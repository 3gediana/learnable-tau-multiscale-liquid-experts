#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compile_probe.py — S3-compile-probe：torch.compile × use_reentrant=False 组合实测
（D14① 遗留可选工程探针；缺一即弃不阻塞，不影响任何科学判据）。

判据（D14① 原文，D16 复述）：
  ① 编译通过（compile 调用+首次前向无异常，产出可执行计算图）
  ② 与 eager 数值一致：max_abs_diff < 1e-6（前向 logits + 反向梯度两路都查）
  ③ 实测加速 ≥ 1.5×（eager 同协议中位 batch 秒 / compiled 中位 batch 秒；
     参考基准：emergency_tier_probe 4.43s/batch、smoke A_s0 实测 3.74s/batch）
  缺一即弃；compile 不得改数值语义。

为什么必须测 use_reentrant=False：本课题 R3/R5 已实测 reentrant 检查点失败
（闭包图双反传），训练路径两处检查点（OrderedTauLTCCell 每 unfold、MRMoE
每专家）均为 use_reentrant=False；本探针只测该组合，不改 mr_moe.py 任何代码。

协议（与 pilot_train.py 训练内循环同构，D14 应急档 frozen 配置）：
  hidden=256 / batch=64 / ode_unfolds=6 / 子采样 2000（DATA_SEED=20261029）/
  臂 B（τ-tie 可学习：同时覆盖 R3 unfold 检查点与 R5 专家序列检查点）
  计时步 = .to + zero_grad + fwd + BCE + backward + gnorm + c_grad + opt.step
           + cuda.synchronize（与 pilot_train.py:285-309 逐行同构）；
  数值对比用固定 batch（x[:64]，两臂同数据同初始化种子 seed 0）。

产物（全部原子写 tmp+os.replace；分阶段落盘，被 kill 也留证据）：
  results/compile_probe.json          探针真源（含分阶段 status）
  results/pilot_timing.json           追加 compile_probe 计时块（保留既有阶段块）

用法：G:/ai/python11/python.exe -u experiments/compile_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJ = os.path.dirname(HERE)
RESULTS = os.path.join(PROJ, "results")
PROBE_JSON = os.path.join(RESULTS, "compile_probe.json")
TIMING_JSON = os.path.join(RESULTS, "pilot_timing.json")

import mr_moe as M            # noqa: E402
import pilot_train as P       # noqa: E402（复用 D3 数据加载与子采样，零改动）

T_START = time.perf_counter()
UNIT_START_ISO = datetime.now().isoformat()


def log(*a):
    print(*a, flush=True)


def atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def jload(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def sec() -> float:
    return round(time.perf_counter() - T_START, 2)


# ---------------------------------------------------------------- 状态记录
state = {
    "schema": "compile-probe/v1",
    "stage": "S3-compile-probe",
    "purpose": "D14①可选工程探针：torch.compile×use_reentrant=False 组合在 τ-tie LTC "
               "训练路径的数值一致性与加速实测；缺一即弃不阻塞，不改科学判据",
    "criteria_d14": {
        "compile_pass": "编译通过（compile+首次前向无异常）",
        "numerical_consistent": "与 eager max_abs_diff < 1e-6（logits 前向 + 梯度反向）",
        "speedup_ge_1.5x": "实测加速 ≥ 1.5×（eager/compiled 同协议中位 batch 秒）",
    },
    "unit_started_at": UNIT_START_ISO,
    "env": {},
    "config": {},
    "phases": {},
    "criteria_eval": {},
    "status": "running",
    "verdict": None,
    "reason": None,
}


def flush():
    state["updated_at"] = datetime.now().isoformat()
    atomic_write_json(PROBE_JSON, state)


def fail(stage_name: str, exc: BaseException) -> None:
    """编译/运行异常 → 记 discarded+reason（不重试；同一实现路径重试上限 2 次的预算留给后续单元）。"""
    state["status"] = "discarded"
    state["verdict"] = "falsified"
    state["reason"] = f"{stage_name} 异常：{type(exc).__name__}: {exc}"
    state["traceback"] = traceback.format_exc()[-2000:]
    log(f"[probe] {stage_name} FAILED: {exc}")
    flush()


# ---------------------------------------------------------------- 主流程
def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["env"] = {
        "torch": torch.__version__,
        "device": str(dev),
        "gpu": (torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
        "ncps": getattr(__import__("ncps"), "__version__", "?"),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "python": sys.version.split()[0],
    }
    log(f"[probe] env={state['env']}")

    # ---------- 数据（D3 冻结来源，与 pilot_train 同子采样种子） ----------
    data = P.load_arrays(P.DATA_DEFAULT)
    n_feat = int(data["x"].shape[-1])
    sub_idx = P.make_subsample(data["splits"], 2000)   # DATA_SEED=20261029
    x_tr = torch.from_numpy(data["x"][sub_idx]).to(dev)
    y_tr = torch.from_numpy(data["y"][sub_idx]).to(dev)
    xb, yb = x_tr[:64], y_tr[:64]     # 固定 batch（避免动态 shape 重编译）
    state["config"] = {
        "hidden": 256, "batch": 64, "ode_unfolds": M.ODE_UNFOLDS,
        "subsample": int(len(sub_idx)), "arm": "B (tau_learnable=True)",
        "lr": P.LR, "loss": "BCEWithLogits", "grad_clip": "无（F3 监控）",
        "checkpoint_unfolds": "R3: per-unfold checkpoint(use_reentrant=False)",
        "checkpoint_experts": "R5: per-expert checkpoint(use_reentrant=False)",
        "reentrant_note": "reentrant=True 组合本课题已实测失败（闭包图双反传），"
                          "故只测 use_reentrant=False；mr_moe.py 零改动",
        "data": os.path.relpath(P.DATA_DEFAULT, PROJ),
        "data_seed": P.DATA_SEED, "seq_len": int(x_tr.shape[1]), "n_feat": n_feat,
        "sub_idx_sha1": __import__("hashlib").sha1(
            sub_idx.tobytes()).hexdigest()[:12],
        "timed_step_protocol": ".to+zero_grad+fwd+BCE+backward+gnorm+c_grad+opt.step+"
                               "synchronize（pilot_train.py:285-309 同构）",
    }
    log(f"[probe] config={state['config']}")

    def fresh_model():
        torch.manual_seed(0)   # D10：同初始化种子
        m = M.build_model(n_feat, 0, tau_learnable=True, hidden=256).to(dev)
        return m

    def train_step(model, opt):
        """pilot_train 内循环同构的一步（返回 loss）；计时含整步。"""
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        gnorm = float(sum(
            (float(p.grad.detach().norm().item()) ** 2)
            for p in model.parameters() if p.grad is not None) ** 0.5)
        cg = model.c_grad_norm()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return float(loss), gnorm, cg

    # ================= 阶段 A：eager 基线计时 =================
    log("[probe] === phase A: eager baseline ===")
    try:
        me = fresh_model()
        oe = torch.optim.Adam(me.parameters(), lr=P.LR)
        me.train()
        for _ in range(2):                     # warmup
            train_step(me, oe)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tsteps, losses = [], []
        for _ in range(3):                     # timed
            t0 = time.perf_counter()
            lv, gn, cg = train_step(me, oe)
            tsteps.append(round(time.perf_counter() - t0, 4))
            losses.append(lv)
        eager = {
            "warmup_steps": 2, "timed_steps": 3, "step_sec": tsteps,
            "batch_sec_median": round(float(np.median(tsteps)), 4),
            "peak_mem_mib": (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                             if dev.type == "cuda" else None),
            "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
            "grad_global_norm_last": round(gn, 6), "c_grad_norm_last": cg,
            "all_finite": bool(np.isfinite(losses).all()),
        }
        state["phases"]["eager"] = eager
        log(f"[probe] eager {eager}")
        del me, oe
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        flush()
    except Exception as e:  # noqa: BLE001
        fail("phase A (eager baseline)", e)
        return 1

    # ================= 阶段 B：数值一致性（compile cold） =================
    log("[probe] === phase B: numerics (eager vs compiled, same init) ===")
    try:
        me2 = fresh_model()
        mc2 = fresh_model()                    # 同种子同初始化 → 权重逐位一致
        mcc = torch.compile(mc2)               # use_reentrant=False 检查点已在模型内
        me2.eval(); mcc.eval()
        with torch.no_grad():
            le = me2(xb)
            t0 = time.perf_counter()
            lc = mcc(xb)                       # 首次调用触发 cold compile
            if dev.type == "cuda":
                torch.cuda.synchronize()
            compile_sec_cold = round(time.perf_counter() - t0, 2)
        logits_diff = float((le - lc).abs().max().item())

        # 反向梯度对比（同 batch 一步 fwd+bwd）
        me2.train(); mcc.train()
        lo_e = F.binary_cross_entropy_with_logits(me2(xb), yb)
        lo_e.backward()
        lo_c = F.binary_cross_entropy_with_logits(mcc(xb), yb)
        lo_c.backward()
        ge = {n: p.grad for n, p in me2.named_parameters()}
        gc = {n: p.grad for n, p in mc2.named_parameters()}
        grad_diff, worst = 0.0, None
        for n in ge:
            a, b = ge[n], gc.get(n)
            if a is None and b is None:
                continue
            if a is None or b is None:
                grad_diff, worst = float("inf"), n
                break
            d = float((a - b).abs().max().item())
            if d > grad_diff:
                grad_diff, worst = d, n
        cge = me2.c_grad_norm()
        cgc = mc2.c_grad_norm()
        cg_diff = (None if (cge is None or cgc is None)
                   else abs(cge - cgc))
        numerics = {
            "compile_sec_cold": compile_sec_cold,
            "logits_max_abs_diff": logits_diff,
            "grad_max_abs_diff": grad_diff, "grad_worst_param": worst,
            "c_grad_norm_eager": cge, "c_grad_norm_compiled": cgc,
            "c_grad_abs_diff": cg_diff,
            "loss_eager": round(float(lo_e), 8),
            "loss_compiled": round(float(lo_c), 8),
            "loss_abs_diff": abs(round(float(lo_e), 8) - round(float(lo_c), 8)),
            "finite_both": bool(torch.isfinite(le).all() and torch.isfinite(lc).all()),
            "init_identical_note": "两模型同 manual_seed(0) 构建（D10），权重逐位一致",
        }
        state["phases"]["numerics"] = numerics
        log(f"[probe] numerics {numerics}")
        del me2, mc2, mcc, ge, gc
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        flush()
    except Exception as e:  # noqa: BLE001
        fail("phase B (compile/numerics)", e)
        return 1

    # ================= 阶段 C：compiled 计时 =================
    log("[probe] === phase C: compiled timing ===")
    try:
        mt = fresh_model()
        mct = torch.compile(mt)
        ot = torch.optim.Adam(mct.parameters(), lr=P.LR)
        mct.train()
        t0 = time.perf_counter()
        train_step(mct, ot)                    # 首次调用（Dynamo 缓存可能热命中）
        if dev.type == "cuda":
            torch.cuda.synchronize()
        compile_sec_warm = round(time.perf_counter() - t0, 2)
        train_step(mct, ot)                    # warmup 2
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tsteps, losses = [], []
        for _ in range(3):
            t1 = time.perf_counter()
            lv, gn, cg = train_step(mct, ot)
            tsteps.append(round(time.perf_counter() - t1, 4))
            losses.append(lv)
        compiled = {
            "compile_sec_first_call_this_model": compile_sec_warm,
            "warmup_steps": 2, "timed_steps": 3, "step_sec": tsteps,
            "batch_sec_median": round(float(np.median(tsteps)), 4),
            "peak_mem_mib": (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                             if dev.type == "cuda" else None),
            "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
            "grad_global_norm_last": round(gn, 6), "c_grad_norm_last": cg,
            "all_finite": bool(np.isfinite(losses).all()),
        }
        state["phases"]["compiled"] = compiled
        log(f"[probe] compiled {compiled}")
        del mt, mct, ot
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        fail("phase C (compiled timing)", e)
        return 1

    # ================= 判据汇总 =================
    e_med = state["phases"]["eager"]["batch_sec_median"]
    c_med = state["phases"]["compiled"]["batch_sec_median"]
    diff = max(state["phases"]["numerics"]["logits_max_abs_diff"],
               state["phases"]["numerics"]["grad_max_abs_diff"])
    speedup = round(e_med / c_med, 3) if c_med > 0 else None
    ev = {
        "compile_pass": True,
        "numerical_consistent": bool(diff < 1e-6),
        "speedup_pass": bool(speedup is not None and speedup >= 1.5),
    }
    state["criteria_eval"] = ev
    state["speedup_vs_eager_median"] = speedup
    state["max_abs_diff_overall"] = diff
    state["reference_baselines"] = {
        "emergency_tier_probe_emerg_h256_b64_batch_sec": 4.433,
        "smoke_A_s0_batch_sec_median": 3.74,
        "note": "eager 同协议中位 %.3fs（本单元现场实测）；加速比对本单元 eager 中位数，"
                "参考基准仅作量级对账" % e_med,
    }
    if ev["compile_pass"] and ev["numerical_consistent"] and ev["speedup_pass"]:
        state["status"] = "passed"
        state["verdict"] = "supported"
        state["reason"] = None
    else:
        state["status"] = "discarded"
        state["verdict"] = "falsified"
        miss = [k for k, v in ev.items() if not v]
        state["reason"] = ("缺一即弃：未达判据 " + ",".join(miss) +
                           f"（max_abs_diff={diff:.3e}, speedup={speedup}）")
    log(f"[probe] criteria_eval={ev} speedup={speedup} diff={diff:.3e} "
        f"status={state['status']}")

    # ================= 追加计时到 pilot_timing.json（原子，保留既有块） =================
    try:
        timing = jload(TIMING_JSON, {})
        timing["step2_compile_probe"] = {
            "status": "executed_by_S3-compile-probe_unit",
            "executed_at": datetime.now().isoformat(),
            "result": "results/compile_probe.json",
            "verdict": state["verdict"],
            "note": "本块为 S3-pilot smoke 段当时的弃用记录（时间预算保护必选项）；"
                    "compile×use_reentrant=False 组合由 S3-compile-probe 独立单元实测，"
                    "判据与结论以 results/compile_probe.json 为准",
        }
        timing.setdefault("stages", [])
        if "S3-compile-probe" not in timing["stages"]:
            timing["stages"].append("S3-compile-probe")
        timing["stage"] = " + ".join(timing["stages"])
        timing["compile_probe"] = {
            "stage": "S3-compile-probe",
            "status": state["status"],
            "verdict": state["verdict"],
            "reason": state["reason"],
            "criteria_eval": ev,
            "eager_batch_sec_median": e_med,
            "compiled_batch_sec_median": c_med,
            "speedup_vs_eager_median": speedup,
            "max_abs_diff_overall": diff,
            "compile_sec_cold": state["phases"]["numerics"]["compile_sec_cold"],
            "peak_mem_mib_eager": state["phases"]["eager"]["peak_mem_mib"],
            "peak_mem_mib_compiled": state["phases"]["compiled"]["peak_mem_mib"],
            "reference_baselines": state["reference_baselines"],
            "source": "results/compile_probe.json",
            "window": {"started_at": UNIT_START_ISO,
                       "done_at": datetime.now().isoformat()},
        }
        atomic_write_json(TIMING_JSON, timing)
        log(f"[probe] appended compile_probe block -> {TIMING_JSON}")
    except Exception as e:  # noqa: BLE001
        log(f"[probe] WARN timing append failed: {e}")
        state["timing_append_error"] = repr(e)

    flush()
    log(f"[probe] DONE status={state['status']} verdict={state['verdict']} "
        f"elapsed={sec()}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
