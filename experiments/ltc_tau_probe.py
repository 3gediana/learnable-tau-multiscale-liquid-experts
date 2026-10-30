#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ltc_tau_probe.py — S2-gate-checks ① ncps 0.0.2 LTC τ-tie 探针五断言。

五断言（design.md §五 S2① 逐字规格）：
  A1 动态可分：同输入下三专家单步状态变化幅度序 ‖Δx1‖>‖Δx2‖>‖Δx3‖（fast 专家动得多）；
  A2 B(0)≡A(0)：同种子同初始化下臂 B 前向输出与臂 A 逐位一致（max abs diff <1e-6）；
  A3 梯度到达 τ 参数：∂L/∂c_j 非零有限，记录 ‖∂L/∂c‖ 与参数梯度中位数对比；
  A4 稳定性与内存：72 步前向有限无 NaN；frozen batch 峰值显存实测（超 8G 按 D9 阶梯
     batch 64→32→16 降档记录）；B>1 逐样本 timespans 行为记录（预期广播崩，本课题不用）；
  A5 参数审计：A/B 可学习参数差恰 3 标量。

实现口径（D4 冻结）：K=3 OrderedTauLTCCell（hidden 1500 / ode_unfolds 6 /
implicit_param_constraints=True / FullyConnected），τ-tie 覆写见 mr_moe.py；
臂 A τ=(1,10,100) buffer，臂 B τ=保序可学习（D2，init 同值）。A1/A4(a) 用臂 B 专家
（init τ 与臂 A 相同，动力学等价）。A4(b) 峰值显存=臂 B 完整 MR-MoE、B=64、72 步
fwd+bwd、frozen batch（与 ⓪ O3 同配置同协议）。

产物：results/ltc_tau_probe.json（分段落盘，--assertions 可子集跑）。
用法：python experiments/ltc_tau_probe.py [--assertions A1,A2,A3,A4,A5]
      [--hidden 256] [--out results/ltc_tau_probe_h256.json]

S3-pilot ⓿（D15 拆分）：五断言在 D14 应急档 hidden 256 复跑，产物默认另存
results/ltc_tau_probe_h256.json（不覆盖 S2 的 hidden 1500 版）；A4 峰值显存
在目标 hidden 用同协议直接实测（reset_peak→B=64 72 步 fwd+bwd→
max_memory_allocated），并交叉引用 emergency_tier_probe.json 的应急档实测。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from ncps.torch import LTC

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJECT = os.path.dirname(HERE)
RESULTS = os.path.join(PROJECT, "results")
OUT_JSON = os.path.join(RESULTS, "ltc_tau_probe.json")

import mr_moe as M  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
IN_SIZE = 80      # 输入维占位（PhysioNet 2019 40 特征 + 40 mask；S2b 定案前占位）
HIDDEN = 1500     # D4
K = 3             # D4
TAUS = (1.0, 10.0, 100.0)  # D1 臂 A 固定 / 臂 B init
SEQ = 72          # D3 截断
BATCH = 64        # D4
PROBE_B = 8       # 探针批量（A2/A3；真实 batch 64 的 ⓪/A4 已覆盖）
A1_SAMPLES = 20   # A1 随机输入样本数

R: dict = {
    "schema": "ltc-tau-probe/v1",
    "stage": "S2-gate-checks",
    "purpose": "ncps 0.0.2 LTC τ-tie 五断言（design.md §五 S2①）",
    "config": {"input_size": IN_SIZE, "hidden": HIDDEN, "k": K, "taus": list(TAUS),
               "seq_len": SEQ, "batch_full": BATCH, "probe_batch": PROBE_B,
               "device": DEV, "seed": SEED, "a1_samples": A1_SAMPLES},
    "assertions": {},
}


def log(*a):
    print(*a, flush=True)


def _finite(t) -> bool:
    return bool(torch.isfinite(t).all().item()) if t is not None else False


def _dump() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- A1
def a1_dynamic_separability() -> None:
    """A1：同输入下三专家单步状态变化幅度序 ‖Δx1‖>‖Δx2‖>‖Δx3‖。"""
    torch.manual_seed(SEED)
    model = M.MRMoE(IN_SIZE, k=K, hidden=HIDDEN, tau_learnable=True).to(DEV).eval()
    taus = model.tau_sched.tau().detach()  # (K,) 初始 (1,10,100)
    cells = [model.experts[k_i].cell for k_i in range(K)]

    per_sample = []  # 每个样本的 (n1,n2,n3)
    with torch.no_grad():
        for s in range(A1_SAMPLES):
            g = torch.Generator(device=DEV).manual_seed(1000 + s)
            x_t = torch.randn(PROBE_B, IN_SIZE, device=DEV, generator=g)
            h0 = torch.zeros(PROBE_B, HIDDEN, device=DEV)
            ns = []
            for k_i in range(K):
                h_out, h1 = cells[k_i](x_t, h0, 1.0, tau=taus[k_i])
                ns.append(float((h1 - h0).norm(dim=-1).mean().item()))
            per_sample.append(ns)
    means = [sum(col) / len(col) for col in zip(*per_sample)]
    strict = sum(1 for row in per_sample if row[0] > row[1] > row[2])
    # 受控变体：同一专家参数，只改 τ（隔离参数噪声，验证 τ→变化幅度的单调性）
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(7)
        x_t = torch.randn(PROBE_B, IN_SIZE, device=DEV, generator=g)
        h0 = torch.zeros(PROBE_B, HIDDEN, device=DEV)
        ctrl = {}
        for tv in (1.0, 10.0, 100.0):
            _, h1 = cells[0](x_t, h0, 1.0, tau=torch.tensor(tv, device=DEV))
            ctrl[f"tau_{tv:g}"] = float((h1 - h0).norm(dim=-1).mean().item())
    mean_ordered = means[0] > means[1] > means[2]
    ctrl_ordered = ctrl["tau_1"] > ctrl["tau_10"] > ctrl["tau_100"]
    R["assertions"]["A1_dynamic_separability"] = {
        "statement": "同输入下三专家单步状态变化幅度序 ‖Δx1‖>‖Δx2‖>‖Δx3‖",
        "setup": "臂 B 三专家（init τ=1:10:100），同输入同时刻从 h0=0 单步；mean ‖Δx‖ 对 batch 取平均再对 20 样本取平均",
        "taus_init": [round(float(v), 6) for v in taus.cpu().tolist()],
        "mean_delta_norm_per_expert": [round(v, 6) for v in means],
        "mean_ordering_strict": bool(mean_ordered),
        "per_sample_strict_order_fraction": round(strict / A1_SAMPLES, 3),
        "per_sample_first3": [[round(v, 5) for v in row] for row in per_sample[:3]],
        "controlled_shared_params_only_tau_differs": {k: round(v, 6) for k, v in ctrl.items()},
        "controlled_ordering_strict": bool(ctrl_ordered),
        "pass": bool(mean_ordered and strict / A1_SAMPLES >= 0.6),
        "pass_criterion": "mean 序严格 ∧ ≥60% 样本严格序（受控变体序必须严格，否则记分支）",
        "interpretation": ("τ 越大膜时间常数越大、单步状态变化越小；受控变体隔离专家参数噪声后"
                           "若仍有序，则 A1 的序由 τ 驱动而非随机初始化"),
    }
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- A2
def a2_init_equivalence() -> None:
    """A2：B(0)≡A(0) 前向 max abs diff <1e-6（同种子同初始化）。"""
    torch.manual_seed(SEED)
    model_a = M.MRMoE(IN_SIZE, k=K, hidden=HIDDEN, tau_learnable=False).to(DEV).eval()
    torch.manual_seed(SEED)
    model_b = M.MRMoE(IN_SIZE, k=K, hidden=HIDDEN, tau_learnable=True).to(DEV).eval()

    # 共享参数逐位一致核验
    sa = dict(model_a.named_parameters())
    sb = dict(model_b.named_parameters())
    shared = sorted(set(sa) & set(sb))
    max_param_diff = max(float((sa[n] - sb[n]).abs().max().item()) for n in shared)
    only_b = sorted(set(sb) - set(sa))
    only_a = sorted(set(sa) - set(sb))
    tau_a = model_a.tau_sched.tau_detached()
    tau_b = model_b.tau_sched.tau_detached()

    g = torch.Generator(device=DEV).manual_seed(SEED)
    x = torch.randn(PROBE_B, SEQ, IN_SIZE, device=DEV, generator=g)
    with torch.no_grad():
        la = model_a(x)
        lb = model_b(x)
    dmax = float((la - lb).abs().max().item())
    R["assertions"]["A2_init_equivalence"] = {
        "statement": "B(0)≡A(0)：同种子同初始化下臂 B 前向输出与臂 A 逐位一致（max abs diff <1e-6）",
        "shared_param_count": len(shared),
        "shared_param_max_abs_diff": max_param_diff,
        "params_only_in_b": only_b,
        "params_only_in_a": only_a,
        "tau_arm_a": tau_a,
        "tau_arm_b": tau_b,
        "forward_batch": PROBE_B, "forward_seq": SEQ,
        "logits_max_abs_diff": dmax,
        "logits_finite_both": bool(_finite(la) and _finite(lb)),
        "pass": bool(dmax < 1e-6 and max_param_diff == 0.0 and not only_a),
        "interpretation": "τ 初始化等价（buffer vs Parameter 同值）+ 同 RNG 种子 ⇒ init 处前向逐位一致，排除 E1 实现伪影的 init 偏置",
    }
    del model_a, model_b
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- A3
def a3_grad_reaches_tau() -> None:
    """A3：∂L/∂c_j 非零有限；记录 ‖∂L/∂c‖ 与参数梯度中位数对比。"""
    torch.manual_seed(SEED)
    model = M.MRMoE(IN_SIZE, k=K, hidden=HIDDEN, tau_learnable=True).to(DEV).train()
    g = torch.Generator(device=DEV).manual_seed(SEED)
    x = torch.randn(PROBE_B, SEQ, IN_SIZE, device=DEV, generator=g)
    tg = torch.Generator(device=DEV).manual_seed(SEED + 1)
    target = torch.randint(0, 2, (PROBE_B, SEQ), device=DEV, generator=tg).float()

    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    model.zero_grad(set_to_none=True)
    loss.backward()

    c = model.tau_sched.c
    cg = c.grad.detach()
    norms = []
    for n, p in model.named_parameters():
        if p.grad is None or n == "tau_sched.c":
            continue
        norms.append(float(p.grad.detach().norm().item()))
    norms_sorted = sorted(norms)
    med = norms_sorted[len(norms_sorted) // 2]
    # 理论值：∂τ_k/∂c_j = sigmoid(c_j)·1[j≤k]
    dtau_dc = torch.sigmoid(c.detach())
    R["assertions"]["A3_grad_reaches_tau"] = {
        "statement": "梯度到达 τ 参数：∂L/∂c_j 非零有限，记录 ‖∂L/∂c‖ 与参数梯度中位数对比",
        "loss": round(float(loss.item()), 6),
        "c_values": [round(float(v), 6) for v in c.detach().cpu().tolist()],
        "grad_c_values": [round(float(v), 10) for v in cg.cpu().tolist()],
        "grad_c_all_finite": bool(_finite(cg)),
        "grad_c_all_nonzero": bool(bool((cg.abs() > 0).all().item())),
        "grad_c_norm": round(float(cg.norm().item()), 10),
        "param_grad_norm_median": round(med, 6),
        "param_grad_norm_min": round(norms_sorted[0], 8),
        "param_grad_norm_max": round(norms_sorted[-1], 6),
        "n_params_with_grad": len(norms),
        "grad_c_norm_over_median": round(float(cg.norm().item()) / med, 6) if med > 0 else None,
        "dtau_dc_theory_sigmoid": [round(float(v), 6) for v in dtau_dc.cpu().tolist()],
        "pass": bool(_finite(cg) and bool((cg.abs() > 0).all().item())),
        "interpretation": "∂τ_k/∂c_j=sigmoid(c_j)·1[j≤k]（D2 推导）；c_j∈{0.54,9,90} 时 sigmoid≈(0.63,0.9999,1.0) 不消失 ⇒ 梯度可达且不消失",
    }
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- A4
def a4_stability_and_memory() -> None:
    """A4：72 步前向有限无 NaN；frozen batch 峰值显存实测；B>1 timespans 行为记录。"""
    res: dict = {}
    # (a) 72 步前向有限性（B=64，臂 B，no_grad）
    torch.manual_seed(SEED)
    model = M.MRMoE(IN_SIZE, k=K, hidden=HIDDEN, tau_learnable=True).to(DEV).eval()
    g = torch.Generator(device=DEV).manual_seed(SEED)
    x64 = torch.randn(BATCH, SEQ, IN_SIZE, device=DEV, generator=g)
    with torch.no_grad():
        logits = model(x64)
    res["forward72_b64_finite"] = _finite(logits)
    res["forward72_b64_max_abs"] = round(float(logits.abs().max().item()), 6)
    del logits
    torch.cuda.empty_cache()

    # (b) frozen batch 峰值显存：B=64 全模型 72 步 fwd+bwd，目标 hidden 直接实测
    # （⓪ O3 同协议：reset_peak → fwd+bwd → max_memory_allocated；臂 B 完整 MR-MoE）。
    # S3-pilot ⓿：hidden 256 复跑改为现场实测（operator_bench.json 的 O3 数据是
    # hidden 1500 档，不能直接引用），并与 emergency_tier_probe.json 应急档实测交叉对账。
    torch.manual_seed(SEED)
    model = M.MRMoE(IN_SIZE, k=K, hidden=HIDDEN, tau_learnable=True).to(DEV).train()
    g = torch.Generator(device=DEV).manual_seed(SEED)
    xb = torch.randn(BATCH, SEQ, IN_SIZE, device=DEV, generator=g)
    yb = torch.randint(0, 2, (BATCH, SEQ), device=DEV, generator=g).float()
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    logits_b = model(xb)
    loss_b = torch.nn.functional.binary_cross_entropy_with_logits(logits_b, yb)
    model.zero_grad(set_to_none=True)
    loss_b.backward()
    if DEV == "cuda":
        torch.cuda.synchronize()
    fwd_bwd_sec = time.perf_counter() - t0
    peak_mib = (torch.cuda.max_memory_allocated() / 1024 ** 2) if DEV == "cuda" else None
    grad_finite = all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    emerg_ref = None
    try:
        with open(os.path.join(RESULTS, "emergency_tier_probe.json"), encoding="utf-8") as f:
            for case in json.load(f).get("cases", []):
                if case.get("tag") == f"emerg_h{HIDDEN}_b{BATCH}":
                    emerg_ref = {"peak_mem_mib": case.get("peak_mem_mib"),
                                 "batch_sec_median": case.get("batch_sec_median"),
                                 "source": "results/emergency_tier_probe.json "
                                           f"emerg_h{HIDDEN}_b{BATCH}"}
    except Exception:  # noqa: BLE001
        emerg_ref = None
    res["frozen_batch_peak_mib_b64"] = round(peak_mib, 1) if peak_mib is not None else None
    res["frozen_batch_peak_source"] = f"本单元现场实测（hidden={HIDDEN}, B={BATCH}, 72 步 fwd+bwd, reset_peak 协议）"
    res["frozen_batch_budget_mib"] = 8192.0
    res["frozen_batch_within_8g"] = bool(peak_mib is not None and peak_mib <= 8192.0)
    res["frozen_batch_fwd_bwd_sec"] = round(fwd_bwd_sec, 3)
    res["backward_grads_finite"] = bool(grad_finite)
    res["emergency_tier_probe_crossref"] = emerg_ref
    res["d9_ladder"] = {
        "b64": {"peak_mib": res["frozen_batch_peak_mib_b64"],
                "within_8g": res["frozen_batch_within_8g"],
                "note": f"hidden={HIDDEN} 现场实测；D14 应急档 h256/B64 为目标档位"},
        "selected_rung": "b64" if res["frozen_batch_within_8g"] else "需降档",
    }
    # 反传有限性证据：本单元现场 backward 梯度有限 + A3（同算子同配置 72 步 fwd+bwd）
    a3_ok = bool(R["assertions"].get("A3_grad_reaches_tau", {}).get("pass"))
    res["backward_finite"] = bool(a3_ok and grad_finite)
    res["backward_finite_evidence"] = {
        "A3_grad_finite": a3_ok,
        "this_run_backward_grads_finite": bool(grad_finite),
        "note": "A4 反传有限性=本单元现场 B=64 全模型 backward 梯度有限 + A3 同配置佐证",
    }
    res["d9_ladder_needed"] = not res["frozen_batch_within_8g"]
    res["d9_ladder_note"] = ("超 8G 才按 D9 阶梯 batch 64→32→16 降档记录；"
                             f"hidden={HIDDEN} 实测见 frozen_batch_peak_mib_b64")
    del model
    torch.cuda.empty_cache()

    # (c) B>1 逐样本 timespans 行为记录（库 LTC 包装器；预期广播崩，本课题不用该路径）
    torch.manual_seed(SEED)
    rnn = LTC(16, 32).to(DEV).eval()
    xm = torch.randn(4, 12, 16, device=DEV)
    rec: dict = {}
    ts_per_sample = torch.rand(4, 12, device=DEV) + 0.5
    try:
        with torch.no_grad():
            out, _ = rnn(xm, timespans=ts_per_sample)
        rec["per_sample_timespans_B4"] = {"works": True, "note": "未崩（与预期不符，记录实测）"}
    except Exception as e:  # noqa: BLE001
        rec["per_sample_timespans_B4"] = {"works": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    ts_const = torch.ones(1, 12, device=DEV)
    try:
        with torch.no_grad():
            out, _ = rnn(xm, timespans=ts_const)
        rec["const_vector_timespans_1xL"] = {"works": True, "note": "(1,L) 常向量 squeeze 后 0 维标量，可广播"}
    except Exception as e:  # noqa: BLE001
        rec["const_vector_timespans_1xL"] = {"works": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    rec["our_path"] = ("本课题 dt=1h 固定网格：专家 manual-loop 传 elapsed_time=1.0 标量、timespans=None，"
                       "不触达逐样本 timespans 路径（D4 冻结）")
    res["timespans_behavior"] = rec
    del rnn
    torch.cuda.empty_cache()

    res["pass"] = bool(res["forward72_b64_finite"] and res["backward_finite"]
                       and res["frozen_batch_within_8g"])
    R["assertions"]["A4_stability_and_memory"] = {
        "statement": "72 步前向有限无 NaN；frozen batch 峰值显存实测（超 8G 按 D9 阶梯降档记录）；B>1 timespans 行为记录",
        **res,
    }


# ---------------------------------------------------------------- A5
def a5_param_audit() -> None:
    """A5：A/B 可学习参数差恰 3 标量。"""
    torch.manual_seed(SEED)
    model_a = M.MRMoE(IN_SIZE, k=K, hidden=HIDDEN, tau_learnable=False).to(DEV)
    torch.manual_seed(SEED)
    model_b = M.MRMoE(IN_SIZE, k=K, hidden=HIDDEN, tau_learnable=True).to(DEV)
    audit = M.param_audit(model_a, model_b)
    dead = [n for n, p in model_b.named_parameters() if p.requires_grad and n.endswith(".cm")]
    buffers_a = [n for n, _ in model_a.named_buffers()]
    buffers_b = [n for n, _ in model_b.named_buffers()]
    R["assertions"]["A5_param_audit"] = {
        "statement": "A/B 可学习参数差恰 3 标量",
        **audit,
        "dead_params_tau_tie_both_arms": dead,
        "dead_params_note": "τ-tie 覆写不走 cm；两臂同样分配同样不用（库 LTCCell 结构保留），不影响 A/B 差集",
        "buffers_arm_a": buffers_a,
        "buffers_arm_b": buffers_b,
        "count_params_arm_a": M.count_params(model_a),
        "count_params_arm_b": M.count_params(model_b),
        "pass": bool(audit["pass_exactly_3_scalars"]),
    }
    del model_a, model_b
    torch.cuda.empty_cache()


ASSERTIONS = {"A1": a1_dynamic_separability, "A2": a2_init_equivalence,
              "A3": a3_grad_reaches_tau, "A4": a4_stability_and_memory,
              "A5": a5_param_audit}


def main() -> int:
    global HIDDEN, OUT_JSON, IN_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--assertions", default=",".join(ASSERTIONS.keys()))
    ap.add_argument("--hidden", type=int, default=HIDDEN,
                    help="目标 hidden（S3-pilot ⓿ 复跑用 256；默认 1500=D4 冻结值）")
    ap.add_argument("--in-size", type=int, default=IN_SIZE)
    ap.add_argument("--out", default=OUT_JSON,
                    help="产物路径（h256 复跑建议 results/ltc_tau_probe_h256.json）")
    args = ap.parse_args()
    want = [s.strip().upper() for s in args.assertions.split(",") if s.strip()]
    HIDDEN = int(args.hidden)
    IN_SIZE = int(args.in_size)
    OUT_JSON = args.out
    R["config"] = {"input_size": IN_SIZE, "hidden": HIDDEN, "k": K, "taus": list(TAUS),
                   "seq_len": SEQ, "batch_full": BATCH, "probe_batch": PROBE_B,
                   "device": DEV, "seed": SEED, "a1_samples": A1_SAMPLES}
    if HIDDEN != 1500:
        R["stage"] = "S3-pilot (D15 smoke: 五断言@hidden 256 复跑)"
        R["purpose"] = (f"ncps 0.0.2 LTC τ-tie 五断言在 D14 应急档 hidden={HIDDEN} 复跑"
                        "（S2 hidden 1500 版见 results/ltc_tau_probe.json）")
        R["rerun_of"] = "results/ltc_tau_probe.json (hidden 1500, S2-gate-checks)"
    R["env"] = {"device": DEV, "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if os.path.exists(OUT_JSON):  # 分段跑合并
        try:
            with open(OUT_JSON, encoding="utf-8") as f:
                R["assertions"].update(json.load(f).get("assertions", {}))
        except Exception as e:  # noqa: BLE001
            log(f"[probe] 合并旧结果失败（忽略）: {e}")
    _dump()
    for name in want:
        fn = ASSERTIONS[name]
        t = __import__("time").time()
        try:
            fn()
            log(f"[probe] {name} done ({__import__('time').time() - t:.1f}s)")
        except Exception as e:  # noqa: BLE001
            R["assertions"][name + "_ERROR"] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
            log(f"[probe] {name} FAILED: {type(e).__name__}: {str(e)[:300]}")
            if DEV == "cuda":
                torch.cuda.empty_cache()
        _dump()
    n_pass = sum(1 for k, v in R["assertions"].items()
                 if not k.endswith("_ERROR") and v.get("pass"))
    log(f"[probe] wrote {OUT_JSON}; assertions_pass={n_pass}/{len(ASSERTIONS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
