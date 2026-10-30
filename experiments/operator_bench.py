#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""operator_bench.py — S2-gate-checks ⓪ 算子数值稳定性与效率基准（fb-ab50d0ec9f/D12）。

动机（用户反馈 2026-10-27 生效至 11-03）：「先优化算子再跑实验」——在进入①探针与
S3 批量训练前，先对本课题科学路径上的 LTC τ-tie 算子（ncps 0.0.2 LTCCell 子类）做
数值稳定性与效率基准；发现问题先在 experiments/mr_moe.py 实现内修并逐条记录修复点
（R1-R4），修不掉带证据早退 F8，不带病算子进①/S3。

基准配置（单元指令逐字）：hidden 1500 / ode_unfolds 6 / batch 64 / 72 步前向。
检查项：前向反向无 NaN、梯度范数量级、frozen batch 峰值显存、单步与单 epoch 计时。

覆盖：
  O0 环境锚定（torch/ncps 版本 + 源码证据行 + 微基准带宽）
  O1 修复前裸形态 TauTieLTCCellRaw（仅 τ-tie 覆写）：单步/72 步 BPTT 显存墙定位
  O2 修复后 OrderedTauLTCCell（R1-R3）：B=16/32/64 全 72 步 fwd+bwd 峰值显存+计时
  O3 完整 MR-MoE（3 专家+门控+读出，臂 B 可学习 τ）真实配置：峰值/计时/梯度/epoch 外推
  O4 数值稳定性：梯度范数 vs 序列长、144 步压力、gleak 极端应力（τ-tie 分母耦合）
  O5 数学恒等证据：裸 vs 修复前向逐位一致；τ-tie(τ=1,cm≡gleak) vs 库 LTCCell 逐位一致
  O6 计时分解：单 unfold 成本与带宽微基准（效率判读依据）

产物：results/operator_bench.json（全部原始数字）+ results/operator_bench.md。
用法：python experiments/operator_bench.py [--sections O0,O1,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJECT = os.path.dirname(HERE)
RESULTS = os.path.join(PROJECT, "results")
OUT_JSON = os.path.join(RESULTS, "operator_bench.json")

import mr_moe as M  # noqa: E402  被基准的实现本体
from ncps.torch import LTCCell  # noqa: E402
from ncps.wirings import FullyConnected  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = 1500
ODE_UNFOLDS = 6
BATCH = 64
SEQ = 72
IN_SIZE = 80   # 输入维占位（PhysioNet 2019 40 特征 + 40 mask ≈ 80；只影响计时量级）
TAUS = (1.0, 10.0, 100.0)

R: dict = {
    "schema": "operator-bench/v1",
    "stage": "S2-gate-checks",
    "purpose": "LTC τ-tie 算子数值稳定性与效率基准（fb-ab50d0ec9f/D12，⓪先于①）",
    "config": {"hidden": HIDDEN, "ode_unfolds": ODE_UNFOLDS, "batch": BATCH,
               "seq_len": SEQ, "input_size_placeholder": IN_SIZE, "taus": list(TAUS),
               "device": DEV},
    "fixes": {
        "R1": "softplus(gleak) 提到 unfold 循环外（库每次 unfold 重算；恒等变换）",
        "R2": "sparsity_mask 折进 w_param 每步一次（FullyConnected 下 mask 全 1，精确恒等）",
        "R3": "每次 unfold 梯度检查点（use_reentrant=False）：反传图从 >20GB 压到 O(小)，反向按 unfold 重算，数学不变",
        "R4": "FullyConnected wiring 构建缓存（18.6s/个→0；erev 为 adjacency 拷贝，数值逐位不变）",
        "R5": "专家序列级梯度检查点（MRMoE.forward）：三专家反传图不再共存，全模型峰值≈单专家（B=32=6863MiB<8G；D9 阶梯 64→32 落预算内）",
    },
    "sections": {},
}


def log(*a):
    print(*a, flush=True)


def _finite(t) -> bool:
    return bool(torch.isfinite(t).all().item()) if t is not None else False


def _grad_report(model: nn.Module, loss: torch.Tensor) -> dict:
    """反向 + 逐模块梯度范数量级 + 全局范数（NaN 检查）。"""
    model.zero_grad(set_to_none=True)
    loss.backward()
    per_module: dict[str, float] = {}
    all_sq, n_bad = 0.0, 0
    c_grad = None
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n = float(g.norm().item())
        if not _finite(g):
            n_bad += 1
            n = float("inf")
        else:
            all_sq += n * n
        if name.endswith("tau_sched.c"):
            c_grad = {"norm": n, "values": [round(float(v), 8) for v in g.reshape(-1).cpu().tolist()]}
        if name.startswith("experts."):
            key = "expert" + name.split(".")[1]
        elif name.startswith("tau_sched."):
            key = "tau_sched"
        else:
            key = name.split(".")[0]
        per_module[key] = per_module.get(key, 0.0) + n
    return {
        "grad_global_norm": float(all_sq ** 0.5) if n_bad == 0 else float("inf"),
        "grad_nonfinite_param_count": n_bad,
        "grad_module_norms": {k: round(v, 6) for k, v in sorted(per_module.items())},
        "grad_c": c_grad,
    }


class ExpertHead(nn.Module):
    """算子基准出口包装：专家（返回 (B,L,H) 序列）接线性头出 (B,L) logits。

    τ 在构造时绑定（标量 float；算子基准不测 τ 梯度，梯度到达 τ 由①探针 A3 测）。
    """

    def __init__(self, expert: nn.Module, tau: float, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.expert = expert
        self.head = nn.Linear(hidden, 1)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.expert(x, self.tau)  # (B,L,H)
        return self.head(out).squeeze(-1)  # (B,L)


def _bench_case(tag: str, make_model, x: torch.Tensor,
                repeats: int = 1, timed: bool = True) -> dict:
    """单案例：frozen batch 前向有限性 + 反向梯度 + 峰值显存 + 计时。

    峰值显存协议：torch.cuda.reset_peak_memory_stats() 在跑前向/反向之前调用，
    跑完读 max_memory_allocated（O1 曾把 reset 放在跑之后读到驻存量，已修正）。
    """
    out: dict = {"tag": tag, "batch": int(x.size(0)), "seq_len": int(x.size(1))}
    torch.manual_seed(0)
    model = make_model().to(DEV)
    model.train()
    out["params"] = M.count_params(model)
    target = torch.randint(0, 2, (x.size(0), x.size(1)), device=DEV).float()

    # --- 前向 + 反向（数值稳定性）+ 峰值显存（同一次 frozen batch 运行内测）---
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    logits = model(x)
    assert logits.dim() == 2, logits.shape
    out["forward_finite"] = _finite(logits)
    out["forward_max_abs"] = round(float(logits.detach().abs().max().item()), 6) if out["forward_finite"] else None
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    out["loss"] = round(float(loss.item()), 6)
    rep = _grad_report(model, loss)
    out.update(rep)
    out["backward_finite"] = out["grad_nonfinite_param_count"] == 0
    if DEV == "cuda":
        torch.cuda.synchronize()
    out["batch_fwd_bwd_sec"] = round(time.perf_counter() - t0, 3)
    if DEV == "cuda":
        out["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)

    # --- 计时：frozen batch 重复（中位数）---
    if timed and DEV == "cuda":
        times = []
        for _ in range(repeats):
            xb = x  # frozen batch：同一输入
            t0 = time.perf_counter()
            model.zero_grad(set_to_none=True)
            lg = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, target)
            loss.backward()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times.sort()
        out["batch_ms_median"] = round(times[len(times) // 2] * 1000, 1)
        out["batch_ms_all"] = [round(t * 1000, 1) for t in times]
        out["per_step_ms"] = round(out["batch_ms_median"] / SEQ, 2)
        out["epoch_estimate_sec"] = {
            f"n{n}_b{x.size(0)}": round(out["batch_ms_median"] * (n // x.size(0)) / 1000, 1)
            for n in (8000, 4000, 2000)
        }
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def _oom_case(tag: str, make_model, x: torch.Tensor) -> dict:
    """预期 OOM 案例：记录原始错误信息（算子墙证据）。"""
    torch.manual_seed(0)
    try:
        model = make_model().to(DEV)
        logits = model(x)
        loss = logits.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        out = {"tag": tag, "works": True,
               "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
               "note": "未 OOM（记录实测峰值）"}
        del model
        torch.cuda.empty_cache()
        return out
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"tag": tag, "works": False,
                "error": f"{type(e).__name__}: {e}",
                "verdict": "τ-tie 裸形态在该配置超出 8G 显存（算子墙，R3 修复靶点）"}
    except Exception as e:  # noqa: BLE001
        torch.cuda.empty_cache()
        return {"tag": tag, "works": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}


# ---------------------------------------------------------------- O0
def sec_o0() -> None:
    import ncps
    import ncps.torch.ltc_cell as ltc_mod
    t0 = time.perf_counter()
    wiring = FullyConnected(HIDDEN)
    wiring_build_sec = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    wiring.build(IN_SIZE)
    wiring_build_sec2 = round(time.perf_counter() - t0, 1)
    # 带宽微基准（效率判读的机器上下文）
    x = torch.randn(BATCH, HIDDEN, HIDDEN, device=DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        y = x * 2.0
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    bw = round(x.numel() * 4 * 2 / 1e9 / dt, 1)
    del x, y
    torch.cuda.empty_cache()
    R["sections"]["O0_env_anchor"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2, 1) if torch.cuda.is_available() else None,
        "ncps": getattr(ncps, "__version__", "unknown"),
        "device": DEV,
        "wiring_build_sec_uncached": [wiring_build_sec, wiring_build_sec2],
        "elementwise_bandwidth_GBs": bw,
        "source_evidence": {
            "ltc_cell.py:223-225": "cm_t = make_positive_fn(params['cm']) / (elapsed_time / ode_unfolds)  # τ-tie 唯一覆写点",
            "ltc_cell.py:242": "gleak = make_positive_fn(params['gleak'])  # 库在每次 unfold 内重算（R1 靶点）",
            "ltc_cell.py:234": "w_activation = w_activation * params['sparsity_mask']  # 库在每次 unfold 内乘 (B,H,H)（R2 靶点）",
            "ltc_cell.py:247": "v_pre = numerator / (denominator + epsilon)  # 隐式(后向欧拉)求解，eps=1e-8",
            "ltc_cell.py:53-54": "make_positive_fn = Softplus() if implicit_param_constraints else Identity()",
            "wirings.py:47-48": "erev_initializer 返回 adjacency_matrix 拷贝（无 RNG 消耗 → R4 缓存安全）",
        },
        "ltc_cell_file": os.path.abspath(ltc_mod.__file__),
    }


# ---------------------------------------------------------------- O1
def sec_o1() -> None:
    """修复前裸形态（TauTieLTCCellRaw：仅 τ-tie 覆写，无 R1/R2/R3）。"""
    res: dict = {}
    tau = 100.0
    # 单步 B=16（应可跑；量化裸形态单步反传图占用）
    x1 = torch.randn(16, 1, IN_SIZE, device=DEV)
    res["b16_single_step"] = _bench_case(
        "raw_tie_b16_1step",
        lambda: ExpertHead(M.OrderedTauLTCExpert(IN_SIZE, HIDDEN, cell_cls=M.TauTieLTCCellRaw), tau),
        x1, timed=False)
    # 72 步 BPTT B=16（预期 OOM：反传图随步数线性累积）
    x16 = torch.randn(16, SEQ, IN_SIZE, device=DEV)
    res["b16_seq72_bptt"] = _oom_case(
        "raw_tie_b16_seq72",
        lambda: ExpertHead(M.OrderedTauLTCExpert(IN_SIZE, HIDDEN, cell_cls=M.TauTieLTCCellRaw), tau),
        x16)
    # 72 步 BPTT B=4（进一步定位：缩小批量也 OOM → 墙在步数×反传图，不在单步批量）
    x4 = torch.randn(4, SEQ, IN_SIZE, device=DEV)
    res["b4_seq72_bptt"] = _oom_case(
        "raw_tie_b4_seq72",
        lambda: ExpertHead(M.OrderedTauLTCExpert(IN_SIZE, HIDDEN, cell_cls=M.TauTieLTCCellRaw), tau),
        x4)
    # 单步 B=64（预期 OOM：单步反传图已 >8G）
    x1b = torch.randn(BATCH, 1, IN_SIZE, device=DEV)
    res["b64_single_step"] = _oom_case(
        "raw_tie_b64_1step",
        lambda: ExpertHead(M.OrderedTauLTCExpert(IN_SIZE, HIDDEN, cell_cls=M.TauTieLTCCellRaw), tau),
        x1b)
    R["sections"]["O1_before_raw"] = res


# ---------------------------------------------------------------- O2
def sec_o2() -> None:
    """修复后（OrderedTauLTCCell，R1-R3）：真实配置 72 步 fwd+bwd。

    BENCH_O2_BATCHES 环境变量可选子集（默认 16,32,64），便于分段跑避前台超时。
    """
    res: dict = {}
    tau = 100.0
    want = [int(s) for s in os.environ.get("BENCH_O2_BATCHES", "16,32,64").split(",")]
    rep64 = int(os.environ.get("BENCH_B64_REPEATS", "2"))
    if 16 in want:
        x16 = torch.randn(16, SEQ, IN_SIZE, device=DEV)
        res["b16_seq72"] = _bench_case(
            "fixed_tie_b16_seq72",
            lambda: ExpertHead(M.OrderedTauLTCExpert(IN_SIZE, HIDDEN), tau),
            x16, repeats=2)
    if 32 in want:
        x32 = torch.randn(32, SEQ, IN_SIZE, device=DEV)
        res["b32_seq72"] = _bench_case(
            "fixed_tie_b32_seq72",
            lambda: ExpertHead(M.OrderedTauLTCExpert(IN_SIZE, HIDDEN), tau),
            x32, repeats=1, timed=False)
    if 64 in want:
        x64 = torch.randn(BATCH, SEQ, IN_SIZE, device=DEV)
        res["b64_seq72"] = _bench_case(
            "fixed_tie_b64_seq72",
            lambda: ExpertHead(M.OrderedTauLTCExpert(IN_SIZE, HIDDEN), tau),
            x64, repeats=rep64)
    R["sections"]["O2_after_fixed"] = res


# ---------------------------------------------------------------- O3
def sec_o3() -> None:
    """完整 MR-MoE（3 专家+门控+读出，臂 B 可学习 τ）真实配置。

    R5 后全模型峰值≈单专家；按 D9 阶梯记录 batch 64→32→16→8 的实测峰值/计时，
    OOM 案例记录原始错误（算子墙证据）。BENCH_O3_BATCHES 可选子集。
    """
    res: dict = {}
    want = [int(s) for s in os.environ.get("BENCH_O3_BATCHES", "8,16,32,64").split(",")]
    rep64 = int(os.environ.get("BENCH_B64_REPEATS", "2"))
    if 8 in want:
        x8 = torch.randn(8, SEQ, IN_SIZE, device=DEV)
        res["b8_seq72"] = _bench_case(
            "mrmoe_armB_b8_seq72", lambda: M.MRMoE(IN_SIZE, tau_learnable=True),
            x8, repeats=1)
    if 16 in want:
        x16 = torch.randn(16, SEQ, IN_SIZE, device=DEV)
        res["b16_seq72"] = _bench_case(
            "mrmoe_armB_b16_seq72", lambda: M.MRMoE(IN_SIZE, tau_learnable=True),
            x16, repeats=1, timed=False)
    if 32 in want:
        x32 = torch.randn(32, SEQ, IN_SIZE, device=DEV)
        res["b32_seq72"] = _bench_case(
            "mrmoe_armB_b32_seq72", lambda: M.MRMoE(IN_SIZE, tau_learnable=True),
            x32, repeats=1, timed=False)
    if 64 in want:
        x64 = torch.randn(BATCH, SEQ, IN_SIZE, device=DEV)
        try:
            res["b64_seq72"] = _bench_case(
                "mrmoe_armB_b64_seq72", lambda: M.MRMoE(IN_SIZE, tau_learnable=True),
                x64, repeats=rep64)
        except torch.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            res["b64_seq72"] = {
                "tag": "mrmoe_armB_b64_seq72", "batch": BATCH, "seq_len": SEQ,
                "works": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "verdict": "R5 后 B=64 仍超 8G（D9 阶梯降档到 32；32/16 见上方实测）"}
    R["sections"]["O3_full_mrmoe"] = res


# ---------------------------------------------------------------- O4
def sec_o4() -> None:
    """数值稳定性：梯度范数 vs 序列长、144 步压力、gleak 极端应力。"""
    res: dict = {}
    tau = 100.0
    # (a) 梯度范数 vs 序列长度（B=16，fixed）：爆炸/消失检查
    for seq in (12, 24, 48, 72):
        x = torch.randn(16, seq, IN_SIZE, device=DEV)
        torch.manual_seed(0)
        model = M.OrderedTauLTCExpert(IN_SIZE, HIDDEN).to(DEV)
        out, _ = model(x, tau)
        loss = out.float().pow(2).mean()
        rep = _grad_report(model, loss)
        res[f"gradnorm_seq{seq}_b16"] = {
            "seq_len": seq,
            "grad_global_norm": rep["grad_global_norm"],
            "grad_module_norms": rep["grad_module_norms"],
            "forward_finite": _finite(out),
            "out_max_abs": round(float(out.detach().abs().max().item()), 6),
        }
        del model
        torch.cuda.empty_cache()
    # (b) 144 步压力（B=16）
    x = torch.randn(16, 144, IN_SIZE, device=DEV)
    torch.manual_seed(0)
    model = M.OrderedTauLTCExpert(IN_SIZE, HIDDEN).to(DEV)
    out, _ = model(x, tau)
    loss = out.float().pow(2).mean()
    rep = _grad_report(model, loss)
    res["stress_seq144_b16"] = {
        "forward_finite": _finite(out), "out_max_abs": round(float(out.detach().abs().max().item()), 6),
        "grad_global_norm": rep["grad_global_norm"], "grad_nonfinite_param_count": rep["grad_nonfinite_param_count"],
    }
    del model, out, loss
    torch.cuda.empty_cache()
    # (c) gleak 极端应力（迷你 hidden 64，快）：τ-tie 把 cm_t 与 gleak 耦合，
    #     gleak→−∞ 时 softplus(gleak)→0、cm_t→0，检验分母是否塌缩到 ε
    mini = {}
    for gv in (-30.0, -10.0, 0.0, 30.0):
        torch.manual_seed(0)
        wiring = FullyConnected(64)
        wiring.build(IN_SIZE)
        cell = M.OrderedTauLTCCell(wiring, in_features=IN_SIZE, ode_unfolds=ODE_UNFOLDS,
                                   implicit_param_constraints=True).to(DEV)
        with torch.no_grad():
            cell._params["gleak"].fill_(gv)
            h = torch.zeros(4, 64, device=DEV)
            vmax = 0.0
            finite = True
            for t in range(SEQ):
                x_t = torch.randn(4, IN_SIZE, device=DEV)
                _, h = cell(x_t, h, 1.0, tau=torch.tensor(100.0, device=DEV))
                if not torch.isfinite(h).all():
                    finite = False
                    break
                vmax = max(vmax, float(h.abs().max().item()))
        mini[f"gleak_{gv:g}"] = {"forward_finite": finite, "max_abs_state": vmax}
        del cell
        torch.cuda.empty_cache()
    # (c2) 双极端（w 与 gleak 同时 → −∞）：库 ε=1e-8 守卫是唯一保护（τ-tie 与库同构）
    torch.manual_seed(0)
    wiring = FullyConnected(64)
    wiring.build(IN_SIZE)
    cell = M.OrderedTauLTCCell(wiring, in_features=IN_SIZE, ode_unfolds=ODE_UNFOLDS,
                               implicit_param_constraints=True).to(DEV)
    with torch.no_grad():
        cell._params["gleak"].fill_(-40.0)
        cell._params["w"].fill_(-40.0)
        h = torch.zeros(4, 64, device=DEV)
        finite, vmax = True, 0.0
        for t in range(72):
            x_t = torch.randn(4, IN_SIZE, device=DEV)
            _, h = cell(x_t, h, 1.0, tau=torch.tensor(100.0, device=DEV))
            if not torch.isfinite(h).all():
                finite = False
                break
            vmax = max(vmax, float(h.abs().max().item()))
    mini["gleak_-40_w_-40"] = {"forward_finite": finite, "max_abs_state": vmax,
                               "note": "库 ε=1e-8 守卫是唯一保护；τ-tie 与库 LTCCell 同构（非本覆写引入）"}
    res["gleak_stress_mini64"] = mini
    R["sections"]["O4_stability"] = res


# ---------------------------------------------------------------- O5
def sec_o5() -> None:
    """数学恒等证据（修复不改数值 + τ-tie 与库求解器同构）。"""
    res: dict = {}
    torch.manual_seed(0)
    wiring = FullyConnected(64)
    wiring.build(IN_SIZE)
    cell_fixed = M.OrderedTauLTCCell(wiring, in_features=IN_SIZE, ode_unfolds=ODE_UNFOLDS,
                                     implicit_param_constraints=True).to(DEV)
    wiring2 = FullyConnected(64)
    wiring2.build(IN_SIZE)
    cell_raw = M.TauTieLTCCellRaw(wiring2, in_features=IN_SIZE, ode_unfolds=ODE_UNFOLDS,
                                  implicit_param_constraints=True).to(DEV)
    cell_raw.load_state_dict(cell_fixed.state_dict(), strict=False)
    xm = torch.randn(4, 12, IN_SIZE, device=DEV)
    h0 = torch.zeros(4, 64, device=DEV)
    with torch.no_grad():
        # 裸 vs 修复：单步
        o1, _ = cell_fixed(xm[:, 0], h0, 1.0, tau=torch.tensor(10.0, device=DEV))
        o2, _ = cell_raw(xm[:, 0], h0, 1.0, tau=torch.tensor(10.0, device=DEV))
        res["raw_vs_fixed_single_step_max_abs_diff"] = float((o1 - o2).abs().max().item())
        # 裸 vs 修复：12 步序列
        def _run_seq(cell):
            h = h0.clone()
            outs = []
            for t in range(12):
                o, h = cell(xm[:, t], h, 1.0, tau=torch.tensor(10.0, device=DEV))
                outs.append(o)
            return torch.stack(outs, dim=1)
        s1 = _run_seq(cell_fixed)
        s2 = _run_seq(cell_raw)
        res["raw_vs_fixed_seq12_max_abs_diff"] = float((s1 - s2).abs().max().item())
        # 库锚点：stock LTCCell 且 cm:=gleak ↔ τ-tie(τ=1)（cm_t = softplus(gleak)/(Δt/u) 两边逐位相同）
        wiring3 = FullyConnected(64)
        wiring3.build(IN_SIZE)
        cell_stock = LTCCell(wiring3, in_features=IN_SIZE, ode_unfolds=ODE_UNFOLDS,
                             implicit_param_constraints=True).to(DEV)
        cell_stock.load_state_dict(cell_fixed.state_dict(), strict=False)
        cell_stock._params["cm"].data = cell_stock._params["gleak"].data.clone()
        os_, _ = cell_stock(xm[:, 0], h0, 1.0)
        ot, _ = cell_raw(xm[:, 0], h0, 1.0, tau=torch.tensor(1.0, device=DEV))
        res["stock_cm_eq_gleak_vs_tie_tau1_max_abs_diff"] = float((os_ - ot).abs().max().item())
    R["sections"]["O5_equivalence"] = res


# ---------------------------------------------------------------- O6
def sec_o6() -> None:
    """计时分解：单 unfold 元素级通路成本（B=16/64）——效率判读依据。"""
    res: dict = {}
    H = HIDDEN
    for b in (16, 64):
        v = torch.randn(b, H, device=DEV)
        mu = torch.randn(H, H, device=DEV)
        sig = torch.randn(H, H, device=DEV)
        w_eff = torch.randn(H, H, device=DEV)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            s = torch.sigmoid(sig * (v.unsqueeze(-1) - mu))
        torch.cuda.synchronize()
        t_sig = (time.perf_counter() - t0) / 5
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            w_act = w_eff * s
            num = torch.sum(w_act * mu, dim=1)
            den = torch.sum(w_act, dim=1)
        torch.cuda.synchronize()
        t_sum = (time.perf_counter() - t0) / 5
        res[f"unfold_parts_b{b}"] = {
            "sigmoid_broadcast_ms": round(t_sig * 1000, 2),
            "weighted_sums_ms": round(t_sum * 1000, 2),
            "per_unfold_estimate_ms": round((t_sig + t_sum) * 1000, 2),
            "tensor_mib": round(b * H * H * 4 / 1024 ** 2, 1),
        }
        del v, mu, sig, w_eff, s, w_act, num, den
        torch.cuda.empty_cache()
    R["sections"]["O6_timing_breakdown"] = res


SECTIONS = {"O0": sec_o0, "O1": sec_o1, "O2": sec_o2, "O3": sec_o3,
            "O4": sec_o4, "O5": sec_o5, "O6": sec_o6}


def _dump() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS.keys()))
    ap.add_argument("--o2-batches", default="16,32,64", help="O2 子集（默认 16,32,64）")
    ap.add_argument("--o3-batches", default="8,64", help="O3 子集（默认 8,64）")
    ap.add_argument("--b64-repeats", type=int, default=2, help="B=64 计时重复次数（默认 2）")
    args = ap.parse_args()
    want = [s.strip() for s in args.sections.split(",") if s.strip()]
    os.environ["BENCH_O2_BATCHES"] = args.o2_batches
    os.environ["BENCH_O3_BATCHES"] = args.o3_batches
    os.environ["BENCH_B64_REPEATS"] = str(args.b64_repeats)
    # 分段跑时合并已有段落（每段落盘，多进程/多次调用不互踩）
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, encoding="utf-8") as f:
                old = json.load(f)
            R["sections"].update(old.get("sections", {}))
        except Exception as e:  # noqa: BLE001
            log(f"[bench] 合并旧结果失败（忽略）: {e}")
    t0 = time.time()
    R["env"] = {"device": DEV, "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    _dump()
    for name in want:
        fn = SECTIONS[name]
        t = time.time()
        try:
            fn()
            log(f"[bench] {name} done ({time.time() - t:.1f}s)")
        except Exception as e:  # noqa: BLE001
            R["sections"][name + "_ERROR"] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
            log(f"[bench] {name} FAILED: {type(e).__name__}: {str(e)[:300]}")
            if DEV == "cuda":
                torch.cuda.empty_cache()
        _dump()  # 每段落盘，崩溃不丢前序结果
    R["total_sec"] = round(time.time() - t0, 1)
    _dump()
    log(f"[bench] wrote {OUT_JSON} total={R['total_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
