#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""emergency_tier_probe.py — D14 应急档实测探针（researcher 判读单元，2026-10-28 05:2x）。

动机：S2-gate-checks ⓪ 效率墙（F8）上交 researcher 判读 S3 路径。判读核心风险=
「降 hidden 应急档（D9 预留条款）是否真的让 pilot/full 预算可行」——operator_bench.md
⓪-4 自估 9-34× 隐含纯 H² 缩放假设，O6 又实测 ~2/3 单步时间为检查点 Python 开销
与小核启动（不随 hidden 缩小），两个证据方向相反，均未在 hidden<1500 实测过。
本探针把估算换成第一手测量，作为 D14 拍板依据。

测量（与 operator_bench.py _bench_case 同协议：reset_peak → 72 步 fwd+bwd →
max_memory_allocated / 中位计时；臂 B=可学习 τ，三专家+门控+读出全模型）：
  - 锚点：hidden 1500 / B=32（应对账 operator_bench O3 的 86s/batch、6863MiB）
  - 应急档：hidden 512 与 256 × B∈{32,64,128}（显存/计时/梯度有限性/∂L/∂c）
  - 每配置外推 epoch(2000/8000) 与 pilot 单训练（E=6）预估

产物：results/emergency_tier_probe.json。
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJECT = os.path.dirname(HERE)
RESULTS = os.path.join(PROJECT, "results")
OUT_JSON = os.path.join(RESULTS, "emergency_tier_probe.json")

import mr_moe as M  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 72
IN_SIZE = 80
PILOT_EPOCHS = 6
FULL_EPOCHS = 20


def log(*a):
    print(*a, flush=True)


def _finite(t) -> bool:
    return bool(torch.isfinite(t).all().item()) if t is not None else False


def _case(tag: str, hidden: int, batch: int, repeats: int) -> dict:
    """单配置：全模型臂 B，72 步 fwd+bwd，峰值显存 + 中位计时 + 梯度体检。"""
    out: dict = {"tag": tag, "hidden": hidden, "batch": batch, "seq_len": SEQ}
    torch.manual_seed(0)
    try:
        model = M.MRMoE(IN_SIZE, tau_learnable=True, hidden=hidden).to(DEV)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"build: {type(e).__name__}: {str(e)[:200]}"
        return out
    model.train()
    out["params"] = M.count_params(model)
    x = torch.randn(batch, SEQ, IN_SIZE, device=DEV)
    target = torch.randint(0, 2, (batch, SEQ), device=DEV).float()

    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    logits = model(x)
    out["forward_finite"] = _finite(logits)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    if DEV == "cuda":
        torch.cuda.synchronize()
    out["first_batch_fwd_bwd_sec"] = round(time.perf_counter() - t0, 3)
    if DEV == "cuda":
        out["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    gnorm = 0.0
    nbad = 0
    cg = None
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not _finite(g):
            nbad += 1
            continue
        gnorm += float(g.norm().item()) ** 2
        if name.endswith("tau_sched.c"):
            cg = round(float(g.norm().item()), 10)
    out["grad_global_norm"] = round(gnorm ** 0.5, 6) if nbad == 0 else None
    out["grad_nonfinite_param_count"] = nbad
    out["grad_c_norm"] = cg

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        lg = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, target)
        loss.backward()
        if DEV == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    out["batch_sec_median"] = round(med, 3)
    out["batch_sec_all"] = [round(t, 3) for t in times]
    out["per_step_ms"] = round(med * 1000 / SEQ, 2)
    out["epoch_estimate_sec"] = {
        f"n{n}": round(med * max(1, round(n / batch)), 1) for n in (8000, 4000, 2000)
    }
    out["pilot_train_estimate_sec_E6"] = round(
        out["epoch_estimate_sec"]["n2000"] * PILOT_EPOCHS, 1)
    out["full_train_estimate_sec_E20"] = round(
        out["epoch_estimate_sec"]["n2000"] * FULL_EPOCHS, 1)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    log(f"[probe] device={DEV} torch={torch.__version__} seq={SEQ} in={IN_SIZE}")
    R = {
        "schema": "emergency-tier-probe/v1",
        "stage": "S3-path-decision (D14, researcher judgment unit 2026-10-28)",
        "purpose": "hidden 应急档（512/256）× batch 档位实测：显存/计时/梯度，作为 D14 拍板依据",
        "protocol": "同 operator_bench._bench_case：reset_peak→72步fwd+bwd→max_memory_allocated/中位计时；臂B全模型（3专家+门控+读出）",
        "cases": [],
    }
    plan = [
        # (tag, hidden, batch, repeats) —— 慢案例少重复（首batch即一次实测），应急档多重复
        ("anchor_h1500_b32", 1500, 32, 1),
        ("emerg_h512_b32", 512, 32, 1),
        ("emerg_h512_b64", 512, 64, 1),
        ("emerg_h256_b32", 256, 32, 3),
        ("emerg_h256_b64", 256, 64, 3),
        ("emerg_h256_b128", 256, 128, 3),
    ]
    for tag, h, b, rep in plan:
        log(f"[probe] {tag} ...")
        t0 = time.perf_counter()
        rec = _case(tag, h, b, rep)
        rec["case_wall_sec"] = round(time.perf_counter() - t0, 1)
        R["cases"].append(rec)
        os.makedirs(RESULTS, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:  # 增量落盘：超时也不丢
            json.dump(R, f, ensure_ascii=False, indent=1)
        log(f"[probe] {tag} -> {json.dumps({k: rec.get(k) for k in ('peak_mem_mib','batch_sec_median','pilot_train_estimate_sec_E6','forward_finite','grad_c_norm')}, ensure_ascii=False)}")
    log(f"[probe] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
