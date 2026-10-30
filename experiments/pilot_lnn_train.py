#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""pilot_lnn_train.py — S3-pilot-continue-d：单体 LNN seed0 sanity（D5(i)）。

目的（design.md D5(i) / admission U4 / workflow S3-pilot-continue-d 动作②）：
  论文 §3.2 "Monolithic LNN" 基线 = 无专家分解的单一 LNN。本脚本在**与 A/B 臂
  完全相同的应急档配置**下训练它，用作「固定 τ 的 MR-MoE（A 臂）优于单体 LNN」
  的方向性健全性检查（U4：不以复现论文绝对数值为目标）。单体 LNN 不是 A/B 臂
  的一员（D10 单变量纪律），不参与 H1a/H1b 配对。

同档匹配规模（D14 应急档；与 pilot_train.py 逐项一致）：
  hidden=256 / batch=64 / ode_unfolds=6 / 子采样 2000 条（data_seed 20261029，
  与 A/B 臂同一子采样，sha1 落盘对账）/ E=6 / Adam lr=1e-3 / BCEWithLogits /
  无 grad clip（F3 监控同阈值）/ seed 0（init + shuffle 同协议）。
  唯一差别=模型：mr_moe.SingleLNN（ncps 0.0.2 标准 LTCCell，非 τ-tie 子类）
  vs MRMoE。单体 LNN 无专家/无 τ → 无 τ 日志（D7 不适用），c_grad 记 N/A。

产物（全部原子写：tmp + os.replace，中断可续）：
  experiments/ckpt/pilot_LNN_s<seed>_ep<N>.pt   每 epoch ckpt
  results/pilot_lnn_progress.json               每 epoch 进度（本脚本专属）
  results/pilot_lnn_sanity.json                 最终 sanity 汇总 + A 臂对比
  results/pilot_train_runs.json                 计时记录（追加 LNN_s0，供
                                                compose_pilot_timing_v2.py 汇总）
  experiments/_pilot_lnn_s<seed>.log            运行日志（runner 由 tool 后台命令写）

用法：
  python experiments/pilot_lnn_train.py --seed 0 [--epochs 6] [--hidden 256]
      [--batch 64] [--subsample 2000] [--data <npz>] [--device auto] [--resume]
      [--ckpt-dir experiments/ckpt]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import mr_moe as M  # noqa: E402  (experiments/ 已在 sys.path)
import pilot_train as PT  # noqa: E402  复用 D3 数据路径/D8 评测/原子写（零改动）

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJ = os.path.dirname(HERE)
PROGRESS_JSON = os.path.join(PROJ, "results", "pilot_lnn_progress.json")
SANITY_JSON = os.path.join(PROJ, "results", "pilot_lnn_sanity.json")
CKPT_DIR_DEFAULT = os.path.join(HERE, "ckpt")
DATA_DEFAULT = os.path.join(PROJ, "data", "processed", "physionet2019_sample.npz")


def log(*a):
    print(*a, flush=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def a_arm_reference() -> dict:
    """A 臂（固定 τ MR-MoE）三种子末 epoch 参考：均值±std + 逐种子明细。"""
    prog = PT.read_json_or(PT.PROGRESS_JSON, {"runs": {}}) or {"runs": {}}
    rows = []
    for r in prog.get("runs", {}).values():
        if r.get("arm") != "A" or r.get("status") != "completed":
            continue
        eps = r.get("epochs", [])
        if not eps:
            continue
        last = eps[-1]
        rows.append({
            "run_key": r["run_key"], "seed": r["seed"],
            "val_loss": last.get("val_loss"),
            "hourly_auroc": (last.get("hourly") or {}).get("auroc"),
            "hourly_auprc": (last.get("hourly") or {}).get("auprc"),
            "adm_auroc": (last.get("admission") or {}).get("auroc"),
            "adm_auprc": (last.get("admission") or {}).get("auprc"),
            "train_loss_last": last.get("train_loss_mean"),
        })
    rows.sort(key=lambda r: r["seed"])

    def ms(key):
        xs = [r[key] for r in rows if r.get(key) is not None]
        if not xs:
            return None, None
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else None
        return mu, sd

    out = {"n_seeds": len(rows), "per_seed": rows}
    for key in ("val_loss", "hourly_auroc", "hourly_auprc", "adm_auroc", "adm_auprc"):
        mu, sd = ms(key)
        out[f"{key}_mean"] = None if mu is None else round(float(mu), 6)
        out[f"{key}_std"] = None if sd is None else round(float(sd), 6)
    return out


def train_run(args) -> int:
    t_start = time.perf_counter()
    seed = int(args.seed)
    run_key = f"LNN_s{seed}"
    device = resolve_device(args.device)
    log(f"[lnn] run={run_key} device={device} hidden={args.hidden} batch={args.batch} "
        f"subsample={args.subsample} epochs={args.epochs} lr={PT.LR} (D5(i) 单体 LNN sanity)")

    data = PT.load_arrays(args.data)
    n_feat = int(data["x"].shape[-1])
    sub_idx = PT.make_subsample(data["splits"], args.subsample)
    val_idx = np.asarray(data["splits"]["val"])
    x_tr = torch.from_numpy(data["x"][sub_idx])
    y_tr = torch.from_numpy(data["y"][sub_idx])
    x_va = torch.from_numpy(data["x"][val_idx])
    y_va = torch.from_numpy(data["y"][val_idx])
    log(f"[lnn] data: n_feat={n_feat} train_sub={len(sub_idx)} val={len(val_idx)} "
        f"sha1={__import__('hashlib').sha1(sub_idx.tobytes()).hexdigest()[:12]} "
        f"(与 A/B 臂同子采样对账)")

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch,
                              shuffle=True, generator=gen, num_workers=0,
                              drop_last=False)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=args.batch,
                            shuffle=False, num_workers=0)

    torch.manual_seed(seed)   # 与 A/B 臂同一 init 协议（同 seed 同数据 → 可配对）
    model = M.SingleLNN(n_feat, hidden=args.hidden,
                        ode_unfolds=M.ODE_UNFOLDS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=PT.LR)
    start_epoch = 0
    if args.resume:
        ckpt_path, ckpt_epoch = PT.find_latest_ckpt(args.ckpt_dir, "LNN", seed)
        if ckpt_path:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            if "opt_state" in ck:
                opt.load_state_dict(ck["opt_state"])
            start_epoch = int(ck["epoch"])
            # R6 同款修复：CPU RNG state 显式压回 CPU uint8
            _tr = ck["torch_rng"]
            if not isinstance(_tr, torch.Tensor):
                _tr = torch.tensor(_tr, dtype=torch.uint8)
            torch.set_rng_state(_tr.to(device="cpu", dtype=torch.uint8))
            if device.type == "cuda" and "cuda_rng" in ck:
                torch.cuda.set_rng_state_all(
                    [s.to(device="cpu", dtype=torch.uint8) for s in ck["cuda_rng"]])
            log(f"[lnn] resume from {ckpt_path} (epoch {start_epoch} done)")
    params = M.count_params(model)

    progress = PT.read_json_or(args.progress, {"schema": "pilot-lnn-progress/v1",
                                               "stage": "S3-pilot-continue-d (D5(i))",
                                               "runs": {}})
    progress.setdefault("runs", {})
    run_rec = progress["runs"].get(run_key, {})
    run_rec.update({
        "arm": "LNN", "seed": seed, "tau_learnable": None, "run_key": run_key,
        "model": "mr_moe.SingleLNN（ncps 0.0.2 标准 LTCCell；非 A/B 臂成员）",
        "status": "running", "started_at": datetime.now().isoformat(),
        "config": {"hidden": args.hidden, "batch": args.batch,
                   "ode_unfolds": M.ODE_UNFOLDS, "epochs": args.epochs,
                   "lr": PT.LR, "subsample": int(len(sub_idx)),
                   "data": os.path.relpath(args.data, PROJ),
                   "data_seed": PT.DATA_SEED, "seq_len": int(x_tr.shape[1]),
                   "n_feat": n_feat, "device": str(device),
                   "params": params, "torch": torch.__version__},
        "sub_idx_sha1": __import__("hashlib").sha1(sub_idx.tobytes()).hexdigest()[:12],
        "epochs": run_rec.get("epochs", []) if args.resume else [],
    })
    progress["runs"][run_key] = run_rec
    progress["updated_at"] = datetime.now().isoformat()
    PT.atomic_write_json(args.progress, progress)

    epoch_secs: list[float] = []
    batch_secs_all: list[float] = []
    f3 = {"triggered": False, "reason": None}
    peak_overall = 0.0

    for ep in range(start_epoch + 1, args.epochs + 1):
        t_ep = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        losses, gnorms = [], []
        for x, y in train_loader:
            t0 = time.perf_counter()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if not torch.isfinite(loss):
                f3 = {"triggered": True,
                      "reason": f"epoch {ep}: loss 非有限 ({float(loss)})"}
                log(f"[lnn] F3 触发: {f3['reason']}")
                break
            loss.backward()
            gnorm = float(sum(
                (float(p.grad.detach().norm().item()) ** 2)
                for p in model.parameters() if p.grad is not None) ** 0.5)
            if not np.isfinite(gnorm) or gnorm > PT.F3_GRAD_NORM_ALERT:
                f3 = {"triggered": True,
                      "reason": f"epoch {ep}: 全局梯度范数异常 gnorm={gnorm}"}
                log(f"[lnn] F3 触发: {f3['reason']}")
                break
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_secs_all.append(time.perf_counter() - t0)
            losses.append(float(loss))
            gnorms.append(gnorm)
        if f3["triggered"]:
            break

        ev = PT.evaluate_with_loss(model, val_loader, device)
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == "cuda" else None
        peak_overall = max(peak_overall, peak or 0.0)
        ep_sec = time.perf_counter() - t_ep
        epoch_secs.append(ep_sec)

        rec = {
            "epoch": ep,
            "train_loss_mean": round(float(np.mean(losses)), 6),
            "train_loss_first_last": [round(losses[0], 6), round(losses[-1], 6)],
            "grad_global_norm_mean": round(float(np.mean(gnorms)), 6),
            "grad_global_norm_max": round(float(np.max(gnorms)), 6),
            "c_grad_norm_mean": None,   # 单体 LNN 无专家 τ（D7 不适用）
            "val_loss": ev.get("val_loss"),
            "hourly": ev.get("hourly"), "admission": ev.get("admission"),
            "tau": None,                # 无专家时间常数（非 A/B 臂成员）
            "n_batches": len(losses),
            "batch_sec_median": round(float(np.median(batch_secs_all[-len(losses):])), 3),
            "epoch_sec": round(ep_sec, 2),
            "peak_mem_mib": round(peak, 1) if peak is not None else None,
            "loss_finite": bool(np.isfinite(losses).all()),
        }
        run_rec["epochs"].append(rec)
        run_rec["updated_at"] = datetime.now().isoformat()
        progress["updated_at"] = datetime.now().isoformat()
        PT.atomic_write_json(args.progress, progress)

        os.makedirs(args.ckpt_dir, exist_ok=True)
        ck = {"epoch": ep, "arm": "LNN", "seed": seed, "run_key": run_key,
              "model_state": model.state_dict(), "opt_state": opt.state_dict(),
              "config": run_rec["config"], "metrics": rec,
              "sub_idx": sub_idx.tolist(),
              "torch_rng": torch.get_rng_state(),
              "cuda_rng": (torch.cuda.get_rng_state_all()
                           if device.type == "cuda" else None)}
        final = os.path.join(args.ckpt_dir, f"pilot_LNN_s{seed}_ep{ep}.pt")
        tmp = final + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, final)
        rec["ckpt"] = os.path.relpath(final, PROJ)
        PT.atomic_write_json(args.progress, progress)
        log(f"[lnn] ep{ep} loss={rec['train_loss_mean']:.4f} "
            f"val_loss={rec['val_loss']} adm_auroc={rec['admission']['auroc']:.4f} "
            f"hourly_auroc={rec['hourly']['auroc']:.4f} "
            f"ep_sec={rec['epoch_sec']} peak={rec['peak_mem_mib']}MiB")

    wall = time.perf_counter() - t_start
    status = "f3_stopped" if f3["triggered"] else (
        "completed" if len(run_rec["epochs"]) >= args.epochs else "partial")
    run_rec["status"] = status
    run_rec["f3"] = f3
    run_rec["finished_at"] = datetime.now().isoformat()
    eps = run_rec["epochs"]
    if len(eps) >= 2:
        run_rec["loss_decrease_ep1_to_last"] = round(
            eps[0]["train_loss_mean"] - eps[-1]["train_loss_mean"], 6)
        run_rec["val_loss_decrease_ep1_to_last"] = round(
            (eps[0]["val_loss"] or 0) - (eps[-1]["val_loss"] or 0), 6)
    PT.atomic_write_json(args.progress, progress)

    med_batch = round(float(np.median(batch_secs_all)), 3) if batch_secs_all else None

    # ---- 计时追加（pilot_train_runs.json；compose_pilot_timing_v2.py 真源）----
    runs_time = PT.read_json_or(PT.RUNS_JSON, {"schema": "pilot-train-runs/v1",
                                               "runs": {}})
    runs_time.setdefault("runs", {})
    runs_time["runs"][run_key] = {
        "arm": "LNN", "seed": seed, "status": status,
        "wall_sec": round(wall, 1), "n_epochs_done": len(eps),
        "epoch_sec": [e["epoch_sec"] for e in eps],
        "batch_sec_median": med_batch,
        "n_batches_per_epoch": (len(batch_secs_all) // max(len(eps), 1)) if eps else 0,
        "peak_mem_mib_overall": round(peak_overall, 1),
        "per_epoch_estimate_sec_E6_extrapolated": (
            round(float(np.mean(epoch_secs)) * args.epochs, 1) if epoch_secs else None),
        "f3": f3, "finished_at": datetime.now().isoformat(),
        "config": run_rec["config"],
    }
    PT.atomic_write_json(PT.RUNS_JSON, runs_time)

    # ---- sanity 汇总 + A 臂对比（D5(i) 方向性判据：A 臂优于单体 LNN）----
    last = eps[-1] if eps else {}
    a_ref = a_arm_reference()
    lnn_block = {
        "val_loss": last.get("val_loss"),
        "hourly_auroc": (last.get("hourly") or {}).get("auroc"),
        "hourly_auprc": (last.get("hourly") or {}).get("auprc"),
        "adm_auroc": (last.get("admission") or {}).get("auroc"),
        "adm_auprc": (last.get("admission") or {}).get("auprc"),
        "train_loss_ep1": eps[0]["train_loss_mean"] if eps else None,
        "train_loss_last": last.get("train_loss_mean"),
    }

    def cmp_key(key):
        lv = lnn_block.get(key)
        av = a_ref.get(f"{key}_mean")
        if lv is None or av is None:
            return None
        return round(float(av) - float(lv), 6)   # A − LNN（>0 = A 臂更优）

    comparison = {
        "statement": "D5(i)/U4 方向性：固定 τ 的 MR-MoE（A 臂）应优于单体 LNN；"
                     "不是 H1a 受控配对（架构不同），仅健全性检查",
        "a_minus_lnn": {k: cmp_key(k) for k in
                        ("adm_auroc", "adm_auprc", "hourly_auroc", "hourly_auprc",
                         "val_loss")},
        "a_arm_reference": a_ref,
        "a_vs_lnn_same_seed0": None,
    }
    a_s0 = next((r for r in a_ref["per_seed"] if r["seed"] == seed), None)
    if a_s0:
        comparison["a_vs_lnn_same_seed0"] = {
            "adm_auroc_a_minus_lnn": (
                None if (a_s0["adm_auroc"] is None or lnn_block["adm_auroc"] is None)
                else round(float(a_s0["adm_auroc"]) - float(lnn_block["adm_auroc"]), 6)),
            "hourly_auroc_a_minus_lnn": (
                None if (a_s0["hourly_auroc"] is None or lnn_block["hourly_auroc"] is None)
                else round(float(a_s0["hourly_auroc"]) - float(lnn_block["hourly_auroc"]), 6)),
        }
    # 方向性判定：AUROC 两项 A>LNN 即记 pass（val_loss 反向指标另看）
    auroc_ok = (comparison["a_minus_lnn"]["adm_auroc"] is not None
                and comparison["a_minus_lnn"]["hourly_auroc"] is not None
                and comparison["a_minus_lnn"]["adm_auroc"] > 0
                and comparison["a_minus_lnn"]["hourly_auroc"] > 0)
    comparison["directional_pass_a_gt_lnn_auroc"] = bool(auroc_ok)

    sanity = {
        "schema": "pilot-lnn-sanity/v1",
        "stage": "S3-pilot-continue-d",
        "purpose": "D5(i) 单体 LNN seed0 sanity：与 A/B 臂同应急档配置，"
                   "验证「固定 τ 的 MR-MoE 优于单体 LNN」的方向性健全性（U4）",
        "ran_at": datetime.now().isoformat(),
        "run_key": run_key, "status": status,
        "config": run_rec["config"],
        "sub_idx_sha1": run_rec["sub_idx_sha1"],
        "params": params,
        "f3": f3,
        "wall_sec": round(wall, 1),
        "batch_sec_median": med_batch,
        "peak_mem_mib_overall": round(peak_overall, 1),
        "epoch_sec": [e["epoch_sec"] for e in eps],
        "epochs": eps,
        "summary": lnn_block,
        "loss_decrease_ep1_to_last": run_rec.get("loss_decrease_ep1_to_last"),
        "val_loss_decrease_ep1_to_last": run_rec.get("val_loss_decrease_ep1_to_last"),
        "comparison_vs_arm_a": comparison,
        "notes": [
            "单体 LNN = mr_moe.SingleLNN（ncps 0.0.2 标准 LTCCell，非 τ-tie 子类）；"
            "hidden 256 与 A/B 臂同档匹配规模（D14 应急档）",
            "无专家/无 τ → 无 τ 日志、c_grad=N/A（D7 不适用）；不是 A/B 臂成员",
            "同 seed 0 / 同子采样（data_seed 20261029）/ 同 lr,batch,E → 与 A_s0 数据顺序一致",
            "结论限应急档规模 + PhysioNet 2019 回退源；S6 终判须标注",
        ],
    }
    PT.atomic_write_json(args.out, sanity)

    log(f"[lnn] DONE run={run_key} status={status} wall={wall:.1f}s "
        f"epochs_done={len(eps)} batch_median={med_batch}s peak={peak_overall:.1f}MiB")
    log(f"[lnn] wrote {args.progress} / {args.out} / {PT.RUNS_JSON}")
    log(f"[lnn] A−LNN: {json.dumps(comparison['a_minus_lnn'], ensure_ascii=False)} "
        f"directional_pass={comparison['directional_pass_a_gt_lnn_auroc']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR_DEFAULT)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--resume", action="store_true",
                    help="从最新 epoch ckpt 续跑（断点移交用）")
    ap.add_argument("--progress", default=PROGRESS_JSON)
    ap.add_argument("--out", default=SANITY_JSON)
    args = ap.parse_args()
    return train_run(args)


if __name__ == "__main__":
    sys.exit(main())
