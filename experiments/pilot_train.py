#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pilot_train.py — S3-pilot / S3-pilot-continue 训练脚本（D14 应急档）。

冻结配置（D14 + D15；改动=改冻结设计，须走 DECISIONS）：
  hidden=256 / batch=64 / ode_unfolds=6 / 子采样 2000 条（train 划分内固定种子抽样）
  / E=6 / Adam lr=1e-3（D4）/ BCEWithLogits / 无 grad clip（F3 监控：loss 非有限
  或全局梯度范数超 F3_GRAD_NORM_ALERT 即停并记录，阈值是监控选择非冻结常数）。
  臂 A：τ=(1,10,100) 固定（buffer）；臂 B：保序可学习 τ（D2）。两臂唯一差=
  tau_sched.c 是否 Parameter（A5 实测恰 3 标量）。
  --seed 控制模型初始化与 batch shuffle；数据子采样用独立固定种子 DATA_SEED
  （两臂同种子同数据 → D10 单变量纪律；子采样索引落 ckpt 可复现）。

评测（D8）：val 划分全量（302 条）小时级 + 入院级（窗口概率最大）AUROC/AUPRC。
τ 日志（D7）：每 epoch τ/增量/‖∂L/∂c‖（batch 均值）/相对移动。

产物（全部原子写：tmp + os.replace，中断可续）：
  experiments/ckpt/pilot_<arm>_s<seed>_ep<N>.pt   每 epoch ckpt（模型/优化器/配置/指标/RNG）
  results/pilot_progress.json                    按 arm/seed 分段的每 epoch 进度
  results/pilot_tau_log.json                     D7 τ 日志
  results/pilot_train_runs.json                  每跑实测计时（epoch/batch 秒、峰值显存）
  experiments/_pilot_<arm>_s<seed>.log           运行日志

用法：
  python experiments/pilot_train.py --arm A --seed 0 [--epochs 6] [--resume]
      [--hidden 256] [--batch 64] [--subsample 2000] [--data <npz>]
      [--no-init-check] [--ckpt-dir experiments/ckpt]
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

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJ = os.path.dirname(HERE)
RESULTS = os.path.join(PROJ, "results")
CKPT_DIR_DEFAULT = os.path.join(HERE, "ckpt")
DATA_DEFAULT = os.path.join(PROJ, "data", "processed", "physionet2019_sample.npz")
PROGRESS_JSON = os.path.join(RESULTS, "pilot_progress.json")
TAU_LOG_JSON = os.path.join(RESULTS, "pilot_tau_log.json")
RUNS_JSON = os.path.join(RESULTS, "pilot_train_runs.json")

DATA_SEED = 20261029       # 数据子采样种子（固定；所有臂/种子共用同一 2000 条）
F3_GRAD_NORM_ALERT = 1e3   # F3 梯度爆炸监控阈值（监控选择；实测量级 ~0.02-0.1）
LR = 1e-3                  # D4 论文 §3.3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(*a, flush=True)


def atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json_or(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def load_arrays(npz_path: str) -> dict:
    """加载 D3 预处理产物 → 模型输入 (N,L,2F) 与标签（train 子采样/val 全量）。"""
    z = np.load(npz_path, allow_pickle=False)
    x = np.concatenate([z["X"], z["M"]], axis=-1).astype(np.float32)
    y = z["Y"].astype(np.float32)
    y_adm = z["y_admission"].astype(np.float32)
    splits = {k: z[k] for k in ("train", "val", "test")}
    return {"x": x, "y": y, "y_adm": y_adm, "splits": splits,
            "ids": [str(v) for v in z["ids"]]}


def make_subsample(splits: dict, n: int, seed: int = DATA_SEED) -> np.ndarray:
    """train 划分内固定种子抽 n 条（排序保序；返回全局索引）。"""
    tr = np.asarray(splits["train"])
    n = min(n, len(tr))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tr), size=n, replace=False)
    return np.sort(tr[pick])


@torch.no_grad()
def evaluate_with_loss(model, loader, device) -> dict:
    """D8 评测 + val BCE（一遍前向同时算 loss 与指标）。"""
    model.eval()
    probs, ys, tot_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        tot_loss += float(F.binary_cross_entropy_with_logits(logits, y)) * x.size(0)
        n += x.size(0)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.cpu().numpy())
    if not probs:
        return {}
    p_h = np.concatenate([a.ravel() for a in probs])
    t_h = np.concatenate([a.ravel() for a in ys])
    p_a = np.concatenate([a.max(axis=1) for a in probs])
    t_a = np.concatenate([a.max(axis=1) for a in ys])
    return {
        "val_loss": round(tot_loss / max(n, 1), 6),
        "hourly": {"auroc": M.binary_auc(p_h, t_h), "auprc": M.binary_auprc(p_h, t_h),
                   "n": int(len(p_h)), "pos_rate": float(t_h.mean())},
        "admission": {"auroc": M.binary_auc(p_a, t_a), "auprc": M.binary_auprc(p_a, t_a),
                      "n": int(len(p_a)), "pos_rate": float(t_a.mean())},
    }


def init_check(x_real: torch.Tensor, n_feat: int, seed: int, hidden: int,
               device) -> dict:
    """D5(ii)：B(0)≡A(0) 在真实数据 batch 上复核（同种子同初始化）。"""
    xb = x_real[:8].to(device)
    torch.manual_seed(seed)
    ma = M.MRMoE(n_feat, k=3, hidden=hidden, tau_learnable=False).to(device)
    torch.manual_seed(seed)
    mb = M.MRMoE(n_feat, k=3, hidden=hidden, tau_learnable=True).to(device)
    with torch.no_grad():
        la, lb = ma(xb), mb(xb)
    dmax = float((la - lb).abs().max().item())
    audit = M.param_audit(ma, mb)
    out = {"statement": "D5(ii) B(0)≡A(0) 真实数据 batch 复核（h256 同档）",
           "batch": int(xb.size(0)), "logits_max_abs_diff": dmax,
           "pass_diff_lt_1e-6": bool(dmax < 1e-6),
           "param_audit_pass_exactly_3_scalars": bool(audit["pass_exactly_3_scalars"]),
           "param_audit": audit,
           "finite_both": bool(torch.isfinite(la).all() and torch.isfinite(lb).all())}
    del ma, mb
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def find_latest_ckpt(ckpt_dir: str, arm: str, seed: int):
    cands = []
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.startswith(f"pilot_{arm}_s{seed}_ep") and f.endswith(".pt"):
                try:
                    cands.append((int(f.rsplit("_ep", 1)[1].split(".")[0]),
                                  os.path.join(ckpt_dir, f)))
                except ValueError:
                    continue
    if not cands:
        return None, 0
    cands.sort()
    return cands[-1][1], cands[-1][0]


def train_run(args) -> int:
    t_start = time.perf_counter()
    arm = args.arm.upper()
    assert arm in ("A", "B"), f"--arm 必须是 A|B，得到 {args.arm}"
    seed = int(args.seed)
    run_key = f"{arm}_s{seed}"
    tau_learnable = (arm == "B")
    device = torch.device(DEV)
    log(f"[train] run={run_key} device={device} hidden={args.hidden} batch={args.batch} "
        f"subsample={args.subsample} epochs={args.epochs} lr={LR} tau_learnable={tau_learnable}")

    data = load_arrays(args.data)
    n_feat = int(data["x"].shape[-1])
    sub_idx = make_subsample(data["splits"], args.subsample)
    val_idx = np.asarray(data["splits"]["val"])
    x_tr = torch.from_numpy(data["x"][sub_idx])
    y_tr = torch.from_numpy(data["y"][sub_idx])
    x_va = torch.from_numpy(data["x"][val_idx])
    y_va = torch.from_numpy(data["y"][val_idx])
    log(f"[train] data: n_feat={n_feat} train_sub={len(sub_idx)} val={len(val_idx)} "
        f"hour_pos_rate={float(y_tr.mean()):.6f} adm_pos_rate={float(y_tr.max(1).values.mean()):.6f}")

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch,
                              shuffle=True, generator=gen, num_workers=0,
                              drop_last=False)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=args.batch,
                            shuffle=False, num_workers=0)

    # ---------- D5(ii) init 复核（真实数据 batch；⓿ A2 已覆盖随机输入版，这里补真实数据版）
    init_rec = None
    if args.init_check:
        t0 = time.perf_counter()
        init_rec = init_check(x_tr, n_feat, seed, args.hidden, device)
        init_rec["sec"] = round(time.perf_counter() - t0, 2)
        log(f"[train] init_check B(0)≡A(0) max_abs_diff={init_rec['logits_max_abs_diff']:.3e} "
            f"pass={init_rec['pass_diff_lt_1e-6']} ({init_rec['sec']}s)")

    # ---------- 模型/优化器（--resume 时加载最新 ckpt）
    model = M.build_model(n_feat, seed, tau_learnable=tau_learnable,
                          hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    start_epoch = 0
    ckpt_path, ckpt_epoch = (None, 0)
    if args.resume:
        ckpt_path, ckpt_epoch = find_latest_ckpt(args.ckpt_dir, arm, seed)
        if ckpt_path:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            if "opt_state" in ck:
                opt.load_state_dict(ck["opt_state"])
            start_epoch = int(ck["epoch"])
            # R6 (2026-10-29T04:2x researcher 判读单元修复): map_location=device 会把
            # CPU RNG state 一起搬到 GPU, 而 set_rng_state/set_rng_state_all 只收 CPU
            # ByteTensor → resume 必崩 (A_s2 实证 TypeError)。显式压回 CPU uint8
            # (已在 CPU 时恒等; 非 Tensor 时兜底转换, 兼容旧 ckpt)。
            _tr = ck["torch_rng"]
            if not isinstance(_tr, torch.Tensor):
                _tr = torch.tensor(_tr, dtype=torch.uint8)
            torch.set_rng_state(_tr.to(device="cpu", dtype=torch.uint8))
            if device.type == "cuda" and "cuda_rng" in ck:
                torch.cuda.set_rng_state_all(
                    [s.to(device="cpu", dtype=torch.uint8) for s in ck["cuda_rng"]])
            log(f"[train] resume from {ckpt_path} (epoch {start_epoch} done)")
    params = M.count_params(model)

    # ---------- 进度/计时记录初始化
    progress = read_json_or(PROGRESS_JSON, {"schema": "pilot-progress/v1",
                                            "stage": "S3-pilot (D15 smoke) / S3-pilot-continue",
                                            "runs": {}})
    progress.setdefault("runs", {})
    tau_log = read_json_or(TAU_LOG_JSON, {"schema": "pilot-tau-log/v1",
                                          "purpose": "D7 τ 日志：τ/增量/‖∂L/∂c‖/相对移动",
                                          "runs": {}})
    tau_log.setdefault("runs", {})
    runs_time = read_json_or(RUNS_JSON, {"schema": "pilot-train-runs/v1", "runs": {}})
    runs_time.setdefault("runs", {})

    run_rec = progress["runs"].get(run_key, {})
    run_rec.update({
        "arm": arm, "seed": seed, "tau_learnable": tau_learnable, "run_key": run_key,
        "status": "running", "started_at": datetime.now().isoformat(),
        "config": {"hidden": args.hidden, "batch": args.batch,
                   "ode_unfolds": M.ODE_UNFOLDS, "epochs": args.epochs,
                   "lr": LR, "subsample": int(len(sub_idx)),
                   "data": os.path.relpath(args.data, PROJ),
                   "data_seed": DATA_SEED, "seq_len": int(x_tr.shape[1]),
                   "n_feat": n_feat, "device": str(device),
                   "params": params, "torch": torch.__version__},
        "init_check": init_rec,
        "sub_idx_sha1": __import__("hashlib").sha1(
            sub_idx.tobytes()).hexdigest()[:12],
        "epochs": run_rec.get("epochs", []) if args.resume else [],
    })
    progress["runs"][run_key] = run_rec
    progress["updated_at"] = datetime.now().isoformat()
    atomic_write_json(PROGRESS_JSON, progress)

    tau_entries = tau_log["runs"].setdefault(
        run_key, {"arm": arm, "seed": seed, "tau_learnable": tau_learnable,
                  "entries": []})
    if not args.resume:
        tau_entries["entries"] = []

    epoch_secs: list[float] = []
    batch_secs_all: list[float] = []
    prev_tau = None
    f3 = {"triggered": False, "reason": None}
    peak_overall = 0.0

    for ep in range(start_epoch + 1, args.epochs + 1):
        t_ep = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        losses, gnorms, cgns = [], [], []
        for x, y in train_loader:
            t0 = time.perf_counter()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if not torch.isfinite(loss):
                f3 = {"triggered": True,
                      "reason": f"epoch {ep}: loss 非有限 ({float(loss)})"}
                log(f"[train] F3 触发: {f3['reason']}")
                break
            loss.backward()
            gnorm = float(sum(
                (float(p.grad.detach().norm().item()) ** 2)
                for p in model.parameters() if p.grad is not None) ** 0.5)
            cg = model.c_grad_norm()
            if not np.isfinite(gnorm) or gnorm > F3_GRAD_NORM_ALERT:
                f3 = {"triggered": True,
                      "reason": f"epoch {ep}: 全局梯度范数异常 gnorm={gnorm}"}
                log(f"[train] F3 触发: {f3['reason']}")
                break
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_secs_all.append(time.perf_counter() - t0)
            losses.append(float(loss))
            gnorms.append(gnorm)
            cgns.append(cg if cg is not None else float("nan"))
        if f3["triggered"]:
            break

        ev = evaluate_with_loss(model, val_loader, device)
        tau_rep = model.tau_report()
        tau_now = tau_rep["tau"]
        rel_move = None
        if prev_tau is not None:
            rel_move = [round(abs(a - b) / b, 8) for a, b in zip(tau_now, prev_tau)]
        prev_tau = tau_now
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == "cuda" else None
        peak_overall = max(peak_overall, peak or 0.0)
        ep_sec = time.perf_counter() - t_ep
        epoch_secs.append(ep_sec)

        cg_finite = [v for v in cgns if np.isfinite(v)]
        rec = {
            "epoch": ep,
            "train_loss_mean": round(float(np.mean(losses)), 6),
            "train_loss_first_last": [round(losses[0], 6), round(losses[-1], 6)],
            "grad_global_norm_mean": round(float(np.mean(gnorms)), 6),
            "grad_global_norm_max": round(float(np.max(gnorms)), 6),
            "c_grad_norm_mean": (round(float(np.mean(cg_finite)), 10)
                                 if cg_finite else None),
            "val_loss": ev.get("val_loss"),
            "hourly": ev.get("hourly"), "admission": ev.get("admission"),
            "tau": tau_now, "increments": tau_rep["increments"],
            "tau_rel_move_vs_prev_epoch": rel_move,
            "n_batches": len(losses), "batch_sec_median": round(
                float(np.median(batch_secs_all[-len(losses):])), 3),
            "epoch_sec": round(ep_sec, 2),
            "peak_mem_mib": round(peak, 1) if peak is not None else None,
            "loss_finite": bool(np.isfinite(losses).all()),
        }
        run_rec["epochs"].append(rec)
        run_rec["updated_at"] = datetime.now().isoformat()
        progress["updated_at"] = datetime.now().isoformat()
        atomic_write_json(PROGRESS_JSON, progress)

        tau_entries["entries"].append({
            "epoch": ep, "tau": tau_now, "increments": tau_rep["increments"],
            "c_grad_norm_mean": rec["c_grad_norm_mean"],
            "tau_rel_move_vs_prev_epoch": rel_move,
            "tau_order_strict_increasing": bool(
                all(a < b for a, b in zip(tau_now, tau_now[1:]))),
        })
        atomic_write_json(TAU_LOG_JSON, tau_log)

        # 每 epoch 原子 ckpt（tmp + replace）
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ck = {"epoch": ep, "arm": arm, "seed": seed, "run_key": run_key,
              "model_state": model.state_dict(), "opt_state": opt.state_dict(),
              "tau": tau_now, "config": run_rec["config"],
              "metrics": rec, "sub_idx": sub_idx.tolist(),
              "torch_rng": torch.get_rng_state(),
              "cuda_rng": (torch.cuda.get_rng_state_all()
                           if device.type == "cuda" else None)}
        final = os.path.join(args.ckpt_dir, f"pilot_{arm}_s{seed}_ep{ep}.pt")
        tmp = final + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, final)
        rec["ckpt"] = os.path.relpath(final, PROJ)
        atomic_write_json(PROGRESS_JSON, progress)
        log(f"[train] ep{ep} loss={rec['train_loss_mean']:.4f} "
            f"val_loss={rec['val_loss']} adm_auroc={rec['admission']['auroc']:.4f} "
            f"hourly_auroc={rec['hourly']['auroc']:.4f} tau={tau_now} "
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
    atomic_write_json(PROGRESS_JSON, progress)

    med_batch = round(float(np.median(batch_secs_all)), 3) if batch_secs_all else None
    runs_time["runs"][run_key] = {
        "arm": arm, "seed": seed, "status": status,
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
    atomic_write_json(RUNS_JSON, runs_time)
    log(f"[train] DONE run={run_key} status={status} wall={wall:.1f}s "
        f"epochs_done={len(eps)} batch_median={med_batch}s "
        f"peak={peak_overall:.1f}MiB")
    log(f"[train] wrote {PROGRESS_JSON} / {TAU_LOG_JSON} / {RUNS_JSON}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="A=固定 τ / B=保序可学习 τ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--subsample", type=int, default=2000)
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR_DEFAULT)
    ap.add_argument("--resume", action="store_true",
                    help="从最新 epoch ckpt 续跑（断点移交用）")
    ap.add_argument("--no-init-check", dest="init_check", action="store_false",
                    help="跳过 D5(ii) 真实数据 init 复核")
    args = ap.parse_args()
    return train_run(args)


if __name__ == "__main__":
    sys.exit(main())
