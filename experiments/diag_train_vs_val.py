# -*- coding: utf-8 -*-
"""diag_train_vs_val.py — 跑后诊断：ep6 ckpt 在 train 子采样 vs val 的 AUROC 对比。

动机：smoke 若出现 val AUROC<0.5，需区分「过拟合/分布差」（train 高 val 低）
vs「结构性错位」（train 也低 → 查预处理/标签/评测）。只读 ckpt + 数据，不训练。
用法：python experiments/diag_train_vs_val.py [--arm A --seed 0] [--epoch 6]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import mr_moe as M  # noqa: E402
from pilot_train import (DATA_DEFAULT, DATA_SEED, evaluate_with_loss,  # noqa: E402
                         load_arrays, make_subsample)
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def eval_on(model, x, y, batch=64):
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=batch, shuffle=False, num_workers=0)
    return evaluate_with_loss(model, loader, torch.device(DEV))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=6)
    ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "ckpt"))
    args = ap.parse_args()

    ck = torch.load(os.path.join(args.ckpt_dir, f"pilot_{args.arm}_s{args.seed}_ep{args.epoch}.pt"),
                    map_location=DEV, weights_only=False)
    cfg = ck["config"]
    data = load_arrays(DATA_DEFAULT)
    sub_idx = np.asarray(ck["sub_idx"])
    val_idx = np.asarray(data["splits"]["val"])
    n_feat = int(data["x"].shape[-1])

    model = M.build_model(n_feat, args.seed, tau_learnable=(args.arm == "B"),
                          hidden=cfg["hidden"]).to(DEV)
    model.load_state_dict(ck["model_state"])
    model.eval()

    tr = eval_on(model, data["x"][sub_idx], data["y"][sub_idx])
    va = eval_on(model, data["x"][val_idx], data["y"][val_idx])
    out = {
        "schema": "train-vs-val-diag/v1", "stage": "S3-pilot smoke 跑后诊断",
        "purpose": "区分过拟合/分布差 vs 结构性错位（train 子采样 vs val 的 AUROC 对比）",
        "run": f"{args.arm}_s{args.seed}", "ckpt_epoch": ck["epoch"],
        "train_sub": {"n": int(len(sub_idx)), **tr},
        "val": {"n": int(len(val_idx)), **va},
        "interpretation": (
            "train AUROC 高 ∧ val AUROC 低 → 过拟合/分布差（应急档 2000 条 1M 参数的"
            "预期风险，交 S3-pilot-continue 判读）；train AUROC 也低 → 结构性错位，"
            "查预处理/标签/评测实现"),
    }
    dst = os.path.join(PROJ, "results", f"diag_train_vs_val_{args.arm}_s{args.seed}.json")
    with open(dst + ".tmp", "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(dst + ".tmp", dst)
    print(f"train_sub: loss={tr['val_loss']} hourly_auroc={tr['hourly']['auroc']:.4f} "
          f"adm_auroc={tr['admission']['auroc']:.4f}")
    print(f"val:       loss={va['val_loss']} hourly_auroc={va['hourly']['auroc']:.4f} "
          f"adm_auroc={va['admission']['auroc']:.4f}")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
