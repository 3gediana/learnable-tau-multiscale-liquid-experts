#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_pipeline.py — S2b-data-model 步骤③：D3 冻结预处理管线（design.md §四 D3）。

D3 冻结内容（逐条对应实现）：
  1. 小时网格 dt=1h（PhysioNet 2019 原始即小时网格，ICULOS 为小时序号）；
  2. 每条入院截断前 72h（SEQ_LEN=72）；
  3. 缺失值前向填充 + 缺失 mask 作为输入通道拼接（禁裸 0 填充进前向/反向：
     观测值=原值，mask=1；填充值=前向最近观测，mask=0；从未观测=0，mask=0）；
  4. min-max 归一化只用 train 划分拟合（val/test 用 train 的 min/max，防泄漏）；
  5. Che et al. 2018 患者级随机 80/10/10 划分（种子固定 0，可复现）；
  6. 标签=Sepsis-3 onset，预测目标=未来 6h 内 onset（y(t)=1[onset∈(t,t+6h]]，
     PhysioNet 2019 的 SepsisLabel 列即该定义，直接沿用）；
  7. 输入通道=34 个时变临床变量（HR..Platelets）+ 34 个 mask 通道 = 68 维；
     Age/Gender/Unit1/Unit2/HospAdmTime/ICULOS 为静态/元数据列，不进时变输入
     （选择记录入 results/baseline_diff.md 口径：MR-MoE 输入 u(t) 为时变向量）。

用法：
  python experiments/data_pipeline.py --raw data/raw/physionet-2019 \
      --out data/processed/physionet2019_s${N}.npz
"""
from __future__ import annotations

import argparse
import csv
import io
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

SEQ_LEN = 72           # D3：前 72h
DT_HOURS = 1           # D3：小时网格
TRAIN_FRAC = 0.8       # Che 2018 80/10/10
VAL_FRAC = 0.1
SPLIT_SEED = 0         # D10 复现性：划分种子固定
FILL_VALUE = 0.0       # 前向填充的占位（mask=0 标注其为填充）

# PhysioNet/CinC 2019 的 41 列：前 34 列为时变临床变量，后 7 列为静态/元数据
TIME_VARYING = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess",
    "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
    "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]
STATIC_META = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"]


def load_psv(path: str) -> dict:
    """解析单个 .psv → values (L,F) / observed mask (L,F) / labels (L,)。"""
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh, delimiter="|") if r]
    header = rows[0]
    assert header[: len(TIME_VARYING)] == TIME_VARYING, f"列结构不符: {path}"
    li = header.index("SepsisLabel")
    L = len(rows) - 1
    F = len(TIME_VARYING)
    values = np.full((L, F), np.nan, dtype=np.float32)
    labels = np.zeros(L, dtype=np.float32)
    for i, r in enumerate(rows[1:]):
        for j in range(F):
            v = r[j]
            if v not in ("", "NA", "NaN"):
                values[i, j] = float(v)
        labels[i] = float(r[li]) if r[li] not in ("", "NA", "NaN") else 0.0
    return {"values": values, "labels": labels}


def forward_fill_with_mask(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """前向填充 + mask。observed=1 表示该格为真实观测；填充/从未观测=0。"""
    L, F = values.shape
    filled = np.array(values, dtype=np.float32, copy=True)
    mask = np.isfinite(values).astype(np.float32)
    last = np.zeros(F, dtype=np.float32)
    has_last = np.zeros(F, dtype=bool)
    for t in range(L):
        obs = mask[t] > 0
        if obs.any():
            last = np.where(obs, values[t], last)
            has_last = has_last | obs
        filled[t] = np.where(has_last, last, FILL_VALUE)
    return filled, mask


def truncate(values: np.ndarray, mask: np.ndarray, labels: np.ndarray,
             max_len: int = SEQ_LEN):
    L = values.shape[0]
    if L > max_len:
        return values[:max_len], mask[:max_len], labels[:max_len]
    if L == max_len:
        return values, mask, labels
    pad = max_len - L
    values = np.concatenate([values, np.zeros((pad, values.shape[1]), np.float32)])
    mask = np.concatenate([mask, np.zeros((pad, mask.shape[1]), np.float32)])
    labels = np.concatenate([labels, np.zeros(pad, np.float32)])
    return values, mask, labels


def minmax_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只用 train 拟合 min/max（D3-4；防泄漏）。"""
    lo = X.min(axis=(0, 1))
    hi = X.max(axis=(0, 1))
    span = np.where(hi - lo > 1e-8, hi - lo, 1.0)
    return lo, span


def minmax_apply(X: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (X - lo) / span


def che2018_split(n: int, seed: int = SPLIT_SEED) -> dict[str, np.ndarray]:
    """Che et al. 2018 患者级随机 80/10/10。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(n * TRAIN_FRAC))
    n_va = int(round(n * VAL_FRAC))
    return {
        "train": np.sort(perm[:n_tr]),
        "val": np.sort(perm[n_tr:n_tr + n_va]),
        "test": np.sort(perm[n_tr + n_va:]),
    }


def build_dataset(raw_dir: str, sets=("training_setA", "training_setB"),
                  max_len: int = SEQ_LEN, normalize: bool = True) -> dict:
    """加载全部已下载入院 → 前向填充/截断/min-max(train fit)/Che 划分。"""
    ids, Xs, Ms, Ys = [], [], [], []
    for s in sets:
        d = os.path.join(raw_dir, s)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".psv"):
                continue
            rec = load_psv(os.path.join(d, f))
            filled, mask = forward_fill_with_mask(rec["values"])
            v, m, y = truncate(filled, mask, rec["labels"], max_len)
            ids.append(f"{s}/{f[:-4]}")
            Xs.append(v)
            Ms.append(m)
            Ys.append(y)
    if not ids:
        raise RuntimeError(f"未找到数据: {raw_dir}")
    X = np.stack(Xs).astype(np.float32)   # (N, L, F)
    M = np.stack(Ms).astype(np.float32)   # (N, L, F)
    Y = np.stack(Ys).astype(np.float32)   # (N, L)
    splits = che2018_split(len(ids))
    if normalize:
        lo, span = minmax_fit(X[splits["train"]])
        X = minmax_apply(X, lo, span).astype(np.float32)
    else:
        lo = span = None
    y_adm = Y.max(axis=1)  # 入院级标签（任一小时为正）
    return {
        "ids": ids, "X": X, "M": M, "Y": Y, "y_admission": y_adm,
        "splits": splits, "minmax_lo": lo, "minmax_span": span,
        "config": {"seq_len": max_len, "dt_hours": DT_HOURS,
                   "n_features": X.shape[-1], "normalize": normalize,
                   "split": "che2018 patient-level 80/10/10 seed0"},
    }


def to_model_input(batch: dict, device=None) -> tuple["np.ndarray", "np.ndarray"]:
    """拼接 mask 通道：x = concat([X, M], dim=-1)（D3-3）。"""
    x = np.concatenate([batch["X"], batch["M"]], axis=-1).astype(np.float32)
    return x, batch["Y"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(PROJ, "data", "raw", "physionet-2019"))
    ap.add_argument("--out", default=os.path.join(PROJ, "data", "processed",
                                                  "physionet2019_sample.npz"))
    ap.add_argument("--max-len", type=int, default=SEQ_LEN)
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    ds = build_dataset(args.raw, max_len=args.max_len, normalize=not args.no_normalize)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        ids=np.array(ds["ids"]),
        X=ds["X"], M=ds["M"], Y=ds["Y"], y_admission=ds["y_admission"],
        train=ds["splits"]["train"], val=ds["splits"]["val"], test=ds["splits"]["test"],
        minmax_lo=ds["minmax_lo"] if ds["minmax_lo"] is not None else np.zeros(1),
        minmax_span=ds["minmax_span"] if ds["minmax_span"] is not None else np.ones(1),
    )
    n = len(ds["ids"])
    print(f"admissions={n} shape={ds['X'].shape} "
          f"train/val/test={len(ds['splits']['train'])}/{len(ds['splits']['val'])}/"
          f"{len(ds['splits']['test'])} "
          f"admission_positive_rate={ds['y_admission'].mean():.4f} "
          f"hour_positive_rate={ds['Y'].mean():.6f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
