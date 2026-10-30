# -*- coding: utf-8 -*-
"""compose_pilot_val_curves.py — 首里程碑③「两臂验证曲线」产物。

从真实落盘进度提取每 epoch 验证曲线（train/val loss + 小时级/入院级 AUROC/AUPRC）：
  results/pilot_progress.json      A/B 臂每 epoch 指标（pilot_train.py 原子落盘）
  results/pilot_lnn_progress.json  单体 LNN 每 epoch 指标（pilot_lnn_train.py 落盘）
输出（原子写；幂等）：
  results/pilot_val_curves.csv     长表：run_key × epoch
  results/pilot_val_curves.md      人读表（按 run 分组）

用法：python experiments/compose_pilot_val_curves.py
"""
import csv
import json
import os
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJ, "results")
CSV_OUT = os.path.join(RESULTS, "pilot_val_curves.csv")
MD_OUT = os.path.join(RESULTS, "pilot_val_curves.md")


def jload(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def main() -> int:
    prog = jload(os.path.join(RESULTS, "pilot_progress.json"), {"runs": {}}) or {"runs": {}}
    lnn = jload(os.path.join(RESULTS, "pilot_lnn_progress.json"), {"runs": {}}) or {"runs": {}}

    runs = []
    for r in prog.get("runs", {}).values():
        runs.append(r)
    for r in lnn.get("runs", {}).values():
        runs.append(r)
    runs.sort(key=lambda r: ({"A": 0, "B": 1, "LNN": 2}.get(r.get("arm"), 9),
                             r.get("seed", 0)))

    cols = ["run_key", "arm", "seed", "epoch", "train_loss_mean", "val_loss",
            "hourly_auroc", "hourly_auprc", "adm_auroc", "adm_auprc",
            "tau", "grad_global_norm_max", "epoch_sec"]
    rows = []
    for r in runs:
        for e in r.get("epochs", []):
            rows.append({
                "run_key": r["run_key"], "arm": r["arm"], "seed": r["seed"],
                "epoch": e.get("epoch"),
                "train_loss_mean": e.get("train_loss_mean"),
                "val_loss": e.get("val_loss"),
                "hourly_auroc": (e.get("hourly") or {}).get("auroc"),
                "hourly_auprc": (e.get("hourly") or {}).get("auprc"),
                "adm_auroc": (e.get("admission") or {}).get("auroc"),
                "adm_auprc": (e.get("admission") or {}).get("auprc"),
                "tau": json.dumps(e["tau"], ensure_ascii=False) if e.get("tau") else "",
                "grad_global_norm_max": e.get("grad_global_norm_max"),
                "epoch_sec": e.get("epoch_sec"),
            })

    tmp = CSV_OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, CSV_OUT)

    L = []
    L.append("# pilot 两臂验证曲线（每 epoch；val 302 条全量，D8）")
    L.append("")
    L.append(f"- 生成：{datetime.now().isoformat()}（experiments/compose_pilot_val_curves.py，幂等）")
    L.append("- 口径：D14 应急档 h256/B64/2000 条/E=6；A=固定 τ / B=保序可学习 τ / LNN=单体基线（D5(i)）")
    L.append("- 用途：admission.first_milestone ③「两臂验证曲线」；ep1 低谷=warm-up 瞬态"
             "（results/diag_train_vs_val_A_s0.json 实证非过拟合/非结构错位）")
    L.append("")
    L.append("| run | ep | train_loss | val_loss | hourly AUROC | hourly AUPRC | adm AUROC | adm AUPRC | τ | grad_max |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        L.append(f"| {row['run_key']} | {row['epoch']} | {row['train_loss_mean']} "
                 f"| {row['val_loss']} | {row['hourly_auroc']} | {row['hourly_auprc']} "
                 f"| {row['adm_auroc']} | {row['adm_auprc']} | {row['tau'] or '-'} "
                 f"| {row['grad_global_norm_max']} |")
    L.append("")
    L.append("- 数据源：results/pilot_progress.json（A/B 6 跑）+ results/pilot_lnn_progress.json（LNN）；全部原子落盘产物，无合成。")
    tmp = MD_OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    os.replace(tmp, MD_OUT)
    print(json.dumps({"wrote": [CSV_OUT, MD_OUT], "n_rows": len(rows),
                      "runs": [r["run_key"] for r in runs]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
