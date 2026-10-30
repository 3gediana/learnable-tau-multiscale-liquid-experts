# -*- coding: utf-8 -*-
"""compose_pilot_table.py — pilot 结论表（固定 τ vs 可学习 τ）→ results/pilot_fixed_vs_learnable.{csv,md}。

数据真源（全部真实落盘产物，无合成）：
  results/pilot_progress.json     每 arm/seed 每 epoch 指标（pilot_train.py 原子落盘）
  results/pilot_train_runs.json   每跑 wall/状态/F3（pilot_train.py 落盘）
  results/pilot_tau_log.json      D7 τ 日志
  results/pilot_lnn_sanity.json   单体 LNN sanity（continue-d 才跑；不存在则记 pending）

口径（与 design.md / D14 / S4 预注册一致）：
  - H1a = B 臂三种子平均 ΔAUROC ≥+0.01 ∧ ΔAUPRC ≥−0.01（B−A 逐种子配对）；
    本表在 B 种子未齐时只给「已到种子」的 mean±std 并显式记 pending，不终判。
  - H1b 预读 = τ 序保持/无塌缩/τ 在学（相对移动）；终判交 S4。
  - 一切数字限 D14 应急档（h256/B64/2000 条/E=6，可训练 ~1.0M）+ PhysioNet 2019
    回退源，S6 终判须标注。

用法：
  python experiments/compose_pilot_table.py [--progress results/pilot_progress.json]
      [--runs results/pilot_train_runs.json] [--lnn results/pilot_lnn_sanity.json]
幂等，可重复运行；continue-c/d 拿到 B_s2/LNN 后重跑即得完整表。
"""
import argparse
import csv
import json
import os
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJ, "results")
CSV_OUT = os.path.join(RESULTS, "pilot_fixed_vs_learnable.csv")
MD_OUT = os.path.join(RESULTS, "pilot_fixed_vs_learnable.md")

TAU_INIT = [1.0, 10.0, 100.0]          # D2 init (1,10,100)
H1A_D_AUROC = 0.01                      # design.md H1a
H1A_D_AUPRC = -0.01
EXPECTED_SEEDS = [0, 1, 2]              # A/B 各三种子


def jload(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def f6(x):
    return None if x is None else round(float(x), 6)


def f4(x):
    return None if x is None else round(float(x), 4)


def run_summary(run: dict, runs_time: dict) -> dict:
    """从 progress run 记录提取末 epoch 汇总行。"""
    eps = run.get("epochs", [])
    last = eps[-1] if eps else {}
    first = eps[0] if eps else {}
    rt = runs_time.get(run["run_key"], {})
    tau_final = last.get("tau")
    rel_total = None
    if tau_final:
        rel_total = [round(abs(a - b) / b, 6) for a, b in zip(tau_final, TAU_INIT)]
    cgs = [e.get("c_grad_norm_mean") for e in eps if e.get("c_grad_norm_mean") is not None]
    order_ok = all(
        all(a < b for a, b in zip(e["tau"], e["tau"][1:]))
        for e in eps if e.get("tau"))
    return {
        "run_key": run["run_key"], "arm": run["arm"], "seed": run["seed"],
        "tau_learnable": run["tau_learnable"], "status": run.get("status"),
        "n_epochs_done": len(eps),
        "train_loss_ep1": f6(first.get("train_loss_mean")),
        "train_loss_last": f6(last.get("train_loss_mean")),
        "val_loss_last": f6(last.get("val_loss")),
        "hourly_auroc_last": f4((last.get("hourly") or {}).get("auroc")),
        "hourly_auprc_last": f6((last.get("hourly") or {}).get("auprc")),
        "adm_auroc_last": f4((last.get("admission") or {}).get("auroc")),
        "adm_auprc_last": f6((last.get("admission") or {}).get("auprc")),
        "tau_final": [round(t, 6) for t in tau_final] if tau_final else None,
        "tau_rel_move_total_vs_init": rel_total,
        "tau_order_strict_increasing_all_epochs": bool(order_ok),
        "c_grad_norm_first": cgs[0] if cgs else None,
        "c_grad_norm_last": cgs[-1] if cgs else None,
        "grad_global_norm_max_max": max(
            (e.get("grad_global_norm_max") or 0.0) for e in eps) if eps else None,
        "loss_finite_all": all(e.get("loss_finite") for e in eps) if eps else None,
        "f3_triggered": (run.get("f3") or {}).get("triggered"),
        "init_check_logits_max_abs_diff": (run.get("init_check") or {}).get(
            "logits_max_abs_diff"),
        "sub_idx_sha1": run.get("sub_idx_sha1"),
        "wall_sec": rt.get("wall_sec"),
        "batch_sec_median": rt.get("batch_sec_median"),
        "peak_mem_mib_overall": rt.get("peak_mem_mib_overall"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", default=os.path.join(RESULTS, "pilot_progress.json"))
    ap.add_argument("--runs", default=os.path.join(RESULTS, "pilot_train_runs.json"))
    ap.add_argument("--lnn", default=os.path.join(RESULTS, "pilot_lnn_sanity.json"))
    args = ap.parse_args()

    prog = jload(args.progress, {"runs": {}}) or {"runs": {}}
    runs_time = ((jload(args.runs, {}) or {}).get("runs", {}) or {})
    lnn = jload(args.lnn)

    rows = [run_summary(r, runs_time) for r in prog["runs"].values()]
    rows.sort(key=lambda r: (r["arm"], r["seed"]))
    by = {(r["arm"], r["seed"]): r for r in rows}

    # ---- 逐种子配对 B−A（只对两臂都 completed 的种子）----
    pairs = []
    for s in EXPECTED_SEEDS:
        a, b = by.get(("A", s)), by.get(("B", s))
        if not a or not b or a["status"] != "completed" or b["status"] != "completed":
            continue
        pairs.append({
            "seed": s,
            "d_adm_auroc": round(b["adm_auroc_last"] - a["adm_auroc_last"], 6),
            "d_adm_auprc": round(b["adm_auprc_last"] - a["adm_auprc_last"], 6),
            "d_hourly_auroc": round(b["hourly_auroc_last"] - a["hourly_auroc_last"], 6),
            "d_hourly_auprc": round(b["hourly_auprc_last"] - a["hourly_auprc_last"], 6),
            "d_val_loss": round(b["val_loss_last"] - a["val_loss_last"], 6),
            "d_train_loss_last": round(b["train_loss_last"] - a["train_loss_last"], 6),
        })

    def agg(arm):
        rs = [r for r in rows if r["arm"] == arm and r["status"] == "completed"]
        return {
            "n": len(rs),
            "adm_auroc_mean": f4(mean([r["adm_auroc_last"] for r in rs])),
            "adm_auroc_std": f4(std([r["adm_auroc_last"] for r in rs])),
            "adm_auprc_mean": f6(mean([r["adm_auprc_last"] for r in rs])),
            "adm_auprc_std": f6(std([r["adm_auprc_last"] for r in rs])),
            "hourly_auroc_mean": f4(mean([r["hourly_auroc_last"] for r in rs])),
            "hourly_auroc_std": f4(std([r["hourly_auroc_last"] for r in rs])),
            "hourly_auprc_mean": f6(mean([r["hourly_auprc_last"] for r in rs])),
            "hourly_auprc_std": f6(std([r["hourly_auprc_last"] for r in rs])),
            "val_loss_mean": f6(mean([r["val_loss_last"] for r in rs])),
            "val_loss_std": f6(std([r["val_loss_last"] for r in rs])),
            "train_loss_last_mean": f6(mean([r["train_loss_last"] for r in rs])),
        }

    agg_a, agg_b = agg("A"), agg("B")
    d_agg = {
        "n_paired_seeds": len(pairs),
        "d_adm_auroc_mean": f4(mean([p["d_adm_auroc"] for p in pairs])),
        "d_adm_auroc_std": f4(std([p["d_adm_auroc"] for p in pairs])),
        "d_adm_auprc_mean": f6(mean([p["d_adm_auprc"] for p in pairs])),
        "d_adm_auprc_std": f6(std([p["d_adm_auprc"] for p in pairs])),
        "d_hourly_auroc_mean": f4(mean([p["d_hourly_auroc"] for p in pairs])),
        "d_hourly_auroc_std": f4(std([p["d_hourly_auroc"] for p in pairs])),
        "d_hourly_auprc_mean": f6(mean([p["d_hourly_auprc"] for p in pairs])),
        "d_hourly_auprc_std": f6(std([p["d_hourly_auprc"] for p in pairs])),
    }

    b_seeds_done = sorted(r["seed"] for r in rows
                          if r["arm"] == "B" and r["status"] == "completed")
    a_seeds_done = sorted(r["seed"] for r in rows
                          if r["arm"] == "A" and r["status"] == "completed")
    pending = {
        "b_seeds_missing": [s for s in EXPECTED_SEEDS if s not in b_seeds_done],
        "a_seeds_missing": [s for s in EXPECTED_SEEDS if s not in a_seeds_done],
        "lnn_sanity": "pending" if lnn is None else "present",
    }
    h1a_provisional = None
    if pairs:
        h1a_provisional = (
            f"provisional ({len(pairs)}/3 种子，不终判): "
            f"Δadm_AUROC mean={d_agg['d_adm_auroc_mean']} (阈值 ≥+{H1A_D_AUROC}), "
            f"Δadm_AUPRC mean={d_agg['d_adm_auprc_mean']} (阈值 ≥{H1A_D_AUPRC})")

    # ---- CSV ----
    os.makedirs(RESULTS, exist_ok=True)
    cols = ["kind", "run_key", "arm", "seed", "tau_learnable", "status", "n_epochs_done",
            "train_loss_ep1", "train_loss_last", "val_loss_last",
            "hourly_auroc_last", "hourly_auprc_last",
            "adm_auroc_last", "adm_auprc_last",
            "tau_final", "tau_rel_move_total_vs_init",
            "tau_order_strict_increasing_all_epochs",
            "c_grad_norm_first", "c_grad_norm_last", "grad_global_norm_max_max",
            "loss_finite_all", "f3_triggered", "init_check_logits_max_abs_diff",
            "sub_idx_sha1", "wall_sec", "batch_sec_median", "peak_mem_mib_overall"]
    tmp = CSV_OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["run"] + [
                json.dumps(r[c], ensure_ascii=False) if isinstance(r[c], (list, dict))
                else r[c] for c in cols[1:]])
        # 配对/聚合行：值一律落进与列头语义相同的列（kind 列区分行型）
        for p in pairs:
            w.writerow(["paired_diff_B_minus_A", f"seed{p['seed']}", "B-A", p["seed"],
                        "", "", "", "", p["d_train_loss_last"], p["d_val_loss"],
                        p["d_hourly_auroc"], p["d_hourly_auprc"],
                        p["d_adm_auroc"], p["d_adm_auprc"]] + [""] * 13)
        for tag, a in (("agg_A", agg_a), ("agg_B", agg_b)):
            w.writerow([tag, "", tag[-1], "", "", "", a["n"], "",
                        a["train_loss_last_mean"], a["val_loss_mean"],
                        a["hourly_auroc_mean"], a["hourly_auprc_mean"],
                        a["adm_auroc_mean"], a["adm_auprc_mean"]] + [""] * 13)
            w.writerow([tag + "_std", "", tag[-1], "", "", "", a["n"], "",
                        "", "", a["hourly_auroc_std"], a["hourly_auprc_std"],
                        a["adm_auroc_std"], a["adm_auprc_std"]] + [""] * 13)
        w.writerow(["agg_delta_B_minus_A", "", "B-A", "", "", "",
                    d_agg["n_paired_seeds"], "", "", "",
                    d_agg["d_hourly_auroc_mean"], d_agg["d_hourly_auprc_mean"],
                    d_agg["d_adm_auroc_mean"], d_agg["d_adm_auprc_mean"]] + [""] * 13)
        w.writerow(["agg_delta_B_minus_A_std", "", "B-A", "", "", "",
                    d_agg["n_paired_seeds"], "", "", "",
                    d_agg["d_hourly_auroc_std"], d_agg["d_hourly_auprc_std"],
                    d_agg["d_adm_auroc_std"], d_agg["d_adm_auprc_std"]] + [""] * 13)
    os.replace(tmp, CSV_OUT)

    # ---- MD ----
    L = []
    L.append("# pilot 结论表：固定 τ（A）vs 保序可学习 τ（B）")
    L.append("")
    L.append(f"- 生成：{datetime.now().isoformat()}（experiments/compose_pilot_table.py，幂等可重跑）")
    L.append("- 口径：D14 应急档 hidden 256 / batch 64 / ode_unfolds 6 / 子采样 2000 条"
             "（data_seed 20261029，与 smoke 同子采样）/ E=6 / Adam lr=1e-3；"
             "数据=PhysioNet/CinC 2019 回退源（D3）；可训练参数 ~1.0M（28.5M→1.0M 规模偏离记账，S6 终判须标注）")
    L.append("- H1a 预注册（design.md）：B 臂三种子平均 ΔAUROC ≥+0.01 ∧ ΔAUPRC ≥−0.01；"
             "H1b 预注册：τ 序保持/无塌缩/τ 在学。**本表不终判 H1a/H1b（交 S4-pilot-verdict）**。")
    L.append(f"- 进度：A 种子 {a_seeds_done}/{EXPECTED_SEEDS}，B 种子 {b_seeds_done}/{EXPECTED_SEEDS}；"
             f"缺失 B {pending['b_seeds_missing']}（continue-c）；单体 LNN sanity {pending['lnn_sanity']}"
             + ("（continue-d）" if lnn is None else ""))
    if h1a_provisional:
        L.append(f"- H1a 临时读数（{h1a_provisional}）")
    L.append("")
    L.append("## 每跑汇总（末 epoch；val 302 条全量，D8）")
    L.append("")
    L.append("| run | arm | seed | 状态 | ep | train_loss(首→末) | val_loss | hourly AUROC | hourly AUPRC | adm AUROC | adm AUPRC | τ_final | τ 相对移动(vs init) | c_grad 首→末 | wall s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        tl = f"{r['train_loss_ep1']}→{r['train_loss_last']}" if r["train_loss_ep1"] is not None else "-"
        cg = "-" if r["c_grad_norm_first"] is None else f"{r['c_grad_norm_first']:.2e}→{r['c_grad_norm_last']:.2e}"
        L.append(f"| {r['run_key']} | {r['arm']} | {r['seed']} | {r['status']} | {r['n_epochs_done']} "
                 f"| {tl} | {r['val_loss_last']} | {r['hourly_auroc_last']} | {r['hourly_auprc_last']} "
                 f"| {r['adm_auroc_last']} | {r['adm_auprc_last']} | {r['tau_final']} "
                 f"| {r['tau_rel_move_total_vs_init']} | {cg} | {r['wall_sec']} |")
    L.append("")
    L.append("## 逐种子配对 B−A（末 epoch 差）")
    L.append("")
    if pairs:
        L.append("| seed | Δadm AUROC | Δadm AUPRC | Δhourly AUROC | Δhourly AUPRC | Δval_loss | Δtrain_loss |")
        L.append("|---|---|---|---|---|---|---|")
        for p in pairs:
            L.append(f"| {p['seed']} | {p['d_adm_auroc']} | {p['d_adm_auprc']} "
                     f"| {p['d_hourly_auroc']} | {p['d_hourly_auprc']} "
                     f"| {p['d_val_loss']} | {p['d_train_loss_last']} |")
        L.append("")
        L.append(f"mean±std（n={len(pairs)}）：Δadm AUROC {d_agg['d_adm_auroc_mean']}±{d_agg['d_adm_auroc_std']}；"
                 f"Δadm AUPRC {d_agg['d_adm_auprc_mean']}±{d_agg['d_adm_auprc_std']}；"
                 f"Δhourly AUROC {d_agg['d_hourly_auroc_mean']}±{d_agg['d_hourly_auroc_std']}；"
                 f"Δhourly AUPRC {d_agg['d_hourly_auprc_mean']}±{d_agg['d_hourly_auprc_std']}")
    else:
        L.append("（暂无配对种子）")
    L.append("")
    L.append("## 臂级 mean±std（已 completed 种子）")
    L.append("")
    L.append("| 臂 | n | adm AUROC | adm AUPRC | hourly AUROC | hourly AUPRC | val_loss |")
    L.append("|---|---|---|---|---|---|---|")
    for tag, a in (("A 固定 τ", agg_a), ("B 可学习 τ", agg_b)):
        L.append(f"| {tag} | {a['n']} | {a['adm_auroc_mean']}±{a['adm_auroc_std']} "
                 f"| {a['adm_auprc_mean']}±{a['adm_auprc_std']} "
                 f"| {a['hourly_auroc_mean']}±{a['hourly_auroc_std']} "
                 f"| {a['hourly_auprc_mean']}±{a['hourly_auprc_std']} | {a['val_loss_mean']}±{a['val_loss_std']} |")
    L.append("")
    L.append("## τ 轨迹（B 臂；D7 日志）")
    L.append("")
    tau_log = jload(os.path.join(RESULTS, "pilot_tau_log.json"), {"runs": {}}) or {"runs": {}}
    for k, v in tau_log.get("runs", {}).items():
        if not v.get("tau_learnable"):
            continue
        ents = v.get("entries", [])
        L.append(f"- {k}: " + " → ".join(
            f"ep{e['epoch']} τ={e['tau']} (c_grad={e['c_grad_norm_mean']}, "
            f"move={e['tau_rel_move_vs_prev_epoch']}, 序={e['tau_order_strict_increasing']})"
            for e in ents))
    L.append("")
    L.append("## 诚实性标注")
    L.append("")
    L.append("- 本表为阶段性汇总：B 臂种子齐（continue-c）+ LNN sanity（continue-d）后由同一脚本重跑得完整表；")
    L.append("- admission 级 val n=302≈17 正例，AUROC 噪声大（S4 给区间不点估计）；")
    L.append("- H1a/H1b 不终判；结论限应急档规模 + PhysioNet 2019 回退源；")
    L.append("- 若 LNN sanity 落盘，本表重跑后追加「A ≥ 单体 LNN」一行（预测判据）。")
    if lnn is not None:
        L.append(f"- LNN sanity 已在盘：{json.dumps(lnn, ensure_ascii=False)[:400]}")
    L.append("")
    tmp = MD_OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    os.replace(tmp, MD_OUT)

    print(json.dumps({"wrote": [CSV_OUT, MD_OUT],
                      "runs": [r["run_key"] + ":" + str(r["status"]) for r in rows],
                      "paired_seeds": [p["seed"] for p in pairs],
                      "agg_A": agg_a, "agg_B": agg_b, "agg_delta": d_agg,
                      "pending": pending, "h1a_provisional": h1a_provisional},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
