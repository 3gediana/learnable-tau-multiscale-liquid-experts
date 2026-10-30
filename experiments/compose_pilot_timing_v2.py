# -*- coding: utf-8 -*-
"""compose_pilot_timing_v2.py — 合并式计时汇总 → results/pilot_timing.json。

v1（compose_pilot_timing.py）把 stage/unit_window 硬编码为 smoke 段，重跑会用
smoke 标签覆盖后续阶段——故 continue 段起用本 v2：以盘上 pilot_timing.json 为
基座保留既有阶段块，把 pilot_train_runs.json（pilot_train.py 每跑结束原子落盘
的实测计时，唯一真源）按阶段标签汇总成 train_runs_all，并新增当前阶段块。

输入（全部真实落盘产物）：
  results/pilot_train_runs.json            每跑实测计时（pilot_train.py 落盘，真源）
  results/pilot_timing.json                既有阶段块（合并基座）
  experiments/_continue_a_runner.log       当前阶段 runner 起止时间（窗口对账；可选）
  results/emergency_tier_probe.json        D14 应急档基准（对账用）
输出：results/pilot_timing.json（原子写；幂等，可重复运行）。

用法：
  python experiments/compose_pilot_timing_v2.py [--stage S3-pilot-continue-a]
      [--block continue_a] [--runner-log experiments/_continue_a_runner.log]
continue-b 复用：--stage S3-pilot-continue-b --block continue_b
    --runner-log <b 段 runner 日志>
"""
import argparse
import json
import os
import re
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJ, "results")
DST = os.path.join(RESULTS, "pilot_timing.json")

# 阶段标签默认表（smoke 段历史标签；其余运行按 --stage 标注）
STAGE_DEFAULT = {"A_s0": "S3-pilot (D15 单格 smoke)"}


def jload(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def parse_runner_window(path):
    """从 runner 日志解析 start/done 时间戳（真实窗口；解析不到返回 None）。"""
    if not path or not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    m0 = re.search(r"runner start (\S+)", txt)
    m1 = re.search(r"runner done (\S+)", txt)
    return {"started_at": m0.group(1) if m0 else None,
            "done_at": m1.group(1) if m1 else None,
            "source": os.path.relpath(path, PROJ)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S3-pilot-continue-a")
    ap.add_argument("--block", default="continue_a")
    ap.add_argument("--runner-log",
                    default=os.path.join("experiments", "_continue_a_runner.log"))
    args = ap.parse_args()

    existing = jload(DST, {})
    runs = jload(os.path.join(RESULTS, "pilot_train_runs.json"), {}).get("runs", {})
    emerg = jload(os.path.join(RESULTS, "emergency_tier_probe.json"), {})
    emerg_ref = next((c for c in emerg.get("cases", [])
                      if c.get("tag") == "emerg_h256_b64"), {})

    # ---- 阶段标签：既有文件标签 > 默认表 > --stage ----
    prev_tags = {k: v.get("stage") for k, v in
                 existing.get("train_runs_all", {}).items()}
    train_runs = {}
    for k, v in runs.items():
        tag = prev_tags.get(k) or STAGE_DEFAULT.get(k) or args.stage
        rec = dict(v)
        rec["stage"] = tag
        train_runs[k] = rec

    # ---- 当前阶段块 ----
    cur = {k: v for k, v in train_runs.items() if v["stage"] == args.stage}
    win = parse_runner_window(os.path.join(PROJ, args.runner_log))
    agg = {}
    if cur:
        walls = [v["wall_sec"] for v in cur.values() if v.get("wall_sec")]
        eps = [e for v in cur.values() for e in (v.get("epoch_sec") or [])]
        bmeds = [v["batch_sec_median"] for v in cur.values()
                 if v.get("batch_sec_median")]
        agg = {
            "n_runs": len(cur),
            "run_keys": sorted(cur),
            "statuses": {k: v.get("status") for k, v in cur.items()},
            "all_completed": all(v.get("status") == "completed"
                                 for v in cur.values()),
            "total_wall_sec": round(sum(walls), 1) if walls else None,
            "mean_epoch_sec": round(sum(eps) / len(eps), 2) if eps else None,
            "mean_batch_sec_median": round(sum(bmeds) / len(bmeds), 3),
            "max_peak_mem_mib_overall": max(
                (v.get("peak_mem_mib_overall") or 0.0) for v in cur.values()),
            "f3_any_triggered": any((v.get("f3") or {}).get("triggered")
                                    for v in cur.values()),
            "window": win,
        }
    out = dict(existing)
    stages = list(out.get("stages", []))
    if args.stage not in stages:
        stages.append(args.stage)
    out["schema"] = "pilot-timing/v2"
    out["stages"] = stages
    out["stage"] = " + ".join(stages)
    out["train_runs_all"] = train_runs
    out[args.block] = {"stage": args.stage, "runs": cur, "aggregate": agg}
    out.setdefault("reference_emergency_tier_probe_emerg_h256_b64", {
        "batch_sec_median": emerg_ref.get("batch_sec_median"),
        "epoch_estimate_sec_n2000":
            (emerg_ref.get("epoch_estimate_sec") or {}).get("n2000"),
        "pilot_train_estimate_sec_E6": emerg_ref.get("pilot_train_estimate_sec_E6"),
        "peak_mem_mib": emerg_ref.get("peak_mem_mib"),
    })
    out["measured_at"] = datetime.now().isoformat()

    os.makedirs(RESULTS, exist_ok=True)
    tmp = DST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, DST)
    print(json.dumps({"wrote": DST, "block": args.block,
                      "stage": args.stage,
                      "runs_in_block": sorted(cur),
                      "aggregate": agg,
                      "all_runs": {k: {"status": v["status"],
                                       "wall_sec": v.get("wall_sec"),
                                       "stage": v["stage"]}
                                   for k, v in train_runs.items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
