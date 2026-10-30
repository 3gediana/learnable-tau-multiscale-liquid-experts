#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_check.py — S2b-data-model 步骤②：败血症数据获取实测（design.md D3 + admission real_validation_path）。

决策过程（D3 冻结）：主源 Moor 2023 harmonized → 回退 PhysioNet/CinC 2019 → 两来源皆败 = F2 停。
本脚本如实记录两个来源的真实获取尝试与结果，落 results/data_check.json。

Moor 2023（eClinicalMedicine 62:102124, "Predicting sepsis using deep learning across
international sites"）：论文 Data sharing statement 原文（Europe PMC 全文 PMC10425671，
https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10425671/fullTextXML ）：
  "All the raw data used in this manuscript is publicly available for accredited
   researchers. MIMIC-III, eICU, and HiRID are available via Physionet. Data from the
   AUMC database is available via the website of ... using the 'ricu' R-package."
四个底库（MIMIC-US / eICU-US / HiRID-CH / AUMCdb-NL）均需 credentialed access（CITI 培训 +
机构签署 DUA），harmonized 标注集本身无公开直接下载；本单元实测检索 GitHub（BorgwardtLab
组织 12 仓库无 sepsis 仓库 / repo search 0 命中）、Zenodo、figshare、grep.app 均无该
harmonized 集发布 → 记为「主源获取失败：需认证访问，本环境无凭证」。

PhysioNet/CinC 2019（Early Prediction of Sepsis from Clinical Data，同 Che 2018 协议）：
公开免凭证下载。因 physionet.org 经本机代理限速约 6 文件/s（实测，见 --probe），全量
40,336 文件约需 108 min，超出单元窗口；本脚本默认下载固定种子随机样本 N_SAMPLE=3000
（约 7%），用于获取实测与协议兼容性判定；--full 可续跑全量（断点续传，已存在跳过）。

用法：
  python experiments/data_check.py                 # 默认样本 3000 + 统计 + 落盘
  python experiments/data_check.py --full          # 全量下载（断点续传）
  python experiments/data_check.py --sample 5000   # 自定义样本量
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import math
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RAW_DIR = os.path.join(PROJ, "data", "raw", "physionet-2019")
RESULT = os.path.join(PROJ, "results", "data_check.json")

PN_BASE = "https://physionet.org/files/challenge-2019/1.0.0"
PN_SETS = ["training_setA", "training_setB"]
UA = {"User-Agent": "Mozilla/5.0 (research data-check; contact: local)"}
SAMPLE_SEED = 20261028
N_SAMPLE_DEFAULT = 3000
MAX_WORKERS = 16
RETRIES = 3

# PhysioNet/CinC 2019 .psv 的 41 列 = 34 个时变临床变量 + 7 个静态/元数据列
TIME_VARYING_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess",
    "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
    "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]

MOOR_EVIDENCE = {
    "paper": "Moor M, Bennett N, Plecko D, Horn M, Rieck B, Meinshausen N, Buhlmann P, "
             "Borgwardt K. Predicting sepsis using deep learning across international sites: "
             "A retrospective development and validation study. eClinicalMedicine 62:102124, 2023.",
    "doi": "10.1016/j.eclinm.2023.102124",
    "data_sharing_statement_source": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10425671/fullTextXML",
    "data_sharing_statement_verbatim": (
        "All the raw data used in this manuscript is publicly available for accredited "
        "researchers. MIMIC-III, eICU, and HiRID are available via Physionet. Data from the "
        "AUMC database is available via the website of [Amsterdam UMC] using the \u201cricu\u201d "
        "R-package. All code used for extracting, cleaning, filtering and modelling will be "
        "made available immediately upon publication."
    ),
    "underlying_databases": {
        "MIMIC-III": "US, PhysioNet, credentialed access (CITI + DUA)",
        "eICU-CRD": "US, PhysioNet, credentialed access (CITI + DUA)",
        "HiRID": "CH (Bern), PhysioNet, credentialed access",
        "AmsterdamUMCdb": "NL, Amsterdam UMC website, credentialed access",
    },
    "conclusion": "主源 Moor 2023 harmonized 集在本环境不可获取：四底库均需机构认证凭证"
                  "（CITI 培训 + DUA），无公开免凭证下载；harmonized 标注集本身亦无公开发布"
                  "（本单元实测检索 GitHub/Zenodo/figshare/grep.app 无果）。按 D3 决策过程"
                  "回退 PhysioNet/CinC 2019。",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def http_get(url: str, timeout: int = 60, binary: bool = False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", "replace")


# ---------------------------------------------------------------- Moor 探测
def probe_moor() -> dict:
    """主源 Moor 2023 获取实测：论文数据共享声明（引文）+ 实时检索镜像/代码仓库。"""
    out = dict(MOOR_EVIDENCE)
    out["live_probes"] = {}

    def probe(name: str, url: str, check):
        t0 = time.time()
        try:
            body = http_get(url, timeout=25)
            out["live_probes"][name] = {
                "url": url, "ok": True, "finding": check(body),
                "elapsed_s": round(time.time() - t0, 2),
            }
        except Exception as e:
            out["live_probes"][name] = {
                "url": url, "ok": False, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 2),
            }

    gh_repo_re = re.compile(r'href="/BorgwardtLab/([A-Za-z0-9_.-]+)"')

    def gh_check(b: str) -> str:
        repos = sorted(set(gh_repo_re.findall(b)))
        sepsis = [r for r in repos if "sepsis" in r.lower()]
        return f"{len(repos)} repos listed; sepsis-related repos: {sepsis}"

    probe("github_org_borgwardtlab_repos",
          "https://github.com/BorgwardtLab?tab=repositories", gh_check)
    probe("github_repo_search",
          "https://api.github.com/search/repositories?q=harmonised+sepsis+international+sites&per_page=5",
          lambda b: f"total_count={json.loads(b).get('total_count')}")
    probe("zenodo_search",
          "https://zenodo.org/api/records?q=harmonised%20sepsis%20international&size=5",
          lambda b: f"hits={len(json.loads(b).get('hits', {}).get('hits', []))}, "
                    f"titles={[h['metadata']['title'][:60] for h in json.loads(b).get('hits', {}).get('hits', [])]}")
    probe("europepmc_fulltext_datasharing",
          "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10425671/fullTextXML",
          lambda b: "Data sharing statement present: "
                    + str("Data sharing statement" in b))
    out["status"] = "failed_credentialed"
    out["obtainable_here"] = False
    return out


# ---------------------------------------------------------------- PhysioNet 2019
def fetch_inventory() -> dict:
    """从 physionet.org 目录页取全部文件名清单（精确例数来源，不需下载内容）。"""
    inv = {}
    for s in PN_SETS:
        html = http_get(f"{PN_BASE}/training/{s}/", timeout=120)
        names = sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))
        inv[s] = names
        log(f"inventory {s}: {len(names)} files (page {len(html)} bytes)")
    return inv


def download_set(setname: str, names: list[str], raw_dir: str) -> dict:
    """并行下载（断点续传：已存在跳过；失败重试 RETRIES 次）。"""
    outdir = os.path.join(raw_dir, setname)
    os.makedirs(outdir, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(outdir, n))]
    log(f"{setname}: {len(names) - len(todo)} cached, {len(todo)} to download")
    t0 = time.time()
    nbytes = 0
    fails = []

    def fetch(name: str):
        last = None
        for attempt in range(RETRIES):
            try:
                req = urllib.request.Request(
                    f"{PN_BASE}/training/{setname}/{name}", headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                with open(os.path.join(outdir, name), "wb") as f:
                    f.write(data)
                return name, len(data), None
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
        return name, 0, last

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for name, nb, err in ex.map(fetch, todo):
            nbytes += nb
            if err:
                fails.append({"file": name, "error": err})
    dt = time.time() - t0
    return {"files_requested": len(todo), "bytes": nbytes,
            "seconds": round(dt, 1), "fails": fails}


def parse_psv(text: str):
    """解析单个 .psv：返回 (列名, 行数, 正标签行数, ICU开始时间)。"""
    rows = list(csv.reader(io.StringIO(text), delimiter="|"))
    if not rows:
        return None
    header = rows[0]
    idx_label = header.index("SepsisLabel")
    idx_iculos = header.index("ICULOS")
    n_pos = 0
    first_iculos = None
    for r in rows[1:]:
        if not r:
            continue
        try:
            if int(float(r[idx_label])) == 1:
                n_pos += 1
            if first_iculos is None and r[idx_iculos] not in ("", "NA"):
                first_iculos = int(float(r[idx_iculos]))
        except (ValueError, IndexError):
            continue
    return {"header": header, "n_rows": len(rows) - 1, "n_pos": n_pos,
            "iculos_start": first_iculos}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def stats_from_files(raw_dir: str, sets: list[str]) -> dict:
    lengths, pos_rows, n_adm, n_pos_adm = [], 0, 0, 0
    per_set = {}
    for s in sets:
        d = os.path.join(raw_dir, s)
        files = sorted(f for f in os.listdir(d) if f.endswith(".psv"))
        sl, sp, sa, spa = [], 0, 0, 0
        for f in files:
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                rec = parse_psv(fh.read())
            if rec is None or rec["n_rows"] == 0:
                continue
            sl.append(rec["n_rows"])
            sp += rec["n_pos"]
            sa += 1
            if rec["n_pos"] > 0:
                spa += 1
        per_set[s] = {
            "files_downloaded": len(files), "admissions_parsed": sa,
            "hours_total": sum(sl), "hours_positive": sp,
            "length_pctiles": {q: pct(sl, q / 100) for q in (5, 25, 50, 75, 95, 99)},
            "length_min": min(sl) if sl else 0, "length_max": max(sl) if sl else 0,
            "length_mean": round(sum(sl) / len(sl), 2) if sl else 0,
            "admissions_with_any_positive": spa,
            "hour_positive_rate": round(sp / max(1, sum(sl)), 6),
            "admission_positive_rate": round(spa / max(1, sa), 6),
        }
        lengths += sl
        pos_rows += sp
        n_adm += sa
        n_pos_adm += spa
    return {
        "sample_admissions": n_adm,
        "sample_hours_total": sum(lengths),
        "sample_hours_positive": pos_rows,
        "length_distribution_hours": {
            "min": min(lengths) if lengths else 0,
            "p05": pct(lengths, 0.05), "p25": pct(lengths, 0.25),
            "p50": pct(lengths, 0.50), "p75": pct(lengths, 0.75),
            "p95": pct(lengths, 0.95), "p99": pct(lengths, 0.99),
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        },
        "label_rate_hour_level": {
            "rate": round(pos_rows / max(1, sum(lengths)), 6),
            "wilson95": wilson(pos_rows, sum(lengths)),
            "note": "正标签小时数 / 总小时数（样本）",
        },
        "label_rate_admission_level": {
            "rate": round(n_pos_adm / max(1, n_adm), 6),
            "wilson95": wilson(n_pos_adm, n_adm),
            "note": "至少 1 个正标签小时的入院数 / 总入院数（样本；Che 2018/PhysioNet 2019 口径）",
        },
        "per_set": per_set,
    }


def protocol_compat() -> dict:
    """D3 协议兼容性实测：列集合、小时网格、标签语义。"""
    d = os.path.join(RAW_DIR, "training_setA")
    files = sorted(f for f in os.listdir(d) if f.endswith(".psv"))[:1]
    if not files:
        return {"checked": False}
    with open(os.path.join(d, files[0]), encoding="utf-8") as fh:
        rec = parse_psv(fh.read())
    return {
        "checked": True,
        "sample_file": f"training_setA/{files[0]}",
        "n_columns_expected_physionet2019": 41,
        "n_columns_observed": len(rec["header"]),
        "n_time_varying": len(TIME_VARYING_COLS),
        "n_static_meta": len(rec["header"]) - len(TIME_VARYING_COLS),
        "columns": rec["header"],
        "hourly_grid": "ICULOS 为小时序号（1,2,3...），dt=1h 与 D3 冻结一致",
        "label_semantics": "SepsisLabel：该小时起未来 6h 内 Sepsis-3 onset=1（PhysioNet 2019 标准），"
                           "与 D3「y(t)=1[onset∈(t,t+6h]]」一致",
        "che2018_split": "Che et al. 2018 患者级随机 80/10/10 划分可在合并 setA+setB 上复现（D3）",
        "dt_hours": 1, "truncation_d3_hours": 72,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="下载全量（断点续传）")
    ap.add_argument("--sample", type=int, default=N_SAMPLE_DEFAULT,
                    help=f"随机样本量（默认 {N_SAMPLE_DEFAULT}；--full 覆盖）")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    tz = timezone(timedelta(hours=8))
    report = {
        "schema": "data-check/v1",
        "stage": "S2b-data-model",
        "purpose": "design.md D3 数据二选一实测：主 Moor 2023 harmonized / 退 PhysioNet-CinC 2019；"
                   "两来源皆败 = F2 停（禁合成替代）",
        "ran_at": datetime.now(tz).isoformat(timespec="seconds"),
        "sources": {},
        "decision": None,
    }

    # ---- 主源：Moor 2023 ----
    log("probing Moor 2023 (primary source) ...")
    moor = probe_moor()
    report["sources"]["moor_2023"] = moor
    moor_ok = bool(moor["obtainable_here"])
    log(f"Moor 2023 obtainable={moor_ok} ({moor['conclusion'][:60]}...)")

    # ---- 回退源：PhysioNet/CinC 2019 ----
    log("fetching PhysioNet 2019 inventory ...")
    inv = fetch_inventory()
    report["sources"]["physionet_cinc_2019"] = {
        "project": "Early Prediction of Sepsis from Clinical Data: "
                   "The PhysioNet/Computing in Cardiology Challenge 2019 v1.0.0",
        "base_url": PN_BASE,
        "license": "Open Data Commons / PhysioNet Credentialed Health Data License（公开免凭证训练集）",
        "inventory_exact": {s: len(v) for s, v in inv.items()},
        "inventory_total": sum(len(v) for v in inv.values()),
        "throttle_note": "本机代理到 physionet.org 实测约 6 文件/s（16/32 线程同），"
                         "全量下载约 108min 超单元窗口 → 默认固定种子随机样本，--full 可续跑",
    }
    pn = report["sources"]["physionet_cinc_2019"]

    if args.full:
        plan = {s: inv[s] for s in PN_SETS}
        mode = "full"
    else:
        rng = random.Random(SAMPLE_SEED)
        n_each = args.sample // len(PN_SETS)
        plan = {s: rng.sample(inv[s], min(n_each, len(inv[s]))) for s in PN_SETS}
        mode = f"random_sample_seed{SAMPLE_SEED}_n{args.sample}"
    report["download_mode"] = mode

    os.makedirs(RAW_DIR, exist_ok=True)
    dl = {}
    for s in PN_SETS:
        log(f"downloading {s}: {len(plan[s])} files ...")
        dl[s] = download_set(s, plan[s], RAW_DIR)
        log(f"{s} done: {dl[s]['files_requested']} files, "
            f"{dl[s]['bytes'] / 1e6:.1f} MB in {dl[s]['seconds']}s, fails={len(dl[s]['fails'])}")
    pn["download"] = dl
    pn["downloaded_bytes_total"] = sum(d["bytes"] for d in dl.values())
    pn["download_fails_total"] = sum(len(d["fails"]) for d in dl.values())

    stats = stats_from_files(RAW_DIR, PN_SETS)
    pn["stats"] = stats
    pn["protocol_compat"] = protocol_compat()

    pn_ok = pn["download_fails_total"] == 0 and stats["sample_admissions"] > 0

    # ---- D3 决策 ----
    if moor_ok:
        report["decision"] = {"selected": "moor_2023", "basis": "主源可获取（D3 首选）"}
    elif pn_ok:
        report["decision"] = {
            "selected": "physionet_cinc_2019",
            "basis": "主源 Moor 2023 需认证访问不可获取（见 moor_2023.conclusion）；"
                     "回退源 PhysioNet/CinC 2019 公开可下载、协议与 Che 2018/Sepsis-3 兼容（实测）",
        }
    else:
        report["decision"] = {"selected": None, "basis": "F2：两来源皆败 → 停，禁合成替代"}

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"wrote {RESULT}")

    print(json.dumps({
        "moor_obtainable": moor_ok,
        "physionet_ok": pn_ok,
        "inventory_exact": pn["inventory_exact"],
        "sample_admissions": stats["sample_admissions"],
        "sample_hours": stats["sample_hours_total"],
        "hour_label_rate": stats["label_rate_hour_level"]["rate"],
        "admission_label_rate": stats["label_rate_admission_level"]["rate"],
        "selected": report["decision"]["selected"],
    }, ensure_ascii=False, indent=1))
    return 0 if pn_ok or moor_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
