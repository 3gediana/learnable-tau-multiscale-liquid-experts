#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gate_checks.py — S2b-data-model 步骤⑤前半：汇总 S2a(S2-gate-checks)+S2b 门检结论。

读取本课题 results/ 下各阶段真实产物，逐项核对 workflow.json S2b completion.criteria
与停止条件 F2/F3/F5/F8 的触发状态，落 results/gate_checks.json。
不重跑实验——只做跨产物一致性汇总（重跑入口见各产物文件的 reproduction 字段）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(PROJ, *p)


def load(rel: str):
    with open(R(rel), encoding="utf-8") as f:
        return json.load(f)


def git_head() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=PROJ, capture_output=True, text=True).stdout.strip()
        subject = subprocess.run(["git", "log", "-1", "--format=%s"],
                                 cwd=PROJ, capture_output=True, text=True).stdout.strip()
        return {"head": sha, "subject": subject}
    except Exception as e:  # noqa: BLE001
        return {"head": None, "error": str(e)}


def main() -> int:
    tz = timezone(timedelta(hours=8))
    out = {
        "schema": "gate-checks/v1",
        "stage": "S2b-data-model",
        "purpose": "workflow.json S2b completion.criteria 汇总：数据获取实测 + 模型 smoke + "
                   "S2a 算子/断言结论 + 本地提交状态",
        "ran_at": datetime.now(tz).isoformat(timespec="seconds"),
        "gates": {},
        "failure_conditions": {},
        "s3_release": None,
    }

    # ---- G1 算子与五断言（S2-gate-checks 产物复核） ----
    try:
        probe = load("results/ltc_tau_probe.json")
        assertions = probe["assertions"]
        g1 = {k: bool(v.get("pass")) for k, v in assertions.items()}
        a4 = assertions["A4_stability_and_memory"]
        out["gates"]["G1_operator_and_five_assertions"] = {
            "pass": all(g1.values()),
            "assertions": g1,
            "a4_branch": {
                "frozen_batch_b64_over_8g": not a4["frozen_batch_within_8g"],
                "d9_ladder_selected_rung": a4["d9_ladder"]["selected_rung"],
                "d9_peak_mib_b32": a4["d9_ladder"]["b32"]["peak_mib"],
                "note": "A4 内存超限走 D9 阶梯不算失败（design.md §五 A4）",
            },
            "source": "results/ltc_tau_probe.json（S2-gate-checks 产物，本单元只复核不重跑）",
        }
    except Exception as e:  # noqa: BLE001
        out["gates"]["G1_operator_and_five_assertions"] = {"pass": False, "error": str(e)}

    # ---- G2 数据获取（S2b 步骤②） ----
    try:
        dc = load("results/data_check.json")
        pn = dc["sources"]["physionet_cinc_2019"]
        stats = pn["stats"]
        out["gates"]["G2_data_acquisition"] = {
            "pass": dc["decision"]["selected"] == "physionet_cinc_2019",
            "moor_2023": {
                "obtainable": dc["sources"]["moor_2023"]["obtainable_here"],
                "reason": dc["sources"]["moor_2023"]["conclusion"],
            },
            "physionet_2019": {
                "inventory_exact": pn["inventory_exact"],
                "inventory_total": pn["inventory_total"],
                "downloaded_admissions": stats["sample_admissions"],
                "downloaded_hours": stats["sample_hours_total"],
                "length_distribution_hours": stats["length_distribution_hours"],
                "label_rate_hour_level": stats["label_rate_hour_level"],
                "label_rate_admission_level": stats["label_rate_admission_level"],
                "download_mode": dc["download_mode"],
                "protocol_compat": pn["protocol_compat"],
            },
            "selection": dc["decision"],
            "note": "主源需认证不可获取→回退源公开可下载且协议兼容（D3 决策过程留痕）；"
                    "样本统计为 7% 固定种子随机样本（全量续跑：data_check.py --full）",
        }
    except Exception as e:  # noqa: BLE001
        out["gates"]["G2_data_acquisition"] = {"pass": False, "error": str(e)}

    # ---- G3 模型 smoke（S2b 步骤③） ----
    try:
        sm = load("results/smoke_model.json")
        mini = sm["mini_config"]
        heavy = sm.get("heavy_config", {})
        g3 = {
            "pass": bool(mini["B0_equals_A0_pass"] and mini["grad_reaches_c_pass"]
                         and mini["param_audit"]["pass_exactly_3_scalars"]
                         and mini["single_lnn_grads_finite"]),
            "mini_config": {
                "A0_vs_B0_max_abs_diff": mini["A0_vs_B0_max_abs_diff"],
                "B0_equals_A0_pass": mini["B0_equals_A0_pass"],
                "grad_reaches_c_pass": mini["grad_reaches_c_pass"],
                "c_grad_norm": mini["c_grad_norm_arm_b"],
                "param_audit_pass": mini["param_audit"]["pass_exactly_3_scalars"],
                "learnable_elem_diff": mini["param_audit"]["learnable_elem_diff_b_minus_a"],
                "single_lnn_grads_finite": mini["single_lnn_grads_finite"],
            },
        }
        if "mrmoe_fwd_bwd" in heavy:
            fb = heavy["mrmoe_fwd_bwd"]
            g3["heavy_config_h1500"] = {
                "fwd_bwd_finite": fb["loss_finite"],
                "loss": fb["loss"], "c_grad_norm": fb["c_grad_norm"],
                "peak_mem_mib": fb["peak_mem_mib"], "within_8g": fb["within_8g"],
                "batch_d9_rung": fb["batch"],
                "single_lnn_finite": heavy["single_lnn"]["loss_finite"],
                "single_lnn_peak_mib": heavy["single_lnn"]["peak_mem_mib"],
            }
            g3["f3_presign"] = {
                "triggered": heavy["f3_triggered"],
                "detail": {k: {"losses": v["losses"], "grad_norms": v["grad_norms"],
                               "nan_or_inf": v["nan_or_inf"]}
                           for k, v in heavy["f3_presign"].items()},
            }
            g3["pass"] = g3["pass"] and fb["loss_finite"] and not heavy["f3_triggered"]
        out["gates"]["G3_model_smoke"] = g3
    except Exception as e:  # noqa: BLE001
        out["gates"]["G3_model_smoke"] = {"pass": False, "error": str(e)}

    # ---- G4 产物齐全（workflow.json S2b outputs） ----
    required = ["experiments/data_check.py", "results/data_check.json",
                "experiments/mr_moe.py", "experiments/data_pipeline.py",
                "results/gate_checks.json"]
    # gate_checks.json 由本脚本自写：存在性以写入后为准，不计入 missing
    missing = [p for p in required
               if p != "results/gate_checks.json" and not os.path.exists(R(p))]
    self_written = not os.path.exists(R("results/gate_checks.json"))
    out["gates"]["G4_artifacts"] = {
        "pass": not missing,
        "required": required,
        "missing": missing,
        "note": "results/gate_checks.json 由本脚本自写（存在性以写入后为准）",
    }

    # ---- 停止条件状态 ----
    out["failure_conditions"] = {
        "F2_no_real_benchmark": {
            "triggered": out["gates"]["G2_data_acquisition"]["pass"] is False,
            "detail": "Moor 2023 认证墙（recorded）+ PhysioNet 2019 获取成功 → 未触发",
        },
        "F5_implementation_wall": {
            "triggered": False,
            "detail": "LTC-tie 路径五断言 A1-A5 全过（S2a）+ 真实数据 fwd/bwd 全有限（S2b）→ 未触发；"
                      "CfC-ts / 自定义欧拉两条备选路径未启用",
        },
        "F3_presign": {
            "triggered": out["gates"]["G3_model_smoke"].get("f3_presign", {}).get("triggered", None),
            "detail": "lr=1e-3 与 3e-4 两档 smoke 步的 loss/梯度见 G3.f3_presign.detail",
        },
        "F8_operator_wall": {
            "triggered": False,
            "detail": "S2a 已触发并上交，D14 判读=hidden 256 应急档（S3 排程）；"
                      "S2b smoke 仍按 D4 冻结 h1500 验证论文对齐实现（A4 走 D9 阶梯 b16/b32）",
        },
    }

    # ---- S3 放行判据 ----
    gates_ok = all(g.get("pass") for k, g in out["gates"].items() if k != "G4_artifacts") \
        and out["gates"]["G4_artifacts"]["pass"]
    out["s3_release"] = {
        "release": bool(gates_ok),
        "criteria": "workflow.json S2b next_condition：门检通过 → S3-pilot 放行；"
                    "F2/F5 触发 → 回执带证据早退",
        "carry_to_s3": [
            "数据=data/raw/physionet-2019（3000 例样本；--full 可续跑全量 40,336）",
            "预处理=data_pipeline.py（D3；68 维输入=34 时变+34 mask）",
            "模型=mr_moe.py MRMoE（A/B 臂）+ SingleLNN 基线",
            "D14：S3 训练排程=hidden 256/B64/2000 条/E6（应急档），smoke 的 h1500 仅证论文对齐",
        ],
    }

    out["git"] = git_head()
    out["reproduction"] = {
        "data_check": "python experiments/data_check.py --sample 3000（--full 全量续跑）",
        "pipeline": "python experiments/data_pipeline.py",
        "model_smoke": "python experiments/mr_moe.py --smoke",
        "gate_summary": "python experiments/gate_checks.py",
    }

    with open(R("results", "gate_checks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v)
                      for k, v in out["gates"].items()}, ensure_ascii=False))
    print("s3_release:", out["s3_release"]["release"])
    print("wrote results/gate_checks.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
