# 首里程碑验收：S3-pilot-continue-d（2026-10-31T01:1x，lab-executor 执行单元）

- 阶段：S3-pilot-continue-d（workflow.json，kind=experiment，est 35min / unit_cap 45min）
- 判据来源：admission.json `first_milestone.acceptance_criteria`（status=accepted 冻结）
  + design.md §五 S3 / §六门线 + workflow.json 本阶段 completion.criteria
- 数据真源（全部真实落盘，无合成）：
  results/pilot_progress.json（A/B 6 跑每 epoch）、results/pilot_tau_log.json（D7 τ 日志）、
  results/pilot_lnn_progress.json + results/pilot_lnn_sanity.json（单体 LNN）、
  results/pilot_train_runs.json / results/pilot_timing.json（计时）、
  experiments/_pilot_*.log（runner 起止）、experiments/ckpt/pilot_*_ep1-6.pt（原子 ckpt）。
- 口径（双标注，S6 终判必须保留）：**D14 应急档** hidden 256 / batch 64 / ode_unfolds 6 /
  子采样 2000 条（data_seed 20261029，sha1=a2ba8268d20d 与 smoke 一致）/ E=6 / Adam lr=1e-3；
  可训练参数 **28.5M（论文 hidden 1500）→1,004,958（A）/1,004,961（B）/333,449（单体 LNN）**；
  数据=**PhysioNet/CinC 2019 回退源**（主源 Moor 2023 认证墙实证见 results/data_check.json）。

## ① 固定 τ 基线（A 臂）损失下降且 AUROC 可算 —— PASS

A 臂三种子末 epoch（val 302 条全量，D8）：

| run | train_loss ep1→ep6 | val_loss | hourly AUROC | hourly AUPRC | adm AUROC | adm AUPRC |
|---|---|---|---|---|---|---|
| A_s0 | 0.228507→0.040567 | 0.040562 | 0.8031 | 0.019727 | 0.6136 | 0.081305 |
| A_s1 | 0.235309→0.039781 | 0.040626 | 0.8018 | 0.019584 | 0.5891 | 0.067592 |
| A_s2 | 0.236789→0.040293 | 0.040487 | 0.8198 | 0.021479 | 0.5889 | 0.067575 |

- 损失下降：train 0.2285-0.2368→0.0398-0.0406（降幅 ≈0.188-0.196，progress `loss_decrease_ep1_to_last` 逐跑落盘）；val loss 同步下降（smoke 期 ep1 低谷=warm-up 瞬态，results/diag_train_vs_val_A_s0.json 实证 train hourly 0.788 vs val 0.803，非过拟合/非结构错位）。
- AUROC 可算：hourly（n=21744）与 admission（n=302≈17 正例）两级每跑每 epoch 全部可算（无 NaN）。
- 轨迹：results/pilot_val_curves.csv/md（42 行=7 跑×6 epoch，含 LNN）。

## ② τ 换保序可学习参数后：梯度到达 τ / 训练不发散 / τ 顺序保持 —— PASS

- 梯度到达 τ：B 臂 `c_grad_norm_mean` 六 epoch 全程非空——B_s0 5.19e-05→6.83e-06、B_s1 4.28e-05→5.81e-06、B_s2 5.44e-05→1.45e-05（pilot_tau_log.json；与 smoke 证据边界②「init 附近损失面平、τ 梯度通路活着但量级小」一致：‖∂L/∂c‖ 探针 5.6e-07→实测 1e-05 量级）。
- 训练不发散：全部 7 跑 loss 全有限、全局梯度范数最大 1.622133<1e3（F3 零触发，pilot_train_runs.json `f3.triggered=false`）。
- τ 顺序保持：每 epoch 每种子 `tau_order_strict_increasing=True`（D2 累积正增量构造性保证 + 实测）；训练后 τ（vs init (1,10,100)，compose 锚点口径）：
  - B_s0 → (0.958478, 9.891654, 99.828354)，总相对移动 4.15%/1.08%/0.17%
  - B_s1 → (0.956341, 9.885924, 99.825455)，4.37%/1.14%/0.17%
  - B_s2 → (0.937164, 9.834926, 99.731377)，6.28%/1.65%/0.27%
  - 三档序保持 0.937<9.83<99.73 型（快≪中≪慢）；**有移动、无塌缩**（D7 判据全过；塌缩=贴 0 或贴边界均未出现）。
- B(0)≡A(0)：**引用 smoke 双证不重跑**（ltc_tau_probe_h256.json A2 随机输入 max_abs_diff=0.0 + pilot_train.py 每跑真实数据 init_check diff=0.0，progress 6/6 落盘；参数审计恰 3 标量 tau_sched.c）。

## ③ 含数字 pilot 日志 / 两臂验证曲线 / 训练后 τ 数值与排序 / 差异清单 —— PASS

| 要求 | 产物 |
|---|---|
| 含数字 pilot 日志 | experiments/_pilot_{A,B}_s*.log + _pilot_lnn_s0.log（runner 起止两行原子记录）；results/pilot_progress.json（每 epoch 数字）；results/pilot_tau_log.json（D7） |
| 两臂验证曲线 | **results/pilot_val_curves.csv / .md**（本阶段新立 experiments/compose_pilot_val_curves.py 生成；7 跑×6 epoch 长表） |
| 训练后 τ 数值与排序 | 结论表 csv/md 的 τ_final/τ 相对移动/序保持列 + pilot_tau_log.json 逐 epoch 轨迹（数值见②） |
| 基线配置差异清单 | results/baseline_diff.md（S2-gate-checks 落盘：D2 vs admission 字面差异/论文未给项冻结值/CfC-vs-LTC 理由/算子层差异 R1-R5） |

## 附加判据：A 臂优于单体 LNN（D5(i) 方向性）—— PASS（带噪声边界注记）

单体 LNN seed0（D5(i)，同应急档 h256/B64/2000 条/E6，同子采样 sha1=a2ba8268d20d，同 seed0/lr）：
train 0.226328→0.041196、val_loss 0.057636→0.041741、hourly AUROC 0.7402→0.7798（AUPRC 0.018358）、
adm AUROC 0.5476→0.5917（AUPRC 0.070446）；wall 178.6s、batch 中位 0.885s、peak 370.1MiB、F3 零触发。

A 臂 mean(n=3) − LNN：adm AUROC **+0.005435**、adm AUPRC +0.001711、hourly AUROC **+0.028468**、
hourly AUPRC +0.001906；同种子 A_s0−LNN_s0：adm AUROC +0.021878、hourly AUROC +0.023278。
方向性判据（两项 AUROC 均 A>LNN）= **pass**，不触发失败条件「A 臂不胜单体 LNN → 查实现 bug」。
**噪声边界（诚实注记，交 S4）**：hourly 级优势 +0.0285 超出 A 臂种子 std（±0.0101）且三种子一致；
admission 级 +0.0054 在 A 臂种子 std（±0.0142）之内，且 A_s1/A_s2 单种子 adm AUROC（0.5891/0.5889）
略低于 LNN（0.5917）——「A≥LNN」在臂级均值成立、admission 单种子级不稳健（val n=302≈17 正例的固有噪声）。

## pilot 结论表（D6/D8；不终判 H1a/H1b，交 S4-pilot-verdict）

- 逐种子配对 B−A（n=3，末 epoch）：Δadm AUROC **0.0±0.0**；Δadm AUPRC **0.0±0.0**；
  Δhourly AUROC **−0.0007±0.0011**；Δhourly AUPRC **−4.3e-05±7.7e-05**（per-seed 明细见 csv/md）。
- 臂级 mean±std：A adm 0.5972±0.0142 / hourly 0.8082±0.0100；B adm 0.5972±0.0142 / hourly 0.8075±0.0089。
- H1a 临时读数（3/3 种子，不终判）：AUROC 项 0.0 < +0.01 未达；AUPRC 项 0.0 ≥ −0.01 达。
- H1b(iii)「≥2/3 种子至少一专家移动>5%」pilot 预览=1/3（仅 B_s2 τ1 6.28%；B_s0 4.15%/B_s1 4.37%）
  ——pilot 尺度未达，full 尺度 E=20 复测交 S6。
- 产物：results/pilot_fixed_vs_learnable.csv / .md（无 pending 行；b/a seeds_missing=[]；lnn_sanity=present）。

## 计时（compose_pilot_timing_v2.py --stage S3-pilot-continue-d --block continue_d）

LNN_s0：window 2026-10-31T01:10:07→01:13:08（+08:00）、wall 178.6s、epoch 均值 29.43s、
batch 中位 0.885s、peak 370.1MiB、status=completed、F3 零触发；continue_a/b/c 既有块保留、
train_runs_all 七跑阶段标签无错归（results/pilot_timing.json）。

## 结论

首里程碑①②③ **全部验收通过**（判据逐项对应上方证据）；附加方向性判据 A>单体 LNN **通过**
（hourly 级稳健、admission 级在噪声内，已注记）。H1a/H1b 不在此终判，交 S4-pilot-verdict；
全部结论限应急档规模 + PhysioNet 2019 回退源，S6 终判须带双标注。
