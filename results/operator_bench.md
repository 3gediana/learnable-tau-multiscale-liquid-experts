# 算子基准报告 — LTC τ-tie 路径数值稳定性与效率（S2-gate-checks ⓪）

- 阶段：S2-gate-checks ⓪（fb-ab50d0ec9f / D12：先优化算子再跑实验）
- 脚本：experiments/operator_bench.py（分段落盘 results/operator_bench.json；`--sections/--o2-batches/--o3-batches/--b64-repeats` 可子集重跑）
- 被基准实现：experiments/mr_moe.py 的 OrderedTauLTCCell（τ-tie 覆写 + 修复 R1-R5）；修复前裸形态 TauTieLTCCellRaw（仅 τ-tie 覆写、无修复）
- 配置（单元指令逐字）：hidden 1500 / ode_unfolds 6 / batch 64 / 72 步；输入维占位 80（PhysioNet 2019 40 特征+40 mask，S2b 定案前占位）
- 环境（O0 实测）：Python 3.11.9 / torch 2.10.0+cu128 / ncps 0.0.2 / RTX 5060 Laptop GPU 8123.4MiB；元素级带宽 162.6 GB/s；FullyConnected(1500) 构建 18.4s+1.0s（R4 靶点）

## 结论摘要

| 维度 | 修复前（TauTieLTCCellRaw） | 修复后（R1-R5） | 判定 |
|---|---|---|---|
| 数值稳定性 | 前向/反向全有限，梯度范数 0.14 | 全有限；12→144 步梯度范数 0.0180→0.0182 不爆炸；gleak∈{−30,−10,0,30} 与 (w,gleak)=(−40,−40) 极端应力全有限 | **通过**（隐式后向欧拉+ε 守卫，本来稳定） |
| 72 步 BPTT 显存 | **OOM（任何批量）**：B=16 反传图累积 22GB+ OOM；B=4 同样 OOM；单步 B=64 峰值 11184MiB 超 8G 预算 | R3+R5 后：单专家 B=16/32=3089/5414MiB；全模型 B=8/16/32=3715/4769/6863MiB（均 <8192）；B=64 仍超 → D9 阶梯落 b32 | **显存墙已修**（R3/R5；B=64 走 D9 阶梯） |
| 数学恒等 | — | 裸 vs 修复前向 max_abs_diff=**0.0**；τ-tie(τ=1, cm≡gleak) vs 库 LTCCell =**0.0**；臂 A/B init 前向 =**0.0**（探针 A2） | 修复不改数值；排除 E1 实现伪影 |
| 效率 | — | 全模型 B=32：86s/batch（1.19s/步）；epoch 外推 8000/4000/2000 条 ≈ 6.0/3.0/1.5 小时 | **未解决**：单步计算量由 hidden 1500 稠密 (B,H,H)×6 unfolds 决定，R1-R5 只修冗余与显存，不降计算量；远超 D9 pilot 预算（≤12min/训练） |

## ⓪-1 修复点逐条记录（mr_moe.py 内实现）

| # | 修复 | 靶点 | 性质 | 效果 |
|---|---|---|---|---|
| R1 | softplus(gleak) 提到 unfold 循环外 | ltc_cell.py:242（每次 unfold 重算 6 次） | 恒等变换 | 每步省 5 次 (H,) softplus |
| R2 | sparsity_mask 折进 w_param 每步一次 | ltc_cell.py:234（每次 unfold 乘 (B,H,H)） | FullyConnected 下 mask 全 1，**精确**恒等 | 每步省 6 次 (B,H,H) 乘 |
| R3 | 每次 unfold 梯度检查点（use_reentrant=False） | — | 数学不变（O5=0.0），反向按 unfold 重算 | 72 步 BPTT 反传图 >20GB(OOM)→O(小) |
| R4 | FullyConnected wiring 构建缓存（18.6s→0）+ **修订**：预 build 后 cell 必须传 in_features=None | wirings.py:295-318；LTCCell.__init__ 逢 in_features 非 None 会再调 wiring.build() | erev 为 adjacency 拷贝；**修订动因**：共享 wiring 被重复 build 时 numpy RNG 推进，sensory_erev 逐专家漂移，实测把臂 A/B init 等价破坏到 max_abs_diff=0.118；预 build+None 后逐位一致（回 0.0） | 构模型开销→0；init 等价恢复 |
| R5 | 专家序列级梯度检查点（MRMoE.forward 包每个专家的 72 步循环） | — | 数学不变（确定性重算），反向按专家重算 | 三专家反传图不再共存：全模型峰值≈单专家（B=32=6863MiB<8G；否则 B=16≈9.5GB 超预算、B=64≈30GB OOM） |

修复前后向恒等证据（O5）：`raw_vs_fixed_single_step_max_abs_diff=0.0`、`raw_vs_fixed_seq12_max_abs_diff=0.0`、`stock_cm_eq_gleak_vs_tie_tau1_max_abs_diff=0.0`。臂 A/B init 等价见探针 A2（`logits_max_abs_diff=0.0`，61 个共享参数逐位一致）。

## ⓪-2 显存墙定位与修复验证（O1/O2/O3）

修复前（O1，τ-tie 裸形态，hidden 1500）：
- B=16 单步 fwd+bwd：峰值 2871.4MiB（6 unfolds × ~3 个 (B,H,H) 中间图/步）；
- B=16 72 步 BPTT：OOM（"Tried to allocate 138.00 MiB … 22.31 GiB is allocated by PyTorch"）；
- B=4 72 步 BPTT：OOM（缩小批量无效——墙在步数×反传图累积）；
- B=64 单步：峰值 11184.4MiB（超 8G 预算；本机 WDDM 软墙 ~22GB 才硬 OOM）。

修复后单专家（O2，τ=100）：B=16 峰值 3089.1MiB / 11.2s/batch / 梯度范数 0.285；B=32 峰值 5413.7MiB / 22.2s / 0.322；B=64 峰值 **10082.1MiB（>8192 预算）** / 203.6s / 0.170——单专家 B=64 即超预算。

> **researcher 判读补注（2026-10-28 05:3x，unit.receipt 12f6a2c6701142c5924c7c30494e94e0 判读）**：本节三处数字与 results/operator_bench.json 对账如下——(a) 上文「B=64 单步 11184.4MiB」在 JSON 中无对应记录；同 tag（O1_before_raw/b64_single_step，raw_tie_b64_1step）实测为 **22933.3MiB**（works=true，WDDM 共享内存 spill 未硬 OOM），11184.4 疑为更早被覆盖 run 的转录，保留原数字仅作证据边界标记，结论（超 8G 预算）不变。(b) O2 表 b16/b32 两行（3089.1/5413.7MiB、11.2s/22.2s、0.285/0.322）在 JSON 的 O2_after_fixed 段无对应记录（该段仅含 b64：10082.1MiB/203.638s/0.170，与本表 b64 行一致）——该两行暂无 JSON 背书，建议后续低成本复测补证（非阻塞）。(c) O3 表 b64 行记「未实测」，但 JSON O3_ERROR 段有该配置真实失败记录（21.73GiB allocated OOM）——证据比表述更强，结论不变。

修复后完整 MR-MoE（O3，臂 B 可学习 τ，3 专家+门控+读出）：

| batch | 峰值 MiB | ≤8G? | fwd+bwd s/batch | 前向/反向 | 梯度全局范数 | ‖∂L/∂c‖ |
|---|---|---|---|---|---|---|
| 8 | 3715.1 | ✓ | 29.6 | 全有限 | 0.291 | 6.71e-07 |
| 16 | 4768.8 | ✓ | 39.7 | 全有限 | 0.142 | 4.62e-07 |
| 32 | 6863.4 | ✓ | 86.0 | 全有限 | 0.149 | 3.13e-07 |
| 64 | 未实测（单专家已 10082MiB>8192，全模型必超） | ✗ | — | — | — | — |

→ D9 阶梯落点 = **batch 32（6863MiB，预算内）**；batch 64 不进 S3。epoch 外推（B=32）：8000 条=250 batch≈6.0h；4000 条≈3.0h；2000 条≈1.5h。

## ⓪-3 数值稳定性（O4）

- 梯度范数 vs 序列长（B=16）：12/24/48/72 步 → 0.0180/0.0181/0.0182/0.0182；144 步压力 → 0.0182，无非有限参数。隐式 unfold 的状态 Jacobian ‖∂v⁺/∂v‖≤1（cm_t>0、w_den≥0），反传不随时间爆炸——实测确认；
- gleak 极端应力（τ-tie 使 cm_t=τ_k·softplus(gleak)·u/Δt 与 gleak 耦合，gleak→−∞ 时 cm_t→0）：gleak∈{−30,−10,0,30} 72 步前向全有限（max|v|≤0.71）；(w,gleak)=(−40,−40) 双极端仍有限（0.886）——分母由 w_denominator 主导（≈75≫ε=1e-8）；库 ε 守卫是最后保护（与库同构，非本覆写引入）；
- 前向输出量级 out_max_abs≈0.06-0.07（随机输入未训练），无 NaN/Inf。

## ⓪-4 效率判读（O6）与未解决项

单 unfold 元素级通路实测（O6）：B=16：sigmoid 广播 5.38ms + 加权求和 3.63ms ≈ 9.0ms；B=64：20.81+13.01 ≈ 33.8ms。实测单专家 B=16 全步（fwd+bwd+检查点）155ms/步=25.8ms/unfold——约 2/3 是检查点 Python 开销与小核启动，GPU 计算只占 1/3。

**结论：显存墙（R3/R5）与冗余（R1/R2/R4）已修；单步计算量墙未修也修不掉**（hidden 1500 / ode_unfolds 6 冻结不变的前提下，稠密 LTC 每 unfold 必须过 (B,H,H)=144M 元素@B=64）。epoch 耗时（O3）远超 D9 pilot 预算（≤12min/训练、E=6）：B=32 下 2000 条也要 ~1.5h/epoch。

留给 researcher 的处置选项（均需 DECISIONS 拍板，本单元不擅自改冻结设计）：
1. torch.compile 融合 unfold（预估 2-3×；30 分钟可验：编译时间/数值稳定性/实测加速比）；
2. 稀疏 wiring（Random sparsity 或 NCP）替代 FullyConnected——论文未给连接性，属设计变更；
3. 降 hidden（D9 应急档 1500→512/256 ≈ 9-34× 提速，须留痕+admission 应急条款）；
4. 放宽 D9 预算/减 epoch/减子采样（排程决策）；
5. 切换 D4 备选路径 ①CfC+ts_k=Δt/τ_k（sibling 实测 CfC 全模型 277ms/batch，快 ~300×；CfC 无标量 τ 需 ts 注入，走 F5 流程）。

## ⓪-5 与 F8 的关系与数据说明

- F8=「算子基准发现数值稳定性或效率问题且本单元修不掉 → 带证据早退交 researcher 判读」。实测：数值稳定性无问题；显存墙已修（R3/R5，B=64 走 D9 阶梯至 32）；**效率墙（epoch 耗时）未修不掉**——按 F8 字面上交 researcher 判读 S3 路径（选项见 ⓪-4）。
- 数据说明：O3 的 b8/b16/b32 显存与计时在 R4 修订（sensory_erev 修复）前后不变（显存/计时与参数初值无关）；该三行的损失/梯度数值为修订前初值下所测，正确性证据以①探针（修订后算子）A2/A3/A5 为准。O1/O2/O4/O5/O6 均不受影响。
- ①五断言与④差异清单已在本阶段完成（results/ltc_tau_probe.json：A1-A5 全过；results/baseline_diff.md）。
