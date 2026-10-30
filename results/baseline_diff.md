# 论文差异清单 — arXiv:2606.12240v1（MR-MoE）vs 本课题冻结实现

- 阶段：S2-gate-checks ④（design.md §五 S2④；D4/U4 指定产物）
- 论文事实源：本地 PDF G:/research-papers/2606.12240.pdf（§2.2 Eq.3/4、§2.4 Eq.8-11、§3.3、§4.2、checklist #5/#7，design 单元复读全文）
- 总则（U4）：论文无官方代码（checklist #5「Not yet」）、单次运行无误差棒（#7）、未给 τ 数值/batch/epochs；本课题**不以复现论文绝对值为目标**，贡献=该具体缺口（§4.2 手工固定 τ）上的受控单变量实验 + 补三种子统计 + 学到的 τ 数值与排序记录。以下差异全部为冻结选择，改任何一项须走 DECISIONS。

## 1. τ 参数化：D2 保序累积增量 vs admission 字面公式（本课题实现口径）

| 项 | admission 预注册字面 | 本课题冻结实现（D2） | 差异理由 |
|---|---|---|---|
| 参数化 | τ_k = softplus(a_k) + Σ_{j<k} softplus(b_j) | τ_k = Σ_{j≤k} softplus(c_j)（累积正增量，c∈R³ 共享） | 字面形式**不保序**（τ_k−τ_{k−1}=softplus(a_k)−softplus(a_{k−1})+softplus(b_{k−1}) 可为负）；累积正增量使 τ_1<τ_2<τ_3 由构造严格保证（顺序不可能反转），与 brief「softplus(a_k)+有序约束/累积正偏移」同一意图，且与云端回执推荐的累积正增量一致（UMNN/S4 为工具先例） |
| init | — | Δ=(1,9,90) → τ=(1,10,100)，c_j=log(e^{Δ_j}−1)≈(0.5413, 9.0, 90.0) | ∂τ_k/∂c_j=sigmoid(c_j)·1[j≤k]≈(0.63, 1.0, 1.0) 不消失（A3 实测梯度到达） |
| 臂语义 | A=固定 τ buffer；B=可学习 | 同左；A/B 唯一差=tau_sched.c 是否 nn.Parameter（A5 实测可学习参数差恰 3 标量） | D10 单变量纪律 |

## 2. 论文未给项的冻结值（D4；均为设计冻结而非论文复现）

| 项 | 论文口径 | 本课题冻结值 | 备注 |
|---|---|---|---|
| τ 数值 | §2.4 只要求 τ₁≪τ₂≪…≪τ_K；§3.3 未给 | 1:10:100（小时，dt=1h 网格） | 作者自认手工指定（§4.2）；备选固定比值 1:3:30 为 H2 条件臂 A2 |
| batch size | §3.3「fixed across all experiments」未给数值 | 64 | D9 阶梯 64→32→16（⓪ 实测显存无需降档见 operator_bench.md） |
| epochs | 未给 | pilot=6 / full=20 | D9 断点：pilot 单训练 ≤12min（⓪ 实测耗时见 operator_bench.md，S3 排程须据此复核） |
| 门控 MLP 尺寸 | §3.3「small MLP」 | Linear(d,64)→Tanh→Linear(64,K)→softmax | 激活未给，冻结 Tanh（与 sibling 58b37930 实现对齐）；门控输入 z(t)=u(t)（论文未给口径，冻结当前输入） |
| 读出 | 每专家线性 C_k | Linear(1500,1)→logit，y=Σπ_k y_k，BCEWithLogits | sigmoid 为 BCE 所需 |
| 专家 hidden | §3.3「1500 neurons」 | 1500（ FullyConnected，ode_unfolds=6，implicit_param_constraints=True ） | 唯一论文给定值；D9：hidden 恒 1500 不动 |
| attention | Eq.8 基础版无 / Eq.9-11 有 | 无 attention（取基础 MR-MoE） | attention 变体=另一个模型，本课题不碰 |
| elapsed_time | dt=1h 网格（D3） | 1.0，timespans=None | B>1 逐样本 timespans 广播崩（A4 记录），本课题不触发 |
| 训练流程 | §4.1 联合训练 | 联合训练全部参数（单优化过程） | 解耦/阶段训练是 sibling 58b37930 靶点，本课题不碰 |
| 统计 | 无误差棒（#7） | 种子 {0,1,2}，mean±std，逐种子配对明细（D6） | 补论文缺失 |

## 3. CfC vs LTC 细胞选择理由（D4 定稿；与 sibling 58b37930 差异）

- 本课题单变量就是 τ：**LTC 有精确的每神经元时间常数 τ=cm/gleak**（implicit 模式下 cm、gleak 均过 softplus），与论文 §2.2 Eq.3/4 的 dx/dt=f/τ、Δt/τ 缩放形式同构；τ-tie 注入=覆写 cm_t 使 softplus(cm):=τ_k·softplus(gleak)（即 τ=cm/gleak=τ_k），一处标量绑定、语义闭合。
- sibling 58b37930 选 **CfC**：其课题改训练流程（阶段化解耦），需要 CfC 的时间插值通路；CfC 默认模式无标量 τ（时间经输入依赖的 t_a·ts 进入），不适合本课题的标量 τ-tie。
- 两课题共享基线实现的分工由白天推进会决定（EV-12）；本课题 mr_moe.py 的 τ-tie 算子与门控/读出结构与 sibling 同构（门控 Linear-Tanh-Linear、逐专家线性读出、BCEWithLogits），差异仅在专家细胞与 τ 注入路径。

## 4. 备选路径（D4 预注册：探针失败时按序切换，两臂必须同路径）

1. **CfC + ts_k=Δt/τ_k**：每专家 CfC 标准胞体，timespans 以 (1,L) 常向量注入（sibling 实测 B>1 可用形态；逐样本 (B,L) 崩）；
2. **最小自定义欧拉细胞**：x←x+(Δt/τ_k)·tanh(Wx+Uu+b)，照论文 Eq.4 自实现；
3. 三条路径都保持 A/B 单变量结构（唯一差=τ const vs 保序可学习）。当前状态：路径 0（LTC-tie）① 探针五断言结论见 results/ltc_tau_probe.json；若断言失败按 F5 处置。

## 5. 保真度缺口与方向性健全性（D5；E6 竞争解释的区分依据）

- 论文 MR-MoE AUROC 0.61 vs Moor 2023 原文 0.846：绝对差值不构成本课题判据（U4 明确不以复现绝对值为目标）；
- 方向性健全性（S3/S4 核）：(i) 臂 A（固定 τ MR-MoE）须优于单体 LNN（同预处理同预算，论文 0.61>0.55 方向）；(ii) B(0)≡A(0) 前向 max abs diff<1e-6（A2 实测）；(iii) 参数审计差恰 3 标量（A5 实测）；
- 单体 LNN 时间常数：论文 §3.5.2 未给，本课题取均匀 dt=1（与臂 A 语义一致），记此节。

## 6. 评测与数据口径（D3/D8；S2b ② 实测获取）

- 数据：主 Moor 2023 harmonized（原论文同数据）/ 退 PhysioNet-CinC 2019；Che et al. 2018 划分；dt=1h 网格；入院截断 72h；标签=未来 6h Sepsis-3 onset；
- 预处理：缺失前向填充 + 缺失 mask 作为输入通道拼接（禁 0 填充裸奔，taste 死法①）；min-max 仅 train 拟合；
- 评测：主=入院级 AUROC/AUPRC（窗口概率最大值参评）；次=小时级；两者都落表（D8）。

## 7. 算子层差异（⓪ 实测；fb-ab50d0ec9f/D12）

- 库 LTCCell → OrderedTauLTCCell：唯一覆写=cm_t 计算（τ-tie），其余求解器逐行照抄；⓪ O5 实测「τ-tie(τ=1, cm≡gleak) ≡ 库 LTCCell」与「裸覆写 ≡ 修复后」前向逐位一致（max_abs_diff=0.0）；
- ⓪ 修复 R1-R5（gleak softplus 外提 / mask 折叠 / 每 unfold 梯度检查点 / wiring 构建缓存+预 build 修订 / 专家序列级检查点）只改冗余与显存、不改数值（O5 + 探针 A2 证据）；
- 显存现实：τ-tie 裸形态 72 步 BPTT 在 8G 任何批量 OOM（sibling 同款实证）；修复后全模型 B=32 峰值 6863MiB（预算内），B=64 超预算 → D9 阶梯 batch 64→32 为 S3 实际档位（探针 A4 记录）；
- 稠密 LTC hidden 1500 的单步计算量现实（O6 实测 ~34ms/unfold@B=64）与 S3 排程影响：全模型 B=32 ≈ 86s/batch，epoch（2000 条）≈1.5h，远超 D9 pilot 预算——**已由 researcher 判读拍板（D14，2026-10-28T05:3x，判读全文 notes/researcher-interventions/bf416052145c4190ba682d9e5e85bbae.md）**：⓪-4 的 5 选项裁定=③降 hidden 应急档（主路）+④预算重锚+①compile 可选探针，②稀疏 wiring 否决，⑤CfC+ts 保留预注册回退（F5b 触发器）。第一手实测（experiments/emergency_tier_probe.py，h1500 锚点与 O3 对账偏差<2%）：hidden 256/batch 64 全模型 4.43s/batch、峰值 1344.6MiB、epoch(2000)=137s、E=6 pilot=13.7min（D9 软顶 12min 实质达成）、E=20 full=46min/run；hidden 512 E=6=52min 超软顶不用；batch 128 实测反劣（16.8s）不升档；
- **规模偏离记账（D14，S6 终判必须标注）**：S3-pilot 与 S5-full 在 hidden 256 应急档运行（batch 恢复 D4 冻结值 64，ode_unfolds=6 不变），可训练参数 28.5M→1.0M（27×），子采样 2000 条。授权链=admission first_milestone 明文「pilot 可缩 hidden」+D9 预留条款「admission 允许的降 hidden 仅作 8G 应急档且留痕」（激活理由=显存墙与计时墙同根于稠密 H=1500 (B,H,H)×6 unfolds 在 8G 卡的固有成本）。H1a/H1b 结论范围限定为「hidden 256 应急档+2000 条子采样+B=64」；偏离对称作用于 A/B 两臂（D10 单变量纪律不破，E2 容量效应竞争解释的区分力不受影响——A2 条件臂仍在）；论文规模（hidden 1500）确认性运行需 >8G 显存资源，列为 S6 后可选后续。若 F5b 触发切 CfC+ts 路径，本节须同步更新（两臂必须同路径）。
- τ-tie 结构性死参数：库分配的 cm（每专家 1500 个）在两臂同样不用（A5 记录），不影响 A/B 差集。

## 8. 声称边界（定位纪律，C4/C5）

- 不声称「让 τ 可学习」机制新颖：LTC 2021（arXiv:2006.04439）τ 为随输入变化的可训练参数；LiquidTAD（arXiv:2604.18274）逐层级可学习衰减率 λ_l=1/τ_l；arXiv:2602.06997 log-space 可学习 τ 自组织三簇——均全文级划界；
- 贡献声称=「MR-MoE §4.2 缺口上的受控单变量实验（固定 1:10:100 vs 保序可学习，init 等价）+ 补三种子统计 + 学到的 τ 数值与排序记录」；不外推为「可学习 τ 普遍更优」（声称范围=分析范围）。
