# 设计冻结 — 让多时间尺度液态专家自己学时间常数（MR-MoE §4.2 可学习时间常数缺口）

- 单元：unit.design（research-planner profile，夜间生产并行车道，监督试点，仅本地落盘）
- 时间：2026-10-27T00:4x–00:5x+08:00（夜窗终点 epoch 1793058899 = 2026-10-27T07:56:39+08:00；本单元为 30 分钟设计单元，预计 30 分钟内收口）
- 状态：novelty/fit gate 已过（admission.json decision=admitted，approval auto:cand-d284141887d3f6b8）；本文件冻结可证伪假设、竞争解释、判决性实验与停止条件。机器可读版=workflow.json（S1-design-freeze status=done + result.milestone.status=accepted，首科学阶段 S2-gate-checks status=ready 可派）。
- 输入基线：admission.json（schema research-topic-admission/v1）、literature/related-work.md（EV-01..EV-15、search_log 15 条）、brief.md、research-approval.json、research/taste.md、research/_state/user-direction.json（2026-10-27T00:19:33 更新，2026-11-03 到期，设计日在效）。
- 本单元边界：不训练、不跑 benchmark、不下载数据、不建仓、不公开推送、不碰主课题与其他课题文件（兄弟课题 58b37930 的文件只读参考、禁写）；本 profile 无 terminal（tool_inventory 全库核实：仅文件/网络/research 工具），git commit 由下一执行单元顺手完成（见 §八）。

## 一、开工复核结论（四项输入逐项过）

### 1.1 admission.json — 复核通过（pass）
- 结构：schema=research-topic-admission/v1、decision=admitted；claims C1–C8 每条可解析到 EV 引用；novelty_checks 六键全真（real_benchmark_exit=true）；uncertainties U1–U8 逐条有归属；first_milestone.status=accepted（验收标准=30 分钟 pilot：固定 τ 基线跑通+梯度到达 τ+不发散+τ 保序+pilot 日志/两臂曲线/τ 数值排序/差异清单）；user_fit_evidence 三条（direct_user_preference / taste_document / active_project）。
- 缺口坐实：主论文 arXiv:2606.12240v1 §4.2 原文（EV-01，本地 PDF G:/research-papers/2606.12240.pdf 第 9 页，本单元复读全文确认原文逐词一致）——「the time constants τk associated with each expert are manually specified and remain fixed during training… An important extension is to make the time constants learnable parameters」；本单元复读同时确认 §2.4 Eq.8（τ₁≪τ₂≪…≪τ_K，小 τ=快动态）、§2.2 Eq.3/4（dx/dt=f/τ 的欧拉离散 x(t+Δt)=x(t)+(Δt/τ)f(x,u)——时间常数以 Δt/τ 缩放状态更新，这是 D4 τ 注入路径的论文依据）、§3.3（K=3、每专家 1500 神经元、Adam lr=1e-3、gating 为 small MLP、batch「fixed across all experiments」未给数值）、checklist #5（无代码数据）、#7（无误差棒与显著性检验）、References 14 条。
- 定位纪律沿用（C4/C5）：**不声称「让 τ 可学习」机制新颖**（LTC 2021 τ 是随输入变化的可训练参数；LiquidTAD arXiv:2604.18274 逐层级可学习衰减率 λ_l=1/τ_l；均全文级划界）；贡献=在 MR-MoE 该具体缺口上的受控单变量实验 + 补原论文缺失的三种子统计 + 学到的 τ 数值与排序记录。
- notes_for_design 四项处置：①τ 保序参数化本单元定稿（D2，与 admission 字面公式的差异及理由记录在案）；②数据二选一冻结为决策过程（D3，S2 实测选择）；③基线配置冻结+差异清单=S2 产物 results/baseline_diff.md（D4）；④S2 向前引文与云端 rid 复查——本单元已完成（§1.5 U1/U2）。

### 1.2 related-work 矩阵 — 复核通过（pass）
- 三条 closest_work 均为全文级：主论文（本地 PDF 全文，EV-01..EV-04）；LTC arXiv:2006.04439（ar5iv 全文补读，EV-05/EV-06）；LiquidTAD arXiv:2604.18274（ar5iv 全文补读，EV-14）。
- 相邻范式均有不占坑定位：Time-MoE/MOIRAI-MoE/Seg-MoE（频率分工、Transformer 骨干、无连续时间 τ）；MTLiquid arXiv:2609.33232（多任务 LNN 不动 τ）；arXiv:2602.06997（log-space 可学习 τ 自组织快中慢三簇，EEG 情绪识别、单体网络）；arXiv:2603.18331（Neural ODE 理论）。
- 引用图：OpenAlex W7164356256 cited_by_count=0 且 citing works=0——本单元第三时点复确认（§1.5 U1）；S2 通道全程 429 未交叉验证（EV-11，unknown）——不得写成「无人跟进」。
- LiquidTAD 划界（U8）：它逐层级学通道共享标量 λ_l，无专家混合/门控路由、无跨专家保序、任务是动作检测、无固定 vs 可学习受控对照——不占本课题坑；本课题「每专家一个保序标量 τ × MoE 门控 × 败血症 × 单变量对照」与之差异写入 design（D2/D4）与后续论文的 related work。

### 1.3 用户品味依据 — 复核通过（pass）
- 方向文件：research/_state/user-direction.json 在效（2026-10-27T00:19:33 更新，2026-11-03 到期）——液态神经网络落地应用、只做别人论文里一个具体缺口的小改进、产出能放 GitHub 的小仓库。
- taste.md [correction] 2026-10-24 选题口径逐项命中：单缺口（§4.2）、单改动（τ 参数化）、单判据（AUROC/AUPRC/τ 保序）、模块级、单卡 8G、1-2 周、pilot 30 分钟见数；无大主题拼接。
- 玩具红线（[correction] 明令禁止合成数据+自制小网络冒充验证）：本课题=真实公开败血症数据 + 按论文 Eqs.1-18 自实现真实 MR-MoE（ncps 0.0.2 LTC cell），无合成数据；数据拿不到即停（F2），不用替代数据。
- 拒稿死法对照入设计：①「实现层缺陷污染全部对照」（0 填充参与前向/反向）→ 预处理冻结为前向填充+缺失 mask 输入通道（D3）；②「声称范围=分析范围」→ 贡献声称限定在受控测量，不泛化「可学习 τ 普遍更优」；③「相邻范式缺失即死」→ §1.2 矩阵已覆盖（LTC/LiquidTAD 全文级）；④「组合无必要性论证」→ 本课题不是组合型贡献，是单缺口单点受控测量，必要性=论文作者自认缺口（§4.2 原文）。

### 1.4 真 benchmark 入口 — 复核通过（pass，获取与接口实测留 S2 门检）
- 数据入口（两来源，admission EV-10）：主=Moor et al. 2023（eClinicalMedicine 62:102124）harmonized 国际 ICU 败血症集（136,478 例、US/NL/CH 四库、Sepsis-3 标注，原论文即用此数据+Che et al. 2018 划分）；回退=PhysioNet/CinC 2019 败血症挑战公开数据（同 Che 2018 协议，公开可下载）。云端 digested 备注（rid 33cf798d：Moor >10GB 需凭证、PhysioNet 50MB CC-BY 4.0、推荐前者）为二手信号，**不得单独作为冻结依据**，S2 按序实测选择并留痕（D3）。
- 模型入口：按论文 Eqs.1-18 自实现 MR-MoE（K=3 LNN 专家 + softmax 门控 MLP + 每专家读出；原论文无代码，checklist #5「Not yet」）。
- 本机栈：ncps 0.0.2 已装（G:/ai/python11/Lib/site-packages/ncps，本单元直读源码核实：LTCCell/LTC 支持 elapsed_time/timespans，默认 ts=1.0；CfCCell 默认模式无标量 τ——时间经 t_a·ts 输入依赖地进入；sibling 实测记录 PATH 首位 G:/ai/python11/python.exe 带 torch 2.10.0+cu128、RTX 5060 8G 可用）；sibling 4a1c4c0a 实测 CfC B>1 批量 timespans 崩溃（LTC 源码核对：LTCCell 内 cm/(elapsed_time/u) 对 (B,) 型 ts 与 (H,) 型 cm 广播冲突，B>1 且传逐样本 timespans 预期同样崩——本课题走固定 dt=1h 网格、timespans=None，不触发该路径，S2 记录在案）。
- **未实测项（如实记录）**：Moor 2023 实际获取方式（公开 harmonized 集 vs 四库申请）与 LTC τ-tie 细胞行为——本单元无 terminal 不能下载/探针，按预注册决策过程冻结（D3/D4），实测放 S2 门检；两来源都拿不到即停（F2），不用替代数据。

### 1.5 本单元增量核验（U1 收窄、U2 收窄、U3/U5 按门检冻结、U6 参数化定稿、U8 划界落设计）
- U1（向前引文）：本单元第一手复确认——lit_query citations(id=W7164356256, dir=in, source=openalex) ok=true count=0（2026-10-27T00:50:42+08:00；前两次：2026-10-25T21:26 直连同值、2026-10-27T00:16:22 sibling 同工具同值）。状态=unknown 收窄为「OpenAlex 三时点 0、S2 仍未交叉验证」；保留「引用窗口约 4.5 个月」caveat，不写「无人跟进」；F4 触发器保留。
- U2（云端 rid eb215c80344add8f 未收割）：已核实收割并入账——research/harvest/INDEX.md 第 27 行（2026-10-26 选题官 digested=true）：事实与已落图谱节点重复（MR-MoE 固定 τ、LiquidTAD 最近邻、UMNN/S4 保序参数化、PhysioNet 2019 基准建议），无同型占坑证据；附转述勘误一条（该报告称缺口在 Limitation 章节，全文口径=Future Work §4.1/§4.2——admission 已按全文口径写，正确）。结论：不重开 screening；F4 触发器保留。
- U3（数据获取）：open_by_design——D3 冻结决策过程（主 Moor 2023 → 退 PhysioNet 2019 → 两败即停 F2），S2 步骤②实测（例数/序列长度分布/标签率落 results/data_check.json）。
- U5（ncps 0.0.2 τ 接口）：本单元源码级收窄——CfC 默认模式无标量 τ；LTC 的 τ=cm/gleak 每神经元一份（implicit 模式下 cm、gleak 均过 softplus）。固定/可学习 τ 的注入路径定稿为 D4 的 τ-tie（子类覆写 cm_t 计算，softplus(cm)=τ_k·softplus(gleak) 的数值爆炸反解问题由此绕开）；S2 步骤①五断言实测。
- U6（保序参数化能否保持顺序、τ 是否塌缩）：本单元定稿——累积正增量参数化从构造上保证 τ_1<τ_2<τ_3（每个增量 softplus(c_j)>0），顺序不可能反转；残余风险=塌缩到边界（D7 判据+每 epoch τ 日志+F6 停止条件）。机制先例不等于本设置已验证（LTC 的 τ 是每神经元随输入变化，与本课题「每专家一个标量」不同参数化）。
- U8（LiquidTAD 划界）：写入 D2/D4 与 §1.2，后续论文 related work 按此口径。

## 二、可证伪假设（冻结）

### H1a 性能主假设（判决性）
- 陈述：在冻结的败血症 benchmark（D3 数据与预处理、Che 2018 划分、D8 评测）上，除 τ 参数化外一切相同（D10），保序可学习 τ 臂 B 相对固定 τ 臂 A，三种子平均 ΔAUROC ≥ +0.01 且 ΔAUPRC ≥ −0.01（逐种子配对，B−A）。
- 立论依据：§4.2 作者自认手工固定 τ「may not optimally reflect the underlying dynamics of the data」，提出可学习化以「automatically discover appropriate temporal scales from data」（EV-01 原文）；数据驱动的 τ 应更贴合败血症多尺度动态；架构、专家数、门控、注意力（无）、数据处理全部不变，差异可归因于 τ 是否由数据决定。
- 可测量预测（预注册）：S5 full 出 mean±std 表；判据两项同时满足→supported；任一不满足→falsified（负结果落盘，按 §六 映射）。
- 失败条件：见 §七 F1；pilot（S3）只做方向性与管线健全性，不终判。

### H1b 结构健全性假设（可解释性前提）
- 陈述：B 臂三种子均满足——(i) 训练后 τ_1<τ_2<τ_3 数值核验通过（参数化构造保证，实现正确性自检）；(ii) 无边界塌缩（D7 判据：任 τ_k<0.01h 或 >1e4 h，或任增量 softplus(c_j)<1e-4 或 >1e4）；(iii) τ 确实在学（≥2/3 种子至少一个专家的相对移动 |τ_final−τ_init|/τ_init > 5%）。
- 立论依据：没有可解释的 τ 就没有可测量的结论（admission stop_if_not_novel 第 2 条同精神）；若 τ 不动，B≡A，负结果不可解释。
- 失败条件：(ii) 或 (iii) 不成立→H1b 证伪；此时 H1a 的读数不可解释，按 F6/§六 inconclusive 分支处置，不得把「B≈A」当作「可学习 τ 无增益」的干净负结果。

### H2 机制归因假设（条件臂，区分「数据驱动发现」与「更优固定比值」）
- 陈述：若 H1a 成立，检验增益来自「数据驱动找到更优时间尺度」而非「任何一组更优的手工有序比值」——条件臂 A2=固定 τ 比值 1:3:30（另一组手工有序值，其余同 A，3 种子）。判据：B−A2 ≥ +0.01 支持「自适应发现」解释；|B−A2| < 0.01 且 A2−A ≥ +0.01 则增益可被固定常数复现，机制解释收缩为「可学习性只是搜索手段，手工选值即可获得」。
- 立论依据：§4.2 的主张是「自动发现合适的时间尺度」；只比 B vs A(1:10:100) 无法区分「学习有用」与「1:10:100 选得不好」。
- 执行条件：仅当 S4 verdict=continue 且预算允许时进 S5（预注册条件臂，默认不进首轮——先保证主对照干净）。
- 失败条件：B−A2 < +0.01 → H2 证伪，论文/README 声称相应收缩（声称范围=分析范围）。

## 三、竞争解释与区分实验（每条预注册区分法）

| # | 竞争解释 | 若真则看到 | 区分实验 |
|---|---|---|---|
| E1 | 实现伪影：τ-tie 子类意外改变别的计算（如 gleak 梯度缩放、cm 语义偏移） | B 臂 loss 曲线与 A 臂系统性偏移但 τ 没动；或 A5 参数审计不符 | S2 断言 A2（B(0)≡A(0) 前向逐位一致）+ A5（两臂参数差恰为 3 个标量 c）+ A3（梯度到达 c）+ 臂代码逐行审计（唯一 diff=τ_k 的 requires_grad） |
| E2 | 优化容量效应：B 臂多 3 个可学习参数=额外自由度，与时间尺度自适应无关 | B−A 增益与 A2−A 同向同幅；τ 移动大但增益与 τ 值无关 | H2 条件臂 A2（固定 1:3:30）+ A5 参数审计；若 B≈A2>B−A 解释收缩 |
| E3 | 种子噪声/小样本：增益在种子间不稳定 | std 压过效应；逐种子符号不一致 | 预注册三种子 mean±std + 逐种子配对明细全落表（D6）；效应落噪声带→inconclusive 分支（加种子）不轻弃 |
| E4 | 预处理/数据伪影 | A 臂 AUROC 异常低或两臂差异集中在特定缺失模式 | D10 单变量纪律构造性排除（两臂共享预处理/数据顺序/init 种子）；A 臂异常先查实现再判方向 |
| E5 | τ 塌缩/边界：序保持沦为数值空话 | τ_3→巨大或 τ_1→0；增量趋 0 | D7 每 epoch τ 日志 + H1b(ii) 判据；≥2/3 种子塌缩→F6 |
| E6 | 保真度缺口：自实现基线与论文绝对数值不同（论文 MR-MoE 0.61 vs Moor 原文 0.846） | 绝对 AUROC 低于论文 | D5 方向性 sanity（A 须胜单体 LNN，论文 0.61>0.55 方向）+ baseline_diff.md 差异清单；明确「不以复现论文绝对值为目标」（U4） |

## 四、设计决策冻结（D1–D11）

- D1 臂语义（沿用 admission 预注册字面）：A=论文式固定 τ（τ=(1,10,100) 小时，const buffer，requires_grad=False）；B=保序可学习 τ（D2 参数化，init 同为 (1,10,100)，nn.Parameter）；两臂均为论文式联合训练（全部参数单一优化过程，E 个 epoch）——训练流程不变量（§4.1 解耦训练是兄弟课题 58b37930 的靶点，本课题不碰）。
- D2 τ 保序参数化定稿：τ_k = Σ_{j=1..k} softplus(c_j)，k=1,2,3；增量 Δ_j=softplus(c_j)>0 → τ_1<τ_2<τ_3 由构造严格保证（顺序不可能反转）。init：Δ=(1,9,90) → τ=(1,10,100)，c_j=softplus^{-1}(Δ_j)=log(e^{Δ_j}−1) ≈ (0.5413, 9.0000, 90.0000)；∂τ_k/∂c_j=sigmoid(c_j)·1[j≤k]（大 c 下≈1，梯度不消失）。**与 admission 字面公式（τ_k=softplus(a_k)+Σ_{j<k}softplus(b_j)）的差异及理由**：该字面形式不保序（τ_k−τ_{k−1}=softplus(a_k)−softplus(a_{k−1})+softplus(b_{k−1}) 可为负），本形式是同一意图（brief「softplus(a_k)+有序约束/累积正偏移」）的保序实现，且与云端回执推荐的「累积正增量 softplus」一致（UMNN/S4 为该工具先例，记 method 节点）；差异记入 results/baseline_diff.md。A 臂 c 为 buffer、B 臂 c 为 Parameter，除此一模一样。
- D3 数据与预处理冻结：主=Moor 2023 harmonized（原论文数据），回退=PhysioNet/CinC 2019（S2 步骤②按序实测获取，选择留痕 DECISIONS，两来源皆败→F2 停）。小时网格重采样 dt=1；每条入院截断前 72h；缺失值前向填充 + 缺失 mask 作为输入通道拼接（禁 0 填充裸奔进前向/反向——taste 死法①）；min-max 归一化只用 train 划分拟合；Che et al. 2018 划分；标签=Sepsis-3 onset，预测目标=未来 6h 内 onset（y(t)=1[onset∈(t,t+6h]]，PhysioNet 2019/Che 2018 标准式）。
- D4 基线配置冻结（与论文差异清单=S2 产物 results/baseline_diff.md）：K=3 专家，每专家=OrderedTauLTCCell（ncps 0.0.2 LTCCell 子类，FullyConnected wiring，hidden 1500，ode_unfolds=6，implicit_param_constraints=True；唯一覆写=cm_t 计算改为 τ_k·softplus(gleak)/(elapsed_time/ode_unfolds)，即 softplus(cm)=τ_k·softplus(gleak)、τ=cm/gleak=τ_k 小时；其余求解器代码逐行照抄库源码）；gleak 每神经元可学习（两臂相同），其余 LTC 参数（vleak/sigma/mu/w/erev/sensory_*）默认初始化并可学习。门控=softmax(MLP hidden 64)（论文「small MLP」未给尺寸），输入 z(t)=u(t)（论文未给门控输入口径，冻结为当前输入）；每专家线性读出 C_k: R^1500→R（logit），y(t)=Σ_k π_k(t)·y_k(t)，BCE。Adam lr=1e-3（论文 §3.3）；batch 64（论文未给数值，冻结并记录）；E：pilot=6、full=20（论文未给）；无 attention（基线取 Eq.8 基础 MR-MoE，attention 变体是另一个模型）；elapsed_time=1.0（dt=1h 网格，timespans=None）。**与 sibling 58b37930（CfC 专家）的细胞选择差异及理由**：本课题单变量就是 τ，CfC 默认模式无标量 τ（时间经输入依赖的 t_a·ts 进入），LTC 有精确的每神经元 τ=cm/gleak 且与论文 Eq.3/4 的 Δt/τ 形式同构；预注册备选路径（S2 探针失败时按序切换，两臂必须同路径）：①CfC + 每专家 ts 缩放 ts_k=Δt/τ_k；②最小自定义欧拉细胞（x←x+(Δt/τ_k)·tanh(Wx+Uu+b)，照论文 Eq.4 实现）。三条路径都保持 A/B 单变量结构。
- D5 方向性健全性（保真度操作化）：(i) A 臂（固定 τ MR-MoE）须优于单体 LNN（同预处理同预算，论文 MR-MoE 0.61>单体 LNN 0.55 方向）；(ii) B(0)≡A(0)：同种子同初始化下 B 臂前向输出与 A 臂逐位一致（τ 初始化等价，max abs diff <1e-6）；(iii) 参数审计：A/B 可学习参数差恰为 3 个标量 c。任一不过→查实现，不算方向失败。
- D6 统计：种子={0,1,2} 固定表；报 mean±std；所有臂间差报逐种子配对明细；std 不省略。
- D7 τ 日志与塌缩判据：每 epoch 记录 τ=(τ_1,τ_2,τ_3)、增量 Δ=(Δ_1,Δ_2,Δ_3)、‖∂L/∂c‖、与初值相对移动 max_k|τ_k−τ_k^init|/τ_k^init；产物 results/*_tau_log.json。塌缩判据（H1b(ii)）：任 τ_k<0.01h 或 >1e4 h，或任增量 <1e-4 或 >1e4；≥2/3 种子塌缩或 loss NaN/爆炸 → F6 分支。
- D8 评测：主指标=入院级（admission-level）AUROC/AUPRC（Che 2018/PhysioNet 口径：每条入院取其窗口概率最大值参评）；次指标=小时级 AUROC/AUPRC；两者都落表。
- D9 算力断点与内存阶梯：pilot 单次训练 ≤12 分钟（硬顶 20 分钟）；子采样阶梯 8000→4000→2000（hidden 恒 1500 不动；admission first_milestone 允许的「pilot 可缩 hidden」仅作 8G 装不下的应急档，降 hidden 必须留痕 DECISIONS）；batch 阶梯 64→32→16、ode_unfolds 恒 6（默认不动）；每 epoch 原子落 checkpoint（experiments/ckpt/ + results/pilot_progress.json），kill 无损续跑；full 按 epoch checkpoint 分段交单元，不硬撑。LTC 1500×1500 稠密中间激活的内存风险由 S2 断言 A4（峰值显存实测）兜底。
- D10 单变量纪律：除 τ 参数化（const buffer vs 保序可学习参数）外一切相同——同初始化种子、同数据顺序、同预处理、同超参、同评测代码路径；A/B 每种子各一次独立训练（不共享 checkpoint，避免初始化偏置）。
- D11 条件臂 A2（H2 判据）：固定 τ 比值 1:3:30（另一组手工有序值，Δ=(1,2,27)），其余同 A；仅当 S4 verdict=continue 且预算允许时执行（3 种子）；用于区分「数据驱动发现更优尺度」与「更优固定比值即可获得」。

## 五、判决性实验（阶段链；机器可读版见 workflow.json）

- S2-gate-checks（kind=experiment，ready，首科学阶段）：①ncps 0.0.2 LTC τ-tie 探针 experiments/ltc_tau_probe.py 五断言——A1 动态可分（同输入下三专家单步状态变化幅度序：‖Δx_1‖>‖Δx_2‖>‖Δx_3‖，fast 专家动得多）；A2 B(0)≡A(0) 前向一致（max abs diff <1e-6）；A3 梯度到达 τ 参数（∂L/∂c_j 非零有限，记录 ‖∂L/∂c‖ 与参数梯度中位数对比）；A4 稳定性与内存（72 步前向有限无 NaN；frozen batch 下峰值显存实测，超 8G 按 D9 阶梯降档并记录；B>1 逐样本 timespans 行为记录——预期崩，本课题不用该路径）；A5 参数审计（A/B 可学习参数差恰 3 标量）。②数据获取实测 experiments/data_check.py：Moor 2023 主 / PhysioNet-CinC 2019 退，记录例数、序列长度分布、标签率，落 results/data_check.json（两来源皆败→F2 停，禁合成替代）。③experiments/mr_moe.py 按 D4 冻结实现 MR-MoE（K=3 OrderedTauLTCCell 专家+门控 MLP+读出，A/B 同构造）+ experiments/data_pipeline.py（D3 预处理）+ 单体 LNN 基线，前向/反向 smoke。④results/baseline_diff.md 论文差异清单（含 D2 参数化与 admission 字面差异、论文未给项冻结值、CfC/LTC 选择理由）。⑤git 提交设计产物（本 profile 无 terminal 的移交项，范围=design.md/workflow.json/STATUS.md/notes/*，research 根仓本地 commit 不推送）。
- S3-pilot（kind=experiment，S2 后放行）：子采样 8000 条（不足则 4000→2000，hidden 恒 1500），E=6；A 臂×种子{0,1,2}、B 臂×种子{0,1,2}；单体 LNN×seed0 sanity（D5(i)）；每 epoch val loss + 入院级/小时级 AUROC/AUPRC；τ 日志（D7）；出 pilot 表（CSV+MD+timing+progress）。首里程碑①②③在此验收（admission.first_milestone：①固定 τ 基线损失下降且 AUROC 可算；②τ 换保序可学习后梯度到达 τ、训练不发散、τ 顺序保持；③含数字的 pilot 日志/两臂验证曲线/训练后 τ 数值与排序/差异清单）。
- S4-pilot-verdict（kind=analysis，verifier）：独立复算（先 python 复算再下判，勿只读 MD 汇总）；臂语义审计（A5 参数差、A2 init 等价、D10 纪律）；D5 sanity；出 pilot 方向性判读（非 H1 终判）；continue→S5；发现臂语义偏离→redesign；发现 F2/F3 级阻塞→stop。
- S5-full-experiment（kind=experiment）：full 划分、E=20（或 D9/D2 修订值，留痕 DECISIONS）；A×3 种子 + B×3 种子；条件臂 A2×3 种子（H2 判据，S4 verdict=continue 时执行）；入院级+小时级指标 mean±std+逐种子明细；τ 全量日志；按 epoch checkpoint 分段交付。
- S6-final-verdict（kind=analysis，verifier）：按 §六 门线终判 H1a/H1b/H2；verdict 落 results/final_verdict.md/json；负结果按 FAILURE-PATTERNS 语义落盘；更新 STATUS/DECISIONS/RETRO。
- S7-repo-package（kind=packaging）：本地小仓库骨架（reproduce 一条命令 + README 中文大白话 + 结果表 + 差异清单）；**公开推送仅凭用户明确授权**（今晚不推送）。

## 六、门线与结果映射（预注册）

- 过门线（H1a，S5 full，两项同时）：ΔAUROC(3-seed mean, B−A) ≥ +0.01 ∧ ΔAUPRC(3-seed mean, B−A) ≥ −0.01。H1b 三项（序保持/无塌缩/τ 在学）为可解释性前提，同时核验。
- 结果映射：
  - H1a 两项全过 且 H1b 全过 → verdict=supported，next_action=continue → S7 打包；H2 由 A2 条件臂 adjudicate（若执行）。
  - H1a 未过 且 H1b 全过（τ 确实学了、没塌缩、实现类问题已排除）→ verdict=falsified：负结果落盘（该设置下保序可学习 τ 不优于手工 1:10:100），方向不再投算力；RETRO 给 ≥3 条替代方向（每条 30 分钟验法+失败条件，选 1 条写 STATUS 下一步）。不调参再试（调参再来不算转向）。
  - 效应落噪声带（−0.01 ≤ ΔAUROC < +0.01 或 AUPRC 降在 0..0.01 且 std 压过）→ verdict=inconclusive：加种子（+3）或加 epoch 放大后再判一轮，不轻弃（2026-10-24 用户拍板「不要太轻易放弃」）。
  - H1b 不过（塌缩/τ 不动/失序）→ H1a 读数不可解释：记 inconclusive 并触发 F6 分支审查。
  - S2/S3 发现实现或数据阻塞 → 按 F2/F3/F5/F6 处置。
- pilot（S3/S4）不终判 H1；pilot 只回答：管线通不通、A 臂正不正常、B 臂方向差多少（带 ±std 的方向性读数）、τ 动不动。

## 七、停止条件（冻结）

- F1 负结果：full 过门线未达且 H1b 全过且实现类问题已排除 → 负结果落盘 + RETRO ≥3 条替代方向，方向不再投算力。
- F2 无真实 benchmark：Moor 2023 与 PhysioNet 2019 两来源获取失败或与 Che 2018 协议不兼容 → 即停（无真实 benchmark 不做实验，禁合成替代）。
- F3 基线不可能：固定 τ MR-MoE 在 pilot 预算内无法稳定训练（lr=1e-3 与 3e-4 两档均 loss 爆炸/NaN）→ 即停（没有基线就没有比较）。
- F4 占坑命中：S2/OpenAlex 出现 2026-06 后发表的「MR-MoE/LNN-MoE 败血症设置 × 可学习/自适应专家时间常数」同型工作 → stop → screening-negative 并记录占坑来源（LTC/LiquidTAD 等机制先例不触发，见 §1.5）。
- F5 实现墙：D4 三条 τ 注入路径（LTC-tie / CfC-ts 缩放 / 自定义欧拉细胞）在 S2 探针均不可行 → redesign（换注入路径）或 stop，不硬撑。
- F6 τ 塌缩/失序：≥2/3 种子 τ 塌缩到边界（D7 判据）或训练不稳定（NaN/发散）→ 即停（没有可解释的 τ 就没有可测量的结论；admission stop_if_not_novel 第 2 条）。
- F7 论文侧变化：发现原论文 v2 或同组跟进工作已包含该扩展 → 即停。
- 单元级：硬超时内先 checkpoint 再继续；拿到改变决策的关键发现即早退（保存证据、更新 STATUS/DECISIONS、出回执）。

## 八、复现入口与遗留（工具墙如实记录）

- 准入与证据链：admission.json（C1–C8→EV 可解析）；literature/related-work.md（EV-01..15、search_log 15 条）。
- U1 复现：lit_query citations(id=W7164356256, dir=in, source=openalex) → count=0（2026-10-27T00:50:42；前次 2026-10-25T21:26 直连、2026-10-27T00:16:22 sibling 同工具，三时点同值）。
- benchmark 入口：Moor et al. 2023（eClinicalMedicine 62:102124）/ PhysioNet-CinC 2019（回退）；论文 arXiv:2606.12240v1（本地 PDF G:/research-papers/2606.12240.pdf 19 页，本单元复读 §2.2/2.4/3.3/4.2/checklist）。
- 模型栈：ncps 0.0.2（G:/ai/python11，本单元直读 cfc_cell.py/ltc_cell.py/ltc.py/wirings.py 源码核实接口；torch 2.10.0+cu128 据 sibling 实测记录，S2 import 探测复核）。
- 状态复验：`python scripts/research-project-state.py` / `python scripts/research-workflow.py show --topic 课题目录`（ready 阶段应=S2-gate-checks）。
- 工具墙记录：本 profile（research-planner）无 terminal/kanban（tool_inventory 全库 76 工具核实）→ ①git commit（research/ 根仓库，本地 main，不推送）由 S2 执行单元顺手完成，范围=design.md、workflow.json、STATUS.md、notes/DECISIONS.md、notes/RETRO.md（同 sibling 4a1c4c0a/58b37930 先例）；②本单元网络只读使用 lit_query（U1 复确认）与本地 PDF/源码读取，无 SSL 墙（web_search/web_extract 未用）；③求助事件脚本（researcher-request.py）需 terminal，本单元无求助需求未使用。
- 机械伪影预告：S2/S5（kind=experiment）回执 next_action=continue 时，驱动器 apply_result 会自动追加同名 -continue 空分析阶段（sibling 已多例）；列表序靠前的 S4/S6 仍为 ready[0] 先派，空阶段由 researcher 判读单元按先例关闭（superseded），不影响派发。
- 与兄弟课题 58b37930 的协调（EV-12）：改动点不同（对方=训练流程解耦，本课题=τ 参数化）、判据不同，不重复；基线实现独立（细胞选择差异见 D4），其 experiments/mr_moe.py 可作只读参考（门控/读出/数据管线模式），禁写对方文件；如后续要共享基线资产，由 researcher 白天推进会决定。
