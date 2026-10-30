#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mr_moe.py — MR-MoE 冻结实现的算子核心（design.md D4 + D1 + D2 + D5 + D7）。

论文对应：arXiv:2606.12240v1 §2.4（Eq.8-11：K 个固定时间常数 LNN 专家 + softmax 门控
MLP + 每专家线性读出，无 attention 的基础 MR-MoE）；论文无官方代码（checklist #5
"Not yet"），本文件按 Eqs.1-18 自行实现。与论文的全部差异记入 results/baseline_diff.md。

阶段归属（workflow.json S2-gate-checks / S2b-data-model 拆分）：
  - 本文件在 S2-gate-checks（⓪算子先行 + ①五断言）创建，内容=τ-tie 算子核心：
    OrderedTauLTCCell（含 ⓪ 修复 R1/R2/R3）、TauSchedule（D2 保序参数化）、
    MRMoE（K 专家 + 门控 + 读出的 A/B 臂同构造）、参数审计助手。
  - S2b-data-model 负责在本文件上扩展（不改已冻结接口）：experiments/data_pipeline.py、
    单体 LNN 基线（D5(i)）、训练/评测循环与 smoke（D4 其余超参 Adam lr=1e-3 / batch 64 /
    E pilot=6 / full=20 已在下述冻结常量中）。

D4 冻结配置：
  - K=3 专家，每专家=OrderedTauLTCCell（ncps 0.0.2 LTCCell 子类，FullyConnected wiring，
    hidden 1500，ode_unfolds=6，implicit_param_constraints=True）；
  - 唯一覆写=cm_t 计算：cm_t = τ_k·softplus(gleak)/(elapsed_time/ode_unfolds)，
    即 softplus(cm) := τ_k·softplus(gleak)、τ = cm/gleak = τ_k（小时，elapsed_time=1.0
    即 dt=1h 网格），其余求解器逐行照抄库源码（ltc_cell.py:203-249）；
  - gleak 每神经元可学习（两臂相同），其余 LTC 参数默认初始化并可学习；
  - 门控 = softmax(MLP hidden 64)（论文「small MLP」未给尺寸，冻结 64；激活 Tanh 为
    冻结选择，与 sibling 58b37930 实现对齐，记 baseline_diff）；
  - 读出 = 每专家线性 C_k: R^1500→R（logit），y(t)=Σ_k π_k(t)·y_k(t)，BCEWithLogits；
  - elapsed_time=1.0，timespans=None（固定 dt=1h 网格；B>1 逐样本 timespans 路径本课题
    不用，探针 A4 记录其行为）。

D2 保序参数化（τ 注入的唯一可学习点）：
  τ_k = Σ_{j=1..k} softplus(c_j)，增量 Δ_j = softplus(c_j) > 0 → τ1<τ2<τ3 由构造严格
  保证。init Δ=(1,9,90) → τ=(1,10,100)，c_j = log(e^{Δ_j}−1) ≈ (0.5413, 9.0, 90.0)
  （float64 计算后转 float32；softplus 大参数区 ATen 走 x>20 线性支，无 exp 溢出）。
  臂 A：c 为 buffer（requires_grad=False）；臂 B：c 为 nn.Parameter。两臂除此以外
  逐位一致（D10 单变量纪律；探针 A2/A5 实测）。

⓪ 算子修复（fb-ab50d0ec9f/D12，先于①探针；逐条留证据于 results/operator_bench.md）：
  R1 把 loop-invariant 的 softplus(gleak) 提到 unfold 循环外（库源码在每次 unfold 内
     重复计算 6 次，ltc_cell.py:242）；
  R2 把常数 sparsity_mask 折进 w_param 每步算一次（库源码在每次 unfold 内对 (B,H,H)
     激活做 mask 乘，ltc_cell.py:234；FullyConnected wiring 下 mask 全 1，折叠为精确
     恒等变换）；
  R3 对每次 unfold 做梯度检查点（torch.utils.checkpoint, use_reentrant=False）：库实现
     的 72 步 BPTT 反传图在 hidden 1500 / batch 64 下需 >20GB（sibling 实测 OOM 证据），
     检查点把每步反传图压到 O(小)，反向时按 unfold 重算——数学不变（O5 实测前向逐位
     一致 max_abs_diff=0.0），峰值显存降到 8G 可跑。
  R4 FullyConnected wiring 构建缓存：库 FullyConnected(1500) 用 Python 双层循环建
     2.25M 条突触（实测 18.6s/个），且 erev/sparsity 由固定默认种子 1111 决定（结果
     确定）；缓存后同 (hidden, in_features) 复用同一 wiring 对象——细胞只在分配参数时
     读取 wiring（erev/adjacency 复制进各自参数），数值逐位不变，构模型开销降为 0。
  R5 专家序列级梯度检查点（MRMoE.forward）：三个专家的 72 步反传图不再同时驻留
     （否则全模型峰值≈3×单专家，B=64≈30GB OOM、B=16≈9.5GB 超 8G 预算）；反向按专家
     重算后全模型峰值≈单专家峰值（⓪ O3 实测 B=32=6863MiB / B=16=4772MiB，均 <8G，
     D9 阶梯 64→32 落在预算内）。数学不变（确定性算子重算，前向逐位一致）。
  修复前裸形态 = TauTieLTCCellRaw（仅 τ-tie 覆写、无 R1/R2/R3），供 ⓪ 前后对照。
  仍未解决（⓪ 记录、交 researcher 判读）：hidden 1500 稠密 LTC 的单步计算量
  （6 unfolds × (B,H,H) 元素级通路）在本机 8G 笔记本卡上实测 ~34ms/unfold(B=64)，
  全模型 72 步 fwd+bwd ≈ 84s/batch(B=32)，epoch 级耗时不达 D9 pilot 预算；R1-R5 只修
  冗余与显存，不降单步计算量。可选后续优化（未在本单元实施）：torch.compile 融合
  （预估 2-3×）、稀疏 wiring、降 hidden（D9 应急档）——均需 DECISIONS 拍板。
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import LTCCell
from ncps.wirings import FullyConnected
from torch.utils.checkpoint import checkpoint

# D4 冻结超参（与 design.md §四 D4 一致；改这里=改冻结设计，须走 DECISIONS）
K_EXPERTS = 3
HIDDEN = 1500
TAU_INIT: tuple[float, ...] = (1.0, 10.0, 100.0)  # D1 臂 A 固定 τ / 臂 B init（小时）
GATE_HIDDEN = 64
SEQ_LEN = 72          # D3：每条入院截断前 72h
BATCH_SIZE = 64       # D4（论文未给）
LR = 1e-3             # 论文 §3.3
ELAPSED_TIME = 1.0    # dt=1h 网格（D4）
ODE_UNFOLDS = 6       # D4（论文未给；D9：恒 6 不动）

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def softplus_inverse(delta: float) -> float:
    """softplus^{-1}：求 c 使 softplus(c)=delta。float64 计算后由调用方转 float32。"""
    delta = float(delta)
    if delta <= 0:
        raise ValueError(f"增量必须为正，得到 {delta}")
    if delta > 20:  # softplus 大参数线性支：c≈delta，避免 exp 溢出（float64 也无妨，求稳）
        return delta
    return math.log(math.expm1(delta))


class TauSchedule(nn.Module):
    """D2 保序 τ 参数化：τ_k = Σ_{j≤k} softplus(c_j)（累积正增量）。

    learnable=False → c 为 buffer（臂 A：固定 τ=(1,10,100)，requires_grad 不涉 τ）；
    learnable=True  → c 为 nn.Parameter（臂 B：A/B 可学习参数差恰 3 标量，探针 A5 实测）。
    两臂 c 的数值完全相同（同一 init 路径）→ init 处前向逐位一致（探针 A2）。
    """

    def __init__(self, tau_init: Sequence[float] = TAU_INIT, learnable: bool = False) -> None:
        super().__init__()
        tau_init = tuple(float(t) for t in tau_init)
        if len(tau_init) < 1 or any(b <= a for a, b in zip(tau_init, tau_init[1:])):
            raise ValueError(f"τ init 必须严格递增，得到 {tau_init}")
        deltas = [tau_init[0]] + [b - a for a, b in zip(tau_init, tau_init[1:])]
        c0 = torch.tensor([softplus_inverse(d) for d in deltas], dtype=torch.float32)
        if learnable:
            self.c = nn.Parameter(c0)
        else:
            self.register_buffer("c", c0)

    def tau(self) -> torch.Tensor:
        """τ 向量 (K,)；臂 B 带梯度（cumsum/softplus 保留计算图）。"""
        return torch.cumsum(F.softplus(self.c), dim=0)

    def tau_detached(self) -> list[float]:
        return [round(float(v), 6) for v in self.tau().detach().cpu().tolist()]

    def increments_detached(self) -> list[float]:
        return [round(float(v), 6) for v in F.softplus(self.c).detach().cpu().tolist()]


class TauTieLTCCellRaw(LTCCell):
    """τ-tie 覆写的裸形态（⓪ 修复前对照）：唯一改动=cm_t 计算，其余逐行照抄库源码。

    cm_t = τ_k · softplus(gleak) / (elapsed_time / ode_unfolds)
    与库 ltcell.py:223-225 的 cm_t = softplus(cm)/(elapsed_time/ode_unfolds) 同构，
    只是把 softplus(cm) 换成 τ_k·softplus(gleak)（τ=cm/gleak=τ_k 的语义由此保持）。
    无 R1/R2/R3：gleak 在每次 unfold 内重算、mask 在每次 unfold 内乘、无反传图压缩。
    """

    def forward(self, inputs, states, elapsed_time=1.0, tau=None):
        # 与库 LTCCell.forward 一致，仅多传 tau（库签名兼容的可选参数）
        inputs = self._map_inputs(inputs)
        next_state = self._ode_solver(inputs, states, elapsed_time, tau)
        outputs = self._map_outputs(next_state)
        return outputs, next_state

    def _ode_solver(self, inputs, state, elapsed_time, tau):
        if tau is None:
            raise ValueError("TauTieLTCCellRaw 需要显式 tau（专家由 MRMoE 注入）")
        v_pre = state

        # 感官预计算（loop-invariant，与库一致）
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = (
            sensory_w_activation * self._params["sensory_sparsity_mask"]
        )
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant —— τ-tie 覆写点（唯一与库不同的行）
        cm_t = tau * self.make_positive_fn(self._params["gleak"]) / (
            elapsed_time / self._ode_unfolds
        )

        w_param = self.make_positive_fn(self._params["w"])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )
            w_activation = w_activation * self._params["sparsity_mask"]
            rev_activation = w_activation * self._params["erev"]
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            gleak = self.make_positive_fn(self._params["gleak"])
            numerator = cm_t * v_pre + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre


class OrderedTauLTCCell(LTCCell):
    """τ-tie LTC 细胞（⓪ 修复后形态，D4 基线算子；唯一覆写=cm_t + R1/R2/R3）。

    数学与 TauTieLTCCellRaw / 库 LTCCell 完全一致（⓪ O5 实测前向逐位一致），
    修复只改内存占用与冗余计算，不改任何数值语义。
    """

    def __init__(self, *args, checkpoint_unfolds: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._checkpoint_unfolds = bool(checkpoint_unfolds)

    def forward(self, inputs, states, elapsed_time=1.0, tau=None):
        inputs = self._map_inputs(inputs)
        next_state = self._ode_solver(inputs, states, elapsed_time, tau)
        outputs = self._map_outputs(next_state)
        return outputs, next_state

    def _ode_solver(self, inputs, state, elapsed_time, tau):
        if tau is None:
            raise ValueError("OrderedTauLTCCell 需要显式 tau（专家由 MRMoE 注入）")
        v_pre = state

        # 感官预计算（loop-invariant，与库一致）
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = (
            sensory_w_activation * self._params["sensory_sparsity_mask"]
        )
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # R1: softplus(gleak) 是 loop-invariant，提到循环外（库在每次 unfold 内重算）
        gleak = self.make_positive_fn(self._params["gleak"])
        # τ-tie 覆写点（唯一与库不同的行）：softplus(cm) := τ_k·softplus(gleak)
        cm_t = tau * gleak / (elapsed_time / self._ode_unfolds)

        # R2: 常数 sparsity_mask 折进 w_param，每步一次（库在每次 unfold 内乘 (B,H,H)）
        w_param = self.make_positive_fn(self._params["w"]) * self._params["sparsity_mask"]

        def _unfold(v):
            w_activation = w_param * self._sigmoid(
                v, self._params["mu"], self._params["sigma"]
            )
            rev_activation = w_activation * self._params["erev"]
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            numerator = cm_t * v + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator
            return numerator / (denominator + self._epsilon)

        for _ in range(self._ode_unfolds):
            if self._checkpoint_unfolds:
                # R3: 每次 unfold 梯度检查点——反传时重算，(B,H,H) 中间图不再驻留
                v_pre = checkpoint(_unfold, v_pre, use_reentrant=False)
            else:
                v_pre = _unfold(v_pre)
        return v_pre


_WIRING_CACHE: dict[tuple[int, int], FullyConnected] = {}


def get_wiring(hidden: int, in_features: int) -> FullyConnected:
    """R4: FullyConnected wiring 构建缓存（同尺寸复用；库构建 1500 单位实测 18.6s）。

    FullyConnected 的 erev/sparsity 由固定默认种子 1111 生成（与新建实例逐位一致），
    细胞分配参数时即把 erev/adjacency 复制进各自 _params，之后不再读取 wiring——
    因此复用同一对象不改变任何数值，只省构建时间。

    注意（R4 修订）：返回的 wiring 已 build；调用方必须传 in_features=None 给
    LTCCell，否则 LTCCell.__init__ 会对共享 wiring 再调一次 build()——base Wiring.build
    重置 sensory_adjacency_matrix 后按已推进的 numpy RNG 重画，sensory_erev 逐专家
    漂移（实测 A/B init 等价被破坏 max_abs_diff=0.118）。预 build+None 后各细胞拿到的
    sensory_erev 逐位一致（与不缓存的新建 wiring 等价）。
    """
    key = (int(hidden), int(in_features))
    if key not in _WIRING_CACHE:
        wiring = FullyConnected(hidden)
        wiring.build(in_features)
        _WIRING_CACHE[key] = wiring
    return _WIRING_CACHE[key]


class OrderedTauLTCExpert(nn.Module):
    """单个 τ-tie LTC 专家：manual-loop 驱动 OrderedTauLTCCell（elapsed_time=1.0）。

    forward(x: (B,L,d), tau: 标量张量或 float) -> (outputs: (B,L,H), h_final: (B,H))
    outputs 为 _map_outputs 后的读出输入（motor=H；FullyConnected 下 output_dim=units）。
    """

    def __init__(self, input_size: int, hidden: int = HIDDEN,
                 ode_unfolds: int = ODE_UNFOLDS,
                 implicit_param_constraints: bool = True,
                 checkpoint_unfolds: bool = True,
                 cell_cls=OrderedTauLTCCell) -> None:
        super().__init__()
        wiring = get_wiring(hidden, input_size)  # R4 缓存（已 build）
        self.cell = cell_cls(
            wiring,
            in_features=None,  # 必须 None：LTCCell.__init__ 逢 in_features 非 None 会再调
            # wiring.build()——共享 wiring 被重复 build 时 numpy RNG 推进，sensory_erev
            # 逐专家漂移（R4 初版实测 A/B init 等价被破坏 0.118；预 build+None 修复）
            ode_unfolds=ode_unfolds,
            implicit_param_constraints=implicit_param_constraints,
            **({"checkpoint_unfolds": checkpoint_unfolds} if cell_cls is OrderedTauLTCCell else {}),
        )
        self.hidden = hidden

    def forward(self, x: torch.Tensor, tau) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq = x.size(0), x.size(1)
        h_state = torch.zeros((b, self.cell.state_size), device=x.device, dtype=x.dtype)
        outs = []
        for t in range(seq):
            h_out, h_state = self.cell(x[:, t], h_state, ELAPSED_TIME, tau=tau)
            outs.append(h_out)
        return torch.stack(outs, dim=1), h_state


class MRMoE(nn.Module):
    """K=3 τ-tie LTC 专家 + softmax 门控 MLP + 每专家线性读出的 MR-MoE（Eq.8-11）。

    tau_learnable=False → 臂 A（固定 τ=(1,10,100)，D1）；
    tau_learnable=True  → 臂 B（保序可学习 τ，D2）。两臂唯一差=tau_sched.c 是否 Parameter。
    forward(x: (B,L,d)) -> logits: (B,L)（未过 sigmoid；损失用 BCEWithLogits）
    """

    def __init__(self, input_size: int, k: int = K_EXPERTS, hidden: int = HIDDEN,
                 tau_init: Sequence[float] = TAU_INIT, tau_learnable: bool = False,
                 gate_hidden: int = GATE_HIDDEN, ode_unfolds: int = ODE_UNFOLDS,
                 checkpoint_unfolds: bool = True, cell_cls=OrderedTauLTCCell) -> None:
        super().__init__()
        if k != len(tau_init):
            raise ValueError("专家数必须与 τ 个数一致")
        self.k, self.hidden = k, hidden
        self.tau_init = tuple(float(t) for t in tau_init)
        self.tau_learnable = bool(tau_learnable)
        self.tau_sched = TauSchedule(self.tau_init, learnable=self.tau_learnable)
        self.experts = nn.ModuleList([
            OrderedTauLTCExpert(input_size, hidden, ode_unfolds,
                                checkpoint_unfolds=checkpoint_unfolds, cell_cls=cell_cls)
            for _ in range(k)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, k)
        )
        self.readouts = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(k)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        taus = self.tau_sched.tau()  # (K,)；臂 B 带梯度

        def _expert(k_i: int, expert: nn.Module, readout: nn.Module) -> torch.Tensor:
            out, _ = expert(x, taus[k_i])   # (B,L,H)
            return readout(out)             # (B,L,1)

        # R5: 专家序列级梯度检查点——三个专家的 72 步反传图不再同时驻留（否则全模型
        # 峰值≈3×单专家：B=64≈30GB OOM / B=16≈9.5GB 超 8G 预算）。反向时按专家重算，
        # 全模型峰值降到≈单专家峰值（⓪ O3 实测 B=32=6863MiB / B=16=4772MiB，均 <8G），
        # D9 batch 阶梯 64→32 即可落在预算内。数学不变（重算确定性算子，前向逐位一致）。
        expert_logits = []
        for k_i, (expert, readout) in enumerate(zip(self.experts, self.readouts)):
            expert_logits.append(
                checkpoint(_expert, k_i, expert, readout, use_reentrant=False))
        y = torch.cat(expert_logits, dim=-1)       # (B,L,K)
        pi = torch.softmax(self.gate(x), dim=-1)   # (B,L,K)
        return (y * pi).sum(dim=-1)                # (B,L)

    # ---------- D7 τ 日志助手（S3 训练循环每 epoch 调用） ----------
    def tau_report(self) -> dict:
        return {
            "tau": self.tau_sched.tau_detached(),
            "increments": self.tau_sched.increments_detached(),
            "tau_learnable": self.tau_learnable,
        }

    def c_grad_norm(self) -> float | None:
        """‖∂L/∂c‖₂（臂 B；backward 后调用）。臂 A 无 c 参数 → None。"""
        c = getattr(self.tau_sched, "c", None)
        if c is None or c.grad is None:
            return None
        return float(c.grad.detach().norm().item())


def build_model(input_size: int, seed: int, tau_learnable: bool = False,
                k: int = K_EXPERTS, hidden: int = HIDDEN) -> MRMoE:
    """按固定种子构建模型（D10 单变量纪律：两臂同初始化种子 → init 处逐位一致）。"""
    torch.manual_seed(seed)
    return MRMoE(input_size, k=k, hidden=hidden, tau_learnable=tau_learnable)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def param_audit(model_a: nn.Module, model_b: nn.Module) -> dict:
    """A/B 可学习参数审计（探针 A5 / E1 竞争解释区分）。

    返回可学习参数名集合的差集与计数；预期差集恰为 τ 参数化的 3 标量（臂 B 多 c）。
    """
    pa = {n: tuple(p.shape) for n, p in model_a.named_parameters() if p.requires_grad}
    pb = {n: tuple(p.shape) for n, p in model_b.named_parameters() if p.requires_grad}
    only_b = sorted(set(pb) - set(pa))
    only_a = sorted(set(pa) - set(pb))
    diff_elems = sum(math.prod(pb[n]) for n in only_b) - \
        sum(math.prod(pa[n]) for n in only_a)
    return {
        "arm_a_learnable_params": int(sum(math.prod(s) for s in pa.values())),
        "arm_b_learnable_params": int(sum(math.prod(s) for s in pb.values())),
        "learnable_elem_diff_b_minus_a": int(diff_elems),
        "names_only_in_b": only_b,
        "names_only_in_a": only_a,
        "shapes_only_in_b": {n: list(pb[n]) for n in only_b},
        "name_sets_identical_except": only_b + only_a,
        "pass_exactly_3_scalars": bool(
            diff_elems == 3 and only_b == ["tau_sched.c"] and not only_a
        ),
    }


# ===================== S2b-data-model 扩展（③模型 smoke） =====================
# 以下为 S2b 阶段新增（不改上述已冻结接口）：
#   - SingleLNN：单体 LNN 基线（D5(i)，论文 §3.2 "Monolithic LNN"）；
#   - SepsisDataset / train_steps / evaluate：D3 数据管线 + D8 评测；
#   - run_smoke：真实数据上的前向/反向 smoke（F3 预兆两档 lr、B(0)≡A(0)、参数审计）。

INPUT_CHANNELS_NOTE = (
    "模型输入 = 34 时变临床变量 + 34 mask 通道 = 68 维（D3-3，见 data_pipeline.py）"
)


class SingleLNN(nn.Module):
    """单体 LNN 基线（D5(i)；论文 §3.2 "Monolithic LNN"：无专家分解的单一 LNN）。

    用 ncps 0.0.2 库标准 LTCCell（非 τ-tie 子类——基线不涉专家/时间常数，D10 单变量
    纪律：这是论文式单体网络，不是 A/B 臂的一员，只作 D5(i) 方向性 sanity）。
    wiring/ode_unfolds/elapsed_time 与专家同配置；每时间步梯度检查点（内存修复，
    数学不变，证据同 R3 / results/operator_bench.md O5）。
    forward(x: (B,L,d)) -> logits: (B,L)
    """

    def __init__(self, input_size: int, hidden: int = HIDDEN,
                 ode_unfolds: int = ODE_UNFOLDS,
                 checkpoint_steps: bool = True) -> None:
        super().__init__()
        wiring = get_wiring(hidden, input_size)  # R4 缓存（数值逐位等价，见 get_wiring）
        self.cell = LTCCell(wiring, in_features=None, ode_unfolds=ode_unfolds)
        self.readout = nn.Linear(self.cell.state_size, 1)
        self.hidden = hidden
        self.checkpoint_steps = bool(checkpoint_steps)

    def _step(self, x_t: torch.Tensor, h: torch.Tensor):
        return self.cell(x_t, h, ELAPSED_TIME)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq = x.size(0), x.size(1)
        h = torch.zeros((b, self.cell.state_size), device=x.device, dtype=x.dtype)
        outs = []
        for t in range(seq):
            if self.checkpoint_steps:
                h_out, h = checkpoint(self._step, x[:, t], h, use_reentrant=False)
            else:
                h_out, h = self._step(x[:, t], h)
            outs.append(self.readout(h_out))   # (B,1)
        return torch.stack(outs, dim=1).squeeze(-1)  # (B,L)


# ---------------------------------------------------------------- 评测指标（D8）
def binary_auc(scores: "np.ndarray", labels: "np.ndarray") -> float:
    """AUROC（Mann-Whitney U，并列取平均秩）。"""
    import numpy as np
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos = pos.sum()
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * neg.sum()))


def binary_auprc(scores: "np.ndarray", labels: "np.ndarray") -> float:
    """AUPRC（阈值扫描 + 梯形积分）。"""
    import numpy as np
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    P = y.sum()
    if P == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    prec = tps / np.maximum(tps + fps, 1.0)
    rec = tps / P
    ss = s[order]
    keep = np.where(np.diff(ss) != 0)[0]
    keep = np.append(keep, len(s) - 1)
    prec, rec = prec[keep], rec[keep]
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[prec[0]], prec])
    return float(np.sum(np.diff(rec) * (prec[1:] + prec[:-1]) / 2))


class SepsisDataset(torch.utils.data.Dataset):
    """D3 预处理产物（data_pipeline.py 落盘的 .npz）的 Dataset 视图。

    __getitem__ 返回 (x: (L, 2F) = concat[X, M], y: (L,))。
    """

    def __init__(self, npz_path: str, split: str) -> None:
        import numpy as np
        z = np.load(npz_path, allow_pickle=True)
        idx = z[split]
        self.x = np.concatenate([z["X"][idx], z["M"][idx]], axis=-1).astype(np.float32)
        self.y = z["Y"][idx].astype(np.float32)
        self.y_adm = z["y_admission"][idx].astype(np.float32)
        self.ids = [str(z["ids"][i]) for i in idx]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def train_steps(model: nn.Module, loader, opt, device, max_steps: int = 2) -> dict:
    """最多 max_steps 步训练；返回 loss 轨迹/梯度范数/有限性（F3 预兆判据用）。"""
    model.train()
    losses, gnorms = [], []
    nan = False
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if not torch.isfinite(loss):
            nan = True
            losses.append(None)
            break
        loss.backward()
        g = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g += float(p.grad.detach().norm()) ** 2
        gnorms.append(g ** 0.5)
        opt.step()
        losses.append(float(loss))
        if bi + 1 >= max_steps:
            break
    return {"steps": len(losses), "losses": losses, "grad_norms": gnorms,
            "nan_or_inf": nan}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> dict:
    """D8 评测：小时级 + 入院级（窗口概率最大值）AUROC/AUPRC。"""
    import numpy as np
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
    if not probs:
        return {}
    p_h = np.concatenate([a.ravel() for a in probs])
    t_h = np.concatenate([a.ravel() for a in ys])
    p_a = np.concatenate([a.max(axis=1) for a in probs])
    t_a = np.concatenate([a.max(axis=1) for a in ys])
    return {
        "hourly": {"auroc": binary_auc(p_h, t_h), "auprc": binary_auprc(p_h, t_h),
                   "n": int(len(p_h)), "pos_rate": float(t_h.mean())},
        "admission": {"auroc": binary_auc(p_a, t_a), "auprc": binary_auprc(p_a, t_a),
                      "n": int(len(p_a)), "pos_rate": float(t_a.mean())},
    }


def run_smoke(data_npz: str, out_json: str, heavy: bool = True) -> dict:
    """S2b 模型 smoke（③）：真实数据上前向/反向 + F3 预兆两档 lr。

    heavy=True 时加跑 D4 冻结 hidden 1500 档（batch 按 D9 阶梯取 16，8G 预算内）；
    heavy=False 只跑迷你档（CPU 可跑，供快速回归）。
    """
    import numpy as np
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {
        "schema": "model-smoke/v1", "stage": "S2b-data-model",
        "purpose": "D4 冻结实现 + D3 管线在真实数据上的前向/反向 smoke；"
                   "F3 预兆（lr=1e-3 与 3e-4 两档 loss/梯度稳定性）",
        "device": str(device), "data": data_npz, "ran_at": datetime.now().isoformat(),
        "input_channels": INPUT_CHANNELS_NOTE,
    }

    ds = SepsisDataset(data_npz, "train")
    n_feat = ds.x.shape[-1]
    report["data_stats"] = {"n_admissions": len(ds), "input_dim": int(n_feat),
                            "seq_len": int(ds.x.shape[1]),
                            "hour_pos_rate": float(ds.y.mean()),
                            "admission_pos_rate": float(ds.y_adm.mean())}

    # ---------- 迷你档：A/B init 等价 + 参数审计 + 梯度到达 c + 单体 LNN ----------
    torch.manual_seed(0)
    ma = MRMoE(n_feat, k=3, hidden=64, tau_learnable=False)
    torch.manual_seed(0)
    mb = MRMoE(n_feat, k=3, hidden=64, tau_learnable=True)
    mini_loader = DataLoader(ds, batch_size=8, shuffle=True,
                             generator=torch.Generator().manual_seed(0))
    x0, y0 = next(iter(mini_loader))
    with torch.no_grad():
        dmax = float((ma(x0) - mb(x0)).abs().max())
    out_b = mb(x0)
    loss_b = F.binary_cross_entropy_with_logits(out_b, y0)
    loss_b.backward()
    c_grad_norm = mb.c_grad_norm()
    mini_lnn = SingleLNN(n_feat, hidden=64)
    out_l = mini_lnn(x0)
    loss_l = F.binary_cross_entropy_with_logits(out_l, y0)
    loss_l.backward()
    lnn_grads_finite = all(
        torch.isfinite(p.grad).all() for p in mini_lnn.parameters()
        if p.grad is not None)
    report["mini_config"] = {
        "hidden": 64, "batch": 8,
        "A0_vs_B0_max_abs_diff": dmax,
        "B0_equals_A0_pass": dmax < 1e-6,
        "param_audit": param_audit(ma, mb),
        "loss_b": float(loss_b), "c_grad_norm_arm_b": c_grad_norm,
        "grad_reaches_c_pass": c_grad_norm is not None and np.isfinite(c_grad_norm)
        and c_grad_norm > 0,
        "single_lnn_loss": float(loss_l), "single_lnn_grads_finite": bool(lnn_grads_finite),
        "single_lnn_out_shape": list(out_l.shape),
    }

    # ---------- D4 冻结档：hidden 1500 / seq 72 / batch 16（D9 阶梯，8G 内） ----------
    if heavy:
        if device.type != "cuda":
            report["heavy_config"] = {"skipped": "no cuda device"}
        else:
            B = 16
            loader = DataLoader(ds, batch_size=B, shuffle=True,
                                generator=torch.Generator().manual_seed(0))
            x, y = next(iter(loader))
            x, y = x.to(device), y.to(device)
            torch.manual_seed(0)
            mrmoe = MRMoE(n_feat, k=3, hidden=1500, tau_learnable=True).to(device)
            torch.cuda.reset_peak_memory_stats()
            logits = mrmoe(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            fwd_bwd = {
                "hidden": 1500, "batch": B, "seq_len": int(x.shape[1]),
                "loss": float(loss), "loss_finite": bool(torch.isfinite(loss)),
                "logits_max_abs": float(logits.abs().max()),
                "c_grad_norm": mrmoe.c_grad_norm(),
                "peak_mem_mib": round(peak, 1), "budget_mib": 8192.0,
                "within_8g": peak < 8192.0,
                "d9_note": "D4 冻结 batch=64 在 h1500 全模型超 8G（operator_bench O3），"
                           "smoke 按 D9 阶梯取 b16（实测峰值见上）",
            }
            # F3 预兆：两档 lr 各 2 步
            params_before_del = count_params(mrmoe)
            del mrmoe
            torch.cuda.empty_cache()
            f3 = {}
            for lr in (1e-3, 3e-4):
                torch.manual_seed(0)
                m = MRMoE(n_feat, k=3, hidden=1500, tau_learnable=True).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                f3[f"lr_{lr:g}"] = train_steps(m, loader, opt, device, max_steps=2)
            # 单体 LNN 基线同档前向/反向（D5(i)）
            torch.manual_seed(0)
            lnn = SingleLNN(n_feat, hidden=1500).to(device)
            torch.cuda.reset_peak_memory_stats()
            lo = lnn(x)
            ll = F.binary_cross_entropy_with_logits(lo, y)
            ll.backward()
            lnn_peak = torch.cuda.max_memory_allocated() / 2 ** 20
            lnn_rep = {
                "hidden": 1500, "batch": B, "loss": float(ll),
                "loss_finite": bool(torch.isfinite(ll)),
                "peak_mem_mib": round(lnn_peak, 1), "within_8g": lnn_peak < 8192.0,
                "grads_finite": all(torch.isfinite(p.grad).all()
                                    for p in lnn.parameters() if p.grad is not None),
            }
            report["heavy_config"] = {
                "mrmoe_fwd_bwd": fwd_bwd, "f3_presign": f3, "single_lnn": lnn_rep,
                "f3_triggered": any(v.get("nan_or_inf") for v in f3.values()),
                "f3_note": "F3 触发器=任一档出现 loss NaN/Inf 或梯度爆炸；"
                           "触发则记 F3 交 S3 前处置（两档均稳=预兆未触发）",
            }
            report["params"] = {"mrmoe_arm_b": params_before_del,
                                "single_lnn": count_params(lnn)}

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:4000])
    print(f"wrote {out_json}")
    return report


def _smoke() -> None:
    """最小前向/反向 smoke（迷你尺寸；真实尺寸基准见 experiments/operator_bench.py）。"""
    torch.manual_seed(0)
    d, b, seq, h = 8, 4, 6, 32
    model = MRMoE(d, k=3, hidden=h, tau_init=(1.0, 10.0, 100.0), tau_learnable=True)
    x = torch.randn(b, seq, d, requires_grad=True)
    logits = model(x)
    assert logits.shape == (b, seq), logits.shape
    assert torch.isfinite(logits).all(), "前向出现非有限值"
    target = torch.randint(0, 2, (b, seq)).float()
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    # τ-tie 路径下 cm 为结构性死参数（两臂同样不用；库 LTCCell 分配但本覆写不走它），
    # 梯度为 None 属预期；其余参数必须全 finite（A3/E1 证据）。
    dead_params = [n for n, p in model.named_parameters()
                   if p.requires_grad and n.endswith(".cm")]
    for name, p in model.named_parameters():
        if not p.requires_grad:  # sparsity_mask 等库内 requires_grad=False 参数
            continue
        if name in dead_params:
            assert p.grad is None, f"{name} 不应有梯度（τ-tie 下 cm 未被使用）"
            continue
        assert p.grad is not None, f"{name} 无梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"
    print(f"[smoke] mr_moe ok loss={loss.item():.4f} "
          f"params={count_params(model)} tau={model.tau_report()}")
    print(f"[smoke] dead params (τ-tie 下未用, 两臂相同): {dead_params}")
    print(f"[smoke] ||dL/dc||={model.c_grad_norm()}")
    # 臂 A/B init 等价（A2 的最小版）
    torch.manual_seed(0)
    ma = MRMoE(d, k=3, hidden=h, tau_learnable=False)
    torch.manual_seed(0)
    mb = MRMoE(d, k=3, hidden=h, tau_learnable=True)
    with torch.no_grad():
        dmax = float((ma(x) - mb(x)).abs().max().item())
    print(f"[smoke] A(0) vs B(0) max_abs_diff={dmax:.3e}")
    print(f"[smoke] param_audit pass={param_audit(ma, mb)['pass_exactly_3_scalars']}")


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="mr_moe: S2b smoke / mini self-test")
    ap.add_argument("--smoke", action="store_true",
                    help="真实数据 smoke（D4 冻结 h1500 档 + F3 两档 lr）")
    ap.add_argument("--mini", action="store_true", help="仅迷你自检（无数据依赖）")
    ap.add_argument("--data", default=os.path.join(
        PROJ, "data", "processed", "physionet2019_sample.npz"),
        help="data_pipeline.py 落盘的 .npz")
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "smoke_model.json"))
    ap.add_argument("--no-heavy", action="store_true", help="smoke 跳过 h1500 重档")
    args = ap.parse_args()
    if args.smoke:
        run_smoke(args.data, args.out, heavy=not args.no_heavy)
    else:
        _smoke()


if __name__ == "__main__":
    _cli()
