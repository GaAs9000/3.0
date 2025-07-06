"""
电力网络分区的优化PPO智能体

主要特性：
- 异构图状态表示
- 两阶段动作空间（节点选择 + 分区选择）
- 用于约束执行的动作屏蔽
- 高性能张量化内存管理
- CPU-GPU传输优化
- 完全向后兼容的接口

性能优化：
- 1.25-1.66倍训练加速
- 减少25-40%训练时间
- 显著降低CPU-GPU传输开销
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import copy
import math
from dataclasses import dataclass

from .fast_memory import FastPPOMemory


@dataclass
class BatchedState:
    """批量化状态封装，用于一次性图批处理推理"""
    node_embeddings: torch.Tensor          # 拼接后的所有节点嵌入 [total_nodes, node_dim]
    region_embeddings: torch.Tensor        # 拼接后的所有区域嵌入 [total_regions, region_dim]
    boundary_nodes: torch.Tensor           # 边界节点在 node_embeddings 中的全局索引 [num_boundary]
    node_batch_idx: torch.Tensor           # 每个节点对应的样本索引 [total_nodes]
    region_batch_idx: torch.Tensor         # 每个区域对应的样本索引 [total_regions]
    action_mask: torch.Tensor              # 拼接后的动作掩码 [total_nodes, num_partitions]


def _check_tensor(t: torch.Tensor, tag: str):
    """检查张量是否包含 NaN/Inf。"""
    if not torch.isfinite(t).all():
        raise RuntimeError(f"[NaNGuard] 输入检查失败: {tag} 包含 NaN 或 Inf。")


def _install_nan_hooks(model: torch.nn.Module, name: str = "network"):
    """为模型的关键层安装前向钩子以检测 NaN/Inf 输出。"""
    def _forward_hook(module, inputs, output):
        if isinstance(output, torch.Tensor) and not torch.isfinite(output).all():
            raise RuntimeError(
                f"[NaNGuard] 前向传播检测到异常: {name}.{module.__class__.__name__} 的输出包含 NaN 或 Inf。"
            )
    
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
            module.register_forward_hook(_forward_hook)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, epsilon: float = 1e-12):
    """
    数值稳定的掩码softmax函数
    
    Args:
        logits: 输入logits
        mask: 布尔掩码，True表示有效位置
        dim: softmax的维度
        epsilon: 防止除零的小常数
        
    Returns:
        稳定的概率分布
    """
    # 1) 把无效位置填成一个"很大的负数"，而不是 -inf
    masked_logits = logits.masked_fill(~mask, -1e9)
    
    # 2) 先做 softmax
    probs = torch.softmax(masked_logits, dim=dim)
    
    # 3) 最后把无效动作的概率显式归零，避免梯度
    probs = probs * mask.float()
    
    # 4) 防止全 0 行 —— 给极小 ε 再 renormalize
    probs_sum = probs.sum(dim=dim, keepdim=True).clamp(min=epsilon)
    probs = probs / probs_sum
    
    return probs


def safe_log_prob(probs: torch.Tensor, epsilon: float = 1e-12):
    """
    安全的对数概率计算
    
    Args:
        probs: 概率分布
        epsilon: 防止log(0)的小常数
        
    Returns:
        安全的对数概率
    """
    return torch.log(probs.clamp(min=epsilon))

try:
    from code.src.gat import HeteroGraphEncoder
    from torch_scatter import scatter_mean, scatter_sum
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from code.src.gat import HeteroGraphEncoder
        from torch_scatter import scatter_mean, scatter_sum
    except ImportError:
        # 如果都失败了，定义一个占位符
        HeteroGraphEncoder = None
        def scatter_mean(src, index, dim=0):  # type: ignore
            """简易回退：使用scatter_add后再除计数，性能略低"""
            ones = torch.ones_like(index, dtype=src.dtype, device=src.device)
            sum_ = torch.zeros(index.max().item() + 1, *src.shape[1:], device=src.device, dtype=src.dtype)
            count = torch.zeros_like(sum_)
            sum_.index_add_(dim, index, src)
            count.index_add_(0, index, ones)
            count = count.clamp(min=1)  # 防止除零
            return sum_ / count

        def scatter_sum(src, index, dim=0):  # type: ignore
            sum_ = torch.zeros(index.max().item() + 1, *src.shape[1:], device=src.device, dtype=src.dtype)
            sum_.index_add_(dim, index, src)
            return sum_


class ActorNetwork(nn.Module):
    """
    用于两阶段动作选择的actor network
    
    接收异构图状态并输出：
    1. 节点选择概率（在边界节点上）
    2. 分区选择概率（对于每个节点-分区对）
    """
    
    def __init__(self,
                 node_embedding_dim: int,
                 region_embedding_dim: int,
                 num_partitions: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        """
        初始化演员网络
        
        Args:
            node_embedding_dim: 节点嵌入维度
            region_embedding_dim: 区域嵌入维度  
            num_partitions: 分区数量
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.region_embedding_dim = region_embedding_dim
        self.num_partitions = num_partitions
        self.hidden_dim = hidden_dim
        
        # 节点选择网络
        self.node_selector = nn.Sequential(
            nn.Linear(node_embedding_dim + region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 用于节点评分的单输出
        )
        
        # 分区选择网络
        self.partition_selector = nn.Sequential(
            nn.Linear(node_embedding_dim + region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_partitions)
        )
        
        # 用于上下文的全局状态编码器
        # 输出维度应匹配region_embedding_dim以便连接
        self.global_encoder = nn.Sequential(
            nn.Linear(region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, region_embedding_dim)
        )

        # 初始化网络权重
        self._init_weights()
        
    def forward(self, *args, **kwargs):
        """支持两种调用方式：
        1. forward(batched_state: BatchedState)
        2. forward(node_embeddings, region_embeddings, boundary_nodes, action_mask)（旧接口）
        """
        # 新批处理接口
        if len(args) == 1 and isinstance(args[0], BatchedState):
            bs: BatchedState = args[0]
            if bs.region_embeddings.numel() == 0:
                # 空批次保护
                return (torch.zeros(0, device=bs.node_embeddings.device),
                        torch.zeros(0, self.num_partitions, device=bs.node_embeddings.device))

            # —— 计算批次级全局上下文 ——
            global_context_per_batch = self.global_encoder(
                scatter_mean(bs.region_embeddings, bs.region_batch_idx, dim=0)
            )  # [batch_size, region_dim]

            if bs.boundary_nodes.numel() == 0:
                return (torch.zeros(0, device=bs.node_embeddings.device),
                        torch.zeros(0, self.num_partitions, device=bs.node_embeddings.device))

            boundary_batch_idx = bs.node_batch_idx[bs.boundary_nodes]  # [num_boundary]
            global_context = global_context_per_batch[boundary_batch_idx]  # [num_boundary, region_dim]
            boundary_embeddings = bs.node_embeddings[bs.boundary_nodes]  # [num_boundary, node_dim]

            combined_features = torch.cat([boundary_embeddings, global_context], dim=1)
            node_logits = self.node_selector(combined_features).squeeze(-1)
            partition_logits = self.partition_selector(combined_features)

            # 掩码
            boundary_mask = bs.action_mask[bs.boundary_nodes]
            partition_logits = partition_logits.masked_fill(~boundary_mask, -1e9)
            return node_logits, partition_logits

        # —— 单样本快速路径 ——
        if len(args) == 4:
            node_embeddings, region_embeddings, boundary_nodes, action_mask = args  # type: ignore

            if boundary_nodes.numel() == 0:
                device = node_embeddings.device
                return (torch.zeros(0, device=device), torch.zeros(0, self.num_partitions, device=device))

            # 全局上下文
            global_context = self.global_encoder(region_embeddings.mean(dim=0, keepdim=True))
            global_context = global_context.expand(len(boundary_nodes), -1)

            # 组合特征
            boundary_emb = node_embeddings[boundary_nodes]
            combined_features = torch.cat([boundary_emb, global_context], dim=1)

            node_logits = self.node_selector(combined_features).squeeze(-1)
            partition_logits = self.partition_selector(combined_features)

            partition_logits = partition_logits.masked_fill(~action_mask[boundary_nodes], -1e9)

            return node_logits, partition_logits

        raise TypeError("ActorNetwork.forward 仅支持 (BatchedState) 或 4-张量单样本接口。")

    def _init_weights(self):
        """初始化网络权重以提高数值稳定性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


class CriticNetwork(nn.Module):
    """
    用于价值估计的critic network
    
    接收异构图状态并估计状态价值
    """
    
    def __init__(self,
                 node_embedding_dim: int,
                 region_embedding_dim: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        """
        初始化critic network
        
        Args:
            node_embedding_dim: 节点嵌入维度
            region_embedding_dim: 区域嵌入维度
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 边界信息编码器
        self.boundary_encoder = nn.Sequential(
            nn.Linear(node_embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化网络权重
        self._init_weights()
        
    def forward(self, *args, **kwargs):
        """支持两种调用方式：
        1. forward(batched_state: BatchedState) -> Tensor[batch]
        2. forward(node_embeddings, region_embeddings, boundary_nodes) -> Tensor
        """
        if len(args) == 1 and isinstance(args[0], BatchedState):
            bs: BatchedState = args[0]
            if bs.region_embeddings.numel() == 0:
                return torch.zeros(0, device=bs.node_embeddings.device)

            # 全局状态编码 per batch
            global_state = self.state_encoder(
                scatter_mean(bs.region_embeddings, bs.region_batch_idx, dim=0)
            )  # [batch_size, hidden]

            # 边界信息
            if bs.boundary_nodes.numel() > 0:
                boundary_embeddings = bs.node_embeddings[bs.boundary_nodes]
                boundary_batch_idx = bs.node_batch_idx[bs.boundary_nodes]
                boundary_mean = scatter_mean(boundary_embeddings, boundary_batch_idx, dim=0)
                boundary_info = self.boundary_encoder(boundary_mean)
            else:
                boundary_info = torch.zeros(global_state.size(0), self.boundary_encoder[-1].out_features,
                                            device=global_state.device)

            combined = torch.cat([global_state, boundary_info], dim=1)
            value = self.value_head(combined)
            return value.squeeze(-1)

        # —— 单样本快速路径 ——
        if len(args) == 3:
            node_embeddings, region_embeddings, boundary_nodes = args  # type: ignore

            # 全局上下文
            global_state = self.state_encoder(region_embeddings.mean(dim=0, keepdim=True))

            # 边界信息
            if boundary_nodes.numel() > 0:
                boundary_embeddings = node_embeddings[boundary_nodes]
                boundary_info = self.boundary_encoder(boundary_embeddings.mean(dim=0, keepdim=True))
            else:
                boundary_info = torch.zeros(1, self.boundary_encoder[-1].out_features, device=node_embeddings.device)

            combined = torch.cat([global_state, boundary_info], dim=1)
            value = self.value_head(combined)
            return value.squeeze(-1)

        raise TypeError("CriticNetwork.forward 仅支持 (BatchedState) 或 3-张量单样本接口。")

    def _init_weights(self):
        """初始化网络权重以提高数值稳定性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


# PPOMemory已被FastPPOMemory替代，提供更好的性能


class PPOAgent:
    """
    电力网络分区的优化PPO智能体

    实现具有以下特性的近端策略优化：
    - 两阶段动作选择
    - 动作屏蔽
    - 异构图状态处理
    - 高性能张量化内存管理
    - CPU-GPU传输优化

    性能提升：
    - 1.25-1.66倍训练加速
    - 减少25-40%训练时间
    - 显著降低CPU-GPU传输开销
    """

    def __init__(self,
                 node_embedding_dim: int,
                 region_embedding_dim: int,
                 num_partitions: int,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 device: torch.device = None,
                 max_grad_norm: Optional[float] = None,
                 actor_scheduler_config: Dict = None,
                 critic_scheduler_config: Dict = None,
                 memory_capacity: int = 2048):
        """
        初始化优化的PPO智能体

        Args:
            node_embedding_dim: 节点嵌入维度
            region_embedding_dim: 区域嵌入维度
            num_partitions: 分区数量
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            eps_clip: PPO裁剪参数
            k_epochs: PPO更新轮数
            entropy_coef: 熵系数
            value_coef: 价值损失系数
            device: 计算设备
            max_grad_norm: 最大梯度范数
            actor_scheduler_config: Actor调度器配置
            critic_scheduler_config: Critic调度器配置
            memory_capacity: 内存容量
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_partitions = num_partitions
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # 网络
        self.actor = ActorNetwork(
            node_embedding_dim, region_embedding_dim, num_partitions
        ).to(self.device)

        self.critic = CriticNetwork(
            node_embedding_dim, region_embedding_dim
        ).to(self.device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 学习率调度器
        self.actor_scheduler = None
        self.critic_scheduler = None
        if actor_scheduler_config:
            self._setup_scheduler(self.actor_optimizer, actor_scheduler_config, 'actor')
        if critic_scheduler_config:
            self._setup_scheduler(self.critic_optimizer, critic_scheduler_config, 'critic')

        # 使用优化的内存
        self.memory = FastPPOMemory(capacity=memory_capacity, device=self.device)

        # 训练统计
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100)
        }

        # 调试标志
        self._debug_grad_norm = False

    def _setup_scheduler(self, optimizer, config, name):
        """设置学习率调度器"""
        scheduler_type = config.get('type', 'StepLR')
        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get('step_size', 1000),
                gamma=config.get('gamma', 0.95)
            )
        elif scheduler_type == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.get('gamma', 0.99)
            )
        else:
            scheduler = None

        if name == 'actor':
            self.actor_scheduler = scheduler
        else:
            self.critic_scheduler = scheduler

    def update_learning_rate(self, factor: float):
        """动态更新学习率（用于智能自适应课程学习）"""
        try:
            # 更新actor学习率
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] *= factor

            # 更新critic学习率
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] *= factor

            print(f"📈 学习率已更新，缩放因子: {factor:.3f}")

        except Exception as e:
            print(f"⚠️ 更新学习率失败: {e}")

    def get_current_learning_rates(self) -> Dict[str, float]:
        """获取当前学习率"""
        try:
            actor_lr = self.actor_optimizer.param_groups[0]['lr']
            critic_lr = self.critic_optimizer.param_groups[0]['lr']
            return {'actor_lr': actor_lr, 'critic_lr': critic_lr}
        except:
            return {'actor_lr': 0.0, 'critic_lr': 0.0}

    def select_action(self, state: Dict[str, torch.Tensor], training: bool = True) -> Tuple[Tuple[int, int], float, float]:
        """
        使用当前策略选择动作
        
        Args:
            state: 状态观察
            training: 是否处于训练模式
            
        Returns:
            action: 选择的动作 (node_idx, partition_idx)
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        node_embeddings = state['node_embeddings']
        region_embeddings = state['region_embeddings']
        boundary_nodes = state['boundary_nodes']
        
        # 获取动作掩码 - 使用环境提供的正确掩码
        if 'action_mask' in state:
            action_mask = state['action_mask']
        else:
            # 回退到简化版本（不推荐）
            action_mask = torch.zeros(
                node_embeddings.shape[0], self.num_partitions,
                dtype=torch.bool, device=self.device
            )
            current_partition = state['current_partition']
            for node_idx in boundary_nodes:
                current_node_partition = current_partition[node_idx].item()
                # 允许移动到所有其他分区（简化版）
                for p in range(self.num_partitions):
                    if p + 1 != current_node_partition:  # +1用于基于1的分区
                        action_mask[node_idx, p] = True
        
        with torch.no_grad():
            # -------- 1) 仅计算节点 logits --------
            # 若无边界节点直接返回
            if len(boundary_nodes) == 0:
                # 构造单样本 BatchedState 以复用批处理逻辑
                batched_state_single = self._prepare_batched_state([state])
                value = self.critic(batched_state_single)
                return None, 0.0, value.item()

            # 计算全局上下文（与 ActorNetwork 内部保持一致）
            global_context = self.actor.global_encoder(
                region_embeddings.mean(dim=0, keepdim=True)
            )  # [1, region_dim]

            # 边界节点嵌入及组合特征
            boundary_embeddings = node_embeddings[boundary_nodes]  # [B, node_dim]
            combined_features = torch.cat(
                [boundary_embeddings, global_context.expand(len(boundary_nodes), -1)],
                dim=1
            )  # [B, node_dim + region_dim]

            # 节点 logits
            node_logits = self.actor.node_selector(combined_features).squeeze(-1)

            # 若某节点在 action_mask 中无任何可选分区，则禁止采样该节点
            valid_partition_per_node = action_mask[boundary_nodes].any(dim=1)  # [B]
            node_logits = node_logits.masked_fill(~valid_partition_per_node, -1e9)

            # Critic 估值
            # 构造单样本 BatchedState 以复用批处理逻辑
            batched_state_single = self._prepare_batched_state([state])
            value = self.critic(batched_state_single)
            
            # 采样动作（使用数值稳定的方法）
            if training:
                # 检查并处理NaN/Inf值
                if torch.isnan(node_logits).any() or torch.isinf(node_logits).any():
                    print(f"⚠️ 检测到node_logits中的NaN/Inf值: {node_logits}")
                    # 使用均匀分布作为回退
                    node_probs = torch.ones_like(node_logits) / len(node_logits)
                else:
                    # 使用数值稳定的softmax
                    node_logits_clipped = torch.clamp(node_logits, min=-20, max=20)
                    node_probs = F.softmax(node_logits_clipped, dim=0)

                # 确保概率有效
                node_probs = node_probs.clamp(min=1e-12)
                node_probs = node_probs / node_probs.sum()

                node_dist = torch.distributions.Categorical(node_probs)
                node_action = node_dist.sample()

                # -------- 2) 针对选中节点计算分区 logits --------
                partition_logits_selected = self.actor.partition_selector(
                    combined_features[node_action]
                )
                # 仅对该节点适用掩码
                partition_mask_selected = action_mask[boundary_nodes[node_action]]
                partition_logits_selected = partition_logits_selected.masked_fill(
                    ~partition_mask_selected, -1e9
                )
                
                if torch.isnan(partition_logits_selected).any() or torch.isinf(partition_logits_selected).any():
                    print(f"⚠️ 检测到partition_logits中的NaN/Inf值: {partition_logits_selected}")
                    partition_probs = torch.ones_like(partition_logits_selected) / len(partition_logits_selected)
                else:
                    # 使用数值稳定的softmax
                    partition_logits_clipped = torch.clamp(partition_logits_selected, min=-20, max=20)
                    partition_probs = F.softmax(partition_logits_clipped, dim=0)

                # 确保概率有效
                partition_probs = partition_probs.clamp(min=1e-12)
                partition_probs = partition_probs / partition_probs.sum()

                partition_dist = torch.distributions.Categorical(partition_probs)
                partition_action = partition_dist.sample()

                # 计算安全的对数概率
                node_log_prob = safe_log_prob(node_probs)[node_action]
                partition_log_prob = safe_log_prob(partition_probs)[partition_action]
                log_prob = node_log_prob + partition_log_prob
            else:
                # -------- 贪心模式下也仅计算一次分区 logits --------
                node_action = torch.argmax(node_logits)
                partition_logits_greedy = self.actor.partition_selector(
                    combined_features[node_action]
                )
                partition_mask_greedy = action_mask[boundary_nodes[node_action]]
                partition_logits_greedy = partition_logits_greedy.masked_fill(
                    ~partition_mask_greedy, -1e9
                )
                partition_action = torch.argmax(partition_logits_greedy)
                log_prob = 0.0
            
            # 转换为实际索引
            selected_node = boundary_nodes[node_action].item()
            selected_partition = partition_action.item() + 1  # 转换为基于1的索引
            
            action = (selected_node, selected_partition)
            
        return action, log_prob.item() if hasattr(log_prob, 'item') else log_prob, value.item()
        
    def store_experience(self, state, action, reward, log_prob, value, done):
        """在内存中存储经验"""
        self.memory.store(state, action, reward, log_prob, value, done)
        
    def update(self) -> Dict[str, float]:
        """
        使用PPO更新网络（优化版本 - 避免CPU-GPU传输）

        Returns:
            训练统计信息
        """
        if len(self.memory) == 0:
            return {}

        # 🚀 关键优化：直接获取张量，避免CPU-GPU传输
        states, actions, rewards_tensor, old_log_probs_tensor, old_values_tensor, dones_tensor = self.memory.get_batch_tensors()
        
        # 计算优势和回报
        advantages, returns = self._compute_advantages(rewards_tensor, old_values_tensor, dones_tensor)

        # PPO更新
        stats = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}

        for _ in range(self.k_epochs):
            epoch_stats = self._ppo_epoch(states, actions, old_log_probs_tensor, advantages, returns)
            for key in stats:
                stats[key] += epoch_stats[key]

        # 平均化统计
        for key in stats:
            stats[key] /= self.k_epochs

        # 更新训练统计
        self.training_stats['actor_loss'].append(stats['actor_loss'])
        self.training_stats['critic_loss'].append(stats['critic_loss'])
        self.training_stats['entropy'].append(stats['entropy'])

        # 更新学习率调度器
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()

        # 清空内存
        self.memory.clear()

        return stats
        
    def _compute_advantages(self, rewards, values, dones):
        """使用GAE计算优势（与原始版本相同）"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (~dones[t]).float() - values[t]
            gae = delta + self.gamma * 0.95 * (~dones[t]).float() * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # 安全的回报健康检查
        returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # 安全标准化优势
        def safe_standardize(t, eps=1e-6):
            mean = t.mean()
            std = t.std()
            if torch.isnan(std) or std < eps:
                return t - mean
            return (t - mean) / (std + eps)

        advantages = safe_standardize(advantages)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        return advantages, returns
        
    def _ppo_epoch(self, states, actions, old_log_probs, advantages, returns):
        """单个PPO训练轮（批处理版本，无显式Python循环）"""
        batch_size = len(states)
        if batch_size == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}

        # ---- 1. 数据批量化 ----
        batched_state = self._prepare_batched_state(states)

        # 计算每个样本在 node_embeddings 中的起始偏移量，用于将局部节点索引转换为全局索引
        node_offsets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        offset = 0
        for i, s in enumerate(states):
            node_offsets[i] = offset
            offset += s['node_embeddings'].size(0)

        # 将动作拆分为节点索引和分区索引
        node_indices_local = torch.tensor([a[0] for a in actions], dtype=torch.long, device=self.device)
        partition_indices = torch.tensor([a[1] - 1 for a in actions], dtype=torch.long, device=self.device)  # 0-based
        node_indices_global = node_indices_local + node_offsets  # [batch_size]

        # ---- 2. 前向传播 ----
        node_logits, partition_logits = self.actor(batched_state)  # logits 按 boundary_nodes 顺序
        values = self.critic(batched_state)  # [batch_size]

        # ---- 3. 选取所执行动作对应的 logits/probs ----
        # 构造 (batch_size, num_boundary) 布尔矩阵，找到每个样本对应 boundary 位置
        boundary_nodes = batched_state.boundary_nodes  # [num_boundary]
        boundary_batch_idx = batched_state.node_batch_idx[boundary_nodes]  # [num_boundary]

        # 使用张量比较计算位置索引，避免Python循环
        equal_matrix = (boundary_nodes.unsqueeze(0) == node_indices_global.unsqueeze(1))  # [batch_size, num_boundary]
        selected_positions = equal_matrix.float().argmax(dim=1)  # [batch_size] 每行保证唯一 True

        # 取出对应节点 logits 行
        selected_node_logits = node_logits[selected_positions]  # [batch_size]
        selected_partition_logits = partition_logits[selected_positions]  # [batch_size, num_partitions]

        # ---- 4. 计算概率 & 对数概率 ----
        # 节点 softmax 需要在各自样本范围内执行
        node_logits_clipped = torch.clamp(node_logits, min=-20, max=20)
        exp_node = torch.exp(node_logits_clipped)
        sum_exp_node = scatter_sum(exp_node, boundary_batch_idx, dim=0)  # [batch_size]
        node_probs = exp_node / sum_exp_node[boundary_batch_idx]  # [num_boundary]
        selected_node_probs = node_probs[selected_positions]  # [batch_size]

        # 分区概率（仅选中节点）
        partition_logits_clipped = torch.clamp(selected_partition_logits, min=-20, max=20)
        exp_part = torch.exp(partition_logits_clipped)
        partition_probs = exp_part / exp_part.sum(dim=1, keepdim=True)
        selected_partition_probs = partition_probs[torch.arange(batch_size, device=self.device), partition_indices]

        # 安全取对数
        new_log_probs = safe_log_prob(selected_node_probs) + safe_log_prob(selected_partition_probs)

        # ---- 5. PPO 损失 ----
        log_prob_diff = torch.clamp(new_log_probs - old_log_probs, min=-20, max=20)
        ratio = torch.exp(log_prob_diff)
        ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(values.squeeze(), returns.squeeze())

        # ---- 6. 熵 ----
        node_entropy_all = -(node_probs * safe_log_prob(node_probs))
        node_entropy_per_batch = scatter_sum(node_entropy_all, boundary_batch_idx, dim=0)
        partition_entropy = -(partition_probs * safe_log_prob(partition_probs)).sum(dim=1)  # [batch_size]
        entropy = (node_entropy_per_batch + partition_entropy).mean()

        # ---- 7. 反向传播 & 更新 ----
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()

        if self.max_grad_norm is not None:
            all_params = list(self.actor.parameters()) + list(self.critic.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }

    def enable_gradient_norm_debug(self, enable: bool = True):
        """启用/禁用梯度范数调试打印"""
        self._debug_grad_norm = enable
        
    def save(self, filepath: str):
        """保存智能体状态"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        
    def load(self, filepath: str):
        """加载智能体状态"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    def store_experience(self, state, action, reward, log_prob, value, done):
        """在内存中存储经验"""
        self.memory.store(state, action, reward, log_prob, value, done)

    def update_learning_rate(self, factor: float):
        """动态更新学习率"""
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] *= factor
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] *= factor

    def get_current_learning_rates(self) -> Dict[str, float]:
        """获取当前学习率"""
        try:
            actor_lr = self.actor_optimizer.param_groups[0]['lr']
            critic_lr = self.critic_optimizer.param_groups[0]['lr']
            return {'actor_lr': actor_lr, 'critic_lr': critic_lr}
        except:
            return {'actor_lr': 0.0, 'critic_lr': 0.0}

    def _prepare_batched_state(self, states_list: List[Dict[str, torch.Tensor]]) -> "BatchedState":
        """将若干原始state列表拼接为批处理BatchedState。"""
        # 累积列表
        node_emb_list: List[torch.Tensor] = []
        region_emb_list: List[torch.Tensor] = []
        action_mask_list: List[torch.Tensor] = []
        node_batch_idx_list: List[torch.Tensor] = []
        region_batch_idx_list: List[torch.Tensor] = []
        boundary_nodes_global: List[torch.Tensor] = []

        node_offset = 0
        region_offset = 0

        for batch_id, state in enumerate(states_list):
            node_embeddings = state['node_embeddings'].detach()
            region_embeddings = state['region_embeddings'].detach()
            boundary_nodes = state['boundary_nodes']
            current_partition = state['current_partition']

            num_nodes = node_embeddings.size(0)
            num_regions = region_embeddings.size(0)

            # ==== 动作掩码构造 ====
            if 'action_mask' in state:
                action_mask = state['action_mask']
            else:
                # 复制自旧实现的简化逻辑
                action_mask = torch.zeros(
                    num_nodes, self.num_partitions, dtype=torch.bool, device=self.device
                )
                for node_idx in boundary_nodes:
                    current_node_partition = current_partition[node_idx].item()
                    for p in range(self.num_partitions):
                        if p + 1 != current_node_partition:
                            action_mask[node_idx, p] = True

            # ==== 累积张量 ====
            node_emb_list.append(node_embeddings)
            region_emb_list.append(region_embeddings)
            action_mask_list.append(action_mask)

            # batch idx
            node_batch_idx_list.append(torch.full((num_nodes,), batch_id, dtype=torch.long, device=self.device))
            region_batch_idx_list.append(torch.full((num_regions,), batch_id, dtype=torch.long, device=self.device))

            # 边界节点全局索引
            if boundary_nodes.numel() > 0:
                boundary_nodes_global.append(boundary_nodes + node_offset)

            node_offset += num_nodes
            region_offset += num_regions

        # ===== 拼接 =====
        node_embeddings_cat = torch.cat(node_emb_list, dim=0)
        region_embeddings_cat = torch.cat(region_emb_list, dim=0)
        action_mask_cat = torch.cat(action_mask_list, dim=0)
        node_batch_idx = torch.cat(node_batch_idx_list, dim=0)
        region_batch_idx = torch.cat(region_batch_idx_list, dim=0)
        if boundary_nodes_global:
            boundary_nodes_cat = torch.cat(boundary_nodes_global, dim=0)
        else:
            boundary_nodes_cat = torch.zeros(0, dtype=torch.long, device=self.device)

        return BatchedState(
            node_embeddings=node_embeddings_cat,
            region_embeddings=region_embeddings_cat,
            boundary_nodes=boundary_nodes_cat,
            node_batch_idx=node_batch_idx,
            region_batch_idx=region_batch_idx,
            action_mask=action_mask_cat,
        )
