"""
电力网络分区的PPO智能体

- 异构图状态表示
- 两阶段动作空间（节点选择 + 分区选择）
- 用于约束执行的动作屏蔽
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import copy
import math  # 确保导入了 math 库


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
    from ..gat import HeteroGraphEncoder
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from gat import HeteroGraphEncoder
    except ImportError:
        # 如果都失败了，定义一个占位符
        HeteroGraphEncoder = None


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
        
    def forward(self, 
                node_embeddings: torch.Tensor,
                region_embeddings: torch.Tensor,
                boundary_nodes: torch.Tensor,
                action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        动作选择的前向传播
        
        Args:
            node_embeddings: 节点嵌入 [total_nodes, node_dim]
            region_embeddings: 区域嵌入 [num_partitions, region_dim]
            boundary_nodes: 边界节点索引 [num_boundary]
            action_mask: 动作掩码 [total_nodes, num_partitions]
            
        Returns:
            node_logits: 节点选择logits [num_boundary]
            partition_logits: 分区选择logits [num_boundary, num_partitions]
        """
        # 来自区域嵌入的全局上下文
        global_context = self.global_encoder(region_embeddings.mean(dim=0, keepdim=True))
        global_context = global_context.expand(len(boundary_nodes), -1)
        
        # 获取边界节点嵌入
        boundary_embeddings = node_embeddings[boundary_nodes]  # [num_boundary, node_dim]
        
        # 将节点嵌入与全局上下文结合
        combined_features = torch.cat([boundary_embeddings, global_context], dim=1)
        
        # 节点选择logits
        node_logits = self.node_selector(combined_features).squeeze(-1)  # [num_boundary]
        
        # 每个边界节点的分区选择logits
        partition_logits = self.partition_selector(combined_features)  # [num_boundary, num_partitions]
        
        # 将动作掩码应用于分区logits（使用数值稳定的方法）
        if len(boundary_nodes) > 0:
            boundary_mask = action_mask[boundary_nodes]  # [num_boundary, num_partitions]
            # 使用-1e9代替-inf，避免NaN/Inf问题
            partition_logits = partition_logits.masked_fill(~boundary_mask, -1e9)
        
        return node_logits, partition_logits

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
        
    def forward(self,
                node_embeddings: torch.Tensor,
                region_embeddings: torch.Tensor,
                boundary_nodes: torch.Tensor) -> torch.Tensor:
        """
        价值估计的前向传播
        
        Args:
            node_embeddings: 节点嵌入 [total_nodes, node_dim]
            region_embeddings: 区域嵌入 [num_partitions, region_dim]
            boundary_nodes: 边界节点索引 [num_boundary]
            
        Returns:
            state_value: 估计的状态价值 [1]
        """
        # 从区域嵌入编码全局状态
        global_state = self.state_encoder(region_embeddings.mean(dim=0, keepdim=True))
        
        # 编码边界信息
        if len(boundary_nodes) > 0:
            boundary_embeddings = node_embeddings[boundary_nodes]
            boundary_info = self.boundary_encoder(boundary_embeddings.mean(dim=0, keepdim=True))
        else:
            boundary_info = torch.zeros(1, self.boundary_encoder[-1].out_features, 
                                       device=node_embeddings.device)
        
        # 结合特征
        combined_features = torch.cat([global_state, boundary_info], dim=1)
        
        # 估计价值
        value = self.value_head(combined_features)
        
        return value.squeeze(-1)

    def _init_weights(self):
        """初始化网络权重以提高数值稳定性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


class PPOMemory:
    """
    PPO训练的内存缓冲区
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def store(self, state, action, reward, log_prob, value, done):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        """清除内存"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
    def get_batch(self):
        """获取经验批次"""
        return (self.states, self.actions, self.rewards, 
                self.log_probs, self.values, self.dones)


class PPOAgent:
    """
    电力网络分区的PPO智能体
    
    实现具有以下特性的近端策略优化：
    - 两阶段动作选择
    - 动作屏蔽
    - 异构图状态处理
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
                 max_grad_norm: float = None,
                 actor_scheduler_config: Dict = None,   # 【新增】
                 critic_scheduler_config: Dict = None): # 【新增】
        """
        初始化PPO智能体

        Args:
            node_embedding_dim: 节点嵌入维度
            region_embedding_dim: 区域嵌入维度
            num_partitions: 分区数量
            lr_actor: 演员学习率
            lr_critic: 评论家学习率
            gamma: 折扣因子
            eps_clip: PPO裁剪参数
            k_epochs: PPO训练轮数
            entropy_coef: 熵系数
            value_coef: 价值损失系数
            device: 计算设备
            max_grad_norm: 最大梯度范数（用于梯度裁剪）
            actor_scheduler_config: Actor学习率调度器配置
            critic_scheduler_config: Critic学习率调度器配置
        """
        self.device = device or torch.device('cpu')
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

        # 【修改】初始化独立的学习率调度器
        self.actor_scheduler = None
        self.critic_scheduler = None

        if actor_scheduler_config and actor_scheduler_config.get('enabled', False):
            print("✅ 启用 Actor 学习率调度器")
            self.actor_scheduler = self._create_scheduler(self.actor_optimizer, actor_scheduler_config)

        if critic_scheduler_config and critic_scheduler_config.get('enabled', False):
            print("✅ 启用 Critic 学习率调度器")
            self.critic_scheduler = self._create_scheduler(self.critic_optimizer, critic_scheduler_config)

        # 内存
        self.memory = PPOMemory()

        # 训练统计
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100)
        }

        # --- 新增：为 Actor 和 Critic 安装 NaN 检测钩子 ---
        _install_nan_hooks(self.actor, name="Actor")
        _install_nan_hooks(self.critic, name="Critic")

    # 【新增/修改】一个泛化的创建调度器的方法
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, config: Dict) -> torch.optim.lr_scheduler._LRScheduler:
        """根据配置为给定的优化器创建学习率调度器（线性预热 + 余弦退火）"""
        warmup_updates = config.get('warmup_updates', 0)
        total_updates = config.get('total_training_updates', 1000)

        print(f"   - 预热更新次数: {warmup_updates}, 总更新次数: {total_updates}")

        def lr_lambda(current_update: int):
            # 线性预热阶段
            if current_update < warmup_updates:
                return float(current_update) / float(max(1, warmup_updates))
            # 余弦退火阶段
            progress = float(current_update - warmup_updates) / float(max(1, total_updates - warmup_updates))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
            # 获取网络输出
            node_logits, partition_logits = self.actor(
                node_embeddings, region_embeddings, boundary_nodes, action_mask
            )
            
            value = self.critic(node_embeddings, region_embeddings, boundary_nodes)
            
            if len(boundary_nodes) == 0:
                # 没有有效动作
                return None, 0.0, value.item()
            
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

                # 对分区logits进行相同的处理
                partition_logits_selected = partition_logits[node_action]
                
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
                # 贪心选择
                node_action = torch.argmax(node_logits)
                partition_action = torch.argmax(partition_logits[node_action])
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
        使用PPO更新网络
        
        Returns:
            训练统计信息
        """
        if len(self.memory.states) == 0:
            return {}
        
        # --- 新增：在学习开始前检查输入和权重的健康状况 ---
        try:
            # 检查作为输入的旧状态
            old_states = [state for state in self.memory.states]
            if old_states:
                # 检查第一个状态的node_embeddings
                first_state = old_states[0]
                if 'node_embeddings' in first_state:
                    _check_tensor(first_state['node_embeddings'], "memory.states.node_embeddings (输入到网络)")
                if 'region_embeddings' in first_state:
                    _check_tensor(first_state['region_embeddings'], "memory.states.region_embeddings (输入到网络)")

            # 检查 Actor 和 Critic 的权重
            for name, param in self.actor.named_parameters():
                _check_tensor(param.data, f"Actor.{name} (权重)")
            for name, param in self.critic.named_parameters():
                _check_tensor(param.data, f"Critic.{name} (权重)")
        except RuntimeError as e:
            # 附加上下文信息后重新抛出异常
            print("❌ 在 PPO update() 的入口检查中发现 NaN/Inf。这表明问题在进入学习步骤之前就已存在。")
            raise e
        # --- 检查结束 ---
            
        # 获取批次
        states, actions, rewards, old_log_probs, old_values, dones = self.memory.get_batch()
        
        # 转换为张量
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(old_values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # 计算优势和回报
        advantages, returns = self._compute_advantages(rewards, old_values, dones)
        
        # PPO更新
        stats = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
        for _ in range(self.k_epochs):
            epoch_stats = self._ppo_epoch(states, actions, old_log_probs, advantages, returns)
            for key in stats:
                stats[key] += epoch_stats[key]
                
        # 对轮数求平均
        for key in stats:
            stats[key] /= self.k_epochs
            self.training_stats[key].append(stats[key])

        # 【修改】在每次更新后独立推进学习率调度器
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()

        # 清除内存
        self.memory.clear()

        return stats
        
    def _compute_advantages(self, rewards, values, dones):
        """使用GAE计算优势"""
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

        # 🔧 修复3: 安全的回报健康检查
        returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(returns).any():
            raise RuntimeError("NaN in returns – check reward pipeline")

        # 🔧 修复1: 安全标准化优势
        def safe_standardize(t, eps=1e-6):
            mean = t.mean()
            std = t.std()
            if torch.isnan(std) or std < eps:
                # 如果方差过小或已损坏，只做去均值
                return t - mean
            return (t - mean) / (std + eps)

        advantages = safe_standardize(advantages)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        return advantages, returns
        
    def _ppo_epoch(self, states, actions, old_log_probs, advantages, returns):
        """单个PPO训练轮"""
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            old_log_prob = old_log_probs[i]
            advantage = advantages[i]
            return_val = returns[i]
            
            # 获取当前策略输出
            node_embeddings = state['node_embeddings']
            region_embeddings = state['region_embeddings']
            boundary_nodes = state['boundary_nodes']
            
            if len(boundary_nodes) == 0:
                continue
                
            # 动作掩码
            action_mask = torch.zeros(
                node_embeddings.shape[0], self.num_partitions,
                dtype=torch.bool, device=self.device
            )
            current_partition = state['current_partition']
            for node_idx in boundary_nodes:
                current_node_partition = current_partition[node_idx].item()
                for p in range(self.num_partitions):
                    if p + 1 != current_node_partition:
                        action_mask[node_idx, p] = True
            
            node_logits, partition_logits = self.actor(
                node_embeddings, region_embeddings, boundary_nodes, action_mask
            )
            
            value = self.critic(node_embeddings, region_embeddings, boundary_nodes)
            
            # 计算新的对数概率
            node_idx, partition_idx = action
            node_pos = (boundary_nodes == node_idx).nonzero(as_tuple=True)[0]
            
            if len(node_pos) == 0:
                continue
                
            node_pos = node_pos[0]
            
            # 添加数值稳定性检查
            if torch.isnan(node_logits).any() or torch.isinf(node_logits).any():
                print(f"⚠️ PPO更新中检测到node_logits的NaN/Inf值")
                node_logits_clipped = torch.zeros_like(node_logits)
            else:
                node_logits_clipped = torch.clamp(node_logits, min=-20, max=20)

            partition_logits_selected = partition_logits[node_pos]
            if torch.isnan(partition_logits_selected).any() or torch.isinf(partition_logits_selected).any():
                print(f"⚠️ PPO更新中检测到partition_logits的NaN/Inf值")
                partition_logits_clipped = torch.zeros_like(partition_logits_selected)
            else:
                partition_logits_clipped = torch.clamp(partition_logits_selected, min=-20, max=20)

            # 使用数值稳定的概率计算
            node_probs = F.softmax(node_logits_clipped, dim=0).clamp(min=1e-12)
            partition_probs = F.softmax(partition_logits_clipped, dim=0).clamp(min=1e-12)
            
            # 安全的对数概率计算
            node_log_prob = safe_log_prob(node_probs)[node_pos]
            partition_log_prob = safe_log_prob(partition_probs)[partition_idx - 1]
            new_log_prob = node_log_prob + partition_log_prob
            
            # 🔧 修复2: ratio = exp(logπ_new – logπ_old) 双重保护
            log_prob_diff = torch.clamp(new_log_prob - old_log_prob, min=-20, max=20)
            ratio = torch.exp(log_prob_diff)
            # 防守式替换所有异常
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2)
            
            # 评论家损失
            critic_loss = F.mse_loss(value, return_val)
            
            # 熵计算（数值稳定）
            entropy = -(node_probs * safe_log_prob(node_probs)).sum()
            entropy += -(partition_probs * safe_log_prob(partition_probs)).sum()
            
            # 总损失
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # 更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            # 🔧 修复4: 梯度裁剪一次性覆盖全部可训练参数
            if self.max_grad_norm is not None:
                all_params = list(self.actor.parameters()) + list(self.critic.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                # 可选：打印梯度范数用于调试
                if hasattr(self, '_debug_grad_norm') and self._debug_grad_norm:
                    print(f"📊 梯度范数: {grad_norm:.4f}")
            else:
                # 即使不裁剪，也计算梯度范数用于监控
                all_params = list(self.actor.parameters()) + list(self.critic.parameters())
                total_norm = 0
                for p in all_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** (1. / 2)
                if hasattr(self, '_debug_grad_norm') and self._debug_grad_norm:
                    print(f"📊 梯度范数(未裁剪): {grad_norm:.4f}")

            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            
        return {
            'actor_loss': total_actor_loss / len(states),
            'critic_loss': total_critic_loss / len(states),
            'entropy': total_entropy / len(states)
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
