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
        
        # 将动作掩码应用于分区logits
        if len(boundary_nodes) > 0:
            boundary_mask = action_mask[boundary_nodes]  # [num_boundary, num_partitions]
            partition_logits = partition_logits.masked_fill(~boundary_mask, float('-inf'))
        
        return node_logits, partition_logits


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
                 device: torch.device = None):
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
        """
        self.device = device or torch.device('cpu')
        self.num_partitions = num_partitions
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
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
        
        # 内存
        self.memory = PPOMemory()
        
        # 训练统计
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100)
        }
        
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
        
        # 获取动作掩码
        action_mask = torch.zeros(
            node_embeddings.shape[0], self.num_partitions,
            dtype=torch.bool, device=self.device
        )
        
        # 基于边界节点和当前分区设置有效动作
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

            # 检查输入数据的有效性
            if torch.isnan(node_embeddings).any():
                print(f"❌ 检测到NaN在node_embeddings中")
                return None, 0.0, value.item()
            if torch.isnan(region_embeddings).any():
                print(f"❌ 检测到NaN在region_embeddings中")
                return None, 0.0, value.item()
            if torch.isnan(node_logits).any():
                print(f"❌ 检测到NaN在node_logits中: {node_logits}")
                print(f"   node_embeddings范围: [{node_embeddings.min():.4f}, {node_embeddings.max():.4f}]")
                print(f"   region_embeddings范围: [{region_embeddings.min():.4f}, {region_embeddings.max():.4f}]")
                return None, 0.0, value.item()
            if torch.isnan(partition_logits).any():
                print(f"❌ 检测到NaN在partition_logits中")
                return None, 0.0, value.item()

            # 采样动作
            if training:
                # 从分布中采样
                node_probs = F.softmax(node_logits, dim=0)
                node_dist = torch.distributions.Categorical(node_probs)
                node_action = node_dist.sample()
                
                partition_probs = F.softmax(partition_logits[node_action], dim=0)
                partition_dist = torch.distributions.Categorical(partition_probs)
                partition_action = partition_dist.sample()
                
                # 计算对数概率
                log_prob = node_dist.log_prob(node_action) + partition_dist.log_prob(partition_action)
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
            
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
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
            
            node_probs = F.softmax(node_logits, dim=0)
            partition_probs = F.softmax(partition_logits[node_pos], dim=0)
            
            new_log_prob = (torch.log(node_probs[node_pos] + 1e-8) + 
                           torch.log(partition_probs[partition_idx - 1] + 1e-8))
            
            # PPO损失
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2)
            
            # 评论家损失
            critic_loss = F.mse_loss(value, return_val)
            
            # 熵
            entropy = -(node_probs * torch.log(node_probs + 1e-8)).sum()
            entropy += -(partition_probs * torch.log(partition_probs + 1e-8)).sum()
            
            # 总损失
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # 更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
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
