import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import os
from collections import defaultdict

# Import types that will be defined in other modules
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from metrics import PowerGridPartitionEnv

class PPOMemory:
    """
    PPO经验存储器
    
    特点：
    1. 高效的批量采样
    2. 支持GAE计算
    3. 内存友好的实现
    """
    
    def __init__(self):
        self.clear()
    
    def store(self, state: Dict, action: Tuple[int, int], action_idx: int,
             log_prob: torch.Tensor, value: torch.Tensor, reward: float,
             done: bool, valid_actions: List[Tuple[int, int]]):
        """存储一步经验"""
        # 只存储必要的状态信息（避免存储大的嵌入矩阵）
        stored_state = {
            'z': state['z'].clone(),
            'boundary_nodes': state['boundary_nodes'].clone(),
            'region_embeddings': state['region_embeddings'].clone(),
            'global_context': state['global_context'].clone(),
            't': state['t']
        }
        
        self.states.append(stored_state)
        self.actions.append(action)
        self.action_indices.append(action_idx)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.dones.append(torch.tensor(done, dtype=torch.float32))
        self.valid_actions_list.append(valid_actions.copy())
    
    def get_batch(self, batch_size: int) -> Dict:
        """获取随机批次用于训练"""
        n = len(self.rewards)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            batch = {
                'states': [self.states[i] for i in batch_indices],
                'actions': [self.actions[i] for i in batch_indices],
                'action_indices': [self.action_indices[i] for i in batch_indices],
                'log_probs': torch.stack([self.log_probs[i] for i in batch_indices]),
                'values': torch.stack([self.values[i] for i in batch_indices]),
                'rewards': torch.stack([self.rewards[i] for i in batch_indices]),
                'dones': torch.stack([self.dones[i] for i in batch_indices]),
                'valid_actions': [self.valid_actions_list[i] for i in batch_indices],
                'advantages': torch.stack([self.advantages[i] for i in batch_indices]),
                'returns': torch.stack([self.returns[i] for i in batch_indices])
            }
            
            yield batch
    
    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95, 
                   next_value: torch.Tensor = None):
        """计算广义优势估计(GAE)和回报"""
        values = torch.stack(self.values)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        
        # 计算GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value if next_value is not None else 0
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        
        # 计算回报
        returns = advantages + values
        
        # 标准化优势（提高训练稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = list(advantages)
        self.returns = list(returns)
    
    def clear(self):
        """清空存储器"""
        self.states = []
        self.actions = []
        self.action_indices = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.valid_actions_list = []
        self.advantages = []
        self.returns = []
    
    def __len__(self):
        return len(self.rewards)


class HierarchicalActor(nn.Module):
    """
    分层Actor网络
    
    两阶段决策：
    1. 选择要分配的节点
    2. 选择目标区域
    
    这种设计可以大幅减少动作空间
    """
    
    def __init__(self, node_dim: int, region_dim: int, context_dim: int,
                 hidden_dim: int = 256, num_layers: int = 3, K: int = 3):
        super().__init__()
        
        self.K = K
        
        # 共享特征提取器
        self.shared_net = self._build_mlp(
            node_dim + region_dim * K + context_dim,
            hidden_dim,
            hidden_dim,
            num_layers - 1
        )
        
        # 动作评分头
        self.action_head = nn.Linear(hidden_dim, 1)
        
        # 正交初始化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        # 输出层使用较小的初始化
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
    
    def _build_mlp(self, input_dim: int, hidden_dim: int, output_dim: int,
                   num_layers: int) -> nn.Module:
        """构建多层感知机"""
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
        
        return nn.Sequential(*layers)
    
    def forward(self, node_embeddings: torch.Tensor, state: Dict,
               valid_actions: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        返回:
            action_probs: 动作概率分布
            action_logits: 原始分数（用于计算熵）
        """
        if len(valid_actions) == 0:
            return torch.empty(0), torch.empty(0)
        
        # 批量计算所有有效动作的特征
        action_features = []
        
        for node_idx, region in valid_actions:
            # 构造动作特征
            node_emb = node_embeddings[node_idx]
            region_emb = state['region_embeddings'][region - 1]
            global_ctx = state['global_context']
            
            # 添加额外的关系特征
            all_region_embs = state['region_embeddings'].flatten()
            
            # 拼接所有特征
            features = torch.cat([
                node_emb,
                all_region_embs,
                global_ctx
            ])
            
            action_features.append(features)
        
        # 批量前向传播
        action_features = torch.stack(action_features)
        hidden = self.shared_net(action_features)
        action_logits = self.action_head(hidden).squeeze(-1)
        
        # 计算概率分布
        action_probs = F.softmax(action_logits, dim=0)
        
        return action_probs, action_logits


class Critic(nn.Module):
    """
    价值网络
    
    估计状态价值函数V(s)
    """
    
    def __init__(self, region_dim: int, context_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, K: int = 3):
        super().__init__()
        
        # 输入维度：区域嵌入 + 全局上下文 + 统计信息
        input_dim = region_dim * K + context_dim + K * 3  # 每个区域3个统计量
        
        # 价值网络
        self.value_net = self._build_mlp(input_dim, hidden_dim, 1, num_layers)
        
        # 正交初始化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
    
    def _build_mlp(self, input_dim: int, hidden_dim: int, output_dim: int,
                   num_layers: int) -> nn.Module:
        """构建多层感知机"""
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.extend([
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
        
        return nn.Sequential(*layers)
    
    def forward(self, state: Dict, env: 'PowerGridPartitionEnv') -> torch.Tensor:
        """
        前向传播
        
        参数:
            state: 状态字典
            env: 环境实例（用于获取额外统计信息）
        """
        # 提取特征
        region_embs = state['region_embeddings'].flatten()
        global_ctx = state['global_context']
        
        # 计算区域统计信息
        region_stats = []
        for k in range(1, env.K + 1):
            mask = (state['z'] == k)
            if mask.any():
                # 节点数比例
                size_ratio = mask.float().mean()
                # 负荷比例
                load_ratio = env.Pd[mask].sum() / (env.Pd.sum() + 1e-10)
                # 发电比例
                gen_ratio = env.Pg[mask].sum() / (env.Pg.sum() + 1e-10)
                
                region_stats.extend([size_ratio, load_ratio, gen_ratio])
            else:
                region_stats.extend([0.0, 0.0, 0.0])
        
        region_stats = torch.tensor(region_stats, device=state['z'].device)
        
        # 组合特征
        features = torch.cat([region_embs, global_ctx, region_stats])
        
        # 计算价值
        value = self.value_net(features)
        
        return value


class PPOAgent:
    """
    PPO智能体（完整实现）
    
    特点：
    1. Clipped surrogate objective
    2. 广义优势估计(GAE)
    3. 多epoch mini-batch更新
    4. 自适应KL惩罚
    5. 熵正则化
    6. 梯度裁剪
    """
    
    def __init__(self, actor: nn.Module, critic: nn.Module, env: 'PowerGridPartitionEnv',
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, lam: float = 0.95, eps_clip: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5, target_kl: float = 0.01,
                 device: str = 'cpu'):
        
        self.device = torch.device(device)
        self.env = env
        
        # 网络
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        
        # 优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 学习率调度器
        self.scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_actor, mode='max', factor=0.5, patience=10
        )
        self.scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_critic, mode='max', factor=0.5, patience=10
        )
        
        # 超参数
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # 经验存储
        self.memory = PPOMemory()
        
        # 训练统计
        self.training_stats = defaultdict(list)
    
    def select_action(self, state: Dict, valid_actions: List[Tuple[int, int]],
                     training: bool = True) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        选择动作
        
        返回:
            (action, value): 动作和状态价值
        """
        if len(valid_actions) == 0:
            return None
        
        # 获取动作概率分布
        with torch.set_grad_enabled(training):
            action_probs, action_logits = self.actor(
                self.env.embeddings, state, valid_actions
            )
            
            # 计算状态价值
            value = self.critic(state, self.env)
        
        if training:
            # 训练时：从概率分布采样
            dist = Categorical(action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            
            # 存储到经验池
            selected_action = valid_actions[action_idx]
            self.memory.store(
                state, selected_action, action_idx.item(),
                log_prob, value.squeeze(), 0, False, valid_actions
            )
        else:
            # 评估时：选择概率最大的动作
            action_idx = torch.argmax(action_probs)
            selected_action = valid_actions[action_idx]
        
        return selected_action, value.item()
    
    def update_last_reward(self, reward: float, done: bool):
        """更新最后一步的奖励和完成标志"""
        if len(self.memory) > 0:
            self.memory.rewards[-1] = torch.tensor(reward, dtype=torch.float32)
            self.memory.dones[-1] = torch.tensor(done, dtype=torch.float32)
    
    def update(self, epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """
        PPO更新
        
        参数:
            epochs: 更新轮数
            batch_size: 批量大小
            
        返回:
            统计信息字典
        """
        if len(self.memory) == 0:
            return {}
        
        # 计算GAE和回报
        with torch.no_grad():
            last_state = self.env.get_state()
            last_value = self.critic(last_state, self.env).squeeze()
        
        self.memory.compute_gae(self.gamma, self.lam, last_value)
        
        # 训练统计
        epoch_stats = defaultdict(list)
        
        # 多轮更新
        for epoch in range(epochs):
            kl_divs = []
            
            # Mini-batch训练
            for batch in self.memory.get_batch(batch_size):
                # 准备批量数据
                old_log_probs = batch['log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)
                
                # 重新计算动作概率（用于比率计算）
                new_log_probs = []
                entropies = []
                
                for i, (state, valid_actions, action_idx) in enumerate(
                    zip(batch['states'], batch['valid_actions'], batch['action_indices'])
                ):
                    # 恢复完整状态（添加嵌入）
                    full_state = {
                        **state,
                        'node_embeddings': self.env.embeddings
                    }
                    
                    # 计算新的动作概率
                    action_probs, _ = self.actor(self.env.embeddings, full_state, valid_actions)
                    
                    if len(action_probs) > 0 and action_idx < len(action_probs):
                        dist = Categorical(action_probs)
                        new_log_probs.append(dist.log_prob(torch.tensor(action_idx)))
                        entropies.append(dist.entropy())
                
                if not new_log_probs:
                    continue
                
                new_log_probs = torch.stack(new_log_probs)
                entropy = torch.stack(entropies).mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - old_log_probs[:len(new_log_probs)])
                
                # KL散度（用于早停）
                kl_div = (old_log_probs[:len(new_log_probs)] - new_log_probs).mean()
                kl_divs.append(kl_div.item())
                
                # Clipped surrogate loss
                surr1 = ratio * advantages[:len(new_log_probs)]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[:len(new_log_probs)]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失 - 重新计算以避免计算图问题
                values = []
                for state in batch['states'][:len(new_log_probs)]:
                    full_state = {
                        **state,
                        'node_embeddings': self.env.embeddings.detach()
                    }
                    value = self.critic(full_state, self.env)
                    values.append(value)
                
                values = torch.cat(values)
                value_loss = F.mse_loss(values, returns[:len(values)].detach())
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                
                # 记录统计（在反向传播后立即提取数值，避免计算图问题）
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                entropy_val = entropy.item()
                kl_div_val = kl_div.item()
                
                epoch_stats['policy_loss'].append(policy_loss_val)
                epoch_stats['value_loss'].append(value_loss_val)
                epoch_stats['entropy'].append(entropy_val)
                epoch_stats['kl_div'].append(kl_div_val)
                
                # 清理计算图，避免重复反向传播
                del loss, policy_loss, value_loss, entropy, ratio, surr1, surr2, new_log_probs, values
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 早停检查
            if kl_divs and np.mean(kl_divs) > self.target_kl:
                print(f"Early stopping at epoch {epoch} due to KL divergence")
                break
        
        # 清空经验池
        self.memory.clear()
        
        # 汇总统计
        stats = {}
        for key, values in epoch_stats.items():
            if values:
                stats[key] = np.mean(values)
                self.training_stats[key].append(stats[key])
        
        return stats
    
    def state_dict(self) -> Dict:
        """返回智能体的状态字典（用于检查点保存）"""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'training_stats': dict(self.training_stats)
        }
    
    def load_state_dict(self, state_dict: Dict):
        """从状态字典加载智能体状态"""
        self.actor.load_state_dict(state_dict['actor_state_dict'])
        self.critic.load_state_dict(state_dict['critic_state_dict'])
        self.optimizer_actor.load_state_dict(state_dict['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(state_dict['optimizer_critic_state_dict'])
        self.training_stats = defaultdict(list, state_dict.get('training_stats', {}))
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'training_stats': dict(self.training_stats)
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.training_stats = defaultdict(list, checkpoint.get('training_stats', {}))
        print(f"📂 Model loaded from {path}")


def initialize_ppo_agent(embeddings, env, device):
    """Test function for PPO agent initialization"""
    # 测试PPO智能体
    print("\n🤖 初始化PPO智能体...")

    # 创建网络
    actor = HierarchicalActor(
        node_dim=embeddings.shape[1],
        region_dim=embeddings.shape[1],
        context_dim=embeddings.shape[1] + 1 + env.K + 1,
        hidden_dim=256,
        K=env.K
    )

    critic = Critic(
        region_dim=embeddings.shape[1],
        context_dim=embeddings.shape[1] + 1 + env.K + 1,
        hidden_dim=256,
        K=env.K
    )

    # 创建PPO智能体
    agent = PPOAgent(
        actor=actor,
        critic=critic,
        env=env,
        device=device
    )

    print(f"✅ PPO智能体初始化成功！")
    print(f"📊 Actor参数量: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"📊 Critic参数量: {sum(p.numel() for p in critic.parameters()):,}")
    
    return agent

def evaluate_agent(agent: PPOAgent, env: 'PowerGridPartitionEnv', 
                   return_env_state: bool = False) -> Any:
    """
    评估智能体在一个完整回合中的表现
    
    参数:
        agent: 要评估的智能体
        env: 评估环境
        return_env_state: 是否返回最终的环境状态
    
    返回:
        最终指标，如果return_env_state为True，则额外返回环境状态
    """
    agent.actor.eval()
    agent.critic.eval()
    
    state = env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            action_result = agent.select_action(state, valid_actions, training=False)
            if action_result is None:
                break
                
            action, _ = action_result
            state, _, done, _ = env.step(action)
    
    # 最终指标
    final_metrics = env.current_metrics
    
    if return_env_state:
        import copy
        return final_metrics, copy.deepcopy(env)
    
    return final_metrics

