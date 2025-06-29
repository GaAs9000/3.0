#!/usr/bin/env python3
"""
PPO训练器：完整的PPO算法实现
用于增强奖励系统的真实训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import logging

class PPOActor(nn.Module):
    """PPO Actor网络"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        """前向传播"""
        logits = self.network(state)
        return F.softmax(logits, dim=-1)
    
    def get_action_and_log_prob(self, state):
        """获取动作和对数概率"""
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()

class PPOCritic(nn.Module):
    """PPO Critic网络"""
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """前向传播"""
        return self.network(state)

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 hidden_dim: int = 256,
                 device: torch.device = None):
        """
        初始化PPO训练器
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            eps_clip: PPO裁剪参数
            k_epochs: 每次更新的训练轮数
            entropy_coef: 熵正则化系数
            value_coef: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = device or torch.device('cpu')
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建网络
        self.actor = PPOActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = PPOCritic(state_dim, hidden_dim).to(self.device)
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验缓冲区
        self.buffer = PPOBuffer()
        
        # 训练统计
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'explained_variance': [],
            'approx_kl': []
        }
        
    def get_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """
        获取动作
        
        Args:
            state: 状态张量
            
        Returns:
            action: 动作
            log_prob: 对数概率
            value: 状态价值
        """
        with torch.no_grad():
            state = state.to(self.device)
            action, log_prob, _ = self.actor.get_action_and_log_prob(state)
            value = self.critic(state)
            
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, 
                        state: torch.Tensor,
                        action: int,
                        reward: float,
                        next_state: torch.Tensor,
                        done: bool,
                        log_prob: float,
                        value: float):
        """存储转换"""
        self.buffer.store(state, action, reward, next_state, done, log_prob, value)
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.buffer) < 64:  # 最小批次大小
            return {}
        
        # 计算优势和回报
        states, actions, rewards, next_states, dones, old_log_probs, values = self.buffer.get()
        
        # 转换为张量
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # 计算回报和优势
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        update_stats = []
        for _ in range(self.k_epochs):
            # 获取当前策略的输出
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            current_values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(current_values, returns)
            
            # 总损失
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # 记录统计信息
            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()
                explained_var = 1 - F.mse_loss(current_values, returns) / returns.var()
                
                update_stats.append({
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'entropy': entropy.item(),
                    'approx_kl': approx_kl,
                    'explained_variance': explained_var.item()
                })
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 计算平均统计
        avg_stats = {}
        for key in update_stats[0].keys():
            avg_stats[key] = np.mean([stat[key] for stat in update_stats])
            self.training_stats[key].append(avg_stats[key])
        
        return avg_stats
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gae_lambda: float = 0.95):
        """计算GAE优势"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

class PPOBuffer:
    """PPO经验缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get(self):
        """获取所有经验"""
        return (self.states, self.actions, self.rewards, self.next_states, 
                self.dones, self.log_probs, self.values)
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)
