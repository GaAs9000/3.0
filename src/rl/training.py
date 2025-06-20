"""
电力网络分区强化学习的训练基础设施

本模块提供全面的训练基础设施，包括：
- 带有经验收集的训练循环
- 模型检查点和恢复
- 日志记录和监控
- 课程学习支持
"""

import torch
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from .agent import PPOAgent
from .environment import PowerGridPartitioningEnv


class TrainingLogger:
    """
    训练过程的综合日志记录
    """
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """
        初始化训练日志记录器
        
        Args:
            log_dir: 日志目录
            use_tensorboard: 是否使用TensorBoard日志记录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(str(self.log_dir / 'tensorboard'))
            
        # 训练指标
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'load_cv': [],
            'coupling_edges': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': []
        }
        
        # 最佳模型跟踪
        self.best_reward = float('-inf')
        self.best_episode = 0
        
    def log_episode(self, episode: int, episode_data: Dict[str, Any]):
        """记录回合数据"""
        # 存储带有键映射的指标
        key_mapping = {
            'episode_reward': 'episode_rewards',
            'episode_length': 'episode_lengths'
        }

        for key, value in episode_data.items():
            # 如果可用使用映射键，否则使用原始键
            target_key = key_mapping.get(key, key)
            if target_key in self.metrics:
                self.metrics[target_key].append(value)
                
        # TensorBoard日志记录
        if self.writer:
            for key, value in episode_data.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Episode/{key}', value, episode)
                    
        # 更新最佳模型
        if 'episode_reward' in episode_data:
            if episode_data['episode_reward'] > self.best_reward:
                self.best_reward = episode_data['episode_reward']
                self.best_episode = episode
                
    def log_training_step(self, step: int, training_data: Dict[str, Any]):
        """记录训练步骤数据"""
        if self.writer:
            for key, value in training_data.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, step)
                    
    def save_metrics(self):
        """将指标保存到文件"""
        metrics_file = self.log_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            # 将numpy数组转换为列表以进行JSON序列化
            serializable_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, list):
                    serializable_metrics[key] = [float(v) if hasattr(v, 'item') else v for v in value]
                else:
                    serializable_metrics[key] = value
                    
            json.dump(serializable_metrics, f, indent=2)
            
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plots = [
            ('episode_rewards', '回合奖励'),
            ('episode_lengths', '回合长度'),
            ('success_rates', '成功率'),
            ('load_cv', '负载变异系数'),
            ('coupling_edges', '耦合边数'),
            ('actor_losses', '演员损失')
        ]
        
        for i, (metric, title) in enumerate(plots):
            if metric in self.metrics and len(self.metrics[metric]) > 0:
                axes[i].plot(self.metrics[metric])
                axes[i].set_title(title)
                axes[i].grid(True)
            else:
                axes[i].text(0.5, 0.5, '无数据', ha='center', va='center')
                axes[i].set_title(title)
                
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def close(self):
        """关闭日志记录器"""
        if self.writer:
            self.writer.close()


class CheckpointManager:
    """
    模型检查点和恢复
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, 
                       agent: PPOAgent,
                       episode: int,
                       metrics: Dict[str, Any],
                       is_best: bool = False):
        """
        保存训练检查点
        
        Args:
            agent: 要保存的PPO智能体
            episode: 当前回合数
            metrics: 训练指标
            is_best: 是否是目前最佳模型
        """
        checkpoint = {
            'episode': episode,
            'agent_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        # 保存常规检查点
        checkpoint_path = self.checkpoint_dir / f'checkpoint_episode_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        # 保存最新模型
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
    def load_checkpoint(self, agent: PPOAgent, checkpoint_path: str) -> Dict[str, Any]:
        """
        加载训练检查点
        
        Args:
            agent: 要加载到的PPO智能体
            checkpoint_path: 检查点文件路径
            
        Returns:
            检查点元数据
        """
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        
        agent.actor.load_state_dict(checkpoint['agent_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        return {
            'episode': checkpoint['episode'],
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', 0)
        }
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点的路径"""
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        return str(latest_path) if latest_path.exists() else None
        
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点的路径"""
        best_path = self.checkpoint_dir / 'best_model.pt'
        return str(best_path) if best_path.exists() else None


class Trainer:
    """
    电力网络分区强化学习的主要训练类
    """
    
    def __init__(self,
                 agent: PPOAgent,
                 env: PowerGridPartitioningEnv,
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints',
                 save_interval: int = 100,
                 eval_interval: int = 50,
                 use_tensorboard: bool = True):
        """
        初始化训练器
        
        Args:
            agent: PPO智能体
            env: 训练环境
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
            save_interval: 保存间隔（回合数）
            eval_interval: 评估间隔（回合数）
            use_tensorboard: 是否使用TensorBoard
        """
        self.agent = agent
        self.env = env
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        
        # 设置日志记录和检查点
        self.logger = TrainingLogger(log_dir, use_tensorboard)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 训练状态
        self.current_episode = 0
        self.training_step = 0
        
        # 成功跟踪
        self.recent_successes = deque(maxlen=100)
        
    def train(self,
              num_episodes: int,
              max_steps_per_episode: int = 200,
              update_interval: int = 10,
              resume_from: Optional[str] = None) -> Dict[str, List[float]]:
        """
        主训练循环
        
        Args:
            num_episodes: 训练的回合数
            max_steps_per_episode: 每回合最大步数
            update_interval: 智能体更新间隔（回合数）
            resume_from: 要恢复的检查点路径
            
        Returns:
            训练历史
        """
        # 如果指定则从检查点恢复
        if resume_from:
            checkpoint_info = self.checkpoint_manager.load_checkpoint(self.agent, resume_from)
            self.current_episode = checkpoint_info['episode']
            print(f"从第{self.current_episode}回合恢复训练")
            
        print(f"开始训练{num_episodes}回合...")
        print(f"环境：{self.env.total_nodes}个节点，{self.env.num_partitions}个分区")
        
        for episode in range(self.current_episode, self.current_episode + num_episodes):
            episode_start_time = time.time()
            
            # 运行回合
            episode_data = self._run_episode(max_steps_per_episode)
            
            # 记录回合
            episode_data['episode_time'] = time.time() - episode_start_time
            self.logger.log_episode(episode, episode_data)
            
            # 更新成功跟踪
            success = self._is_successful_episode(episode_data)
            self.recent_successes.append(success)
            
            # 打印进度
            if episode % 10 == 0:
                success_rate = np.mean(self.recent_successes) if self.recent_successes else 0.0
                print(f"回合{episode}：奖励={episode_data['episode_reward']:.3f}，"
                      f"长度={episode_data['episode_length']}，"
                      f"成功率={success_rate:.3f}")
                      
            # 更新智能体
            if episode % update_interval == 0 and len(self.agent.memory.states) > 0:
                training_stats = self.agent.update()
                self.logger.log_training_step(self.training_step, training_stats)
                self.training_step += 1
                
            # 保存检查点
            if episode % self.save_interval == 0:
                is_best = episode_data['episode_reward'] >= self.logger.best_reward
                self.checkpoint_manager.save_checkpoint(
                    self.agent, episode, self.logger.metrics, is_best
                )
                
            # 定期保存指标
            if episode % 50 == 0:
                self.logger.save_metrics()
                self.logger.plot_training_curves()
                
        # 最终保存
        self.checkpoint_manager.save_checkpoint(
            self.agent, self.current_episode + num_episodes - 1, self.logger.metrics
        )
        self.logger.save_metrics()
        self.logger.plot_training_curves()
        
        print("训练完成！")
        print(f"最佳奖励：{self.logger.best_reward:.3f}，在第{self.logger.best_episode}回合")
        
        return self.logger.metrics
        
    def _run_episode(self, max_steps: int) -> Dict[str, Any]:
        """运行单个训练回合"""
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(max_steps):
            # 选择动作
            action, log_prob, value = self.agent.select_action(obs, training=True)
            
            if action is None:
                # 没有有效动作
                break
                
            # 执行步骤
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            # 存储经验
            self.agent.store_experience(obs, action, reward, log_prob, value, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            info = next_info
            
            if terminated or truncated:
                break
                
        # 获取最终指标
        final_metrics = info.get('metrics', {})
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'load_cv': final_metrics.get('load_cv', 0.0),
            'coupling_edges': final_metrics.get('coupling_edges', 0),
            'connectivity': final_metrics.get('connectivity', 0.0),
            'final_metrics': final_metrics
        }
        
    def _is_successful_episode(self, episode_data: Dict[str, Any]) -> bool:
        """判断回合是否成功"""
        # 定义成功标准
        load_cv_threshold = 0.3
        connectivity_threshold = 0.9
        min_length_threshold = 10
        
        return (
            episode_data.get('load_cv', 1.0) < load_cv_threshold and
            episode_data.get('connectivity', 0.0) > connectivity_threshold and
            episode_data.get('episode_length', 0) >= min_length_threshold
        )
        
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            评估指标
        """
        print(f"评估智能体{num_episodes}回合...")
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        eval_metrics = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                # 选择动作（贪心）
                action, _, _ = self.agent.select_action(obs, training=False)
                
                if action is None:
                    break
                    
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
                    
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(self._is_successful_episode({
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                **info.get('metrics', {})
            }))
            eval_metrics.append(info.get('metrics', {}))
            
        # 计算评估统计
        eval_stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes),
            'mean_load_cv': np.mean([m.get('load_cv', 0) for m in eval_metrics]),
            'mean_coupling': np.mean([m.get('coupling_edges', 0) for m in eval_metrics])
        }
        
        print(f"评估结果：")
        for key, value in eval_stats.items():
            print(f"  {key}: {value:.4f}")
            
        return eval_stats
        
    def close(self):
        """清理训练器"""
        self.logger.close()
        self.env.close()
