#!/usr/bin/env python3
"""
电力网络分区强化学习统一训练脚本

集成了完整的训练基础设施：
- 训练循环和经验收集
- 模型检查点和恢复
- 日志记录和监控
- 可视化支持
- 配置文件驱动
"""

import torch
import numpy as np
import argparse
import os
import sys
import time
import json
import warnings
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.append(str(Path(__file__).parent))

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# 禁用警告
warnings.filterwarnings('ignore')


class TrainingLogger:
    """
    训练过程的综合日志记录
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练日志记录器
        
        Args:
            config: 完整配置字典
        """
        # 提取日志配置
        logging_config = config.get('logging', {})
        training_config = config.get('training', {})
        
        self.log_dir = Path(training_config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置参数
        self.use_tensorboard = training_config.get('use_tensorboard', True)
        self.metrics_save_interval = logging_config.get('metrics_save_interval', 50)
        self.plot_save_interval = logging_config.get('plot_save_interval', 100)
        self.console_log_interval = logging_config.get('console_log_interval', 10)
        
        # TensorBoard writer
        self.writer = None
        if self.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(str(self.log_dir / 'tensorboard'))
            
        # 训练指标
        metrics_to_track = logging_config.get('training_metrics', [
            'episode_rewards', 'episode_lengths', 'success_rates', 'load_cv',
            'coupling_edges', 'actor_losses', 'critic_losses', 'entropies'
        ])
        
        self.metrics = {metric: [] for metric in metrics_to_track}
        
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
                
        # 控制台日志
        if episode % self.console_log_interval == 0:
            print(f"回合 {episode}: 奖励={episode_data.get('episode_reward', 0):.3f}, "
                  f"长度={episode_data.get('episode_length', 0)}")
                
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
            
    def plot_training_curves(self, viz_manager):
        """绘制训练曲线"""
        if viz_manager.enabled:
            viz_manager.plot_training_curves(self.metrics, save_path="training_curves.png")
        
    def close(self):
        """关闭日志记录器"""
        if self.writer:
            self.writer.close()


class CheckpointManager:
    """
    模型检查点和恢复
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检查点管理器
        
        Args:
            config: 完整配置字典
        """
        training_config = config.get('training', {})
        self.checkpoint_dir = Path(training_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, 
                       agent,
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
        
    def load_checkpoint(self, agent, checkpoint_path: str) -> Dict[str, Any]:
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


class UnifiedTrainer:
    """
    电力网络分区强化学习的统一训练类
    整合了训练、日志、检查点、可视化等所有功能
    """
    
    def __init__(self,
                 agent,
                 env,
                 config: Dict[str, Any]):
        """
        初始化统一训练器
        
        Args:
            agent: PPO智能体
            env: 训练环境
            config: 完整配置字典
        """
        self.agent = agent
        self.env = env
        self.config = config
        
        # 提取训练配置
        training_config = config.get('training', {})
        self.save_interval = training_config.get('save_interval', 100)
        self.eval_interval = training_config.get('eval_interval', 50)
        
        # 成功判定标准
        success_criteria = training_config.get('success_criteria', {})
        self.load_cv_threshold = success_criteria.get('load_cv_threshold', 0.3)
        self.connectivity_threshold = success_criteria.get('connectivity_threshold', 0.9)
        self.min_length_threshold = success_criteria.get('min_length_threshold', 10)
        
        # 收敛检测
        convergence_config = training_config.get('convergence', {})
        self.convergence_window = convergence_config.get('window_size', 10)
        self.convergence_threshold = convergence_config.get('threshold', 0.01)
        
        # 设置组件
        self.logger = TrainingLogger(config)
        self.checkpoint_manager = CheckpointManager(config)
        
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
                
            # 定期保存指标和图表
            if episode % self.logger.metrics_save_interval == 0:
                self.logger.save_metrics()
                
            # 检查收敛
            if self._check_convergence():
                print(f"训练在第{episode}回合收敛！")
                break
                
        # 最终保存
        self.checkpoint_manager.save_checkpoint(
            self.agent, self.current_episode + num_episodes - 1, self.logger.metrics
        )
        self.logger.save_metrics()
        
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
        return (
            episode_data.get('load_cv', 1.0) < self.load_cv_threshold and
            episode_data.get('connectivity', 0.0) > self.connectivity_threshold and
            episode_data.get('episode_length', 0) >= self.min_length_threshold
        )
        
    def _check_convergence(self) -> bool:
        """检查训练是否收敛"""
        if len(self.logger.metrics['episode_rewards']) < self.convergence_window:
            return False
            
        recent_rewards = self.logger.metrics['episode_rewards'][-self.convergence_window:]
        return np.std(recent_rewards) < self.convergence_threshold
        
    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            评估指标
        """
        if num_episodes is None:
            eval_config = self.config.get('evaluation', {})
            num_episodes = eval_config.get('num_episodes', 20)
            
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
        
    def run_final_visualization(self):
        """运行最终可视化"""
        # 动态导入可视化模块
        try:
            from visualization import VisualizationManager
            viz_manager = VisualizationManager(self.config)
            if viz_manager.enabled:
                print("\n📈 生成最终可视化...")
                viz_manager.run_basic_visualization(self.env, self.logger.metrics)
        except ImportError:
            print("⚠️ 可视化模块不可用")
        
    def close(self):
        """清理训练器"""
        self.logger.close()
        if hasattr(self.env, 'close'):
            self.env.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        'data': {
            'case_name': 'ieee14',
            'normalize': True,
            'cache_dir': 'cache'
        },
        'environment': {
            'num_partitions': 3,
            'max_steps': 200,
            'reward_weights': {
                'load_balance': 0.4,
                'electrical_decoupling': 0.4,
                'power_balance': 0.2
            }
        },
        'gat': {
            'hidden_channels': 64,
            'gnn_layers': 3,
            'heads': 4,
            'output_dim': 128,
            'dropout': 0.1
        },
        'agent': {
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'k_epochs': 4,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'hidden_dim': 256,
            'dropout': 0.1
        },
        'training': {
            'num_episodes': 1000,
            'max_steps_per_episode': 200,
            'update_interval': 10,
            'save_interval': 100,
            'eval_interval': 50,
            'log_dir': 'logs',
            'checkpoint_dir': 'checkpoints',
            'use_tensorboard': True,
            'success_criteria': {
                'load_cv_threshold': 0.3,
                'connectivity_threshold': 0.9,
                'min_length_threshold': 10
            },
            'convergence': {
                'window_size': 10,
                'threshold': 0.01
            }
        },
        'evaluation': {
            'num_episodes': 20
        },
        'logging': {
            'metrics_save_interval': 50,
            'plot_save_interval': 100,
            'console_log_interval': 10
        },
        'visualization': {
            'enabled': True,
            'save_figures': True,
            'figures_dir': 'figures'
        },
        'system': {
            'device': 'auto',
            'seed': 42,
            'num_threads': 1,
            'warnings': 'ignore'
        }
    }


def load_power_grid_data(case_name: str) -> Dict[str, Any]:
    """加载电网数据"""
    if case_name == 'ieee14':
        # IEEE 14-bus测试系统
        return {
            'baseMVA': 100.0,
            'bus': np.array([
                [1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.06, 0.0, 138, 1, 1.1, 0.9],
                [2, 2, 21.7, 12.7, 0.0, 0.0, 1, 1.045, -4.98, 138, 1, 1.1, 0.9],
                [3, 2, 94.2, 19.0, 0.0, 0.0, 1, 1.01, -12.72, 138, 1, 1.1, 0.9],
                [4, 1, 47.8, -3.9, 0.0, 0.0, 1, 1.019, -10.33, 138, 1, 1.1, 0.9],
                [5, 1, 7.6, 1.6, 0.0, 0.0, 1, 1.02, -8.78, 138, 1, 1.1, 0.9],
                [6, 2, 11.2, 7.5, 0.0, 0.0, 1, 1.07, -14.22, 138, 1, 1.1, 0.9],
                [7, 1, 0.0, 0.0, 0.0, 0.0, 1, 1.062, -13.37, 138, 1, 1.1, 0.9],
                [8, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.09, -13.36, 138, 1, 1.1, 0.9],
                [9, 1, 29.5, 16.6, 0.0, 19.0, 1, 1.056, -14.94, 138, 1, 1.1, 0.9],
                [10, 1, 9.0, 5.8, 0.0, 0.0, 1, 1.051, -15.1, 138, 1, 1.1, 0.9],
                [11, 1, 3.5, 1.8, 0.0, 0.0, 1, 1.057, -14.79, 138, 1, 1.1, 0.9],
                [12, 1, 6.1, 1.6, 0.0, 0.0, 1, 1.055, -15.07, 138, 1, 1.1, 0.9],
                [13, 1, 13.5, 5.8, 0.0, 0.0, 1, 1.05, -15.16, 138, 1, 1.1, 0.9],
                [14, 1, 14.9, 5.0, 0.0, 0.0, 1, 1.036, -16.04, 138, 1, 1.1, 0.9],
            ]),
            'branch': np.array([
                [1, 2, 0.01938, 0.05917, 0.0528, 100, 110, 120, 0, 0, 1, -360, 360],
                [1, 5, 0.05403, 0.22304, 0.0492, 100, 110, 120, 0, 0, 1, -360, 360],
                [2, 3, 0.04699, 0.19797, 0.0438, 100, 110, 120, 0, 0, 1, -360, 360],
                [2, 4, 0.05811, 0.17632, 0.034, 100, 110, 120, 0, 0, 1, -360, 360],
                [2, 5, 0.05695, 0.17388, 0.0346, 100, 110, 120, 0, 0, 1, -360, 360],
                [3, 4, 0.06701, 0.17103, 0.0128, 100, 110, 120, 0, 0, 1, -360, 360],
                [4, 5, 0.01335, 0.04211, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [4, 7, 0.0, 0.20912, 0.0, 100, 110, 120, 0.978, 0, 1, -360, 360],
                [4, 9, 0.0, 0.55618, 0.0, 100, 110, 120, 0.969, 0, 1, -360, 360],
                [5, 6, 0.0, 0.25202, 0.0, 100, 110, 120, 0.932, 0, 1, -360, 360],
                [6, 11, 0.09498, 0.1989, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [6, 12, 0.12291, 0.25581, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [6, 13, 0.06615, 0.13027, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [7, 8, 0.0, 0.17615, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [7, 9, 0.0, 0.11001, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [9, 10, 0.03181, 0.0845, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [9, 14, 0.12711, 0.27038, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [10, 11, 0.08205, 0.19207, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [12, 13, 0.22092, 0.19988, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [13, 14, 0.17093, 0.34802, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            ]),
            'gen': np.array([
                [1, 232.4, -16.9, 10, -10, 1.06, 100, 1, 332.4, 0],
                [2, 40.0, 43.56, 50, -40, 1.045, 100, 1, 140.0, 0],
                [3, 0.0, 25.075, 40, 0, 1.01, 100, 1, 100.0, 0],
                [6, 0.0, 12.73, 24, -6, 1.07, 100, 1, 100.0, 0],
                [8, 0.0, 17.623, 24, -6, 1.09, 100, 1, 100.0, 0],
            ])
        }
    else:
        # 尝试从文件加载
        if os.path.exists(case_name):
            raise NotImplementedError("MATPOWER文件加载尚未实现")
        else:
            raise ValueError(f"未知案例: {case_name}")


def setup_device(device_config: str) -> torch.device:
    """设置计算设备"""
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
        
    print(f"使用设备: {device}")
    return device


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='电力网络分区强化学习训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--case', type=str, default='ieee14', help='电网案例名称')
    parser.add_argument('--episodes', type=int, help='训练回合数')
    parser.add_argument('--partitions', type=int, help='分区数量')
    parser.add_argument('--resume', type=str, help='恢复检查点路径')
    parser.add_argument('--eval-only', action='store_true', help='仅运行评估')
    parser.add_argument('--save-config', type=str, help='保存默认配置到文件')
    parser.add_argument('--preset', type=str, choices=['quick', 'full', 'large'], 
                       help='使用预设配置')
    
    args = parser.parse_args()
    
    # 保存默认配置
    if args.save_config:
        config = create_default_config()
        with open(args.save_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
        print(f"默认配置已保存到 {args.save_config}")
        return
    
    # 动态导入模块（避免在保存配置时出错）
    try:
        from data_processing import PowerGridDataProcessor
        from gat import create_hetero_graph_encoder
        from rl.environment import PowerGridPartitioningEnv
        from rl.agent import PPOAgent
        from visualization import VisualizationManager
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        return 1
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        
    # 应用预设配置
    if args.preset:
        preset_key = f"{args.preset}_training"
        if preset_key in config:
            preset_config = config[preset_key]
            # 递归更新配置
            def update_config(base, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in base:
                        update_config(base[key], value)
                    else:
                        base[key] = value
            update_config(config, preset_config)
        
    # 命令行参数覆盖
    if args.case:
        config['data']['case_name'] = args.case
    if args.episodes:
        config['training']['num_episodes'] = args.episodes
    if args.partitions:
        config['environment']['num_partitions'] = args.partitions
        
    print("🚀 启动电力网络分区强化学习训练")
    print("=" * 60)
    print(f"配置信息:")
    print(f"  案例: {config['data']['case_name']}")
    print(f"  分区数: {config['environment']['num_partitions']}")
    print(f"  训练回合: {config['training']['num_episodes']}")
    
    # 系统设置
    torch.manual_seed(config['system']['seed'])
    np.random.seed(config['system']['seed'])
    torch.set_num_threads(config['system']['num_threads'])
    device = setup_device(config['system']['device'])
    
    try:
        # 1. 加载和处理数据
        print("\n📊 加载和处理电网数据...")
        mpc = load_power_grid_data(config['data']['case_name'])
        
        processor = PowerGridDataProcessor(
            normalize=config['data']['normalize'],
            cache_dir=config['data']['cache_dir']
        )
        hetero_data = processor.graph_from_mpc(mpc)
        hetero_data = hetero_data.to(device)
        
        print(f"✅ 数据加载完成: {hetero_data}")
        
        # 2. 创建GAT编码器
        print("\n🧠 创建GAT编码器...")
        gat_config = config['gat']
        encoder = create_hetero_graph_encoder(
            hetero_data,
            hidden_channels=gat_config['hidden_channels'],
            gnn_layers=gat_config['gnn_layers'],
            heads=gat_config['heads'],
            output_dim=gat_config['output_dim']
        ).to(device)

        with torch.no_grad():
            node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data)

        total_nodes = sum(emb.shape[0] for emb in node_embeddings.values())
        print(f"✅ 节点嵌入计算完成: {total_nodes}个节点")
        print(f"✅ 注意力权重提取完成: {len(attention_weights)}种边类型")
        
        # 3. 创建强化学习环境
        print("\n🌍 创建强化学习环境...")
        env_config = config['environment']
        env = PowerGridPartitioningEnv(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=env_config['num_partitions'],
            reward_weights=env_config['reward_weights'],
            max_steps=env_config['max_steps'],
            device=device,
            attention_weights=attention_weights
        )
        
        print(f"✅ 环境创建完成: {env.total_nodes}个节点, {env.num_partitions}个分区")
        
        # 4. 创建PPO智能体
        print("\n🤖 创建PPO智能体...")
        agent_config = config['agent']
        node_embedding_dim = env.state_manager.embedding_dim
        region_embedding_dim = node_embedding_dim * 2

        print(f"   增强节点嵌入维度: {node_embedding_dim}")
        print(f"   区域嵌入维度: {region_embedding_dim}")
        
        agent = PPOAgent(
            node_embedding_dim=node_embedding_dim,
            region_embedding_dim=region_embedding_dim,
            num_partitions=env.num_partitions,
            lr_actor=agent_config['lr_actor'],
            lr_critic=agent_config['lr_critic'],
            gamma=agent_config['gamma'],
            eps_clip=agent_config['eps_clip'],
            k_epochs=agent_config['k_epochs'],
            entropy_coef=agent_config['entropy_coef'],
            value_coef=agent_config['value_coef'],
            device=device
        )
        
        print(f"✅ 智能体创建完成，参数量: {sum(p.numel() for p in agent.actor.parameters()):,}")
        
        # 5. 创建统一训练器
        print("\n🏋️ 设置统一训练器...")
        trainer = UnifiedTrainer(agent=agent, env=env, config=config)
        
        # 6. 运行训练或评估
        training_config = config['training']
        if args.eval_only:
            print("\n📈 运行评估...")
            if args.resume:
                trainer.checkpoint_manager.load_checkpoint(agent, args.resume)
                print(f"从{args.resume}加载模型")
            eval_stats = trainer.evaluate()
            print("\n评估结果:")
            for key, value in eval_stats.items():
                print(f"  {key}: {value:.4f}")
        else:
            print("\n🚀 开始训练...")
            history = trainer.train(
                num_episodes=training_config['num_episodes'],
                max_steps_per_episode=training_config['max_steps_per_episode'],
                update_interval=training_config['update_interval'],
                resume_from=args.resume
            )
            
            print("\n训练完成！")
            print(f"最终平均奖励: {np.mean(history['episode_rewards'][-10:]):.3f}")
            
            # 运行最终评估
            print("\n📈 运行最终评估...")
            eval_stats = trainer.evaluate()
            print("\n最终评估结果:")
            for key, value in eval_stats.items():
                print(f"  {key}: {value:.4f}")
                
            # 生成可视化
            trainer.run_final_visualization()
        
        # 清理
        trainer.close()
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\n🎉 训练成功完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
