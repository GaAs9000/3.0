"""
Training Infrastructure for Power Grid Partitioning RL

This module provides comprehensive training infrastructure including:
- Training loops with experience collection
- Model checkpointing and resuming
- Logging and monitoring
- Curriculum learning support
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
    Comprehensive logging for training process
    """
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """
        Initialize Training Logger
        
        Args:
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(str(self.log_dir / 'tensorboard'))
            
        # Training metrics
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
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.best_episode = 0
        
    def log_episode(self, episode: int, episode_data: Dict[str, Any]):
        """Log episode data"""
        # Store metrics with key mapping
        key_mapping = {
            'episode_reward': 'episode_rewards',
            'episode_length': 'episode_lengths'
        }

        for key, value in episode_data.items():
            # Use mapped key if available, otherwise use original key
            target_key = key_mapping.get(key, key)
            if target_key in self.metrics:
                self.metrics[target_key].append(value)
                
        # TensorBoard logging
        if self.writer:
            for key, value in episode_data.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Episode/{key}', value, episode)
                    
        # Update best model
        if 'episode_reward' in episode_data:
            if episode_data['episode_reward'] > self.best_reward:
                self.best_reward = episode_data['episode_reward']
                self.best_episode = episode
                
    def log_training_step(self, step: int, training_data: Dict[str, Any]):
        """Log training step data"""
        if self.writer:
            for key, value in training_data.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, step)
                    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.log_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, list):
                    serializable_metrics[key] = [float(v) if hasattr(v, 'item') else v for v in value]
                else:
                    serializable_metrics[key] = value
                    
            json.dump(serializable_metrics, f, indent=2)
            
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plots = [
            ('episode_rewards', 'Episode Rewards'),
            ('episode_lengths', 'Episode Lengths'),
            ('success_rates', 'Success Rate'),
            ('load_cv', 'Load CV'),
            ('coupling_edges', 'Coupling Edges'),
            ('actor_losses', 'Actor Loss')
        ]
        
        for i, (metric, title) in enumerate(plots):
            if metric in self.metrics and len(self.metrics[metric]) > 0:
                axes[i].plot(self.metrics[metric])
                axes[i].set_title(title)
                axes[i].grid(True)
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center')
                axes[i].set_title(title)
                
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def close(self):
        """Close logger"""
        if self.writer:
            self.writer.close()


class CheckpointManager:
    """
    Model checkpointing and resuming
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize Checkpoint Manager
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, 
                       agent: PPOAgent,
                       episode: int,
                       metrics: Dict[str, Any],
                       is_best: bool = False):
        """
        Save training checkpoint
        
        Args:
            agent: PPO agent to save
            episode: Current episode number
            metrics: Training metrics
            is_best: Whether this is the best model so far
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
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_episode_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        # Save latest model
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
    def load_checkpoint(self, agent: PPOAgent, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint
        
        Args:
            agent: PPO agent to load into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint metadata
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
        """Get path to latest checkpoint"""
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        return str(latest_path) if latest_path.exists() else None
        
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint"""
        best_path = self.checkpoint_dir / 'best_model.pt'
        return str(best_path) if best_path.exists() else None


class Trainer:
    """
    Main training class for Power Grid Partitioning RL
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
        Initialize Trainer
        
        Args:
            agent: PPO agent
            env: Training environment
            log_dir: Logging directory
            checkpoint_dir: Checkpoint directory
            save_interval: Episodes between saves
            eval_interval: Episodes between evaluations
            use_tensorboard: Whether to use TensorBoard
        """
        self.agent = agent
        self.env = env
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        
        # Setup logging and checkpointing
        self.logger = TrainingLogger(log_dir, use_tensorboard)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Training state
        self.current_episode = 0
        self.training_step = 0
        
        # Success tracking
        self.recent_successes = deque(maxlen=100)
        
    def train(self,
              num_episodes: int,
              max_steps_per_episode: int = 200,
              update_interval: int = 10,
              resume_from: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
            update_interval: Episodes between agent updates
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training history
        """
        # Resume from checkpoint if specified
        if resume_from:
            checkpoint_info = self.checkpoint_manager.load_checkpoint(self.agent, resume_from)
            self.current_episode = checkpoint_info['episode']
            print(f"Resumed training from episode {self.current_episode}")
            
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Environment: {self.env.total_nodes} nodes, {self.env.num_partitions} partitions")
        
        for episode in range(self.current_episode, self.current_episode + num_episodes):
            episode_start_time = time.time()
            
            # Run episode
            episode_data = self._run_episode(max_steps_per_episode)
            
            # Log episode
            episode_data['episode_time'] = time.time() - episode_start_time
            self.logger.log_episode(episode, episode_data)
            
            # Update success tracking
            success = self._is_successful_episode(episode_data)
            self.recent_successes.append(success)
            
            # Print progress
            if episode % 10 == 0:
                success_rate = np.mean(self.recent_successes) if self.recent_successes else 0.0
                print(f"Episode {episode}: Reward={episode_data['episode_reward']:.3f}, "
                      f"Length={episode_data['episode_length']}, "
                      f"Success Rate={success_rate:.3f}")
                      
            # Update agent
            if episode % update_interval == 0 and len(self.agent.memory.states) > 0:
                training_stats = self.agent.update()
                self.logger.log_training_step(self.training_step, training_stats)
                self.training_step += 1
                
            # Save checkpoint
            if episode % self.save_interval == 0:
                is_best = episode_data['episode_reward'] >= self.logger.best_reward
                self.checkpoint_manager.save_checkpoint(
                    self.agent, episode, self.logger.metrics, is_best
                )
                
            # Save metrics periodically
            if episode % 50 == 0:
                self.logger.save_metrics()
                self.logger.plot_training_curves()
                
        # Final save
        self.checkpoint_manager.save_checkpoint(
            self.agent, self.current_episode + num_episodes - 1, self.logger.metrics
        )
        self.logger.save_metrics()
        self.logger.plot_training_curves()
        
        print("Training completed!")
        print(f"Best reward: {self.logger.best_reward:.3f} at episode {self.logger.best_episode}")
        
        return self.logger.metrics
        
    def _run_episode(self, max_steps: int) -> Dict[str, Any]:
        """Run a single training episode"""
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action, log_prob, value = self.agent.select_action(obs, training=True)
            
            if action is None:
                # No valid actions
                break
                
            # Take step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(obs, action, reward, log_prob, value, terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            info = next_info
            
            if terminated or truncated:
                break
                
        # Get final metrics
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
        """Determine if episode was successful"""
        # Define success criteria
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
        Evaluate agent performance
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        eval_metrics = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                # Select action (greedy)
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
            
        # Compute evaluation statistics
        eval_stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes),
            'mean_load_cv': np.mean([m.get('load_cv', 0) for m in eval_metrics]),
            'mean_coupling': np.mean([m.get('coupling_edges', 0) for m in eval_metrics])
        }
        
        print(f"Evaluation Results:")
        for key, value in eval_stats.items():
            print(f"  {key}: {value:.4f}")
            
        return eval_stats
        
    def close(self):
        """Clean up trainer"""
        self.logger.close()
        self.env.close()
