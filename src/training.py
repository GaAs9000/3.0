import torch
import numpy as np
import time
import os
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Tuple, Optional

# Import types that will be defined in other modules
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import PPOAgent
    from env import PowerGridPartitionEnv, CurriculumLearningEnv

def train_ppo(agent: 'PPOAgent', env: Union['PowerGridPartitionEnv', 'CurriculumLearningEnv'],
              n_episodes: int = 1000, max_steps: int = 500,
              update_interval: int = 10, save_interval: int = 100,
              use_tensorboard: bool = True, start_episode: int = 0,
              checkpoint_dir: str = 'models') -> Dict[str, List[float]]:
    """
    训练PPO智能体
    
    参数:
        agent: PPO智能体
        env: 训练环境
        n_episodes: 训练回合数
        max_steps: 每回合最大步数
        update_interval: 更新间隔
        save_interval: 保存间隔
        use_tensorboard: 是否使用TensorBoard
        
    返回:
        训练历史
    """
    # TensorBoard记录器
    if use_tensorboard:
        writer = SummaryWriter(f'runs/power_grid_partition_{time.strftime("%Y%m%d_%H%M%S")}')
    
    # 训练历史
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'load_cv': [],
        'total_coupling': [],
        'success_rate': []
    }
    
    # 成功率追踪
    recent_successes = deque(maxlen=100)
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 进度条
    pbar = tqdm(range(start_episode, n_episodes), desc="Training Progress")
    
    for episode in pbar:
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 执行一个回合
        for step in range(max_steps):
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # 选择动作
            action_value = agent.select_action(state, valid_actions, training=True)
            if action_value is None:
                break
            
            action, value = action_value
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新经验池中的奖励
            agent.update_last_reward(reward, done)
            
            # 累积奖励
            episode_reward += reward
            episode_length += 1
            
            # 更新状态
            state = next_state
            
            if done:
                break
        
        # 记录回合结果
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        
        # 记录最终指标
        final_metrics = env.current_metrics
        history['load_cv'].append(final_metrics.load_cv)
        history['total_coupling'].append(final_metrics.total_coupling)
        
        # 判断是否成功
        success = (final_metrics.load_cv < 0.3 and 
                  final_metrics.connectivity == 1.0 and
                  episode_length == env.N)
        recent_successes.append(success)
        
        # 更新智能体（每隔一定回合）
        if (episode + 1) % update_interval == 0 and len(agent.memory) > 0:
            update_stats = agent.update(epochs=4, batch_size=64)
            
            # 更新学习率
            avg_reward = np.mean(history['episode_rewards'][-update_interval:])
            agent.scheduler_actor.step(avg_reward)
            agent.scheduler_critic.step(avg_reward)
        
        # TensorBoard记录
        if use_tensorboard:
            writer.add_scalar('Train/Episode_Reward', episode_reward, episode)
            writer.add_scalar('Train/Episode_Length', episode_length, episode)
            writer.add_scalar('Train/Load_CV', final_metrics.load_cv, episode)
            writer.add_scalar('Train/Total_Coupling', final_metrics.total_coupling, episode)
            writer.add_scalar('Train/Success_Rate', np.mean(recent_successes), episode)
            writer.add_scalar('Train/Connectivity', final_metrics.connectivity, episode)
            writer.add_scalar('Train/Power_Balance', final_metrics.power_balance, episode)
            writer.add_scalar('Train/Modularity', final_metrics.modularity, episode)
            
            # 学习率监控
            if hasattr(agent, 'scheduler_actor'):
                writer.add_scalar('Learning_Rate/Actor', agent.scheduler_actor.get_last_lr()[0], episode)
            if hasattr(agent, 'scheduler_critic'):
                writer.add_scalar('Learning_Rate/Critic', agent.scheduler_critic.get_last_lr()[0], episode)
            
            # 损失函数监控
            if (episode + 1) % update_interval == 0 and len(update_stats) > 0:
                if 'policy_loss' in update_stats:
                    writer.add_scalar('Loss/Policy', update_stats['policy_loss'], episode)
                if 'value_loss' in update_stats:
                    writer.add_scalar('Loss/Value', update_stats['value_loss'], episode)
                if 'entropy' in update_stats:
                    writer.add_scalar('Loss/Entropy', update_stats['entropy'], episode)
                if 'kl_divergence' in update_stats:
                    writer.add_scalar('Loss/KL_Divergence', update_stats['kl_divergence'], episode)
            
            # 每100回合记录分布统计
            if (episode + 1) % 100 == 0:
                recent_rewards = history['episode_rewards'][-100:] if len(history['episode_rewards']) >= 100 else history['episode_rewards']
                writer.add_histogram('Stats/Episode_Rewards_Distribution', np.array(recent_rewards), episode)
                writer.add_scalar('Stats/Reward_Std', np.std(recent_rewards), episode)
                writer.add_scalar('Stats/Best_Reward_Last100', np.max(recent_rewards), episode)
        
        # 更新进度条
        success_rate = np.mean(recent_successes) if recent_successes else 0
        pbar.set_postfix({
            'Reward': f'{episode_reward:.2f}',
            'CV': f'{final_metrics.load_cv:.3f}',
            'Success': f'{success_rate:.2%}'
        })
        
        # 保存模型和检查点
        if (episode + 1) % save_interval == 0:
            # 保存智能体模型
            model_path = f'{checkpoint_dir}/ppo_checkpoint_ep{episode+1}.pth'
            agent.save(model_path)
            
            # 保存完整检查点（包含训练状态）
            checkpoint_path = f'{checkpoint_dir}/training_checkpoint_ep{episode+1}.pth'
            checkpoint = {
                'episode': episode + 1,
                'agent_state': agent.state_dict(),
                'history': history,
                'recent_successes': list(recent_successes),
                'env_state': getattr(env, 'get_state', lambda: None)(),
                'training_config': {
                    'n_episodes': n_episodes,
                    'max_steps': max_steps,
                    'update_interval': update_interval,
                    'save_interval': save_interval
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"\n💾 检查点已保存: {checkpoint_path}")
    
    # 关闭TensorBoard
    if use_tensorboard:
        writer.close()
    
    # 保存最终检查点
    final_checkpoint_path = f'{checkpoint_dir}/training_final.pth'
    final_checkpoint = {
        'episode': n_episodes,
        'agent_state': agent.state_dict(),
        'history': history,
        'recent_successes': list(recent_successes),
        'env_state': getattr(env, 'get_state', lambda: None)(),
        'training_completed': True
    }
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"\n🏁 最终检查点已保存: {final_checkpoint_path}")
    
    return history


def load_training_checkpoint(checkpoint_path: str) -> Dict:
    """加载训练检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ 检查点加载成功:")
    print(f"   - 回合: {checkpoint.get('episode', 0)}")
    print(f"   - 历史长度: {len(checkpoint.get('history', {}).get('episode_rewards', []))}")
    print(f"   - 训练完成: {checkpoint.get('training_completed', False)}")
    
    return checkpoint

def full_training(agent, curriculum_env, resume_from: str = None):
    """完整训练函数 - 生产环境配置"""
    print("\n🚀 开始完整训练...")
    print("这是完整的训练配置，将进行长时间训练以获得最佳性能")
    
    # 检查是否从检查点恢复
    start_episode = 0
    previous_history = None
    if resume_from and os.path.exists(resume_from):
        print(f"📥 从检查点恢复训练: {resume_from}")
        checkpoint = load_training_checkpoint(resume_from)
        agent.load_state_dict(checkpoint['agent_state'])
        start_episode = checkpoint.get('episode', 0)
        previous_history = checkpoint.get('history', None)
        print(f"从第 {start_episode} 回合继续训练")
    
    # 完整训练配置
    history = train_ppo(
        agent=agent,
        env=curriculum_env,
        n_episodes=2000,  # 完整训练回合数
        max_steps=500,    # 每回合最大步数
        update_interval=10,  # 每10回合更新一次
        save_interval=50,    # 每50回合保存一次
        use_tensorboard=True,  # 启用详细日志
        start_episode=start_episode
    )
    
    # 如果有之前的历史，合并历史数据
    if previous_history:
        for key in history:
            if key in previous_history:
                history[key] = previous_history[key] + history[key]
    
    # 保存最终模型
    final_model_path = 'models/ppo_final_model.pth'
    agent.save(final_model_path)
    print(f"\n💾 最终模型已保存至: {final_model_path}")
    
    print("\n✅ 完整训练完成！")
    print(f"📊 总训练回合: {len(history['episode_rewards'])}")
    print(f"📊 平均奖励: {np.mean(history['episode_rewards']):.3f}")
    print(f"📊 最佳回合奖励: {np.max(history['episode_rewards']):.3f}")
    print(f"📊 最终Load CV: {history['load_cv'][-1]:.3f}")
    print(f"📊 最终耦合度: {history['total_coupling'][-1]:.3f}")
    
    return history

def quick_training(agent, curriculum_env):
    """快速训练函数 - 演示和测试用"""
    print("\n🚀 开始快速训练...")
    print("这是快速演示配置，适合测试和调试")

    # 快速训练配置
    history = train_ppo(
        agent=agent,
        env=curriculum_env,
        n_episodes=100,   # 适中的回合数
        max_steps=200,    # 适中的步数
        update_interval=10,
        save_interval=25,
        use_tensorboard=True
    )

    print("\n✅ 快速训练完成！")
    print(f"📊 平均奖励: {np.mean(history['episode_rewards']):.3f}")
    print(f"📊 最终Load CV: {history['load_cv'][-1]:.3f}")
    print(f"📊 最终耦合度: {history['total_coupling'][-1]:.3f}")
    
    return history

