#!/usr/bin/env python3
"""
电力网络分区强化学习训练模块

专注于模型训练功能：
- 配置加载和验证
- 数据处理和环境创建
- 模型训练和优化
- 基础可视化和结果保存
- 训练监控和日志记录
"""

import torch
import numpy as np
import argparse
import yaml
import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

# 添加code/src到路径
sys.path.append(str(Path(__file__).parent / 'code' / 'src'))
# 添加code到路径以便导入baseline
sys.path.append(str(Path(__file__).parent / 'code'))

# --- 全局 NaN/Inf 异常检测开关 ---
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("error", message=".*NaN.*", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 导入电力系统数据加载函数
try:
    from scipy.io import loadmat
except ImportError:
    print("⚠️ 警告: scipy未安装，无法加载MATPOWER文件")

try:
    import pandapower as pp
    import pandapower.networks as pn
    _pandapower_available = True
except ImportError:
    _pandapower_available = False
    print("⚠️ 警告: pandapower未安装，无法加载IEEE标准测试系统")


def check_dependencies():
    """检查可选依赖"""
    deps = {
        'stable_baselines3': False,
        'use_tensorboard': False,
        'plotly': False,
        'networkx': False
    }

    try:
        import stable_baselines3
        deps['stable_baselines3'] = True
    except ImportError:
        pass

    try:
        from torch.utils.tensorboard import SummaryWriter
        deps['use_tensorboard'] = True
    except ImportError:
        pass

    try:
        import plotly.graph_objects as go
        deps['plotly'] = True
    except ImportError:
        pass

    try:
        import networkx as nx
        deps['networkx'] = True
    except ImportError:
        pass

    return deps


def load_power_grid_data(case_name: str) -> Dict:
    """
    加载电力网络数据

    Args:
        case_name: 案例名称 (ieee14, ieee30, ieee57, ieee118)

    Returns:
        MATPOWER格式的电网数据字典
    """
    # 首先尝试从PandaPower加载
    if _pandapower_available:
        try:
            return load_from_pandapower(case_name)
        except Exception as e:
            print(f"⚠️ PandaPower加载失败: {e}")
            print("🔄 回退到内置数据...")

    # 回退到内置的IEEE标准测试系统数据
    ieee_cases = {
        'ieee14': {
            'baseMVA': 100.0,
            'bus': np.array([
                [1, 3, 0, 0, 0, 0, 1, 1.06, 0, 345, 1, 1.06, 0.94],
                [2, 2, 21.7, 12.7, 0, 0, 1, 1.045, -4.98, 345, 1, 1.06, 0.94],
                [3, 2, 94.2, 19, 0, 0, 1, 1.01, -12.72, 345, 1, 1.06, 0.94],
                [4, 1, 47.8, -3.9, 0, 0, 1, 1.019, -10.33, 345, 1, 1.06, 0.94],
                [5, 1, 7.6, 1.6, 0, 0, 1, 1.02, -8.78, 345, 1, 1.06, 0.94],
                [6, 2, 11.2, 7.5, 0, 12.2, 1, 1.07, -14.22, 345, 1, 1.06, 0.94],
                [7, 1, 0, 0, 0, 0, 1, 1.062, -13.37, 345, 1, 1.06, 0.94],
                [8, 2, 0, 0, 0, 17.4, 1, 1.09, -13.36, 345, 1, 1.06, 0.94],
                [9, 1, 29.5, 16.6, 0, 0, 1, 1.056, -14.94, 345, 1, 1.06, 0.94],
                [10, 1, 9, 5.8, 0, 0, 1, 1.051, -15.1, 345, 1, 1.06, 0.94],
                [11, 1, 3.5, 1.8, 0, 0, 1, 1.057, -14.79, 345, 1, 1.06, 0.94],
                [12, 1, 6.1, 1.6, 0, 0, 1, 1.055, -15.07, 345, 1, 1.06, 0.94],
                [13, 1, 13.5, 5.8, 0, 0, 1, 1.05, -15.16, 345, 1, 1.06, 0.94],
                [14, 1, 14.9, 5, 0, 0, 1, 1.036, -16.04, 345, 1, 1.06, 0.94]
            ]),
            'branch': np.array([
                [1, 2, 0.01938, 0.05917, 0.0528, 0, 0, 0, 0, 0, 1, -360, 360],
                [1, 5, 0.05403, 0.22304, 0.0492, 0, 0, 0, 0, 0, 1, -360, 360],
                [2, 3, 0.04699, 0.19797, 0.0438, 0, 0, 0, 0, 0, 1, -360, 360],
                [2, 4, 0.05811, 0.17632, 0.034, 0, 0, 0, 0, 0, 1, -360, 360],
                [2, 5, 0.05695, 0.17388, 0.0346, 0, 0, 0, 0, 0, 1, -360, 360],
                [3, 4, 0.06701, 0.17103, 0.0128, 0, 0, 0, 0, 0, 1, -360, 360],
                [4, 5, 0.01335, 0.04211, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [4, 7, 0, 0.20912, 0, 0.978, 0, 0, 0, 0, 1, -360, 360],
                [4, 9, 0, 0.55618, 0, 0.969, 0, 0, 0, 0, 1, -360, 360],
                [5, 6, 0, 0.25202, 0, 0.932, 0, 0, 0, 0, 1, -360, 360],
                [6, 11, 0.09498, 0.1989, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [6, 12, 0.12291, 0.25581, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [6, 13, 0.06615, 0.13027, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [7, 8, 0, 0.17615, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [7, 9, 0, 0.11001, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [9, 10, 0.03181, 0.0845, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [9, 14, 0.12711, 0.27038, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [10, 11, 0.08205, 0.19207, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [12, 13, 0.22092, 0.19988, 0, 0, 0, 0, 0, 0, 1, -360, 360],
                [13, 14, 0.17093, 0.34802, 0, 0, 0, 0, 0, 0, 1, -360, 360]
            ])
        }
    }

    if case_name in ieee_cases:
        return ieee_cases[case_name]
    else:
        # 对于其他案例，返回默认的IEEE14数据
        print(f"⚠️ 案例 {case_name} 未找到，使用IEEE14默认数据")
        return ieee_cases['ieee14']


def load_from_pandapower(case_name: str) -> Dict:
    """
    从PandaPower加载IEEE标准测试系统

    Args:
        case_name: 案例名称 (ieee14, ieee30, ieee57, ieee118)

    Returns:
        MATPOWER格式的电网数据字典
    """
    print(f"🔌 从PandaPower加载 {case_name.upper()} 数据...")

    # 映射案例名称到PandaPower函数
    case_mapping = {
        'ieee14': pn.case14,
        'ieee30': pn.case30,
        'ieee57': pn.case57,
        'ieee118': pn.case118
    }

    if case_name not in case_mapping:
        raise ValueError(f"不支持的案例: {case_name}")

    # 加载PandaPower网络
    net = case_mapping[case_name]()
    from rich_output import rich_success
    rich_success(f"成功加载 {case_name.upper()}: {len(net.bus)} 节点, {len(net.line)} 线路")

    # 转换为MATPOWER格式
    mpc = convert_pandapower_to_matpower(net)

    return mpc


def convert_pandapower_to_matpower(net) -> Dict:
    """
    将PandaPower网络转换为MATPOWER格式

    Args:
        net: PandaPower网络对象

    Returns:
        MATPOWER格式的电网数据字典
    """
    # 基础MVA
    baseMVA = net.sn_mva

    # 节点数据转换
    bus_data = []
    for idx, bus in net.bus.iterrows():
        # MATPOWER bus格式: [bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
        bus_type = 1  # PQ节点

        # 检查是否为发电机节点
        if idx in net.gen.bus.values:
            gen_at_bus = net.gen[net.gen.bus == idx]
            if not gen_at_bus.empty:
                # 检查是否为slack节点
                if any(gen_at_bus.slack):
                    bus_type = 3  # slack节点
                else:
                    bus_type = 2  # PV节点

        # 获取负荷数据
        pd = qd = 0
        if idx in net.load.bus.values:
            load_at_bus = net.load[net.load.bus == idx]
            pd = load_at_bus.p_mw.sum()
            qd = load_at_bus.q_mvar.sum()

        # 获取电压等级
        vn_kv = bus.vn_kv if hasattr(bus, 'vn_kv') else 345.0  # 默认345kV

        bus_row = [
            idx + 1,  # bus number (1-indexed)
            bus_type,  # bus type
            pd,  # Pd (MW)
            qd,  # Qd (MVAr)
            0,  # Gs (MW)
            0,  # Bs (MVAr)
            1,  # area
            1.0,  # Vm (p.u.)
            0,  # Va (degrees)
            vn_kv,  # baseKV
            1,  # zone
            1.1,  # Vmax
            0.9   # Vmin
        ]
        bus_data.append(bus_row)

    # 线路数据转换
    branch_data = []
    for idx, line in net.line.iterrows():
        # 获取线路电压等级（从连接的节点获取）
        from_bus_vn = net.bus.loc[line.from_bus, 'vn_kv'] if line.from_bus in net.bus.index else 345.0

        # MATPOWER branch格式: [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]
        # 简化的阻抗计算
        r_pu = line.r_ohm_per_km * line.length_km / (from_bus_vn**2 / baseMVA) if hasattr(line, 'r_ohm_per_km') else 0.01
        x_pu = line.x_ohm_per_km * line.length_km / (from_bus_vn**2 / baseMVA) if hasattr(line, 'x_ohm_per_km') else 0.05
        b_pu = 0.0  # 简化处理
        rate_a = line.max_i_ka * from_bus_vn * np.sqrt(3) / 1000 if hasattr(line, 'max_i_ka') else 100.0

        branch_row = [
            line.from_bus + 1,  # from bus (1-indexed)
            line.to_bus + 1,    # to bus (1-indexed)
            r_pu,  # r (p.u.)
            x_pu,  # x (p.u.)
            b_pu,  # b (p.u.)
            rate_a,  # rateA (MVA)
            0,  # rateB
            0,  # rateC
            0,  # ratio
            0,  # angle
            1,  # status
            -360,  # angmin
            360    # angmax
        ]
        branch_data.append(branch_row)

    # 变压器数据（如果有）
    for idx, trafo in net.trafo.iterrows():
        # 简化的变压器模型
        branch_row = [
            trafo.hv_bus + 1,  # from bus (1-indexed)
            trafo.lv_bus + 1,  # to bus (1-indexed)
            trafo.vk_percent / 100 * (trafo.vn_hv_kv**2 / baseMVA),  # r (p.u.)
            trafo.vkr_percent / 100 * (trafo.vn_hv_kv**2 / baseMVA),  # x (p.u.)
            0,  # b (p.u.)
            trafo.sn_mva,  # rateA (MVA)
            0,  # rateB
            0,  # rateC
            trafo.vn_hv_kv / trafo.vn_lv_kv,  # ratio
            0,  # angle
            1,  # status
            -360,  # angmin
            360    # angmax
        ]
        branch_data.append(branch_row)

    return {
        'baseMVA': baseMVA,
        'bus': np.array(bus_data),
        'branch': np.array(branch_data)
    }


class TrainingLogger:
    """训练日志记录器 - 现在支持TensorBoard和W&B"""

    def __init__(self, config: Dict[str, Any], total_episodes: int):
        """初始化日志记录器"""
        self.config = config
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.success_rates = []
        self.load_cvs = []
        self.coupling_edges = []
        self.best_reward = -float('inf')

        # 智能自适应课程学习相关指标
        self.curriculum_stages = []
        self.stage_transitions = []
        self.director_decisions = []

        log_config = config.get('logging', {})
        self.metrics_save_interval = log_config.get('metrics_save_interval', 100)

        # 导入并设置 Rich 输出管理器
        try:
            from rich_output import set_output_manager, rich_progress
            set_output_manager(config)
            self.progress_bar = rich_progress("🚀 训练进度", total_episodes)
            self.progress_bar.__enter__()  # 启动进度条
            self.use_rich = True
        except ImportError:
            from tqdm import tqdm
            self.progress_bar = tqdm(total=total_episodes, desc="🚀 训练进度")
            self.use_rich = False

        # 设置TensorBoard
        self.use_tensorboard = log_config.get('use_tensorboard', False)
        self.tensorboard_writer = self._setup_tensorboard() if self.use_tensorboard else None

        # 设置Weights & Biases
        wandb_config = config.get('wandb', {})
        self.use_wandb = wandb_config.get('enabled', False)
        if self.use_wandb:
            self._setup_wandb(wandb_config)

    def _setup_wandb(self, wandb_config: Dict[str, Any]):
        """初始化Weights & Biases"""
        try:
            wandb.init(
                project=wandb_config.get('project', 'power-grid-partitioning'),
                entity=wandb_config.get('entity'), # entity可以为None
                config=self.config, # 记录所有超参数
                reinit=True
            )
            from rich_output import rich_success
            rich_success(f"W&B: 项目: {wandb.run.project}, 名称: {wandb.run.name}")
        except Exception as e:
            from rich_output import rich_error
            rich_error(f"W&B: 初始化失败: {e}")
            self.use_wandb = False

    def _setup_tensorboard(self):
        """设置TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.config['logging']['log_dir']
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            tensorboard_writer = SummaryWriter(f"{log_dir}/training_{timestamp}")
            from rich_output import rich_info
            rich_info(f"TensorBoard日志目录: {log_dir}/training_{timestamp}", show_always=True)
            return tensorboard_writer
        except ImportError:
            from rich_output import rich_warning
            rich_warning("TensorBoard不可用，跳过TensorBoard日志")
            self.use_tensorboard = False
            return None
        except Exception as e:
            from rich_output import rich_warning
            rich_warning(f"TensorBoard初始化失败: {e}")
            self.use_tensorboard = False
            return None

    def log_episode(self, episode: int, reward: float, length: int, info: Dict = None):
        """记录单个回合"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        if reward > self.best_reward:
            self.best_reward = reward

        # 更新进度条
        if self.use_rich:
            self.progress_bar.update(1, 奖励=f"{reward:.2f}", 最佳=f"{self.best_reward:.2f}")
        else:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix({
                "奖励": f"{reward:.2f}",
                "最佳": f"{self.best_reward:.2f}"
            })

        # 记录额外信息
        if info:
            if 'success' in info:
                self.success_rates.append(1.0 if info['success'] else 0.0)
            if 'load_cv' in info:
                self.load_cvs.append(info['load_cv'])
            if 'coupling_edges' in info:
                self.coupling_edges.append(info['coupling_edges'])

        # TensorBoard日志
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Episode/Reward', reward, episode)
            self.tensorboard_writer.add_scalar('Episode/Length', length, episode)
            if info:
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f'Episode/{key}', value, episode)
            # 定期刷新以确保数据写入
            if episode % 10 == 0:
                self.tensorboard_writer.flush()

        # W&B日志
        if self.use_wandb:
            log_data = {'Episode/Reward': reward, 'Episode/Length': length}
            if info:
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        log_data[f'Episode/{key}'] = value
            wandb.log(log_data, step=episode)

    def log_training_step(self, episode: int, actor_loss: float = None,
                         critic_loss: float = None, entropy: float = None):
        """记录训练步骤"""
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/ActorLoss', actor_loss, episode)

        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/CriticLoss', critic_loss, episode)

        if entropy is not None:
            self.entropies.append(entropy)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/Entropy', entropy, episode)

        # 刷新TensorBoard数据
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.flush()

        # W&B日志
        if self.use_wandb:
            log_data = {}
            if actor_loss is not None:
                log_data['Training/ActorLoss'] = actor_loss
            if critic_loss is not None:
                log_data['Training/CriticLoss'] = critic_loss
            if entropy is not None:
                log_data['Training/Entropy'] = entropy
            if log_data:
                wandb.log(log_data, step=episode)

    def get_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.episode_rewards:
            return {}

        stats = {
            'total_episodes': len(self.episode_rewards),
            'best_reward': self.best_reward,
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'training_time': time.time() - self.start_time
        }

        if self.success_rates:
            stats['success_rate'] = np.mean(self.success_rates)
        if self.load_cvs:
            stats['mean_load_cv'] = np.mean(self.load_cvs)
        if self.actor_losses:
            stats['mean_actor_loss'] = np.mean(self.actor_losses)
        if self.critic_losses:
            stats['mean_critic_loss'] = np.mean(self.critic_losses)

        return stats

    def close(self):
        """关闭日志记录器"""
        if self.use_rich:
            self.progress_bar.__exit__(None, None, None)
        else:
            self.progress_bar.close()
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        if self.use_wandb:
            wandb.finish()


class UnifiedTrainer:
    """统一训练器"""

    def __init__(self, agent, env, config, gym_env=None):
        self.agent = agent
        self.env = env
        self.config = config
        self.gym_env = gym_env  # 场景生成环境包装器（如果使用）
        # 处理设备配置
        device_config = config['system']['device']
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        self.logger = None # 将在train方法中初始化

    def train(self, num_episodes: int, max_steps_per_episode: int, update_interval: int = 10):
        """训练智能体"""
        self.logger = TrainingLogger(self.config, num_episodes)

        from rich_output import rich_info
        if not self.config.get('debug', {}).get('training_output', {}).get('only_show_errors', True):
            rich_info(f"TensorBoard: {'已启用' if self.logger.use_tensorboard else '已禁用'}")
            rich_info(f"指标保存间隔: {self.logger.metrics_save_interval} 回合")

        for episode in range(num_episodes):
            # 如果使用场景生成环境，需要重置Gym环境以生成新场景
            if self.gym_env is not None:
                obs_array, info = self.gym_env.reset()
                # 更新内部环境引用（因为每次重置都会创建新的内部环境）
                self.env = self.gym_env.internal_env
                state, _ = self.env.reset()  # 重置内部环境状态
            else:
                state, _ = self.env.reset()  # 标准环境重置

            episode_reward = 0
            episode_length = 0
            episode_info = {}

            for step in range(max_steps_per_episode):
                # 选择动作
                action, log_prob, value = self.agent.select_action(state)

                if action is None:  # 没有有效动作
                    break

                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 存储经验
                self.agent.store_experience(state, action, reward, log_prob, value, done)

                episode_reward += reward
                episode_length += 1
                state = next_state

                # 收集回合信息
                if info:
                    episode_info.update(info)

                if done:
                    break

            # 记录到logger（包含详细信息）
            self.logger.log_episode(episode, episode_reward, episode_length, episode_info)

            # 更新智能体并记录训练指标
            if episode % update_interval == 0 and episode > 0:
                try:
                    # --- 使用 try-except 包裹 agent.update() ---
                    training_stats = self.agent.update()
                    if training_stats:
                        self.logger.log_training_step(
                            episode,
                            training_stats.get('actor_loss'),
                            training_stats.get('critic_loss'),
                            training_stats.get('entropy')
                        )

                except RuntimeError as e:
                    if "[NaNGuard]" in str(e) or "NaN" in str(e):
                        print(f"\n❌ [NaN Dumper] 捕获到致命错误: {e}")

                        dump_dir = "diagnostics"
                        os.makedirs(dump_dir, exist_ok=True)
                        dump_path = os.path.join(dump_dir, f"nan_dump_episode_{episode}.pt")

                        print(f"  ... 正在保存诊断信息到: {dump_path}")

                        # 收集触发异常的批量数据
                        batch_data = {
                            'states': self.agent.memory.states,
                            'actions': self.agent.memory.actions,
                            'logprobs': self.agent.memory.logprobs,
                            'rewards': self.agent.memory.rewards,
                            'is_terminals': self.agent.memory.is_terminals,
                        }

                        torch.save({
                            "error_message": str(e),
                            "episode": episode,
                            "actor_state_dict": self.agent.actor.state_dict(),
                            "critic_state_dict": self.agent.critic.state_dict(),
                            "actor_optimizer_state_dict": self.agent.actor_optimizer.state_dict(),
                            "critic_optimizer_state_dict": self.agent.critic_optimizer.state_dict(),
                            "triggering_batch": batch_data,
                        }, dump_path)

                        print("  ... 保存完成。正在中断训练。")
                        # 重新抛出异常，以正常退出训练循环
                        raise e
                    else:
                        # 如果是其他运行时错误，则直接抛出
                        raise

            # 保存中间结果
            if episode % self.logger.metrics_save_interval == 0 and episode > 0:
                self._save_intermediate_results(episode)

        # 训练完成统计
        final_stats = self.logger.get_statistics()
        print(f"\n🎯 训练完成统计:")
        print(f"   - 总回合数: {final_stats.get('total_episodes', 0)}")
        print(f"   - 最佳奖励: {final_stats.get('best_reward', 0):.4f}")
        print(f"   - 平均奖励: {final_stats.get('mean_reward', 0):.4f}")
        print(f"   - 训练时间: {final_stats.get('training_time', 0)/60:.1f} 分钟")
        if 'success_rate' in final_stats:
            print(f"   - 成功率: {final_stats['success_rate']:.3f}")

        return {
            'episode_rewards': self.logger.episode_rewards,
            'episode_lengths': self.logger.episode_lengths,
            'training_stats': final_stats
        }

    def _save_intermediate_results(self, episode: int):
        """保存中间训练结果"""
        try:
            stats = self.logger.get_statistics()
            checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # 保存统计信息
            stats_file = checkpoint_dir / f"training_stats_episode_{episode}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"💾 中间结果已保存: episode {episode}")
        except Exception as e:
            print(f"⚠️ 保存中间结果失败: {e}")

    def evaluate(self, num_episodes: int = 10):
        """评估智能体"""
        print(f"🔍 评估智能体 {num_episodes} 轮...")

        eval_rewards = []
        success_count = 0

        for episode in range(num_episodes):
            state, _ = self.env.reset()  # 解包元组
            episode_reward = 0

            for step in range(200):  # 最大步数
                action, _, _ = self.agent.select_action(state, training=False)

                if action is None:  # 没有有效动作
                    break

                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if done:
                    if info.get('success', False):
                        success_count += 1
                    break

            eval_rewards.append(episode_reward)

        return {
            'avg_reward': np.mean(eval_rewards),
            'success_rate': success_count / num_episodes,
            'rewards': eval_rewards
        }

    def run_final_visualization(self):
        """运行最终可视化"""
        try:
            from visualization import VisualizationManager
            from rich_output import rich_success
            viz = VisualizationManager(self.config)
            viz.visualize_partition(self.env, title="Final Partition")
            rich_success("可视化完成")
        except Exception as e:
            from rich_output import rich_warning
            rich_warning(f"可视化失败: {e}")

    def close(self):
        """清理资源"""
        if self.logger:
            self.logger.close()


class UnifiedTrainingSystem:
    """统一训练系统 - 专注于训练功能"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化统一训练系统"""
        self.deps = check_dependencies()
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.setup_directories()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        # 如果没有指定配置文件，尝试使用默认的 config.yaml
        if not config_path:
            default_config_path = 'config.yaml'
            if os.path.exists(default_config_path):
                config_path = default_config_path
                from rich_output import rich_info
                rich_info(f"使用默认配置文件: {config_path}")

        # 检查是否是文件路径
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                from rich_output import rich_success, rich_info
                rich_success(f"配置文件加载成功: {config_path}")
                rich_info(f"案例名称: {config['data']['case_name']}", show_always=True)
                return config

        # 检查是否是预设配置名称
        elif config_path and os.path.exists('config.yaml'):
            with open('config.yaml', 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)

                # 检查是否存在预设配置
                if config_path in base_config:
                    from rich_output import rich_success, rich_info
                    rich_success(f"使用预设配置: {config_path}")
                    preset_config = base_config[config_path]

                    # 深度合并预设配置到基础配置
                    merged_config = self._deep_merge_config(base_config, preset_config)
                    rich_info(f"案例名称: {merged_config['data']['case_name']}", show_always=True)
                    return merged_config
                else:
                    from rich_output import rich_warning
                    rich_warning(f"未找到预设配置 '{config_path}'，使用默认配置")
                    return base_config
        else:
            from rich_output import rich_warning
            rich_warning("未找到配置文件，使用默认配置")
            return self._create_default_config()

    def _deep_merge_config(self, base_config: Dict[str, Any], preset_config: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置字典"""
        result = base_config.copy()

        for key, value in preset_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        return {
            'system': {
                'name': 'unified_power_grid_partitioning',
                'version': '2.0',
                'device': 'auto',
                'seed': 42,
                'num_threads': 1
            },
            'data': {
                'case_name': 'ieee14',
                'normalize': True,
                'cache_dir': 'cache'
            },
            'training': {
                'mode': 'standard',  # standard, parallel, curriculum, large_scale
                'num_episodes': 1000,
                'max_steps_per_episode': 200,
                'update_interval': 10,
                'save_interval': 100,
                'eval_interval': 50
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
                'type': 'ppo',  # ppo, sb3_ppo
                'lr_actor': 3e-5,  # 降低10倍用于数值稳定
                'lr_critic': 1e-4,  # 降低10倍用于数值稳定
                'gamma': 0.99,
                'eps_clip': 0.2,
                'k_epochs': 4,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'hidden_dim': 256
            },
            'parallel_training': {
                'enabled': False,
                'num_cpus': 12,
                'total_timesteps': 5_000_000,
                'scenario_generation': True
            },
            'scenario_generation': {
                'enabled': True,  # 默认启用场景生成
                'perturb_prob': 0.8,
                'perturb_types': ['n-1', 'load_gen_fluctuation', 'both', 'none'],
                'scale_range': [0.8, 1.2]
            },
            'curriculum': {
                'enabled': False,
                'start_partitions': 2,
                'end_partitions': 5,
                'episodes_per_stage': 200
            },
            'evaluation': {
                'num_episodes': 20,
                'include_baselines': True,
                'baseline_methods': ['spectral', 'kmeans', 'random']
            },
            'visualization': {
                'enabled': True,
                'save_figures': True,
                'figures_dir': 'figures',
                'interactive': True
            },
            'logging': {
                'use_tensorboard': True,
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',
                'console_log_interval': 10,
                'metrics_save_interval': 50
            }
        }

    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_config = self.config['system'].get('device', 'auto')
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)

        print(f"🔧 使用设备: {device}")
        return device

    def setup_directories(self):
        """创建必要的目录"""
        dirs = [
            self.config['data']['cache_dir'],
            self.config['logging']['log_dir'],
            self.config['logging']['checkpoint_dir'],
            self.config['visualization']['figures_dir'],
            'models', 'output', 'experiments'
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_training_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取不同训练模式的配置"""
        base_config = self.config.copy()

        configs = {
            'fast': {
                **base_config,
                'training': {
                    **base_config['training'],
                    'mode': 'fast',
                    'num_episodes': 1500
                },
                'parallel_training': {
                    **base_config['parallel_training'],
                    'enabled': False
                },
                'scenario_generation': {
                    **base_config['scenario_generation'],
                    'enabled': True
                }
            },
            'full': {
                **base_config,
                'training': {
                    **base_config['training'],
                    'num_episodes': 5000,
                    'max_steps_per_episode': 500,
                    'update_interval': 20
                },
                'parallel_training': {
                    **base_config['parallel_training'],
                    'enabled': True
                },
                'scenario_generation': {
                    **base_config['scenario_generation'],
                    'enabled': True
                }
            },
            'ieee118': {
                **base_config,
                'data': {
                    **base_config['data'],
                    'case_name': 'ieee118'
                },
                'environment': {
                    **base_config['environment'],
                    'num_partitions': 8,
                    'max_steps': 500
                },
                'gat': {
                    **base_config['gat'],
                    'hidden_channels': 128,
                    'gnn_layers': 4,
                    'heads': 8,
                    'output_dim': 256
                },
                'training': {
                    **base_config['training'],
                    'num_episodes': 3000,
                    'max_steps_per_episode': 500
                },
                'parallel_training': {
                    **base_config['parallel_training'],
                    'enabled': True
                },
                'scenario_generation': {
                    **base_config['scenario_generation'],
                    'enabled': True
                }
            },
            'parallel': {
                **base_config,
                'parallel_training': {
                    **base_config['parallel_training'],
                    'enabled': True
                },
                'scenario_generation': {
                    **base_config['scenario_generation'],
                    'enabled': True
                }
            },
            'curriculum': {
                **base_config,
                'curriculum': {
                    **base_config['curriculum'],
                    'enabled': True
                }
            },
            'adaptive': {
                **base_config,
                'training': {
                    **base_config['training'],
                    'num_episodes': 2000,
                    'max_steps_per_episode': 200,
                    'success_criteria': {
                        **base_config['training'].get('success_criteria', {}),
                        'load_cv_threshold': 0.25,
                        'connectivity_threshold': 0.95
                    }
                },
                'adaptive_curriculum': {
                    **base_config.get('adaptive_curriculum', {}),
                    'enabled': True
                },
                'parallel_training': {
                    **base_config['parallel_training'],
                    'enabled': False
                },
                'scenario_generation': {
                    **base_config['scenario_generation'],
                    'enabled': True
                }
            }
        }

        return configs

    def run_training(self, mode: str = 'standard', **kwargs) -> Dict[str, Any]:
        """运行训练"""
        print(f"\n🚀 开始{mode.upper()}模式训练")
        print("=" * 60)

        # 获取模式配置
        configs = self.get_training_configs()
        if mode not in configs:
            print(f"⚠️ 未知训练模式: {mode}，使用快速模式")
            mode = 'fast'

        config = configs[mode]

        # 应用命令行参数覆盖
        for key, value in kwargs.items():
            if '.' in key:
                # 支持嵌套配置，如 training.num_episodes
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value

        # 设置随机种子
        torch.manual_seed(config['system']['seed'])
        np.random.seed(config['system']['seed'])

        try:
            if mode == 'parallel' or config['parallel_training']['enabled']:
                return self._run_parallel_training(config)
            elif mode == 'curriculum' or config['curriculum']['enabled']:
                return self._run_curriculum_training(config)
            elif config.get('adaptive_curriculum', {}).get('enabled', False):
                return self._run_curriculum_training(config)  # 智能自适应也使用课程学习流程
            else:
                return self._run_standard_training(config)

        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _run_standard_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行标准训练"""
        print("📊 标准训练模式")

        # 导入必要模块
        from data_processing import PowerGridDataProcessor
        from gat import create_hetero_graph_encoder
        from rl.environment import PowerGridPartitioningEnv
        from rl.agent import PPOAgent

        # 1. 数据处理
        print("\n1️⃣ 数据处理...")
        processor = PowerGridDataProcessor(
            normalize=config['data']['normalize'],
            cache_dir=config['data']['cache_dir']
        )

        # 加载数据
        mpc = load_power_grid_data(config['data']['case_name'])

        hetero_data = processor.graph_from_mpc(mpc, config).to(self.device)
        print(f"✅ 数据加载完成: {hetero_data}")

        # 2. GAT编码器
        print("\n2️⃣ GAT编码器...")
        gat_config = config['gat']
        encoder = create_hetero_graph_encoder(
            hetero_data,
            hidden_channels=gat_config['hidden_channels'],
            gnn_layers=gat_config['gnn_layers'],
            heads=gat_config['heads'],
            output_dim=gat_config['output_dim']
        ).to(self.device)

        with torch.no_grad():
            node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data, config)

        print(f"✅ 编码器初始化完成")

        # 3. 环境（支持场景生成）
        print("\n3️⃣ 强化学习环境...")
        env_config = config['environment']
        scenario_config = config.get('scenario_generation', {})
        use_scenario_generation = scenario_config.get('enabled', True)  # 默认启用场景生成

        if use_scenario_generation:
            print("🎭 启用场景生成功能...")
            # 使用支持场景生成的Gym环境包装器
            try:
                from rl.gym_wrapper import PowerGridPartitionGymEnv

                # 确保设备配置正确传递
                config_copy = config.copy()
                config_copy['system']['device'] = str(self.device)  # 转换为字符串

                gym_env = PowerGridPartitionGymEnv(
                    base_case_data=mpc,
                    config=config_copy,
                    use_scenario_generator=True,
                    scenario_seed=config['system']['seed']
                )

                # 重置环境以获取初始状态
                obs_array, info = gym_env.reset()
                env = gym_env.internal_env  # 获取内部的PowerGridPartitioningEnv

                print(f"✅ 场景生成环境创建完成: {env.total_nodes}节点, {env.num_partitions}分区")

            except ImportError as e:
                print(f"⚠️ 场景生成模块导入失败: {e}")
                print("🔄 回退到标准环境...")
                use_scenario_generation = False

        if not use_scenario_generation:
            print("📊 使用标准环境（无场景生成）...")
            env = PowerGridPartitioningEnv(
                hetero_data=hetero_data,
                node_embeddings=node_embeddings,
                num_partitions=env_config['num_partitions'],
                reward_weights=env_config['reward_weights'],
                max_steps=env_config['max_steps'],
                device=self.device,
                attention_weights=attention_weights,
                config=config
            )

            print(f"✅ 标准环境创建完成: {env.total_nodes}节点, {env.num_partitions}分区")

        # 4. 智能体
        print("\n4️⃣ PPO智能体...")
        agent_config = config['agent']
        node_embedding_dim = env.state_manager.embedding_dim
        region_embedding_dim = node_embedding_dim * 2

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
            device=self.device,
            max_grad_norm=agent_config.get('max_grad_norm', None),
            # 【修改】传递独立的调度器配置
            actor_scheduler_config=agent_config.get('actor_scheduler', {}),
            critic_scheduler_config=agent_config.get('critic_scheduler', {})
        )

        print(f"✅ 智能体创建完成")

        # 5. 训练
        print("\n5️⃣ 开始训练...")
        # 如果使用了场景生成，传递gym_env给训练器
        gym_env_ref = gym_env if use_scenario_generation and 'gym_env' in locals() else None
        trainer = UnifiedTrainer(agent=agent, env=env, config=config, gym_env=gym_env_ref)

        training_config = config['training']
        history = trainer.train(
            num_episodes=training_config['num_episodes'],
            max_steps_per_episode=training_config['max_steps_per_episode'],
            update_interval=training_config['update_interval']
        )

        # 6. 评估
        print("\n6️⃣ 评估...")
        eval_stats = trainer.evaluate()

        # 7. 基线对比
        baseline_results = None
        if config['evaluation']['include_baselines']:
            print("\n7️⃣ 基线方法对比...")
            try:
                from baseline import run_baseline_comparison
                baseline_results = run_baseline_comparison(env, agent, seed=42)
                print("✅ 基线对比完成")
            except Exception as e:
                print(f"⚠️ 基线对比失败: {e}")

        # 8. 可视化
        if config['visualization']['enabled']:
            print("\n8️⃣ 生成可视化...")
            try:
                trainer.run_final_visualization()
                if baseline_results is not None and config['visualization']['interactive']:
                    from visualization import run_interactive_visualization
                    run_interactive_visualization(env, baseline_results)
            except Exception as e:
                print(f"⚠️ 可视化失败: {e}")

        trainer.close()

        return {
            'success': True,
            'mode': 'standard',
            'config': config,
            'history': history,
            'eval_stats': eval_stats,
            'baseline_results': baseline_results,
            'best_reward': trainer.logger.best_reward
        }

    def _run_parallel_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行并行训练"""
        print("🌐 并行训练模式")

        if not self.deps['stable_baselines3']:
            print("❌ 并行训练需要stable-baselines3，请安装：pip install stable-baselines3")
            return {'success': False, 'error': 'Missing stable-baselines3'}

        try:
            # 检查是否有gym和stable_baselines3
            try:
                import gym
                from stable_baselines3 import PPO
                has_sb3 = True
            except ImportError:
                has_sb3 = False

            if has_sb3:
                # 使用Stable-Baselines3的并行训练
                return self._run_sb3_parallel_training(config)
            else:
                print("⚠️ 缺少stable-baselines3，跳过并行训练")
                return {'success': False, 'error': 'Missing stable-baselines3'}

        except Exception as e:
            print(f"❌ 并行训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _run_sb3_parallel_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """使用Stable-Baselines3的并行训练"""
        from data_processing import PowerGridDataProcessor
        from rl.gym_wrapper import make_parallel_env
        from stable_baselines3 import PPO

        # 【修复】在创建并行环境前，将主进程中已解析好的设备名称更新到配置字典中
        # 避免子进程收到 "auto" 字符串导致 torch.device() 报错
        config['system']['device'] = str(self.device)
        print(f"🔧 并行训练设备配置已更新: {config['system']['device']}")

        # 1. 数据处理
        print("\n1️⃣ 数据处理...")
        processor = PowerGridDataProcessor(
            normalize=config['data']['normalize'],
            cache_dir=config['data']['cache_dir']
        )

        # 加载数据
        mpc = load_power_grid_data(config['data']['case_name'])

        # 2. 创建并行环境
        print("\n2️⃣ 创建并行环境...")
        parallel_config = config['parallel_training']

        parallel_env = make_parallel_env(
            base_case_data=mpc,
            config=config,
            num_envs=parallel_config['num_cpus'],
            use_scenario_generator=parallel_config['scenario_generation']
        )

        # 3. 创建并训练PPO智能体
        print("\n3️⃣ 创建PPO智能体...")
        model = PPO(
            "MlpPolicy",
            parallel_env,
            verbose=1,
            tensorboard_log=config['logging']['log_dir']
        )

        print("\n4️⃣ 开始并行训练...")
        model.learn(total_timesteps=parallel_config['total_timesteps'])

        # 4. 保存模型
        model_path = Path(config['logging']['checkpoint_dir']) / "parallel_model"
        model_path.parent.mkdir(exist_ok=True, parents=True)
        model.save(str(model_path))

        return {
            'success': True,
            'mode': 'parallel',
            'model_path': str(model_path),
            'total_timesteps': parallel_config['total_timesteps'],
            'config': config
        }

    def _run_curriculum_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行课程学习训练"""
        # 检查是否启用智能自适应课程学习
        if config.get('adaptive_curriculum', {}).get('enabled', False):
            return self._run_adaptive_curriculum_training(config)
        else:
            return self._run_traditional_curriculum_training(config)

    def _run_traditional_curriculum_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行传统课程学习训练"""
        print("📚 传统课程学习训练模式")

        curriculum_config = config['curriculum']
        start_partitions = curriculum_config['start_partitions']
        end_partitions = curriculum_config['end_partitions']
        episodes_per_stage = curriculum_config['episodes_per_stage']

        results = []

        for num_partitions in range(start_partitions, end_partitions + 1):
            print(f"\n📖 课程阶段: {num_partitions}个分区")

            # 更新配置
            stage_config = config.copy()
            stage_config['environment']['num_partitions'] = num_partitions
            stage_config['training']['num_episodes'] = episodes_per_stage

            # 运行该阶段的训练
            stage_result = self._run_standard_training(stage_config)
            results.append(stage_result)

            if not stage_result['success']:
                break

        return {
            'success': all(r['success'] for r in results),
            'mode': 'traditional_curriculum',
            'config': config,
            'stage_results': results
        }

    def _run_adaptive_curriculum_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行智能自适应课程学习训练"""
        print("🧠 智能自适应课程学习训练模式")
        print("=" * 60)

        try:
            # 导入智能导演系统
            from code.src.rl.adaptive import AdaptiveDirector

            # 获取基础训练模式
            base_mode = self._detect_base_mode(config)

            # 初始化智能导演
            director = AdaptiveDirector(config, base_mode)
            print(f"✅ 智能导演系统已初始化 (基础模式: {base_mode})")

            # 运行自适应训练
            result = self._run_adaptive_training_with_director(config, director)

            return {
                'success': result.get('success', False),
                'mode': 'adaptive_curriculum',
                'config': config,
                'director_status': director.get_status_summary(),
                **result
            }

        except ImportError as e:
            print(f"❌ 无法导入智能导演系统: {e}")
            print("🔄 回退到传统课程学习模式")
            return self._run_traditional_curriculum_training(config)
        except Exception as e:
            print(f"❌ 智能自适应课程学习失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e), 'mode': 'adaptive_curriculum'}

    def _detect_base_mode(self, config: Dict[str, Any]) -> str:
        """检测基础训练模式"""
        # 根据配置特征检测基础模式
        num_episodes = config.get('training', {}).get('num_episodes', 1500)
        parallel_enabled = config.get('parallel_training', {}).get('enabled', False)

        if num_episodes >= 4000 or parallel_enabled:
            if num_episodes >= 4000:
                return 'ieee118'
            else:
                return 'full'
        else:
            return 'fast'

    def _run_adaptive_training_with_director(self, config: Dict[str, Any], director) -> Dict[str, Any]:
        """使用智能导演运行自适应训练"""
        print("\n🎬 启动智能导演训练流程...")

        # 1. 环境和智能体初始化
        print("\n1️⃣ 初始化环境和智能体...")
        mpc = load_power_grid_data(config['data']['case_name'])

        # 创建环境（使用场景生成）
        if config['scenario_generation']['enabled']:
            from rl.gym_wrapper import PowerGridPartitionGymEnv

            # 确保设备配置正确传递
            config_copy = config.copy()
            config_copy['system']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

            gym_env = PowerGridPartitionGymEnv(
                base_case_data=mpc,
                config=config_copy,
                use_scenario_generator=True,
                scenario_seed=config['system']['seed']
            )
            # 重置环境以获取初始状态
            obs_array, info = gym_env.reset()
            env = gym_env.internal_env
        else:
            from rl.environment import PowerGridPartitioningEnv
            env = PowerGridPartitioningEnv(mpc, config)
            gym_env = None

        # 创建智能体
        from rl.agent import PPOAgent

        # 获取正确的嵌入维度
        node_embedding_dim = env.state_manager.embedding_dim
        region_embedding_dim = node_embedding_dim * 2

        agent = PPOAgent(
            node_embedding_dim=node_embedding_dim,
            region_embedding_dim=region_embedding_dim,
            num_partitions=env.num_partitions,
            lr_actor=config['agent']['lr_actor'],
            lr_critic=config['agent']['lr_critic'],
            gamma=config['agent']['gamma'],
            eps_clip=config['agent']['eps_clip'],
            k_epochs=config['agent']['k_epochs'],
            entropy_coef=config['agent']['entropy_coef'],
            value_coef=config['agent']['value_coef'],
            device=env.device,
            actor_scheduler_config=config['agent'].get('actor_scheduler'),
            critic_scheduler_config=config['agent'].get('critic_scheduler')
        )

        # 2. 智能自适应训练循环
        print("\n2️⃣ 开始智能自适应训练...")
        num_episodes = config['training']['num_episodes']
        max_steps_per_episode = config['training']['max_steps_per_episode']
        update_interval = config['training']['update_interval']

        # 初始化训练日志
        logger = TrainingLogger(config, num_episodes)

        # 智能导演状态跟踪
        director_decisions = []
        stage_transitions = []

        # 创建实时状态表格
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.live import Live
            from rich.panel import Panel
            from rich.columns import Columns
            from rich.text import Text
            console = Console()
            use_rich_table = True
        except ImportError:
            use_rich_table = False

        def create_status_table(episode, director_decision, episode_reward, best_reward):
            """创建智能导演状态表格"""
            if not use_rich_table:
                return None

            # 主状态表格
            base_mode = 'unknown'
            if director_decision and 'stage_info' in director_decision:
                base_mode = director_decision['stage_info'].get('base_mode', 'unknown')
            table = Table(title=f"🧠 智能自适应训练状态 ({base_mode.upper()})", show_header=True, header_style="bold magenta")
            table.add_column("指标", style="cyan", width=15)
            table.add_column("当前值", style="green", width=20)
            table.add_column("说明", style="yellow", width=25)

            # 基础信息
            table.add_row("Episode", f"{episode}/{num_episodes}", "当前训练进度")
            table.add_row("当前奖励", f"{episode_reward:.3f}", "本轮episode奖励")
            table.add_row("最佳奖励", f"{best_reward:.3f}", "历史最佳奖励")

            if director_decision and 'stage_info' in director_decision:
                stage_info = director_decision['stage_info']
                stage_name = stage_info['stage_name']
                stage_progress = stage_info['stage_progress']

                # 阶段信息
                stage_emoji = {
                    'exploration': '🔍',
                    'transition': '🔄',
                    'refinement': '⚡',
                    'fine_tuning': '🎯',
                    'emergency_recovery': '🚨'
                }.get(stage_name, '❓')

                table.add_row("当前阶段", f"{stage_emoji} {stage_name}", f"进度: {stage_progress:.1%}")

                # 参数信息
                if 'reward_weights' in director_decision:
                    weights = director_decision['reward_weights']
                    balance_w = weights.get('balance_weight', 0)
                    decoupling_w = weights.get('decoupling_weight', 0)
                    power_w = weights.get('power_weight', 0)
                    table.add_row("奖励权重", f"B:{balance_w:.1f} D:{decoupling_w:.1f} P:{power_w:.1f}", "平衡/解耦/功率")

                if 'learning_rate_factor' in director_decision:
                    lr_factor = director_decision['learning_rate_factor']
                    table.add_row("学习率因子", f"{lr_factor:.3f}", "相对于基础学习率")

            # 统计信息
            emergency_count = sum(1 for t in stage_transitions if t.get('stage_name') == 'emergency_recovery')
            normal_count = len(stage_transitions) - emergency_count
            table.add_row("阶段转换", f"正常:{normal_count} 紧急:{emergency_count}", "智能转换统计")

            return Panel(table, border_style="blue")

        # 初始化Live显示
        if use_rich_table:
            initial_table = create_status_table(0, None, 0.0, -float('inf'))
            live = Live(initial_table, console=console, refresh_per_second=2)
            live.start()

        try:
            for episode in range(num_episodes):
                # 重置环境
                if gym_env is not None:
                    obs_array, info = gym_env.reset()
                    env = gym_env.internal_env
                    state, _ = env.reset()
                else:
                    state, _ = env.reset()

                episode_reward = 0
                episode_length = 0

                # Episode执行
                for step in range(max_steps_per_episode):
                    action, log_prob, value = agent.select_action(state, training=True)

                    if action is None:
                        break

                    state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    agent.store_experience(state, action, reward, log_prob, value, done)
                    episode_reward += reward
                    episode_length += 1

                    if done:
                        break

                # 收集episode信息
                episode_info = {
                    'episode': episode,
                    'reward': episode_reward,
                    'episode_length': episode_length,
                    'success': info.get('success', False),
                    'load_cv': info.get('load_cv', 1.0),
                    'coupling_ratio': info.get('coupling_ratio', 1.0),
                    'connectivity': info.get('connectivity', 0.0),
                    **info
                }

                # 智能导演决策
                director_decision = director.step(episode, episode_info)
                director_decisions.append(director_decision)

                # 应用智能导演的参数调整
                self._apply_director_decision(env, agent, director_decision)

                # 记录阶段转换
                if 'stage_info' in director_decision:
                    stage_info = director_decision['stage_info']
                    if len(stage_transitions) == 0 or stage_transitions[-1]['stage'] != stage_info['current_stage']:
                        stage_transitions.append({
                            'episode': episode,
                            'stage': stage_info['current_stage'],
                            'stage_name': stage_info['stage_name']
                        })

                # 更新实时表格
                if use_rich_table and episode % 2 == 0:  # 每2个episode更新一次表格
                    current_best = max([r for r in logger.episode_rewards if r is not None] + [-float('inf')])
                    updated_table = create_status_table(episode, director_decision, episode_reward, current_best)
                    live.update(updated_table)

                # 记录训练日志
                logger.log_episode(episode, episode_reward, episode_length, episode_info)

                # 智能体更新
                if episode % update_interval == 0 and episode > 0:
                    try:
                        training_stats = agent.update()
                        if training_stats:
                            logger.log_training_step(
                                episode,
                                training_stats.get('actor_loss'),
                                training_stats.get('critic_loss'),
                                training_stats.get('entropy')
                            )
                    except Exception as e:
                        print(f"⚠️ Episode {episode} 智能体更新失败: {e}")

                # 定期保存中间结果
                if episode % config['training']['save_interval'] == 0 and episode > 0:
                    self._save_adaptive_intermediate_results(episode, director, logger)

            # 训练完成统计
            final_stats = logger.get_statistics()
            director_summary = director.get_status_summary()

            # 关闭Live显示
            if use_rich_table:
                live.stop()

            # 统计阶段转换信息
            emergency_transitions = sum(1 for t in stage_transitions if t['stage_name'] == 'emergency_recovery')
            normal_transitions = len(stage_transitions) - emergency_transitions

            # 创建最终结果表格
            if use_rich_table:
                final_table = Table(title="🎯 智能自适应训练完成", show_header=True, header_style="bold green")
                final_table.add_column("指标", style="cyan", width=15)
                final_table.add_column("结果", style="green", width=20)

                final_table.add_row("总回合数", str(final_stats.get('total_episodes', 0)))
                final_table.add_row("最佳奖励", f"{final_stats.get('best_reward', 0):.4f}")
                final_table.add_row("平均奖励", f"{final_stats.get('mean_reward', 0):.4f}")
                final_table.add_row("最终阶段", str(director_summary['current_stage']))
                final_table.add_row("智能转换", f"{normal_transitions}次正常 + {emergency_transitions}次紧急恢复")

                console.print(Panel(final_table, border_style="green"))
            else:
                print(f"\n🎯 智能自适应训练完成:")
                print(f"   - 总回合数: {final_stats.get('total_episodes', 0)}")
                print(f"   - 最佳奖励: {final_stats.get('best_reward', 0):.4f}")
                print(f"   - 平均奖励: {final_stats.get('mean_reward', 0):.4f}")
                print(f"   - 最终阶段: {director_summary['current_stage']}")
                print(f"   - 智能转换: {normal_transitions}次正常 + {emergency_transitions}次紧急恢复")

            return {
                'success': True,
                'episode_rewards': logger.episode_rewards,
                'episode_lengths': logger.episode_lengths,
                'training_stats': final_stats,
                'director_decisions': director_decisions,
                'stage_transitions': stage_transitions,
                'director_summary': director_summary
            }

        except Exception as e:
            # 确保关闭Live显示
            if use_rich_table and 'live' in locals():
                live.stop()
            print(f"❌ 智能自适应训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

        finally:
            # 清理资源
            if use_rich_table and 'live' in locals():
                try:
                    live.stop()
                except:
                    pass
            if hasattr(logger, 'progress_bar'):
                logger.progress_bar.__exit__(None, None, None)

    def _apply_director_decision(self, env, agent, decision: Dict[str, Any]):
        """应用智能导演的决策到环境和智能体"""
        import builtins

        try:
            # 静默模式：减少日志输出
            verbose = self.config.get('debug', {}).get('adaptive_curriculum_verbose', False)

            # 更新环境参数
            if hasattr(env, 'reward_function') and 'reward_weights' in decision:
                reward_weights = decision['reward_weights']
                if hasattr(env.reward_function, 'update_weights'):
                    if verbose:
                        env.reward_function.update_weights(reward_weights)
                    else:
                        # 临时禁用打印
                        original_print = builtins.print
                        builtins.print = lambda *args, **kwargs: None
                        env.reward_function.update_weights(reward_weights)
                        builtins.print = original_print

            # 更新连通性惩罚
            if 'connectivity_penalty' in decision:
                if hasattr(env, 'connectivity_penalty'):
                    env.connectivity_penalty = decision['connectivity_penalty']

            # 更新智能体学习率
            if 'learning_rate_factor' in decision and decision['learning_rate_factor'] != 1.0:
                if hasattr(agent, 'update_learning_rate'):
                    if verbose:
                        agent.update_learning_rate(decision['learning_rate_factor'])
                    else:
                        # 临时禁用打印
                        original_print = builtins.print
                        builtins.print = lambda *args, **kwargs: None
                        agent.update_learning_rate(decision['learning_rate_factor'])
                        builtins.print = original_print

        except Exception as e:
            builtins.print(f"⚠️ 应用导演决策时出错: {e}")

    def _save_adaptive_intermediate_results(self, episode: int, director, logger):
        """保存智能自适应训练的中间结果"""
        try:
            from pathlib import Path
            import json

            checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # 保存训练统计
            stats = logger.get_statistics()
            stats_file = checkpoint_dir / f"adaptive_training_stats_episode_{episode}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            # 保存智能导演状态
            director_status = director.get_status_summary()
            director_file = checkpoint_dir / f"director_status_episode_{episode}.json"
            with open(director_file, 'w') as f:
                json.dump(director_status, f, indent=2)

            print(f"💾 智能自适应中间结果已保存: episode {episode}")

        except Exception as e:
            print(f"⚠️ 保存智能自适应中间结果失败: {e}")

    def save_results(self, results: Dict[str, Any], output_dir: str = 'experiments'):
        """保存训练结果"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        exp_dir = Path(output_dir) / f"training_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果
        results_file = exp_dir / "results.json"
        with open(results_file, 'w') as f:
            # 过滤不能序列化的对象
            serializable_results = {}
            for key, value in results.items():
                if key not in ['env', 'agent', 'trainer']:
                    try:
                        json.dumps(value)
                        serializable_results[key] = value
                    except (TypeError, ValueError):
                        serializable_results[key] = str(value)

            json.dump(serializable_results, f, indent=2)

        print(f"💾 训练结果已保存到: {exp_dir}")
        return exp_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='电力网络分区强化学习训练系统')

    # 基础参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径或预设配置名称')
    parser.add_argument('--mode', type=str, default='fast',
                       choices=['fast', 'full', 'ieee118', 'parallel', 'curriculum'],
                       help='训练模式')
    parser.add_argument('-a', '--adaptive', action='store_true',
                       help='启用智能自适应训练 (可与任何模式组合)')

    # 训练参数
    parser.add_argument('--episodes', type=int, help='训练回合数')
    parser.add_argument('--case', type=str, help='电网案例名称')
    parser.add_argument('--partitions', type=int, help='分区数量')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--workers', type=int, help='并行工作进程数')

    # 输出参数
    parser.add_argument('--save-results', action='store_true', help='保存训练结果')
    parser.add_argument('--output-dir', type=str, default='experiments', help='输出目录')

    # 系统参数
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], help='计算设备')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--check-deps', action='store_true', help='检查依赖')

    args = parser.parse_args()

    # 检查依赖
    if args.check_deps:
        deps = check_dependencies()
        print("📦 依赖检查结果:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"   - {dep}: {status}")
        return

    # 创建训练系统
    try:
        system = UnifiedTrainingSystem(config_path=args.config)

        # 准备训练参数
        train_kwargs = {}
        if args.episodes:
            train_kwargs['training.num_episodes'] = args.episodes
        if args.case:
            train_kwargs['data.case_name'] = args.case
        if args.partitions:
            train_kwargs['environment.num_partitions'] = args.partitions
        if args.lr:
            train_kwargs['agent.lr_actor'] = args.lr
            train_kwargs['agent.lr_critic'] = args.lr * 2
        if args.workers:
            train_kwargs['parallel_training.num_cpus'] = args.workers
        if args.device:
            train_kwargs['system.device'] = args.device
        if args.seed:
            train_kwargs['system.seed'] = args.seed

        # 处理智能自适应参数
        if args.adaptive:
            train_kwargs['adaptive_curriculum.enabled'] = True
            print(f"🧠 启用智能自适应训练 (基础模式: {args.mode})")

        # 运行训练
        results = system.run_training(mode=args.mode, **train_kwargs)

        # 保存结果
        if args.save_results and results.get('success', False):
            system.save_results(results, args.output_dir)

        # 输出结果摘要
        if results.get('success', False):
            print(f"\n🎉 训练成功完成!")
            if 'best_reward' in results:
                print(f"🏆 最佳奖励: {results['best_reward']:.4f}")
            if 'eval_stats' in results:
                eval_stats = results['eval_stats']
                print(f"📊 评估结果: 平均奖励 {eval_stats.get('avg_reward', 0):.4f}, "
                      f"成功率 {eval_stats.get('success_rate', 0):.3f}")
        else:
            print(f"\n❌ 训练失败: {results.get('error', '未知错误')}")
            return 1

    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
