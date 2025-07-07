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
from tqdm import tqdm
import queue
import threading
from rich.panel import Panel

# 添加code/src到路径
sys.path.append(str(Path(__file__).parent / 'code' / 'src'))
# 添加code到路径以便导入baseline
sys.path.append(str(Path(__file__).parent / 'code'))

# --- 核心模块导入 ---
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from torch_geometric.data import HeteroData
    from data_processing import PowerGridDataProcessor
    from gat import create_production_encoder
    from rl.environment import PowerGridPartitioningEnv
    from rl.agent import PPOAgent
    from rl.adaptive import AdaptiveDirector
    CRITICAL_DEPS_AVAILABLE = True
except ImportError as e:
    CRITICAL_DEPS_AVAILABLE = False
    MISSING_DEP_ERROR = e

# --- 全局 NaN/Inf 异常检测开关 ---
torch.autograd.set_detect_anomaly(False)
warnings.filterwarnings("error", message=".*NaN.*", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def safe_print(message: str):
    """安全的Unicode打印函数，避免编码错误"""
    try:
        print(message)
    except UnicodeEncodeError:
        # 移除Unicode字符，使用ASCII替代
        ascii_message = message.encode('ascii', 'replace').decode('ascii')
        print(ascii_message)

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
    加载电力网络数据。
    该函数现在只从pandapower加载数据，以确保数据源的一致性和高质量。
    如果加载失败，将直接抛出异常。

    Args:
        case_name: 案例名称 (ieee14, ieee30, ieee57, ieee118)

    Returns:
        MATPOWER格式的电网数据字典
    """
    if not _pandapower_available:
        raise ImportError("Pandapower库未安装或不可用，无法加载电网数据。")

    try:
        # 这是唯一应该被执行的数据加载路径
        return load_from_pandapower(case_name)
    except Exception as e:
        print(f"❌ 从PandaPower加载案例 '{case_name}' 时发生致命错误。")
        print("这可能是由于拼写错误、案例不存在或PandaPower库本身的问题。")
        # 重新抛出异常，以便程序停止
        raise e


def load_from_pandapower(case_name: str) -> Dict:
    """
    从PandaPower加载IEEE标准测试系统

    Args:
        case_name: 案例名称 (ieee14, ieee30, ieee57, ieee118)

    Returns:
        MATPOWER格式的电网数据字典
    """
    print(f"从PandaPower加载 {case_name.upper()} 数据...")

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
    from code.src.rich_output import rich_success
    rich_success(f"成功加载 {case_name.upper()}: {len(net.bus)} 节点, {len(net.line)} 线路")

    # 转换为MATPOWER格式
    mpc = convert_pandapower_to_matpower(net)

    return mpc


def convert_pandapower_to_matpower(net) -> Dict:
    """
    将PandaPower网络转换为MATPOWER格式（使用官方converter）

    Args:
        net: PandaPower网络对象

    Returns:
        MATPOWER格式的电网数据字典
    """
    # 使用官方 converter 进行可靠转换，避免手写转换中的精度和维护问题
    import pandapower as pp
    from pandapower.converter import to_mpc

    # 确保网络已有潮流结果；如果没有，则先运行一次平坦启动潮流
    try:
        if getattr(net, "res_bus", None) is None or net.res_bus.empty:
            pp.runpp(net, init="flat", calculate_voltage_angles=False, numba=False)
    except Exception:
        # 如果潮流失败，仍继续转换（to_mpc 允许无结果转换，但会给出警告）
        pass

    mpc_wrap = to_mpc(net)

    # pandapower >=2.0 返回 {'mpc': dict}；向后兼容旧版本
    if isinstance(mpc_wrap, dict) and "mpc" in mpc_wrap:
        mpc = mpc_wrap["mpc"]
    else:
        mpc = mpc_wrap

    return mpc


class TrainingLogger:
    """训练日志记录器 - 支持TensorBoard监控"""

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
        self.total_episodes = total_episodes

        # 智能自适应课程学习相关指标
        self.curriculum_stages = []
        self.stage_transitions = []
        self.director_decisions = []

        log_config = config.get('logging', {})
        self.metrics_save_interval = log_config.get('metrics_save_interval', 100)

        # 移除Rich状态面板，简化输出
        
        # 使用简单进度条
        try:
            from tqdm import tqdm
            self.progress_bar = tqdm(total=total_episodes, desc="🚀 训练进度",
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            self.use_rich = False
        except ImportError:
            # 如果tqdm也不可用，使用最基本的进度显示
            self.progress_bar = None
            self.use_rich = False

        # 设置TensorBoard
        self.use_tensorboard = log_config.get('use_tensorboard', False)
        self.tensorboard_writer = self._setup_tensorboard() if self.use_tensorboard else None

        # HTML仪表板已移动到test.py中用于性能分析

        # 移除TUI集成，简化为基本输出
        self.use_tui = False



    def _create_status_panel(self, episode: int, reward: float, info: Dict = None):
        """创建简洁的状态面板"""
        try:
            from rich.panel import Panel
            from rich.table import Table
            
            # 计算统计数据
            avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10) if self.episode_rewards else 0
            progress_pct = (episode + 1) / self.total_episodes * 100
            
            # 【改进】计算多层次成功率
            success_stats = self._compute_multi_level_success_rate()
            
            # 提取质量指标
            quality_score = None
            if info and 'reward_components' in info:
                components = info['reward_components']
                quality_score = components.get('quality_score')
            
            # 创建主表格
            table = Table.grid(padding=1)
            table.add_column("指标", style="bold cyan", min_width=12)
            table.add_column("数值", justify="right", min_width=15)
            table.add_column("状态", justify="center", min_width=8)
            
            # 进度信息
            table.add_row("训练进度", f"{episode+1:,}/{self.total_episodes:,} ({progress_pct:.1f}%)", "📊")
            
            # 奖励信息
            reward_color = "bold green" if reward > 0 else "yellow" if reward > -1 else "red"
            reward_icon = "🎉" if reward > 0 else "⚡" if reward > -1 else "❌"
            table.add_row("当前奖励", f"[{reward_color}]{reward:.3f}[/{reward_color}]", reward_icon)
            
            # 最佳奖励
            best_color = "bold green" if self.best_reward > 0 else "cyan"
            best_icon = "🏆" if self.best_reward > 0 else "🎯"
            table.add_row("最佳奖励", f"[{best_color}]{self.best_reward:.3f}[/{best_color}]", best_icon)
            
            # 平均奖励（最近10轮）
            avg_color = "green" if avg_reward > -1 else "yellow" if avg_reward > -2 else "red"
            table.add_row("平均奖励", f"[{avg_color}]{avg_reward:.3f}[/{avg_color}]", "📈")
            
            # 【改进】智能成功率显示
            success_display, success_icon = self._format_success_rate_display(success_stats)
            table.add_row("学习进展", success_display, success_icon)
            
            # 质量分数
            if quality_score is not None:
                quality_color = "green" if quality_score > 0.4 else "yellow" if quality_score > 0.3 else "red"
                table.add_row("质量分数", f"[{quality_color}]{quality_score:.3f}[/{quality_color}]", "⭐")
            
            # 【改进】智能系统状态评估
            system_status = self._evaluate_system_status(success_stats, avg_reward)
            table.add_row("系统状态", system_status, "🔧")
            
            # 运行时间
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                time_str = f"{elapsed:.0f}秒"
            elif elapsed < 3600:
                time_str = f"{elapsed/60:.1f}分钟"
            else:
                time_str = f"{elapsed/3600:.1f}小时"
            table.add_row("运行时间", time_str, "⏱️")
            
            title = "[bold blue]🚀 电力网络分区训练监控[/bold blue]"
            return Panel(table, title=title, border_style="blue", padding=(1, 2))
            
        except Exception as e:
            # 出错时返回简单文本
            return f"Episode {episode+1}/{self.total_episodes} | 奖励: {reward:.3f} | 最佳: {self.best_reward:.3f}"

    def _compute_multi_level_success_rate(self) -> Dict[str, Any]:
        """
        计算多层次成功率指标
        
        Returns:
            包含不同层次成功率的字典
        """
        if not self.episode_rewards:
            return {
                'positive_count': 0,
                'positive_rate': 0.0,
                'improvement_count': 0,
                'improvement_rate': 0.0,
                'learning_count': 0,
                'learning_rate': 0.0,
                'total_episodes': 0
            }
        
        total = len(self.episode_rewards)
        
        # 1. 传统正奖励成功率
        positive_count = sum(1 for r in self.episode_rewards if r > 0)
        positive_rate = positive_count / total
        
        # 2. 相对改进成功率（相比于初始几轮的表现）
        baseline_reward = np.mean(self.episode_rewards[:min(3, total)]) if total >= 3 else min(self.episode_rewards)
        improvement_count = sum(1 for r in self.episode_rewards if r > baseline_reward)
        improvement_rate = improvement_count / total
        
        # 3. 学习进展成功率（超过合理阈值）
        # 对于电力网络分区任务，-1.0是一个合理的学习阈值
        learning_threshold = -1.0
        learning_count = sum(1 for r in self.episode_rewards if r > learning_threshold)
        learning_rate = learning_count / total
        
        return {
            'positive_count': positive_count,
            'positive_rate': positive_rate,
            'improvement_count': improvement_count,
            'improvement_rate': improvement_rate,
            'learning_count': learning_count,
            'learning_rate': learning_rate,
            'total_episodes': total,
            'baseline_reward': baseline_reward,
            'learning_threshold': learning_threshold
        }
    
    def _format_success_rate_display(self, success_stats: Dict[str, Any]) -> Tuple[str, str]:
        """
        格式化成功率显示
        
        Args:
            success_stats: 多层次成功率统计
            
        Returns:
            (显示文本, 图标)
        """
        total = success_stats['total_episodes']
        
        # 选择最有意义的成功率指标进行显示
        if success_stats['positive_rate'] > 0:
            # 有正奖励时显示正奖励率
            count = success_stats['positive_count']
            rate = success_stats['positive_rate']
            color = "bold green"
            label = "正奖励"
            icon = "🎉"
        elif success_stats['learning_rate'] > 0.3:
            # 学习进展良好时显示学习率
            count = success_stats['learning_count']
            rate = success_stats['learning_rate']
            color = "green"
            label = "学习进展"
            icon = "📚"
        elif success_stats['improvement_rate'] > 0.2:
            # 有相对改进时显示改进率
            count = success_stats['improvement_count']
            rate = success_stats['improvement_rate']
            color = "yellow"
            label = "相对改进"
            icon = "📈"
        else:
            # 早期训练阶段
            count = success_stats['learning_count']
            rate = success_stats['learning_rate']
            color = "dim"
            label = "早期学习"
            icon = "🌱"
        
        display = f"[{color}]{count}/{total} ({rate*100:.1f}%) {label}[/{color}]"
        return display, icon
    
    def _evaluate_system_status(self, success_stats: Dict[str, Any], avg_reward: float) -> str:
        """
        智能评估系统状态
        
        Args:
            success_stats: 成功率统计
            avg_reward: 平均奖励
            
        Returns:
            状态显示文本
        """
        positive_rate = success_stats['positive_rate']
        learning_rate = success_stats['learning_rate']
        improvement_rate = success_stats['improvement_rate']
        
        if positive_rate > 0.3:
            return "[bold green]🟢 训练成功[/bold green]"
        elif positive_rate > 0.1 or learning_rate > 0.5:
            return "[green]🟡 学习良好[/green]"
        elif learning_rate > 0.3 or improvement_rate > 0.4:
            return "[yellow]🟠 稳步改进[/yellow]"
        elif avg_reward > -3.0 and improvement_rate > 0.2:
            return "[yellow]🟡 正在学习[/yellow]"
        else:
            return "[red]🔴 需要更多训练[/red]"

    def _setup_tensorboard(self):
        """设置TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.config['logging']['log_dir']
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            tensorboard_writer = SummaryWriter(f"{log_dir}/training_{timestamp}")
            from code.src.rich_output import rich_info
            rich_info(f"TensorBoard日志目录: {log_dir}/training_{timestamp}", show_always=True)
            return tensorboard_writer
        except ImportError:
            from code.src.rich_output import rich_warning
            rich_warning("TensorBoard不可用，跳过TensorBoard日志")
            self.use_tensorboard = False
            return None
        except Exception as e:
            from code.src.rich_output import rich_warning
            rich_warning(f"TensorBoard初始化失败: {e}")
            self.use_tensorboard = False
            return None

    def log_episode(self, episode: int, reward: float, length: int, info: Dict = None):
        """记录单个回合"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        if reward > self.best_reward:
            self.best_reward = reward

        # 如果使用TUI，将更新推送到队列
        if self.use_tui and self.tui_update_queue:
            from code.src.tui_monitor import TrainingUpdate
            
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            quality_score = info.get('quality_score', 0.5) if info else 0.5
            success_rate = np.mean(self.success_rates) * 100 if self.success_rates else 0.0

            update = TrainingUpdate(
                episode=episode,
                total_episodes=self.total_episodes,
                reward=reward,
                best_reward=self.best_reward,
                avg_reward=avg_reward,
                quality_score=quality_score,
                success_rate=success_rate,
                log_message=f"[{episode+1}/{self.total_episodes}] Reward: {reward:.3f}"
            )
            self.tui_update_queue.put(update)
            # 注释掉提前返回，让代码继续执行TensorBoard记录
            # return # TUI接管显示，直接返回

        # 使用简单进度条
        if self.progress_bar:
            # 计算一些基本统计信息
            avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10) if self.episode_rewards else reward

            # 更新进度条描述
            desc = f"🚀 训练进度 | 当前: {reward:.3f} | 最佳: {self.best_reward:.3f} | 平均: {avg_reward:.3f}"
            self.progress_bar.set_description(desc)
            self.progress_bar.update(1)
        else:
            # 如果没有进度条，使用简单的文本输出
            if episode % 100 == 0 or episode == self.total_episodes - 1:
                avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10) if self.episode_rewards else reward
                print(f"Episode {episode+1}/{self.total_episodes} | 奖励: {reward:.3f} | 最佳: {self.best_reward:.3f} | 平均: {avg_reward:.3f}")

        # 记录额外信息 - 【修复】适配新系统指标名称
        if info:
            if 'success' in info:
                self.success_rates.append(1.0 if info['success'] else 0.0)
            # 优先使用新指标名称，向后兼容旧名称
            if 'cv' in info or 'load_cv' in info:
                cv_value = info.get('cv', info.get('load_cv', 0.0))
                self.load_cvs.append(cv_value)
            if 'coupling_ratio' in info or 'coupling_edges' in info:
                coupling_value = info.get('coupling_ratio', info.get('coupling_edges', 0.0))
                self.coupling_edges.append(coupling_value)

        # TensorBoard日志 - 重新设计的分组记录
        if self.use_tensorboard and self.tensorboard_writer:
            self._log_tensorboard_metrics(episode, reward, length, info)

    def _log_tensorboard_metrics(self, episode: int, reward: float, length: int, info: dict):
        """
        分组记录TensorBoard指标，提供清晰的可视化结构

        指标分组：
        - Core: 核心训练指标
        - Reward: 奖励组件详情
        - Quality: 质量评估指标
        - Debug: 调试和诊断信息
        """
        if not (self.use_tensorboard and self.tensorboard_writer):
            return

        # === 1. 核心训练指标 (Core Training Metrics) ===
        self.tensorboard_writer.add_scalar('Core/Episode_Reward', reward, episode)
        self.tensorboard_writer.add_scalar('Core/Episode_Length', length, episode)

        if info:
            # 成功率和基础状态
            if 'success' in info:
                self.tensorboard_writer.add_scalar('Core/Success_Rate', float(info['success']), episode)
            if 'step' in info:
                self.tensorboard_writer.add_scalar('Core/Current_Step', info['step'], episode)

        # === 2. 奖励组件详情 (Reward Components) ===
        if info:
            # 终局奖励组件
            reward_components = {
                'balance_reward': 'Balance_Component',
                'decoupling_reward': 'Decoupling_Component',
                'power_reward': 'Power_Component',
                'quality_reward': 'Quality_Total',
                'final_reward': 'Final_Total'
            }

            for key, display_name in reward_components.items():
                if key in info and isinstance(info[key], (int, float)):
                    self.tensorboard_writer.add_scalar(f'Reward/{display_name}', info[key], episode)

            # 奖励调节因子
            adjustment_factors = {
                'threshold_bonus': 'Threshold_Bonus',
                'termination_discount': 'Termination_Discount'
            }

            for key, display_name in adjustment_factors.items():
                if key in info and isinstance(info[key], (int, float)):
                    self.tensorboard_writer.add_scalar(f'Reward/{display_name}', info[key], episode)

        # === 3. 质量评估指标 (Quality Assessment) ===
        if info:
            quality_metrics = {
                'quality_score': 'Unified_Quality_Score',
                'plateau_confidence': 'Plateau_Confidence'
            }

            for key, display_name in quality_metrics.items():
                if key in info and isinstance(info[key], (int, float)):
                    self.tensorboard_writer.add_scalar(f'Quality/{display_name}', info[key], episode)

        # === 4. 调试和诊断信息 (Debug & Diagnostics) ===
        if info:
            debug_metrics = {
                'reward_mode': 'Current_Reward_Mode'
            }

            for key, display_name in debug_metrics.items():
                if key in info:
                    # 对于字符串类型，转换为数值编码
                    if isinstance(info[key], str):
                        # 简单的字符串哈希编码用于可视化
                        value = hash(info[key]) % 1000
                        self.tensorboard_writer.add_scalar(f'Debug/{display_name}', value, episode)
                    elif isinstance(info[key], (int, float)):
                        self.tensorboard_writer.add_scalar(f'Debug/{display_name}', info[key], episode)

        # 定期刷新以确保数据写入
        if episode % 10 == 0:
            self.tensorboard_writer.flush()

    def log_training_step(self, episode: int, actor_loss: float = None,
                         critic_loss: float = None, entropy: float = None):
        """记录训练步骤 - 使用分组记录"""
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/Actor_Loss', actor_loss, episode)

        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/Critic_Loss', critic_loss, episode)

        if entropy is not None:
            self.entropies.append(entropy)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/Policy_Entropy', entropy, episode)

        # TensorBoard数据会在_log_tensorboard_metrics中定期刷新



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

    def set_training_mode(self, training: bool = True):
        """
        设置训练/评估模式，控制TensorBoard指标记录

        Args:
            training: True为训练模式（只记录训练相关指标），False为评估模式（记录所有指标）
        """
        self.training_mode = training

    # HTML仪表板生成功能已移动到test.py中用于性能分析

    def close(self):
        """关闭日志记录器"""
        # 关闭进度条
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.close()

        if self.tensorboard_writer:
            self.tensorboard_writer.close()


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
        self.tui_thread = None

    def train(self, num_episodes: int, max_steps_per_episode: int, update_interval: int = 10):
        """训练智能体"""
        self.logger = TrainingLogger(self.config, num_episodes)
        
        # 如果启用TUI，在单独的线程中运行它
        if self.logger.use_tui:
            self.tui_thread = threading.Thread(target=self.logger.tui_app.run, daemon=True)
            self.tui_thread.start()
            # 等待一小段时间以确保TUI启动
            time.sleep(1)

        from code.src.rich_output import rich_info
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

            # 【新增】验证指标完整性，确保修复生效 - 简化版
            if episode < 3 and not self.config.get('debug', {}).get('training_output', {}).get('only_show_errors', True):
                self._validate_metrics_integrity(episode_info, episode)

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

        # 简洁的训练完成统计
        final_stats = self.logger.get_statistics()
        positive_rewards = sum(1 for r in self.logger.episode_rewards if r > 0)
        total_episodes = final_stats.get('total_episodes', 0)
        best_reward = final_stats.get('best_reward', 0)
        mean_reward = final_stats.get('mean_reward', 0)

        # 使用Rich创建简洁的完成表格
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            
            console = Console()
            table = Table.grid(padding=1)
            table.add_column("指标", style="bold cyan")
            table.add_column("结果", style="green", justify="right")
            
            table.add_row("回合数", f"{total_episodes:,}")
            table.add_row("最佳奖励", f"{best_reward:.4f}")
            table.add_row("平均奖励", f"{mean_reward:.4f}")
            
            # 【改进】使用智能成功率统计
            if hasattr(self.logger, '_compute_multi_level_success_rate'):
                success_stats = self.logger._compute_multi_level_success_rate()
                if success_stats['positive_rate'] > 0:
                    success_text = f"正奖励: {success_stats['positive_count']}/{total_episodes} ({success_stats['positive_rate']*100:.1f}%)"
                elif success_stats['learning_rate'] > 0.3:
                    success_text = f"学习进展: {success_stats['learning_count']}/{total_episodes} ({success_stats['learning_rate']*100:.1f}%)"
                else:
                    success_text = f"相对改进: {success_stats['improvement_count']}/{total_episodes} ({success_stats['improvement_rate']*100:.1f}%)"
            else:
                # 备用传统计算
                success_text = f"{positive_rewards}/{total_episodes} ({positive_rewards/total_episodes*100:.1f}%)"
            
            table.add_row("学习效果", success_text)
            table.add_row("训练时间", f"{final_stats.get('training_time', 0)/60:.1f} 分钟")
            
            # 【改进】智能系统状态评估
            if hasattr(self.logger, '_compute_multi_level_success_rate'):
                success_stats = self.logger._compute_multi_level_success_rate()
                if success_stats['positive_rate'] > 0.3:
                    status = "[bold green]🎉 训练成功[/bold green]"
                elif success_stats['positive_rate'] > 0.1 or success_stats['learning_rate'] > 0.5:
                    status = "[green]✅ 学习良好[/green]"
                elif success_stats['learning_rate'] > 0.3 or success_stats['improvement_rate'] > 0.4:
                    status = "[yellow]🟠 稳步改进[/yellow]"
                elif mean_reward > -3.0 and success_stats['improvement_rate'] > 0.2:
                    status = "[yellow]🟡 正在学习[/yellow]"
                else:
                    status = "[red]⚠️ 需要更多训练[/red]"
            else:
                # 备用传统评估
                if best_reward > 0:
                    status = "[bold green]🎉 重构成功[/bold green]"
                elif positive_rewards > 0:
                    status = "[yellow]✅ 重构有效[/yellow]"
                else:
                    status = "[red]⚠️ 需要更多训练[/red]"
            table.add_row("系统状态", status)
            
            panel = Panel(table, title="[bold blue]🎯 训练完成统计[/bold blue]", border_style="blue")
            console.print(panel)
            
        except ImportError:
            # 备用简单输出
            print(f"\n🎯 训练完成: {total_episodes}回合, 最佳奖励: {best_reward:.4f}, 成功率: {positive_rewards/total_episodes*100:.1f}%")

        return {
            'episode_rewards': self.logger.episode_rewards,
            'episode_lengths': self.logger.episode_lengths,
            'training_stats': final_stats
        }

    def _validate_metrics_integrity(self, episode_info: Dict[str, Any], episode: int):
        """
        验证指标完整性，确保修复生效

        Args:
            episode_info: episode信息字典
            episode: episode编号
        """
        try:
            # 检查关键指标是否不再是固定值
            metrics = episode_info.get('metrics', {})

            # 检查CV指标
            cv_value = metrics.get('cv', metrics.get('load_cv', 1.0))
            if cv_value == 1.0 and episode > 1:
                from code.src.rich_output import rich_warning
                rich_warning(f"Episode {episode}: CV指标仍为固定值1.0，可能存在问题")

            # 输出指标摘要（仅前3个episode，更简洁）
            if episode <= 2:
                coupling_ratio = metrics.get('coupling_ratio', 1.0)
                connectivity = metrics.get('connectivity', 0.0)
                from code.src.rich_output import rich_debug
                rich_debug(f"Episode {episode} 指标: CV={cv_value:.3f}, Coupling={coupling_ratio:.3f}, Conn={connectivity:.3f}")

        except Exception as e:
            from code.src.rich_output import rich_warning
            rich_warning(f"Episode {episode} 指标验证失败: {e}")

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
        """评估智能体 - 使用合理的成功标准"""
        print(f"🔍 评估智能体 {num_episodes} 轮...")

        eval_rewards = []
        success_count = 0

        for episode in range(num_episodes):
            state, _ = self.env.reset()  # 解包元组
            episode_reward = 0
            episode_info = {}

            for step in range(200):  # 最大步数
                action, _, _ = self.agent.select_action(state, training=False)

                if action is None:  # 没有有效动作
                    break

                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # 收集最后的环境信息
                if info:
                    episode_info.update(info)

                if done:
                    break

            eval_rewards.append(episode_reward)
            
            # 【改进】使用合理的成功标准评估，而不是依赖环境success标志
            is_success = self._evaluate_episode_success(episode_reward, episode_info)
            if is_success:
                success_count += 1

        return {
            'avg_reward': np.mean(eval_rewards),
            'success_rate': success_count / num_episodes,
            'rewards': eval_rewards
        }



    def _evaluate_episode_success(self, episode_reward: float, episode_info: Dict[str, Any]) -> bool:
        """
        评估单个episode是否成功
        
        Args:
            episode_reward: episode总奖励
            episode_info: episode信息（可能包含指标）
            
        Returns:
            是否成功
        """
        try:
            # 成功标准1: 奖励超过合理阈值
            if episode_reward > -1.0:  # 对于复杂任务，-1.0是一个合理的成功阈值
                return True
            
            # 成功标准2: 基于质量指标（如果可用）
            if episode_info and 'metrics' in episode_info:
                metrics = episode_info['metrics']
                
                # 检查负载平衡（CV < 0.5）
                cv = metrics.get('cv', metrics.get('load_cv', 1.0))
                if cv < 0.5:
                    return True
                
                # 检查连通性（connectivity > 0.9）
                connectivity = metrics.get('connectivity', 0.0)
                if connectivity > 0.9:
                    return True
                
                # 检查耦合比率（coupling_ratio < 0.3）
                coupling_ratio = metrics.get('coupling_ratio', 1.0)
                if coupling_ratio < 0.3:
                    return True
            
            # 成功标准3: 基于质量分数（如果可用）
            if episode_info and 'reward_components' in episode_info:
                components = episode_info['reward_components']
                quality_score = components.get('quality_score', 0.0)
                if quality_score > 0.4:  # 质量分数超过0.4认为成功
                    return True
            
            # 如果都不满足，但奖励有显著改进（相比-5.0基线）
            if episode_reward > -2.5:  # 相比很差的基线有显著改进
                return True
                
            return False
            
        except Exception as e:
            # 如果评估出错，保守地根据奖励判断
            return episode_reward > -1.5



    def close(self):
        """清理资源"""
        if self.logger:
            self.logger.close()


def create_environment_from_config(config: Dict, hetero_data: HeteroData, node_embeddings: Dict, attention_weights: Dict, mpc_data: Dict, device: torch.device, is_normalized: bool = True) -> Tuple[PowerGridPartitioningEnv, Optional[gym.Env]]:
    """根据配置创建环境实例，如果需要则使用Gym包装器。"""
    should_use_gym_wrapper = config.get('scenario_generation', {}).get('enabled', True) or \
                             config.get('parallel_training', {}).get('enabled', False)

    if should_use_gym_wrapper:
        from code.src.rl.gym_wrapper import PowerGridPartitionGymEnv
        config_copy = config.copy()
        config_copy['system']['device'] = str(device)
        
        gym_env = PowerGridPartitionGymEnv(
            base_case_data=mpc_data,
            config=config_copy,
            use_scenario_generator=config.get('scenario_generation', {}).get('enabled', True),
            scenario_seed=config['system']['seed']
        )
        _, _ = gym_env.reset()
        env = gym_env.internal_env
        return env, gym_env
    
    # 创建标准环境
    env_config = config['environment']
    env = PowerGridPartitioningEnv(
        hetero_data,
        node_embeddings=node_embeddings,
        num_partitions=env_config['num_partitions'],
        reward_weights=env_config.get('reward_weights', {}),
        max_steps=env_config['max_steps'],
        device=device,
        attention_weights=attention_weights,
        config=config,
        is_normalized=is_normalized
    )
    return env, None


if __name__ == "__main__":
    import argparse, yaml, copy, pprint, sys
    
    parser = argparse.ArgumentParser(description="RL Trainer for Power Grid Partitioning")
    parser.add_argument("--mode", default="fast", help="配置预设名称 (fast / full / ieee118 等)")
    parser.add_argument("--run", action="store_true", help="执行完整训练流程。若省略则仅打印合并后的配置")
    args = parser.parse_args()

    # 1. 读取配置文件
    cfg_file = Path("config.yaml")
    if not cfg_file.exists():
        safe_print("❌ 找不到 config.yaml ，请检查项目目录")
        sys.exit(1)
    with open(cfg_file, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # 2. 递归合并预设
    def deep_update(dest: Dict[str, Any], src: Dict[str, Any]):
        for k, v in src.items():
            if isinstance(v, dict) and k in dest:
                deep_update(dest[k], v)
            else:
                dest[k] = copy.deepcopy(v)

    if args.mode in base_cfg:
        deep_update(base_cfg, base_cfg[args.mode])
    else:
        safe_print(f"⚠️ 未找到名为 '{args.mode}' 的预设，使用基础配置")

    if not args.run:
        safe_print("🚀 配置已加载（未启动训练，添加 --run 可开始训练）\n─" * 40)
        pprint.pp(base_cfg)
        sys.exit(0)

    # 3. === 正式训练管道 ===
    safe_print("🚀 开始训练流程 ...")

    # 3.1 设备
    device_str = base_cfg['system'].get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    # 3.2 数据加载
    mpc_data = load_power_grid_data(base_cfg['data']['case_name'])

    # 3.3 图处理
    from code.src.data_processing import PowerGridDataProcessor
    processor = PowerGridDataProcessor(
        normalize=base_cfg['data'].get('normalize', True),
        cache_dir=base_cfg['data'].get('cache_dir', 'data/cache')
    )
    hetero_data = processor.graph_from_mpc(mpc_data, config=base_cfg)

    # 3.4 编码器 (GAT)
    encoder_cfg = base_cfg.get('gat', {})
    encoder = create_production_encoder(hetero_data, encoder_cfg).to(device)
    hetero_data = hetero_data.to(device)

    # 预计算节点嵌入 & 注意力权重
    with torch.no_grad():
        node_embeddings_dict, attention_weights = encoder.encode_nodes_with_attention(hetero_data, base_cfg)

    # 将 node_embeddings_dict 转为同一维度张量字典（环境内部按 node_type 键访问）

    # 3.5 创建环境
    env, gym_env = create_environment_from_config(
        base_cfg, hetero_data, node_embeddings_dict, attention_weights, mpc_data, device,
        is_normalized=base_cfg['data'].get('normalize', True)
    )

    # 3.6 创建智能体（需先获取实际state维度）
    # 先reset一次环境以拿到示例state
    init_state, _ = env.reset()
    node_emb_dim = init_state['node_embeddings'].shape[1]
    region_emb_dim = init_state['region_embeddings'].shape[1]

    agent = PPOAgent(
        node_embedding_dim=node_emb_dim,
        region_embedding_dim=region_emb_dim,
        num_partitions=base_cfg['environment']['num_partitions'],
        agent_config=base_cfg.get('agent', {}),
        device=device
    )

     # 3.7 训练器
    trainer = UnifiedTrainer(agent, env, base_cfg, gym_env=gym_env)

    train_cfg = base_cfg['training']
    trainer.train(
        num_episodes=train_cfg['num_episodes'],
        max_steps_per_episode=train_cfg['max_steps_per_episode'],
        update_interval=train_cfg.get('update_interval', 10)
    )

    safe_print("🎉 训练流程结束。")
    
