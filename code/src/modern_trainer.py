#!/usr/bin/env python3
"""
现代化训练监控系统

专注于简洁、稳定、美观的训练过程显示：
- 清晰的进度指示
- 实时性能监控
- 智能错误处理
- 跨平台兼容
"""

import time
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import math
import shutil


@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    episode: int
    reward: float
    best_reward: float
    avg_reward: float
    success_rate: float
    quality_score: float
    learning_rate: float
    elapsed_time: float
    status: str


class SimpleProgressBar:
    """简单稳定的进度条"""
    
    def __init__(self, total: int, width: int = 50, desc: str = "训练进度"):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        
    def update(self, current: int = None):
        """更新进度"""
        if current is not None:
            self.current = current
        else:
            self.current += 1
            
        # 计算进度
        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)
        
        # 计算速度和ETA
        elapsed = time.time() - self.start_time
        if elapsed > 0 and self.current > 0:
            speed = self.current / elapsed
            eta = (self.total - self.current) / speed if speed > 0 else 0
            eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
        else:
            eta_str = "?"
            
        # 输出进度条
        percent = progress * 100
        print(f"\r{self.desc}: {bar} {percent:5.1f}% ({self.current}/{self.total}) ETA: {eta_str}", 
              end='', flush=True)
    
    def finish(self):
        """完成进度条"""
        print()  # 换行


class ModernTrainingMonitor:
    """现代化训练监控器"""
    
    def __init__(self, total_episodes: int, update_interval: int = 10):
        self.total_episodes = total_episodes
        self.update_interval = update_interval
        self.start_time = time.time()
        
        # 历史数据
        self.reward_history = deque(maxlen=100)
        self.success_history = deque(maxlen=50)
        self.quality_history = deque(maxlen=50)
        
        # 状态追踪
        self.last_update_time = 0
        self.best_reward = float('-inf')
        self.episode_count = 0
        
        # 终端宽度
        self.terminal_width = self._get_terminal_width()
        
        print(f"\n🚀 启动现代化训练监控 (总回合: {total_episodes:,})")
        print("=" * min(60, self.terminal_width))
        
    def _get_terminal_width(self) -> int:
        """获取终端宽度"""
        try:
            return shutil.get_terminal_size().columns
        except:
            return 80
    
    def update(self, episode: int, reward: float, info: Dict[str, Any] = None):
        """更新训练状态"""
        self.episode_count = episode + 1
        current_time = time.time()
        
        # 更新历史数据
        self.reward_history.append(reward)
        self.best_reward = max(self.best_reward, reward)
        
        if info:
            success = info.get('success', reward > -1.0)
            quality = info.get('quality_score', 0.5)
            self.success_history.append(success)
            self.quality_history.append(quality)
        
        # 按间隔更新显示
        if (current_time - self.last_update_time >= 1.0 or 
            episode % self.update_interval == 0 or 
            episode == self.total_episodes - 1):
            
            self._display_status(episode, reward, info)
            self.last_update_time = current_time
    
    def _display_status(self, episode: int, reward: float, info: Dict[str, Any] = None):
        """显示训练状态"""
        # 清屏并回到顶部（优雅的方式）
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/Mac
            print('\033[2J\033[1;1H', end='')
        
        # 标题
        print("🚀 电力网络分区强化学习训练监控")
        print("=" * min(50, self.terminal_width))
        
        # 基本信息
        progress = (episode + 1) / self.total_episodes * 100
        elapsed = time.time() - self.start_time
        
        print(f"📊 进度: {episode + 1:,}/{self.total_episodes:,} ({progress:.1f}%)")
        print(f"⏱️  运行时间: {self._format_time(elapsed)}")
        
        # 奖励信息
        avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0
        print(f"🎯 当前奖励: {reward:.3f}")
        print(f"⭐ 最佳奖励: {self.best_reward:.3f}")
        print(f"📈 平均奖励: {avg_reward:.3f}")
        
        # 成功率和质量
        if self.success_history:
            success_rate = sum(self.success_history) / len(self.success_history) * 100
            print(f"✅ 成功率: {success_rate:.1f}%")
        
        if self.quality_history:
            avg_quality = sum(self.quality_history) / len(self.quality_history)
            print(f"🌟 质量分数: {avg_quality:.3f}")
        
        # 状态评估
        status = self._evaluate_status(reward, avg_reward)
        print(f"🔧 训练状态: {status}")
        
        # 进度条
        bar_width = min(40, self.terminal_width - 10)
        filled = int(bar_width * progress / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"\n{bar} {progress:.1f}%")
        
        # 额外信息
        if info and 'metrics' in info:
            metrics = info['metrics']
            cv = metrics.get('cv', 0)
            coupling = metrics.get('coupling_ratio', 0)
            print(f"\n📋 技术指标: CV={cv:.3f}, 耦合={coupling:.3f}")
        
        print("\n" + "─" * min(50, self.terminal_width))
        
        # 刷新显示
        sys.stdout.flush()
    
    def _evaluate_status(self, current_reward: float, avg_reward: float) -> str:
        """评估训练状态"""
        if current_reward > self.best_reward * 0.9:
            return "🟢 表现优秀"
        elif current_reward > avg_reward:
            return "🔵 稳步提升"
        elif current_reward > avg_reward * 0.8:
            return "🟡 缓慢进步"
        else:
            return "🔴 需要调整"
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    def final_summary(self, history: List[Dict[str, Any]]):
        """显示最终总结"""
        print("\n" + "=" * min(60, self.terminal_width))
        print("🎉 训练完成！")
        print("=" * min(60, self.terminal_width))
        
        if history:
            rewards = [h.get('reward', 0) for h in history]
            best_reward = max(rewards)
            avg_reward = sum(rewards) / len(rewards)
            final_reward = rewards[-1]
            
            print(f"📊 最终统计:")
            print(f"   总回合数: {len(history):,}")
            print(f"   最佳奖励: {best_reward:.4f}")
            print(f"   平均奖励: {avg_reward:.4f}")
            print(f"   最终奖励: {final_reward:.4f}")
            
            # 计算成功率
            positive_rewards = sum(1 for r in rewards if r > -1.0)
            success_rate = positive_rewards / len(rewards) * 100
            print(f"   成功率: {success_rate:.1f}%")
            
            # 训练效果评估
            improvement = final_reward - rewards[0] if len(rewards) > 1 else 0
            print(f"   改善程度: {improvement:.4f}")
            
            if success_rate > 60:
                status = "🌟 训练成功！"
            elif success_rate > 30:
                status = "✅ 训练良好"
            elif improvement > 0.5:
                status = "📈 有所改善"
            else:
                status = "⚠️ 需要调优"
            
            print(f"   训练评价: {status}")
        
        total_time = time.time() - self.start_time
        print(f"   总用时: {self._format_time(total_time)}")
        print()


class MinimalTrainingLogger:
    """极简训练日志器 - 用于替代复杂的Rich显示"""
    
    def __init__(self, total_episodes: int, log_interval: int = 50):
        self.total_episodes = total_episodes
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = 0
        
        print(f"🚀 开始训练 (共 {total_episodes:,} 回合)")
        print("─" * 50)
    
    def log(self, episode: int, reward: float, best_reward: float, avg_reward: float):
        """记录训练进度"""
        current_time = time.time()
        
        # 按间隔记录或最后一个回合
        if (episode % self.log_interval == 0 or 
            episode == self.total_episodes - 1 or
            current_time - self.last_log_time >= 10):  # 至少10秒记录一次
            
            progress = (episode + 1) / self.total_episodes * 100
            elapsed = current_time - self.start_time
            
            print(f"[{episode + 1:>6}/{self.total_episodes}] "
                  f"({progress:5.1f}%) "
                  f"奖励: {reward:>7.3f} "
                  f"最佳: {best_reward:>7.3f} "
                  f"平均: {avg_reward:>7.3f} "
                  f"用时: {elapsed:>6.0f}s")
            
            self.last_log_time = current_time
    
    def finish(self, final_stats: Dict[str, Any]):
        """完成训练日志"""
        total_time = time.time() - self.start_time
        print("─" * 50)
        print(f"✅ 训练完成！用时: {total_time:.0f}秒")
        
        if final_stats:
            print(f"📊 最佳奖励: {final_stats.get('best_reward', 0):.4f}")
            print(f"📈 平均奖励: {final_stats.get('mean_reward', 0):.4f}")


def create_modern_logger(total_episodes: int, style: str = 'auto') -> Any:
    """
    创建现代化训练日志器
    
    Args:
        total_episodes: 总回合数
        style: 显示风格 ('modern', 'minimal', 'auto')
    
    Returns:
        训练日志器实例
    """
    # 自动选择风格
    if style == 'auto':
        # 根据环境和回合数选择
        if total_episodes > 500 and sys.stdout.isatty():
            style = 'modern'
        else:
            style = 'minimal'
    
    if style == 'modern':
        return ModernTrainingMonitor(total_episodes)
    else:
        return MinimalTrainingLogger(total_episodes)


# 使用示例和测试
if __name__ == "__main__":
    import random
    import time
    
    # 测试现代化监控器
    print("测试现代化训练监控器...")
    monitor = ModernTrainingMonitor(total_episodes=100)
    
    history = []
    for i in range(100):
        # 模拟训练数据
        reward = -3.0 + i * 0.02 + random.uniform(-0.5, 0.5)
        success = reward > -1.0
        quality = min(1.0, max(0.0, 0.3 + i * 0.005))
        
        info = {
            'success': success,
            'quality_score': quality,
            'metrics': {
                'cv': max(0.1, 1.0 - i * 0.008),
                'coupling_ratio': max(0.1, 0.8 - i * 0.006)
            }
        }
        
        monitor.update(i, reward, info)
        history.append({'reward': reward, 'episode': i})
        
        # 模拟训练延迟
        time.sleep(0.05)
    
    monitor.final_summary(history) 