#!/usr/bin/env python3
"""
实时训练监控脚本
用于观察训练过程中的各种指标
"""

import os
import sys
import time
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from collections import deque

# 添加src到路径
sys.path.append(str(Path(__file__).parent / 'src'))

class TrainingMonitor:
    """实时训练监控器"""
    
    def __init__(self, log_dir: str = "logs", checkpoint_dir: str = "checkpoints"):
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # 监控数据
        self.episodes = []
        self.rewards = []
        self.lengths = []
        self.success_rates = []
        self.load_cvs = []
        self.actor_losses = []
        self.critic_losses = []
        
        # 实时显示设置
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('实时训练监控', fontsize=16, fontweight='bold')
        
        # 数据缓存
        self.recent_rewards = deque(maxlen=100)
        self.last_update_time = time.time()
        
    def find_latest_tensorboard_log(self) -> Optional[Path]:
        """查找最新的TensorBoard日志文件"""
        if not self.log_dir.exists():
            return None
            
        # 查找最新的训练日志目录
        training_dirs = [d for d in self.log_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('training_')]
        
        if not training_dirs:
            return None
            
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        return latest_dir
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """查找最新的检查点文件"""
        if not self.checkpoint_dir.exists():
            return None
            
        checkpoint_files = list(self.checkpoint_dir.glob("training_stats_episode_*.json"))
        if not checkpoint_files:
            return None
            
        latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        return latest_file
    
    def load_checkpoint_data(self, checkpoint_file: Path) -> Dict:
        """加载检查点数据"""
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载检查点失败: {e}")
            return {}
    
    def parse_tensorboard_logs(self, log_dir: Path) -> Dict:
        """解析TensorBoard日志（简化版本）"""
        # 这里可以实现TensorBoard日志解析
        # 由于复杂性，这里返回空字典，实际可以使用tensorboard库解析
        return {}
    
    def update_data(self):
        """更新监控数据"""
        # 查找最新的检查点
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            data = self.load_checkpoint_data(latest_checkpoint)
            if data:
                self._update_from_checkpoint(data)
        
        # 查找TensorBoard日志
        latest_tb_log = self.find_latest_tensorboard_log()
        if latest_tb_log:
            tb_data = self.parse_tensorboard_logs(latest_tb_log)
            if tb_data:
                self._update_from_tensorboard(tb_data)
    
    def _update_from_checkpoint(self, data: Dict):
        """从检查点数据更新"""
        # 这里可以根据实际的检查点数据格式进行解析
        if 'total_episodes' in data:
            self.episodes.append(data['total_episodes'])
        if 'mean_reward' in data:
            self.rewards.append(data['mean_reward'])
            self.recent_rewards.append(data['mean_reward'])
        if 'mean_length' in data:
            self.lengths.append(data['mean_length'])
        if 'success_rate' in data:
            self.success_rates.append(data['success_rate'])
        if 'mean_load_cv' in data:
            self.load_cvs.append(data['mean_load_cv'])
        if 'mean_actor_loss' in data:
            self.actor_losses.append(data['mean_actor_loss'])
        if 'mean_critic_loss' in data:
            self.critic_losses.append(data['mean_critic_loss'])
    
    def _update_from_tensorboard(self, data: Dict):
        """从TensorBoard数据更新"""
        # 实现TensorBoard数据解析逻辑
        pass
    
    def update_plots(self, frame):
        """更新图表"""
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        # 更新数据
        self.update_data()
        
        # 如果没有数据，显示等待信息
        if not self.rewards:
            self.axes[0, 0].text(0.5, 0.5, '等待训练数据...', 
                               ha='center', va='center', transform=self.axes[0, 0].transAxes)
            return
        
        # 1. 奖励曲线
        ax = self.axes[0, 0]
        if self.episodes and self.rewards:
            ax.plot(self.episodes, self.rewards, 'b-', alpha=0.7, label='平均奖励')
            if len(self.rewards) > 10:
                # 移动平均
                window = min(10, len(self.rewards) // 3)
                moving_avg = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
                episodes_ma = self.episodes[window-1:]
                ax.plot(episodes_ma, moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        ax.set_title('训练奖励')
        ax.set_xlabel('回合')
        ax.set_ylabel('奖励')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 回合长度
        ax = self.axes[0, 1]
        if self.episodes and self.lengths:
            ax.plot(self.episodes, self.lengths, 'g-', alpha=0.7)
        ax.set_title('回合长度')
        ax.set_xlabel('回合')
        ax.set_ylabel('步数')
        ax.grid(True, alpha=0.3)
        
        # 3. 成功率
        ax = self.axes[0, 2]
        if self.episodes and self.success_rates:
            ax.plot(self.episodes, self.success_rates, 'm-', alpha=0.7)
        ax.set_title('成功率')
        ax.set_xlabel('回合')
        ax.set_ylabel('成功率')
        ax.grid(True, alpha=0.3)
        
        # 4. 负载CV
        ax = self.axes[1, 0]
        if self.episodes and self.load_cvs:
            ax.plot(self.episodes, self.load_cvs, 'c-', alpha=0.7)
        ax.set_title('负载变异系数')
        ax.set_xlabel('回合')
        ax.set_ylabel('CV')
        ax.grid(True, alpha=0.3)
        
        # 5. Actor损失
        ax = self.axes[1, 1]
        if self.episodes and self.actor_losses:
            ax.plot(self.episodes, self.actor_losses, 'orange', alpha=0.7)
        ax.set_title('Actor损失')
        ax.set_xlabel('回合')
        ax.set_ylabel('损失')
        ax.grid(True, alpha=0.3)
        
        # 6. Critic损失
        ax = self.axes[1, 2]
        if self.episodes and self.critic_losses:
            ax.plot(self.episodes, self.critic_losses, 'purple', alpha=0.7)
        ax.set_title('Critic损失')
        ax.set_xlabel('回合')
        ax.set_ylabel('损失')
        ax.grid(True, alpha=0.3)
        
        # 更新时间戳
        current_time = time.strftime('%H:%M:%S')
        self.fig.suptitle(f'实时训练监控 - 更新时间: {current_time}', 
                         fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def start_monitoring(self, update_interval: int = 5):
        """开始实时监控"""
        print(f"🔍 开始实时训练监控...")
        print(f"📂 日志目录: {self.log_dir}")
        print(f"📂 检查点目录: {self.checkpoint_dir}")
        print(f"🔄 更新间隔: {update_interval} 秒")
        print("💡 关闭窗口停止监控")
        
        # 创建动画
        ani = animation.FuncAnimation(
            self.fig, self.update_plots, 
            interval=update_interval * 1000,  # 转换为毫秒
            cache_frame_data=False
        )
        
        plt.show()
        return ani

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时训练监控')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--interval', type=int, default=5, help='更新间隔（秒）')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = TrainingMonitor(args.log_dir, args.checkpoint_dir)
    
    # 开始监控
    try:
        ani = monitor.start_monitoring(args.interval)
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")

if __name__ == "__main__":
    main()
