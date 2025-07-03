#!/usr/bin/env python3
"""
基于 Textual 的现代化、交互式训练监控 TUI 系统
"""

import time
from typing import Dict, Any, List
from collections import deque
from dataclasses import dataclass, field

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, ProgressBar, Log
from textual.reactive import reactive

@dataclass
class TrainingUpdate:
    """训练数据更新包"""
    episode: int
    total_episodes: int
    reward: float
    best_reward: float
    avg_reward: float
    quality_score: float
    success_rate: float
    log_message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

class MetricsWidget(Static):
    """显示核心训练指标的组件 - 重新设计为分行显示"""
    
    reward = reactive(0.0)
    best_reward = reactive(0.0)
    avg_reward = reactive(0.0)
    quality_score = reactive(0.0)
    success_rate = reactive(0.0)

    def render(self) -> str:
        return f"""🎯 当前奖励:  {self.reward:>8.3f}

⭐ 最佳奖励:  {self.best_reward:>8.3f}

📈 平均奖励:  {self.avg_reward:>8.3f}

✅ 成功率:    {self.success_rate:>7.1f}%

🌟 质量分数:  {self.quality_score:>7.3f}"""

class ProgressWidget(Static):
    """进度显示组件"""
    
    episode = reactive(0)
    total_episodes = reactive(1500)
    
    def render(self) -> str:
        progress_pct = (self.episode / self.total_episodes * 100) if self.total_episodes > 0 else 0
        # 创建简单的进度条
        bar_width = 20
        filled = int(bar_width * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        return f"""📊 训练进度

{self.episode:>4}/{self.total_episodes} ({progress_pct:5.1f}%)

{bar}

⏱️ Episode {self.episode}"""

class TrainingMonitorApp(App):
    """电力网络分区训练监控 Textual TUI 应用"""

    CSS_PATH = "tui_monitor.css"
    BINDINGS = [
        ("d", "toggle_dark", "切换主题"),
        ("q", "quit", "退出"),
    ]

    def __init__(self, update_queue, total_episodes: int):
        super().__init__()
        self.update_queue = update_queue
        self.total_episodes = total_episodes
        self.should_stop = False
    
    def compose(self) -> ComposeResult:
        """创建应用组件"""
        yield Header(name="⚡️ 电力网络分区强化学习训练监控 ⚡️")
        
        with Container(id="main_container"):
            # 顶部面板：左侧指标，右侧进度
            with Horizontal(id="top_panel"):
                with Vertical(id="metrics_section"):
                    yield Static("📊 核心指标", classes="section_title")
                    yield MetricsWidget(id="metrics_widget")
                
                with Vertical(id="progress_section"):
                    yield ProgressWidget(id="progress_widget")

            # 底部：实时日志
            with Vertical(id="log_section"):
                yield Static("📜 实时日志", classes="section_title")
                yield Log(id="log_view", max_lines=100)
        
        yield Footer()

    def on_mount(self) -> None:
        """应用挂载后启动更新任务"""
        # 初始化进度组件的总集数
        progress_widget = self.query_one(ProgressWidget)
        progress_widget.total_episodes = self.total_episodes
        
        self.set_interval(0.1, self.check_for_updates)

    def check_for_updates(self) -> None:
        """检查队列中的更新"""
        if self.should_stop:
            return
            
        try:
            while not self.update_queue.empty():
                update = self.update_queue.get_nowait()
                self.process_update(update)
        except:
            # 队列为空或其他错误，继续
            pass

    def process_update(self, update: TrainingUpdate):
        """处理来自训练线程的更新"""
        try:
            # 更新指标
            metrics_widget = self.query_one(MetricsWidget)
            metrics_widget.reward = update.reward
            metrics_widget.best_reward = update.best_reward
            metrics_widget.avg_reward = update.avg_reward
            metrics_widget.quality_score = update.quality_score
            metrics_widget.success_rate = update.success_rate

            # 更新进度
            progress_widget = self.query_one(ProgressWidget)
            progress_widget.episode = update.episode + 1
            
            # 更新日志
            if update.log_message:
                log_widget = self.query_one(Log)
                log_widget.write(update.log_message)
        except Exception as e:
            # 静默处理更新错误
            pass

    def action_quit(self) -> None:
        """退出应用"""
        self.should_stop = True
        super().action_quit()

if __name__ == '__main__':
    # 测试用例
    import queue
    import threading
    import random

    q = queue.Queue()
    total_episodes = 100
    app = TrainingMonitorApp(update_queue=q, total_episodes=total_episodes)
    
    def mock_training(q, total):
        for i in range(total):
            update = TrainingUpdate(
                episode=i,
                total_episodes=total,
                reward=-5.0 + i * 0.02 + (random.random() - 0.5),
                best_reward=max(0, -5.0 + i * 0.02),
                avg_reward=-4.0 + i * 0.01,
                quality_score=0.2 + i * 0.001,
                success_rate= i / total * 50,
                log_message=f"Episode {i+1} done. Reward: {-5.0 + i * 0.02:.2f}"
            )
            q.put(update)
            time.sleep(0.1)
        time.sleep(1)
        app.action_quit()

    training_thread = threading.Thread(target=mock_training, args=(q, total_episodes))
    training_thread.start()
    
    app.run()
    training_thread.join() 