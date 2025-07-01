#!/usr/bin/env python3
"""
Rich 输出管理器

使用 Rich 库提供美化的终端输出，支持：
- 彩色进度条
- 格式化的错误和警告信息
- 优雅的表格显示
- 控制输出详细程度
"""

import sys
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.style import Style
    from rich.theme import Theme
    from rich.traceback import install as install_rich_traceback
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich 库未安装，使用标准输出")


class RichOutputManager:
    """Rich 输出管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化输出管理器"""
        self.config = config or {}
        self.debug_config = self.config.get('debug', {})
        self.output_config = self.debug_config.get('training_output', {})
        
        # 输出控制开关
        self.only_show_errors = self.output_config.get('only_show_errors', True)
        self.show_cache_loading = self.output_config.get('show_cache_loading', False)
        self.show_attention_collection = self.output_config.get('show_attention_collection', False)
        self.show_state_manager_details = self.output_config.get('show_state_manager_details', False)
        self.show_metis_details = self.output_config.get('show_metis_details', False)
        self.show_scenario_generation = self.output_config.get('show_scenario_generation', False)
        
        if RICH_AVAILABLE:
            # 创建自定义主题
            custom_theme = Theme({
                "info": "cyan",
                "warning": "yellow",
                "error": "bold red",
                "success": "bold green",
                "progress": "blue",
                "debug": "dim white"
            })
            
            self.console = Console(theme=custom_theme, stderr=False)
            self.error_console = Console(theme=custom_theme, stderr=True)
            
            # 安装 Rich 异常处理
            install_rich_traceback(show_locals=False)
        else:
            self.console = None
            self.error_console = None
    
    def info(self, message: str, show_always: bool = False):
        """显示信息消息"""
        if self.only_show_errors and not show_always:
            return
            
        if RICH_AVAILABLE:
            self.console.print(f"ℹ️  {message}", style="info")
        else:
            print(f"ℹ️  {message}")
    
    def success(self, message: str, show_always: bool = False):
        """显示成功消息"""
        if self.only_show_errors and not show_always:
            return
            
        if RICH_AVAILABLE:
            self.console.print(f"✅ {message}", style="success")
        else:
            print(f"✅ {message}")
    
    def warning(self, message: str):
        """显示警告消息（总是显示）"""
        if RICH_AVAILABLE:
            self.console.print(f"⚠️  {message}", style="warning")
        else:
            print(f"⚠️  {message}")
    
    def error(self, message: str, exception: Exception = None):
        """显示错误消息（总是显示）"""
        if RICH_AVAILABLE:
            self.error_console.print(f"❌ {message}", style="error")
            if exception and self.debug_config.get('verbose_logging', False):
                self.error_console.print_exception(show_locals=False)
        else:
            print(f"❌ {message}", file=sys.stderr)
            if exception:
                import traceback
                traceback.print_exc()
    
    def debug(self, message: str, category: str = "general"):
        """
        显示调试消息
        
        Args:
            message: 调试消息
            category: 消息类别 (cache, attention, state, metis, scenario)
        """
        if self.only_show_errors:
            return
            
        # 检查特定类别的显示开关
        category_switches = {
            'cache': self.show_cache_loading,
            'attention': self.show_attention_collection,
            'state': self.show_state_manager_details,
            'metis': self.show_metis_details,
            'scenario': self.show_scenario_generation
        }
        
        if category in category_switches and not category_switches[category]:
            return
            
        if RICH_AVAILABLE:
            self.console.print(f"🔍 {message}", style="debug")
        else:
            print(f"🔍 {message}")
    
    def create_progress_bar(self, description: str, total: int = None) -> 'RichProgressBar':
        """创建进度条"""
        return RichProgressBar(description, total, self.console)
    
    def create_table(self, title: str = None) -> 'RichTable':
        """创建表格"""
        return RichTable(title, self.console)
    
    def print_panel(self, content: str, title: str = None, style: str = "info"):
        """显示面板"""
        if self.only_show_errors and style not in ["error", "warning"]:
            return
            
        if RICH_AVAILABLE:
            panel = Panel(content, title=title, border_style=style)
            self.console.print(panel)
        else:
            if title:
                print(f"\n=== {title} ===")
            print(content)
            print("=" * (len(title) + 8) if title else "")


class RichProgressBar:
    """Rich 进度条包装器"""
    
    def __init__(self, description: str, total: int = None, console: Console = None):
        self.description = description
        self.total = total
        self.console = console
        self.current = 0
        self.start_time = time.time()
        
        if RICH_AVAILABLE and console:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
                expand=True
            )
            self.task_id = None
            self.is_started = False
        else:
            self.progress = None
            # 使用 tqdm 作为备用
            try:
                from tqdm import tqdm
                self.tqdm_progress = tqdm(total=total, desc=description)
            except ImportError:
                self.tqdm_progress = None
    
    def __enter__(self):
        if self.progress:
            self.progress.start()
            self.task_id = self.progress.add_task(self.description, total=self.total)
            self.is_started = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress and self.is_started:
            self.progress.stop()
        if hasattr(self, 'tqdm_progress') and self.tqdm_progress:
            self.tqdm_progress.close()
    
    def update(self, advance: int = 1, **kwargs):
        """更新进度"""
        self.current += advance

        if self.progress and self.task_id is not None:
            # 更新 Rich 进度条 - 增强显示重构后的系统状态
            update_dict = {'advance': advance}
            if kwargs:
                # 特殊处理奖励显示，突出显示正奖励
                formatted_parts = []
                for k, v in kwargs.items():
                    if k == '奖励':
                        try:
                            reward = float(v)
                            if reward > 0:
                                formatted_parts.append(f"[bold green]{k}={v}[/bold green] 🎉")
                            elif reward > -1:
                                formatted_parts.append(f"[yellow]{k}={v}[/yellow] ⚡")
                            else:
                                formatted_parts.append(f"[red]{k}={v}[/red]")
                        except:
                            formatted_parts.append(f"{k}={v}")
                    elif k == '最佳':
                        try:
                            best = float(v)
                            if best > 0:
                                formatted_parts.append(f"[bold cyan]{k}={v}[/bold cyan] 🏆")
                            else:
                                formatted_parts.append(f"{k}={v}")
                        except:
                            formatted_parts.append(f"{k}={v}")
                    else:
                        formatted_parts.append(f"{k}={v}")

                postfix = ", ".join(formatted_parts)
                update_dict['description'] = f"{self.description} ({postfix})"

            self.progress.update(self.task_id, **update_dict)
        elif hasattr(self, 'tqdm_progress') and self.tqdm_progress:
            # 更新 tqdm 进度条
            self.tqdm_progress.update(advance)
            if kwargs:
                self.tqdm_progress.set_postfix(kwargs)
    
    def set_description(self, description: str):
        """设置描述"""
        self.description = description
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=description)
        elif hasattr(self, 'tqdm_progress') and self.tqdm_progress:
            self.tqdm_progress.set_description(description)


class RichTable:
    """Rich 表格包装器"""
    
    def __init__(self, title: str = None, console: Console = None):
        self.console = console
        if RICH_AVAILABLE and console:
            self.table = Table(title=title, box=box.ROUNDED)
        else:
            self.title = title
            self.columns = []
            self.rows = []
    
    def add_column(self, name: str, style: str = None, justify: str = "left"):
        """添加列"""
        if RICH_AVAILABLE and hasattr(self, 'table'):
            self.table.add_column(name, style=style, justify=justify)
        else:
            self.columns.append(name)
    
    def add_row(self, *row_data):
        """添加行"""
        if RICH_AVAILABLE and hasattr(self, 'table'):
            self.table.add_row(*[str(item) for item in row_data])
        else:
            self.rows.append(row_data)
    
    def print(self):
        """打印表格"""
        if RICH_AVAILABLE and hasattr(self, 'table') and self.console:
            self.console.print(self.table)
        else:
            # 备用文本表格
            if self.title:
                print(f"\n{self.title}")
                print("=" * len(self.title))
            
            # 打印列头
            if self.columns:
                print(" | ".join(self.columns))
                print("-" * (len(" | ".join(self.columns))))
            
            # 打印行
            for row in self.rows:
                print(" | ".join(str(item) for item in row))
            print()


# 全局输出管理器实例
_global_output_manager = None


def get_output_manager(config: Dict[str, Any] = None) -> RichOutputManager:
    """获取全局输出管理器实例"""
    global _global_output_manager
    if _global_output_manager is None:
        _global_output_manager = RichOutputManager(config)
    return _global_output_manager


def set_output_manager(config: Dict[str, Any]):
    """设置全局输出管理器"""
    global _global_output_manager
    _global_output_manager = RichOutputManager(config)


# 便捷函数
def rich_info(message: str, show_always: bool = False):
    """显示信息消息"""
    get_output_manager().info(message, show_always)


def rich_success(message: str, show_always: bool = False):
    """显示成功消息"""
    get_output_manager().success(message, show_always)


def rich_warning(message: str):
    """显示警告消息"""
    get_output_manager().warning(message)


def rich_error(message: str, exception: Exception = None):
    """显示错误消息"""
    get_output_manager().error(message, exception)


def rich_debug(message: str, category: str = "general"):
    """显示调试消息"""
    get_output_manager().debug(message, category)


def rich_progress(description: str, total: int = None):
    """创建进度条"""
    return get_output_manager().create_progress_bar(description, total)


def create_training_status_panel(episode: int, reward: float, best_reward: float,
                                avg_reward: float, positive_rewards: int,
                                quality_score: float = None, plateau_confidence: float = None):
    """创建训练状态面板"""
    if not RICH_AVAILABLE:
        return None

    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    # 创建状态表格
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("指标", style="bold cyan")
    table.add_column("数值", justify="right")
    table.add_column("状态", justify="center")

    # 回合信息
    table.add_row("回合", f"{episode}", "📊")

    # 奖励信息
    reward_color = "green" if reward > 0 else "yellow" if reward > -1 else "red"
    reward_icon = "🎉" if reward > 0 else "⚡" if reward > -1 else "❌"
    table.add_row("当前奖励", f"[{reward_color}]{reward:.3f}[/{reward_color}]", reward_icon)

    # 最佳奖励
    best_color = "bold green" if best_reward > 0 else "cyan"
    best_icon = "🏆" if best_reward > 0 else "🎯"
    table.add_row("最佳奖励", f"[{best_color}]{best_reward:.3f}[/{best_color}]", best_icon)

    # 平均奖励
    avg_color = "green" if avg_reward > -1 else "yellow" if avg_reward > -2 else "red"
    table.add_row("平均奖励", f"[{avg_color}]{avg_reward:.3f}[/{avg_color}]", "📈")

    # 正奖励次数
    positive_color = "bold green" if positive_rewards > 0 else "dim"
    table.add_row("正奖励次数", f"[{positive_color}]{positive_rewards}[/{positive_color}]", "✨")

    # 质量分数（如果可用）
    if quality_score is not None:
        quality_color = "green" if quality_score > 0.4 else "yellow" if quality_score > 0.3 else "red"
        table.add_row("质量分数", f"[{quality_color}]{quality_score:.3f}[/{quality_color}]", "⭐")

    # 平台期置信度（如果可用）
    if plateau_confidence is not None:
        conf_color = "bold green" if plateau_confidence > 0.9 else "green" if plateau_confidence > 0.8 else "yellow"
        table.add_row("平台期置信度", f"[{conf_color}]{plateau_confidence:.3f}[/{conf_color}]", "🎯")

    # 系统状态
    system_status = "🟢 重构成功" if best_reward > 0 else "🟡 学习中" if avg_reward > -2 else "🔴 需优化"
    table.add_row("系统状态", system_status, "🔧")

    return Panel(
        table,
        title="[bold blue]🚀 训练状态监控 - 系统现代化重构版[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    )


def print_training_summary(stats: Dict[str, Any]):
    """打印训练摘要"""
    output_manager = get_output_manager()
    
    if not output_manager.only_show_errors:
        table = output_manager.create_table("🎯 训练摘要")
        table.add_column("指标", style="cyan", justify="right")
        table.add_column("数值", style="magenta")
        
        # 添加关键统计信息
        table.add_row("总回合数", str(stats.get('total_episodes', 0)))
        table.add_row("最佳奖励", f"{stats.get('best_reward', 0):.4f}")
        table.add_row("平均奖励", f"{stats.get('mean_reward', 0):.4f}")
        table.add_row("训练时间", f"{stats.get('training_time', 0):.2f}秒")
        if 'success_rate' in stats:
            table.add_row("成功率", f"{stats.get('success_rate', 0):.2%}")
        
        table.print()


def print_evaluation_summary(results: Dict[str, Any]):
    """打印评估摘要"""
    output_manager = get_output_manager()
    
    if results.get('success', False):
        output_manager.success("评估成功完成!", show_always=True)
        
        # 创建结果表格
        table = output_manager.create_table("📊 评估结果")
        table.add_column("项目", style="cyan")
        table.add_column("结果", style="green")
        
        if 'baseline_comparison' in results and results['baseline_comparison']['success']:
            baseline_summary = results['baseline_comparison'].get('summary', {})
            table.add_row("基线对比", f"最佳方法: {baseline_summary.get('best_method', 'N/A')}")
            table.add_row("最佳分数", f"{baseline_summary.get('best_score', 0):.4f}")
        
        if 'report_path' in results:
            table.add_row("报告路径", str(results['report_path']))
        
        if 'results_path' in results:
            table.add_row("结果路径", str(results['results_path']))
        
        table.print()
    else:
        output_manager.error(f"评估失败: {results.get('error', '未知错误')}") 