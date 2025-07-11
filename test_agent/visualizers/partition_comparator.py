#!/usr/bin/env python3
"""
分区对比可视化器
简洁高效地对比展示baseline和agent的分区结果
支持中文控制台输出和英文图片文本
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import networkx as nx
import numpy as np
from typing import Dict, Tuple, Optional, Any
import seaborn as sns


class PartitionComparator:
    """分区结果对比可视化器"""
    
    def __init__(self):
        # 设置matplotlib样式和字体
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 智能字体配置
        self._setup_fonts()
        
        self.colors = sns.color_palette("husl", 8)  # 支持最多8个分区
        
    def _setup_fonts(self):
        """智能设置字体，优先使用系统可用的最佳字体"""
        import matplotlib
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # 刷新缓存
        
        # 获取系统可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 字体优先级列表 (按质量和兼容性排序)
        preferred_fonts = [
            'DejaVu Sans',          # 优秀的Unicode支持
            'Liberation Sans',      # Linux标准字体
            'Arial',                # Windows标准字体  
            'Helvetica',            # Mac标准字体
            'Bitstream Vera Sans',  # 备用字体
            'sans-serif'            # 系统默认
        ]
        
        # 选择第一个可用的字体
        selected_fonts = []
        for font in preferred_fonts:
            if font in available_fonts or font == 'sans-serif':
                selected_fonts.append(font)
        
        # 应用字体配置
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = selected_fonts
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"📝 可视化使用字体: {selected_fonts[0] if selected_fonts else 'default'}")
        
    def _get_safe_text(self, text: str) -> str:
        """获取安全的文本，避免特殊字符渲染问题"""
        # 替换可能有问题的特殊字符
        replacements = {
            '✓': 'Pass',
            '✗': 'Fail', 
            '×': 'x',
            '√': 'OK',
            '±': '+/-',
            '≥': '>=',
            '≤': '<=',
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v'
        }
        
        safe_text = text
        for old, new in replacements.items():
            safe_text = safe_text.replace(old, new)
        
        return safe_text
        
    def compare_partitions(self,
                          adj_matrix: np.ndarray,
                          baseline_partition: np.ndarray,
                          agent_partition: np.ndarray,
                          baseline_metrics: Dict[str, float],
                          agent_metrics: Dict[str, float],
                          node_positions: Optional[Dict] = None,
                          scenario_name: str = "Normal",
                          save_path: str = None) -> str:
        """
        对比两种分区方法的结果
        
        Args:
            adj_matrix: 邻接矩阵
            baseline_partition: baseline分区结果 (1-indexed)
            agent_partition: agent分区结果 (1-indexed)
            baseline_metrics: baseline性能指标
            agent_metrics: agent性能指标
            node_positions: 节点位置字典，如果为None则自动计算
            scenario_name: 场景名称
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        # 创建NetworkX图
        G = nx.from_numpy_array(adj_matrix)
        
        # 计算节点位置（如果未提供）
        if node_positions is None:
            # 使用spring layout，但固定随机种子保证一致性
            node_positions = nx.spring_layout(G, k=2/np.sqrt(len(G)), seed=42)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Partition Comparison - {scenario_name} Scenario', fontsize=16, fontweight='bold')
        
        # 绘制baseline结果
        self._draw_partition(G, baseline_partition, node_positions, ax1, 
                           "Baseline (Spectral)", baseline_metrics)
        
        # 绘制agent结果
        self._draw_partition(G, agent_partition, node_positions, ax2, 
                           "RL Agent", agent_metrics)
        
        # 添加整体性能对比
        self._add_comparison_text(fig, baseline_metrics, agent_metrics)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = f"result/output/partition_comparison_{scenario_name.lower()}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _draw_partition(self, G: nx.Graph, partition: np.ndarray, 
                       pos: Dict, ax: plt.Axes, title: str, metrics: Dict[str, float]):
        """绘制单个分区结果"""
        
        # 节点颜色映射
        node_colors = []
        for i in range(len(G)):
            if i < len(partition) and partition[i] > 0:
                color_idx = int(partition[i] - 1) % len(self.colors)
                node_colors.append(self.colors[color_idx])
            else:
                node_colors.append('gray')
        
        # 边的样式：跨分区边用虚线
        edge_styles = []
        edge_colors = []
        edge_widths = []
        
        for u, v in G.edges():
            if u < len(partition) and v < len(partition):
                if int(partition[u]) == int(partition[v]):
                    edge_styles.append('solid')
                    edge_colors.append('gray')
                    edge_widths.append(1.0)
                else:
                    edge_styles.append('dashed')
                    edge_colors.append('red')
                    edge_widths.append(2.0)
            else:
                edge_styles.append('solid')
                edge_colors.append('gray')
                edge_widths.append(1.0)
        
        # 绘制网络
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=500, ax=ax, alpha=0.9)
        
        # 分别绘制不同样式的边
        solid_edges = [(u, v) for i, (u, v) in enumerate(G.edges()) if edge_styles[i] == 'solid']
        dashed_edges = [(u, v) for i, (u, v) in enumerate(G.edges()) if edge_styles[i] == 'dashed']
        
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, 
                             edge_color='gray', width=1.0, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, 
                             edge_color='red', width=2.0, style='dashed', ax=ax)
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
        
        # 设置标题和指标
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # 添加完整的test.py指标，使用安全文本避免渲染问题
        metrics_text = "Raw Metrics (lower is better):\n"
        metrics_text += f"Load_B (CV): {metrics.get('cv', 0):.3f}\n"
        metrics_text += f"Coupling: {metrics.get('coupling', 0):.3f}\n"
        metrics_text += f"Power_B: {metrics.get('power_b', 0):.3f}\n\n"
        
        metrics_text += "Converted Scores (higher is better):\n"
        metrics_text += f"Load_B Score: {metrics.get('load_b_score', 0):.3f}\n"
        metrics_text += f"Decoupling Score: {metrics.get('decoupling_score', 0):.3f}\n"
        metrics_text += f"Power_B Score: {metrics.get('power_b_score', 0):.3f}\n\n"
        
        metrics_text += f"Comprehensive Score: {metrics.get('reward', 0):.3f}\n"
        # 使用安全文本函数处理特殊字符
        success_text = "Pass" if metrics.get('success', False) else "Fail"
        metrics_text += f"Success: {success_text}"
        
        # 应用安全文本转换
        safe_metrics_text = self._get_safe_text(metrics_text)
        
        ax.text(0.02, 0.98, safe_metrics_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                fontsize=8)  # 缩小字体以适应更多信息
        
        # 添加分区图例
        unique_partitions = np.unique(partition[partition > 0])
        if len(unique_partitions) > 0:
            patches = []
            for p in unique_partitions:
                color_idx = (p - 1) % len(self.colors)
                patches.append(mpatches.Patch(color=self.colors[color_idx], 
                                            label=f'Partition {p}'))
            ax.legend(handles=patches, loc='upper right', fontsize=9)
        
        ax.axis('off')
    
    def _add_comparison_text(self, fig: plt.figure, 
                           baseline_metrics: Dict[str, float], 
                           agent_metrics: Dict[str, float]):
        """Add complete test.py metrics comparison information"""
        
        # Calculate all metric improvements
        comparison_text = "Agent vs Baseline Improvements:\n"
        
        # Raw metric improvements (lower is better, so positive % means improvement)
        if baseline_metrics.get('cv', 0) > 0:
            cv_improvement = (baseline_metrics['cv'] - agent_metrics['cv']) / baseline_metrics['cv'] * 100
            comparison_text += f"Load_B Improvement: {cv_improvement:+.1f}%\n"
        
        if baseline_metrics.get('coupling', 0) > 0:
            coupling_improvement = (baseline_metrics['coupling'] - agent_metrics['coupling']) / baseline_metrics['coupling'] * 100
            comparison_text += f"Coupling Improvement: {coupling_improvement:+.1f}%\n"
        
        if baseline_metrics.get('power_b', 0) > 0:
            power_improvement = (baseline_metrics['power_b'] - agent_metrics['power_b']) / baseline_metrics['power_b'] * 100
            comparison_text += f"Power_B Improvement: {power_improvement:+.1f}%\n"
        
        # Comprehensive score improvement (higher is better)
        if baseline_metrics.get('reward', 0) > 0:
            reward_improvement = (agent_metrics['reward'] - baseline_metrics['reward']) / baseline_metrics['reward'] * 100
            comparison_text += f"Comprehensive Score Improvement: {reward_improvement:+.1f}%"
        
        # 应用安全文本转换
        safe_comparison_text = self._get_safe_text(comparison_text)
        
        # Add comparison text at the bottom
        fig.text(0.5, 0.02, safe_comparison_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def compare_all_methods(self,
                           adj_matrix: np.ndarray,
                           partitions: Dict[str, np.ndarray],
                           metrics: Dict[str, Dict[str, float]],
                           node_positions: Optional[Dict] = None,
                           scenario_name: str = "Normal",
                           save_path: str = None) -> str:
        """
        对比所有方法的分区结果 - 2x4布局
        
        Args:
            adj_matrix: 邻接矩阵
            partitions: 所有方法的分区结果 {method_name: partition}
            metrics: 所有方法的性能指标 {method_name: metrics_dict}
            node_positions: 节点位置字典，如果为None则自动计算
            scenario_name: 场景名称
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        # 创建NetworkX图
        G = nx.from_numpy_array(adj_matrix)
        
        # 计算节点位置（如果未提供）
        if node_positions is None:
            # 使用spring layout，但固定随机种子保证一致性
            node_positions = nx.spring_layout(G, k=2/np.sqrt(len(G)), seed=42)
        
        # 创建2x4布局的图形
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'All Methods Comparison - {scenario_name} Scenario', fontsize=16, fontweight='bold')
        
        # 方法名称顺序 (确保Agent在第一位)
        method_order = ['Agent']
        other_methods = [name for name in partitions.keys() if name != 'Agent']
        method_order.extend(other_methods)
        
        # 绘制所有方法的结果
        for idx, method_name in enumerate(method_order):
            if idx >= 8:  # 只显示前8个方法
                break
                
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]
            
            if method_name in partitions and method_name in metrics:
                self._draw_partition(G, partitions[method_name], node_positions, ax, 
                                   method_name, metrics[method_name])
            else:
                # 如果某个方法不存在，显示空白
                ax.set_title(f'{method_name} (Not Available)', fontsize=14)
                ax.axis('off')
        
        # 如果方法数量少于8个，隐藏多余的子图
        total_methods = len(method_order)
        for idx in range(total_methods, 8):
            row = idx // 4
            col = idx % 4
            axes[row, col].axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = f"result/output/all_methods_comparison_{scenario_name.lower()}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


# 便捷函数
def compare_partitions(adj_matrix: np.ndarray,
                      baseline_partition: np.ndarray,
                      agent_partition: np.ndarray,
                      baseline_metrics: Dict[str, float],
                      agent_metrics: Dict[str, float],
                      scenario_name: str = "Normal") -> str:
    """便捷函数：对比两种分区方法"""
    comparator = PartitionComparator()
    return comparator.compare_partitions(
        adj_matrix, baseline_partition, agent_partition,
        baseline_metrics, agent_metrics, 
        scenario_name=scenario_name
    )