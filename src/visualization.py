import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from matplotlib.patches import Patch
from typing import Optional, Dict, List, Any
import pandas as pd
import seaborn as sns
from pathlib import Path
try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        PLOTLY_AVAILABLE = True
except ImportError:
        PLOTLY_AVAILABLE = False

# Import types that will be defined in other modules
from typing import TYPE_CHECKING
if TYPE_CHECKING:
        from env import PowerGridPartitionEnv

class VisualizationManager:
    """
    可视化管理器 - 统一管理所有可视化功能
    支持配置文件驱动的可视化参数设置
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化管理器
        
        Args:
            config: 完整的配置字典
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.enabled = self.viz_config.get('enabled', True)
        self.save_figures = self.viz_config.get('save_figures', True)
        self.figures_dir = Path(self.viz_config.get('figures_dir', 'figures'))
        
        # 创建图片目录
        if self.save_figures:
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            
        # 设置matplotlib参数
        self._setup_matplotlib()
        
    def _setup_matplotlib(self):
        """设置matplotlib全局参数"""
        figure_settings = self.viz_config.get('figure_settings', {})
        plt.rcParams['figure.dpi'] = figure_settings.get('dpi', 300)
        plt.rcParams['savefig.bbox'] = figure_settings.get('bbox_inches', 'tight')
        plt.rcParams['savefig.format'] = figure_settings.get('format', 'png')
        
    def get_color_palette(self, num_colors: int) -> List[str]:
        """
        动态生成颜色方案
        
        Args:
            num_colors: 需要的颜色数量
            
        Returns:
            颜色列表 (HEX格式)
        """
        colors_config = self.viz_config.get('colors', {})
        palette_type = colors_config.get('palette_type', 'husl')
        
        # 使用配置的颜色方案
        palette = sns.color_palette(palette_type, num_colors)
        return [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in palette]
        
    def visualize_partition(self, env: 'PowerGridPartitionEnv', 
                          title: str = "Power Grid Partition",
                          save_path: Optional[str] = None) -> None:
        """
        可视化电网分区结果 (已修复以兼容新版环境)
        """
        if not self.enabled:
            return
            
        # 从 env 对象的正确位置获取数据
        try:
            # 使用 state_manager 获取分区和节点总数
            current_partition = env.state_manager.current_partition
            total_nodes = env.total_nodes
            num_partitions = env.num_partitions

            # 使用 edge_info 获取边信息
            edge_array = env.edge_info['edge_index'].cpu().numpy()
            
            # 使用 evaluator 获取物理数据
            node_pd = env.evaluator.load_active
            node_pg = env.evaluator.gen_active
            edge_admittance = env.evaluator.edge_admittances

        except AttributeError as e:
            print(f"❌ 可视化失败：无法从env对象获取必要的属性。请确保env对象结构正确。错误: {e}")
            return
            
        try:
            import gc
            
            # 获取分区可视化配置
            partition_config = self.viz_config.get('partition_plot', {})
            figsize = partition_config.get('figsize', [16, 10])
            show_metrics = partition_config.get('show_metrics', True)
            node_size_scale = partition_config.get('node_size_scale', 500)
            edge_alpha = partition_config.get('edge_alpha', 0.2)
            coupling_edge_width = partition_config.get('coupling_edge_width', 2)
            coupling_edge_alpha = partition_config.get('coupling_edge_alpha', 0.6)
            font_size = partition_config.get('font_size', 8)
            
            # 创建图形 (简化版本，专注于核心可视化)
            fig, ax_main = plt.subplots(figsize=figsize)
            
            # 创建NetworkX图
            G = nx.Graph()
            
            # 添加节点
            for i in range(total_nodes):
                G.add_node(i)
            
            # 添加边（去重）
            edge_set = set()
            for i in range(edge_array.shape[1]):
                u, v = edge_array[0, i], edge_array[1, i]
                edge_set.add((min(u, v), max(u, v)))
            
            G.add_edges_from(list(edge_set))
            
            # 计算布局
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            
            # 颜色方案
            unassigned_color = self.viz_config.get('colors', {}).get('unassigned_color', '#E0E0E0')
            colors = [unassigned_color] + self.get_color_palette(num_partitions)
            
            # 节点颜色和大小
            node_colors = [colors[current_partition[i].item()] for i in range(total_nodes)]
            node_sizes = [300 + node_pd[i].item() * node_size_scale for i in range(total_nodes)]
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, alpha=edge_alpha, ax=ax_main)
            
            # 高亮跨区域边
            inter_edges = []
            for u, v in edge_set:
                if current_partition[u] > 0 and current_partition[v] > 0 and current_partition[u] != current_partition[v]:
                    inter_edges.append((u, v))
            
            nx.draw_networkx_edges(G, pos, edgelist=inter_edges, edge_color='red',
                                  width=coupling_edge_width, alpha=coupling_edge_alpha, ax=ax_main)
            
            # 绘制节点和标签
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                  alpha=0.9, ax=ax_main)
            nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold', ax=ax_main)
            
            # 图例
            legend_elements = []
            for k in range(num_partitions + 1):
                count = (current_partition == k).sum().item()
                if k == 0:
                    label = f'Unassigned ({count} nodes)'
                else:
                    label = f'Region {k} ({count} nodes)'
                legend_elements.append(mpatches.Patch(color=colors[k], label=label))
            
            ax_main.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            ax_main.set_title(title, fontsize=16, fontweight='bold')
            ax_main.axis('off')

            plt.tight_layout()
            
            # 保存图片
            if self.save_figures:
                if save_path is None:
                    save_path = self.figures_dir / 'partition_result.png'
                else:
                    save_path = self.figures_dir / save_path
                plt.savefig(save_path)
                print(f"💾 分区图已保存到: {save_path}")
            
            plt.show()
            
            # 清理内存
            gc.collect()
            
        except Exception as e:
            print(f"⚠️ 可视化执行出错: {e}")
            import traceback
            traceback.print_exc()
            try:
                plt.close('all')
                gc.collect()
            except:
                pass

    def plot_training_curves(self, history: Dict[str, List[float]], 
                           env_N: int = None, save_path: Optional[str] = None) -> None:
        """
        绘制训练曲线
        """
        if not self.enabled:
            return
            
        # 获取训练曲线配置
        curves_config = self.viz_config.get('training_curves', {})
        figsize = curves_config.get('figsize', [12, 10])
        moving_average_window = curves_config.get('moving_average_window', 20)
        grid_alpha = curves_config.get('grid_alpha', 0.3)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 奖励曲线
        ax = axes[0, 0]
        episodes = range(len(history['episode_rewards']))
        ax.plot(episodes, history['episode_rewards'], alpha=0.6, label='Episode Reward')
        
        # 移动平均
        if len(history['episode_rewards']) > 10:
            window = min(moving_average_window, len(history['episode_rewards']) // 5)
            moving_avg = pd.Series(history['episode_rewards']).rolling(window).mean()
            ax.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Rewards')
        ax.legend()
        ax.grid(True, alpha=grid_alpha)
        
        # 2. Load CV曲线
        ax = axes[0, 1]
        if 'load_cv' in history:
            ax.plot(episodes, history['load_cv'], 'g-', alpha=0.8)
            ax.set_ylim(0, max(history['load_cv']) * 1.1 if history['load_cv'] else 1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Load CV')
        ax.set_title('Load Balance (CV)')
        ax.grid(True, alpha=grid_alpha)
        
        # 3. 耦合度曲线
        ax = axes[1, 0]
        if 'coupling_edges' in history:
            ax.plot(episodes, history['coupling_edges'], 'b-', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Coupling Edges')
        ax.set_title('Inter-region Coupling')
        ax.grid(True, alpha=grid_alpha)
        
        # 4. Episode长度
        ax = axes[1, 1]
        ax.plot(episodes, history['episode_lengths'], 'm-', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=grid_alpha)
        
        # 添加完成线（如果提供了env_N）
        if env_N is not None:
            ax.axhline(y=env_N, color='r', linestyle='--', alpha=0.5, label='Complete')
            ax.legend()
        
        fig.suptitle('Training Process Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图片
        if self.save_figures:
            if save_path is None:
                save_path = self.figures_dir / 'training_curves.png'
            else:
                save_path = self.figures_dir / save_path
            plt.savefig(save_path)
            print(f"🖼️ 训练曲线图已保存到: {save_path}")
        plt.close()

    def create_interactive_visualization(self, env: 'PowerGridPartitionEnv', 
                                       comparison_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        创建交互式可视化（使用Plotly）
        """
        interactive_config = self.viz_config.get('interactive', {})
        if not interactive_config.get('enabled', True) or not PLOTLY_AVAILABLE:
            print("⚠️ 交互式可视化未启用或Plotly不可用")
            return None
            
        import torch
        
        template = interactive_config.get('template', 'plotly_white')
        height = interactive_config.get('height', 800)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Network Topology', 'Method Comparison', 'Load Distribution',
                'Coupling Matrix', 'Metrics Radar', 'Region Statistics'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'heatmap'}, {'type': 'scatterpolar'}, {'type': 'table'}]
            ],
            column_widths=[0.4, 0.3, 0.3],
            row_heights=[0.6, 0.4]
        )
        
        # 1. 网络拓扑
        G = nx.Graph()
        edge_array = env.edge_index.cpu().numpy()
        
        for i in range(env.N):
            G.add_node(i)
        
        edge_set = set()
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            edge_set.add((min(u, v), max(u, v)))
        
        G.add_edges_from(list(edge_set))
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # 绘制边
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(
            go.Scatter(x=edge_x, y=edge_y, mode='lines',
                      line=dict(width=0.5, color='#888'),
                      hoverinfo='none', showlegend=False),
            row=1, col=1
        )
        
        # 绘制节点
        colors = ['#E0E0E0'] + self.get_color_palette(env.K)
        
        for k in range(env.K + 1):
            mask = (env.z == k)
            if mask.any():
                node_indices = torch.where(mask)[0].cpu().numpy()
                node_x = [pos[i][0] for i in node_indices]
                node_y = [pos[i][1] for i in node_indices]
                node_text = [f'Node {i}<br>Load: {env.Pd[i]:.3f}' for i in node_indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        marker=dict(size=10, color=colors[k]),
                        text=[str(i) for i in node_indices],
                        textposition='top center',
                        hovertext=node_text,
                        hoverinfo='text',
                        name=f'Region {k}' if k > 0 else 'Unassigned'
                    ),
                    row=1, col=1
                )
        
        # 2. 方法比较
        fig.add_trace(
            go.Bar(
                x=comparison_df.index,
                y=comparison_df['overall_score'],
                marker_color=['green' if x == 'RL (PPO)' else 'lightblue' 
                             for x in comparison_df.index],
                text=[f'{v:.3f}' for v in comparison_df['overall_score']],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. 负荷分布
        region_data = {'Region': [], 'Load': [], 'Generation': []}
        for k in range(1, env.K + 1):
            mask = (env.z == k)
            if mask.any():
                region_data['Region'].append(f'R{k}')
                region_data['Load'].append(env.Pd[mask].sum().item())
                region_data['Generation'].append(env.Pg[mask].sum().item())
        
        # 动态生成颜色
        bar_colors = self.get_color_palette(env.K)
        
        fig.add_trace(go.Bar(
            x=[f'R{i+1}' for i in range(env.K)],
            y=region_data['Load'],
            marker_color=bar_colors,
            name='Load'
        ), row=1, col=3)
        
        fig.add_trace(go.Bar(
            x=[f'R{i+1}' for i in range(env.K)],
            y=region_data['Generation'],
            marker_color=bar_colors,
            name='Generation'
        ), row=1, col=3)
        
        # 4. 耦合矩阵
        coupling_matrix = np.zeros((env.K, env.K))
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            if env.z[u] > 0 and env.z[v] > 0 and env.z[u] != env.z[v]:
                coupling_matrix[env.z[u]-1, env.z[v]-1] += env.admittance[i].item()
        
        fig.add_trace(
            go.Heatmap(
                z=coupling_matrix,
                x=[f'R{i+1}' for i in range(env.K)],
                y=[f'R{i+1}' for i in range(env.K)],
                colorscale='YlOrRd',
                text=np.round(coupling_matrix, 3),
                texttemplate='%{text}',
                textfont={'size': 10}
            ),
            row=2, col=1
        )
        
        # 5. 指标雷达图（只显示RL方法）
        metrics_to_plot = ['load_cv', 'connectivity', 'modularity']
        rl_values = [comparison_df.loc['RL (PPO)', m] for m in metrics_to_plot]
        
        # 归一化到0-1（用于雷达图）
        norm_values = []
        for i, m in enumerate(metrics_to_plot):
            if m in ['load_cv']:  # 越小越好
                norm_values.append(1 - rl_values[i])
            else:  # 越大越好
                norm_values.append(rl_values[i])
        
        fig.add_trace(
            go.Scatterpolar(
                r=norm_values + [norm_values[0]],  # 闭合
                theta=metrics_to_plot + [metrics_to_plot[0]],
                fill='toself',
                name='RL (PPO)'
            ),
            row=2, col=2
        )
        
        # 6. 统计表格
        stats_data = [
            ['Total Nodes', str(env.N)],
            ['Regions', str(env.K)],
            ['Load CV', f'{env.current_metrics.load_cv:.4f}'],
            ['Total Coupling', f'{env.current_metrics.total_coupling:.4f}'],
            ['Connectivity', f'{env.current_metrics.connectivity:.2f}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=list(zip(*stats_data)),
                          fill_color='lavender',
                          align='left')
            ),
            row=2, col=3
        )
        
        # 更新布局
        fig.update_layout(
            title_text="Power Grid Partition Analysis Dashboard",
            title_font_size=20,
            showlegend=True,
            height=height,
            template=template
        )
        
        # 更新子图标题
        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, row=1, col=1)
        
        # 保存HTML文件
        if interactive_config.get('save_html', True) and self.save_figures:
            html_path = self.figures_dir / 'interactive_partition_analysis.html'
            fig.write_html(str(html_path))
            print(f"🌐 交互式图表已保存到: {html_path}")
        
        return fig

    def run_basic_visualization(self, env, history):
        """运行基础可视化"""
        if not self.enabled:
            return
            
        print("\n📈 生成基础可视化图表...")
        
        # 可视化最终分区
        self.visualize_partition(env, "Final Partition Result (RL)", "rl_partition_result.png")
        
        # 绘制训练曲线
        self.plot_training_curves(history, env_N=env.N, save_path="training_curves.png")

    def run_interactive_visualization(self, env, comparison_df):
        """运行交互式可视化"""
        interactive_config = self.viz_config.get('interactive', {})
        if not interactive_config.get('enabled', True):
            return
            
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly未安装，跳过交互式可视化。")
            return
            
        print("📊 生成交互式对比图表...")
        fig = self.create_interactive_visualization(env, comparison_df)
        
        if fig is not None:
            print(f"🌐 交互式图表创建成功")


# 向后兼容的函数接口
def get_color_palette(num_colors: int) -> List[str]:
    """向后兼容的颜色生成函数"""
    palette = sns.color_palette("husl", num_colors)
    return [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in palette]

def visualize_partition(env: 'PowerGridPartitionEnv', title: str = "Power Grid Partition",
                       save_path: Optional[str] = None, show_metrics: bool = True):
    """向后兼容的分区可视化函数 (已修复)"""
    # 使用默认配置
    default_config = {'visualization': {'enabled': True, 'save_figures': True}}
    viz_manager = VisualizationManager(default_config)
    viz_manager.visualize_partition(env, title, save_path)

def plot_training_curves(history: Dict[str, List[float]], env_N: int = None, save_path: Optional[str] = None):
    """向后兼容的训练曲线函数"""
    # 使用默认配置
    default_config = {'visualization': {'enabled': True, 'save_figures': True}}
    viz_manager = VisualizationManager(default_config)
    viz_manager.plot_training_curves(history, env_N, save_path)

def run_basic_visualization(env, history):
    """向后兼容的基础可视化函数"""
    default_config = {'visualization': {'enabled': True, 'save_figures': True}}
    viz_manager = VisualizationManager(default_config)
    viz_manager.run_basic_visualization(env, history)

def create_interactive_visualization(env: 'PowerGridPartitionEnv', 
                                   comparison_df: pd.DataFrame) -> go.Figure:
    """向后兼容的交互式可视化函数"""
    default_config = {'visualization': {'enabled': True, 'save_figures': True}}
    viz_manager = VisualizationManager(default_config)
    return viz_manager.create_interactive_visualization(env, comparison_df)

def run_interactive_visualization(env, comparison_df):
    """向后兼容的交互式可视化函数"""
    default_config = {'visualization': {'enabled': True, 'save_figures': True}}
    viz_manager = VisualizationManager(default_config)
    viz_manager.run_interactive_visualization(env, comparison_df)

