import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from matplotlib.patches import Patch
from typing import Optional, Dict, List
import pandas as pd
import seaborn as sns
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

def get_color_palette(num_colors: int) -> List[str]:
    """
    动态生成颜色方案
    
    参数:
        num_colors: 需要的颜色数量
        
    返回:
        颜色列表 (HEX格式)
    """
    # 使用Seaborn的husl色板，适合类别区分
    palette = sns.color_palette("husl", num_colors)
    # 转换为matplotlib和plotly都兼容的HEX格式
    return [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in palette]

def visualize_partition(env: 'PowerGridPartitionEnv', title: str = "Power Grid Partition",
                           save_path: Optional[str] = None, show_metrics: bool = True):
        """
        可视化电网分区结果
        """
        try:
            import gc
            # 创建图形
            if show_metrics:
                fig = plt.figure(figsize=(16, 10))
                gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[3, 1])
                ax_main = fig.add_subplot(gs[:, 0])
                ax_metrics = fig.add_subplot(gs[0, 1])
                ax_load = fig.add_subplot(gs[0, 2])
                ax_coupling = fig.add_subplot(gs[1, 1:])
            else:
                fig, ax_main = plt.subplots(figsize=(12, 10))
            
            # 创建NetworkX图
            G = nx.Graph()
            edge_array = env.edge_index.cpu().numpy()
            
            # 添加节点
            for i in range(env.N):
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
            colors = ['#E0E0E0'] + get_color_palette(env.K)
            
            # 节点颜色和大小
            node_colors = [colors[env.z[i].item()] for i in range(env.N)]
            node_sizes = [300 + env.Pd[i].item() * 500 for i in range(env.N)]
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax_main)
            
            # 高亮跨区域边
            inter_edges = []
            for u, v in edge_set:
                if env.z[u] > 0 and env.z[v] > 0 and env.z[u] != env.z[v]:
                    inter_edges.append((u, v))
            
            nx.draw_networkx_edges(G, pos, edgelist=inter_edges, edge_color='red',
                                  width=2, alpha=0.6, ax=ax_main)
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                  alpha=0.9, ax=ax_main)
            
            # 节点标签
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax_main)
            
            # 添加图例
            legend_elements = []
            for k in range(env.K + 1):
                count = (env.z == k).sum().item()
                if k == 0:
                    label = f'Unassigned ({count} nodes)'
                else:
                    label = f'Region {k} ({count} nodes)'
                legend_elements.append(mpatches.Patch(color=colors[k], label=label))
            
            ax_main.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            ax_main.set_title(title, fontsize=16, fontweight='bold')
            ax_main.axis('off')
            
            if show_metrics:
                # 显示指标
                metrics = env.current_metrics
            
            # 指标表格
            ax_metrics.axis('off')
            ax_metrics.set_title('Partition Metrics', fontsize=14, fontweight='bold')
            
            metric_data = [
                ['Load CV', f'{metrics.load_cv:.4f}'],
                ['Load Gini', f'{metrics.load_gini:.4f}'],
                ['Total Coupling', f'{metrics.total_coupling:.4f}'],
                ['Inter-region Lines', f'{metrics.inter_region_lines}'],
                ['Connectivity', f'{metrics.connectivity:.2f}'],
                ['Modularity', f'{metrics.modularity:.4f}']
            ]
            
            table = ax_metrics.table(cellText=metric_data, loc='center',
                                    cellLoc='left', colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # 负荷分布图
            ax_load.set_title('Load Distribution', fontsize=12)
            region_loads = []
            region_labels = []
            
            for k in range(1, env.K + 1):
                mask = (env.z == k)
                if mask.any():
                    load = env.Pd[mask].sum().item()
                    gen = env.Pg[mask].sum().item()
                    region_loads.append([load, gen])
                    region_labels.append(f'R{k}')
            
            if region_loads:
                region_loads = np.array(region_loads)
                x = np.arange(len(region_labels))
                width = 0.35
                
                bars1 = ax_load.bar(x - width/2, region_loads[:, 0], width,
                                   label='Load', color='lightcoral')
                bars2 = ax_load.bar(x + width/2, region_loads[:, 1], width,
                                   label='Generation', color='lightgreen')
                
                ax_load.set_ylabel('Power (p.u.)')
                ax_load.set_xticks(x)
                ax_load.set_xticklabels(region_labels)
                ax_load.legend()
                
                # 添加数值标签
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax_load.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            # 耦合矩阵热图
            ax_coupling.set_title('Region Coupling Matrix', fontsize=12)
            coupling_matrix = np.zeros((env.K, env.K))
            
            for i in range(edge_array.shape[1]):
                u, v = edge_array[0, i], edge_array[1, i]
                if env.z[u] > 0 and env.z[v] > 0 and env.z[u] != env.z[v]:
                    coupling_matrix[env.z[u]-1, env.z[v]-1] += env.admittance[i].item()
            
            sns.heatmap(coupling_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=[f'R{i+1}' for i in range(env.K)],
                       yticklabels=[f'R{i+1}' for i in range(env.K)],
                       ax=ax_coupling, cbar_kws={'label': 'Coupling Strength'})
        
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Figure saved to {save_path}")
            
            plt.show()
            
            # 清理内存
            gc.collect()
            
        except Exception as e:
            print(f"⚠️ 可视化出错: {e}")
            # 尝试清理任何已创建的图形
            try:
                plt.close('all')
                gc.collect()
            except:
                pass


def plot_training_curves(history: Dict[str, List[float]], env_N: int = None, save_path: Optional[str] = None):
        """
        绘制训练曲线
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 奖励曲线
        ax = axes[0, 0]
        episodes = range(len(history['episode_rewards']))
        ax.plot(episodes, history['episode_rewards'], alpha=0.6, label='Episode Reward')
        
        # 移动平均
        if len(history['episode_rewards']) > 10:
            window = min(20, len(history['episode_rewards']) // 5)
            moving_avg = pd.Series(history['episode_rewards']).rolling(window).mean()
            ax.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Load CV曲线
        ax = axes[0, 1]
        ax.plot(episodes, history['load_cv'], 'g-', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Load CV')
        ax.set_title('Load Balance (CV)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(history['load_cv']) * 1.1)
        
        # 3. 耦合度曲线
        ax = axes[1, 0]
        ax.plot(episodes, history['total_coupling'], 'b-', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Coupling')
        ax.set_title('Inter-region Coupling')
        ax.grid(True, alpha=0.3)
        
        # 4. Episode长度
        ax = axes[1, 1]
        ax.plot(episodes, history['episode_lengths'], 'm-', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        
        # 添加完成线（如果提供了env_N）
        if env_N is not None:
            ax.axhline(y=env_N, color='r', linestyle='--', alpha=0.5, label='Complete')
            ax.legend()
        
        fig.suptitle('Training Process Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"🖼️ 训练曲线图已保存到: {save_path}")
        plt.close()


def run_basic_visualization(env, history):
    """可视化基础结果"""
    print("\n📈 生成基础可视化图表...")
    
    # 可视化最终分区
    visualize_partition(env, "Final Partition Result (RL)", "figures/rl_partition_result.png")
    
    # 绘制训练曲线
    plot_training_curves(history, env_N=env.N, save_path="figures/training_curves.png")

def create_interactive_visualization(env: 'PowerGridPartitionEnv', 
                                   comparison_df: pd.DataFrame) -> go.Figure:
        """
        创建交互式可视化（使用Plotly）
        """
        import torch  # 添加缺失的torch导入
        
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
        colors = ['#E0E0E0'] + get_color_palette(env.K)
        
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
        bar_colors = get_color_palette(env.K)
        
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
            height=800,
            template='plotly_white'
        )
        
        # 更新子图标题
        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, row=1, col=1)
        
        return fig


def run_interactive_visualization(env, comparison_df):
    """创建并保存交互式可视化图表"""
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly未安装，跳过交互式可视化。")
        return
        
    print("📊 生成交互式对比图表...")
    fig = create_interactive_visualization(env, comparison_df)
    
    # 保存为HTML文件
    fig.write_html("figures/interactive_partition_analysis.html")
    print(f"🌐 交互式图表已保存到: figures/interactive_partition_analysis.html")

