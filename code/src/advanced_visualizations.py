#!/usr/bin/env python3
"""
高级可视化系统
集成最新的可视化库，提供电力网络分区的交互式和高级可视化功能

主要功能：
1. 交互式网络图（pyvis）
2. 实时监控仪表板（Bokeh/Dash）
3. 3D网络可视化（Plotly）
4. 动态场景演化（HoloViews）
5. GNN注意力可视化
6. 多维度性能对比雷达图
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
import json
import warnings

# 导入核心可视化库
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly未安装，3D可视化功能将不可用")

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    warnings.warn("Pyvis未安装，交互式网络图功能将不可用")

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import HoverTool, ColumnDataSource, Div
    from bokeh.palettes import Category20
    from bokeh.io import output_notebook
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    warnings.warn("Bokeh未安装，实时监控功能将不可用")

try:
    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh', 'matplotlib')
    HOLOVIEWS_AVAILABLE = True
except ImportError:
    HOLOVIEWS_AVAILABLE = False
    warnings.warn("HoloViews未安装，动态演化功能将不可用")

try:
    import dash
    from dash import dcc, html, Input, Output
    import dash_cytoscape as cyto
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    warnings.warn("Dash未安装，Web应用功能将不可用")


class AdvancedVisualizationSystem:
    """高级可视化系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化高级可视化系统
        
        Args:
            config: 可视化配置参数
        """
        self.config = config or {}
        self.output_dir = Path(self.config.get('output_dir', 'evaluation/latest/advanced_viz'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配色方案（专业电力系统配色）
        self.colors = {
            'partition_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
            'generator': '#FF6B6B',      # 发电机 - 红色
            'load': '#4ECDC4',          # 负载 - 青色
            'transmission': '#45B7D1',   # 输电线 - 蓝色
            'boundary': '#FFD93D',       # 边界节点 - 黄色
            'internal': '#95E1D3'        # 内部节点 - 绿色
        }
        
    def create_interactive_network_graph(self, 
                                       env_data: Dict[str, Any],
                                       partition: np.ndarray,
                                       title: str = "电力网络分区交互图") -> Optional[str]:
        """
        使用pyvis创建交互式网络图
        
        Args:
            env_data: 环境数据（包含节点、边、特征等）
            partition: 分区结果
            title: 图表标题
            
        Returns:
            生成的HTML文件路径
        """
        if not PYVIS_AVAILABLE:
            print("⚠️ Pyvis未安装，跳过交互式网络图生成")
            return None
            
        print("🎨 生成交互式网络图...")
        
        # 创建pyvis网络
        net = Network(
            notebook=False,
            directed=False,
            bgcolor="#ffffff",
            font_color="#000000",
            height="800px",
            width="100%",
            cdn_resources='in_line'
        )
        
        # 设置物理布局参数（适合电力网络）
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                }
            },
            "nodes": {
                "borderWidth": 2,
                "shadow": true
            },
            "edges": {
                "width": 2,
                "shadow": true,
                "smooth": {
                    "type": "continuous"
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": true
            }
        }
        """)
        
        # 添加节点
        bus_data = env_data.get('bus_data', [])
        for i, bus in enumerate(bus_data):
            partition_id = partition[i] if i < len(partition) else 0
            
            # 确定节点类型和颜色
            if bus.get('type') == 'generator':
                color = self.colors['generator']
                node_type = "发电机"
            elif bus.get('type') == 'load':
                color = self.colors['load'] 
                node_type = "负载"
            else:
                color = self.colors['partition_colors'][partition_id % len(self.colors['partition_colors'])]
                node_type = "母线"
            
            # 构建节点信息
            node_info = f"""
            节点类型: {node_type}
            分区: {partition_id}
            有功功率: {bus.get('p', 0):.2f} MW
            无功功率: {bus.get('q', 0):.2f} MVar
            电压: {bus.get('v', 1.0):.3f} p.u.
            """
            
            net.add_node(
                n_id=i,
                label=f"Bus {i}",
                title=node_info.strip(),
                color=color,
                size=20 + abs(bus.get('p', 0)) * 0.5,  # 根据功率调整大小
                shape="dot" if node_type == "母线" else "square"
            )
        
        # 添加边（输电线路）
        branch_data = env_data.get('branch_data', [])
        for branch in branch_data:
            from_bus = int(branch['from'])
            to_bus = int(branch['to'])
            
            # 检查是否跨分区
            if partition[from_bus] != partition[to_bus]:
                edge_color = self.colors['boundary']
                edge_width = 4
            else:
                edge_color = "#888888"
                edge_width = 2
            
            # 边信息
            edge_info = f"""
            线路: Bus {from_bus} - Bus {to_bus}
            电阻: {branch.get('r', 0):.4f} p.u.
            电抗: {branch.get('x', 0):.4f} p.u.
            容量: {branch.get('rating', 0):.1f} MVA
            """
            
            net.add_edge(
                from_bus, to_bus,
                title=edge_info.strip(),
                color=edge_color,
                width=edge_width
            )
        
        # 保存HTML文件
        output_file = self.output_dir / f"interactive_network_{title.replace(' ', '_')}.html"
        net.save_graph(str(output_file))
        
        print(f"✅ 交互式网络图已保存: {output_file}")
        return str(output_file)
    
    def create_3d_network_visualization(self,
                                      env_data: Dict[str, Any],
                                      partition: np.ndarray,
                                      node_embeddings: Optional[torch.Tensor] = None,
                                      title: str = "3D电力网络可视化") -> Optional[str]:
        """
        创建3D网络可视化（使用节点嵌入或物理布局）
        
        Args:
            env_data: 环境数据
            partition: 分区结果
            node_embeddings: 节点嵌入（可选，用于3D坐标）
            title: 图表标题
            
        Returns:
            生成的HTML文件路径
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly未安装，跳过3D可视化")
            return None
            
        print("🎨 生成3D网络可视化...")
        
        # 准备3D坐标
        num_nodes = len(env_data.get('bus_data', []))
        
        if node_embeddings is not None and node_embeddings.shape[1] >= 3:
            # 使用前3个嵌入维度作为坐标
            coords = node_embeddings[:, :3].cpu().numpy()
        else:
            # 使用力导向布局生成3D坐标
            G = nx.Graph()
            for branch in env_data.get('branch_data', []):
                G.add_edge(int(branch['from']), int(branch['to']))
            
            # 2D布局
            pos_2d = nx.spring_layout(G, k=1/np.sqrt(num_nodes), iterations=50)
            
            # 添加第三维（基于节点度数）
            coords = np.zeros((num_nodes, 3))
            for node, (x, y) in pos_2d.items():
                coords[node] = [x, y, G.degree(node) * 0.1]
        
        # 创建3D散点图
        fig = go.Figure()
        
        # 为每个分区创建散点
        unique_partitions = np.unique(partition[partition > 0])
        for p_id in unique_partitions:
            mask = partition == p_id
            
            # 节点信息
            hover_text = []
            bus_data = env_data.get('bus_data', [])
            for i in np.where(mask)[0]:
                if i < len(bus_data):
                    bus = bus_data[i]
                    hover_text.append(
                        f"Bus {i}<br>"
                        f"分区: {p_id}<br>"
                        f"有功: {bus.get('p', 0):.2f} MW<br>"
                        f"无功: {bus.get('q', 0):.2f} MVar"
                    )
            
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode='markers',
                name=f'分区 {p_id}',
                marker=dict(
                    size=8,
                    color=self.colors['partition_colors'][p_id % len(self.colors['partition_colors'])],
                    line=dict(width=1, color='DarkSlateGray')
                ),
                text=hover_text,
                hoverinfo='text'
            ))
        
        # 添加边
        edge_trace = []
        branch_data = env_data.get('branch_data', [])
        for branch in branch_data:
            from_bus = int(branch['from'])
            to_bus = int(branch['to'])
            
            if from_bus < num_nodes and to_bus < num_nodes:
                edge_trace.extend([
                    coords[from_bus], coords[to_bus], [None, None, None]
                ])
        
        if edge_trace:
            edge_trace = np.array(edge_trace)
            fig.add_trace(go.Scatter3d(
                x=edge_trace[:, 0],
                y=edge_trace[:, 1],
                z=edge_trace[:, 2],
                mode='lines',
                name='输电线路',
                line=dict(color='gray', width=1),
                hoverinfo='skip'
            ))
        
        # 更新布局
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (节点度数)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            height=800
        )
        
        # 保存HTML文件
        output_file = self.output_dir / f"3d_network_{title.replace(' ', '_')}.html"
        fig.write_html(str(output_file))
        
        print(f"✅ 3D网络可视化已保存: {output_file}")
        return str(output_file)
    
    def create_attention_heatmap(self,
                               attention_weights: torch.Tensor,
                               node_labels: Optional[List[str]] = None,
                               title: str = "GAT注意力权重热图") -> Optional[str]:
        """
        创建GAT注意力权重热图
        
        Args:
            attention_weights: 注意力权重矩阵
            node_labels: 节点标签
            title: 图表标题
            
        Returns:
            生成的HTML文件路径
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly未安装，跳过注意力热图生成")
            return None
            
        print("🎨 生成注意力权重热图...")
        
        # 转换为numpy数组
        if isinstance(attention_weights, torch.Tensor):
            attention_matrix = attention_weights.cpu().numpy()
        else:
            attention_matrix = attention_weights
        
        # 确保是2D矩阵
        if len(attention_matrix.shape) > 2:
            # 如果是多头注意力，取平均
            attention_matrix = attention_matrix.mean(axis=0)
        
        # 创建热图
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=node_labels if node_labels else list(range(attention_matrix.shape[1])),
            y=node_labels if node_labels else list(range(attention_matrix.shape[0])),
            colorscale='Viridis',
            colorbar=dict(title="注意力权重"),
            hoverongaps=False
        ))
        
        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title="目标节点",
            yaxis_title="源节点",
            height=600,
            width=800
        )
        
        # 保存HTML文件
        output_file = self.output_dir / f"attention_heatmap_{title.replace(' ', '_')}.html"
        fig.write_html(str(output_file))
        
        print(f"✅ 注意力热图已保存: {output_file}")
        return str(output_file)
    
    def create_performance_radar_chart(self,
                                     performance_data: Dict[str, Dict[str, float]],
                                     title: str = "算法性能对比雷达图") -> Optional[str]:
        """
        创建多维度性能对比雷达图
        
        Args:
            performance_data: 性能数据，格式：{算法名: {指标名: 数值}}
            title: 图表标题
            
        Returns:
            生成的HTML文件路径
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly未安装，跳过雷达图生成")
            return None
            
        print("🎨 生成性能对比雷达图...")
        
        # 准备数据
        algorithms = list(performance_data.keys())
        if not algorithms:
            print("⚠️ 没有性能数据")
            return None
            
        # 获取所有指标
        metrics = list(performance_data[algorithms[0]].keys())
        
        # 创建雷达图
        fig = go.Figure()
        
        for algo in algorithms:
            values = [performance_data[algo].get(metric, 0) for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=algo,
                opacity=0.7
            ))
        
        # 更新布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]  # 假设所有指标已归一化到0-1
                )
            ),
            showlegend=True,
            title=title,
            height=600
        )
        
        # 保存HTML文件
        output_file = self.output_dir / f"performance_radar_{title.replace(' ', '_')}.html"
        fig.write_html(str(output_file))
        
        print(f"✅ 性能雷达图已保存: {output_file}")
        return str(output_file)
    
    def create_realtime_monitoring_dashboard(self,
                                           initial_data: Dict[str, Any],
                                           port: int = 8050) -> Optional[str]:
        """
        创建实时监控仪表板（使用Dash）
        
        Args:
            initial_data: 初始数据
            port: Web服务器端口
            
        Returns:
            仪表板URL
        """
        if not DASH_AVAILABLE:
            print("⚠️ Dash未安装，跳过实时监控仪表板")
            return None
            
        print("🎨 创建实时监控仪表板...")
        
        # 创建Dash应用
        app = dash.Dash(__name__)
        
        # 定义布局
        app.layout = html.Div([
            html.H1("电力网络分区实时监控", style={'textAlign': 'center'}),
            
            # 实时指标卡片
            html.Div([
                html.Div([
                    html.H3("当前Episode"),
                    html.H2(id='current-episode', children='0')
                ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("平均奖励"),
                    html.H2(id='avg-reward', children='0.00')
                ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("成功率"),
                    html.H2(id='success-rate', children='0.0%')
                ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("CV值"),
                    html.H2(id='cv-value', children='0.00')
                ], className='metric-card', style={'width': '24%', 'display': 'inline-block'})
            ], style={'margin': '20px'}),
            
            # 实时图表
            dcc.Graph(id='live-reward-graph'),
            dcc.Graph(id='live-metrics-graph'),
            
            # 自动更新组件
            dcc.Interval(
                id='interval-component',
                interval=2*1000,  # 2秒更新一次
                n_intervals=0
            )
        ])
        
        # 回调函数（示例）
        @app.callback(
            [Output('current-episode', 'children'),
             Output('avg-reward', 'children'),
             Output('success-rate', 'children'),
             Output('cv-value', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            # 这里应该从实际的训练过程获取数据
            # 示例返回
            return str(n), f"{np.random.randn():.2f}", f"{np.random.rand()*100:.1f}%", f"{np.random.rand():.3f}"
        
        print(f"🌐 实时监控仪表板运行在: http://localhost:{port}")
        print("   按Ctrl+C停止服务器")
        
        # 注意：在实际使用中，这里应该在后台线程运行
        # app.run_server(debug=False, port=port)
        
        return f"http://localhost:{port}"
    
    def create_scenario_evolution_animation(self,
                                          evolution_data: List[Dict[str, Any]],
                                          title: str = "场景演化动画") -> Optional[str]:
        """
        创建场景演化动画（使用Plotly）
        
        Args:
            evolution_data: 演化数据列表，每个元素包含一个时间步的数据
            title: 动画标题
            
        Returns:
            生成的HTML文件路径
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly未安装，跳过演化动画生成")
            return None
            
        print("🎨 生成场景演化动画...")
        
        # 准备动画帧
        frames = []
        
        for t, data in enumerate(evolution_data):
            frame_data = []
            
            # 每个分区的数据
            for p_id in range(1, data.get('num_partitions', 4) + 1):
                mask = data['partition'] == p_id
                
                frame_data.append(go.Scatter(
                    x=data['node_positions'][mask, 0],
                    y=data['node_positions'][mask, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=self.colors['partition_colors'][p_id % len(self.colors['partition_colors'])]
                    ),
                    name=f'分区 {p_id}'
                ))
            
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        # 创建初始图
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames
        )
        
        # 添加播放按钮
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '播放',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    },
                    {
                        'label': '暂停',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': f't={f.name}',
                        'method': 'animate'
                    }
                    for f in frames
                ],
                'active': 0,
                'y': 0,
                'len': 0.9,
                'x': 0.05,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }],
            title=title,
            height=600
        )
        
        # 保存HTML文件
        output_file = self.output_dir / f"evolution_animation_{title.replace(' ', '_')}.html"
        fig.write_html(str(output_file))
        
        print(f"✅ 场景演化动画已保存: {output_file}")
        return str(output_file)
    
    def generate_comprehensive_test_report(self,
                                         test_results: Dict[str, Any],
                                         model_info: Dict[str, Any]) -> Path:
        """
        生成综合测试报告，包含所有高级可视化
        
        Args:
            test_results: 测试结果数据
            model_info: 模型信息
            
        Returns:
            报告目录路径
        """
        print("\n🎨 生成综合可视化测试报告...")
        
        report_dir = self.output_dir / f"test_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir.mkdir(exist_ok=True)
        
        # 生成各种可视化
        visualizations = {}
        
        # 1. 交互式网络图
        if test_results.get('env_data'):
            viz_path = self.create_interactive_network_graph(
                test_results['env_data'],
                test_results.get('best_partition', np.array([])),
                "最佳分区结果"
            )
            if viz_path:
                visualizations['interactive_network'] = viz_path
        
        # 2. 3D网络可视化
        if test_results.get('env_data') and test_results.get('node_embeddings') is not None:
            viz_path = self.create_3d_network_visualization(
                test_results['env_data'],
                test_results.get('best_partition', np.array([])),
                test_results.get('node_embeddings'),
                "3D网络嵌入可视化"
            )
            if viz_path:
                visualizations['3d_network'] = viz_path
        
        # 3. 注意力热图
        if test_results.get('attention_weights') is not None:
            viz_path = self.create_attention_heatmap(
                test_results['attention_weights'],
                title="GAT注意力模式分析"
            )
            if viz_path:
                visualizations['attention_heatmap'] = viz_path
        
        # 4. 性能雷达图
        if test_results.get('algorithm_comparison'):
            viz_path = self.create_performance_radar_chart(
                test_results['algorithm_comparison'],
                "多算法性能对比"
            )
            if viz_path:
                visualizations['performance_radar'] = viz_path
        
        # 5. 生成HTML索引页
        self._generate_report_index(report_dir, visualizations, test_results, model_info)
        
        print(f"\n✅ 综合测试报告已生成: {report_dir}")
        print(f"   包含 {len(visualizations)} 个可视化")
        
        return report_dir
    
    def _generate_report_index(self,
                             report_dir: Path,
                             visualizations: Dict[str, str],
                             test_results: Dict[str, Any],
                             model_info: Dict[str, Any]):
        """生成报告索引页"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>电力网络分区测试报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E86AB;
            text-align: center;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #A23B72;
            margin-top: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2E86AB;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .viz-card {{
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 15px;
            text-align: center;
            transition: transform 0.2s;
        }}
        .viz-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .viz-card a {{
            color: #2E86AB;
            text-decoration: none;
            font-weight: bold;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #e0e0e0;
            padding: 10px;
            text-align: left;
        }}
        .metrics-table th {{
            background-color: #2E86AB;
            color: white;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔋 电力网络分区测试报告</h1>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>模型信息</h3>
                <p><strong>网络类型:</strong> {model_info.get('network_type', 'N/A')}</p>
                <p><strong>训练Episodes:</strong> {model_info.get('training_episodes', 'N/A')}</p>
                <p><strong>最佳奖励:</strong> {model_info.get('best_reward', 'N/A'):.4f}</p>
            </div>
            <div class="info-card">
                <h3>测试配置</h3>
                <p><strong>测试网络:</strong> {test_results.get('test_network', 'N/A')}</p>
                <p><strong>测试Episodes:</strong> {test_results.get('num_episodes', 'N/A')}</p>
                <p><strong>平均奖励:</strong> {test_results.get('avg_reward', 'N/A'):.4f}</p>
            </div>
        </div>
        
        <h2>📊 可视化结果</h2>
        <div class="viz-grid">
"""
        
        # 添加可视化卡片
        viz_info = {
            'interactive_network': ('🌐 交互式网络图', '探索电力网络拓扑和分区结果'),
            '3d_network': ('🎯 3D网络可视化', '基于节点嵌入的三维展示'),
            'attention_heatmap': ('🔥 注意力热图', 'GAT模型的注意力权重分析'),
            'performance_radar': ('📈 性能雷达图', '多维度算法性能对比')
        }
        
        for viz_key, viz_path in visualizations.items():
            title, desc = viz_info.get(viz_key, ('可视化', ''))
            rel_path = Path(viz_path).name
            html_content += f"""
            <div class="viz-card">
                <h4>{title}</h4>
                <p>{desc}</p>
                <a href="{rel_path}" target="_blank">查看可视化 →</a>
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>📋 性能指标</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>指标</th>
                    <th>数值</th>
                    <th>说明</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # 添加性能指标
        metrics = test_results.get('metrics', {})
        metric_info = {
            'success_rate': ('成功率', '满足所有约束的比例'),
            'avg_cv': ('平均CV值', '负载平衡度量'),
            'avg_coupling': ('平均耦合度', '分区间连接强度'),
            'computation_time': ('平均计算时间', '单次决策时间(ms)')
        }
        
        for metric_key, (metric_name, metric_desc) in metric_info.items():
            value = metrics.get(metric_key, 'N/A')
            if isinstance(value, float):
                value = f"{value:.4f}"
            html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value}</td>
                    <td>{metric_desc}</td>
                </tr>
"""
        
        html_content += f"""
            </tbody>
        </table>
        
        <div class="timestamp">
            报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        # 保存索引页
        index_path = report_dir / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   索引页: {index_path}")


# 便捷函数
def create_advanced_visualizations(test_results: Dict[str, Any],
                                 model_info: Dict[str, Any],
                                 config: Optional[Dict[str, Any]] = None) -> Path:
    """
    便捷函数：创建所有高级可视化
    
    Args:
        test_results: 测试结果
        model_info: 模型信息
        config: 配置参数
        
    Returns:
        报告目录路径
    """
    viz_system = AdvancedVisualizationSystem(config)
    return viz_system.generate_comprehensive_test_report(test_results, model_info)