#!/usr/bin/env python3
"""
增强的测试可视化集成
将高级可视化功能集成到现有的测试流程中
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time

# 添加路径
sys.path.append(str(Path(__file__).parent))

# 导入现有模块
from visualization import VisualizationManager
from plotly_chart_factory import PlotlyChartFactory
from html_dashboard_generator import HTMLDashboardGenerator
from advanced_visualizations import AdvancedVisualizationSystem, create_advanced_visualizations


class EnhancedTestVisualizer:
    """增强的测试可视化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增强测试可视化器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 初始化各个可视化组件
        self.basic_viz = VisualizationManager(config)
        self.chart_factory = PlotlyChartFactory(config.get('chart_factory', {}))
        self.dashboard_gen = HTMLDashboardGenerator(config.get('dashboard', {}))
        self.advanced_viz = AdvancedVisualizationSystem(config.get('advanced', {}))
        
        # 输出目录
        self.output_dir = Path(self.config.get('output_dir', 'evaluation/latest/enhanced_viz'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_test_results(self, 
                             test_results: Dict[str, Any],
                             model_info: Dict[str, Any],
                             env_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        生成完整的测试结果可视化
        
        Args:
            test_results: 测试结果数据
            model_info: 模型信息
            env_data: 环境数据（可选）
            
        Returns:
            生成的可视化文件路径字典
        """
        print("\n🎨 开始生成增强的测试可视化...")
        
        generated_files = {}
        
        # 1. 基础可视化
        print("\n📊 [1/6] 生成基础可视化...")
        basic_files = self._generate_basic_visualizations(test_results, env_data)
        generated_files.update(basic_files)
        
        # 2. 性能对比图表
        print("\n📈 [2/6] 生成性能对比图表...")
        perf_files = self._generate_performance_charts(test_results)
        generated_files.update(perf_files)
        
        # 3. 交互式网络图
        print("\n🌐 [3/6] 生成交互式网络图...")
        network_files = self._generate_network_visualizations(test_results, env_data)
        generated_files.update(network_files)
        
        # 4. 算法对比分析
        print("\n🔬 [4/6] 生成算法对比分析...")
        comparison_files = self._generate_algorithm_comparisons(test_results)
        generated_files.update(comparison_files)
        
        # 5. 场景分析可视化
        print("\n🎭 [5/6] 生成场景分析可视化...")
        scenario_files = self._generate_scenario_analysis(test_results)
        generated_files.update(scenario_files)
        
        # 6. 综合HTML报告
        print("\n📝 [6/6] 生成综合HTML报告...")
        report_file = self._generate_comprehensive_report(
            test_results, model_info, generated_files
        )
        generated_files['comprehensive_report'] = str(report_file)
        
        print(f"\n✅ 测试可视化完成！生成了 {len(generated_files)} 个文件")
        print(f"   输出目录: {self.output_dir}")
        
        return generated_files
    
    def _generate_basic_visualizations(self, 
                                     test_results: Dict[str, Any],
                                     env_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """生成基础可视化"""
        files = {}
        
        # 训练曲线
        if 'episode_rewards' in test_results:
            fig = self.chart_factory.create_training_curves({
                'episode_rewards': test_results['episode_rewards'],
                'episode_lengths': test_results.get('episode_lengths', []),
                'success_rates': test_results.get('success_rates', [])
            })
            
            if fig:
                output_file = self.output_dir / "training_curves.html"
                fig.write_html(str(output_file))
                files['training_curves'] = str(output_file)
        
        # 分区可视化（如果有环境数据）
        if env_data and hasattr(self.basic_viz, 'visualize_partition'):
            try:
                # 这里需要创建一个模拟的env对象或直接使用数据
                output_file = self.output_dir / "partition_layout.png"
                # self.basic_viz.visualize_partition(env, save_path=str(output_file))
                # files['partition_layout'] = str(output_file)
            except Exception as e:
                print(f"   ⚠️ 分区可视化失败: {e}")
        
        return files
    
    def _generate_performance_charts(self, test_results: Dict[str, Any]) -> Dict[str, str]:
        """生成性能对比图表"""
        files = {}
        
        # 性能指标时间序列
        if 'metrics_history' in test_results:
            metrics = test_results['metrics_history']
            
            # CV值变化曲线
            if 'cv_values' in metrics:
                fig = self.chart_factory.create_metrics_timeline(
                    metrics={'CV值': metrics['cv_values']},
                    title="CV值变化趋势"
                )
                if fig:
                    output_file = self.output_dir / "cv_timeline.html"
                    fig.write_html(str(output_file))
                    files['cv_timeline'] = str(output_file)
            
            # 耦合度变化
            if 'coupling_ratios' in metrics:
                fig = self.chart_factory.create_metrics_timeline(
                    metrics={'耦合度': metrics['coupling_ratios']},
                    title="耦合度变化趋势"
                )
                if fig:
                    output_file = self.output_dir / "coupling_timeline.html"
                    fig.write_html(str(output_file))
                    files['coupling_timeline'] = str(output_file)
        
        # 计算时间分布
        if 'computation_times' in test_results:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=test_results['computation_times'],
                nbinsx=30,
                name='计算时间分布',
                marker_color='#2E86AB'
            ))
            
            fig.update_layout(
                title="Agent决策时间分布",
                xaxis_title="计算时间 (ms)",
                yaxis_title="频次",
                showlegend=False
            )
            
            output_file = self.output_dir / "computation_time_dist.html"
            fig.write_html(str(output_file))
            files['computation_time_dist'] = str(output_file)
        
        return files
    
    def _generate_network_visualizations(self, 
                                       test_results: Dict[str, Any],
                                       env_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """生成网络可视化"""
        files = {}
        
        if not env_data:
            return files
        
        # 交互式网络图
        if 'best_partition' in test_results:
            output_file = self.advanced_viz.create_interactive_network_graph(
                env_data,
                test_results['best_partition'],
                title="最优分区方案"
            )
            if output_file:
                files['interactive_network'] = output_file
        
        # 3D网络可视化
        if 'node_embeddings' in test_results:
            output_file = self.advanced_viz.create_3d_network_visualization(
                env_data,
                test_results.get('best_partition', np.array([])),
                test_results['node_embeddings'],
                title="节点嵌入3D可视化"
            )
            if output_file:
                files['3d_network'] = output_file
        
        # GAT注意力热图
        if 'attention_weights' in test_results:
            output_file = self.advanced_viz.create_attention_heatmap(
                test_results['attention_weights'],
                title="GAT注意力权重分析"
            )
            if output_file:
                files['attention_heatmap'] = output_file
        
        return files
    
    def _generate_algorithm_comparisons(self, test_results: Dict[str, Any]) -> Dict[str, str]:
        """生成算法对比分析"""
        files = {}
        
        # 算法性能对比
        if 'algorithm_comparison' in test_results:
            # 雷达图对比
            output_file = self.advanced_viz.create_performance_radar_chart(
                test_results['algorithm_comparison'],
                title="算法性能多维对比"
            )
            if output_file:
                files['performance_radar'] = output_file
            
            # 性能条形图
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            comparison_data = test_results['algorithm_comparison']
            algorithms = list(comparison_data.keys())
            
            if algorithms:
                metrics = list(comparison_data[algorithms[0]].keys())
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=metrics[:4] if len(metrics) >= 4 else metrics
                )
                
                for i, metric in enumerate(metrics[:4]):
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    values = [comparison_data[algo].get(metric, 0) for algo in algorithms]
                    
                    fig.add_trace(
                        go.Bar(
                            x=algorithms,
                            y=values,
                            name=metric,
                            marker_color=self.chart_factory.color_sequence[i % len(self.chart_factory.color_sequence)]
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    title_text="算法性能详细对比",
                    showlegend=False,
                    height=600
                )
                
                output_file = self.output_dir / "algorithm_comparison_bars.html"
                fig.write_html(str(output_file))
                files['algorithm_comparison_bars'] = str(output_file)
        
        # 收敛速度对比
        if 'convergence_data' in test_results:
            fig = go.Figure()
            
            for algo, data in test_results['convergence_data'].items():
                fig.add_trace(go.Scatter(
                    x=list(range(len(data))),
                    y=data,
                    mode='lines',
                    name=algo,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="算法收敛速度对比",
                xaxis_title="迭代次数",
                yaxis_title="目标函数值",
                hovermode='x unified'
            )
            
            output_file = self.output_dir / "convergence_comparison.html"
            fig.write_html(str(output_file))
            files['convergence_comparison'] = str(output_file)
        
        return files
    
    def _generate_scenario_analysis(self, test_results: Dict[str, Any]) -> Dict[str, str]:
        """生成场景分析可视化"""
        files = {}
        
        # 场景性能热图
        if 'scenario_performance' in test_results:
            import plotly.graph_objects as go
            
            scenario_data = test_results['scenario_performance']
            scenarios = list(scenario_data.keys())
            metrics = list(scenario_data[scenarios[0]].keys()) if scenarios else []
            
            # 创建热图数据
            z_data = []
            for scenario in scenarios:
                row = [scenario_data[scenario].get(metric, 0) for metric in metrics]
                z_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=metrics,
                y=scenarios,
                colorscale='RdYlGn',
                colorbar=dict(title="性能分数")
            ))
            
            fig.update_layout(
                title="不同场景下的性能表现",
                xaxis_title="性能指标",
                yaxis_title="场景类型"
            )
            
            output_file = self.output_dir / "scenario_performance_heatmap.html"
            fig.write_html(str(output_file))
            files['scenario_heatmap'] = str(output_file)
        
        # 场景分布饼图
        if 'scenario_distribution' in test_results:
            import plotly.express as px
            
            dist_data = test_results['scenario_distribution']
            
            fig = px.pie(
                values=list(dist_data.values()),
                names=list(dist_data.keys()),
                title="测试场景分布",
                color_discrete_sequence=self.chart_factory.color_sequence
            )
            
            output_file = self.output_dir / "scenario_distribution.html"
            fig.write_html(str(output_file))
            files['scenario_distribution'] = str(output_file)
        
        return files
    
    def _generate_comprehensive_report(self,
                                     test_results: Dict[str, Any],
                                     model_info: Dict[str, Any],
                                     generated_files: Dict[str, str]) -> Path:
        """生成综合HTML报告"""
        
        # 准备报告数据
        report_data = {
            'title': '电力网络分区增强测试报告',
            'model_info': model_info,
            'test_summary': {
                'total_episodes': test_results.get('num_episodes', 0),
                'avg_reward': test_results.get('avg_reward', 0),
                'success_rate': test_results.get('success_rate', 0),
                'best_cv': test_results.get('best_cv', 0),
                'avg_computation_time': np.mean(test_results.get('computation_times', [0]))
            },
            'visualizations': generated_files,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 使用高级可视化系统的报告生成器
        report_dir = self.advanced_viz.generate_comprehensive_test_report(
            test_results, model_info
        )
        
        # 添加额外的可视化链接
        index_file = report_dir / "index.html"
        
        # 这里可以修改index.html添加更多内容
        
        return index_file
    
    def create_live_monitoring_dashboard(self, port: int = 8050):
        """
        创建实时监控仪表板
        
        Args:
            port: Web服务器端口
        """
        print(f"\n🌐 启动实时监控仪表板在端口 {port}...")
        
        # 使用高级可视化系统的实时监控功能
        dashboard_url = self.advanced_viz.create_realtime_monitoring_dashboard(
            initial_data={},
            port=port
        )
        
        return dashboard_url


# 集成到test.py的函数
def enhance_test_visualization(test_results: Dict[str, Any],
                             model_info: Dict[str, Any],
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    便捷函数：为test.py提供增强的可视化功能
    
    Args:
        test_results: 测试结果
        model_info: 模型信息  
        config: 配置参数
        
    Returns:
        生成的文件路径字典
    """
    visualizer = EnhancedTestVisualizer(config)
    
    # 处理环境数据
    env_data = None
    if 'env_data' in test_results:
        env_data = test_results['env_data']
    elif 'bus_data' in test_results and 'branch_data' in test_results:
        env_data = {
            'bus_data': test_results['bus_data'],
            'branch_data': test_results['branch_data']
        }
    
    return visualizer.visualize_test_results(test_results, model_info, env_data)


# 示例：如何在test.py中使用
def example_usage():
    """示例用法"""
    
    # 模拟测试结果
    test_results = {
        'num_episodes': 100,
        'avg_reward': -2.5,
        'success_rate': 0.85,
        'best_cv': 0.25,
        'episode_rewards': np.random.randn(100).cumsum(),
        'computation_times': np.random.exponential(10, 100),
        'best_partition': np.random.randint(1, 5, 57),
        'algorithm_comparison': {
            'RL Agent': {'CV': 0.25, '耦合度': 0.15, '成功率': 0.85, '计算时间': 10},
            'Spectral': {'CV': 0.35, '耦合度': 0.25, '成功率': 0.65, '计算时间': 5},
            'K-means': {'CV': 0.45, '耦合度': 0.35, '成功率': 0.45, '计算时间': 3}
        },
        'scenario_performance': {
            '正常': {'CV': 0.25, '耦合度': 0.15, '成功率': 0.90},
            '高负载': {'CV': 0.30, '耦合度': 0.20, '成功率': 0.80},
            'N-1故障': {'CV': 0.35, '耦合度': 0.25, '成功率': 0.70}
        }
    }
    
    model_info = {
        'network_type': 'IEEE57',
        'training_episodes': 3000,
        'best_reward': -1.8
    }
    
    # 生成增强可视化
    generated_files = enhance_test_visualization(test_results, model_info)
    
    print("\n生成的可视化文件:")
    for name, path in generated_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    example_usage()