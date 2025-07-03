#!/usr/bin/env python3
"""
HTML仪表板生成器

专门负责生成交互式HTML仪表板：
- 自包含的HTML文件生成
- 集成Plotly图表和交互功能
- 响应式布局和专业样式
- 数据导出和分享功能
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from jinja2 import Template
import plotly
import pandas as pd
import numpy as np

# 导入我们的组件
try:
    from .metrics_analyzer import MetricsAnalyzer
    from .plotly_chart_factory import PlotlyChartFactory
except ImportError:
    # 处理直接导入的情况
    from metrics_analyzer import MetricsAnalyzer
    from plotly_chart_factory import PlotlyChartFactory


class HTMLDashboardGenerator:
    """HTML仪表板生成器 - 生成完整的交互式HTML报告"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化HTML仪表板生成器
        
        Args:
            config: 生成器配置参数
        """
        self.config = config or {}
        self.analyzer = MetricsAnalyzer(config.get('analyzer', {}))
        self.chart_factory = PlotlyChartFactory(config.get('chart_factory', {}))
        
        # 输出配置
        self.output_dir = Path(self.config.get('output_dir', 'output/dashboards'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # HTML模板配置
        self.template_config = self.config.get('template', {})
        self.chart_config = self.chart_factory.get_chart_config() or {}
        
        # 数据优化配置
        self.max_data_points = self.config.get('max_data_points', 2000)
        self.enable_compression = self.config.get('enable_compression', True)
        
    def generate_training_dashboard(self, training_data: Dict[str, Any], 
                                  output_filename: Optional[str] = None) -> Path:
        """
        生成训练仪表板
        
        Args:
            training_data: 训练数据字典
            output_filename: 输出文件名
            
        Returns:
            生成的HTML文件路径
        """
        print("🎨 开始生成HTML训练仪表板...")
        
        # 准备数据
        dashboard_data = self._prepare_dashboard_data(training_data)
        
        # 生成图表
        charts = self._generate_all_charts(dashboard_data)
        
        # 分析统计数据
        analysis_results = self._generate_analysis_results(dashboard_data)
        
        # 生成HTML
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"training_dashboard_{timestamp}.html"
            
        output_path = self.output_dir / output_filename
        
        self._generate_html_file(
            output_path=output_path,
            dashboard_data=dashboard_data,
            charts=charts,
            analysis_results=analysis_results
        )
        
        print(f"✅ HTML仪表板生成完成: {output_path}")
        return output_path
    
    def generate_comparison_dashboard(self, comparison_data: Dict[str, Dict[str, Any]], 
                                    output_filename: Optional[str] = None) -> Path:
        """
        生成对比分析仪表板
        
        Args:
            comparison_data: 对比数据字典 {method_name: training_data}
            output_filename: 输出文件名
            
        Returns:
            生成的HTML文件路径
        """
        print("🎨 开始生成HTML对比仪表板...")
        
        # 处理对比数据
        processed_data = {}
        for method, data in comparison_data.items():
            processed_data[method] = self._prepare_dashboard_data(data)
        
        # 生成对比图表
        comparison_charts = self._generate_comparison_charts(processed_data)
        
        # 生成HTML
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"comparison_dashboard_{timestamp}.html"
            
        output_path = self.output_dir / output_filename
        
        self._generate_comparison_html_file(
            output_path=output_path,
            comparison_data=processed_data,
            charts=comparison_charts
        )
        
        print(f"✅ HTML对比仪表板生成完成: {output_path}")
        return output_path
    
    def _prepare_dashboard_data(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备仪表板数据"""
        # 提取基础数据
        episodes = training_data.get('episodes', [])
        rewards = training_data.get('rewards', [])
        
        # 优化数据点数量
        if self.enable_compression and len(episodes) > self.max_data_points:
            episodes, rewards = self._compress_data(episodes, rewards)
        
        # 计算移动平均
        moving_averages = self._calculate_moving_averages(rewards)
        
        # 准备完整数据集
        dashboard_data = {
            'episodes': episodes,
            'rewards': rewards,
            'moving_averages': moving_averages,
            'actor_losses': training_data.get('actor_losses', []),
            'critic_losses': training_data.get('critic_losses', []),
            'entropies': training_data.get('entropies', []),
            'cv_values': training_data.get('cv_values', []),
            'coupling_ratios': training_data.get('coupling_ratios', []),
            'stages': training_data.get('stages', []),
            'transitions': training_data.get('transitions', []),
            'success_rates': training_data.get('success_rates', []),
            'training_time': training_data.get('training_time', 0),
            'config': training_data.get('config', {}),
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'total_episodes': len(episodes),
                'data_compressed': len(training_data.get('episodes', [])) > self.max_data_points
            }
        }
        
        return dashboard_data
    
    def _generate_all_charts(self, dashboard_data: Dict[str, Any]) -> Dict[str, str]:
        """生成所有图表"""
        charts = {}
        
        # 1. 奖励分析图表
        if dashboard_data['rewards']:
            # 检测异常
            anomalies = self.analyzer.detect_anomalies(dashboard_data['rewards'])
            
            reward_chart = self.chart_factory.create_reward_analysis_chart(
                episodes=dashboard_data['episodes'],
                rewards=dashboard_data['rewards'],
                moving_averages=dashboard_data['moving_averages'],
                anomalies=anomalies.get('anomalies', [])
            )
            charts['reward_analysis'] = self._fig_to_html(reward_chart)
        
        # 2. 电力系统图表
        if dashboard_data['cv_values'] or dashboard_data['coupling_ratios']:
            # 计算质量评分
            quality_score = None
            if dashboard_data['cv_values'] and dashboard_data['coupling_ratios']:
                power_analysis = self.analyzer.analyze_power_system_quality(
                    dashboard_data['cv_values'], 
                    dashboard_data['coupling_ratios']
                )
                quality_score = power_analysis.get('quality_score', 0)
            
            power_chart = self.chart_factory.create_power_system_chart(
                episodes=dashboard_data['episodes'],
                cv_values=dashboard_data['cv_values'],
                coupling_ratios=dashboard_data['coupling_ratios'],
                quality_score=quality_score
            )
            charts['power_system'] = self._fig_to_html(power_chart)
        
        # 3. 课程学习图表
        if dashboard_data['stages']:
            curriculum_chart = self.chart_factory.create_curriculum_chart(
                episodes=dashboard_data['episodes'],
                stages=dashboard_data['stages'],
                transitions=dashboard_data['transitions'],
                performance=dashboard_data['rewards']
            )
            charts['curriculum'] = self._fig_to_html(curriculum_chart)
        
        # 4. 损失分析图表
        if dashboard_data['actor_losses'] or dashboard_data['critic_losses']:
            loss_chart = self.chart_factory.create_loss_analysis_chart(
                episodes=dashboard_data['episodes'],
                actor_losses=dashboard_data['actor_losses'],
                critic_losses=dashboard_data['critic_losses'],
                entropies=dashboard_data['entropies']
            )
            charts['loss_analysis'] = self._fig_to_html(loss_chart)
        
        return charts
    
    def _generate_analysis_results(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析结果"""
        analysis = {}
        
        # 收敛性分析
        if dashboard_data['rewards']:
            convergence = self.analyzer.calculate_convergence_metrics(dashboard_data['rewards'])
            analysis['convergence'] = convergence
        
        # 电力系统分析
        if dashboard_data['cv_values'] and dashboard_data['coupling_ratios']:
            power_system = self.analyzer.analyze_power_system_quality(
                dashboard_data['cv_values'], 
                dashboard_data['coupling_ratios']
            )
            analysis['power_system'] = power_system
        
        # 课程学习分析
        if dashboard_data['stages'] and dashboard_data['rewards']:
            curriculum = self.analyzer.analyze_curriculum_effectiveness(
                dashboard_data['stages'], 
                dashboard_data['rewards']
            )
            analysis['curriculum'] = curriculum
        
        # 基础统计
        analysis['basic_stats'] = self._calculate_basic_stats(dashboard_data)
        
        return analysis
    
    def _generate_comparison_charts(self, comparison_data: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """生成对比图表"""
        charts = {}
        
        # 准备对比数据
        methods = list(comparison_data.keys())
        
        # 奖励对比图表
        reward_comparison_data = {}
        for method, data in comparison_data.items():
            if data['rewards']:
                final_performance = data['rewards'][-100:] if len(data['rewards']) > 100 else data['rewards']
                reward_comparison_data[method] = [
                    np.mean(final_performance),  # 平均奖励
                    max(data['rewards']),        # 最佳奖励
                    self._calculate_convergence_speed(data['rewards']),  # 收敛速度
                    self._calculate_stability(data['rewards'])           # 稳定性
                ]
        
        if reward_comparison_data:
            comparison_chart = self.chart_factory.create_comparison_chart(reward_comparison_data)
            charts['method_comparison'] = self._fig_to_html(comparison_chart)
        
        return charts
    
    def _generate_html_file(self, output_path: Path, dashboard_data: Dict[str, Any], 
                          charts: Dict[str, str], analysis_results: Dict[str, Any]):
        """生成HTML文件"""
        
        # 准备模板数据
        template_data = {
            'title': '强化学习训练仪表板',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_session': dashboard_data.get('config', {}).get('session_name', 'Training Session'),
            
            # 总览数据
            'total_episodes': dashboard_data['metadata']['total_episodes'],
            'training_duration': self._format_duration(dashboard_data['training_time']),
            'current_status': self._determine_status(analysis_results),
            'status_text': self._get_status_text(analysis_results),
            
            # 性能指标
            'best_reward': f"{max(dashboard_data['rewards']) if dashboard_data['rewards'] else 0:.4f}",
            'avg_reward': f"{np.mean(dashboard_data['rewards']) if dashboard_data['rewards'] else 0:.4f}",
            'convergence_confidence': f"{analysis_results.get('convergence', {}).get('convergence_confidence', 0) * 100:.1f}",
            
            # 电力系统指标
            'current_cv': f"{dashboard_data['cv_values'][-1] if dashboard_data['cv_values'] else 1.0:.3f}",
            'coupling_ratio': f"{dashboard_data['coupling_ratios'][-1] if dashboard_data['coupling_ratios'] else 1.0:.3f}",
            'system_quality': f"{analysis_results.get('power_system', {}).get('quality_score', 0):.0f}",
            
            # 智能课程
            'current_stage': dashboard_data['stages'][-1] if dashboard_data['stages'] else 0,
            'stage_progress': "100.0",  # 可以根据具体逻辑计算
            'transition_count': len(dashboard_data['transitions']),
            
            # 图表
            'charts': charts,
            
            # 数据表格
            'training_log_data': self._prepare_log_table_data(dashboard_data),
            
            # 配置
            'chart_config': json.dumps(self.chart_config),
            'color_palette': json.dumps(self.chart_factory.get_color_palette()),
            
            # 导出数据
            'csv_data': self._prepare_csv_data(dashboard_data)
        }
        
        # 渲染HTML
        html_content = self._render_html_template(template_data)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _render_html_template(self, template_data: Dict[str, Any]) -> str:
        """渲染HTML模板"""
        
        template_str = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* 现代化响应式样式 */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5em;
            font-weight: 300;
            margin-bottom: 10px;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #2E86AB;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }
        
        .card h3 {
            color: #2E86AB;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .card p {
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        
        .controls {
            margin: 30px 0;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            border: 1px solid #dee2e6;
        }
        
        .control-group {
            display: inline-block;
            margin-right: 30px;
            margin-bottom: 10px;
        }
        
        .control-group label {
            font-weight: 600;
            color: #495057;
            margin-right: 10px;
        }
        
        .control-group input, .control-group select {
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 8px;
            background: white;
        }
        
        .control-group button {
            padding: 10px 20px;
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .control-group button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .chart-container:hover {
            transform: translateY(-3px);
        }
        
        .data-table {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }
        
        .data-table h3 {
            color: #2E86AB;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        th {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            font-weight: 600;
        }
        
        tr:hover {
            background-color: #f8f9fa;
        }
        
        .metric-value {
            font-weight: 600;
            color: #2E86AB;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        
        @media (max-width: 768px) {
            .dashboard-container { padding: 15px; }
            .summary-cards { grid-template-columns: 1fr; }
            .chart-grid { grid-template-columns: 1fr; }
            .control-group { display: block; margin-bottom: 15px; }
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- 页头信息 -->
        <header>
            <h1>🎯 {{ title }}</h1>
            <p>训练会话: {{ training_session }} | 生成时间: {{ generation_time }}</p>
        </header>
        
        <!-- 总览卡片 -->
        <section class="summary-cards">
            <div class="card">
                <h3>📊 训练总览</h3>
                <p><strong>总Episodes:</strong> <span class="metric-value">{{ total_episodes }}</span></p>
                <p><strong>训练时间:</strong> <span class="metric-value">{{ training_duration }}</span></p>
                <p><strong>当前状态:</strong> 
                   <span class="status-indicator status-{{ current_status }}"></span>
                   {{ status_text }}
                </p>
            </div>
            
            <div class="card">
                <h3>🏆 性能指标</h3>
                <p><strong>最佳奖励:</strong> <span class="metric-value">{{ best_reward }}</span></p>
                <p><strong>平均奖励:</strong> <span class="metric-value">{{ avg_reward }}</span></p>
                <p><strong>收敛置信度:</strong> <span class="metric-value">{{ convergence_confidence }}%</span></p>
            </div>
            
            <div class="card">
                <h3>⚡ 电力系统</h3>
                <p><strong>当前CV:</strong> <span class="metric-value">{{ current_cv }}</span></p>
                <p><strong>耦合边比例:</strong> <span class="metric-value">{{ coupling_ratio }}</span></p>
                <p><strong>系统质量:</strong> <span class="metric-value">{{ system_quality }}/100</span></p>
            </div>
            
            <div class="card">
                <h3>🎓 智能课程</h3>
                <p><strong>当前阶段:</strong> <span class="metric-value">{{ current_stage }}</span></p>
                <p><strong>阶段进度:</strong> <span class="metric-value">{{ stage_progress }}%</span></p>
                <p><strong>转换次数:</strong> <span class="metric-value">{{ transition_count }}</span></p>
            </div>
        </section>
        
        <!-- 交互控件 -->
        <section class="controls">
            <div class="control-group">
                <label>Episode范围: </label>
                <input type="range" id="episodeRange" min="0" max="{{ total_episodes }}" value="{{ total_episodes }}">
                <span id="episodeValue">{{ total_episodes }}</span>
            </div>
            
            <div class="control-group">
                <label>显示模式: </label>
                <select id="displayMode">
                    <option value="all">全部数据</option>
                    <option value="recent">最近数据</option>
                    <option value="smooth">平滑显示</option>
                </select>
            </div>
            
            <div class="control-group">
                <button onclick="exportData()">📥 导出数据</button>
                <button onclick="printReport()">🖨️ 打印报告</button>
                <button onclick="refreshCharts()">🔄 刷新图表</button>
            </div>
        </section>
        
        <!-- 图表网格 -->
        <section class="chart-grid">
            {% for chart_id, chart_html in charts.items() %}
            <div class="chart-container">
                <div id="{{ chart_id }}-container">{{ chart_html | safe }}</div>
            </div>
            {% endfor %}
        </section>
        
        <!-- 详细数据表格 -->
        <section class="data-table">
            <h3>📋 训练日志详情</h3>
            <div id="training-log-table">
                {{ training_log_data | safe }}
            </div>
        </section>
    </div>
    
    <script>
        // 图表交互功能
        const chartConfig = {{ chart_config | safe }};
        const colorPalette = {{ color_palette | safe }};
        
        // Episode范围滑块
        document.getElementById('episodeRange').addEventListener('input', function() {
            document.getElementById('episodeValue').textContent = this.value;
            // 这里可以添加图表更新逻辑
        });
        
        // 数据导出
        function exportData() {
            const csvContent = "data:text/csv;charset=utf-8," + "{{ csv_data | safe }}";
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "training_data.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        // 打印报告
        function printReport() {
            window.print();
        }
        
        // 刷新图表
        function refreshCharts() {
            location.reload();
        }
        
        // 响应式图表
        window.addEventListener('resize', function() {
            setTimeout(function() {
                Plotly.Plots.resize();
            }, 100);
        });
        
        // 页面加载完成后的初始化
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🎨 HTML仪表板加载完成');
            
            // 这里可以添加更多初始化逻辑
        });
    </script>
</body>
</html>'''
        
        template = Template(template_str)
        return template.render(**template_data)
    
    def _fig_to_html(self, fig) -> str:
        """将Plotly图表转换为HTML"""
        if fig is None:
            return "<p>图表生成失败</p>"
            
        # 优化图表用于HTML显示
        optimized_fig = self.chart_factory.optimize_for_html(fig, self.max_data_points)
        
        if optimized_fig is None:
            return "<p>图表优化失败</p>"
        
        # 生成HTML
        return plotly.offline.plot(
            optimized_fig,
            output_type='div',
            include_plotlyjs=False,
            config=self.chart_config
        )
    
    def _compress_data(self, episodes: List[int], rewards: List[float]) -> Tuple[List[int], List[float]]:
        """压缩数据"""
        if len(episodes) <= self.max_data_points:
            return episodes, rewards
            
        # 使用图表工厂的智能采样
        indices = self.chart_factory._smart_sampling(len(episodes), self.max_data_points)
        
        compressed_episodes = [episodes[i] for i in indices]
        compressed_rewards = [rewards[i] for i in indices]
        
        return compressed_episodes, compressed_rewards
    
    def _calculate_moving_averages(self, rewards: List[float]) -> Dict[str, List[float]]:
        """计算移动平均"""
        if not rewards:
            return {}
            
        moving_averages = {}
        windows = [5, 20, 50]
        
        for window in windows:
            if len(rewards) > window:
                ma = pd.Series(rewards).rolling(window=window).mean().tolist()
                # 移除NaN值
                ma = [x for x in ma if not pd.isna(x)]
                if ma:
                    moving_averages[f'{window}'] = ma
                    
        return moving_averages
    
    def _calculate_basic_stats(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算基础统计"""
        stats = {}
        
        if dashboard_data['rewards']:
            rewards = dashboard_data['rewards']
            stats['reward_stats'] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(min(rewards)),
                'max': float(max(rewards)),
                'median': float(np.median(rewards))
            }
        
        stats['episode_count'] = len(dashboard_data['episodes'])
        stats['training_duration'] = dashboard_data['training_time']
        
        return stats
    
    def _determine_status(self, analysis_results: Dict[str, Any]) -> str:
        """确定训练状态"""
        convergence = analysis_results.get('convergence', {})
        
        if convergence.get('is_converged', False):
            return 'active'
        elif convergence.get('plateau_detected', False):
            return 'warning'
        else:
            return 'error'
    
    def _get_status_text(self, analysis_results: Dict[str, Any]) -> str:
        """获取状态文本"""
        convergence = analysis_results.get('convergence', {})
        
        if convergence.get('is_converged', False):
            return '已收敛'
        elif convergence.get('plateau_detected', False):
            return '平台期'
        else:
            return '训练中'
    
    def _format_duration(self, seconds: float) -> str:
        """格式化持续时间"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    def _prepare_log_table_data(self, dashboard_data: Dict[str, Any]) -> str:
        """准备日志表格数据"""
        if not dashboard_data['episodes']:
            return "<p>暂无训练数据</p>"
        
        # 创建表格
        table_html = "<table><thead><tr>"
        table_html += "<th>Episode</th><th>奖励</th><th>CV</th><th>耦合比例</th><th>阶段</th>"
        table_html += "</tr></thead><tbody>"
        
        # 只显示最近50条记录
        max_rows = 50
        start_idx = max(0, len(dashboard_data['episodes']) - max_rows)
        
        for i in range(start_idx, len(dashboard_data['episodes'])):
            episode = dashboard_data['episodes'][i]
            reward = dashboard_data['rewards'][i] if i < len(dashboard_data['rewards']) else 0
            cv = dashboard_data['cv_values'][i] if i < len(dashboard_data['cv_values']) else 0
            coupling = dashboard_data['coupling_ratios'][i] if i < len(dashboard_data['coupling_ratios']) else 0
            stage = dashboard_data['stages'][i] if i < len(dashboard_data['stages']) else 0
            
            table_html += f"<tr>"
            table_html += f"<td>{episode}</td>"
            table_html += f"<td>{reward:.4f}</td>"
            table_html += f"<td>{cv:.3f}</td>"
            table_html += f"<td>{coupling:.3f}</td>"
            table_html += f"<td>{stage}</td>"
            table_html += f"</tr>"
        
        table_html += "</tbody></table>"
        return table_html
    
    def _prepare_csv_data(self, dashboard_data: Dict[str, Any]) -> str:
        """准备CSV导出数据"""
        if not dashboard_data['episodes']:
            return ""
        
        csv_lines = ["Episode,Reward,CV,Coupling_Ratio,Stage"]
        
        for i in range(len(dashboard_data['episodes'])):
            episode = dashboard_data['episodes'][i]
            reward = dashboard_data['rewards'][i] if i < len(dashboard_data['rewards']) else 0
            cv = dashboard_data['cv_values'][i] if i < len(dashboard_data['cv_values']) else 0
            coupling = dashboard_data['coupling_ratios'][i] if i < len(dashboard_data['coupling_ratios']) else 0
            stage = dashboard_data['stages'][i] if i < len(dashboard_data['stages']) else 0
            
            csv_lines.append(f"{episode},{reward:.4f},{cv:.3f},{coupling:.3f},{stage}")
        
        return "\\n".join(csv_lines)
    
    def _generate_comparison_html_file(self, output_path: Path, comparison_data: Dict[str, Dict[str, Any]], 
                                     charts: Dict[str, str]):
        """生成对比分析HTML文件"""
        # 这里可以实现对比分析的HTML生成
        # 暂时使用简化版本
        comparison_html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>训练方法对比分析</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>训练方法对比分析</h1>
            <div id="comparison-charts">
                {charts.get('method_comparison', '<p>暂无对比图表</p>')}
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(comparison_html)
    
    def _calculate_convergence_speed(self, rewards: List[float]) -> float:
        """计算收敛速度"""
        if len(rewards) < 10:
            return 0.0
        
        # 简化的收敛速度计算
        improvement_window = min(50, len(rewards) // 2)
        early_avg = np.mean(rewards[:improvement_window])
        late_avg = np.mean(rewards[-improvement_window:])
        
        improvement_rate = (late_avg - early_avg) / (abs(early_avg) + 1e-8)
        speed_score = min(100, max(0, improvement_rate * 100 + 50))
        
        return speed_score
    
    def _calculate_stability(self, rewards: List[float]) -> float:
        """计算稳定性"""
        if len(rewards) < 10:
            return 0.0
        
        # 计算最近数据的变异系数
        recent_rewards = rewards[-min(50, len(rewards)):]
        cv = np.std(recent_rewards) / (abs(np.mean(recent_rewards)) + 1e-8)
        
        # 转换为稳定性评分（0-100）
        stability_score = max(0, min(100, (1 - cv) * 100))
        
        return stability_score

    def generate_performance_dashboard(self, performance_data: Dict[str, Any],
                                     output_filename: Optional[str] = None) -> Path:
        """
        生成性能分析仪表板

        Args:
            performance_data: 性能测试数据
            output_filename: 输出文件名，如果为None则自动生成

        Returns:
            生成的HTML文件路径
        """
        print("🎨 开始生成性能分析HTML仪表板...")

        # 设置输出路径
        if output_filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_filename = f"performance_analysis_{timestamp}.html"

        output_path = self.output_dir / output_filename
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建图表工厂
        chart_factory = PlotlyChartFactory(self.config.get('chart_factory', {}))

        # 生成性能分析图表
        charts = self._create_performance_charts(performance_data, chart_factory)

        # 生成HTML内容
        html_content = self._generate_performance_html(performance_data, charts)

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 性能分析仪表板生成完成: {output_path}")
        return output_path

    def _create_performance_charts(self, performance_data: Dict[str, Any],
                                 chart_factory: 'PlotlyChartFactory') -> Dict[str, str]:
        """创建性能分析图表"""
        charts = {}

        try:
            # 1. 跨网络成功率对比图
            if 'test_networks' in performance_data and 'success_rates' in performance_data:
                networks = performance_data['test_networks']
                success_rates = performance_data['success_rates']

                # 使用现有的comparison_chart方法
                comparison_data = {
                    'networks': networks,
                    'success_rates': [rate * 100 for rate in success_rates]  # 转换为百分比
                }
                charts['success_rate_comparison'] = chart_factory.create_comparison_chart(comparison_data).to_json()

            # 2. 平均奖励对比图 - 使用简单的文本表格代替复杂图表
            if 'test_networks' in performance_data and 'avg_rewards' in performance_data:
                networks = performance_data['test_networks']
                avg_rewards = performance_data['avg_rewards']

                # 创建简单的HTML表格
                table_html = "<table style='width:100%; border-collapse: collapse;'>"
                table_html += "<tr style='background:#f0f0f0;'><th style='padding:10px; border:1px solid #ddd;'>网络</th><th style='padding:10px; border:1px solid #ddd;'>平均奖励</th></tr>"
                for network, reward in zip(networks, avg_rewards):
                    table_html += f"<tr><td style='padding:10px; border:1px solid #ddd;'>{network}</td><td style='padding:10px; border:1px solid #ddd;'>{reward:.3f}</td></tr>"
                table_html += "</table>"
                charts['reward_comparison'] = table_html

        except Exception as e:
            print(f"⚠️ 创建性能图表时出错: {e}")

        return charts

    def _generate_performance_html(self, performance_data: Dict[str, Any],
                                 charts: Dict[str, str]) -> str:
        """生成性能分析HTML内容"""

        # 基础HTML模板
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>性能分析仪表板</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #2c3e50; margin-bottom: 10px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .metric-label { font-size: 0.9em; opacity: 0.9; }
        .chart-container { margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 8px; }
        .chart-title { font-size: 1.2em; font-weight: bold; margin-bottom: 15px; color: #34495e; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 模型性能分析仪表板</h1>
            <p>训练网络: {{ train_network }} | 测试时间: {{ test_time }}</p>
        </div>

        <div class="summary">
            <div class="metric-card">
                <div class="metric-value">{{ overall_success_rate }}%</div>
                <div class="metric-label">总体成功率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ overall_avg_reward }}</div>
                <div class="metric-label">平均奖励</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ test_count }}</div>
                <div class="metric-label">测试网络数</div>
            </div>
        </div>

        {{ charts_html }}
    </div>
</body>
</html>
        """

        # 准备模板变量
        template_vars = {
            'train_network': performance_data.get('train_network', 'Unknown'),
            'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_success_rate': f"{performance_data.get('overall_success_rate', 0) * 100:.1f}",
            'overall_avg_reward': f"{performance_data.get('overall_avg_reward', 0):.3f}",
            'test_count': len(performance_data.get('test_networks', [])),
            'charts_html': self._format_charts_html(charts)
        }

        # 替换模板变量
        html_content = html_template
        for key, value in template_vars.items():
            html_content = html_content.replace(f"{{{{ {key} }}}}", str(value))

        return html_content

    def _format_charts_html(self, charts: Dict[str, str]) -> str:
        """格式化图表HTML"""
        charts_html = ""

        for chart_id, chart_content in charts.items():
            if chart_id == 'success_rate_comparison' and chart_content.startswith('{'):
                # Plotly图表
                charts_html += f"""
                <div class="chart-container">
                    <div class="chart-title">跨网络泛化成功率对比</div>
                    <div id="{chart_id}" style="height: 500px;"></div>
                    <script>
                        Plotly.newPlot('{chart_id}', {chart_content});
                    </script>
                </div>
                """
            elif chart_id == 'reward_comparison':
                # HTML表格
                charts_html += f"""
                <div class="chart-container">
                    <div class="chart-title">跨网络平均奖励对比</div>
                    {chart_content}
                </div>
                """
            else:
                # 其他类型的图表
                charts_html += f"""
                <div class="chart-container">
                    <div class="chart-title">{chart_id}</div>
                    <div>{chart_content}</div>
                </div>
                """

        return charts_html