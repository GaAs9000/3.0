#!/usr/bin/env python3
"""
Plotly图表工厂

统一管理图表样式、配色方案和图表生成：
- 统一的视觉主题和配色
- 专业的电力系统可视化
- 智能课程学习进展图表
- 交互式性能分析图表
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import json


class PlotlyChartFactory:
    """Plotly图表工厂 - 统一图表样式和生成"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化图表工厂
        
        Args:
            config: 图表配置参数
        """
        self.config = config or {}
        
        # 统一配色方案
        self.colors = {
            'primary': '#2E86AB',      # 主要颜色 - 蓝色
            'secondary': '#A23B72',    # 次要颜色 - 紫红色  
            'success': '#F18F01',      # 成功/正向 - 橙色
            'warning': '#C73E1D',      # 警告/负向 - 红色
            'info': '#7209B7',         # 信息 - 紫色
            'neutral': '#6C757D',      # 中性 - 灰色
            'background': '#F8F9FA',   # 背景 - 浅灰色
            'text': '#2C3E50',         # 文字 - 深蓝灰色
            'grid': '#E5E5E5',         # 网格线 - 淡灰色
            'accent1': '#FF6B6B',      # 强调色1 - 亮红色
            'accent2': '#4ECDC4',      # 强调色2 - 青绿色
            'accent3': '#45B7D1',      # 强调色3 - 天蓝色
            'accent4': '#96CEB4',      # 强调色4 - 薄荷绿
            'accent5': '#FFEAA7'       # 强调色5 - 浅黄色
        }
        
        # 颜色序列（用于多系列图表）
        self.color_sequence = [
            self.colors['primary'], self.colors['secondary'], self.colors['success'],
            self.colors['info'], self.colors['accent1'], self.colors['accent2'],
            self.colors['accent3'], self.colors['accent4'], self.colors['accent5']
        ]
        
        # 统一布局配置
        self.layout_template = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': self.colors['text']},
            'title': {'font': {'size': 16, 'color': self.colors['text']}, 'x': 0.5, 'xanchor': 'center'},
            'xaxis': {
                'gridcolor': self.colors['grid'], 
                'gridwidth': 1,
                'zeroline': False,
                'title': {'font': {'size': 14}}
            },
            'yaxis': {
                'gridcolor': self.colors['grid'], 
                'gridwidth': 1,
                'zeroline': False,
                'title': {'font': {'size': 14}}
            },
            'legend': {
                'orientation': 'h', 
                'y': -0.15, 
                'x': 0.5, 
                'xanchor': 'center',
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': self.colors['grid'],
                'borderwidth': 1
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': dict(l=60, r=60, t=80, b=100)
        }
        
        # 图表默认配置
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'training_chart',
                'height': 600,
                'width': 800,
                'scale': 2
            }
        }
    
    def create_reward_analysis_chart(self, episodes: List[int], rewards: List[float],
                                   moving_averages: Optional[Dict[str, List[float]]] = None,
                                   anomalies: Optional[List[Dict]] = None) -> go.Figure:
        """
        创建奖励分析图表
        
        Args:
            episodes: Episode序列
            rewards: 奖励序列
            moving_averages: 移动平均数据 {'window_size': values}
            anomalies: 异常点数据
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['奖励趋势', '奖励分布', '收敛分析', '性能改善'],
            specs=[[{"secondary_y": True}, {"type": "histogram"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. 奖励趋势图
        fig.add_trace(go.Scatter(
            x=episodes, y=rewards,
            name='原始奖励', 
            opacity=0.6,
            line=dict(color=self.colors['neutral'], width=1),
            mode='lines'
        ), row=1, col=1)
        
        # 添加移动平均线
        if moving_averages:
            for i, (window, ma_values) in enumerate(moving_averages.items()):
                if len(ma_values) > 0:
                    color_idx = i % len(self.color_sequence)
                    fig.add_trace(go.Scatter(
                        x=episodes[-len(ma_values):], y=ma_values,
                        name=f'{window}期移动平均',
                        line=dict(color=self.color_sequence[color_idx], width=3),
                        mode='lines'
                    ), row=1, col=1)
        
        # 标记异常点
        if anomalies:
            anomaly_episodes = [episodes[a['index']] for a in anomalies if a['index'] < len(episodes)]
            anomaly_rewards = [rewards[a['index']] for a in anomalies if a['index'] < len(rewards)]
            if anomaly_episodes:
                fig.add_trace(go.Scatter(
                    x=anomaly_episodes, y=anomaly_rewards,
                    mode='markers',
                    name='异常点',
                    marker=dict(color=self.colors['warning'], size=8, symbol='x'),
                    showlegend=True
                ), row=1, col=1)
        
        # 2. 奖励分布
        fig.add_trace(go.Histogram(
            x=rewards, 
            name='奖励分布',
            marker_color=self.colors['primary'],
            opacity=0.7,
            nbinsx=50
        ), row=1, col=2)
        
        # 添加分位数线
        if len(rewards) > 0:
            q25, q50, q75 = np.percentile(rewards, [25, 50, 75])
            fig.add_vline(x=q50, line_dash="dash", 
                         annotation_text=f"中位数: {q50:.3f}", 
                         line_color=self.colors['secondary'],
                         row=1, col=2)
        
        # 3. 收敛分析 - 滚动标准差
        if len(rewards) > 30:
            window_size = min(30, len(rewards) // 3)
            rolling_std = []
            rolling_episodes = []
            
            for i in range(window_size, len(rewards)):
                window = rewards[i-window_size:i]
                rolling_std.append(np.std(window))
                rolling_episodes.append(episodes[i])
            
            if rolling_std:
                fig.add_trace(go.Scatter(
                    x=rolling_episodes, y=rolling_std,
                    name=f'{window_size}期滚动标准差',
                    line=dict(color=self.colors['warning'], width=2),
                    mode='lines',
                    fill='tonexty' if len(rolling_std) > 0 else None,
                    fillcolor=f'rgba{tuple(list(bytes.fromhex(self.colors["warning"][1:])) + [20])}'
                ), row=2, col=1)
        
        # 更新布局
        layout_config = self.layout_template.copy()
        layout_config['title'] = '奖励分析仪表板'
        layout_config['height'] = 700
        layout_config['showlegend'] = True
        fig.update_layout(**layout_config)
        
        # 更新子图标题
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_yaxes(title_text="奖励值", row=1, col=1)
        fig.update_xaxes(title_text="奖励值", row=1, col=2)
        fig.update_yaxes(title_text="频次", row=1, col=2)
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_yaxes(title_text="标准差", row=2, col=1)
        
        return fig
    
    def create_power_system_chart(self, episodes: List[int], cv_values: List[float],
                                coupling_ratios: List[float], 
                                quality_score: Optional[float] = None) -> go.Figure:
        """
        创建电力系统专业指标图表
        
        Args:
            episodes: Episode序列
            cv_values: 负载变异系数序列
            coupling_ratios: 耦合边比例序列
            quality_score: 系统质量评分
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['负载变异系数', '耦合边比例', 'CV vs 耦合度关系', '系统质量评分'],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"type": "scatter"}, {"type": "indicator"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. CV时间序列
        if cv_values:
            fig.add_trace(go.Scatter(
                x=episodes[:len(cv_values)], y=cv_values,
                name='负载CV', 
                line=dict(color=self.colors['warning'], width=3),
                mode='lines+markers',
                marker=dict(size=4)
            ), row=1, col=1)
            
            # 添加CV阈值线
            cv_threshold = 0.2  # 可配置
            fig.add_hline(y=cv_threshold, line_dash="dash", 
                         annotation_text=f"目标阈值: {cv_threshold}", 
                         line_color=self.colors['success'],
                         row=1, col=1)
        
        # 2. 耦合边比例
        if coupling_ratios:
            fig.add_trace(go.Scatter(
                x=episodes[:len(coupling_ratios)], y=coupling_ratios,
                name='耦合边比例', 
                line=dict(color=self.colors['secondary'], width=3),
                mode='lines+markers',
                marker=dict(size=4)
            ), row=1, col=2)
            
            # 添加耦合阈值线
            coupling_threshold = 0.15  # 可配置
            fig.add_hline(y=coupling_threshold, line_dash="dash",
                         annotation_text=f"目标阈值: {coupling_threshold}", 
                         line_color=self.colors['success'],
                         row=1, col=2)
        
        # 3. CV vs 耦合度散点图
        if cv_values and coupling_ratios:
            min_len = min(len(cv_values), len(coupling_ratios))
            cv_subset = cv_values[:min_len]
            coupling_subset = coupling_ratios[:min_len]
            episodes_subset = episodes[:min_len]
            
            fig.add_trace(go.Scatter(
                x=cv_subset, y=coupling_subset,
                mode='markers',
                marker=dict(
                    color=episodes_subset,
                    colorscale='Viridis',
                    size=8,
                    colorbar=dict(title="Episode", x=0.45),
                    showscale=True
                ),
                name='CV-耦合关系',
                text=[f"Episode {ep}" for ep in episodes_subset],
                hovertemplate="CV: %{x:.3f}<br>耦合比例: %{y:.3f}<br>%{text}<extra></extra>"
            ), row=2, col=1)
        
        # 4. 系统质量综合评分
        if quality_score is not None:
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "系统质量评分"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.colors['primary']},
                    'steps': [
                        {'range': [0, 40], 'color': self.colors['warning']},
                        {'range': [40, 70], 'color': self.colors['accent5']},
                        {'range': [70, 100], 'color': self.colors['success']}
                    ],
                    'threshold': {
                        'line': {'color': self.colors['text'], 'width': 4},
                        'thickness': 0.75, 
                        'value': 80
                    }
                }
            ), row=2, col=2)
        
        # 更新布局
        layout_config = self.layout_template.copy()
        layout_config['title'] = '电力系统专业指标分析'
        layout_config['height'] = 700
        fig.update_layout(**layout_config)
        
        # 更新坐标轴标题
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_yaxes(title_text="变异系数", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_yaxes(title_text="耦合边比例", row=1, col=2)
        fig.update_xaxes(title_text="负载变异系数", row=2, col=1)
        fig.update_yaxes(title_text="耦合边比例", row=2, col=1)
        
        return fig
    
    def create_curriculum_chart(self, episodes: List[int], stages: List[int],
                              transitions: List[Dict], performance: List[float]) -> go.Figure:
        """
        创建智能课程学习图表
        
        Args:
            episodes: Episode序列
            stages: 训练阶段序列
            transitions: 阶段转换事件
            performance: 性能数据序列
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['课程阶段进展', '阶段转换时机', '各阶段性能分布', '学习进展趋势'],
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "box"}, {"secondary_y": True}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. 阶段进展时间线
        if stages and episodes:
            fig.add_trace(go.Scatter(
                x=episodes[:len(stages)], y=stages,
                mode='lines+markers',
                name='课程阶段',
                line=dict(color=self.colors['info'], width=3),
                marker=dict(size=6)
            ), row=1, col=1)
            
            # 标注阶段转换点
            for transition in transitions:
                if 'episode' in transition:
                    fig.add_vline(
                        x=transition['episode'], 
                        line_dash="dash", 
                        annotation_text=f"→阶段{transition.get('to_stage', '?')}",
                        line_color=self.colors['accent1'],
                        row=1, col=1
                    )
        
        # 2. 各阶段持续时间
        if transitions:
            stage_durations = [t.get('duration', 0) for t in transitions]
            stage_names = [f"阶段{t.get('from_stage', '?')}" for t in transitions]
            
            if stage_durations:
                fig.add_trace(go.Bar(
                    x=stage_names,
                    y=stage_durations,
                    name='阶段持续时间',
                    marker_color=self.colors['primary'],
                    text=stage_durations,
                    textposition='outside'
                ), row=1, col=2)
        
        # 3. 各阶段性能分布
        if stages and performance:
            unique_stages = sorted(set(stages))
            for i, stage in enumerate(unique_stages):
                stage_performance = [performance[j] for j, s in enumerate(stages) 
                                   if s == stage and j < len(performance)]
                
                if stage_performance:
                    color_idx = i % len(self.color_sequence)
                    fig.add_trace(go.Box(
                        y=stage_performance,
                        name=f'阶段{stage}',
                        marker_color=self.color_sequence[color_idx],
                        boxmean=True
                    ), row=2, col=1)
        
        # 4. 学习进展趋势（阶段叠加性能）
        if episodes and performance and stages:
            fig.add_trace(go.Scatter(
                x=episodes[:len(performance)], 
                y=performance,
                mode='lines',
                name='性能趋势',
                line=dict(color=self.colors['success'], width=2),
                opacity=0.8
            ), row=2, col=2)
            
            # 添加阶段背景色
            if transitions:
                for i, transition in enumerate(transitions):
                    start_ep = transition.get('episode', 0)
                    end_ep = transitions[i+1].get('episode', len(episodes)) if i+1 < len(transitions) else len(episodes)
                    
                    fig.add_vrect(
                        x0=start_ep, x1=end_ep,
                        fillcolor=self.color_sequence[i % len(self.color_sequence)],
                        opacity=0.1,
                        line_width=0,
                        row=2, col=2
                    )
        
        # 更新布局
        layout_config = self.layout_template.copy()
        layout_config['title'] = '智能课程学习分析'
        layout_config['height'] = 700
        fig.update_layout(**layout_config)
        
        # 更新坐标轴标题
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_yaxes(title_text="课程阶段", row=1, col=1)
        fig.update_xaxes(title_text="阶段", row=1, col=2)
        fig.update_yaxes(title_text="持续时间(Episodes)", row=1, col=2)
        fig.update_xaxes(title_text="阶段", row=2, col=1)
        fig.update_yaxes(title_text="性能值", row=2, col=1)
        fig.update_xaxes(title_text="Episode", row=2, col=2)
        fig.update_yaxes(title_text="性能值", row=2, col=2)
        
        return fig
    
    def create_loss_analysis_chart(self, episodes: List[int], 
                                 actor_losses: List[float],
                                 critic_losses: List[float],
                                 entropies: List[float]) -> go.Figure:
        """
        创建损失分析图表
        
        Args:
            episodes: Episode序列
            actor_losses: Actor损失序列
            critic_losses: Critic损失序列
            entropies: 熵值序列
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Actor损失', 'Critic损失', '策略熵变化', '损失相关性'],
            specs=[[{}, {}], [{}, {"type": "scatter"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Actor损失
        if actor_losses:
            episodes_subset = episodes[:len(actor_losses)]
            fig.add_trace(go.Scatter(
                x=episodes_subset, y=actor_losses,
                name='Actor损失',
                line=dict(color=self.colors['primary'], width=2),
                mode='lines'
            ), row=1, col=1)
            
            # 添加移动平均
            if len(actor_losses) > 10:
                window = min(20, len(actor_losses) // 3)
                ma = pd.Series(actor_losses).rolling(window=window).mean()
                fig.add_trace(go.Scatter(
                    x=episodes_subset, y=ma,
                    name=f'Actor损失{window}期均值',
                    line=dict(color=self.colors['accent1'], width=3),
                    mode='lines'
                ), row=1, col=1)
        
        # 2. Critic损失
        if critic_losses:
            episodes_subset = episodes[:len(critic_losses)]
            fig.add_trace(go.Scatter(
                x=episodes_subset, y=critic_losses,
                name='Critic损失',
                line=dict(color=self.colors['secondary'], width=2),
                mode='lines'
            ), row=1, col=2)
            
            # 添加移动平均
            if len(critic_losses) > 10:
                window = min(20, len(critic_losses) // 3)
                ma = pd.Series(critic_losses).rolling(window=window).mean()
                fig.add_trace(go.Scatter(
                    x=episodes_subset, y=ma,
                    name=f'Critic损失{window}期均值',
                    line=dict(color=self.colors['accent2'], width=3),
                    mode='lines'
                ), row=1, col=2)
        
        # 3. 策略熵
        if entropies:
            episodes_subset = episodes[:len(entropies)]
            fig.add_trace(go.Scatter(
                x=episodes_subset, y=entropies,
                name='策略熵',
                line=dict(color=self.colors['success'], width=2),
                mode='lines',
                fill='tonexty'
            ), row=2, col=1)
        
        # 4. 损失相关性散点图
        if actor_losses and critic_losses:
            min_len = min(len(actor_losses), len(critic_losses))
            fig.add_trace(go.Scatter(
                x=actor_losses[:min_len], 
                y=critic_losses[:min_len],
                mode='markers',
                name='损失相关性',
                marker=dict(
                    color=episodes[:min_len],
                    colorscale='Viridis',
                    size=6,
                    showscale=True,
                    colorbar=dict(title="Episode", x=1.02)
                ),
                text=[f"Episode {ep}" for ep in episodes[:min_len]],
                hovertemplate="Actor损失: %{x:.4f}<br>Critic损失: %{y:.4f}<br>%{text}<extra></extra>"
            ), row=2, col=2)
        
        # 更新布局
        layout_config = self.layout_template.copy()
        layout_config['title'] = '训练损失分析'
        layout_config['height'] = 700
        fig.update_layout(**layout_config)
        
        # 更新坐标轴标题
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_yaxes(title_text="Actor损失", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_yaxes(title_text="Critic损失", row=1, col=2)
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_yaxes(title_text="策略熵", row=2, col=1)
        fig.update_xaxes(title_text="Actor损失", row=2, col=2)
        fig.update_yaxes(title_text="Critic损失", row=2, col=2)
        
        return fig
    
    def create_overview_chart(self, training_stats: Dict[str, Any]) -> go.Figure:
        """
        创建训练总览图表
        
        Args:
            training_stats: 训练统计数据
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['训练进度', '奖励统计', '系统健康', '收敛状态', '学习效率', '总体评分'],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )
        
        # 1. 训练进度
        total_episodes = training_stats.get('total_episodes', 0)
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=total_episodes,
            title={'text': "已完成Episodes"},
            gauge={'axis': {'range': [None, max(1000, total_episodes)]},
                   'bar': {'color': self.colors['primary']}}
        ), row=1, col=1)
        
        # 2. 奖励统计
        reward_data = training_stats.get('reward_summary', {})
        categories = ['最佳', '平均', '最差']
        values = [
            reward_data.get('best', 0),
            reward_data.get('mean', 0), 
            reward_data.get('worst', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=categories, y=values,
            name='奖励统计',
            marker_color=[self.colors['success'], self.colors['primary'], self.colors['warning']],
            text=[f"{v:.3f}" for v in values],
            textposition='outside'
        ), row=1, col=2)
        
        # 3. 系统健康评分
        health_score = training_stats.get('system_health', {}).get('health_score', 50)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={'text': "系统健康评分"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self.colors['success'] if health_score > 70 else self.colors['warning']},
                'steps': [
                    {'range': [0, 40], 'color': 'lightgray'},
                    {'range': [40, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'lightgreen'}
                ]
            }
        ), row=1, col=3)
        
        # 4. 收敛状态
        convergence_confidence = training_stats.get('convergence', {}).get('confidence', 0) * 100
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=convergence_confidence,
            title={'text': "收敛置信度 (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': self.colors['info']}}
        ), row=2, col=1)
        
        # 5. 学习效率
        learning_efficiency = training_stats.get('learning_efficiency', 0.5) * 100
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=learning_efficiency,
            title={'text': "学习效率 (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': self.colors['accent3']}}
        ), row=2, col=2)
        
        # 6. 总体评分
        overall_score = training_stats.get('overall_score', 50)
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            title={'text': "总体评分"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': self.colors['primary']}}
        ), row=2, col=3)
        
        # 更新布局
        fig.update_layout(
            **self.layout_template,
            title='训练总览仪表板',
            height=600
        )
        
        return fig
    
    def create_comparison_chart(self, comparison_data: Dict[str, List]) -> go.Figure:
        """
        创建对比分析图表
        
        Args:
            comparison_data: 对比数据
            
        Returns:
            Plotly图表对象
        """
        methods = list(comparison_data.keys())
        metrics = ['平均奖励', '最佳奖励', '收敛速度', '稳定性']
        
        fig = go.Figure()
        
        # 创建雷达图
        for i, method in enumerate(methods):
            values = comparison_data[method]
            # 闭合雷达图
            values_closed = values + [values[0]]
            metrics_closed = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=metrics_closed,
                fill='toself',
                name=method,
                fillcolor=f'rgba{tuple(list(bytes.fromhex(self.color_sequence[i % len(self.color_sequence)][1:])) + [50])}',
                line=dict(color=self.color_sequence[i % len(self.color_sequence)], width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor=self.colors['grid']
                ),
                angularaxis=dict(
                    gridcolor=self.colors['grid']
                )
            ),
            showlegend=True,
            title='方法对比分析',
            **self.layout_template
        )
        
        return fig
    
    def optimize_for_html(self, fig: go.Figure, max_points: int = 2000) -> go.Figure:
        """
        优化图表用于HTML显示
        
        Args:
            fig: 原始图表
            max_points: 最大数据点数
            
        Returns:
            优化后的图表
        """
        # 数据点采样优化
        for trace in fig.data:
            if hasattr(trace, 'x') and hasattr(trace, 'y'):
                # 检查x和y是否不为None且有长度
                if (trace.x is not None and trace.y is not None and 
                    hasattr(trace.x, '__len__') and hasattr(trace.y, '__len__') and
                    len(trace.x) > max_points):
                    # 智能采样
                    indices = self._smart_sampling(len(trace.x), max_points)
                    if hasattr(trace, 'update'):
                        trace.update(
                            x=[trace.x[i] for i in indices],
                            y=[trace.y[i] for i in indices]
                        )
        
        # 性能优化设置
        fig.update_layout(
            autosize=True
        )
        
        return fig
    
    def _smart_sampling(self, data_length: int, target_size: int) -> List[int]:
        """
        智能采样 - 保留关键点
        
        Args:
            data_length: 原始数据长度
            target_size: 目标大小
            
        Returns:
            采样索引列表
        """
        if data_length <= target_size:
            return list(range(data_length))
        
        # 保留开始和结束点
        indices = [0, data_length - 1]
        
        # 均匀采样中间点
        step = data_length // (target_size - 2)
        for i in range(step, data_length - 1, step):
            if len(indices) < target_size:
                indices.append(i)
        
        return sorted(set(indices))
    
    def get_chart_config(self) -> Dict[str, Any]:
        """获取图表配置"""
        return self.chart_config.copy()
    
    def get_color_palette(self) -> Dict[str, str]:
        """获取配色方案"""
        return self.colors.copy() 