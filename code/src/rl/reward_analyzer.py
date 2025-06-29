#!/usr/bin/env python3
"""
奖励分析工具：深度分析奖励组件的贡献和演化
为调试和优化奖励函数提供科学依据
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import pandas as pd
from pathlib import Path
import json

class RewardAnalyzer:
    """
    奖励分析工具
    
    功能：
    1. 奖励组件分解分析
    2. 奖励演化趋势分析
    3. 组件相关性分析
    4. 异常检测和诊断
    5. 可视化报告生成
    """
    
    def __init__(self, output_dir: str = "reward_analysis"):
        """
        初始化奖励分析器
        
        Args:
            output_dir: 分析结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 数据存储
        self.reward_history: List[Dict[str, float]] = []
        self.episode_info: List[Dict[str, Any]] = []
        self.component_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 分析配置
        self.moving_average_window = 50
        self.anomaly_threshold = 3.0  # 标准差倍数
        
        # 可视化配置
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def add_episode_data(self, 
                        episode: int,
                        total_reward: float,
                        reward_components: Dict[str, float],
                        episode_info: Optional[Dict[str, Any]] = None):
        """
        添加单回合的奖励数据
        
        Args:
            episode: 回合编号
            total_reward: 总奖励
            reward_components: 奖励组件分解
            episode_info: 额外的回合信息
        """
        # 添加总奖励到组件中
        components = reward_components.copy()
        components['total_reward'] = total_reward
        components['episode'] = episode
        
        self.reward_history.append(components)
        
        if episode_info:
            episode_info['episode'] = episode
            self.episode_info.append(episode_info)
        
        # 更新组件统计
        self._update_component_stats(components)
    
    def _update_component_stats(self, components: Dict[str, float]):
        """更新组件统计信息"""
        for comp_name, value in components.items():
            if comp_name == 'episode':
                continue
                
            if comp_name not in self.component_stats:
                self.component_stats[comp_name] = {
                    'values': [],
                    'mean': 0.0,
                    'std': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'count': 0
                }
            
            stats = self.component_stats[comp_name]
            stats['values'].append(value)
            stats['count'] += 1
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            
            # 更新均值和标准差
            values = stats['values']
            stats['mean'] = np.mean(values)
            stats['std'] = np.std(values)
    
    def analyze_component_evolution(self) -> Dict[str, Any]:
        """分析奖励组件的演化趋势"""
        if not self.reward_history:
            return {}
        
        analysis = {}
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(self.reward_history)
        
        for component in df.columns:
            if component == 'episode':
                continue
                
            values = df[component].values
            episodes = df['episode'].values
            
            # 趋势分析
            trend_analysis = self._analyze_trend(episodes, values)
            
            # 稳定性分析
            stability_analysis = self._analyze_stability(values)
            
            # 相关性分析
            correlation_analysis = self._analyze_correlations(df, component)
            
            analysis[component] = {
                'trend': trend_analysis,
                'stability': stability_analysis,
                'correlation': correlation_analysis,
                'statistics': self.component_stats[component]
            }
        
        return analysis
    
    def _analyze_trend(self, episodes: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """分析趋势"""
        # 线性回归分析趋势
        if len(values) < 2:
            return {'slope': 0, 'trend': 'insufficient_data'}
        
        # 计算斜率
        slope = np.polyfit(episodes, values, 1)[0]
        
        # 移动平均
        if len(values) >= self.moving_average_window:
            moving_avg = pd.Series(values).rolling(window=self.moving_average_window).mean()
            recent_trend = moving_avg.iloc[-10:].mean() - moving_avg.iloc[-20:-10].mean()
        else:
            recent_trend = 0
        
        # 趋势分类
        if abs(slope) < 1e-6:
            trend_type = 'stable'
        elif slope > 0:
            trend_type = 'increasing'
        else:
            trend_type = 'decreasing'
        
        return {
            'slope': slope,
            'trend_type': trend_type,
            'recent_trend': recent_trend,
            'trend_strength': abs(slope)
        }
    
    def _analyze_stability(self, values: np.ndarray) -> Dict[str, Any]:
        """分析稳定性"""
        if len(values) < 10:
            return {'stability': 'insufficient_data'}
        
        # 变异系数
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
        
        # 异常值检测
        z_scores = np.abs((values - mean_val) / std_val) if std_val > 0 else np.zeros_like(values)
        anomalies = np.sum(z_scores > self.anomaly_threshold)
        anomaly_rate = anomalies / len(values)
        
        # 稳定性评级
        if cv < 0.1:
            stability_level = 'very_stable'
        elif cv < 0.3:
            stability_level = 'stable'
        elif cv < 0.5:
            stability_level = 'moderate'
        else:
            stability_level = 'unstable'
        
        return {
            'coefficient_of_variation': cv,
            'anomaly_count': anomalies,
            'anomaly_rate': anomaly_rate,
            'stability_level': stability_level
        }
    
    def _analyze_correlations(self, df: pd.DataFrame, target_component: str) -> Dict[str, float]:
        """分析与其他组件的相关性"""
        correlations = {}
        
        target_values = df[target_component]
        
        for component in df.columns:
            if component in ['episode', target_component]:
                continue
                
            try:
                corr = target_values.corr(df[component])
                if not np.isnan(corr):
                    correlations[component] = corr
            except:
                continue
        
        return correlations
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测奖励异常"""
        anomalies = []
        
        if not self.reward_history:
            return anomalies
        
        df = pd.DataFrame(self.reward_history)
        
        for component in df.columns:
            if component == 'episode':
                continue
                
            values = df[component].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                continue
            
            z_scores = np.abs((values - mean_val) / std_val)
            anomaly_indices = np.where(z_scores > self.anomaly_threshold)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    'episode': df.iloc[idx]['episode'],
                    'component': component,
                    'value': values[idx],
                    'z_score': z_scores[idx],
                    'expected_range': (mean_val - 2*std_val, mean_val + 2*std_val)
                })
        
        return anomalies
    
    def generate_visualizations(self):
        """生成可视化图表"""
        if not self.reward_history:
            print("⚠️ 没有数据可供可视化")
            return
        
        df = pd.DataFrame(self.reward_history)
        
        # 1. 奖励组件演化图
        self._plot_component_evolution(df)
        
        # 2. 组件分布图
        self._plot_component_distributions(df)
        
        # 3. 相关性热力图
        self._plot_correlation_heatmap(df)
        
        # 4. 异常检测图
        self._plot_anomaly_detection(df)
        
        print(f"📊 可视化图表已保存到: {self.output_dir}")
    
    def _plot_component_evolution(self, df: pd.DataFrame):
        """绘制组件演化图"""
        components = [col for col in df.columns if col != 'episode']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, component in enumerate(components[:4]):  # 只显示前4个组件
            if i >= len(axes):
                break
                
            ax = axes[i]
            episodes = df['episode']
            values = df[component]
            
            # 原始数据
            ax.plot(episodes, values, alpha=0.3, label='原始数据')
            
            # 移动平均
            if len(values) >= self.moving_average_window:
                moving_avg = pd.Series(values).rolling(window=self.moving_average_window).mean()
                ax.plot(episodes, moving_avg, linewidth=2, label=f'{self.moving_average_window}回合移动平均')
            
            ax.set_title(f'{component} 演化趋势')
            ax.set_xlabel('回合')
            ax.set_ylabel('奖励值')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(components), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_component_distributions(self, df: pd.DataFrame):
        """绘制组件分布图"""
        components = [col for col in df.columns if col != 'episode']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, component in enumerate(components[:4]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = df[component].dropna()
            
            # 直方图
            ax.hist(values, bins=30, alpha=0.7, density=True)
            
            # 核密度估计
            try:
                values.plot.density(ax=ax, linewidth=2)
            except:
                pass
            
            ax.set_title(f'{component} 分布')
            ax.set_xlabel('奖励值')
            ax.set_ylabel('密度')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(components), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """绘制相关性热力图"""
        # 只包含数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        plt.title('奖励组件相关性热力图')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomaly_detection(self, df: pd.DataFrame):
        """绘制异常检测图"""
        anomalies = self.detect_anomalies()
        
        if not anomalies:
            return
        
        # 按组件分组异常
        component_anomalies = defaultdict(list)
        for anomaly in anomalies:
            component_anomalies[anomaly['component']].append(anomaly)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (component, comp_anomalies) in enumerate(list(component_anomalies.items())[:4]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 绘制正常数据
            episodes = df['episode']
            values = df[component]
            ax.plot(episodes, values, alpha=0.5, label='正常数据')
            
            # 标记异常点
            anomaly_episodes = [a['episode'] for a in comp_anomalies]
            anomaly_values = [a['value'] for a in comp_anomalies]
            ax.scatter(anomaly_episodes, anomaly_values, 
                      color='red', s=50, label=f'异常点 ({len(comp_anomalies)}个)')
            
            ax.set_title(f'{component} 异常检测')
            ax.set_xlabel('回合')
            ax.set_ylabel('奖励值')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(component_anomalies), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_analysis_report(self) -> str:
        """生成分析报告"""
        if not self.reward_history:
            return "没有数据可供分析"
        
        analysis = self.analyze_component_evolution()
        anomalies = self.detect_anomalies()
        
        report_file = self.output_dir / 'reward_analysis_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 奖励系统分析报告\n\n")
            f.write(f"分析数据: {len(self.reward_history)} 个回合\n\n")
            
            # 组件概览
            f.write("## 奖励组件概览\n\n")
            f.write("| 组件名称 | 均值 | 标准差 | 最小值 | 最大值 | 趋势 | 稳定性 |\n")
            f.write("|---------|------|--------|--------|--------|------|--------|\n")
            
            for component, comp_analysis in analysis.items():
                stats = comp_analysis['statistics']
                trend = comp_analysis['trend']['trend_type']
                stability = comp_analysis['stability']['stability_level']
                
                f.write(f"| {component} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                       f"{stats['min']:.4f} | {stats['max']:.4f} | {trend} | {stability} |\n")
            
            # 异常检测结果
            f.write(f"\n## 异常检测结果\n\n")
            f.write(f"检测到 {len(anomalies)} 个异常点\n\n")
            
            if anomalies:
                f.write("| 回合 | 组件 | 异常值 | Z分数 |\n")
                f.write("|------|------|--------|-------|\n")
                
                for anomaly in anomalies[:10]:  # 只显示前10个
                    f.write(f"| {anomaly['episode']} | {anomaly['component']} | "
                           f"{anomaly['value']:.4f} | {anomaly['z_score']:.2f} |\n")
            
            # 建议
            f.write("\n## 优化建议\n\n")
            
            # 基于分析结果生成建议
            unstable_components = [comp for comp, analysis in analysis.items() 
                                 if analysis['stability']['stability_level'] == 'unstable']
            
            if unstable_components:
                f.write(f"### 稳定性改进\n")
                f.write(f"以下组件表现不稳定，建议调整权重或实现:\n")
                for comp in unstable_components:
                    f.write(f"- {comp}\n")
                f.write("\n")
            
            # 趋势建议
            decreasing_components = [comp for comp, analysis in analysis.items() 
                                   if analysis['trend']['trend_type'] == 'decreasing']
            
            if decreasing_components:
                f.write(f"### 趋势优化\n")
                f.write(f"以下组件呈下降趋势，可能需要调整:\n")
                for comp in decreasing_components:
                    f.write(f"- {comp}\n")
        
        print(f"📋 分析报告已生成: {report_file}")
        return str(report_file)

if __name__ == "__main__":
    # 示例使用
    analyzer = RewardAnalyzer()
    
    # 模拟添加一些数据
    for episode in range(100):
        components = {
            'local_connectivity': np.random.normal(0.5, 0.1),
            'incremental_balance': np.random.normal(0.3, 0.05),
            'boundary_compression': np.random.normal(0.7, 0.15),
            'exploration_bonus': np.random.normal(0.2, 0.08)
        }
        total_reward = sum(components.values()) + np.random.normal(0, 0.1)
        
        analyzer.add_episode_data(episode, total_reward, components)
    
    # 生成分析
    analyzer.generate_visualizations()
    analyzer.generate_analysis_report()
    
    print("✅ 奖励分析完成")
