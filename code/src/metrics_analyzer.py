#!/usr/bin/env python3
"""
训练指标分析器

专注于深度分析训练数据，计算各种统计指标和趋势分析：
- 收敛性分析
- 电力系统质量评估
- 智能课程学习效果分析
- 异常检测和性能瓶颈识别
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.signal import find_peaks
from collections import deque
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class MetricsAnalyzer:
    """训练指标分析器 - 提供深度统计分析功能"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化指标分析器
        
        Args:
            config: 分析配置参数
        """
        self.config = config or {}
        self.default_window_size = self.config.get('default_window_size', 50)
        self.convergence_threshold = self.config.get('convergence_threshold', 0.1)
        self.stability_threshold = self.config.get('stability_threshold', 0.05)
        
    def calculate_convergence_metrics(self, rewards: List[float], 
                                    window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        计算收敛性指标
        
        Args:
            rewards: 奖励序列
            window_size: 分析窗口大小
            
        Returns:
            收敛性分析结果
        """
        if not rewards or len(rewards) < 3:
            return self._empty_convergence_metrics()
            
        window_size = window_size or min(self.default_window_size, len(rewards) // 2)
        if window_size < 3:
            window_size = 3
            
        try:
            # 计算滚动统计
            rolling_stats = self._calculate_rolling_stats(rewards, window_size)
            
            # 趋势分析
            trend_analysis = self._analyze_trend(rewards, window_size)
            
            # 稳定性分析
            stability_analysis = self._analyze_stability(rewards, window_size)
            
            # 收敛置信度
            convergence_confidence = self._calculate_convergence_confidence(
                rolling_stats, trend_analysis, stability_analysis
            )
            
            return {
                'variance_trend': rolling_stats['variance'],
                'mean_trend': rolling_stats['mean'],
                'improvement_rate': trend_analysis['improvement_rate'],
                'trend_direction': trend_analysis['direction'],
                'trend_strength': trend_analysis['strength'],
                'stability_score': stability_analysis['stability_score'],
                'volatility': stability_analysis['volatility'],
                'convergence_confidence': convergence_confidence,
                'is_converged': convergence_confidence > 0.8,
                'plateau_detected': stability_analysis['plateau_detected'],
                'analysis_window': window_size
            }
            
        except Exception as e:
            print(f"⚠️ 收敛性分析失败: {e}")
            return self._empty_convergence_metrics()
    
    def analyze_power_system_quality(self, cv_values: List[float], 
                                   coupling_ratios: List[float]) -> Dict[str, Any]:
        """
        电力系统质量分析
        
        Args:
            cv_values: 负载变异系数序列
            coupling_ratios: 耦合边比例序列
            
        Returns:
            电力系统质量分析结果
        """
        if not cv_values or not coupling_ratios:
            return self._empty_power_system_analysis()
            
        try:
            # 综合质量评分
            quality_score = self._composite_quality_score(cv_values, coupling_ratios)
            
            # 平衡度趋势分析
            balance_trend = self._trend_analysis(cv_values) if cv_values else {}
            
            # 耦合效率分析
            coupling_efficiency = self._coupling_efficiency(coupling_ratios) if coupling_ratios else {}
            
            # 系统健康度评估
            system_health = self._system_health_assessment(cv_values, coupling_ratios)
            
            # 目标达成率
            target_achievement = self._calculate_target_achievement(cv_values, coupling_ratios)
            
            return {
                'quality_score': quality_score,
                'balance_trend': balance_trend,
                'coupling_efficiency': coupling_efficiency,
                'system_health': system_health,
                'target_achievement': target_achievement,
                'current_cv': cv_values[-1] if cv_values else 1.0,
                'current_coupling_ratio': coupling_ratios[-1] if coupling_ratios else 1.0,
                'cv_improvement': self._calculate_improvement(cv_values, lower_is_better=True),
                'coupling_improvement': self._calculate_improvement(coupling_ratios, lower_is_better=True)
            }
            
        except Exception as e:
            print(f"⚠️ 电力系统质量分析失败: {e}")
            return self._empty_power_system_analysis()
    
    def analyze_curriculum_effectiveness(self, stage_data: List[int], 
                                       performance_data: List[float]) -> Dict[str, Any]:
        """
        课程学习效果分析
        
        Args:
            stage_data: 训练阶段序列
            performance_data: 性能数据序列
            
        Returns:
            课程学习效果分析结果
        """
        if not stage_data or not performance_data:
            return self._empty_curriculum_analysis()
            
        try:
            # 各阶段性能统计
            stage_performance = self._stage_wise_performance(stage_data, performance_data)
            
            # 转换时机分析
            transition_analysis = self._transition_timing_analysis(stage_data, performance_data)
            
            # 学习效率分析
            learning_efficiency = self._learning_efficiency_per_stage(stage_data, performance_data)
            
            # 课程策略评估
            curriculum_strategy = self._evaluate_curriculum_strategy(stage_data, performance_data)
            
            return {
                'stage_performance': stage_performance,
                'transition_analysis': transition_analysis,
                'learning_efficiency': learning_efficiency,
                'curriculum_strategy': curriculum_strategy,
                'total_stages': len(set(stage_data)),
                'average_stage_duration': np.mean(transition_analysis.get('stage_durations', [1])),
                'curriculum_effectiveness': curriculum_strategy.get('effectiveness_score', 0.5)
            }
            
        except Exception as e:
            print(f"⚠️ 课程学习分析失败: {e}")
            return self._empty_curriculum_analysis()
    
    def detect_anomalies(self, data: List[float], 
                        method: str = 'statistical') -> Dict[str, Any]:
        """
        异常检测
        
        Args:
            data: 数据序列
            method: 检测方法 ('statistical', 'iqr', 'isolation')
            
        Returns:
            异常检测结果
        """
        if not data or len(data) < 3:
            return {'anomalies': [], 'anomaly_ratio': 0.0}
            
        try:
            if method == 'statistical':
                return self._statistical_anomaly_detection(data)
            elif method == 'iqr':
                return self._iqr_anomaly_detection(data)
            else:
                return self._statistical_anomaly_detection(data)
                
        except Exception as e:
            print(f"⚠️ 异常检测失败: {e}")
            return {'anomalies': [], 'anomaly_ratio': 0.0}
    
    def _calculate_rolling_stats(self, data: List[float], window_size: int) -> Dict[str, List[float]]:
        """计算滚动统计"""
        if len(data) < window_size:
            return {'mean': [np.mean(data)], 'std': [np.std(data)], 'variance': [np.var(data)]}
            
        rolling_mean = []
        rolling_std = []
        rolling_var = []
        
        for i in range(window_size, len(data) + 1):
            window = data[i-window_size:i]
            rolling_mean.append(np.mean(window))
            rolling_std.append(np.std(window))
            rolling_var.append(np.var(window))
            
        return {
            'mean': rolling_mean,
            'std': rolling_std,
            'variance': rolling_var
        }
    
    def _analyze_trend(self, data: List[float], window_size: int) -> Dict[str, Any]:
        """趋势分析"""
        if len(data) < 3:
            return {'improvement_rate': 0.0, 'direction': 'stable', 'strength': 0.0}
            
        # 线性回归分析趋势
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        # 计算改善率
        if len(data) >= 10:
            recent_mean = np.mean(data[-10:])
            early_mean = np.mean(data[:10])
            improvement_rate = (recent_mean - early_mean) / (abs(early_mean) + 1e-8)
        else:
            improvement_rate = (data[-1] - data[0]) / (abs(data[0]) + 1e-8)
        
        # 趋势方向
        if slope > 0.001:
            direction = 'improving'
        elif slope < -0.001:
            direction = 'declining'
        else:
            direction = 'stable'
            
        return {
            'improvement_rate': improvement_rate,
            'direction': direction,
            'strength': abs(r_value),
            'slope': slope,
            'r_squared': r_value**2
        }
    
    def _analyze_stability(self, data: List[float], window_size: int) -> Dict[str, Any]:
        """稳定性分析"""
        if len(data) < window_size:
            return {'stability_score': 0.0, 'volatility': 1.0, 'plateau_detected': False}
            
        # 计算滚动变异系数
        rolling_cv = []
        for i in range(window_size, len(data) + 1):
            window = data[i-window_size:i]
            mean_val = np.mean(window)
            std_val = np.std(window)
            cv = std_val / (abs(mean_val) + 1e-8)
            rolling_cv.append(cv)
        
        # 稳定性评分
        avg_cv = np.mean(rolling_cv)
        stability_score = max(0, 1 - avg_cv / 0.5)  # 归一化到0-1
        
        # 检测平台期
        recent_cv = np.mean(rolling_cv[-min(10, len(rolling_cv)):])
        plateau_detected = recent_cv < self.stability_threshold
        
        return {
            'stability_score': stability_score,
            'volatility': avg_cv,
            'plateau_detected': plateau_detected,
            'recent_stability': recent_cv
        }
    
    def _calculate_convergence_confidence(self, rolling_stats: Dict, 
                                        trend_analysis: Dict, 
                                        stability_analysis: Dict) -> float:
        """计算收敛置信度"""
        try:
            # 稳定性权重
            stability_weight = 0.4
            stability_score = stability_analysis.get('stability_score', 0.0)
            
            # 趋势权重
            trend_weight = 0.3
            trend_strength = trend_analysis.get('strength', 0.0)
            trend_direction = trend_analysis.get('direction', 'stable')
            
            # 趋势加分：改善趋势加分，下降趋势减分
            if trend_direction == 'improving':
                trend_score = trend_strength
            elif trend_direction == 'declining':
                trend_score = -trend_strength
            else:
                trend_score = 0.5  # 稳定也是一种好的状态
                
            # 方差权重
            variance_weight = 0.3
            variance_trend = rolling_stats.get('variance', [1.0])
            recent_variance = np.mean(variance_trend[-min(5, len(variance_trend)):])
            variance_score = max(0, 1 - recent_variance / 1.0)  # 归一化
            
            # 综合评分
            confidence = (stability_weight * stability_score + 
                         trend_weight * max(0, trend_score) + 
                         variance_weight * variance_score)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception:
            return 0.5
    
    def _composite_quality_score(self, cv_values: List[float], 
                                coupling_ratios: List[float]) -> float:
        """综合质量评分"""
        if not cv_values or not coupling_ratios:
            return 0.0
            
        # CV评分（越小越好）
        current_cv = cv_values[-1]
        cv_score = max(0, 1 - current_cv / 0.5)  # 假设0.5是可接受的上限
        
        # 耦合比例评分（越小越好）
        current_coupling = coupling_ratios[-1]
        coupling_score = max(0, 1 - current_coupling / 0.3)  # 假设0.3是可接受的上限
        
        # 加权平均
        quality_score = (cv_score * 0.6 + coupling_score * 0.4) * 100
        
        return min(100, max(0, quality_score))
    
    def _trend_analysis(self, data: List[float]) -> Dict[str, Any]:
        """通用趋势分析"""
        if len(data) < 3:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'improvement': 0.0}
            
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        # 计算改善百分比
        if len(data) >= 10:
            recent_avg = np.mean(data[-5:])
            early_avg = np.mean(data[:5])
        else:
            recent_avg = data[-1]
            early_avg = data[0]
            
        improvement = (recent_avg - early_avg) / (abs(early_avg) + 1e-8) * 100
        
        return {
            'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
            'slope': slope,
            'improvement': improvement,
            'r_squared': r_value**2
        }
    
    def _coupling_efficiency(self, coupling_ratios: List[float]) -> Dict[str, Any]:
        """耦合效率分析"""
        if not coupling_ratios:
            return {'efficiency': 0.0, 'trend': 'unknown'}
            
        # 效率定义为：1 - 耦合比例（越少耦合越好）
        current_efficiency = (1 - coupling_ratios[-1]) * 100
        
        # 趋势分析
        trend_analysis = self._trend_analysis(coupling_ratios)
        
        return {
            'efficiency': max(0, current_efficiency),
            'trend': 'improving' if trend_analysis['slope'] < 0 else 'declining' if trend_analysis['slope'] > 0 else 'stable',
            'improvement_rate': -trend_analysis['improvement']  # 负号因为耦合减少是改善
        }
    
    def _system_health_assessment(self, cv_values: List[float], 
                                coupling_ratios: List[float]) -> Dict[str, Any]:
        """系统健康度评估"""
        health_score = 50  # 基础分
        
        # CV健康度
        if cv_values:
            current_cv = cv_values[-1]
            if current_cv < 0.1:
                health_score += 25
            elif current_cv < 0.2:
                health_score += 15
            elif current_cv < 0.3:
                health_score += 5
            else:
                health_score -= 10
                
        # 耦合健康度
        if coupling_ratios:
            current_coupling = coupling_ratios[-1]
            if current_coupling < 0.1:
                health_score += 25
            elif current_coupling < 0.2:
                health_score += 15
            elif current_coupling < 0.3:
                health_score += 5
            else:
                health_score -= 10
        
        health_score = min(100, max(0, health_score))
        
        # 健康等级
        if health_score >= 80:
            health_level = 'excellent'
        elif health_score >= 60:
            health_level = 'good'
        elif health_score >= 40:
            health_level = 'fair'
        else:
            health_level = 'poor'
            
        return {
            'health_score': health_score,
            'health_level': health_level,
            'recommendations': self._generate_health_recommendations(cv_values, coupling_ratios)
        }
    
    def _calculate_target_achievement(self, cv_values: List[float], 
                                    coupling_ratios: List[float]) -> Dict[str, Any]:
        """目标达成率计算"""
        # 设定目标值
        cv_target = 0.2
        coupling_target = 0.15
        
        achievements = {}
        
        if cv_values:
            current_cv = cv_values[-1]
            cv_achievement = max(0, (1 - current_cv / cv_target) * 100) if cv_target > 0 else 0
            achievements['cv_achievement'] = min(100, cv_achievement)
            
        if coupling_ratios:
            current_coupling = coupling_ratios[-1]
            coupling_achievement = max(0, (1 - current_coupling / coupling_target) * 100) if coupling_target > 0 else 0
            achievements['coupling_achievement'] = min(100, coupling_achievement)
            
        # 总体达成率
        if achievements:
            overall_achievement = np.mean(list(achievements.values()))
            achievements['overall_achievement'] = overall_achievement
            
        return achievements
    
    def _calculate_improvement(self, data: List[float], 
                             lower_is_better: bool = True) -> float:
        """计算改善率"""
        if len(data) < 2:
            return 0.0
            
        if len(data) >= 10:
            recent_avg = np.mean(data[-5:])
            early_avg = np.mean(data[:5])
        else:
            recent_avg = data[-1]
            early_avg = data[0]
            
        if lower_is_better:
            improvement = (early_avg - recent_avg) / (abs(early_avg) + 1e-8) * 100
        else:
            improvement = (recent_avg - early_avg) / (abs(early_avg) + 1e-8) * 100
            
        return improvement
    
    def _stage_wise_performance(self, stages: List[int], 
                               performance: List[float]) -> Dict[str, Any]:
        """各阶段性能统计"""
        if not stages or not performance:
            return {}
            
        stage_stats = {}
        unique_stages = sorted(set(stages))
        
        for stage in unique_stages:
            stage_indices = [i for i, s in enumerate(stages) if s == stage]
            stage_performance = [performance[i] for i in stage_indices if i < len(performance)]
            
            if stage_performance:
                stage_stats[f'stage_{stage}'] = {
                    'mean_performance': np.mean(stage_performance),
                    'std_performance': np.std(stage_performance),
                    'episodes': len(stage_performance),
                    'best_performance': max(stage_performance),
                    'worst_performance': min(stage_performance)
                }
                
        return stage_stats
    
    def _transition_timing_analysis(self, stages: List[int], 
                                  performance: List[float]) -> Dict[str, Any]:
        """转换时机分析"""
        if not stages or len(stages) < 2:
            return {'transitions': [], 'stage_durations': []}
            
        transitions = []
        stage_durations = []
        current_stage = stages[0]
        stage_start = 0
        
        for i, stage in enumerate(stages[1:], 1):
            if stage != current_stage:
                # 记录转换
                transition = {
                    'from_stage': current_stage,
                    'to_stage': stage,
                    'episode': i,
                    'duration': i - stage_start
                }
                
                # 添加转换时的性能信息
                if i-1 < len(performance):
                    transition['performance_before'] = performance[i-1]
                if i < len(performance):
                    transition['performance_after'] = performance[i]
                    
                transitions.append(transition)
                stage_durations.append(i - stage_start)
                
                current_stage = stage
                stage_start = i
                
        # 最后一个阶段的持续时间
        if len(stages) > stage_start:
            stage_durations.append(len(stages) - stage_start)
            
        return {
            'transitions': transitions,
            'stage_durations': stage_durations,
            'total_transitions': len(transitions),
            'average_stage_duration': np.mean(stage_durations) if stage_durations else 0
        }
    
    def _learning_efficiency_per_stage(self, stages: List[int], 
                                     performance: List[float]) -> Dict[str, Any]:
        """各阶段学习效率"""
        stage_performance = self._stage_wise_performance(stages, performance)
        
        efficiency_scores = {}
        
        for stage_key, stats in stage_performance.items():
            episodes = stats['episodes']
            improvement = stats['best_performance'] - stats['worst_performance']
            
            # 效率定义为：改善程度 / 训练时间
            if episodes > 0:
                efficiency = improvement / episodes
            else:
                efficiency = 0.0
                
            efficiency_scores[stage_key] = {
                'efficiency': efficiency,
                'improvement': improvement,
                'episodes': episodes
            }
            
        return efficiency_scores
    
    def _evaluate_curriculum_strategy(self, stages: List[int], 
                                    performance: List[float]) -> Dict[str, Any]:
        """评估课程策略"""
        if not stages or not performance:
            return {'effectiveness_score': 0.0, 'strategy': 'unknown'}
            
        # 计算不同阶段的平均性能
        stage_performance = self._stage_wise_performance(stages, performance)
        
        if len(stage_performance) < 2:
            return {'effectiveness_score': 0.5, 'strategy': 'single_stage'}
            
        # 检查阶段间的性能提升
        stage_means = []
        for stage_key in sorted(stage_performance.keys()):
            stage_means.append(stage_performance[stage_key]['mean_performance'])
            
        # 计算阶段间改善
        improvements = []
        for i in range(1, len(stage_means)):
            improvement = (stage_means[i] - stage_means[i-1]) / (abs(stage_means[i-1]) + 1e-8)
            improvements.append(improvement)
            
        # 效果评分
        if improvements:
            avg_improvement = np.mean(improvements)
            positive_improvements = sum(1 for imp in improvements if imp > 0)
            improvement_ratio = positive_improvements / len(improvements)
            
            effectiveness_score = (avg_improvement * 0.5 + improvement_ratio * 0.5)
            effectiveness_score = min(1.0, max(0.0, effectiveness_score))
        else:
            effectiveness_score = 0.5
            
        return {
            'effectiveness_score': effectiveness_score,
            'strategy': 'progressive' if effectiveness_score > 0.6 else 'needs_improvement',
            'stage_improvements': improvements,
            'average_improvement': np.mean(improvements) if improvements else 0.0
        }
    
    def _statistical_anomaly_detection(self, data: List[float]) -> Dict[str, Any]:
        """统计异常检测"""
        if len(data) < 3:
            return {'anomalies': [], 'anomaly_ratio': 0.0}
            
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        # 使用3σ准则
        threshold = 3 * std
        anomalies = []
        
        for i, value in enumerate(data_array):
            if abs(value - mean) > threshold:
                anomalies.append({
                    'index': i,
                    'value': value,
                    'deviation': abs(value - mean),
                    'severity': 'high' if abs(value - mean) > 4 * std else 'medium'
                })
                
        return {
            'anomalies': anomalies,
            'anomaly_ratio': len(anomalies) / len(data),
            'detection_method': 'statistical',
            'threshold': threshold
        }
    
    def _iqr_anomaly_detection(self, data: List[float]) -> Dict[str, Any]:
        """IQR异常检测"""
        if len(data) < 3:
            return {'anomalies': [], 'anomaly_ratio': 0.0}
            
        data_array = np.array(data)
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = []
        for i, value in enumerate(data_array):
            if value < lower_bound or value > upper_bound:
                anomalies.append({
                    'index': i,
                    'value': value,
                    'type': 'low' if value < lower_bound else 'high',
                    'severity': 'high' if (value < Q1 - 3 * IQR or value > Q3 + 3 * IQR) else 'medium'
                })
                
        return {
            'anomalies': anomalies,
            'anomaly_ratio': len(anomalies) / len(data),
            'detection_method': 'iqr',
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def _generate_health_recommendations(self, cv_values: List[float], 
                                       coupling_ratios: List[float]) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        if cv_values:
            current_cv = cv_values[-1]
            if current_cv > 0.3:
                recommendations.append("负载平衡需要改善，建议优化分区策略")
            elif current_cv > 0.2:
                recommendations.append("负载平衡表现良好，可进一步优化")
            else:
                recommendations.append("负载平衡表现优秀")
                
        if coupling_ratios:
            current_coupling = coupling_ratios[-1]
            if current_coupling > 0.3:
                recommendations.append("耦合边过多，建议调整奖励权重")
            elif current_coupling > 0.2:
                recommendations.append("耦合边比例合理，可适当降低")
            else:
                recommendations.append("耦合边控制优秀")
                
        return recommendations
    
    def _empty_convergence_metrics(self) -> Dict[str, Any]:
        """空的收敛性指标"""
        return {
            'variance_trend': [],
            'mean_trend': [],
            'improvement_rate': 0.0,
            'trend_direction': 'unknown',
            'trend_strength': 0.0,
            'stability_score': 0.0,
            'volatility': 1.0,
            'convergence_confidence': 0.0,
            'is_converged': False,
            'plateau_detected': False,
            'analysis_window': 0
        }
    
    def _empty_power_system_analysis(self) -> Dict[str, Any]:
        """空的电力系统分析"""
        return {
            'quality_score': 0.0,
            'balance_trend': {},
            'coupling_efficiency': {},
            'system_health': {'health_score': 0.0, 'health_level': 'unknown'},
            'target_achievement': {},
            'current_cv': 1.0,
            'current_coupling_ratio': 1.0,
            'cv_improvement': 0.0,
            'coupling_improvement': 0.0
        }
    
    def _empty_curriculum_analysis(self) -> Dict[str, Any]:
        """空的课程分析"""
        return {
            'stage_performance': {},
            'transition_analysis': {'transitions': [], 'stage_durations': []},
            'learning_efficiency': {},
            'curriculum_strategy': {'effectiveness_score': 0.0, 'strategy': 'unknown'},
            'total_stages': 0,
            'average_stage_duration': 0.0,
            'curriculum_effectiveness': 0.0
        } 