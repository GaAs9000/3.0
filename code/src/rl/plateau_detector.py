"""
增强的平台期检测系统

实现"从急性子变成老司机"的智能阶段转换机制：
- 稳定性检测：不只看平均值，还要分析方差、趋势、波动
- 趋势验证：区分真实提升和随机波动
- 置信度计算：量化转换决策的可靠性
- 回退保护：检测转换后性能恶化并自动恢复

作者：Augment Agent
日期：2025-06-30
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlateauDetectionConfig:
    """平台期检测配置"""
    # 启用开关
    enabled: bool = True                    # 是否启用平台期检测

    # 稳定性检测参数
    stability_window: int = 50              # 稳定性检测窗口
    stability_cv_threshold: float = 0.15    # 变异系数阈值
    stability_trend_threshold: float = 0.05 # 趋势稳定性阈值
    
    # 趋势验证参数
    trend_window_short: int = 20            # 短期趋势窗口
    trend_window_medium: int = 50           # 中期趋势窗口
    trend_window_long: int = 100            # 长期趋势窗口
    trend_consistency_threshold: float = 0.7 # 趋势一致性阈值
    
    # 置信度计算参数
    confidence_threshold: float = 0.8       # 转换置信度阈值
    stability_weight: float = 0.4           # 稳定性权重
    trend_weight: float = 0.3               # 趋势权重
    performance_weight: float = 0.3         # 性能水平权重
    
    # 回退保护参数
    fallback_enabled: bool = True           # 是否启用回退
    fallback_performance_threshold: float = 0.8  # 回退性能阈值
    fallback_observation_window: int = 30   # 回退观察窗口
    
    # 性能改善要求
    min_improvement_rate: float = 0.05      # 最小改善率
    improvement_consistency_window: int = 30 # 改善一致性窗口


class StabilityDetector:
    """稳定性检测器"""
    
    def __init__(self, config: PlateauDetectionConfig):
        self.config = config
        
    def analyze_stability(self, values: List[float]) -> Dict[str, float]:
        """分析数值序列的稳定性"""
        if len(values) < self.config.stability_window:
            return {'stability_score': 0.0, 'cv': float('inf'), 'trend_stability': 0.0}
        
        recent_values = values[-self.config.stability_window:]
        
        # 计算变异系数
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        cv = std_val / (abs(mean_val) + 1e-8)  # 避免除零
        
        # 计算趋势稳定性（线性回归的R²）
        x = np.arange(len(recent_values))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
            trend_stability = r_value ** 2  # R²值
        except:
            trend_stability = 0.0
        
        # 计算局部波动性（相邻值差异的标准差）
        diffs = np.diff(recent_values)
        local_volatility = np.std(diffs) / (abs(mean_val) + 1e-8)
        
        # 综合稳定性评分
        cv_score = max(0, 1 - cv / self.config.stability_cv_threshold)
        trend_score = min(1, trend_stability / self.config.stability_trend_threshold)
        volatility_score = max(0, 1 - local_volatility)
        
        stability_score = (cv_score + trend_score + volatility_score) / 3
        
        return {
            'stability_score': stability_score,
            'cv': cv,
            'trend_stability': trend_stability,
            'local_volatility': local_volatility,
            'cv_score': cv_score,
            'trend_score': trend_score,
            'volatility_score': volatility_score
        }


class TrendValidator:
    """趋势验证器"""
    
    def __init__(self, config: PlateauDetectionConfig):
        self.config = config
        
    def validate_trend(self, values: List[float]) -> Dict[str, Any]:
        """验证趋势的真实性和一致性"""
        if len(values) < self.config.trend_window_long:
            return {'trend_confidence': 0.0, 'trend_direction': 'unknown'}
        
        # 多时间尺度趋势分析
        trends = {}
        for window_name, window_size in [
            ('short', self.config.trend_window_short),
            ('medium', self.config.trend_window_medium),
            ('long', self.config.trend_window_long)
        ]:
            if len(values) >= window_size:
                recent_values = values[-window_size:]
                x = np.arange(len(recent_values))
                try:
                    slope, _, r_value, p_value, _ = stats.linregress(x, recent_values)
                    trends[window_name] = {
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'direction': 'up' if slope > 0 else 'down' if slope < 0 else 'flat'
                    }
                except:
                    trends[window_name] = {'slope': 0, 'r_squared': 0, 'p_value': 1, 'direction': 'flat'}
        
        # 趋势一致性检查
        directions = [t['direction'] for t in trends.values()]
        direction_consistency = len(set(directions)) == 1  # 所有时间尺度方向一致
        
        # 趋势强度评估
        avg_r_squared = np.mean([t['r_squared'] for t in trends.values()])
        avg_p_value = np.mean([t['p_value'] for t in trends.values()])
        
        # 趋势置信度计算
        consistency_score = 1.0 if direction_consistency else 0.5
        strength_score = min(1.0, avg_r_squared * 2)  # R²越高趋势越明显
        significance_score = max(0, 1 - avg_p_value * 2)  # p值越小越显著
        
        trend_confidence = (consistency_score + strength_score + significance_score) / 3
        
        return {
            'trend_confidence': trend_confidence,
            'trend_direction': directions[0] if direction_consistency else 'inconsistent',
            'trends': trends,
            'direction_consistency': direction_consistency,
            'avg_r_squared': avg_r_squared,
            'avg_p_value': avg_p_value,
            'consistency_score': consistency_score,
            'strength_score': strength_score,
            'significance_score': significance_score
        }


class PlateauConfidenceCalculator:
    """平台期置信度计算器"""
    
    def __init__(self, config: PlateauDetectionConfig):
        self.config = config
        
    def calculate_confidence(self, 
                           stability_analysis: Dict[str, float],
                           trend_analysis: Dict[str, Any],
                           performance_level: float,
                           target_performance: float) -> Dict[str, float]:
        """计算平台期转换的置信度"""
        
        # 稳定性置信度
        stability_confidence = stability_analysis['stability_score']
        
        # 趋势置信度
        trend_confidence = trend_analysis['trend_confidence']
        
        # 性能水平置信度
        performance_ratio = performance_level / (target_performance + 1e-8)
        performance_confidence = min(1.0, performance_ratio)
        
        # 加权综合置信度
        total_confidence = (
            self.config.stability_weight * stability_confidence +
            self.config.trend_weight * trend_confidence +
            self.config.performance_weight * performance_confidence
        )
        
        return {
            'total_confidence': total_confidence,
            'stability_confidence': stability_confidence,
            'trend_confidence': trend_confidence,
            'performance_confidence': performance_confidence,
            'meets_threshold': total_confidence >= self.config.confidence_threshold
        }


class EnhancedPlateauDetector:
    """增强的平台期检测器"""
    
    def __init__(self, config: PlateauDetectionConfig):
        self.config = config
        self.stability_detector = StabilityDetector(config)
        self.trend_validator = TrendValidator(config)
        self.confidence_calculator = PlateauConfidenceCalculator(config)
        
        # 性能记忆库
        self.performance_memory = {}
        self.stage_best_points = {}
        
        # 回退状态跟踪
        self.fallback_history = []
        self.last_transition_episode = -1
        
    def should_transition(self, 
                         performance_history: List[Dict[str, Any]],
                         current_stage: int,
                         target_metric: str,
                         target_value: float,
                         episode: int) -> Dict[str, Any]:
        """
        判断是否应该进行阶段转换
        
        Args:
            performance_history: 性能历史记录
            current_stage: 当前阶段
            target_metric: 目标指标名称
            target_value: 目标值
            episode: 当前episode
            
        Returns:
            包含转换决策和详细分析的字典
        """
        
        if len(performance_history) < self.config.stability_window:
            return {
                'should_transition': False,
                'reason': 'insufficient_data',
                'confidence': 0.0,
                'analysis': {}
            }
        
        # 提取目标指标的历史值
        metric_values = [p.get(target_metric, 0) for p in performance_history]
        
        # 1. 稳定性分析
        stability_analysis = self.stability_detector.analyze_stability(metric_values)
        
        # 2. 趋势验证
        trend_analysis = self.trend_validator.validate_trend(metric_values)
        
        # 3. 当前性能水平
        current_performance = metric_values[-1] if metric_values else 0
        
        # 4. 置信度计算
        confidence_analysis = self.confidence_calculator.calculate_confidence(
            stability_analysis, trend_analysis, current_performance, target_value
        )
        
        # 5. 综合决策
        should_transition = (
            confidence_analysis['meets_threshold'] and
            current_performance >= target_value and
            stability_analysis['stability_score'] >= 0.6 and
            trend_analysis['trend_confidence'] >= 0.5
        )
        
        # 6. 记录最佳稳定点
        if should_transition:
            self._record_best_stable_point(current_stage, performance_history[-1], episode)
        
        return {
            'should_transition': should_transition,
            'confidence': confidence_analysis['total_confidence'],
            'reason': self._get_decision_reason(should_transition, confidence_analysis, stability_analysis, trend_analysis),
            'analysis': {
                'stability': stability_analysis,
                'trend': trend_analysis,
                'confidence': confidence_analysis,
                'current_performance': current_performance,
                'target_value': target_value
            }
        }
    
    def _record_best_stable_point(self, stage: int, performance_data: Dict[str, Any], episode: int):
        """记录阶段的最佳稳定点"""
        self.stage_best_points[stage] = {
            'performance': performance_data.copy(),
            'episode': episode,
            'timestamp': episode
        }
        
    def _get_decision_reason(self, should_transition: bool, confidence_analysis: Dict, 
                           stability_analysis: Dict, trend_analysis: Dict) -> str:
        """获取决策原因的详细说明"""
        if should_transition:
            return f"平台期确认：置信度{confidence_analysis['total_confidence']:.3f}，稳定性{stability_analysis['stability_score']:.3f}，趋势{trend_analysis['trend_confidence']:.3f}"
        else:
            reasons = []
            if confidence_analysis['total_confidence'] < self.config.confidence_threshold:
                reasons.append(f"置信度不足({confidence_analysis['total_confidence']:.3f}<{self.config.confidence_threshold})")
            if stability_analysis['stability_score'] < 0.6:
                reasons.append(f"稳定性不足({stability_analysis['stability_score']:.3f}<0.6)")
            if trend_analysis['trend_confidence'] < 0.5:
                reasons.append(f"趋势不明确({trend_analysis['trend_confidence']:.3f}<0.5)")
            
            return "等待平台期：" + "，".join(reasons)

    def check_fallback_needed(self,
                             performance_history: List[Dict[str, Any]],
                             current_stage: int,
                             transition_episode: int) -> Dict[str, Any]:
        """检查是否需要回退到上一阶段"""
        if not self.config.fallback_enabled:
            return {'needs_fallback': False, 'reason': 'fallback_disabled'}

        if current_stage <= 1:
            return {'needs_fallback': False, 'reason': 'first_stage'}

        # 检查转换后的观察窗口
        episodes_since_transition = len(performance_history) - transition_episode
        if episodes_since_transition < self.config.fallback_observation_window:
            return {'needs_fallback': False, 'reason': 'observation_period'}

        # 获取转换前后的性能对比
        pre_transition_performance = self._get_pre_transition_performance(
            performance_history, transition_episode
        )
        post_transition_performance = self._get_post_transition_performance(
            performance_history, transition_episode
        )

        if not pre_transition_performance or not post_transition_performance:
            return {'needs_fallback': False, 'reason': 'insufficient_data'}

        # 性能恶化检测
        performance_ratio = post_transition_performance / (pre_transition_performance + 1e-8)

        needs_fallback = performance_ratio < self.config.fallback_performance_threshold

        if needs_fallback:
            self.fallback_history.append({
                'from_stage': current_stage,
                'to_stage': current_stage - 1,
                'episode': len(performance_history),
                'performance_ratio': performance_ratio,
                'reason': 'performance_deterioration'
            })

        return {
            'needs_fallback': needs_fallback,
            'reason': f'performance_ratio_{performance_ratio:.3f}' if needs_fallback else 'performance_stable',
            'performance_ratio': performance_ratio,
            'pre_performance': pre_transition_performance,
            'post_performance': post_transition_performance
        }

    def _get_pre_transition_performance(self, performance_history: List[Dict], transition_episode: int) -> Optional[float]:
        """获取转换前的平均性能"""
        if transition_episode < self.config.fallback_observation_window:
            return None

        pre_data = performance_history[
            max(0, transition_episode - self.config.fallback_observation_window):transition_episode
        ]

        if not pre_data:
            return None

        # 使用复合评分作为性能指标
        scores = [p.get('composite_score', p.get('episode_reward', 0)) for p in pre_data]
        return np.mean(scores)

    def _get_post_transition_performance(self, performance_history: List[Dict], transition_episode: int) -> Optional[float]:
        """获取转换后的平均性能"""
        post_data = performance_history[transition_episode:]

        if len(post_data) < self.config.fallback_observation_window:
            return None

        # 使用最近的观察窗口数据
        recent_data = post_data[-self.config.fallback_observation_window:]
        scores = [p.get('composite_score', p.get('episode_reward', 0)) for p in recent_data]
        return np.mean(scores)

    def get_fallback_target(self, current_stage: int) -> Dict[str, Any]:
        """获取回退目标状态"""
        target_stage = current_stage - 1

        if target_stage in self.stage_best_points:
            return {
                'target_stage': target_stage,
                'target_parameters': self.stage_best_points[target_stage]['performance'],
                'fallback_type': 'best_stable_point'
            }
        else:
            return {
                'target_stage': target_stage,
                'target_parameters': None,
                'fallback_type': 'default_stage_parameters'
            }

    def get_status_summary(self) -> Dict[str, Any]:
        """获取检测器状态摘要"""
        return {
            'config': {
                'confidence_threshold': self.config.confidence_threshold,
                'stability_window': self.config.stability_window,
                'fallback_enabled': self.config.fallback_enabled
            },
            'memory': {
                'best_points_recorded': len(self.stage_best_points),
                'fallback_history_count': len(self.fallback_history)
            },
            'recent_fallbacks': self.fallback_history[-3:] if self.fallback_history else []
        }


class PlateauDetectorIntegration:
    """平台期检测器集成接口"""

    def __init__(self, adaptive_director):
        """
        集成到现有的AdaptiveDirector中

        Args:
            adaptive_director: 现有的AdaptiveDirector实例
        """
        self.director = adaptive_director

        # 从配置中获取平台期检测配置
        plateau_config_dict = self.director.config.get('adaptive_curriculum', {}).get('plateau_detection', {})
        self.plateau_config = PlateauDetectionConfig(**plateau_config_dict)

        # 创建增强检测器
        self.plateau_detector = EnhancedPlateauDetector(self.plateau_config)

        # 保存原始转换方法的引用
        self.original_check_stage_transition = self.director._check_stage_transition

        # 替换转换检查方法
        self.director._check_stage_transition = self._enhanced_check_stage_transition

        logger.info("平台期检测器已集成到AdaptiveDirector")

    def _enhanced_check_stage_transition(self, performance: Dict[str, float]):
        """增强的阶段转换检查"""
        current_stage = self.director.parameter_scheduler.current_stage

        # 根据当前阶段选择合适的检测策略
        if current_stage == 1:
            self._check_enhanced_stage1_to_2_transition(performance)
        elif current_stage == 2:
            self._check_enhanced_stage2_to_3_transition(performance)
        elif current_stage == 3:
            self._check_enhanced_stage3_to_4_transition(performance)

    def _check_enhanced_stage1_to_2_transition(self, performance: Dict[str, float]):
        """增强的阶段1到2转换检查"""
        decision = self.plateau_detector.should_transition(
            performance_history=list(self.director.performance_analyzer.episode_history),
            current_stage=1,
            target_metric='episode_length',
            target_value=self.director.transition_criteria.episode_length_target,
            episode=self.director.episode_count
        )

        if decision['should_transition']:
            self.director.parameter_scheduler.update_stage(2, self.director.episode_count)
            self.director.stage_transition_history.append({
                'episode': self.director.episode_count,
                'from_stage': 1,
                'to_stage': 2,
                'trigger': 'enhanced_plateau_detection',
                'confidence': decision['confidence'],
                'reason': decision['reason']
            })

            if self.director.verbose:
                logger.info(f"阶段转换 1→2: {decision['reason']}")

    def _check_enhanced_stage2_to_3_transition(self, performance: Dict[str, float]):
        """增强的阶段2到3转换检查"""
        decision = self.plateau_detector.should_transition(
            performance_history=list(self.director.performance_analyzer.performance_history),
            current_stage=2,
            target_metric='composite_score',
            target_value=self.director.transition_criteria.composite_score_target,
            episode=self.director.episode_count
        )

        if decision['should_transition']:
            self.director.parameter_scheduler.update_stage(3, self.director.episode_count)
            self.director.stage_transition_history.append({
                'episode': self.director.episode_count,
                'from_stage': 2,
                'to_stage': 3,
                'trigger': 'enhanced_plateau_detection',
                'confidence': decision['confidence'],
                'reason': decision['reason']
            })

            if self.director.verbose:
                logger.info(f"阶段转换 2→3: {decision['reason']}")

    def _check_enhanced_stage3_to_4_transition(self, performance: Dict[str, float]):
        """增强的阶段3到4转换检查"""
        # 阶段3到4的转换保持相对保守
        episodes_in_stage3 = self.director.episode_count - self.director.parameter_scheduler.stage_start_episode

        if episodes_in_stage3 >= 200:  # 减少等待时间
            decision = self.plateau_detector.should_transition(
                performance_history=list(self.director.performance_analyzer.performance_history),
                current_stage=3,
                target_metric='composite_score',
                target_value=0.8,  # 更高的要求
                episode=self.director.episode_count
            )

            if decision['should_transition']:
                self.director.parameter_scheduler.update_stage(4, self.director.episode_count)
                self.director.stage_transition_history.append({
                    'episode': self.director.episode_count,
                    'from_stage': 3,
                    'to_stage': 4,
                    'trigger': 'enhanced_plateau_detection',
                    'confidence': decision['confidence'],
                    'reason': decision['reason']
                })

                if self.director.verbose:
                    logger.info(f"阶段转换 3→4: {decision['reason']}")

    def check_and_handle_fallback(self) -> bool:
        """检查并处理回退需求"""
        if not self.director.stage_transition_history:
            return False

        last_transition = self.director.stage_transition_history[-1]
        current_stage = self.director.parameter_scheduler.current_stage

        fallback_check = self.plateau_detector.check_fallback_needed(
            performance_history=list(self.director.performance_analyzer.performance_history),
            current_stage=current_stage,
            transition_episode=last_transition['episode']
        )

        if fallback_check['needs_fallback']:
            fallback_target = self.plateau_detector.get_fallback_target(current_stage)

            # 执行回退
            self.director.parameter_scheduler.update_stage(
                fallback_target['target_stage'],
                self.director.episode_count
            )

            # 记录回退事件
            self.director.stage_transition_history.append({
                'episode': self.director.episode_count,
                'from_stage': current_stage,
                'to_stage': fallback_target['target_stage'],
                'trigger': 'automatic_fallback',
                'reason': fallback_check['reason'],
                'performance_ratio': fallback_check.get('performance_ratio', 0)
            })

            logger.warning(f"执行自动回退 {current_stage}→{fallback_target['target_stage']}: {fallback_check['reason']}")
            return True

        return False
