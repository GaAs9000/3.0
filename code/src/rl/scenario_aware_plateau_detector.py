"""
场景感知奖励系统 - 场景感知平台期检测器

实现基于场景感知历史追踪器的平台期检测，确保平台期检测在同类难度场景内进行比较。

作者：Augment Agent
日期：2025-01-15
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
import warnings
from .scenario_context import ScenarioContext
from .scenario_aware_tracker import ScenarioAwareHistoryTracker

# 抑制numpy的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)


class PlateauResult(NamedTuple):
    """平台期检测结果"""
    plateau_detected: bool
    confidence: float
    improvement_rate: float
    stability_score: float
    historical_percentile: float
    details: Dict[str, float]


class ScenarioAwarePlateauDetector:
    """
    场景感知平台期检测器
    
    基于场景感知历史追踪器实现平台期检测，确保同类场景内进行比较
    """

    def __init__(self,
                 history_tracker: ScenarioAwareHistoryTracker,
                 config: Optional[Dict] = None):
        """
        初始化场景感知平台期检测器

        Args:
            history_tracker: 场景感知历史追踪器
            config: 检测配置
        """
        self.history_tracker = history_tracker
        self.config = config or {}
        
        # 检测参数
        self.window_size = self.config.get('window_size', 15)
        self.min_improvement_rate = self.config.get('min_improvement_rate', 0.005)
        self.stability_threshold = self.config.get('stability_threshold', 0.8)
        self.min_percentile = self.config.get('min_percentile', 0.7)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        
        # 数值稳定性参数
        self.epsilon = 1e-9
        
        # 当前场景上下文
        self.current_scenario_context = None

    def detect_plateau(self, 
                      quality_score: float, 
                      scenario_context: ScenarioContext) -> PlateauResult:
        """
        检测平台期

        Args:
            quality_score: 当前质量分数 [0, 1]
            scenario_context: 场景上下文

        Returns:
            PlateauResult: 检测结果
        """
        # 输入验证
        if np.isnan(quality_score) or np.isinf(quality_score):
            quality_score = 0.0
        quality_score = np.clip(quality_score, 0.0, 1.0)
        
        # 更新历史数据
        self.history_tracker.update_history(quality_score, scenario_context)
        self.current_scenario_context = scenario_context
        
        # 获取最近分数用于局部分析
        recent_scores = self.history_tracker.get_recent_scores()
        
        # 如果数据不足，不进行检测
        if len(recent_scores) < min(5, self.window_size):
            return PlateauResult(
                plateau_detected=False,
                confidence=0.0,
                improvement_rate=1.0,  # 假设还在改善
                stability_score=0.0,
                historical_percentile=0.0,
                details={'reason': 'insufficient_data', 'data_count': len(recent_scores)}
            )

        # 执行三层检测
        improvement_rate = self._compute_improvement_rate(recent_scores)
        stability_score = self._compute_stability_score(recent_scores)
        
        # 🔥 关键改进：场景内百分位计算
        historical_percentile = self.history_tracker.compute_scenario_percentile(
            quality_score, scenario_context
        )

        # 综合判断
        plateau_detected = (
            improvement_rate < self.min_improvement_rate and
            stability_score > self.stability_threshold and
            historical_percentile > self.min_percentile
        )

        # 计算置信度
        confidence = self._compute_confidence(
            improvement_rate, stability_score, historical_percentile
        )

        # 详细信息
        scenario_stats = self.history_tracker.get_scenario_statistics(scenario_context)
        details = {
            'window_size': len(recent_scores),
            'scenario_key': scenario_stats.get('scenario_key', 'unknown'),
            'scenario_count': scenario_stats.get('count', 0),
            'current_score': quality_score,
            'mean_recent': np.mean(recent_scores),
            'std_recent': np.std(recent_scores),
            'min_recent': np.min(recent_scores),
            'max_recent': np.max(recent_scores),
            'scenario_mean': scenario_stats.get('mean', 0.0),
            'scenario_std': scenario_stats.get('std', 0.0)
        }

        return PlateauResult(
            plateau_detected=plateau_detected,
            confidence=confidence,
            improvement_rate=improvement_rate,
            stability_score=stability_score,
            historical_percentile=historical_percentile,
            details=details
        )

    def _compute_improvement_rate(self, recent_scores: List[float]) -> float:
        """
        计算改善率（基于线性回归斜率）

        Args:
            recent_scores: 最近的质量分数列表

        Returns:
            改善率的绝对值，越小表示改善越慢
        """
        if len(recent_scores) < 3:
            return 1.0  # 数据不足，假设还在改善

        try:
            scores = np.array(recent_scores)
            x = np.arange(len(scores))

            # 线性回归计算斜率
            slope = np.polyfit(x, scores, 1)[0]

            # 返回斜率的绝对值作为改善率
            improvement_rate = abs(slope)

            # 数值稳定性保护
            if np.isnan(improvement_rate) or np.isinf(improvement_rate):
                improvement_rate = 1.0

            return improvement_rate

        except Exception:
            return 1.0

    def _compute_stability_score(self, recent_scores: List[float]) -> float:
        """
        计算稳定性分数

        Args:
            recent_scores: 最近的质量分数列表

        Returns:
            稳定性分数 [0, 1]，越大表示越稳定
        """
        if len(recent_scores) < 2:
            return 0.0

        try:
            variance = np.var(recent_scores)

            # 使用反比函数：stability = 1 / (1 + variance)
            stability_score = 1.0 / (1.0 + variance + self.epsilon)

            # 数值稳定性保护
            if np.isnan(stability_score) or np.isinf(stability_score):
                stability_score = 0.0

            return np.clip(stability_score, 0.0, 1.0)

        except Exception:
            return 0.0

    def _compute_confidence(self,
                           improvement_rate: float,
                           stability_score: float,
                           historical_percentile: float) -> float:
        """
        计算综合置信度

        Args:
            improvement_rate: 改善率
            stability_score: 稳定性分数  
            historical_percentile: 历史百分位

        Returns:
            综合置信度 [0, 1]
        """
        try:
            # 改善率置信度：改善率越小，置信度越高
            improvement_confidence = 1.0 - min(improvement_rate / self.min_improvement_rate, 1.0)

            # 稳定性置信度：直接使用稳定性分数
            stability_confidence = stability_score

            # 历史表现置信度：百分位越高，置信度越高
            historical_confidence = max(0.0, historical_percentile - self.min_percentile) / (1.0 - self.min_percentile)

            # 加权平均计算综合置信度
            weights = [0.4, 0.3, 0.3]  # 改善率、稳定性、历史表现的权重
            confidence = (
                weights[0] * improvement_confidence +
                weights[1] * stability_confidence +
                weights[2] * historical_confidence
            )

            return np.clip(confidence, 0.0, 1.0)

        except Exception:
            return 0.0

    def should_early_stop(self, 
                         quality_score: float, 
                         scenario_context: ScenarioContext) -> Tuple[bool, float]:
        """
        判断是否应该提前停止

        Args:
            quality_score: 当前质量分数
            scenario_context: 场景上下文

        Returns:
            (是否提前停止, 置信度)
        """
        result = self.detect_plateau(quality_score, scenario_context)
        
        early_stop = (
            result.plateau_detected and 
            result.confidence > self.confidence_threshold
        )
        
        return early_stop, result.confidence

    def get_plateau_statistics(self) -> Dict[str, any]:
        """
        获取平台期检测统计信息

        Returns:
            统计信息字典
        """
        base_stats = self.history_tracker.get_global_statistics()
        
        if self.current_scenario_context:
            current_scenario_stats = self.history_tracker.get_scenario_statistics(
                self.current_scenario_context
            )
        else:
            current_scenario_stats = {}
        
        return {
            'global_stats': base_stats,
            'current_scenario_stats': current_scenario_stats,
            'detection_config': {
                'window_size': self.window_size,
                'min_improvement_rate': self.min_improvement_rate,
                'stability_threshold': self.stability_threshold,
                'min_percentile': self.min_percentile,
                'confidence_threshold': self.confidence_threshold
            }
        }

    def reset(self):
        """重置检测器状态"""
        self.current_scenario_context = None