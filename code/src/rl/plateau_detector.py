"""
自适应质量导向训练系统 - 平台期检测器

实现基于相对改善率的自适应平台期检测算法，用于自适应质量导向训练系统。
核心功能：
1. 改善率检测：基于线性回归斜率分析质量改善趋势
2. 稳定性检测：基于方差分析质量分数的稳定性
3. 相对水平检测：基于历史百分位数评估当前表现水平
4. 综合判断：多维度置信度计算和平台期判定

作者：Augment Agent
日期：2025-07-01
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from collections import deque
import warnings

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


class QualityPlateauDetector:
    """
    质量平台期检测器

    基于技术方案中的算法实现：
    1. 改善率检测：|slope| < min_improvement_threshold
    2. 稳定性检测：stability_score > stability_threshold
    3. 相对水平检测：historical_percentile > min_percentile_threshold
    4. 综合判断：加权平均置信度计算
    """

    def __init__(self,
                 window_size: int = 15,
                 min_improvement_rate: float = 0.005,
                 stability_threshold: float = 0.8,
                 min_percentile: float = 0.7,
                 confidence_threshold: float = 0.8):
        """
        初始化平台期检测器

        Args:
            window_size: 观察窗口大小
            min_improvement_rate: 最小改善率阈值
            stability_threshold: 稳定性要求
            min_percentile: 历史表现要求(前70%)
            confidence_threshold: 早停置信度要求
        """
        self.window_size = window_size
        self.min_improvement_rate = min_improvement_rate
        self.stability_threshold = stability_threshold
        self.min_percentile = min_percentile
        self.confidence_threshold = confidence_threshold

        # 数据存储
        self.recent_scores = deque(maxlen=window_size)
        self.all_history_scores = []

        # 数值稳定性参数
        self.epsilon = 1e-9



    def update(self, quality_score: float) -> PlateauResult:
        """
        更新质量分数并检测平台期

        Args:
            quality_score: 当前质量分数 [0, 1]，越大越好

        Returns:
            PlateauResult: 检测结果
        """
        # 输入验证
        if np.isnan(quality_score) or np.isinf(quality_score):
            quality_score = 0.0

        quality_score = np.clip(quality_score, 0.0, 1.0)

        # 更新数据
        self.recent_scores.append(quality_score)
        self.all_history_scores.append(quality_score)

        # 如果数据不足，不进行检测
        if len(self.recent_scores) < min(5, self.window_size):
            return PlateauResult(
                plateau_detected=False,
                confidence=0.0,
                improvement_rate=1.0,  # 假设还在改善
                stability_score=0.0,
                historical_percentile=0.0,
                details={}
            )

        # 执行三层检测
        improvement_rate = self._compute_improvement_rate()
        stability_score = self._compute_stability_score()
        historical_percentile = self._compute_historical_percentile()

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
        details = {
            'window_size': len(self.recent_scores),
            'total_history': len(self.all_history_scores),
            'current_score': quality_score,
            'mean_recent': np.mean(self.recent_scores),
            'std_recent': np.std(self.recent_scores),
            'min_recent': np.min(self.recent_scores),
            'max_recent': np.max(self.recent_scores)
        }

        return PlateauResult(
            plateau_detected=plateau_detected,
            confidence=confidence,
            improvement_rate=improvement_rate,
            stability_score=stability_score,
            historical_percentile=historical_percentile,
            details=details
        )



    def _compute_improvement_rate(self) -> float:
        """
        计算改善率（基于线性回归斜率）

        Returns:
            改善率的绝对值，越小表示改善越慢
        """
        if len(self.recent_scores) < 3:
            return 1.0  # 数据不足，假设还在改善

        try:
            scores = np.array(self.recent_scores)
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

    def _compute_stability_score(self) -> float:
        """
        计算稳定性分数

        Returns:
            稳定性分数 [0, 1]，越大表示越稳定
        """
        if len(self.recent_scores) < 2:
            return 0.0

        try:
            variance = np.var(self.recent_scores)

            # 使用反比函数：stability = 1 / (1 + variance)
            stability_score = 1.0 / (1.0 + variance + self.epsilon)

            # 数值稳定性保护
            if np.isnan(stability_score) or np.isinf(stability_score):
                stability_score = 0.0

            return np.clip(stability_score, 0.0, 1.0)

        except Exception:
            return 0.0



    def _compute_historical_percentile(self) -> float:
        """
        计算历史百分位数

        Returns:
            当前分数在历史中的百分位数 [0, 1]
        """
        if len(self.all_history_scores) < 2:
            return 0.0

        try:
            current_score = self.recent_scores[-1]

            # 计算百分位数
            percentile = np.mean(np.array(self.all_history_scores) <= current_score)

            # 数值稳定性保护
            if np.isnan(percentile) or np.isinf(percentile):
                percentile = 0.0

            return np.clip(percentile, 0.0, 1.0)

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
            historical_percentile: 历史百分位数

        Returns:
            综合置信度 [0, 1]
        """
        try:
            # 改善率置信度：改善越慢，置信度越高
            max_rate = 0.1  # 假设的最大改善率
            improvement_confidence = 1.0 - min(improvement_rate / max_rate, 1.0)

            # 稳定性置信度：直接使用稳定性分数
            stability_confidence = stability_score

            # 历史表现置信度：直接使用百分位数
            performance_confidence = historical_percentile

            # 加权平均（可以根据需要调整权重）
            weights = [0.4, 0.3, 0.3]  # [改善率, 稳定性, 历史表现]
            confidence = (
                weights[0] * improvement_confidence +
                weights[1] * stability_confidence +
                weights[2] * performance_confidence
            )

            # 数值稳定性保护
            if np.isnan(confidence) or np.isinf(confidence):
                confidence = 0.0

            return np.clip(confidence, 0.0, 1.0)

        except Exception:
            return 0.0



    def reset(self):
        """重置检测器状态"""
        self.recent_scores.clear()
        self.all_history_scores.clear()

    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.recent_scores:
            return {}

        return {
            'recent_mean': np.mean(self.recent_scores),
            'recent_std': np.std(self.recent_scores),
            'recent_min': np.min(self.recent_scores),
            'recent_max': np.max(self.recent_scores),
            'history_mean': np.mean(self.all_history_scores) if self.all_history_scores else 0.0,
            'history_std': np.std(self.all_history_scores) if self.all_history_scores else 0.0,
            'total_samples': len(self.all_history_scores),
            'window_samples': len(self.recent_scores)
        }


