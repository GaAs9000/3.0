"""
智能自适应训练系统

从"固定剧本"升级为"智能导演"模式的核心实现。
提供性能驱动的阶段转换、平滑参数演化和安全保障机制。
支持与fast/full/ieee118等模式的智能集成。

作者：Augment Agent
日期：2025-06-30
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

# 导入平台期检测系统
try:
    from .plateau_detector import QualityPlateauDetector, PlateauResult
    from .scenario_aware_plateau_detector import ScenarioAwarePlateauDetector
    _plateau_detector_available = True
except ImportError:
    _plateau_detector_available = False
    logger.warning("平台期检测系统不可用，将使用基础转换机制")


@dataclass
class StageTransitionCriteria:
    """阶段转换判定标准"""
    episode_length_target: int = 15
    episode_length_window: int = 50
    episode_length_stability: float = 0.8
    
    coupling_improvement_threshold: float = 0.05
    coupling_trend_window: int = 100
    
    composite_score_target: float = 0.7
    composite_score_window: int = 100
    
    connectivity_rate_threshold: float = 0.9
    plateau_detection_enabled: bool = True


@dataclass
class ParameterEvolutionConfig:
    """参数演化配置"""
    connectivity_penalty_range: Tuple[float, float] = (0.1, 1.5)
    action_mask_relaxation_range: Tuple[float, float] = (0.0, 0.7)
    balance_weight_range: Tuple[float, float] = (0.6, 0.8)
    decoupling_weight_range: Tuple[float, float] = (0.2, 0.6)
    power_weight_range: Tuple[float, float] = (0.0, 0.3)
    learning_rate_decay_factor: float = 0.1


@dataclass
class SafetyConfig:
    """安全监控配置"""
    min_episode_length: int = 3
    max_reward_threshold: float = -100
    max_loss_threshold: float = 10
    performance_deterioration_patience: int = 50
    performance_deterioration_threshold: float = 0.8


@dataclass
class ModeAdaptationConfig:
    """模式适配配置"""
    mode: str = "fast"
    
    # 不同模式的转换条件调整
    fast_mode_adjustments: Dict[str, float] = None
    full_mode_adjustments: Dict[str, float] = None
    ieee118_mode_adjustments: Dict[str, float] = None
    
    def __post_init__(self):
        if self.fast_mode_adjustments is None:
            self.fast_mode_adjustments = {
                'episode_length_stability': 0.7,  # 更宽松
                'composite_score_target': 0.6,    # 更宽松
                'safety_threshold_factor': 1.5    # 更宽松
            }
        
        if self.full_mode_adjustments is None:
            self.full_mode_adjustments = {
                'episode_length_stability': 0.8,  # 标准
                'composite_score_target': 0.7,    # 标准
                'safety_threshold_factor': 1.0    # 标准
            }
        
        if self.ieee118_mode_adjustments is None:
            self.ieee118_mode_adjustments = {
                'episode_length_stability': 0.9,  # 更严格
                'composite_score_target': 0.8,    # 更严格
                'safety_threshold_factor': 0.8    # 更严格
            }


class PerformanceAnalyzer:
    """性能分析器 - 多维度性能指标分析"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_history = deque(maxlen=200)
        self.performance_history = deque(maxlen=200)
        
    def analyze_episode(self, episode_info: Dict[str, Any]) -> Dict[str, float]:
        """分析单个episode的性能"""
        self.episode_history.append(episode_info)
        
        # 【修复】适配新奖励系统的指标名称映射
        performance = {
            'episode_length': episode_info.get('episode_length', 0),
            'reward': episode_info.get('reward', -float('inf')),
            # 新系统使用'cv'而不是'load_cv'
            'load_cv': episode_info.get('cv', episode_info.get('load_cv', 1.0)),
            'coupling_ratio': episode_info.get('coupling_ratio', 1.0),
            'connectivity': episode_info.get('connectivity', 0.0),
            'success': episode_info.get('success', False)
        }
        
        # 计算复合评分
        performance['composite_score'] = self._compute_composite_score(performance)
        
        self.performance_history.append(performance)
        return performance
    
    def _compute_composite_score(self, performance: Dict[str, float]) -> float:
        """计算综合性能评分"""
        # 负载平衡分数 (CV越小越好)
        balance_score = max(0, 1 - performance['load_cv'])
        
        # 解耦分数 (coupling_ratio越小越好)
        decoupling_score = max(0, 1 - performance['coupling_ratio'])
        
        # 连通性分数
        connectivity_score = performance['connectivity']
        
        # 加权综合评分
        composite = (
            0.4 * balance_score +
            0.4 * decoupling_score +
            0.2 * connectivity_score
        )
        
        return composite
    
    def get_trend_analysis(self, window_size: int = 50) -> Dict[str, float]:
        """获取性能趋势分析"""
        if len(self.performance_history) < window_size:
            return {'trend': 0.0, 'stability': 0.0}
        
        recent_scores = [p['composite_score'] for p in list(self.performance_history)[-window_size:]]
        
        # 计算趋势 (线性回归斜率)
        x = np.arange(len(recent_scores))
        trend = np.polyfit(x, recent_scores, 1)[0]
        
        # 计算稳定性 (变异系数的倒数)
        stability = 1.0 / (1.0 + np.std(recent_scores))
        
        return {
            'trend': trend,
            'stability': stability,
            'mean_score': np.mean(recent_scores),
            'recent_improvement': recent_scores[-10:] if len(recent_scores) >= 10 else recent_scores
        }


class ParameterScheduler:
    """参数调度器 - 平滑参数演化"""
    
    def __init__(self, config: ParameterEvolutionConfig, mode_config: ModeAdaptationConfig):
        self.config = config
        self.mode_config = mode_config
        self.current_stage = 1
        self.stage_progress = 0.0
        self.stage_start_episode = 0
        
    def get_stage_parameters(self, stage: int, progress: float) -> Dict[str, Any]:
        """根据阶段和进度获取参数"""
        if stage == 1:
            return self._stage1_params(progress)
        elif stage == 2:
            return self._stage2_params(progress)
        elif stage == 3:
            return self._stage3_params(progress)
        else:  # stage >= 4
            return self._stage4_params(progress)
    
    def _stage1_params(self, progress: float) -> Dict[str, Any]:
        """阶段1：探索期 - 高度放松约束"""
        return {
            'connectivity_penalty': 0.05,
            'action_mask_relaxation': 0.7,
            'reward_weights': {
                'balance_weight': 0.8,
                'decoupling_weight': 0.2,
                'power_weight': 0.0
            },
            'learning_rate_factor': 1.0,
            'stage_name': 'exploration'
        }
    
    def _stage2_params(self, progress: float) -> Dict[str, Any]:
        """阶段2：过渡期 - 逐渐收紧约束"""
        # 平滑插值
        connectivity_penalty = 0.1 + 1.4 * progress
        action_mask_relaxation = 0.7 * (1 - progress)
        balance_weight = 0.8 - 0.2 * progress
        decoupling_weight = 0.2 + 0.4 * progress
        power_weight = 0.3 * progress
        
        return {
            'connectivity_penalty': connectivity_penalty,
            'action_mask_relaxation': action_mask_relaxation,
            'reward_weights': {
                'balance_weight': balance_weight,
                'decoupling_weight': decoupling_weight,
                'power_weight': power_weight
            },
            'learning_rate_factor': 1.0,
            'stage_name': 'transition'
        }
    
    def _stage3_params(self, progress: float) -> Dict[str, Any]:
        """阶段3：精炼期 - 严格约束"""
        return {
            'connectivity_penalty': 1.5,
            'action_mask_relaxation': 0.0,
            'reward_weights': {
                'balance_weight': 0.6,
                'decoupling_weight': 0.6,
                'power_weight': 0.3
            },
            'learning_rate_factor': 0.5,
            'stage_name': 'refinement'
        }
    
    def _stage4_params(self, progress: float) -> Dict[str, Any]:
        """阶段4：微调期 - 学习率退火"""
        lr_decay = self.config.learning_rate_decay_factor * (1 - progress * 0.5)
        
        return {
            'connectivity_penalty': 1.5,
            'action_mask_relaxation': 0.0,
            'reward_weights': {
                'balance_weight': 0.6,
                'decoupling_weight': 0.6,
                'power_weight': 0.3
            },
            'learning_rate_factor': lr_decay,
            'stage_name': 'fine_tuning'
        }
    
    def update_stage(self, new_stage: int, episode: int):
        """更新当前阶段"""
        if new_stage != self.current_stage:
            logger.info(f"阶段转换: {self.current_stage} -> {new_stage} (Episode {episode})")
            self.current_stage = new_stage
            self.stage_start_episode = episode
            self.stage_progress = 0.0
    
    def update_progress(self, episode: int, episodes_in_stage: int = 500):
        """更新阶段内进度"""
        episodes_since_start = episode - self.stage_start_episode
        self.stage_progress = min(1.0, episodes_since_start / episodes_in_stage)


class SafetyMonitor:
    """安全监控器 - 训练崩溃检测与恢复"""
    
    def __init__(self, config: SafetyConfig, mode_config: ModeAdaptationConfig):
        self.config = config
        self.mode_config = mode_config
        self.best_performance = -float('inf')
        self.deterioration_count = 0
        self.emergency_mode = False
        
        # 根据模式调整安全阈值
        self._adjust_safety_thresholds()
        
    def _adjust_safety_thresholds(self):
        """根据训练模式调整安全阈值"""
        mode = self.mode_config.mode
        
        if mode == "fast":
            adjustments = self.mode_config.fast_mode_adjustments
        elif mode == "ieee118":
            adjustments = self.mode_config.ieee118_mode_adjustments
        else:  # full or default
            adjustments = self.mode_config.full_mode_adjustments
        
        factor = adjustments.get('safety_threshold_factor', 1.0)
        
        # 调整安全阈值
        self.adjusted_min_episode_length = max(1, int(self.config.min_episode_length * factor))
        self.adjusted_max_reward_threshold = self.config.max_reward_threshold * factor
        self.adjusted_max_loss_threshold = self.config.max_loss_threshold * factor
    
    def check_training_safety(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """检查训练安全状态"""
        safety_status = {
            'is_safe': True,
            'emergency_mode': False,
            'warnings': [],
            'actions': []
        }
        
        # 检查训练崩溃信号
        collapse_signals = [
            performance.get('episode_length', 0) < self.adjusted_min_episode_length,
            performance.get('reward', -float('inf')) < self.adjusted_max_reward_threshold,
            # 可以添加更多崩溃检测信号
        ]
        
        if any(collapse_signals):
            safety_status['is_safe'] = False
            safety_status['emergency_mode'] = True
            safety_status['warnings'].append("检测到训练崩溃信号")
            safety_status['actions'].append("启动紧急恢复模式")
            self.emergency_mode = True
        
        # 检查性能回退
        current_score = performance.get('composite_score', 0)
        if current_score < self.best_performance * self.config.performance_deterioration_threshold:
            self.deterioration_count += 1
            if self.deterioration_count > self.config.performance_deterioration_patience:
                safety_status['warnings'].append("检测到持续性能回退")
                safety_status['actions'].append("建议回退到上一阶段")
        else:
            self.deterioration_count = 0
            self.best_performance = max(self.best_performance, current_score)
        
        return safety_status
    
    def get_emergency_params(self) -> Dict[str, Any]:
        """获取紧急恢复参数"""
        return {
            'connectivity_penalty': 0.01,
            'action_mask_relaxation': 0.9,
            'reward_weights': {
                'balance_weight': 1.0,
                'decoupling_weight': 0.1,
                'power_weight': 0.0
            },
            'learning_rate_factor': 0.1,
            'stage_name': 'emergency_recovery'
        }


class AdaptiveDirector:
    """智能自适应训练导演 - 核心控制器"""

    def __init__(self, config: Dict[str, Any], base_mode: str = "fast"):
        self.config = config
        self.base_mode = base_mode

        # 初始化子组件
        curriculum_config = config.get('adaptive_curriculum', {})

        # 模式适配配置
        self.mode_config = ModeAdaptationConfig(mode=base_mode)

        # 根据基础模式调整转换条件
        self.transition_criteria = self._create_mode_adapted_criteria(curriculum_config)
        self.param_evolution_config = ParameterEvolutionConfig(**curriculum_config.get('parameter_evolution', {}))

        # 优先使用模式特定的安全监控配置
        safety_monitoring_config = curriculum_config.get('safety_monitoring', {})
        if not safety_monitoring_config:
            # 如果没有模式特定配置，使用全局配置
            safety_monitoring_config = config.get('adaptive_curriculum', {}).get('safety_monitoring', {})

        self.safety_config = SafetyConfig(**safety_monitoring_config)

        self.performance_analyzer = PerformanceAnalyzer(config)
        self.parameter_scheduler = ParameterScheduler(self.param_evolution_config, self.mode_config)
        self.safety_monitor = SafetyMonitor(self.safety_config, self.mode_config)

        # 状态跟踪
        self.episode_count = 0
        self.stage_transition_history = []
        self.last_stage_transition_episode = -1
        self.emergency_count = 0
        self.normal_stage_count = 0

        # 简化日志输出
        self.verbose = config.get('debug', {}).get('adaptive_curriculum_verbose', False)

        # 初始化平台期检测系统
        self.plateau_detector = None
        if _plateau_detector_available and curriculum_config.get('plateau_detection', {}).get('enabled', False):
            try:
                # 从配置创建平台期检测器
                plateau_config = curriculum_config.get('plateau_detection', {})
                self.plateau_detector = QualityPlateauDetector(
                    window_size=plateau_config.get('stability_window', 40),
                    min_improvement_rate=plateau_config.get('min_improvement_rate', 0.03),
                    stability_threshold=plateau_config.get('stability_cv_threshold', 0.2),
                    min_percentile=plateau_config.get('fallback_performance_threshold', 0.75),
                    confidence_threshold=plateau_config.get('confidence_threshold', 0.75)
                )
                logger.info(f"平台期检测系统已启用")
            except Exception as e:
                logger.warning(f"平台期检测系统初始化失败: {e}")
                self.plateau_detector = None

        logger.info(f"智能自适应训练导演已初始化 (基础模式: {base_mode})")

    def _create_mode_adapted_criteria(self, curriculum_config: Dict[str, Any]) -> StageTransitionCriteria:
        """根据基础模式创建适配的转换条件"""
        base_criteria = curriculum_config.get('stage_transition', {})

        # 获取模式特定的调整
        if self.base_mode == "fast":
            adjustments = self.mode_config.fast_mode_adjustments
        elif self.base_mode == "ieee118":
            adjustments = self.mode_config.ieee118_mode_adjustments
        else:  # full or default
            adjustments = self.mode_config.full_mode_adjustments

        # 应用调整
        adapted_criteria = base_criteria.copy()
        adapted_criteria['episode_length_stability'] = adjustments.get(
            'episode_length_stability',
            base_criteria.get('episode_length_stability', 0.8)
        )
        adapted_criteria['composite_score_target'] = adjustments.get(
            'composite_score_target',
            base_criteria.get('composite_score_target', 0.7)
        )

        return StageTransitionCriteria(**adapted_criteria)

    def step(self, episode: int, episode_info: Dict[str, Any]) -> Dict[str, Any]:
        """每个episode后的调度决策"""
        self.episode_count = episode

        # 1. 性能分析
        performance = self.performance_analyzer.analyze_episode(episode_info)

        # 2. 安全检查
        safety_status = self.safety_monitor.check_training_safety(performance)

        # 3. 紧急模式处理
        if safety_status['emergency_mode']:
            return self._handle_emergency_mode(safety_status)

        # 4. 阶段转换判定
        self._check_stage_transition(performance)

        # 4.5. 平台期检测（如果启用）
        plateau_result = None
        if self.plateau_detector:
            # 计算质量分数用于平台期检测
            quality_score = self._compute_quality_score(performance)
            plateau_result = self.plateau_detector.update(quality_score)

            # 如果检测到平台期且置信度足够高，可以考虑阶段转换
            if plateau_result.plateau_detected and plateau_result.confidence > self.plateau_detector.confidence_threshold:
                if self.verbose:
                    logger.info(f"检测到平台期 (置信度: {plateau_result.confidence:.3f})")
                # 这里可以添加平台期处理逻辑，比如触发阶段转换

        # 5. 更新参数调度
        self.parameter_scheduler.update_progress(episode)
        current_params = self.parameter_scheduler.get_stage_parameters(
            self.parameter_scheduler.current_stage,
            self.parameter_scheduler.stage_progress
        )

        # 6. 返回完整调度结果
        return {
            **current_params,
            'stage_info': {
                'current_stage': self.parameter_scheduler.current_stage,
                'stage_progress': self.parameter_scheduler.stage_progress,
                'stage_name': current_params['stage_name'],
                'base_mode': self.base_mode
            },
            'performance_info': performance,
            'safety_status': safety_status,
            'episode': episode
        }

    def _check_stage_transition(self, performance: Dict[str, float]):
        """检查是否需要阶段转换"""
        current_stage = self.parameter_scheduler.current_stage

        if current_stage == 1:
            # 阶段1 -> 阶段2 的转换条件
            if self._check_stage1_to_2_transition(performance):
                self.parameter_scheduler.update_stage(2, self.episode_count)
                self.stage_transition_history.append({
                    'episode': self.episode_count,
                    'from_stage': 1,
                    'to_stage': 2,
                    'trigger': 'episode_length_stability'
                })

        elif current_stage == 2:
            # 阶段2 -> 阶段3 的转换条件
            if self._check_stage2_to_3_transition(performance):
                self.parameter_scheduler.update_stage(3, self.episode_count)
                self.stage_transition_history.append({
                    'episode': self.episode_count,
                    'from_stage': 2,
                    'to_stage': 3,
                    'trigger': 'composite_score_target'
                })

        elif current_stage == 3:
            # 阶段3 -> 阶段4 的转换条件
            if self._check_stage3_to_4_transition(performance):
                self.parameter_scheduler.update_stage(4, self.episode_count)
                self.stage_transition_history.append({
                    'episode': self.episode_count,
                    'from_stage': 3,
                    'to_stage': 4,
                    'trigger': 'refinement_complete'
                })

    def _check_stage1_to_2_transition(self, performance: Dict[str, float]) -> bool:
        """检查阶段1到阶段2的转换条件"""
        if len(self.performance_analyzer.episode_history) < self.transition_criteria.episode_length_window:
            return False

        # 检查episode长度稳定性
        recent_lengths = [
            ep.get('episode_length', 0)
            for ep in list(self.performance_analyzer.episode_history)[-self.transition_criteria.episode_length_window:]
        ]

        target_achieved_count = sum(1 for length in recent_lengths if length >= self.transition_criteria.episode_length_target)
        stability_rate = target_achieved_count / len(recent_lengths)

        return stability_rate >= self.transition_criteria.episode_length_stability

    def _check_stage2_to_3_transition(self, performance: Dict[str, float]) -> bool:
        """检查阶段2到阶段3的转换条件"""
        if len(self.performance_analyzer.performance_history) < self.transition_criteria.composite_score_window:
            return False

        # 检查复合评分目标
        recent_scores = [
            p['composite_score']
            for p in list(self.performance_analyzer.performance_history)[-self.transition_criteria.composite_score_window:]
        ]

        mean_score = np.mean(recent_scores)
        return mean_score >= self.transition_criteria.composite_score_target

    def _check_stage3_to_4_transition(self, performance: Dict[str, float]) -> bool:
        """检查阶段3到阶段4的转换条件"""
        # 简单条件：在阶段3停留足够长时间且性能稳定
        episodes_in_stage3 = self.episode_count - self.parameter_scheduler.stage_start_episode

        if episodes_in_stage3 >= 300:  # 在阶段3至少300个episodes
            trend_analysis = self.performance_analyzer.get_trend_analysis(50)
            return trend_analysis['stability'] > 0.8  # 性能稳定

        return False

    def _handle_emergency_mode(self, safety_status: Dict[str, Any]) -> Dict[str, Any]:
        """处理紧急模式"""
        emergency_params = self.safety_monitor.get_emergency_params()

        self.emergency_count += 1

        # 只在verbose模式或首次紧急恢复时显示详细信息
        if self.verbose or self.emergency_count == 1:
            logger.warning(f"进入紧急恢复模式: {safety_status['warnings']}")
        elif self.emergency_count % 10 == 0:  # 每10次显示一次统计
            logger.warning(f"紧急恢复模式 (第{self.emergency_count}次)")

        return {
            **emergency_params,
            'stage_info': {
                'current_stage': 0,  # 特殊阶段标识
                'stage_progress': 0.0,
                'stage_name': 'emergency_recovery',
                'base_mode': self.base_mode
            },
            'safety_status': safety_status,
            'episode': self.episode_count
        }

    def get_status_summary(self) -> Dict[str, Any]:
        """获取当前状态摘要"""
        return {
            'base_mode': self.base_mode,
            'current_stage': self.parameter_scheduler.current_stage,
            'stage_progress': self.parameter_scheduler.stage_progress,
            'episode_count': self.episode_count,
            'transition_history': self.stage_transition_history,
            'emergency_mode': self.safety_monitor.emergency_mode,
            'best_performance': self.safety_monitor.best_performance,
            'emergency_count': self.emergency_count,
            'normal_stage_count': self.normal_stage_count
        }

    def _compute_quality_score(self, performance: Dict[str, Any]) -> float:
        """
        计算质量分数用于平台期检测

        Args:
            performance: 性能分析结果

        Returns:
            质量分数 [0, 1]，越大越好
        """
        try:
            # 获取关键指标
            cv = performance.get('cv', 1.0)  # 负载平衡系数，越小越好
            coupling_ratio = performance.get('coupling_ratio', 1.0)  # 耦合比，越小越好
            connectivity_rate = performance.get('connectivity_rate', 0.0)  # 连通性，越大越好
            episode_length = performance.get('episode_length', 1)  # episode长度，适中为好

            # 归一化处理
            # CV分数：使用指数函数，CV越小分数越高
            cv_score = np.exp(-2.0 * max(0, cv))

            # 耦合分数：使用指数函数，耦合比越小分数越高
            coupling_score = np.exp(-3.0 * max(0, coupling_ratio))

            # 连通性分数：直接使用连通率
            connectivity_score = max(0, min(1, connectivity_rate))

            # episode长度分数：适中长度得分最高
            target_length = 10  # 目标长度
            length_score = np.exp(-0.1 * abs(episode_length - target_length) / target_length)

            # 加权平均计算综合质量分数
            weights = [0.3, 0.3, 0.3, 0.1]  # [CV, 耦合, 连通性, 长度]
            quality_score = (
                weights[0] * cv_score +
                weights[1] * coupling_score +
                weights[2] * connectivity_score +
                weights[3] * length_score
            )

            # 确保结果在 [0, 1] 范围内
            quality_score = np.clip(quality_score, 0.0, 1.0)

            return quality_score

        except Exception as e:
            logger.warning(f"计算质量分数时出错: {e}")
            return 0.0
