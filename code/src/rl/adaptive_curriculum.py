"""
智能自适应课程学习系统

从"固定剧本"升级为"智能导演"模式的核心实现。
提供性能驱动的阶段转换、平滑参数演化和安全保障机制。

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


class PerformanceAnalyzer:
    """性能分析器 - 多维度性能指标分析"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_history = deque(maxlen=200)
        self.performance_history = deque(maxlen=200)
        
    def analyze_episode(self, episode_info: Dict[str, Any]) -> Dict[str, float]:
        """分析单个episode的性能"""
        self.episode_history.append(episode_info)
        
        performance = {
            'episode_length': episode_info.get('episode_length', 0),
            'reward': episode_info.get('reward', -float('inf')),
            'load_cv': episode_info.get('load_cv', 1.0),
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
    
    def __init__(self, config: ParameterEvolutionConfig):
        self.config = config
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
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.best_performance = -float('inf')
        self.deterioration_count = 0
        self.emergency_mode = False
        
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
            performance.get('episode_length', 0) < self.config.min_episode_length,
            performance.get('reward', -float('inf')) < self.config.max_reward_threshold,
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


class AdaptiveCurriculumDirector:
    """智能自适应课程学习导演 - 核心控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化子组件
        curriculum_config = config.get('adaptive_curriculum', {})

        self.transition_criteria = StageTransitionCriteria(**curriculum_config.get('stage_transition', {}))
        self.param_evolution_config = ParameterEvolutionConfig(**curriculum_config.get('parameter_evolution', {}))

        # 优先使用模式特定的安全监控配置
        safety_monitoring_config = curriculum_config.get('safety_monitoring', {})
        if not safety_monitoring_config:
            # 如果没有模式特定配置，使用全局配置
            safety_monitoring_config = config.get('adaptive_curriculum', {}).get('safety_monitoring', {})

        self.safety_config = SafetyConfig(**safety_monitoring_config)
        
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.parameter_scheduler = ParameterScheduler(self.param_evolution_config)
        self.safety_monitor = SafetyMonitor(self.safety_config)
        
        # 状态跟踪
        self.episode_count = 0
        self.stage_transition_history = []
        self.last_stage_transition_episode = -1
        self.emergency_count = 0
        self.normal_stage_count = 0

        # 简化日志输出
        self.verbose = config.get('debug', {}).get('adaptive_curriculum_verbose', False)

        logger.info("智能自适应课程学习导演已初始化")
    
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
                'stage_name': current_params['stage_name']
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
                'stage_name': 'emergency_recovery'
            },
            'safety_status': safety_status,
            'episode': self.episode_count
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取当前状态摘要"""
        return {
            'current_stage': self.parameter_scheduler.current_stage,
            'stage_progress': self.parameter_scheduler.stage_progress,
            'episode_count': self.episode_count,
            'transition_history': self.stage_transition_history,
            'emergency_mode': self.safety_monitor.emergency_mode,
            'best_performance': self.safety_monitor.best_performance
        }
