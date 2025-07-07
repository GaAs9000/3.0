"""
场景感知奖励系统 - 相对改进奖励计算器

实现基于相对改进率的奖励计算，解决奖励量纲不统一问题。

作者：Augment Agent
日期：2025-01-15
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from .scenario_context import ScenarioContext
from .scenario_aware_tracker import ScenarioAwareHistoryTracker
from collections import defaultdict


class RelativeImprovementReward:
    """
    相对改进奖励计算器
    
    基于相对改进率计算奖励，确保相同的相对努力获得相同幅度的奖励
    """
    
    def __init__(self, 
                 history_tracker: ScenarioAwareHistoryTracker,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化相对改进奖励计算器
        
        Args:
            history_tracker: 场景感知历史追踪器
            config: 配置字典
        """
        self.history_tracker = history_tracker
        self.config = config or {}
        
        # 配置参数
        self.method = self.config.get('method', 'relative_improvement')
        self.use_difficulty_scaling = self.config.get('use_difficulty_scaling', True)
        self.clip_range = self.config.get('clip_range', [-1.0, 1.0])
        self.epsilon = self.config.get('epsilon', 1e-8)
        
        # PopArt标准化器（如果使用）
        if self.method == 'popart':
            self.popart_normalizer = PopArtNormalizer(self.config.get('popart', {}))
        else:
            self.popart_normalizer = None
    
    def compute_relative_reward(self, 
                              prev_score: float, 
                              curr_score: float, 
                              scenario_context: ScenarioContext) -> float:
        """
        计算相对改进奖励
        
        Args:
            prev_score: 前一步的质量分数
            curr_score: 当前步的质量分数
            scenario_context: 场景上下文
            
        Returns:
            相对改进奖励
        """
        if self.method == 'relative_improvement':
            return self._compute_relative_improvement_reward(prev_score, curr_score, scenario_context)
        elif self.method == 'range_normalized':
            return self._compute_range_normalized_reward(prev_score, curr_score, scenario_context)
        elif self.method == 'popart':
            return self._compute_popart_normalized_reward(prev_score, curr_score, scenario_context)
        else:
            # 默认使用原始势函数奖励
            return 0.99 * curr_score - prev_score
    
    def _compute_relative_improvement_reward(self, 
                                           prev_score: float, 
                                           curr_score: float, 
                                           scenario_context: ScenarioContext) -> float:
        """
        计算基于相对改进率的奖励
        
        Args:
            prev_score: 前一步质量分数
            curr_score: 当前质量分数
            scenario_context: 场景上下文
            
        Returns:
            相对改进奖励
        """
        # 1. 计算相对改进率
        if prev_score > self.epsilon:
            relative_improvement = (curr_score - prev_score) / prev_score
        else:
            # 处理边界情况：从零或接近零开始
            relative_improvement = curr_score - prev_score
        
        # 2. 可选：基于场景难度的缩放
        if self.use_difficulty_scaling:
            difficulty_factor = self.history_tracker.estimate_scenario_difficulty(scenario_context)
            scaled_reward = relative_improvement * difficulty_factor
        else:
            scaled_reward = relative_improvement
        
        # 3. 数值保护
        return np.clip(scaled_reward, self.clip_range[0], self.clip_range[1])
    
    def _compute_range_normalized_reward(self, 
                                       prev_score: float, 
                                       curr_score: float, 
                                       scenario_context: ScenarioContext) -> float:
        """
        计算基于场景范围归一化的奖励
        
        Args:
            prev_score: 前一步质量分数
            curr_score: 当前质量分数
            scenario_context: 场景上下文
            
        Returns:
            范围归一化奖励
        """
        # 1. 获取场景统计信息
        scenario_stats = self.history_tracker.get_scenario_statistics(scenario_context)
        
        # 2. 获取场景质量范围
        q_min = scenario_stats.get('min', 0.0)
        q_max = scenario_stats.get('max', 1.0)
        range_span = max(q_max - q_min, 0.1)  # 避免除零
        
        # 3. 范围归一化的改进
        delta_q = 0.99 * curr_score - prev_score
        normalized_reward = delta_q / range_span
        
        # 4. 可选：难度缩放
        if self.use_difficulty_scaling:
            difficulty_factor = self.history_tracker.estimate_scenario_difficulty(scenario_context)
            normalized_reward *= difficulty_factor
        
        return np.clip(normalized_reward, self.clip_range[0], self.clip_range[1])
    
    def _compute_popart_normalized_reward(self, 
                                        prev_score: float, 
                                        curr_score: float, 
                                        scenario_context: ScenarioContext) -> float:
        """
        计算PopArt标准化奖励
        
        Args:
            prev_score: 前一步质量分数
            curr_score: 当前质量分数
            scenario_context: 场景上下文
            
        Returns:
            PopArt标准化奖励
        """
        # 1. 计算原始奖励
        raw_reward = 0.99 * curr_score - prev_score
        
        # 2. PopArt标准化
        if self.popart_normalizer:
            normalized_reward = self.popart_normalizer.normalize_reward(raw_reward)
        else:
            normalized_reward = raw_reward
        
        return np.clip(normalized_reward, self.clip_range[0], self.clip_range[1])


class PopArtNormalizer:
    """
    PopArt自适应标准化器
    
    实现自适应标准化reward分布
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化PopArt标准化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 统计量
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        
        # 动量系数
        self.beta = self.config.get('beta', 0.99)
        self.epsilon = self.config.get('epsilon', 1e-8)
        
        # 最小更新次数
        self.min_updates = self.config.get('min_updates', 10)
    
    def normalize_reward(self, reward: float) -> float:
        """
        标准化奖励
        
        Args:
            reward: 原始奖励
            
        Returns:
            标准化后的奖励
        """
        # 1. 更新统计量
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var = self.beta * self.var + (1 - self.beta) * delta * delta
        
        # 2. 标准化（当有足够数据时）
        if self.count >= self.min_updates:
            std = np.sqrt(self.var + self.epsilon)
            normalized_reward = (reward - self.mean) / std
        else:
            # 数据不足时返回原始奖励
            normalized_reward = reward
        
        return normalized_reward
    
    def denormalize_value(self, normalized_value: float) -> float:
        """
        反标准化值（用于critic网络输出）
        
        Args:
            normalized_value: 标准化的值
            
        Returns:
            反标准化的值
        """
        if self.count >= self.min_updates:
            std = np.sqrt(self.var + self.epsilon)
            return normalized_value * std + self.mean
        else:
            return normalized_value
    
    def get_statistics(self) -> Dict[str, float]:
        """获取标准化统计信息"""
        return {
            'mean': self.mean,
            'var': self.var,
            'std': np.sqrt(self.var + self.epsilon),
            'count': self.count
        }


class HybridRewardFunction:
    """
    混合奖励函数 - 升级版
    
    支持：
    1. 两级查询基线系统（精确场景→类别平均→全局回退）
    2. 探索奖励机制
    3. 动态权重调整
    4. 在训练过程中从绝对奖励切换到相对奖励
    """
    
    def __init__(self, 
                 history_tracker: ScenarioAwareHistoryTracker,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化混合奖励函数
        
        Args:
            history_tracker: 场景感知历史追踪器
            config: 配置字典
        """
        self.history_tracker = history_tracker
        self.config = config or {}
        
        # 切换参数
        self.use_relative = self.config.get('use_relative', False)
        self.switch_episode = self.config.get('switch_episode', 500)
        self.current_episode = 0
        
        # 基础奖励权重配置
        self.base_reward_weight = self.config.get('base_reward_weight', 0.5)
        self.improvement_bonus = self.config.get('improvement_bonus', 0.1)
        
        # 探索奖励配置
        self.exploration_config = self.config.get('exploration_reward', {})
        self.exploration_enabled = self.exploration_config.get('enabled', True)
        self.exploration_base_bonus = self.exploration_config.get('base_bonus', 0.05)
        self.exploration_priority_multiplier = self.exploration_config.get('priority_multiplier', 2.0)
        self.exploration_decay_episodes = self.exploration_config.get('decay_episodes', 100)
        
        # 动态权重调整配置
        self.dynamic_weighting = self.config.get('dynamic_weighting', {})
        self.use_dynamic_weights = self.dynamic_weighting.get('enabled', True)
        self.min_data_threshold = self.dynamic_weighting.get('min_data_threshold', 5)
        self.weight_adjustment_factor = self.dynamic_weighting.get('adjustment_factor', 0.3)
        
        # 相对奖励计算器
        self.relative_calculator = RelativeImprovementReward(
            history_tracker, 
            self.config.get('relative_reward', {})
        )
        
        # 统计信息
        self.total_computations = 0
        self.exploration_rewards_given = 0
        self.baseline_source_stats = defaultdict(int)

    def compute_reward(self, 
                      current_quality: float, 
                      previous_quality: Optional[float],
                      scenario_context: ScenarioContext) -> Tuple[float, Dict[str, Any]]:
        """
        计算混合奖励 - 升级版
        
        Args:
            current_quality: 当前质量分数
            previous_quality: 前一步质量分数
            scenario_context: 场景上下文
            
        Returns:
            Tuple[float, Dict]: (奖励值, 详细信息)
        """
        self.total_computations += 1
        
        # 获取两级查询基线
        baseline_quality, baseline_source, baseline_details = \
            self.history_tracker.get_scenario_baseline(scenario_context)
        
        # 统计基线来源
        self.baseline_source_stats[baseline_source] += 1
        
        # 计算基础奖励和相对改进奖励
        base_reward = current_quality - baseline_quality
        
        if previous_quality is not None:
            relative_reward = self.relative_calculator.compute_relative_reward(
                previous_quality, current_quality, scenario_context
            )
        else:
            relative_reward = 0.0
        
        # 动态权重调整
        if self.use_dynamic_weights:
            adjusted_weights = self._calculate_dynamic_weights(baseline_details)
        else:
            adjusted_weights = {
                'base_weight': self.base_reward_weight,
                'relative_weight': 1.0 - self.base_reward_weight
            }
        
        # 计算主要奖励
        main_reward = (
            adjusted_weights['base_weight'] * base_reward + 
            adjusted_weights['relative_weight'] * relative_reward
        )
        
        # 改进奖励加成
        improvement_bonus = 0.0
        if previous_quality is not None and current_quality > previous_quality:
            improvement_bonus = self.improvement_bonus
        
        # 探索奖励
        exploration_reward = 0.0
        exploration_info = {}
        
        if self.exploration_enabled:
            exploration_reward, exploration_info = self._calculate_exploration_reward(
                scenario_context, current_quality
            )
        
        # 总奖励
        total_reward = main_reward + improvement_bonus + exploration_reward
        
        # 详细信息
        reward_details = {
            'main_reward': float(main_reward),
            'base_reward': float(base_reward),
            'relative_reward': float(relative_reward),
            'improvement_bonus': float(improvement_bonus),
            'exploration_reward': float(exploration_reward),
            'total_reward': float(total_reward),
            'baseline_info': {
                'baseline_quality': float(baseline_quality),
                'baseline_source': baseline_source,
                'query_level': baseline_details.get('query_level', 0),
                'category': baseline_details.get('scenario_category', 'unknown'),
                'details': baseline_details
            },
            'weight_info': adjusted_weights,
            'exploration_info': exploration_info,
            'computation_stats': {
                'total_computations': self.total_computations,
                'exploration_rewards_given': self.exploration_rewards_given,
                'baseline_source_distribution': dict(self.baseline_source_stats)
            }
        }
        
        return total_reward, reward_details

    def _calculate_dynamic_weights(self, baseline_details: Dict[str, Any]) -> Dict[str, float]:
        """
        根据数据充足性动态调整权重
        
        Args:
            baseline_details: 基线查询详细信息
            
        Returns:
            动态调整后的权重字典
        """
        query_level = baseline_details.get('query_level', 0)
        
        # 根据查询级别调整权重
        if query_level == 1:  # 精确场景匹配
            # 数据充足，更多依赖基础奖励
            base_weight = self.base_reward_weight + self.weight_adjustment_factor
        elif query_level == 2:  # 类别平均
            # 中等数据，使用配置权重
            base_weight = self.base_reward_weight
        elif query_level == 3:  # 全局回退
            # 数据稀疏，更多依赖相对改进
            base_weight = self.base_reward_weight - self.weight_adjustment_factor
        else:  # 无数据
            # 完全依赖相对改进
            base_weight = 0.1
        
        # 权重范围限制
        base_weight = np.clip(base_weight, 0.1, 0.9)
        relative_weight = 1.0 - base_weight
        
        return {
            'base_weight': float(base_weight),
            'relative_weight': float(relative_weight),
            'adjustment_reason': f'query_level_{query_level}'
        }

    def _calculate_exploration_reward(self, 
                                    scenario_context: ScenarioContext,
                                    current_quality: float) -> Tuple[float, Dict[str, Any]]:
        """
        计算探索奖励
        
        Args:
            scenario_context: 场景上下文
            current_quality: 当前质量分数
            
        Returns:
            Tuple[float, Dict]: (探索奖励, 探索信息)
        """
        if not self.exploration_enabled:
            return 0.0, {'enabled': False}
        
        # 获取探索上下文
        exploration_context = self.history_tracker.get_exploration_context(scenario_context)
        
        is_new_scenario = exploration_context['is_new_scenario']
        exploration_priority = exploration_context['exploration_priority']
        
        if not is_new_scenario:
            return 0.0, {
                'enabled': True,
                'is_new_scenario': False,
                'scenario_type': exploration_context['new_scenario_type']
            }
        
        # 计算基础探索奖励
        base_exploration_reward = self.exploration_base_bonus
        
        # 优先级调整
        priority_adjusted_reward = base_exploration_reward * (
            1.0 + exploration_priority * self.exploration_priority_multiplier
        )
        
        # 衰减调整（随着episode增加逐渐减少探索奖励）
        if self.exploration_decay_episodes > 0 and self.current_episode > 0:
            decay_factor = max(0.1, 1.0 - (self.current_episode / self.exploration_decay_episodes))
        else:
            decay_factor = 1.0
        
        final_exploration_reward = priority_adjusted_reward * decay_factor
        
        # 质量调整（质量越高，探索奖励越大）
        quality_multiplier = 0.5 + current_quality * 0.5  # 0.5-1.0的范围
        final_exploration_reward *= quality_multiplier
        
        self.exploration_rewards_given += 1
        
        exploration_info = {
            'enabled': True,
            'is_new_scenario': True,
            'new_scenario_type': exploration_context['new_scenario_type'],
            'scenario_category': exploration_context['scenario_category'],
            'exploration_priority': float(exploration_priority),
            'base_reward': float(base_exploration_reward),
            'priority_adjusted': float(priority_adjusted_reward),
            'decay_factor': float(decay_factor),
            'quality_multiplier': float(quality_multiplier),
            'final_reward': float(final_exploration_reward),
            'data_sparsity': exploration_context['data_sparsity']
        }
        
        return final_exploration_reward, exploration_info

    def set_episode(self, episode: int):
        """设置当前episode（用于衰减计算）"""
        self.current_episode = episode

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_computations': self.total_computations,
            'exploration_rewards_given': self.exploration_rewards_given,
            'exploration_rate': (
                self.exploration_rewards_given / max(1, self.total_computations)
            ),
            'baseline_source_distribution': dict(self.baseline_source_stats),
            'current_episode': self.current_episode,
            'config': {
                'exploration_enabled': self.exploration_enabled,
                'exploration_base_bonus': self.exploration_base_bonus,
                'base_reward_weight': self.base_reward_weight,
                'use_dynamic_weights': self.use_dynamic_weights
            }
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.total_computations = 0
        self.exploration_rewards_given = 0
        self.baseline_source_stats.clear() 