"""
场景感知奖励系统 - 相对改进奖励计算器

实现基于相对改进率的奖励计算，解决奖励量纲不统一问题。

作者：Augment Agent
日期：2025-01-15
"""

import numpy as np
import torch
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
    混合奖励函数 - 性能优化版
    
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
        
        # 性能优化：预计算常量
        self.clip_min, self.clip_max = self.config.get('clip_range', [-1.0, 1.0])
        
        # 统计信息
        self.total_computations = 0
        self.exploration_rewards_given = 0
        self.baseline_source_stats = defaultdict(int)

    def compute_reward(self, 
                      current_quality: float, 
                      previous_quality: Optional[float],
                      scenario_context: ScenarioContext) -> Tuple[float, Dict[str, Any]]:
        """
        计算混合奖励 - 性能优化版
        
        Args:
            current_quality: 当前质量分数
            previous_quality: 前一步质量分数
            scenario_context: 场景上下文
            
        Returns:
            Tuple[float, Dict]: (奖励值, 详细信息)
        """
        self.total_computations += 1
        
        # 获取两级查询基线（已GPU优化）
        baseline_quality, baseline_source, baseline_details = \
            self.history_tracker.get_scenario_baseline(scenario_context)
        
        # 统计基线来源
        self.baseline_source_stats[baseline_source] += 1
        
        # 计算基础奖励和相对改进奖励
        base_reward = current_quality - baseline_quality
        
        if previous_quality is not None:
            # 简化的相对改进计算
            if previous_quality > 1e-8:
                relative_reward = (current_quality - previous_quality) / previous_quality
            else:
                relative_reward = current_quality - previous_quality
            # 快速裁剪
            relative_reward = max(self.clip_min, min(self.clip_max, relative_reward))
        else:
            relative_reward = 0.0
        
        # 简化的动态权重计算
        if self.use_dynamic_weights:
            query_level = baseline_details.get('query_level', 0)
            if query_level == 1:  # 精确场景匹配
                base_weight = self.base_reward_weight + self.weight_adjustment_factor
            elif query_level == 2:  # 类别平均
                base_weight = self.base_reward_weight
            elif query_level == 3:  # 全局回退
                base_weight = self.base_reward_weight - self.weight_adjustment_factor
            else:  # 无数据
                base_weight = 0.1
            # 快速裁剪
            base_weight = max(0.1, min(0.9, base_weight))
            relative_weight = 1.0 - base_weight
        else:
            base_weight = self.base_reward_weight
            relative_weight = 1.0 - self.base_reward_weight
        
        # 计算主要奖励
        main_reward = base_weight * base_reward + relative_weight * relative_reward
        
        # 改进奖励加成
        improvement_bonus = self.improvement_bonus if (previous_quality is not None and current_quality > previous_quality) else 0.0
        
        # 简化的探索奖励
        exploration_reward = 0.0
        exploration_info = {'enabled': self.exploration_enabled}
        
        if self.exploration_enabled:
            exploration_reward, exploration_info = self._calculate_exploration_reward_optimized(
                scenario_context, current_quality
            )
        
        # 总奖励
        total_reward = main_reward + improvement_bonus + exploration_reward
        
        # 简化的详细信息
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
                'query_level': baseline_details.get('query_level', 0)
            },
            'weight_info': {
                'base_weight': float(base_weight),
                'relative_weight': float(relative_weight)
            },
            'exploration_info': exploration_info
        }
        
        return total_reward, reward_details

    def _calculate_exploration_reward_optimized(self, 
                                              scenario_context: ScenarioContext,
                                              current_quality: float) -> Tuple[float, Dict[str, Any]]:
        """
        计算探索奖励 - 性能优化版
        
        Args:
            scenario_context: 场景上下文
            current_quality: 当前质量分数
            
        Returns:
            Tuple[float, Dict]: (探索奖励, 探索信息)
        """
        if not self.exploration_enabled:
            return 0.0, {'enabled': False}
        
        # 快速检查是否为新场景
        is_new_scenario, new_scenario_type = self.history_tracker.is_new_scenario(scenario_context)
        
        if not is_new_scenario:
            return 0.0, {
                'enabled': True,
                'is_new_scenario': False,
                'scenario_type': new_scenario_type
            }
        
        # 简化的探索奖励计算
        base_exploration_reward = self.exploration_base_bonus
        
        # 基于场景类型的简单调整
        if new_scenario_type == 'new_category':
            priority_multiplier = 1.5
        elif new_scenario_type == 'new_exact_scenario':
            priority_multiplier = 1.2
        else:  # insufficient_data_scenario
            priority_multiplier = 1.0
        
        priority_adjusted_reward = base_exploration_reward * priority_multiplier
        
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
            'new_scenario_type': new_scenario_type,
            'priority_multiplier': float(priority_multiplier),
            'decay_factor': float(decay_factor),
            'quality_multiplier': float(quality_multiplier),
            'final_reward': float(final_exploration_reward)
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
        
    def get_exploration_context(self, scenario_context: ScenarioContext) -> Dict[str, Any]:
        """
        获取完整的探索上下文信息（保持接口一致性）
        
        Args:
            scenario_context: 场景上下文
            
        Returns:
            探索上下文字典
        """
        is_new_scenario, new_scenario_type = self.history_tracker.is_new_scenario(scenario_context)
        
        if not is_new_scenario:
            return {
                'is_new_scenario': False,
                'new_scenario_type': 'known_scenario',
                'exploration_priority': 0.0,
                'data_sparsity': {
                    'exact_scenario_data_count': len(
                        self.history_tracker.histories.get(
                            self.history_tracker.classifier.classify(scenario_context), []
                        )
                    ),
                    'category_data_count': 0,
                    'data_sufficiency': 'sufficient'
                }
            }
        
        # 计算探索优先级
        scenario_key = self.history_tracker.classifier.classify(scenario_context)
        scenario_category = self.history_tracker.classifier.extract_category(scenario_key)
        
        exact_data_count = len(self.history_tracker.histories.get(scenario_key, []))
        
        category_scenarios = self.history_tracker.classifier.list_scenarios_in_category(scenario_category)
        category_data_count = sum(
            len(self.history_tracker.histories.get(cat_scenario, [])) 
            for cat_scenario in category_scenarios
        )
        
        # 计算探索优先级
        if new_scenario_type == 'new_category':
            exploration_priority = 1.0
        elif new_scenario_type == 'new_exact_scenario':
            # 基于类别数据的探索优先级
            if category_data_count < self.min_data_threshold:
                exploration_priority = 0.8
            else:
                exploration_priority = 0.5
        else:  # insufficient_data_scenario
            # 基于数据稀疏性的探索优先级
            exploration_priority = max(0.2, 1.0 - (exact_data_count / self.min_data_threshold))
        
        return {
            'is_new_scenario': True,
            'new_scenario_type': new_scenario_type,
            'scenario_category': scenario_category,
            'exploration_priority': float(exploration_priority),
            'data_sparsity': {
                'exact_scenario_data_count': exact_data_count,
                'category_data_count': category_data_count,
                'data_sufficiency': 'insufficient' if exact_data_count < self.min_data_threshold else 'sufficient',
                'sparsity_level': 'high' if exact_data_count == 0 else 'medium' if exact_data_count < self.min_data_threshold else 'low'
            }
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        动态更新配置（保持接口一致性）
        
        Args:
            new_config: 新的配置字典
        """
        # 更新基础配置
        if 'base_reward_weight' in new_config:
            self.base_reward_weight = new_config['base_reward_weight']
        
        if 'improvement_bonus' in new_config:
            self.improvement_bonus = new_config['improvement_bonus']
            
        # 更新探索奖励配置
        exploration_config = new_config.get('exploration_reward', {})
        if exploration_config:
            self.exploration_enabled = exploration_config.get('enabled', self.exploration_enabled)
            self.exploration_base_bonus = exploration_config.get('base_bonus', self.exploration_base_bonus)
            self.exploration_priority_multiplier = exploration_config.get('priority_multiplier', self.exploration_priority_multiplier)
            self.exploration_decay_episodes = exploration_config.get('decay_episodes', self.exploration_decay_episodes)
        
        # 更新动态权重配置
        dynamic_config = new_config.get('dynamic_weighting', {})
        if dynamic_config:
            self.use_dynamic_weights = dynamic_config.get('enabled', self.use_dynamic_weights)
            self.min_data_threshold = dynamic_config.get('min_data_threshold', self.min_data_threshold)
            self.weight_adjustment_factor = dynamic_config.get('adjustment_factor', self.weight_adjustment_factor)
        
        # 更新裁剪范围
        if 'clip_range' in new_config:
            self.clip_min, self.clip_max = new_config['clip_range']
        
        # 更新相对奖励计算器配置
        relative_config = new_config.get('relative_reward', {})
        if relative_config and self.relative_calculator:
            self.relative_calculator.config.update(relative_config)
            
            # 更新相关属性
            self.relative_calculator.method = relative_config.get('method', self.relative_calculator.method)
            self.relative_calculator.use_difficulty_scaling = relative_config.get('use_difficulty_scaling', self.relative_calculator.use_difficulty_scaling)
            self.relative_calculator.clip_range = relative_config.get('clip_range', self.relative_calculator.clip_range)
            
        print(f"✅ HybridRewardFunction配置已更新")
    
    def get_reward_breakdown(self, 
                           current_quality: float, 
                           previous_quality: Optional[float],
                           scenario_context: ScenarioContext) -> Dict[str, Any]:
        """
        获取详细的奖励分解（用于调试和分析）
        
        Args:
            current_quality: 当前质量分数
            previous_quality: 前一步质量分数
            scenario_context: 场景上下文
            
        Returns:
            详细的奖励分解字典
        """
        # 计算完整奖励
        total_reward, reward_details = self.compute_reward(
            current_quality, previous_quality, scenario_context
        )
        
        # 添加额外的分析信息
        breakdown = {
            'reward_computation': reward_details,
            'scenario_analysis': {
                'scenario_key': self.history_tracker.classifier.classify(scenario_context),
                'scenario_category': self.history_tracker.classifier.extract_category(
                    self.history_tracker.classifier.classify(scenario_context)
                ),
                'scenario_statistics': self.history_tracker.get_scenario_statistics(scenario_context)
            },
            'comparison_analysis': {
                'quality_improvement': current_quality - (previous_quality or 0.0),
                'quality_improvement_percentage': (
                    ((current_quality - previous_quality) / previous_quality * 100) 
                    if previous_quality and previous_quality > 1e-8 else 0.0
                ),
                'baseline_comparison': current_quality - reward_details['baseline_info']['baseline_quality'],
                'quality_ranking': self.history_tracker.compute_scenario_percentile(
                    current_quality, scenario_context
                ) * 100  # 转换为百分比
            },
            'system_status': {
                'exploration_active': reward_details['exploration_info'].get('is_new_scenario', False),
                'data_availability': reward_details['baseline_info']['baseline_source'],
                'reward_mode': 'exploration' if reward_details['exploration_reward'] > 0 else 'exploitation'
            }
        }
        
        return breakdown 