"""
场景感知奖励系统 - 相对改进奖励计算器

实现基于相对改进率的奖励计算，解决奖励量纲不统一问题。

作者：Augment Agent
日期：2025-01-15
"""

import numpy as np
from typing import Dict, Any, Optional
from .scenario_context import ScenarioContext
from .scenario_aware_tracker import ScenarioAwareHistoryTracker


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
    混合奖励函数
    
    支持在训练过程中从绝对奖励切换到相对奖励
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
        
        # 相对奖励计算器
        self.relative_calculator = RelativeImprovementReward(
            history_tracker, 
            self.config.get('relative_reward', {})
        )
    
    def compute_reward(self, 
                      prev_score: float, 
                      curr_score: float, 
                      scenario_context: ScenarioContext,
                      episode_num: Optional[int] = None) -> float:
        """
        计算混合奖励
        
        Args:
            prev_score: 前一步质量分数
            curr_score: 当前质量分数
            scenario_context: 场景上下文
            episode_num: 当前回合数
            
        Returns:
            计算的奖励
        """
        # 更新当前回合数
        if episode_num is not None:
            self.current_episode = episode_num
        
        # 决定使用哪种奖励计算方式
        should_use_relative = (
            self.use_relative and 
            self.current_episode > self.switch_episode
        )
        
        if should_use_relative:
            return self.relative_calculator.compute_relative_reward(
                prev_score, curr_score, scenario_context
            )
        else:
            # 使用原始势函数奖励
            return 0.99 * curr_score - prev_score
    
    def update_episode(self, episode_num: int):
        """更新当前回合数"""
        self.current_episode = episode_num
    
    def enable_relative_reward(self, enable: bool = True):
        """启用或禁用相对奖励"""
        self.use_relative = enable
    
    def get_current_mode(self) -> str:
        """获取当前奖励模式"""
        if self.use_relative and self.current_episode > self.switch_episode:
            return 'relative'
        else:
            return 'absolute' 