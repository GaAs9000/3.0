"""
场景感知奖励系统 - 场景感知历史追踪器

实现按场景分类的历史数据存储和管理，支持场景内百分位计算。

作者：Augment Agent
日期：2025-01-15
"""

import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
from .scenario_context import ScenarioContext, ScenarioClassifier


class ScenarioAwareHistoryTracker:
    """
    场景感知历史追踪器
    
    按场景分类存储和管理历史质量分数，支持场景内百分位计算
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化历史追踪器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 配置参数
        self.window_size = self.config.get('window_size', 15)
        self.min_samples_for_percentile = self.config.get('min_samples_for_percentile', 5)
        self.max_history_per_scenario = self.config.get('max_history_per_scenario', 1000)
        self.min_samples_threshold = self.config.get('min_samples_threshold', 3)
        
        # 核心数据结构
        self.histories = defaultdict(list)  # {scenario_key: [scores]}
        self.recent_window = deque(maxlen=self.window_size)  # 最近的全局窗口
        self.scenario_stats = defaultdict(dict)  # 场景统计信息
        
        # 场景分类器
        classifier_config = self.config.get('scenario_classification', {})
        self.classifier = ScenarioClassifier(classifier_config)
        
        # 统计信息
        self.total_updates = 0
        self.last_update_time = time.time()
        
        # EMA参数
        self.ema_alpha = self.config.get('ema_alpha', 0.1)
    
    def update_history(self, score: float, scenario_context: ScenarioContext):
        """
        更新历史数据
        
        Args:
            score: 质量分数
            scenario_context: 场景上下文
        """
        # 输入验证
        if np.isnan(score) or np.isinf(score):
            score = 0.0
        score = np.clip(score, 0.0, 1.0)
        
        # 1. 场景分类
        scenario_key = self.classifier.classify(scenario_context)
        
        # 2. 更新场景历史
        self.histories[scenario_key].append(score)
        
        # 3. 维护历史长度限制
        if len(self.histories[scenario_key]) > self.max_history_per_scenario:
            # 删除最旧的20%数据
            remove_count = int(self.max_history_per_scenario * 0.2)
            del self.histories[scenario_key][:remove_count]
        
        # 4. 更新最近窗口（用于改善率计算）
        self.recent_window.append(score)
        
        # 5. 更新场景统计
        self._update_scenario_stats(scenario_key, score)
        
        # 6. 更新全局统计
        self.total_updates += 1
        self.last_update_time = time.time()
    
    def compute_scenario_percentile(self, score: float, scenario_context: ScenarioContext) -> float:
        """
        计算场景内百分位
        
        Args:
            score: 当前质量分数
            scenario_context: 场景上下文
            
        Returns:
            场景内百分位 [0, 1]
        """
        scenario_key = self.classifier.classify(scenario_context)
        scenario_scores = self.histories[scenario_key]
        
        # 数据不足时返回中位数
        if len(scenario_scores) < self.min_samples_threshold:
            return 0.5
        
        # 计算百分位
        percentile = sum(1 for s in scenario_scores if s <= score) / len(scenario_scores)
        return percentile
    
    def get_scenario_statistics(self, scenario_context: ScenarioContext) -> Dict[str, Any]:
        """
        获取场景统计信息
        
        Args:
            scenario_context: 场景上下文
            
        Returns:
            场景统计字典
        """
        scenario_key = self.classifier.classify(scenario_context)
        
        if scenario_key not in self.scenario_stats:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'q25': 0.0,
                'q50': 0.0,
                'q75': 0.0
            }
        
        scores = self.histories[scenario_key]
        if not scores:
            return self.scenario_stats[scenario_key]
        
        scores_array = np.array(scores)
        
        return {
            'scenario_key': scenario_key,
            'count': len(scores),
            'mean': np.mean(scores_array),
            'std': np.std(scores_array),
            'min': np.min(scores_array),
            'max': np.max(scores_array),
            'q25': np.percentile(scores_array, 25),
            'q50': np.percentile(scores_array, 50),
            'q75': np.percentile(scores_array, 75),
            'recent_mean': np.mean(scores_array[-10:]) if len(scores) >= 10 else np.mean(scores_array)
        }
    
    def get_recent_scores(self, count: Optional[int] = None) -> List[float]:
        """
        获取最近的质量分数
        
        Args:
            count: 返回的分数数量，None表示返回所有
            
        Returns:
            最近分数列表
        """
        if count is None:
            return list(self.recent_window)
        else:
            recent_list = list(self.recent_window)
            return recent_list[-count:] if len(recent_list) >= count else recent_list
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        获取全局统计信息
        
        Returns:
            全局统计字典
        """
        all_scores = []
        scenario_info = {}
        
        for scenario_key, scores in self.histories.items():
            all_scores.extend(scores)
            scenario_info[scenario_key] = {
                'count': len(scores),
                'recent_count': len([s for s in scores[-50:]]) if len(scores) >= 50 else len(scores)
            }
        
        if not all_scores:
            return {
                'total_updates': self.total_updates,
                'total_scenarios': 0,
                'total_scores': 0,
                'scenarios': scenario_info
            }
        
        all_scores_array = np.array(all_scores)
        
        return {
            'total_updates': self.total_updates,
            'total_scenarios': len(self.histories),
            'total_scores': len(all_scores),
            'global_mean': np.mean(all_scores_array),
            'global_std': np.std(all_scores_array),
            'global_min': np.min(all_scores_array),
            'global_max': np.max(all_scores_array),
            'scenarios': scenario_info,
            'last_update': self.last_update_time
        }
    
    def _update_scenario_stats(self, scenario_key: str, score: float):
        """
        更新场景统计信息
        
        Args:
            scenario_key: 场景标识符
            score: 质量分数
        """
        if scenario_key not in self.scenario_stats:
            # 初始化场景统计
            self.scenario_stats[scenario_key] = {
                'q_min': score,
                'q_max': score,
                'q_mean': score,
                'q_var': 0.0,
                'count': 1,
                'last_update': time.time()
            }
        else:
            # 使用EMA更新统计
            stats = self.scenario_stats[scenario_key]
            
            # 更新极值
            stats['q_min'] = min(stats['q_min'], score)
            stats['q_max'] = max(stats['q_max'], score)
            
            # EMA更新均值
            old_mean = stats['q_mean']
            stats['q_mean'] = (1 - self.ema_alpha) * old_mean + self.ema_alpha * score
            
            # EMA更新方差
            delta = score - old_mean
            stats['q_var'] = (1 - self.ema_alpha) * stats['q_var'] + self.ema_alpha * delta * delta
            
            # 更新计数和时间
            stats['count'] += 1
            stats['last_update'] = time.time()
    
    def estimate_scenario_difficulty(self, scenario_context: ScenarioContext) -> float:
        """
        估计场景难度因子
        
        Args:
            scenario_context: 场景上下文
            
        Returns:
            难度因子 [0.5, 2.0]，越大表示越难
        """
        scenario_key = self.classifier.classify(scenario_context)
        
        if scenario_key not in self.scenario_stats:
            return 1.0  # 默认难度
        
        stats = self.scenario_stats[scenario_key]
        
        # 基于平均质量分数估计难度
        # 质量分数越低，认为场景越难
        mean_quality = stats.get('q_mean', 0.5)
        
        # 映射到难度因子：质量0.2->难度2.0, 质量0.8->难度0.5
        if mean_quality <= 0.2:
            difficulty = 2.0
        elif mean_quality >= 0.8:
            difficulty = 0.5
        else:
            # 线性插值
            difficulty = 2.0 - (mean_quality - 0.2) / (0.8 - 0.2) * 1.5
        
        return np.clip(difficulty, 0.5, 2.0)
    
    def reset_scenario_data(self, scenario_context: ScenarioContext):
        """
        重置特定场景的数据
        
        Args:
            scenario_context: 要重置的场景上下文
        """
        scenario_key = self.classifier.classify(scenario_context)
        if scenario_key in self.histories:
            del self.histories[scenario_key]
        if scenario_key in self.scenario_stats:
            del self.scenario_stats[scenario_key]
    
    def clear_all_data(self):
        """清空所有历史数据"""
        self.histories.clear()
        self.scenario_stats.clear()
        self.recent_window.clear()
        self.total_updates = 0
        self.classifier.reset_statistics()
    
    def export_data(self) -> Dict[str, Any]:
        """
        导出历史数据
        
        Returns:
            包含所有历史数据的字典
        """
        return {
            'histories': dict(self.histories),
            'scenario_stats': dict(self.scenario_stats),
            'recent_window': list(self.recent_window),
            'total_updates': self.total_updates,
            'config': self.config,
            'classifier_stats': self.classifier.get_scenario_statistics()
        }
    
    def import_data(self, data: Dict[str, Any]):
        """
        导入历史数据
        
        Args:
            data: 要导入的历史数据字典
        """
        self.histories = defaultdict(list, data.get('histories', {}))
        self.scenario_stats = defaultdict(dict, data.get('scenario_stats', {}))
        
        recent_window_data = data.get('recent_window', [])
        self.recent_window = deque(recent_window_data, maxlen=self.window_size)
        
        self.total_updates = data.get('total_updates', 0)
        
        # 恢复分类器统计
        classifier_stats = data.get('classifier_stats', {})
        self.classifier.scenario_counts = classifier_stats 