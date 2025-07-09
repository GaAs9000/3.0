"""
场景感知奖励系统 - 场景感知历史追踪器

实现按场景分类的历史数据存储和管理，支持场景内百分位计算。

作者：Augment Agent
日期：2025-01-15
"""

import time
import numpy as np
import torch
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
from .scenario_context import ScenarioContext, ScenarioClassifier


class ScenarioAwareHistoryTracker:
    """
    场景感知历史追踪器
    
    按场景分类存储和管理历史质量分数，支持场景内百分位计算
    优化版：GPU加速的核心统计计算
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化历史追踪器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 从新的配置结构中提取参数
        history_config = self.config.get('history_tracking', {})
        scenario_classification_config = self.config.get('scenario_classification', {})
        
        # 配置参数（使用新的分层配置结构）
        self.window_size = history_config.get('window_size', 15)
        self.min_samples_for_percentile = history_config.get('min_samples_for_percentile', 5)
        self.max_history_per_scenario = history_config.get('max_history_per_scenario', 1000)
        self.min_samples_threshold = history_config.get('min_samples_threshold', 3)
        
        # GPU设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 核心数据结构
        self.histories = defaultdict(list)  # {scenario_key: [scores]}
        self.recent_window = deque(maxlen=self.window_size)  # 最近的全局窗口
        self.scenario_stats = defaultdict(dict)  # 场景统计信息
        
        # GPU加速的统计缓存
        self.gpu_stats_cache = {}  # {scenario_key: torch.Tensor}
        self.cache_dirty = set()  # 需要更新的场景集合
        
        # 场景分类器
        self.classifier = ScenarioClassifier(scenario_classification_config)
        
        # 统计信息
        self.total_updates = 0
        self.last_update_time = time.time()
        
        # EMA参数
        self.ema_alpha = history_config.get('ema_alpha', 0.1)
    
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
        
        # 3. 标记GPU缓存需要更新
        self.cache_dirty.add(scenario_key)
        
        # 4. 维护历史长度限制
        if len(self.histories[scenario_key]) > self.max_history_per_scenario:
            # 删除最旧的20%数据
            remove_count = int(self.max_history_per_scenario * 0.2)
            del self.histories[scenario_key][:remove_count]
            # 需要重新计算GPU缓存
            self.cache_dirty.add(scenario_key)
        
        # 5. 更新最近窗口（用于改善率计算）
        self.recent_window.append(score)
        
        # 6. 更新场景统计（使用GPU加速）
        self._update_scenario_stats_gpu(scenario_key, score)
        
        # 7. 更新全局统计
        self.total_updates += 1
        self.last_update_time = time.time()
    
    def _update_scenario_stats_gpu(self, scenario_key: str, score: float):
        """
        使用GPU加速更新场景统计信息
        
        Args:
            scenario_key: 场景标识符
            score: 质量分数
        """
        # 将分数转换为GPU张量
        score_tensor = torch.tensor(score, device=self.device, dtype=torch.float32)
        
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
            # 使用GPU加速的EMA更新
            stats = self.scenario_stats[scenario_key]
            
            # 转换为GPU张量进行计算
            old_mean_tensor = torch.tensor(stats['q_mean'], device=self.device, dtype=torch.float32)
            old_var_tensor = torch.tensor(stats['q_var'], device=self.device, dtype=torch.float32)
            alpha_tensor = torch.tensor(self.ema_alpha, device=self.device, dtype=torch.float32)
            
            # GPU加速的统计更新
            stats['q_min'] = min(stats['q_min'], score)
            stats['q_max'] = max(stats['q_max'], score)
            
            # EMA更新均值（GPU）
            new_mean = (1 - alpha_tensor) * old_mean_tensor + alpha_tensor * score_tensor
            stats['q_mean'] = new_mean.item()
            
            # EMA更新方差（GPU）
            delta = score_tensor - old_mean_tensor
            new_var = (1 - alpha_tensor) * old_var_tensor + alpha_tensor * delta * delta
            stats['q_var'] = new_var.item()
            
            # 更新计数和时间
            stats['count'] += 1
            stats['last_update'] = time.time()
    
    def _get_gpu_scenario_stats(self, scenario_key: str) -> Optional[torch.Tensor]:
        """
        获取GPU加速的场景统计信息
        
        Args:
            scenario_key: 场景标识符
            
        Returns:
            GPU张量 [mean, std, min, max] 或 None
        """
        # 检查是否需要更新缓存
        if scenario_key in self.cache_dirty or scenario_key not in self.gpu_stats_cache:
            if scenario_key in self.histories and len(self.histories[scenario_key]) >= self.min_samples_threshold:
                # 将历史数据转换为GPU张量
                scores_list = self.histories[scenario_key]
                scores_tensor = torch.tensor(scores_list, device=self.device, dtype=torch.float32)
                
                # GPU上计算统计信息
                stats_tensor = torch.stack([
                    scores_tensor.mean(),
                    scores_tensor.std(),
                    scores_tensor.min(),
                    scores_tensor.max()
                ])
                
                self.gpu_stats_cache[scenario_key] = stats_tensor
                self.cache_dirty.discard(scenario_key)
            else:
                return None
        
        return self.gpu_stats_cache.get(scenario_key)

    def compute_scenario_percentile(self, score: float, scenario_context: ScenarioContext) -> float:
        """
        计算场景内百分位 - GPU加速版
        
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
        
        # GPU加速的百分位计算
        scores_tensor = torch.tensor(scenario_scores, device=self.device, dtype=torch.float32)
        score_tensor = torch.tensor(score, device=self.device, dtype=torch.float32)
        
        # 使用GPU计算百分位
        percentile_tensor = (scores_tensor <= score_tensor).float().mean()
        return percentile_tensor.item()
    
    def get_scenario_statistics(self, scenario_context: ScenarioContext) -> Dict[str, Any]:
        """
        获取场景统计信息 - GPU加速版
        
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
        
        # 尝试使用GPU加速统计
        gpu_stats = self._get_gpu_scenario_stats(scenario_key)
        if gpu_stats is not None:
            # 使用GPU计算的统计信息
            scores_list = self.histories[scenario_key]
            scores_tensor = torch.tensor(scores_list, device=self.device, dtype=torch.float32)
            
            # GPU上计算百分位数
            q25, q50, q75 = torch.quantile(scores_tensor, torch.tensor([0.25, 0.5, 0.75], device=self.device))
            
            return {
                'scenario_key': scenario_key,
                'count': len(scores_list),
                'mean': gpu_stats[0].item(),
                'std': gpu_stats[1].item(),
                'min': gpu_stats[2].item(),
                'max': gpu_stats[3].item(),
                'q25': q25.item(),
                'q50': q50.item(),
                'q75': q75.item(),
                'recent_mean': scores_tensor[-10:].mean().item() if len(scores_list) >= 10 else gpu_stats[0].item()
            }
        else:
            # 回退到CPU计算
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

    def get_scenario_baseline(self, scenario_context: ScenarioContext) -> Tuple[float, str, Dict[str, Any]]:
        """
        获取场景基线质量分数 - GPU加速的两级查询
        
        查询策略：
        Level 1: 精确场景匹配
        Level 2: 同类场景平均
        Level 3: 全局基线回退
        
        Args:
            scenario_context: 场景上下文
            
        Returns:
            Tuple[float, str, Dict]: (基线分数, 数据来源, 详细信息)
        """
        scenario_key = self.classifier.classify(scenario_context)
        scenario_category = self.classifier.extract_category(scenario_key)
        
        details = {
            'scenario_key': scenario_key,
            'scenario_category': scenario_category,
            'query_level': None,
            'data_source_details': {}
        }
        
        # === Level 1: 精确场景匹配（GPU加速） ===
        if scenario_key in self.histories and len(self.histories[scenario_key]) >= self.min_samples_threshold:
            # 使用GPU加速计算
            gpu_stats = self._get_gpu_scenario_stats(scenario_key)
            if gpu_stats is not None:
                baseline = gpu_stats[0].item()  # mean
                scenario_scores = self.histories[scenario_key]
                
                details.update({
                    'query_level': 1,
                    'data_source_details': {
                        'exact_scenario_count': len(scenario_scores),
                        'recent_scores': scenario_scores[-5:] if len(scenario_scores) >= 5 else scenario_scores,
                        'score_range': [gpu_stats[2].item(), gpu_stats[3].item()]  # [min, max]
                    }
                })
                return float(baseline), 'exact_scenario', details
        
        # === Level 2: 同类场景平均（GPU加速） ===
        category_scenarios = self.classifier.list_scenarios_in_category(scenario_category)
        category_scores_tensors = []
        category_details = {}
        
        for category_scenario in category_scenarios:
            if category_scenario in self.histories and len(self.histories[category_scenario]) >= self.min_samples_threshold:
                scores = self.histories[category_scenario]
                scores_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
                category_scores_tensors.append(scores_tensor)
                
                category_details[category_scenario] = {
                    'count': len(scores),
                    'mean': float(scores_tensor.mean().item()),
                    'recent_mean': float(scores_tensor[-5:].mean().item()) if len(scores) >= 5 else float(scores_tensor.mean().item())
                }
        
        if category_scores_tensors:
            # GPU上合并并计算类别基线
            all_category_scores = torch.cat(category_scores_tensors, dim=0)
            category_baseline = all_category_scores.mean().item()
            
            details.update({
                'query_level': 2,
                'data_source_details': {
                    'category_description': self.classifier.get_category_description(scenario_category),
                    'contributing_scenarios': len(category_details),
                    'total_category_scores': len(all_category_scores),
                    'scenario_details': category_details,
                    'category_score_range': [float(all_category_scores.min().item()), float(all_category_scores.max().item())]
                }
            })
            return float(category_baseline), 'category_average', details
        
        # === Level 3: 全局基线回退（GPU加速） ===
        all_scores_tensors = []
        global_details = {}
        
        for scenario_id, scores in self.histories.items():
            if len(scores) >= self.min_samples_threshold:
                scores_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
                all_scores_tensors.append(scores_tensor)
                global_details[scenario_id] = {
                    'count': len(scores),
                    'category': self.classifier.extract_category(scenario_id)
                }
        
        if all_scores_tensors:
            # GPU上计算全局基线
            all_global_scores = torch.cat(all_scores_tensors, dim=0)
            global_baseline = all_global_scores.mean().item()
            
            details.update({
                'query_level': 3,
                'data_source_details': {
                    'total_scenarios': len(global_details),
                    'total_scores': len(all_global_scores),
                    'global_score_range': [float(all_global_scores.min().item()), float(all_global_scores.max().item())],
                    'scenario_summary': global_details
                }
            })
            return float(global_baseline), 'global_fallback', details
        
        # === 无历史数据 ===
        details.update({
            'query_level': 0,
            'data_source_details': {
                'reason': 'no_historical_data',
                'total_known_scenarios': len(self.histories),
                'total_updates': self.total_updates
            }
        })
        return 0.5, 'no_data', details  # 保守的默认值

    def is_new_scenario(self, scenario_context: ScenarioContext) -> Tuple[bool, str]:
        """
        判断是否为新场景（用于探索奖励判断）
        
        Args:
            scenario_context: 场景上下文
            
        Returns:
            Tuple[bool, str]: (是否为新场景, 新场景类型)
        """
        scenario_key = self.classifier.classify(scenario_context)
        scenario_category = self.classifier.extract_category(scenario_key)
        
        # 检查精确场景
        if scenario_key not in self.histories:
            return True, 'new_exact_scenario'
        
        if len(self.histories[scenario_key]) < self.min_samples_threshold:
            return True, 'insufficient_data_scenario'
        
        # 检查类别
        category_scenarios = self.classifier.list_scenarios_in_category(scenario_category)
        category_has_sufficient_data = False
        
        for category_scenario in category_scenarios:
            if (category_scenario in self.histories and 
                len(self.histories[category_scenario]) >= self.min_samples_threshold):
                category_has_sufficient_data = True
                break
        
        if not category_has_sufficient_data:
            return True, 'new_category'
        
        return False, 'known_scenario'

    def get_exploration_context(self, scenario_context: ScenarioContext) -> Dict[str, Any]:
        """
        获取探索上下文信息（用于计算探索奖励）
        
        Args:
            scenario_context: 场景上下文
            
        Returns:
            探索上下文字典
        """
        scenario_key = self.classifier.classify(scenario_context)
        scenario_category = self.classifier.extract_category(scenario_key)
        
        is_new, new_type = self.is_new_scenario(scenario_context)
        
        # 计算数据稀疏性
        exact_data_count = len(self.histories.get(scenario_key, []))
        
        category_scenarios = self.classifier.list_scenarios_in_category(scenario_category)
        category_data_count = sum(
            len(self.histories.get(cat_scenario, [])) 
            for cat_scenario in category_scenarios
        )
        
        global_data_count = sum(len(scores) for scores in self.histories.values())
        
        return {
            'scenario_key': scenario_key,
            'scenario_category': scenario_category,
            'category_description': self.classifier.get_category_description(scenario_category),
            'is_new_scenario': is_new,
            'new_scenario_type': new_type,
            'data_sparsity': {
                'exact_scenario_count': exact_data_count,
                'category_total_count': category_data_count,
                'global_total_count': global_data_count,
                'category_scenario_types': len(category_scenarios)
            },
            'exploration_priority': self._calculate_exploration_priority(
                is_new, new_type, exact_data_count, category_data_count
            )
        }

    def _calculate_exploration_priority(self, is_new: bool, new_type: str, 
                                      exact_count: int, category_count: int) -> float:
        """
        计算探索优先级（0-1，越高越需要探索）
        
        Args:
            is_new: 是否为新场景
            new_type: 新场景类型
            exact_count: 精确场景数据量
            category_count: 类别数据量
            
        Returns:
            探索优先级分数
        """
        if not is_new:
            return 0.0
        
        priority_map = {
            'new_exact_scenario': 0.8,
            'insufficient_data_scenario': 0.6,
            'new_category': 1.0
        }
        
        base_priority = priority_map.get(new_type, 0.5)
        
        # 数据稀疏性调整
        if exact_count == 0:
            scarcity_boost = 0.2
        elif exact_count < self.min_samples_threshold:
            scarcity_boost = 0.1 * (self.min_samples_threshold - exact_count) / self.min_samples_threshold
        else:
            scarcity_boost = 0.0
        
        # 类别稀疏性调整
        if category_count < 10:  # 类别数据不足10个样本
            category_boost = 0.1
        else:
            category_boost = 0.0
        
        final_priority = np.clip(base_priority + scarcity_boost + category_boost, 0.0, 1.0)
        return float(final_priority) 