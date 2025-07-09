"""
场景感知奖励系统 - 场景上下文数据结构

定义场景上下文相关的数据结构，用于场景分类和历史追踪。

作者：Augment Agent  
日期：2025-01-15
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import hashlib
from collections import defaultdict


@dataclass
class ScenarioContext:
    """
    场景上下文数据结构
    
    记录场景的关键特征，用于场景分类和奖励计算
    """
    has_n1_fault: bool = False
    load_scale_factor: float = 1.0
    has_gen_fluctuation: bool = False
    fault_branch_idx: Optional[int] = None
    scenario_tag: str = ""
    
    def __post_init__(self):
        """生成场景标签"""
        if not self.scenario_tag:
            self.scenario_tag = self._generate_tag()
    
    def _generate_tag(self) -> str:
        """生成人类可读的场景标签"""
        parts = []
        
        # 故障状态
        if self.has_n1_fault:
            parts.append(f"故障_{self.fault_branch_idx}" if self.fault_branch_idx is not None else "故障")
        else:
            parts.append("正常")
        
        # 负荷水平
        if abs(self.load_scale_factor - 1.0) > 0.05:
            parts.append(f"负荷{self.load_scale_factor:.1f}")
        
        # 发电波动
        if self.has_gen_fluctuation:
            parts.append("发电波动")
        
        return "_".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'has_n1_fault': self.has_n1_fault,
            'load_scale_factor': self.load_scale_factor,
            'has_gen_fluctuation': self.has_gen_fluctuation,
            'fault_branch_idx': self.fault_branch_idx,
            'scenario_tag': self.scenario_tag
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioContext':
        """从字典创建场景上下文"""
        return cls(
            has_n1_fault=data.get('has_n1_fault', False),
            load_scale_factor=data.get('load_scale_factor', 1.0),
            has_gen_fluctuation=data.get('has_gen_fluctuation', False),
            fault_branch_idx=data.get('fault_branch_idx'),
            scenario_tag=data.get('scenario_tag', "")
        )


class ScenarioClassifier:
    """
    场景分类器
    
    将场景上下文映射为场景标识符，用于历史数据分组
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化场景分类器
        
        Args:
            config: 分类配置，包含负荷分段等参数
        """
        self.config = config or {}
        
        # 负荷缩放分段
        self.load_scale_bins = self.config.get('load_scale_bins', [0.8, 1.0, 1.2])
        
        # 最大场景数限制
        self.max_scenarios = self.config.get('max_scenarios', 20)
        
        # 已知场景计数
        self.scenario_counts = {}
        
        # 场景类别映射
        self.category_mappings = {
            'F': '故障类',
            'N': '正常类', 
            'F_G': '故障发电波动类',
            'N_G': '正常发电波动类'
        }
    
    def classify(self, scenario_context: ScenarioContext) -> str:
        """
        分类场景上下文为场景标识符
        
        Args:
            scenario_context: 场景上下文
            
        Returns:
            场景标识符字符串
        """
        # 1. 故障状态
        fault_status = "F" if scenario_context.has_n1_fault else "N"
        
        # 2. 负荷水平分段
        load_scale = self._round_to_bin(scenario_context.load_scale_factor, self.load_scale_bins)
        
        # 3. 发电波动状态
        gen_fluctuation = "G" if scenario_context.has_gen_fluctuation else ""
        
        # 4. 组合场景标识符
        parts = [fault_status, f"L{load_scale}"]
        if gen_fluctuation:
            parts.append(gen_fluctuation)
        
        scenario_id = "_".join(parts)
        
        # 5. 更新场景计数
        self.scenario_counts[scenario_id] = self.scenario_counts.get(scenario_id, 0) + 1
        
        return scenario_id
    
    def extract_category(self, scenario_id: str) -> str:
        """
        从场景ID中提取类别标识符
        
        Args:
            scenario_id: 完整的场景标识符（如"F_L1.0"或"N_L1.2_G"）
            
        Returns:
            场景类别标识符（如"F"或"N_G"）
        """
        if not scenario_id:
            return "UNKNOWN"
            
        parts = scenario_id.split('_')
        
        # 基础类别（故障状态）
        base_category = parts[0] if parts else "UNKNOWN"
        
        # 检查是否有发电波动
        has_gen_fluctuation = 'G' in parts
        
        if has_gen_fluctuation:
            return f"{base_category}_G"
        else:
            return base_category
    
    def get_category_description(self, category: str) -> str:
        """
        获取类别的人类可读描述
        
        Args:
            category: 类别标识符
            
        Returns:
            类别描述
        """
        return self.category_mappings.get(category, f"未知类别({category})")
    
    def list_scenarios_in_category(self, category: str) -> List[str]:
        """
        列出指定类别下的所有已知场景
        
        Args:
            category: 类别标识符
            
        Returns:
            属于该类别的场景ID列表
        """
        scenarios_in_category = []
        
        for scenario_id in self.scenario_counts.keys():
            if self.extract_category(scenario_id) == category:
                scenarios_in_category.append(scenario_id)
                
        return scenarios_in_category

    def get_signature(self, scenario_context: ScenarioContext) -> Tuple:
        """
        获取可哈希的场景签名，用于字典键
        
        Args:
            scenario_context: 场景上下文
            
        Returns:
            可哈希的元组
        """
        return (
            scenario_context.has_n1_fault,
            round(scenario_context.load_scale_factor, 1),
            scenario_context.has_gen_fluctuation
        )
    
    def _round_to_bin(self, value: float, bins: list) -> float:
        """
        将数值舍入到最近的分段
        
        Args:
            value: 待舍入的值
            bins: 分段列表
            
        Returns:
            最近的分段值
        """
        if not bins:
            return round(value, 1)
        
        # 找到最近的分段
        closest_bin = min(bins, key=lambda x: abs(x - value))
        return closest_bin
    
    def get_scenario_statistics(self) -> Dict[str, int]:
        """获取场景统计信息"""
        return self.scenario_counts.copy()
    
    def get_category_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取按类别统计的信息
        
        Returns:
            类别统计字典
        """
        category_stats = defaultdict(lambda: {'count': 0, 'scenarios': []})
        
        for scenario_id, count in self.scenario_counts.items():
            category = self.extract_category(scenario_id)
            category_stats[category]['count'] += count
            category_stats[category]['scenarios'].append({
                'scenario_id': scenario_id,
                'count': count
            })
        
        # 转换为普通字典并添加描述
        result = {}
        for category, stats in category_stats.items():
            result[category] = {
                'description': self.get_category_description(category),
                'total_count': stats['count'],
                'scenario_count': len(stats['scenarios']),
                'scenarios': stats['scenarios']
            }
        
        return result
    
    def reset_statistics(self):
        """重置场景统计"""
        self.scenario_counts.clear() 