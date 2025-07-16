"""
向后兼容性管理模块

确保v3.0抽象决策上下文架构与传统具体动作代码路径的无缝集成。
提供自动检测、回退机制和配置管理功能。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityConfig:
    """兼容性配置"""
    enable_v3_architecture: bool = True  # 是否启用v3.0架构
    auto_fallback: bool = True           # 是否自动回退到传统模式
    strict_validation: bool = False      # 是否严格验证格式
    log_compatibility_events: bool = True # 是否记录兼容性事件


class CompatibilityManager:
    """
    兼容性管理器
    
    负责管理v3.0架构与传统代码路径的兼容性，
    提供自动检测、转换和回退功能。
    """
    
    def __init__(self, config: Optional[CompatibilityConfig] = None):
        """
        初始化兼容性管理器
        
        Args:
            config: 兼容性配置，None使用默认配置
        """
        self.config = config or CompatibilityConfig()
        self._stats = {
            'v3_calls': 0,
            'legacy_calls': 0,
            'fallback_events': 0,
            'conversion_events': 0
        }
    
    def detect_action_format(self, action_data: Any) -> str:
        """
        检测动作数据格式
        
        Args:
            action_data: 动作数据（可能是tuple或AbstractDecisionContext）
            
        Returns:
            'v3_context' | 'legacy_tuple' | 'unknown'
        """
        if hasattr(action_data, 'node_embedding') and hasattr(action_data, 'strategy_vector'):
            return 'v3_context'
        elif isinstance(action_data, (tuple, list)) and len(action_data) == 2:
            return 'legacy_tuple'
        else:
            return 'unknown'
    
    def detect_select_action_result(self, result: Tuple) -> str:
        """
        检测select_action返回结果的格式
        
        Args:
            result: select_action的返回结果
            
        Returns:
            'v3_format' | 'legacy_format' | 'unknown_format'
        """
        if len(result) == 3:
            # (decision_context, value, embeddings) or (action, log_prob, value)
            first_item = result[0]
            if hasattr(first_item, 'node_embedding'):
                return 'v3_format'
            else:
                return 'legacy_format'
        elif len(result) == 4:
            # (action, log_prob, value, embeddings)
            return 'legacy_format'
        else:
            return 'unknown_format'
    
    def ensure_concrete_action(self, action_data: Any, env: Any = None) -> Optional[Tuple[int, int]]:
        """
        确保获得具体的执行动作
        
        Args:
            action_data: 动作数据（DecisionContext或具体动作）
            env: 环境实例（用于materialize）
            
        Returns:
            具体动作tuple，或None如果无法获得
        """
        format_type = self.detect_action_format(action_data)
        
        if format_type == 'legacy_tuple':
            self._stats['legacy_calls'] += 1
            return action_data
        elif format_type == 'v3_context':
            self._stats['v3_calls'] += 1
            
            # 尝试实体化
            if env and hasattr(env, 'materialize_action'):
                concrete_action = env.materialize_action(action_data)
                if concrete_action is not None:
                    return concrete_action
            
            # 回退到上下文中的调试信息
            if self.config.auto_fallback and action_data.node_context:
                self._stats['fallback_events'] += 1
                node_id = action_data.node_context.get('node_id', 0)
                available_partitions = action_data.node_context.get('available_partitions', [1])
                partition_id = available_partitions[0] if available_partitions else 1
                
                if self.config.log_compatibility_events:
                    logger.warning(f"回退到上下文调试信息：node_id={node_id}, partition_id={partition_id}")
                
                return (node_id, partition_id)
        
        # 无法处理的格式
        if self.config.log_compatibility_events:
            logger.error(f"无法处理的动作格式：{type(action_data)}")
        
        return None
    
    def normalize_select_action_result(self, result: Tuple, agent_type: str = 'auto') -> Tuple:
        """
        标准化select_action的返回结果
        
        Args:
            result: 原始返回结果
            agent_type: agent类型 ('enhanced', 'legacy', 'auto')
            
        Returns:
            标准化的结果tuple
        """
        format_type = self.detect_select_action_result(result)
        
        if format_type == 'v3_format':
            # v3.0格式: (decision_context, value, embeddings)
            return result
        elif format_type == 'legacy_format':
            if len(result) == 3:
                # (action, log_prob, value) -> 添加空embeddings
                return result + (None,)
            else:
                # (action, log_prob, value, embeddings)
                return result
        else:
            # 未知格式，返回原始结果并记录警告
            if self.config.log_compatibility_events:
                logger.warning(f"未知的select_action返回格式：{len(result)}元组")
            return result
    
    def validate_memory_storage(self, memory: Any) -> Dict[str, Any]:
        """
        验证内存存储的兼容性
        
        Args:
            memory: 内存缓冲区实例
            
        Returns:
            验证结果字典
        """
        validation_result = {
            'is_compatible': True,
            'supports_v3': False,
            'supports_legacy': False,
            'issues': []
        }
        
        # 检查是否支持v3.0存储
        if hasattr(memory, 'decision_contexts') and hasattr(memory, 'store'):
            validation_result['supports_v3'] = True
        else:
            validation_result['issues'].append("内存缓冲区不支持v3.0决策上下文存储")
        
        # 检查是否支持传统存储
        if hasattr(memory, 'actions') and hasattr(memory, 'store'):
            validation_result['supports_legacy'] = True
        else:
            validation_result['issues'].append("内存缓冲区不支持传统动作存储")
        
        # 检查store方法的签名
        if hasattr(memory, 'store'):
            try:
                import inspect
                signature = inspect.signature(memory.store)
                params = list(signature.parameters.keys())
                
                if 'decision_context' in params:
                    validation_result['supports_v3'] = True
                if 'action' in params:
                    validation_result['supports_legacy'] = True
            except Exception as e:
                validation_result['issues'].append(f"无法检查store方法签名：{e}")
        
        validation_result['is_compatible'] = (
            validation_result['supports_v3'] and validation_result['supports_legacy']
        )
        
        return validation_result
    
    def create_legacy_adapter(self, v3_agent: Any) -> 'LegacyAgentAdapter':
        """
        为v3.0 agent创建传统接口适配器
        
        Args:
            v3_agent: v3.0架构的agent
            
        Returns:
            传统接口适配器
        """
        return LegacyAgentAdapter(v3_agent, self)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取兼容性统计信息"""
        total_calls = self._stats['v3_calls'] + self._stats['legacy_calls']
        
        stats = self._stats.copy()
        stats.update({
            'total_calls': total_calls,
            'v3_ratio': self._stats['v3_calls'] / max(total_calls, 1),
            'fallback_ratio': self._stats['fallback_events'] / max(self._stats['v3_calls'], 1),
            'config': self.config
        })
        
        return stats


class LegacyAgentAdapter:
    """
    传统Agent接口适配器
    
    为v3.0架构的agent提供传统的select_action接口，
    确保与旧版训练代码的完全兼容性。
    """
    
    def __init__(self, v3_agent: Any, compatibility_manager: CompatibilityManager):
        """
        初始化适配器
        
        Args:
            v3_agent: v3.0架构的agent
            compatibility_manager: 兼容性管理器
        """
        self.v3_agent = v3_agent
        self.compat_mgr = compatibility_manager
        
        # 代理所有其他属性到原始agent
        for attr_name in dir(v3_agent):
            if not attr_name.startswith('_') and attr_name != 'select_action':
                try:
                    setattr(self, attr_name, getattr(v3_agent, attr_name))
                except (AttributeError, TypeError):
                    pass  # 跳过无法设置的属性
    
    def select_action(self, state: Dict[str, torch.Tensor], 
                     training: bool = True,
                     return_embeddings: bool = False) -> Tuple[Tuple[int, int], float, float, Optional[Dict]]:
        """
        传统格式的select_action接口
        
        Returns:
            (action, log_prob, value, embeddings) - 传统格式
        """
        # 调用v3.0 agent的select_action
        result = self.v3_agent.select_action(state, training, return_embeddings)
        
        # 检测返回格式
        format_type = self.compat_mgr.detect_select_action_result(result)
        
        if format_type == 'v3_format':
            # v3.0格式: (decision_context, value, embeddings)
            decision_context, value, embeddings = result
            
            # 转换为传统格式
            if decision_context is not None:
                # 从决策上下文提取具体动作（使用回退机制）
                concrete_action = self.compat_mgr.ensure_concrete_action(
                    decision_context, 
                    getattr(self.v3_agent, 'env', None)
                )
                
                if concrete_action is None:
                    # 最后的回退：使用默认动作
                    concrete_action = (0, 1)
                    logger.warning("使用默认动作作为最后回退")
                
                # 从决策上下文重建log_prob
                log_prob = decision_context.get_selected_log_prob().item() if decision_context else 0.0
                
                return (concrete_action, log_prob, value, embeddings)
            else:
                return (None, 0.0, value, embeddings)
        else:
            # 已经是传统格式，直接返回
            return self.compat_mgr.normalize_select_action_result(result)


def create_compatibility_layer(config: Optional[CompatibilityConfig] = None) -> CompatibilityManager:
    """
    创建兼容性层
    
    便捷函数，用于快速设置兼容性管理
    
    Args:
        config: 兼容性配置
        
    Returns:
        配置好的兼容性管理器
    """
    return CompatibilityManager(config)


def ensure_backward_compatibility(agent: Any, 
                                 memory: Any = None,
                                 config: Optional[CompatibilityConfig] = None) -> Dict[str, Any]:
    """
    确保系统的向后兼容性
    
    Args:
        agent: agent实例
        memory: 内存缓冲区实例
        config: 兼容性配置
        
    Returns:
        兼容性检查结果
    """
    compat_mgr = CompatibilityManager(config)
    
    results = {
        'agent_compatible': True,
        'memory_compatible': True,
        'issues': [],
        'recommendations': []
    }
    
    # 检查agent兼容性
    if not hasattr(agent, 'select_action'):
        results['agent_compatible'] = False
        results['issues'].append("Agent缺少select_action方法")
    
    # 检查内存兼容性
    if memory is not None:
        memory_validation = compat_mgr.validate_memory_storage(memory)
        results['memory_compatible'] = memory_validation['is_compatible']
        results['issues'].extend(memory_validation['issues'])
        
        if not memory_validation['supports_v3']:
            results['recommendations'].append("考虑升级内存缓冲区以支持v3.0架构")
    
    # 提供解决方案
    if not results['agent_compatible'] or not results['memory_compatible']:
        results['recommendations'].append("使用LegacyAgentAdapter进行接口适配")
        results['recommendations'].append("启用自动回退模式")
    
    return results


# 全局兼容性管理器实例
global_compatibility_manager = CompatibilityManager()


def set_global_compatibility_config(config: CompatibilityConfig):
    """设置全局兼容性配置"""
    global global_compatibility_manager
    global_compatibility_manager = CompatibilityManager(config)


def get_global_compatibility_manager() -> CompatibilityManager:
    """获取全局兼容性管理器"""
    return global_compatibility_manager 