#!/usr/bin/env python3
"""
Enhanced PPO Agent with Action Embedding Support

基于动作嵌入的PPO智能体实现，支持：
- 双塔架构（状态塔+动作塔）
- 动态K值支持
- 连通性硬约束
- 高效的动作选择流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import time

from .agent import PPOAgent, BatchedState
from models.partition_encoder import PartitionEncoder, create_partition_encoder
from models.actor_network import EnhancedActorNetwork, create_actor_network

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessmentResult:
    """风险评估结果"""
    risk_level: str  # 'safe', 'low', 'medium', 'high'
    risk_score: float  # 0.0-1.0
    available_partitions: List[int]
    fallback_reason: Optional[str] = None
    cache_hit: bool = False


class ConnectivityCache:
    """连通性检查缓存系统"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _get_cache_key(self, partition_state: torch.Tensor, node_id: int, target_partition: int) -> str:
        """生成缓存键"""
        # 使用分区状态的哈希和动作信息作为键
        partition_hash = hash(tuple(partition_state.cpu().numpy()))
        return f"{partition_hash}_{node_id}_{target_partition}"

    def get(self, partition_state: torch.Tensor, node_id: int, target_partition: int) -> Optional[Dict]:
        """获取缓存的连通性检查结果"""
        key = self._get_cache_key(partition_state, node_id, target_partition)
        current_time = time.time()

        if key in self.cache:
            cached_time, result = self.cache[key]
            if current_time - cached_time < self.ttl_seconds:
                self.access_times[key] = current_time
                return result
            else:
                # 过期，删除
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]

        return None

    def put(self, partition_state: torch.Tensor, node_id: int, target_partition: int, result: Dict):
        """缓存连通性检查结果"""
        key = self._get_cache_key(partition_state, node_id, target_partition)
        current_time = time.time()

        # 如果缓存已满，删除最旧的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = (current_time, result)
        self.access_times[key] = current_time


class PerformanceMonitor:
    """性能监控器 - 用于自适应约束调节"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.episode_data = []
        self.current_episode = 0

    def record_episode(self, episode_info: Dict[str, Any]):
        """记录单个episode的性能数据"""
        self.current_episode += 1

        # 提取关键指标
        performance_data = {
            'episode': self.current_episode,
            'reward': episode_info.get('reward', -float('inf')),
            'success': episode_info.get('success', False),
            'connectivity_score': episode_info.get('connectivity_score', 0.0),
            'quality_score': episode_info.get('quality_score', 0.0),
            'episode_length': episode_info.get('episode_length', 0),
            'termination_reason': episode_info.get('termination_reason', 'unknown')
        }

        # 从metrics中提取更多指标
        if 'metrics' in episode_info:
            metrics = episode_info['metrics']
            performance_data.update({
                'cv': metrics.get('cv', 1.0),
                'coupling_ratio': metrics.get('coupling_ratio', 1.0),
                'power_imbalance': metrics.get('power_imbalance_normalized', 1.0)
            })

        self.episode_data.append(performance_data)

        # 保持窗口大小
        if len(self.episode_data) > self.window_size:
            self.episode_data = self.episode_data[-self.window_size:]

    def get_performance_metrics(self) -> Dict[str, float]:
        """计算当前性能指标"""
        if not self.episode_data:
            return {
                'success_rate': 0.0,
                'avg_connectivity': 0.0,
                'avg_reward': -float('inf'),
                'avg_quality': 0.0,
                'constraint_violation_rate': 1.0
            }

        recent_data = self.episode_data[-min(20, len(self.episode_data)):]  # 最近20个episode

        # 计算成功率
        success_rate = sum(1 for d in recent_data if d['success']) / len(recent_data)

        # 计算平均连通性分数
        avg_connectivity = sum(d['connectivity_score'] for d in recent_data) / len(recent_data)

        # 计算平均奖励
        avg_reward = sum(d['reward'] for d in recent_data) / len(recent_data)

        # 计算平均质量分数
        avg_quality = sum(d['quality_score'] for d in recent_data) / len(recent_data)

        # 计算约束违规率（基于终止原因）
        violation_count = sum(1 for d in recent_data
                            if d['termination_reason'] in ['connectivity_violation', 'constraint_violation'])
        constraint_violation_rate = violation_count / len(recent_data)

        return {
            'success_rate': success_rate,
            'avg_connectivity': avg_connectivity,
            'avg_reward': avg_reward,
            'avg_quality': avg_quality,
            'constraint_violation_rate': constraint_violation_rate,
            'episode_count': len(recent_data)
        }

    def should_adjust_constraints(self, target_success_rate: float = 0.6,
                                target_connectivity: float = 0.5) -> Tuple[bool, str]:
        """判断是否需要调整约束"""
        metrics = self.get_performance_metrics()

        if metrics['episode_count'] < 10:  # 数据不足
            return False, "insufficient_data"

        # 成功率过低 -> 放松约束
        if metrics['success_rate'] < target_success_rate * 0.7:
            return True, "relax_low_success"

        # 连通性过低 -> 加强约束
        if metrics['avg_connectivity'] < target_connectivity:
            return True, "tighten_low_connectivity"

        # 约束违规率过高 -> 放松约束
        if metrics['constraint_violation_rate'] > 0.3:
            return True, "relax_high_violation"

        # 表现良好 -> 逐渐加强约束
        if (metrics['success_rate'] > target_success_rate and
            metrics['avg_connectivity'] > target_connectivity * 1.2):
            return True, "tighten_good_performance"

        return False, "no_adjustment"


class RiskPredictor:
    """轻量级风险预测器"""

    def __init__(self, device: torch.device):
        self.device = device
        self.history = defaultdict(list)  # 节点历史决策模式

    def predict_risk(self,
                    node_id: int,
                    current_partition: torch.Tensor,
                    boundary_nodes: torch.Tensor,
                    hetero_data: Any) -> RiskAssessmentResult:
        """
        预测节点移动的风险等级

        Args:
            node_id: 目标节点ID
            current_partition: 当前分区状态
            boundary_nodes: 边界节点
            hetero_data: 图数据

        Returns:
            风险评估结果
        """
        # 检查输入参数
        if current_partition is None:
            # 提供默认的可用分区而不是空列表
            return RiskAssessmentResult(
                risk_level='medium',
                risk_score=0.5,
                available_partitions=list(range(1, 4)),  # 默认3个分区
                fallback_reason="current_partition_is_none"
            )

        # 1. 基础风险评估：节点度数和当前分区大小
        node_degree = self._get_node_degree(node_id, hetero_data)

        # 安全检查节点ID
        if node_id >= len(current_partition):
            return RiskAssessmentResult(
                risk_level='medium',
                risk_score=0.5,
                available_partitions=list(range(1, 4)),  # 默认3个分区
                fallback_reason="invalid_node_id"
            )

        current_node_partition = current_partition[node_id].item()
        partition_size = (current_partition == current_node_partition).sum().item()

        # 2. 计算风险分数
        risk_score = 0.0

        # 高度数节点风险更高（可能是关键连接点）
        if node_degree > 3:
            risk_score += 0.3
        elif node_degree > 2:
            risk_score += 0.1

        # 小分区中的节点风险更高
        if partition_size <= 2:
            risk_score += 0.4
        elif partition_size <= 3:
            risk_score += 0.2

        # 3. 历史模式分析
        if node_id in self.history:
            recent_failures = sum(1 for result in self.history[node_id][-5:] if not result)
            if recent_failures > 2:
                risk_score += 0.3

        # 4. 确定风险等级和可用分区
        if risk_score < 0.2:
            risk_level = 'safe'
            # 安全级别：提供邻居分区作为可用选项
            available_partitions = self._get_neighbor_partitions(node_id, current_partition, hetero_data)
        elif risk_score < 0.4:
            risk_level = 'low'
            # 低风险：提供邻居分区
            available_partitions = self._get_neighbor_partitions(node_id, current_partition, hetero_data)
        elif risk_score < 0.7:
            risk_level = 'medium'
            # 中等风险：只提供当前分区的邻居分区
            available_partitions = self._get_neighbor_partitions(node_id, current_partition, hetero_data)
        else:
            risk_level = 'high'
            # 高风险：仍然提供选项，但标记为高风险
            available_partitions = self._get_neighbor_partitions(node_id, current_partition, hetero_data)

        # 确保至少有一个可用分区
        if not available_partitions:
            # 如果没有邻居分区，至少提供除当前分区外的其他分区
            num_partitions = int(current_partition.max().item())
            current_partition_id = current_node_partition
            available_partitions = [p for p in range(1, num_partitions + 1) if p != current_partition_id]
            if not available_partitions:
                available_partitions = list(range(1, num_partitions + 1))

        return RiskAssessmentResult(
            risk_level=risk_level,
            risk_score=risk_score,
            available_partitions=available_partitions,
            fallback_reason=None
        )

    def _get_neighbor_partitions(self, node_id: int, current_partition: torch.Tensor, hetero_data: Any) -> List[int]:
        """
        获取节点邻居所在的分区列表

        Args:
            node_id: 节点ID
            current_partition: 当前分区状态
            hetero_data: 图数据

        Returns:
            邻居分区ID列表
        """
        neighbor_partitions = set()

        try:
            if hetero_data and 'bus' in hetero_data:
                # 从异构图数据中获取边信息
                edge_data = None
                for edge_type in hetero_data.edge_types:
                    if edge_type[0] == 'bus' and edge_type[2] == 'bus':
                        edge_data = hetero_data[edge_type]
                        break

                if edge_data is not None and hasattr(edge_data, 'edge_index'):
                    edge_index = edge_data.edge_index
                    # 找到与node_id相连的所有节点
                    neighbors = []
                    # 出边
                    out_neighbors = edge_index[1][edge_index[0] == node_id]
                    # 入边
                    in_neighbors = edge_index[0][edge_index[1] == node_id]
                    neighbors = torch.cat([out_neighbors, in_neighbors]).unique()

                    # 获取邻居节点的分区
                    for neighbor in neighbors:
                        neighbor_id = neighbor.item()
                        if neighbor_id < len(current_partition):
                            partition_id = current_partition[neighbor_id].item()
                            if partition_id > 0:  # 有效分区
                                neighbor_partitions.add(partition_id)
        except Exception as e:
            logger.debug(f"Error getting neighbor partitions for node {node_id}: {e}")

        # 移除当前节点所在的分区
        current_node_partition = current_partition[node_id].item()
        neighbor_partitions.discard(current_node_partition)

        return list(neighbor_partitions)

        return RiskAssessmentResult(
            risk_level=risk_level,
            risk_score=risk_score,
            available_partitions=[],  # 将在后续填充
            fallback_reason=None
        )

    def _get_node_degree(self, node_id: int, hetero_data: Any) -> int:
        """获取节点度数"""
        try:
            edge_index = hetero_data['bus', 'connects_to', 'bus'].edge_index
            degree = (edge_index[0] == node_id).sum() + (edge_index[1] == node_id).sum()
            return degree.item()
        except:
            return 1  # 默认度数

    def update_history(self, node_id: int, success: bool):
        """更新节点历史决策结果"""
        self.history[node_id].append(success)
        # 只保留最近10次记录
        if len(self.history[node_id]) > 10:
            self.history[node_id] = self.history[node_id][-10:]


class EnhancedPPOAgent(PPOAgent):
    """
    增强的PPO智能体 - 支持动作嵌入
    
    继承自原PPOAgent，重写关键方法以支持双塔架构
    """
    
    def __init__(self,
                 state_dim: int,
                 num_partitions: int,
                 config: Dict[str, Any],
                 device: str = 'auto'):
        """
        初始化增强的PPO智能体

        Args:
            state_dim: 状态维度
            num_partitions: 初始分区数
            config: 配置字典，需包含:
                - node_dim: 节点特征维度
                - edge_dim: 边特征维度
                - gnn_hidden: GNN隐藏层维度
                - strategy_vector_dim: 策略向量维度
                - partition_encoder: 分区编码器配置
                - use_two_tower: 是否使用双塔架构
            device: 计算设备
        """
        # 【关键修复】使用实际的状态维度，而不是配置中的固定值
        # state_dim是从环境实际获取的节点嵌入维度，能够适应不同模式
        node_embedding_dim = state_dim  # 使用实际的节点嵌入维度

        # 区域嵌入维度 = 2 * 实际节点嵌入维度
        region_embedding_dim = 2 * node_embedding_dim
        
        # 处理设备参数
        if isinstance(device, str):
            if device == 'auto':
                device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device_obj = torch.device(device)
        else:
            device_obj = device
        
        # 调用父类初始化，传递正确的参数
        super().__init__(
            node_embedding_dim=node_embedding_dim,
            region_embedding_dim=region_embedding_dim,
            num_partitions=num_partitions,
            agent_config=config,  # 传递整个config作为agent_config
            device=device_obj
        )

        # 【新增】初始化预测性回退机制组件
        self.connectivity_cache = ConnectivityCache(
            max_size=config.get('cache_size', 1000),
            ttl_seconds=config.get('cache_ttl', 300)
        )
        self.risk_predictor = RiskPredictor(device_obj)

        # 【新增】自适应约束参数
        self.adaptive_constraint_config = config.get('adaptive_constraints', {
            'enabled': True,
            'initial_strictness': 0.8,
            'min_strictness': 0.3,
            'max_strictness': 1.0,
            'adjustment_rate': 0.05,
            'evaluation_window': 50,  # 评估窗口大小
            'success_threshold': 0.6,  # 成功率阈值
            'connectivity_threshold': 0.5  # 连通性阈值
        })
        self.current_constraint_strictness = self.adaptive_constraint_config['initial_strictness']

        # 【新增】性能监控组件
        self.performance_monitor = PerformanceMonitor(
            window_size=self.adaptive_constraint_config['evaluation_window']
        )
        
        # 检查是否启用双塔架构（父类已经创建了双塔架构）
        self.use_two_tower = config.get('use_two_tower', True)
        self.num_partitions = num_partitions

        # 【调试】打印双塔架构信息
        if self.use_two_tower and hasattr(self, 'actor') and hasattr(self, 'partition_encoder'):
            print(f"🏗️ EnhancedPPOAgent双塔架构配置:")
            print(f"   ✅ State Tower (Actor): {type(self.actor).__name__}")
            print(f"   ✅ Action Tower (PartitionEncoder): {type(self.partition_encoder).__name__}")
            print(f"   📏 Strategy Vector Dim: {self.strategy_vector_dim}")
            print(f"   🔗 Node Embedding Dim: {self.node_embedding_dim}")
            print(f"   🔗 Region Embedding Dim: {self.region_embedding_dim}")
            print(f"   🎯 Partitions: {self.num_partitions}")
            logger.info("EnhancedPPOAgent using two-tower architecture from parent class")
        else:
            print(f"⚠️  EnhancedPPOAgent标准架构配置:")
            print(f"   node_embedding_dim: {self.node_embedding_dim}")
            print(f"   region_embedding_dim: {self.region_embedding_dim}")
            print(f"   state_dim: {state_dim}")
            print(f"   num_partitions: {self.num_partitions}")
            logger.info("EnhancedPPOAgent using standard architecture")
    
    def select_action(self,
                     state: Dict[str, torch.Tensor],
                     training: bool = True,
                     return_embeddings: bool = False) -> Tuple[Optional[Tuple[int, int]], float, float, Optional[Dict]]:
        """
        使用动作嵌入机制选择动作
        
        Args:
            state: 状态字典，需包含:
                - available_partitions: 每个节点的可用分区信息
                - partition_features: 分区特征信息
            training: 是否处于训练模式
            return_embeddings: 是否返回嵌入向量（用于调试）
        
        Returns:
            action: 选择的动作 (node_idx, partition_idx)
            log_prob: 动作的对数概率
            value: 状态价值估计
            embeddings: 嵌入信息（如果return_embeddings=True）
        """
        # 增强智能体始终使用双塔架构
        
        with torch.no_grad():
            # 1. 准备批处理状态
            batched_state = self._prepare_batched_state([state])
            
            # 2. 获取价值估计
            value = self.critic(batched_state).item()
            
            # 3. 检查边界节点
            if batched_state.boundary_nodes.numel() == 0:
                return None, 0.0, value, None
            
            # 4. 获取节点logits和策略向量
            node_logits, strategy_vectors = self.actor(batched_state)
            
            # 5. 节点选择
            # 获取每个边界节点是否有有效分区
            boundary_mask = batched_state.action_mask[batched_state.boundary_nodes]
            valid_node_mask = boundary_mask.any(dim=1)
            node_logits = node_logits.masked_fill(~valid_node_mask, -1e9)
            
            if training:
                node_probs = F.softmax(node_logits, dim=0)
                node_dist = torch.distributions.Categorical(probs=node_probs)
                node_action_idx = node_dist.sample()
            else:
                node_action_idx = torch.argmax(node_logits)
            
            # 6. 获取选中节点的策略向量
            selected_strategy_vector = strategy_vectors[node_action_idx]
            selected_node = batched_state.boundary_nodes[node_action_idx].item()
            
            # 7. 获取该节点的可用分区信息（四层智能约束处理）
            risk_result = self._get_available_partitions_with_risk_assessment(
                state, selected_node, batched_state
            )
            available_partitions = risk_result.available_partitions
            
            # 8. 编码可用分区
            partition_embeddings = self._encode_partitions(
                state, available_partitions
            )
            
            # 9. 计算相似度并选择分区
            similarities = self.partition_encoder.compute_similarity(
                selected_strategy_vector,
                partition_embeddings,
                temperature=1.0  # 可以从config中获取
            )
            
            # 10. 基于相似度选择分区
            if training:
                partition_probs = F.softmax(similarities, dim=0)
                partition_dist = torch.distributions.Categorical(probs=partition_probs)
                partition_idx = partition_dist.sample()
                
                # 计算总的log_prob
                node_log_prob = node_dist.log_prob(node_action_idx)
                partition_log_prob = partition_dist.log_prob(partition_idx)
                log_prob = (node_log_prob + partition_log_prob).item()
            else:
                partition_idx = torch.argmax(similarities)
                log_prob = 0.0
            
            # 11. 转换为实际的分区ID
            selected_partition = available_partitions[partition_idx.item()]

            # 确保分区ID是整数而不是张量
            if isinstance(selected_partition, torch.Tensor):
                selected_partition = selected_partition.item()

            action = (selected_node, selected_partition)
            
            # 准备返回的嵌入信息
            embeddings = None
            if return_embeddings:
                embeddings = {
                    'strategy_vector': selected_strategy_vector.cpu().numpy(),
                    'partition_embeddings': partition_embeddings.cpu().numpy(),
                    'similarities': similarities.cpu().numpy(),
                    'node_logits': node_logits.cpu().numpy()
                }
            
            return action, log_prob, value, embeddings
    
    def _get_available_partitions(self, 
                                 state: Dict,
                                 node_id: int,
                                 batched_state: BatchedState) -> List[int]:
        """
        获取节点的可用分区列表（确保连通性）
        
        Args:
            state: 原始状态字典
            node_id: 节点ID
            batched_state: 批处理状态
        
        Returns:
            可用分区ID列表
        """
        # 从环境提供的信息中获取
        if 'connectivity_safe_partitions' in state:
            node_safe_partitions = state['connectivity_safe_partitions'].get(node_id, [])
            return node_safe_partitions
        
        # 使用action_mask
        if hasattr(batched_state, 'action_mask'):
            # 找到节点在boundary_nodes中的位置
            node_positions = (batched_state.boundary_nodes == node_id).nonzero(as_tuple=True)[0]
            if len(node_positions) > 0:
                node_pos = node_positions[0]
                # 修复：直接使用node_id作为action_mask的索引，而不是boundary_nodes[node_pos]
                # 因为action_mask的第一维是按节点ID排列的，不是按boundary_nodes位置排列的
                if node_id < batched_state.action_mask.size(0):
                    mask = batched_state.action_mask[node_id]
                    available = torch.where(mask)[0].cpu().tolist()
                    return [p + 1 for p in available]  # 转换为1-indexed
                else:
                    logger.warning(f"节点ID {node_id} 超出action_mask范围 {batched_state.action_mask.size(0)}")
                    return []
        
        # 默认返回所有分区
        return list(range(1, self.num_partitions + 1))

    def _get_action_mask_partitions(self,
                                   node_id: int,
                                   batched_state: BatchedState) -> List[int]:
        """
        使用action_mask获取节点的可用分区（与原始PPOAgent一致）

        Args:
            node_id: 节点ID
            batched_state: 批处理状态

        Returns:
            可用分区ID列表
        """
        if hasattr(batched_state, 'action_mask') and batched_state.action_mask is not None:
            # 检查action_mask的维度
            if batched_state.action_mask.dim() == 2:
                # 二维mask: [num_nodes, num_partitions]
                if node_id < batched_state.action_mask.size(0):
                    mask = batched_state.action_mask[node_id]
                    available = torch.where(mask)[0].cpu().tolist()
                    return [p + 1 for p in available]  # 转换为1-indexed
            else:
                # 一维mask或其他格式，尝试找到节点在boundary_nodes中的位置
                node_positions = (batched_state.boundary_nodes == node_id).nonzero(as_tuple=True)[0]
                if len(node_positions) > 0:
                    node_pos = node_positions[0]
                    # 获取该节点的action_mask
                    if node_pos < batched_state.action_mask.size(0):
                        mask = batched_state.action_mask[node_pos]
                        available = torch.where(mask)[0].cpu().tolist()
                        return [p + 1 for p in available]  # 转换为1-indexed

        # 如果action_mask不可用，返回默认分区（除当前分区外）
        if hasattr(batched_state, 'partition_assignment') and batched_state.partition_assignment is not None:
            current_partition = batched_state.partition_assignment[node_id].item()
            num_partitions = int(batched_state.partition_assignment.max().item())
            return [p for p in range(1, num_partitions + 1) if p != current_partition]

        # 最后的回退：返回默认分区
        return [1, 2, 3]

    def _get_available_partitions_with_risk_assessment(self,
                                                      state: Dict,
                                                      node_id: int,
                                                      batched_state: BatchedState) -> RiskAssessmentResult:
        """
        四层智能约束处理机制

        层次结构：
        1. 连通性安全分区（最严格，避免割点问题）
        2. 风险评估层（基于预测的中等风险分区）
        3. Action Mask回退（与原始PPOAgent一致）
        4. 软约束模式（确保系统不卡死）

        Args:
            state: 原始状态字典
            node_id: 节点ID
            batched_state: 批处理状态

        Returns:
            风险评估结果，包含可用分区和风险信息
        """
        # 第一层：连通性安全分区（最优选择）
        safe_partitions = self._get_connectivity_safe_partitions(state, node_id, batched_state)

        if safe_partitions:
            return RiskAssessmentResult(
                risk_level='safe',
                risk_score=0.0,
                available_partitions=safe_partitions,
                fallback_reason=None,
                cache_hit=False
            )

        # 第二层：风险评估层（智能预测）
        risk_result = self.risk_predictor.predict_risk(
            node_id,
            batched_state.partition_assignment if hasattr(batched_state, 'partition_assignment') else state.get('partition_assignment'),
            batched_state.boundary_nodes,
            state.get('hetero_data')
        )

        # 基于风险等级获取分区
        if risk_result.risk_level in ['safe', 'low']:
            # 低风险：使用缓存优化的连通性检查
            risk_assessed_partitions = self._get_risk_assessed_partitions(
                state, node_id, batched_state, risk_result.risk_level
            )
            if risk_assessed_partitions:
                risk_result.available_partitions = risk_assessed_partitions
                risk_result.fallback_reason = f"risk_assessment_{risk_result.risk_level}"
                return risk_result

        # 第三层：Action Mask回退（与原始PPOAgent一致）
        logger.warning(f"No safe/low-risk partitions for node {node_id}, falling back to action_mask")
        action_mask_partitions = self._get_action_mask_partitions(node_id, batched_state)

        if action_mask_partitions:
            return RiskAssessmentResult(
                risk_level='medium',
                risk_score=0.6,
                available_partitions=action_mask_partitions,
                fallback_reason="action_mask_fallback",
                cache_hit=False
            )

        # 第四层：软约束模式（最后保障）
        logger.warning(f"All fallbacks failed for node {node_id}, using soft constraint mode")
        all_partitions = list(range(1, self.num_partitions + 1))

        return RiskAssessmentResult(
            risk_level='high',
            risk_score=0.9,
            available_partitions=all_partitions,
            fallback_reason="soft_constraint_fallback",
            cache_hit=False
        )

    def _get_connectivity_safe_partitions(self,
                                        state: Dict,
                                        node_id: int,
                                        batched_state: BatchedState) -> List[int]:
        """获取连通性安全的分区（第一层）"""
        # 优先使用环境提供的安全分区信息
        if 'connectivity_safe_partitions' in state:
            safe_partitions = state['connectivity_safe_partitions'].get(node_id, [])
            if safe_partitions:
                return safe_partitions

        # 如果环境没有提供，尝试从环境对象直接获取
        if hasattr(state, 'env') and hasattr(state['env'], 'get_connectivity_safe_partitions'):
            try:
                safe_partitions = state['env'].get_connectivity_safe_partitions(node_id)
                if safe_partitions:
                    return safe_partitions
            except Exception as e:
                logger.debug(f"Failed to get safe partitions from env: {e}")

        # 如果没有预计算的安全分区，返回空列表
        # 这将触发风险评估层
        return []

    def _get_risk_assessed_partitions(self,
                                    state: Dict,
                                    node_id: int,
                                    batched_state: BatchedState,
                                    risk_level: str) -> List[int]:
        """
        基于风险评估获取分区（第二层）
        使用缓存优化的连通性检查
        """
        available_partitions = []

        # 获取基础可用分区（来自action_mask）
        base_partitions = self._get_action_mask_partitions(node_id, batched_state)

        if not base_partitions:
            return []

        # 根据风险等级决定检查策略
        if risk_level == 'safe':
            # 安全级别：检查所有分区
            partitions_to_check = base_partitions
        else:  # 'low'
            # 低风险级别：只检查前几个分区（性能优化）
            partitions_to_check = base_partitions[:min(3, len(base_partitions))]

        # 使用缓存优化的连通性检查
        current_partition = (batched_state.partition_assignment
                           if hasattr(batched_state, 'partition_assignment')
                           else state.get('partition_assignment'))

        for target_partition in partitions_to_check:
            # 检查缓存
            cached_result = self.connectivity_cache.get(current_partition, node_id, target_partition)

            if cached_result is not None:
                # 缓存命中
                if not cached_result.get('violates_connectivity', True):
                    available_partitions.append(target_partition)
            else:
                # 缓存未命中，进行实际连通性检查
                violation_result = self._perform_connectivity_check(
                    current_partition, node_id, target_partition, state
                )

                # 缓存结果
                self.connectivity_cache.put(
                    current_partition,
                    node_id,
                    target_partition,
                    violation_result
                )

                # 如果不违反连通性，添加到可用分区
                if not violation_result.get('violates_connectivity', True):
                    available_partitions.append(target_partition)

        return available_partitions

    def update_constraint_strictness(self, episode_info: Dict[str, Any] = None):
        """
        智能自适应约束调节机制

        Args:
            episode_info: 单个episode的信息（可选，用于记录）
        """
        if not self.adaptive_constraint_config['enabled']:
            return

        # 记录episode数据
        if episode_info:
            self.performance_monitor.record_episode(episode_info)

        # 检查是否需要调整
        should_adjust, reason = self.performance_monitor.should_adjust_constraints(
            self.adaptive_constraint_config['success_threshold'],
            self.adaptive_constraint_config['connectivity_threshold']
        )

        if not should_adjust:
            return

        # 获取当前性能指标
        metrics = self.performance_monitor.get_performance_metrics()
        old_strictness = self.current_constraint_strictness
        adjustment_rate = self.adaptive_constraint_config['adjustment_rate']

        # 根据原因调整约束严格程度
        if reason == "relax_low_success":
            # 成功率过低，大幅放松约束
            self.current_constraint_strictness -= adjustment_rate * 2.0
            logger.info(f"🔧 放松约束 (成功率过低: {metrics['success_rate']:.2f})")

        elif reason == "relax_high_violation":
            # 违规率过高，中等放松约束
            self.current_constraint_strictness -= adjustment_rate * 1.5
            logger.info(f"🔧 放松约束 (违规率过高: {metrics['constraint_violation_rate']:.2f})")

        elif reason == "tighten_low_connectivity":
            # 连通性过低，中等加强约束
            self.current_constraint_strictness += adjustment_rate * 1.5
            logger.info(f"🔧 加强约束 (连通性过低: {metrics['avg_connectivity']:.2f})")

        elif reason == "tighten_good_performance":
            # 表现良好，小幅加强约束
            self.current_constraint_strictness += adjustment_rate * 0.5
            logger.info(f"🔧 渐进加强约束 (表现良好)")

        # 限制在合理范围内
        self.current_constraint_strictness = max(
            self.adaptive_constraint_config['min_strictness'],
            min(self.adaptive_constraint_config['max_strictness'], self.current_constraint_strictness)
        )

        # 记录调整信息
        if abs(self.current_constraint_strictness - old_strictness) > 0.01:
            logger.info(f"📊 约束严格程度: {old_strictness:.3f} → {self.current_constraint_strictness:.3f}")
            logger.debug(f"   原因: {reason}")
            logger.debug(f"   性能指标: 成功率={metrics['success_rate']:.2f}, "
                        f"连通性={metrics['avg_connectivity']:.2f}, "
                        f"违规率={metrics['constraint_violation_rate']:.2f}")

    def get_adaptive_constraint_info(self) -> Dict[str, Any]:
        """获取自适应约束的当前状态信息"""
        metrics = self.performance_monitor.get_performance_metrics()

        return {
            'current_strictness': self.current_constraint_strictness,
            'config': self.adaptive_constraint_config,
            'performance_metrics': metrics,
            'total_episodes': self.performance_monitor.current_episode,
            'recent_episodes': len(self.performance_monitor.episode_data)
        }

    def _perform_connectivity_check(self,
                                  current_partition: torch.Tensor,
                                  node_id: int,
                                  target_partition: int,
                                  state: Dict) -> Dict[str, Any]:
        """
        执行实际的连通性检查

        Args:
            current_partition: 当前分区状态
            node_id: 要移动的节点ID
            target_partition: 目标分区ID
            state: 状态字典，包含图结构信息

        Returns:
            连通性检查结果字典
        """
        try:
            # 尝试使用环境的action_space进行连通性检查
            if 'env' in state and hasattr(state['env'], 'action_space'):
                env = state['env']
                if hasattr(env.action_space, 'mask_handler'):
                    action = (node_id, target_partition)
                    return env.action_space.mask_handler.check_connectivity_violation(
                        current_partition, action
                    )

            # 回退到简化的连通性检查
            return self._simplified_connectivity_check(
                current_partition, node_id, target_partition, state
            )

        except Exception as e:
            logger.warning(f"连通性检查失败: {e}")
            # 失败时假设违反连通性（保守策略）
            return {
                'violates_connectivity': True,
                'violation_severity': 0.5,
                'isolated_nodes_count': 1,
                'affected_partition_size': 1
            }

    def _simplified_connectivity_check(self,
                                     current_partition: torch.Tensor,
                                     node_id: int,
                                     target_partition: int,
                                     state: Dict) -> Dict[str, Any]:
        """
        简化的连通性检查（当无法访问完整的mask_handler时使用）

        基于启发式规则：
        1. 如果原分区只有1-2个节点，移动是安全的
        2. 如果节点度数很低，移动风险较小
        3. 其他情况保守处理
        """
        current_node_partition = current_partition[node_id].item()

        # 计算原分区大小
        partition_size = (current_partition == current_node_partition).sum().item()

        # 小分区移动通常是安全的
        if partition_size <= 2:
            return {
                'violates_connectivity': False,
                'violation_severity': 0.0,
                'isolated_nodes_count': 0,
                'affected_partition_size': partition_size
            }

        # 获取节点度数（如果可能）
        node_degree = 1  # 默认度数
        if 'hetero_data' in state:
            try:
                hetero_data = state['hetero_data']
                edge_index = hetero_data['bus', 'connects_to', 'bus'].edge_index
                node_degree = (edge_index[0] == node_id).sum() + (edge_index[1] == node_id).sum()
                node_degree = node_degree.item()
            except:
                pass

        # 基于度数和分区大小的启发式判断
        if node_degree <= 2 and partition_size >= 4:
            # 低度数节点在大分区中移动相对安全
            violation_severity = 0.2
        elif node_degree >= 4 and partition_size <= 4:
            # 高度数节点在小分区中移动风险较高
            violation_severity = 0.8
        else:
            # 中等风险
            violation_severity = 0.4

        return {
            'violates_connectivity': violation_severity > 0.5,
            'violation_severity': violation_severity,
            'isolated_nodes_count': 1 if violation_severity > 0.7 else 0,
            'affected_partition_size': partition_size
        }

    def _prepare_constraint_aware_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        为约束感知策略向量生成准备增强状态

        这个方法为后续的方案1实现预留接口，将约束信息融入状态编码

        Args:
            state: 原始状态字典

        Returns:
            增强的状态字典，包含约束感知信息
        """
        enhanced_state = state.copy()

        # 【扩展点1】添加约束严格程度信息
        enhanced_state['constraint_strictness'] = torch.tensor(
            self.current_constraint_strictness,
            device=self.device,
            dtype=torch.float32
        )

        # 【扩展点2】添加历史违规信息
        if hasattr(self, 'performance_monitor'):
            metrics = self.performance_monitor.get_performance_metrics()
            enhanced_state['violation_rate'] = torch.tensor(
                metrics.get('constraint_violation_rate', 0.0),
                device=self.device,
                dtype=torch.float32
            )
            enhanced_state['success_rate'] = torch.tensor(
                metrics.get('success_rate', 0.0),
                device=self.device,
                dtype=torch.float32
            )

        # 【扩展点3】添加连通性风险预测
        if 'boundary_nodes' in state:
            boundary_nodes = state['boundary_nodes']
            risk_scores = []

            for node in boundary_nodes:
                if torch.is_tensor(node):
                    node_id = node.item()
                else:
                    node_id = int(node)

                # 使用风险预测器计算风险分数
                risk_result = self.risk_predictor.predict_risk(
                    node_id,
                    state.get('partition_assignment', torch.zeros(1)),
                    boundary_nodes,
                    state.get('hetero_data')
                )
                risk_scores.append(risk_result.risk_score)

            enhanced_state['node_risk_scores'] = torch.tensor(
                risk_scores, device=self.device, dtype=torch.float32
            )

        return enhanced_state

    def _generate_constraint_aware_strategy_vector(self,
                                                  node_features: torch.Tensor,
                                                  constraint_context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        生成约束感知的策略向量（方案1的核心接口）

        这个方法为后续实现约束感知策略向量生成预留接口

        Args:
            node_features: 节点特征
            constraint_context: 约束上下文信息

        Returns:
            约束感知的策略向量
        """
        # 【当前实现】使用现有的策略生成器
        # 【未来扩展】这里将融入约束信息

        # 提取约束特征
        constraint_features = []

        if 'constraint_strictness' in constraint_context:
            constraint_features.append(constraint_context['constraint_strictness'])

        if 'violation_rate' in constraint_context:
            constraint_features.append(constraint_context['violation_rate'])

        if 'success_rate' in constraint_context:
            constraint_features.append(constraint_context['success_rate'])

        # 如果有约束特征，将其与节点特征结合
        if constraint_features:
            constraint_tensor = torch.stack(constraint_features).unsqueeze(0)
            # 扩展约束特征以匹配节点特征的批次维度
            if node_features.dim() > 1:
                constraint_tensor = constraint_tensor.expand(node_features.size(0), -1)

            # 约束感知特征融合：将约束信息与节点特征结合
            enhanced_features = torch.cat([node_features, constraint_tensor], dim=-1)
        else:
            enhanced_features = node_features

        # 约束感知策略生成
        # 当前使用现有的策略生成器（需要适配输入维度）
        return enhanced_features

    def get_constraint_aware_capabilities(self) -> Dict[str, bool]:
        """
        获取约束感知能力状态

        Returns:
            能力状态字典
        """
        return {
            'predictive_fallback': True,
            'connectivity_cache': True,
            'adaptive_constraints': self.adaptive_constraint_config['enabled'],
            'risk_prediction': True,
            'constraint_aware_state': True,  # 架构准备完成
            'constraint_aware_strategy': False,  # 待实现（方案1）
            'performance_monitoring': hasattr(self, 'performance_monitor')
        }
    
    def _encode_partitions(self, 
                          state: Dict,
                          partition_ids: List[int]) -> torch.Tensor:
        """
        编码分区特征
        
        Args:
            state: 状态字典，需包含partition_info
            partition_ids: 要编码的分区ID列表
        
        Returns:
            分区嵌入矩阵 [num_partitions, embedding_dim]
        """
        # 从状态中获取分区信息
        partition_info = state.get('partition_info', {})
        grid_info = state.get('grid_info', {})
        
        # 准备分区特征
        partitions_data = []
        for pid in partition_ids:
            if pid in partition_info:
                partitions_data.append(partition_info[pid])
            else:
                # 创建默认特征
                partitions_data.append({
                    'nodes': [],
                    'load': 0.0,
                    'generation': 0.0,
                    'internal_edges': 0,
                    'boundary_nodes': 0
                })
        
        # 使用分区编码器编码
        embeddings = self.partition_encoder.encode_partitions(
            partitions_data,
            grid_info,
            self.device
        )
        
        return embeddings
    
    def update_partition_cache(self, 
                             partition_changes: Dict[int, List[int]]):
        """
        更新分区缓存（当分区发生变化时）
        
        Args:
            partition_changes: 变化的分区 {partition_id: affected_nodes}
        """
        if hasattr(self.partition_encoder, 'invalidate_cache'):
            self.partition_encoder.invalidate_cache(list(partition_changes.keys()))
    
    def get_strategy_vector(self, 
                           state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        获取当前状态的策略向量（用于分析）
        
        Args:
            state: 状态观察
        
        Returns:
            所有边界节点的策略向量 [num_boundary_nodes, strategy_vector_dim]
        """
        if not self.use_two_tower:
            raise ValueError("Strategy vector only available in two-tower mode")
        
        with torch.no_grad():
            batched_state = self._prepare_batched_state([state])
            _, strategy_vectors = self.actor(batched_state)
            return strategy_vectors


def create_enhanced_agent(state_dim: int,
                         num_partitions: int,
                         config: Dict[str, Any],
                         device: str = 'auto') -> EnhancedPPOAgent:
    """
    工厂函数：创建增强PPO智能体

    Args:
        state_dim: 状态维度
        num_partitions: 分区数
        config: 配置字典
        device: 设备

    Returns:
        增强PPO智能体实例
    """
    return EnhancedPPOAgent(state_dim, num_partitions, config, device)