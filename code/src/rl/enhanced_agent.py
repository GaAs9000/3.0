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

from .agent import PPOAgent, BatchedState
from models.partition_encoder import PartitionEncoder, create_partition_encoder
from models.actor_network import EnhancedActorNetwork, create_actor_network

logger = logging.getLogger(__name__)


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
        # 【修复】从配置中获取正确的维度信息，而不是从state_dim推断
        # state_dim是状态向量的维度，不等于嵌入维度
        node_embedding_dim = config.get('node_embedding_dim', 128)  # 节点嵌入维度
        region_embedding_dim = config.get('region_embedding_dim', 256)  # 区域嵌入维度
        
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
        
        # 检查是否启用双塔架构
        self.use_two_tower = config.get('use_two_tower', True)

        # 【调试】打印维度信息
        print(f"🔧 EnhancedPPOAgent维度配置:")
        print(f"   node_embedding_dim: {self.node_embedding_dim}")
        print(f"   region_embedding_dim: {self.region_embedding_dim}")
        print(f"   state_dim: {state_dim}")
        print(f"   num_partitions: {self.num_partitions}")
        
        if self.use_two_tower:
            # 策略向量维度
            self.strategy_vector_dim = config.get('strategy_vector_dim', 128)
            
            # 替换actor网络为增强版本
            self.actor = create_actor_network({
                'node_embedding_dim': self.node_embedding_dim,
                'region_embedding_dim': self.region_embedding_dim,
                'strategy_vector_dim': self.strategy_vector_dim,
                'hidden_dim': config.get('actor_hidden_dim', 256),
                'dropout': config.get('dropout', 0.1),

            }, use_two_tower=True).to(self.device)
            
            # 创建分区编码器
            partition_encoder_config = config.get('partition_encoder', {})
            partition_encoder_config['embedding_dim'] = self.strategy_vector_dim
            self.partition_encoder = create_partition_encoder(
                partition_encoder_config
            ).to(self.device)
            
            # 重新初始化优化器以包含新参数
            # 保持与父类一致的优化器命名
            lr = config.get('lr', 3e-4)
            self.actor_optimizer = torch.optim.Adam([
                {'params': self.actor.parameters()},
                {'params': self.partition_encoder.parameters()}
            ], lr=lr)
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=lr
            )
            
            logger.info("EnhancedPPOAgent initialized with two-tower architecture")
        else:
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
            
            # 7. 获取该节点的可用分区信息
            available_partitions = self._get_available_partitions(
                state, selected_node, batched_state
            )
            
            if not available_partitions:
                # 如果没有连通性安全的分区，使用软约束回退
                logger.warning(f"No available partitions for node {selected_node}")
                # 回退到所有相邻分区（软约束模式）
                available_partitions = self._get_fallback_partitions(
                    state, selected_node, batched_state
                )

                if not available_partitions:
                    # 如果仍然没有分区，返回None
                    return None, 0.0, value, None
            
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