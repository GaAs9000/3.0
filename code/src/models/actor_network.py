#!/usr/bin/env python3
"""
Enhanced Actor Network with Two-Tower Architecture

重构的Actor网络，实现双塔架构：
- 状态塔：输出固定维度的策略向量
- 动作塔：由PartitionEncoder实现（外部组件）

关键改进：
- 解耦K值依赖
- 支持动作嵌入机制
- 保持与现有接口的兼容性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    计算分组平均（scatter mean）

    Args:
        src: 源张量
        index: 分组索引
        dim: 聚合维度

    Returns:
        分组平均结果
    """
    if src.numel() == 0:
        return torch.zeros(0, *src.shape[1:], dtype=src.dtype, device=src.device)

    dim_size = int(index.max()) + 1
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)

    # 确保index的形状与src的第一个维度匹配
    if index.dim() != 1 or index.size(0) != src.size(0):
        print(f"DEBUG: src shape: {src.shape}, index shape: {index.shape}")
        print(f"DEBUG: src device: {src.device}, index device: {index.device}")
        print(f"DEBUG: src dtype: {src.dtype}, index dtype: {index.dtype}")
        raise ValueError(f"Index shape {index.shape} doesn't match src first dimension {src.size(0)}")

    ones = torch.ones(index.size(0), dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.scatter_add_(0, index, ones)

    # 扩展索引张量以匹配源张量的维度
    if src.dim() > 1:
        # 将1维索引扩展为与src相同的维度
        index_expanded = index.unsqueeze(-1).expand(-1, src.size(1))
        out.scatter_add_(0, index_expanded, src)
    else:
        out.scatter_add_(0, index, src)
    count = count.clamp(min=1)
    count = count.view(-1, *([1] * (src.dim() - 1)))
    return out / count


class EnhancedActorNetwork(nn.Module):
    """
    增强的Actor网络 - 状态塔实现
    
    不再直接输出分区概率，而是输出策略向量，
    用于与分区嵌入进行相似度匹配。
    """
    
    def __init__(self,
                 node_embedding_dim: int,
                 region_embedding_dim: int,
                 strategy_vector_dim: int = 128,  # 替代num_partitions
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        """
        Args:
            node_embedding_dim: 节点嵌入维度
            region_embedding_dim: 区域嵌入维度
            strategy_vector_dim: 策略向量维度（需与PartitionEncoder的embedding_dim一致）
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
            use_layer_norm: 是否使用层归一化
        """
        super().__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.region_embedding_dim = region_embedding_dim
        self.strategy_vector_dim = strategy_vector_dim
        self.hidden_dim = hidden_dim
        
        # 节点选择网络（保持不变）
        self.node_selector = nn.Sequential(
            nn.Linear(node_embedding_dim + region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 节点评分
        )
        
        # 策略生成器（替代原partition_selector）
        # 输出固定维度的策略向量，而非K个分区的概率
        layers = [
            nn.Linear(node_embedding_dim + region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, strategy_vector_dim)
        ]
        
        # 可选的输出归一化
        if use_layer_norm:
            layers.append(nn.LayerNorm(strategy_vector_dim))
            
        self.strategy_generator = nn.Sequential(*layers)
        
        # 全局状态编码器（保持不变）
        self.global_encoder = nn.Sequential(
            nn.Linear(region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, region_embedding_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"EnhancedActorNetwork initialized: "
                   f"strategy_vector_dim={strategy_vector_dim}, "
                   f"hidden_dim={hidden_dim}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, batched_state):
        """
        前向传播
        
        Args:
            batched_state: 批处理状态（与原接口兼容）
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - node_logits: 边界节点选择分数 [num_boundary_nodes]
                - strategy_vectors: 每个边界节点的策略向量 [num_boundary_nodes, strategy_vector_dim]
        """
        bs = batched_state
        
        # 空批次保护
        if bs.region_embeddings.numel() == 0 or bs.boundary_nodes.numel() == 0:
            return (torch.zeros(0, device=bs.node_embeddings.device),
                    torch.zeros(0, self.strategy_vector_dim, device=bs.node_embeddings.device))
        
        # 计算批次级全局上下文
        global_context_per_batch = self.global_encoder(
            scatter_mean(bs.region_embeddings, bs.region_batch_idx, dim=0)
        )  # [batch_size, region_dim]
        
        # 获取边界节点的批次索引和上下文
        boundary_batch_idx = bs.node_batch_idx[bs.boundary_nodes]
        global_context = global_context_per_batch[boundary_batch_idx]
        
        # 获取边界节点嵌入
        boundary_embeddings = bs.node_embeddings[bs.boundary_nodes]
        
        # 组合特征
        combined_features = torch.cat([boundary_embeddings, global_context], dim=1)
        
        # 计算节点选择logits
        node_logits = self.node_selector(combined_features).squeeze(-1)
        
        # 生成策略向量（替代partition_logits）
        strategy_vectors = self.strategy_generator(combined_features)
        
        return node_logits, strategy_vectors
    
    def get_node_probabilities(self, node_logits: torch.Tensor, 
                              temperature: float = 1.0) -> torch.Tensor:
        """
        计算节点选择概率
        
        Args:
            node_logits: 节点logits
            temperature: 温度参数
        
        Returns:
            节点选择概率分布
        """
        return F.softmax(node_logits / temperature, dim=0)
    
    def compute_action_probabilities(self, 
                                   strategy_vector: torch.Tensor,
                                   partition_embeddings: torch.Tensor,
                                   action_mask: torch.Tensor,
                                   temperature: float = 1.0) -> torch.Tensor:
        """
        基于策略向量和分区嵌入计算动作概率
        
        Args:
            strategy_vector: 策略向量 [strategy_vector_dim]
            partition_embeddings: 分区嵌入 [num_partitions, embedding_dim]
            action_mask: 动作掩码 [num_partitions]
            temperature: 温度参数
        
        Returns:
            动作概率分布 [num_partitions]
        """
        # 计算相似度（点积）
        similarities = torch.matmul(strategy_vector, partition_embeddings.t())
        
        # 应用温度缩放
        logits = similarities / temperature
        
        # 应用掩码
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        
        # 计算概率
        return F.softmax(masked_logits, dim=-1)


class BackwardCompatibleActorNetwork(EnhancedActorNetwork):
    """
    向后兼容的Actor网络
    
    提供与原ActorNetwork相同的接口，
    但内部使用双塔架构。
    """
    
    def __init__(self, 
                 node_embedding_dim: int,
                 region_embedding_dim: int,
                 num_partitions: int,  # 保留以兼容
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 strategy_vector_dim: int = 128):
        """
        保持原有接口，但添加strategy_vector_dim参数
        """
        super().__init__(
            node_embedding_dim=node_embedding_dim,
            region_embedding_dim=region_embedding_dim,
            strategy_vector_dim=strategy_vector_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.num_partitions = num_partitions
        
        # 添加一个简单的线性投影层，用于兼容模式
        # 将策略向量投影到num_partitions维
        self.compatibility_projection = nn.Linear(strategy_vector_dim, num_partitions)
        
        logger.info(f"BackwardCompatibleActorNetwork: num_partitions={num_partitions}")
    
    def forward_compatible(self, batched_state):
        """
        兼容模式的前向传播
        
        返回与原ActorNetwork相同格式的输出
        """
        bs = batched_state
        
        # 获取基础输出
        node_logits, strategy_vectors = super().forward(batched_state)
        
        if strategy_vectors.size(0) == 0:
            return (node_logits, 
                    torch.zeros(0, self.num_partitions, device=node_logits.device))
        
        # 将策略向量投影到partition空间（仅用于兼容）
        partition_logits = self.compatibility_projection(strategy_vectors)
        
        # 应用动作掩码
        if hasattr(bs, 'action_mask') and bs.action_mask is not None:
            boundary_mask = bs.action_mask[bs.boundary_nodes]
            partition_logits = partition_logits.masked_fill(~boundary_mask, -1e9)
        
        return node_logits, partition_logits


def create_actor_network(config: Dict, use_two_tower: bool = True) -> nn.Module:
    """
    工厂函数：创建Actor网络
    
    Args:
        config: 配置字典
        use_two_tower: 是否使用双塔架构
    
    Returns:
        Actor网络实例
    """
    if use_two_tower:
        if config.get('backward_compatible', False):
            return BackwardCompatibleActorNetwork(
                node_embedding_dim=config['node_embedding_dim'],
                region_embedding_dim=config['region_embedding_dim'],
                num_partitions=config.get('num_partitions', 3),
                hidden_dim=config.get('hidden_dim', 256),
                dropout=config.get('dropout', 0.1),
                strategy_vector_dim=config.get('strategy_vector_dim', 128)
            )
        else:
            return EnhancedActorNetwork(
                node_embedding_dim=config['node_embedding_dim'],
                region_embedding_dim=config['region_embedding_dim'],
                strategy_vector_dim=config.get('strategy_vector_dim', 128),
                hidden_dim=config.get('hidden_dim', 256),
                dropout=config.get('dropout', 0.1)
            )
    else:
        # [已废弃] 不再支持旧的ActorNetwork
        # 如果需要兼容模式，使用backward_compatible=True的EnhancedActorNetwork
        if config.get('backward_compatible', False):
            return EnhancedActorNetwork(
                node_embedding_dim=config['node_embedding_dim'],
                region_embedding_dim=config['region_embedding_dim'],
                strategy_vector_dim=config.get('strategy_vector_dim', 128),
                hidden_dim=config.get('hidden_dim', 256),
                dropout=config.get('dropout', 0.1)
            )
        else:
            raise ValueError("Old ActorNetwork is deprecated. Use use_two_tower=True or set backward_compatible=True")