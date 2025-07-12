#!/usr/bin/env python3
"""
Partition Encoder - Action Tower for Two-Tower Architecture

生产级的分区特征编码器，用于将分区的物理和拓扑特征
编码为固定维度的嵌入向量，支持动作嵌入机制。

关键特性：
- 标准化的分区特征（6个核心物理量）
- 层归一化处理特征尺度差异
- 支持批量编码和缓存
- 与策略向量维度对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PartitionFeatures:
    """分区特征数据类"""
    size_ratio: float          # 分区节点数 / 总节点数
    load_ratio: float          # 分区总负荷 / 电网总负荷
    generation_ratio: float    # 分区总发电量 / 电网总发电量
    internal_connectivity: float # 分区内部边数 / 分区总边数
    boundary_ratio: float      # 分区边界节点数 / 分区总节点数
    power_imbalance: float     # (分区发电 - 分区负荷) / 分区负荷
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """转换为张量"""
        return torch.tensor([
            self.size_ratio,
            self.load_ratio,
            self.generation_ratio,
            self.internal_connectivity,
            self.boundary_ratio,
            self.power_imbalance
        ], dtype=torch.float32, device=device)


class PartitionEncoder(nn.Module):
    """
    分区编码器 - 双塔架构中的动作塔
    
    将分区的物理特征编码为固定维度的嵌入向量，
    用于与策略向量进行相似度匹配。
    """
    
    def __init__(self, 
                 partition_feature_dim: int = 6,
                 embedding_dim: int = 128,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1,
                 use_layer_norm: bool = True):
        """
        Args:
            partition_feature_dim: 分区特征维度（默认6个标准特征）
            embedding_dim: 输出嵌入维度（需与策略向量维度一致）
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            dropout_rate: Dropout率
            use_layer_norm: 是否使用层归一化
        """
        super().__init__()
        
        self.partition_feature_dim = partition_feature_dim
        self.embedding_dim = embedding_dim
        self.use_layer_norm = use_layer_norm
        
        # 构建编码器网络
        layers = []
        
        # 输入层归一化（处理不同规模电网的特征尺度差异）
        if use_layer_norm:
            layers.append(nn.LayerNorm(partition_feature_dim))
        
        # 输入层
        layers.append(nn.Linear(partition_feature_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # 最终的嵌入归一化（可选）
        self.output_norm = nn.LayerNorm(embedding_dim) if use_layer_norm else None
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"PartitionEncoder initialized: "
                   f"feature_dim={partition_feature_dim}, "
                   f"embedding_dim={embedding_dim}, "
                   f"hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, partition_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            partition_features: 分区特征张量 [batch_size, partition_feature_dim]
                               或 [num_partitions, partition_feature_dim]
        
        Returns:
            分区嵌入向量 [batch_size, embedding_dim] 或 [num_partitions, embedding_dim]
        """
        # 输入验证
        if torch.isnan(partition_features).any() or torch.isinf(partition_features).any():
            logger.warning("Input features contain NaN or Inf values, replacing with zeros")
            partition_features = torch.nan_to_num(partition_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 编码
        embeddings = self.encoder(partition_features)
        
        # 输出归一化（如果启用）
        if self.output_norm is not None:
            embeddings = self.output_norm(embeddings)
        
        return embeddings
    
    def encode_partitions(self, 
                         partitions_info: List[Dict[str, Union[List[int], float]]],
                         grid_info: Dict[str, torch.Tensor],
                         device: torch.device) -> torch.Tensor:
        """
        编码多个分区
        
        Args:
            partitions_info: 分区信息列表，每个包含:
                - 'nodes': 节点ID列表
                - 'load': 总负荷
                - 'generation': 总发电
                - 'internal_edges': 内部边数
                - 'boundary_nodes': 边界节点数
            grid_info: 电网全局信息:
                - 'total_nodes': 总节点数
                - 'total_load': 总负荷
                - 'total_generation': 总发电
            device: 计算设备
        
        Returns:
            分区嵌入矩阵 [num_partitions, embedding_dim]
        """
        features_list = []
        
        total_nodes = grid_info['total_nodes']
        total_load = grid_info['total_load']
        total_generation = grid_info['total_generation']
        
        for partition in partitions_info:
            # 计算标准化特征
            num_nodes = len(partition['nodes'])
            load = partition['load']
            generation = partition['generation']
            internal_edges = partition['internal_edges']
            boundary_nodes = partition['boundary_nodes']
            
            # 计算6个标准特征
            size_ratio = num_nodes / total_nodes if total_nodes > 0 else 0
            load_ratio = load / total_load if total_load > 0 else 0
            generation_ratio = generation / total_generation if total_generation > 0 else 0
            
            # 内部连通性（避免除零）
            max_edges = num_nodes * (num_nodes - 1) / 2
            internal_connectivity = internal_edges / max_edges if max_edges > 0 else 0
            
            # 边界比率
            boundary_ratio = boundary_nodes / num_nodes if num_nodes > 0 else 0
            
            # 功率不平衡
            power_imbalance = (generation - load) / load if load > 0 else 0
            
            # 创建特征向量
            features = PartitionFeatures(
                size_ratio=size_ratio,
                load_ratio=load_ratio,
                generation_ratio=generation_ratio,
                internal_connectivity=internal_connectivity,
                boundary_ratio=boundary_ratio,
                power_imbalance=power_imbalance
            )
            
            features_list.append(features.to_tensor(device))
        
        # 堆叠所有特征
        features_tensor = torch.stack(features_list, dim=0)
        
        # 编码
        return self.forward(features_tensor)
    
    def compute_similarity(self, 
                          strategy_vector: torch.Tensor,
                          partition_embeddings: torch.Tensor,
                          temperature: float = 1.0) -> torch.Tensor:
        """
        计算策略向量与分区嵌入的相似度
        
        Args:
            strategy_vector: 策略向量 [embedding_dim] 或 [batch_size, embedding_dim]
            partition_embeddings: 分区嵌入 [num_partitions, embedding_dim]
            temperature: 温度参数（控制分布的尖锐程度）
        
        Returns:
            相似度分数 [num_partitions] 或 [batch_size, num_partitions]
        """
        # 确保维度正确
        if strategy_vector.dim() == 1:
            strategy_vector = strategy_vector.unsqueeze(0)  # [1, embedding_dim]
        
        # 计算点积相似度
        similarity = torch.matmul(strategy_vector, partition_embeddings.t())  # [batch_size, num_partitions]
        
        # 应用温度缩放
        similarity = similarity / temperature
        
        # 如果输入是单个向量，去除批次维度
        if similarity.size(0) == 1:
            similarity = similarity.squeeze(0)  # [num_partitions]
        
        return similarity


class CachedPartitionEncoder(PartitionEncoder):
    """
    带缓存的分区编码器
    
    为了提高效率，缓存分区特征和嵌入，
    只在分区发生变化时更新。
    """
    
    def __init__(self, *args, cache_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.cache_size = cache_size
        self.feature_cache = {}  # partition_id -> features
        self.embedding_cache = {}  # partition_id -> embedding
        self.cache_hits = 0
        self.cache_misses = 0
    
    def encode_with_cache(self,
                         partition_id: int,
                         partition_info: Dict[str, Union[List[int], float]],
                         grid_info: Dict[str, torch.Tensor],
                         device: torch.device,
                         force_update: bool = False) -> torch.Tensor:
        """
        带缓存的编码
        
        Args:
            partition_id: 分区ID
            partition_info: 分区信息
            grid_info: 电网信息
            device: 计算设备
            force_update: 是否强制更新缓存
        
        Returns:
            分区嵌入 [embedding_dim]
        """
        # 检查缓存
        if not force_update and partition_id in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[partition_id]
        
        self.cache_misses += 1
        
        # 计算嵌入
        embedding = self.encode_partitions([partition_info], grid_info, device)[0]
        
        # 更新缓存
        self.embedding_cache[partition_id] = embedding
        
        # 限制缓存大小
        if len(self.embedding_cache) > self.cache_size:
            # 移除最旧的条目（简单的FIFO策略）
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
            if oldest_key in self.feature_cache:
                del self.feature_cache[oldest_key]
        
        return embedding
    
    def invalidate_cache(self, partition_ids: Optional[List[int]] = None):
        """
        使缓存失效
        
        Args:
            partition_ids: 要失效的分区ID列表，None表示清空所有缓存
        """
        if partition_ids is None:
            self.feature_cache.clear()
            self.embedding_cache.clear()
        else:
            for pid in partition_ids:
                self.feature_cache.pop(pid, None)
                self.embedding_cache.pop(pid, None)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.embedding_cache)
        }


def create_partition_encoder(config: Dict) -> PartitionEncoder:
    """
    工厂函数：创建分区编码器
    
    Args:
        config: 配置字典，包含:
            - partition_feature_dim: 特征维度
            - embedding_dim: 嵌入维度
            - hidden_dim: 隐藏层维度
            - num_layers: 层数
            - dropout_rate: Dropout率
            - use_cache: 是否使用缓存
            - cache_size: 缓存大小
    
    Returns:
        PartitionEncoder实例
    """
    use_cache = config.get('use_cache', True)
    
    encoder_params = {
        'partition_feature_dim': config.get('partition_feature_dim', 6),
        'embedding_dim': config.get('embedding_dim', 128),
        'hidden_dim': config.get('hidden_dim', 64),
        'num_layers': config.get('num_layers', 2),
        'dropout_rate': config.get('dropout_rate', 0.1),
        'use_layer_norm': config.get('use_layer_norm', True)
    }
    
    if use_cache:
        cache_size = config.get('cache_size', 100)
        return CachedPartitionEncoder(**encoder_params, cache_size=cache_size)
    else:
        return PartitionEncoder(**encoder_params)
