"""
相似度计算工具模块

为v3.0双塔架构提供全面的嵌入相似度计算功能，支持：
- 节点嵌入相似度匹配
- 分区嵌入相似度计算  
- 批量相似度计算
- 多种相似度度量方法
- 阈值过滤和排序功能
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    """相似度计算结果"""
    similarities: torch.Tensor  # 相似度得分
    indices: torch.Tensor       # 排序后的索引
    best_match_idx: int        # 最佳匹配的索引
    best_similarity: float     # 最佳匹配的相似度
    above_threshold_count: int # 超过阈值的匹配数量


class SimilarityCalculator:
    """
    嵌入相似度计算器
    
    支持多种相似度度量方法和计算策略
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化相似度计算器
        
        Args:
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def cosine_similarity(self, 
                         query: torch.Tensor, 
                         candidates: torch.Tensor,
                         normalize: bool = True) -> torch.Tensor:
        """
        计算余弦相似度
        
        Args:
            query: 查询向量 [embedding_dim]
            candidates: 候选向量集合 [num_candidates, embedding_dim]
            normalize: 是否预先归一化向量
            
        Returns:
            相似度得分 [num_candidates]
        """
        if normalize:
            query = F.normalize(query, p=2, dim=0)
            candidates = F.normalize(candidates, p=2, dim=1)
        
        similarities = torch.cosine_similarity(
            query.unsqueeze(0), 
            candidates, 
            dim=1
        )
        
        return similarities
    
    def euclidean_distance(self, 
                          query: torch.Tensor, 
                          candidates: torch.Tensor) -> torch.Tensor:
        """
        计算欧几里得距离（转换为相似度）
        
        Args:
            query: 查询向量 [embedding_dim]
            candidates: 候选向量集合 [num_candidates, embedding_dim]
            
        Returns:
            相似度得分 [num_candidates] (距离越小，相似度越高)
        """
        distances = torch.norm(candidates - query.unsqueeze(0), p=2, dim=1)
        # 转换为相似度：相似度 = 1 / (1 + 距离)
        similarities = 1.0 / (1.0 + distances)
        return similarities
    
    def manhattan_distance(self, 
                          query: torch.Tensor, 
                          candidates: torch.Tensor) -> torch.Tensor:
        """
        计算曼哈顿距离（转换为相似度）
        
        Args:
            query: 查询向量 [embedding_dim]
            candidates: 候选向量集合 [num_candidates, embedding_dim]
            
        Returns:
            相似度得分 [num_candidates]
        """
        distances = torch.norm(candidates - query.unsqueeze(0), p=1, dim=1)
        similarities = 1.0 / (1.0 + distances)
        return similarities
    
    def dot_product_similarity(self, 
                              query: torch.Tensor, 
                              candidates: torch.Tensor) -> torch.Tensor:
        """
        计算点积相似度
        
        Args:
            query: 查询向量 [embedding_dim]
            candidates: 候选向量集合 [num_candidates, embedding_dim]
            
        Returns:
            相似度得分 [num_candidates]
        """
        similarities = torch.matmul(candidates, query)
        return similarities
    
    def compute_similarity(self, 
                          query: torch.Tensor, 
                          candidates: torch.Tensor,
                          method: str = 'cosine',
                          threshold: float = 0.0) -> SimilarityResult:
        """
        计算相似度并返回完整结果
        
        Args:
            query: 查询向量
            candidates: 候选向量集合
            method: 相似度计算方法 ('cosine', 'euclidean', 'manhattan', 'dot_product')
            threshold: 相似度阈值
            
        Returns:
            SimilarityResult对象
        """
        # 选择计算方法
        if method == 'cosine':
            similarities = self.cosine_similarity(query, candidates)
        elif method == 'euclidean':
            similarities = self.euclidean_distance(query, candidates)
        elif method == 'manhattan':
            similarities = self.manhattan_distance(query, candidates)
        elif method == 'dot_product':
            similarities = self.dot_product_similarity(query, candidates)
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")
        
        # 排序索引（降序）
        sorted_indices = torch.argsort(similarities, descending=True)
        
        # 最佳匹配
        best_match_idx = sorted_indices[0].item()
        best_similarity = similarities[best_match_idx].item()
        
        # 超过阈值的数量
        above_threshold_count = (similarities >= threshold).sum().item()
        
        return SimilarityResult(
            similarities=similarities,
            indices=sorted_indices,
            best_match_idx=best_match_idx,
            best_similarity=best_similarity,
            above_threshold_count=above_threshold_count
        )
    
    def batch_similarity(self, 
                        queries: torch.Tensor, 
                        candidates: torch.Tensor,
                        method: str = 'cosine') -> torch.Tensor:
        """
        批量计算相似度
        
        Args:
            queries: 查询向量集合 [num_queries, embedding_dim]
            candidates: 候选向量集合 [num_candidates, embedding_dim]
            method: 相似度计算方法
            
        Returns:
            相似度矩阵 [num_queries, num_candidates]
        """
        if method == 'cosine':
            # 归一化
            queries = F.normalize(queries, p=2, dim=1)
            candidates = F.normalize(candidates, p=2, dim=1)
            # 批量点积
            similarities = torch.matmul(queries, candidates.t())
        elif method == 'dot_product':
            similarities = torch.matmul(queries, candidates.t())
        else:
            # 对于距离度量，逐个计算
            similarities = []
            for query in queries:
                sim = self.compute_similarity(query, candidates, method).similarities
                similarities.append(sim)
            similarities = torch.stack(similarities)
        
        return similarities
    
    def find_top_k_matches(self, 
                          query: torch.Tensor, 
                          candidates: torch.Tensor,
                          k: int = 5,
                          method: str = 'cosine',
                          threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查找前K个最相似的匹配
        
        Args:
            query: 查询向量
            candidates: 候选向量集合
            k: 返回的匹配数量
            method: 相似度计算方法
            threshold: 最小相似度阈值
            
        Returns:
            (top_k_indices, top_k_similarities)
        """
        result = self.compute_similarity(query, candidates, method, threshold)
        
        # 过滤低于阈值的结果
        valid_mask = result.similarities >= threshold
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        
        if len(valid_indices) == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([])
        
        # 获取有效结果的相似度
        valid_similarities = result.similarities[valid_indices]
        
        # 排序并取前K个
        sorted_valid_indices = torch.argsort(valid_similarities, descending=True)
        top_k_count = min(k, len(sorted_valid_indices))
        
        top_k_valid_indices = sorted_valid_indices[:top_k_count]
        top_k_indices = valid_indices[top_k_valid_indices]
        top_k_similarities = valid_similarities[top_k_valid_indices]
        
        return top_k_indices, top_k_similarities


class NodeSimilarityMatcher:
    """节点相似度匹配器"""
    
    def __init__(self, similarity_calculator: SimilarityCalculator = None):
        self.calc = similarity_calculator or SimilarityCalculator()
    
    def find_similar_nodes(self, 
                          target_node_embedding: torch.Tensor,
                          current_node_embeddings: torch.Tensor,
                          threshold: float = 0.8,
                          top_k: int = 3) -> List[Tuple[int, float]]:
        """
        查找相似的节点
        
        Args:
            target_node_embedding: 目标节点嵌入
            current_node_embeddings: 当前环境的节点嵌入
            threshold: 相似度阈值
            top_k: 返回的最大匹配数量
            
        Returns:
            [(node_id, similarity), ...] 匹配列表
        """
        top_k_indices, top_k_similarities = self.calc.find_top_k_matches(
            target_node_embedding, current_node_embeddings, top_k, threshold=threshold
        )
        
        matches = []
        for idx, sim in zip(top_k_indices, top_k_similarities):
            matches.append((idx.item(), sim.item()))
        
        return matches
    
    def match_node_with_fallback(self, 
                                target_node_embedding: torch.Tensor,
                                current_node_embeddings: torch.Tensor,
                                fallback_strategies: List[str] = None) -> Optional[int]:
        """
        带回退策略的节点匹配
        
        Args:
            target_node_embedding: 目标节点嵌入
            current_node_embeddings: 当前环境节点嵌入
            fallback_strategies: 回退策略列表
            
        Returns:
            匹配的节点ID，或None
        """
        if fallback_strategies is None:
            fallback_strategies = ['cosine', 'euclidean', 'dot_product']
        
        thresholds = [0.8, 0.6, 0.4, 0.2]  # 逐渐降低阈值
        
        for method in fallback_strategies:
            for threshold in thresholds:
                result = self.calc.compute_similarity(
                    target_node_embedding, current_node_embeddings, method, threshold
                )
                
                if result.above_threshold_count > 0:
                    return result.best_match_idx
        
        # 最后的回退：直接返回最相似的
        result = self.calc.compute_similarity(
            target_node_embedding, current_node_embeddings, 'cosine'
        )
        return result.best_match_idx


class PartitionSimilarityMatcher:
    """分区相似度匹配器"""
    
    def __init__(self, similarity_calculator: SimilarityCalculator = None):
        self.calc = similarity_calculator or SimilarityCalculator()
    
    def find_similar_partitions(self, 
                               target_partition_embedding: torch.Tensor,
                               current_partition_embeddings: torch.Tensor,
                               partition_ids: torch.Tensor,
                               threshold: float = 0.7,
                               top_k: int = 3) -> List[Tuple[int, float]]:
        """
        查找相似的分区
        
        Args:
            target_partition_embedding: 目标分区嵌入
            current_partition_embeddings: 当前环境分区嵌入
            partition_ids: 分区ID列表
            threshold: 相似度阈值
            top_k: 返回的最大匹配数量
            
        Returns:
            [(partition_id, similarity), ...] 匹配列表
        """
        top_k_indices, top_k_similarities = self.calc.find_top_k_matches(
            target_partition_embedding, current_partition_embeddings, top_k, threshold=threshold
        )
        
        matches = []
        for idx, sim in zip(top_k_indices, top_k_similarities):
            partition_id = partition_ids[idx.item()].item()
            matches.append((partition_id, sim.item()))
        
        return matches
    
    def match_partition_by_features(self, 
                                   target_features: Dict[str, float],
                                   current_partitions_features: List[Dict[str, float]],
                                   feature_weights: Dict[str, float] = None) -> Optional[int]:
        """
        基于特征的分区匹配（作为嵌入匹配的补充）
        
        Args:
            target_features: 目标分区特征
            current_partitions_features: 当前分区特征列表
            feature_weights: 特征权重
            
        Returns:
            最匹配的分区索引
        """
        if feature_weights is None:
            feature_weights = {
                'size_ratio': 0.2,
                'load_ratio': 0.3,
                'generation_ratio': 0.2,
                'internal_connectivity': 0.15,
                'boundary_ratio': 0.1,
                'power_imbalance': 0.05
            }
        
        best_score = -float('inf')
        best_idx = None
        
        for idx, current_features in enumerate(current_partitions_features):
            score = 0.0
            
            for feature_name, weight in feature_weights.items():
                target_val = target_features.get(feature_name, 0.0)
                current_val = current_features.get(feature_name, 0.0)
                
                # 使用负的平方差作为相似度
                diff = abs(target_val - current_val)
                feature_similarity = 1.0 / (1.0 + diff)
                score += weight * feature_similarity
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx


# 便捷函数

def compute_embedding_similarity(embedding1: torch.Tensor, 
                               embedding2: torch.Tensor,
                               method: str = 'cosine') -> float:
    """
    计算两个嵌入之间的相似度
    
    Args:
        embedding1: 第一个嵌入向量
        embedding2: 第二个嵌入向量
        method: 相似度计算方法
        
    Returns:
        相似度得分
    """
    calc = SimilarityCalculator()
    result = calc.compute_similarity(embedding1, embedding2.unsqueeze(0), method)
    return result.best_similarity


def find_best_match(query_embedding: torch.Tensor,
                   candidate_embeddings: torch.Tensor,
                   threshold: float = 0.5,
                   method: str = 'cosine') -> Optional[int]:
    """
    查找最佳匹配的候选项
    
    Args:
        query_embedding: 查询嵌入
        candidate_embeddings: 候选嵌入集合
        threshold: 最小相似度阈值
        method: 相似度计算方法
        
    Returns:
        最佳匹配的索引，或None
    """
    calc = SimilarityCalculator()
    result = calc.compute_similarity(query_embedding, candidate_embeddings, method, threshold)
    
    if result.best_similarity >= threshold:
        return result.best_match_idx
    else:
        return None


def batch_find_matches(query_embeddings: torch.Tensor,
                      candidate_embeddings: torch.Tensor,
                      threshold: float = 0.5,
                      method: str = 'cosine') -> List[Optional[int]]:
    """
    批量查找匹配
    
    Args:
        query_embeddings: 查询嵌入集合 [num_queries, embedding_dim]
        candidate_embeddings: 候选嵌入集合 [num_candidates, embedding_dim]
        threshold: 最小相似度阈值
        method: 相似度计算方法
        
    Returns:
        匹配索引列表，每个查询对应一个索引（或None）
    """
    calc = SimilarityCalculator()
    similarity_matrix = calc.batch_similarity(query_embeddings, candidate_embeddings, method)
    
    matches = []
    for similarities in similarity_matrix:
        best_idx = torch.argmax(similarities).item()
        best_similarity = similarities[best_idx].item()
        
        if best_similarity >= threshold:
            matches.append(best_idx)
        else:
            matches.append(None)
    
    return matches 