#!/usr/bin/env python3
"""
Abstract Decision Context for v3.0 Architecture

实现抽象决策上下文，支持：
- 完全嵌入表示（无具体ID依赖）
- 双塔架构决策机制
- 跨拓扑泛化能力
- 批量处理和序列化
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class AbstractDecisionContext:
    """
    抽象决策上下文 - v3.0架构的核心数据结构
    
    完全基于嵌入表示，不依赖具体的节点ID或分区ID，
    支持跨拓扑泛化和相似度决策机制。
    """
    
    # 核心嵌入表示
    strategy_vector: torch.Tensor                    # 策略向量 [embedding_dim]
    candidate_embeddings: torch.Tensor               # 候选分区嵌入 [num_candidates, embedding_dim]
    node_embedding: torch.Tensor                     # 目标节点嵌入 [embedding_dim]
    
    # 决策计算结果
    similarity_scores: torch.Tensor                  # 相似度得分 [num_candidates]
    probability_distribution: torch.Tensor           # 概率分布 [num_candidates]
    
    # 上下文信息
    boundary_nodes_embeddings: torch.Tensor          # 边界节点嵌入 [num_boundary, embedding_dim]
    state_embedding: torch.Tensor                    # 全局状态嵌入 [state_dim]
    
    # 决策元数据
    timestamp: float = field(default_factory=time.time)
    temperature: float = 1.0
    decision_confidence: float = 0.0
    
    # 可选的排名信息（用于分析）
    candidate_rankings: Optional[torch.Tensor] = None  # 候选项排名 [num_candidates]
    
    def __post_init__(self):
        """后处理验证"""
        self.validate_consistency()
    
    def validate_consistency(self) -> bool:
        """验证内部一致性"""
        try:
            # 检查张量维度
            num_candidates = self.candidate_embeddings.size(0)
            
            assert self.similarity_scores.size(0) == num_candidates, \
                f"相似度得分维度不匹配: {self.similarity_scores.size(0)} vs {num_candidates}"
            
            assert self.probability_distribution.size(0) == num_candidates, \
                f"概率分布维度不匹配: {self.probability_distribution.size(0)} vs {num_candidates}"
            
            # 检查概率分布有效性
            assert torch.allclose(self.probability_distribution.sum(), torch.tensor(1.0), atol=1e-5), \
                f"概率分布和不为1: {self.probability_distribution.sum()}"
            
            assert torch.all(self.probability_distribution >= 0), \
                "概率分布包含负值"
            
            # 检查嵌入维度一致性
            embedding_dim = self.strategy_vector.size(0)
            assert self.candidate_embeddings.size(1) == embedding_dim, \
                "候选嵌入维度不一致"
            assert self.node_embedding.size(0) == embedding_dim, \
                "节点嵌入维度不一致"
            
            return True
            
        except Exception as e:
            logger.error(f"DecisionContext一致性验证失败: {e}")
            return False
    
    def compute_probability_distribution(self, temperature: float = None) -> torch.Tensor:
        """基于相似度得分重新计算概率分布"""
        if temperature is None:
            temperature = self.temperature
        
        # 温度缩放的softmax
        scaled_scores = self.similarity_scores / temperature
        probabilities = F.softmax(scaled_scores, dim=0)
        
        return probabilities
    
    def sample_action(self, temperature: float = None) -> Tuple[int, float]:
        """
        从概率分布中采样动作
        
        Returns:
            (candidate_idx, log_prob): 选择的候选索引和对数概率
        """
        probabilities = self.compute_probability_distribution(temperature)
        
        # 多项式采样
        candidate_idx = torch.multinomial(probabilities, 1).item()
        log_prob = torch.log(probabilities[candidate_idx])
        
        return candidate_idx, log_prob.item()
    
    def get_best_candidate(self) -> Tuple[int, float]:
        """获取最佳候选项（贪婪选择）"""
        best_idx = torch.argmax(self.similarity_scores).item()
        confidence = self.similarity_scores[best_idx].item()
        return best_idx, confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'strategy_vector': self.strategy_vector.cpu().numpy(),
            'candidate_embeddings': self.candidate_embeddings.cpu().numpy(),
            'node_embedding': self.node_embedding.cpu().numpy(),
            'similarity_scores': self.similarity_scores.cpu().numpy(),
            'probability_distribution': self.probability_distribution.cpu().numpy(),
            'boundary_nodes_embeddings': self.boundary_nodes_embeddings.cpu().numpy(),
            'state_embedding': self.state_embedding.cpu().numpy(),
            'timestamp': self.timestamp,
            'temperature': self.temperature,
            'decision_confidence': self.decision_confidence,
            'candidate_rankings': self.candidate_rankings.cpu().numpy() if self.candidate_rankings is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device = None) -> 'AbstractDecisionContext':
        """从字典反序列化"""
        if device is None:
            device = torch.device('cpu')
        
        kwargs = {}
        
        # 转换numpy数组为tensor
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                kwargs[key] = torch.from_numpy(value).to(device)
            elif key == 'candidate_rankings' and value is not None:
                kwargs[key] = torch.from_numpy(value).to(device)
            else:
                kwargs[key] = value
        
        return cls(**kwargs)


class DecisionContextBatch:
    """
    决策上下文批量处理器
    
    支持高效的批量操作和并行计算
    """
    
    def __init__(self, contexts: List[AbstractDecisionContext], batch_size: int = None):
        self.contexts = contexts
        self.batch_size = batch_size or len(contexts)
        self._validate_batch()
    
    def _validate_batch(self):
        """验证批量数据的一致性"""
        if not self.contexts:
            raise ValueError("空的上下文列表")
        
        # 检查嵌入维度一致性
        embedding_dim = self.contexts[0].strategy_vector.size(0)
        for i, ctx in enumerate(self.contexts):
            if ctx.strategy_vector.size(0) != embedding_dim:
                raise ValueError(f"上下文{i}的嵌入维度不一致: {ctx.strategy_vector.size(0)} vs {embedding_dim}")
    
    def compute_batch_log_probs(self, temperature: float = 1.0) -> torch.Tensor:
        """
        批量计算对数概率
        
        Returns:
            log_probs: [batch_size] 每个决策的对数概率
        """
        log_probs = []
        
        for ctx in self.contexts:
            # 重新计算概率分布
            probabilities = ctx.compute_probability_distribution(temperature)
            
            # 获取选择的候选项（最高相似度）
            selected_idx = torch.argmax(ctx.similarity_scores)
            log_prob = torch.log(probabilities[selected_idx])
            
            log_probs.append(log_prob)
        
        return torch.stack(log_probs)
    
    def compute_batch_similarities(self) -> torch.Tensor:
        """批量计算最大相似度得分"""
        max_similarities = []
        
        for ctx in self.contexts:
            max_sim = torch.max(ctx.similarity_scores)
            max_similarities.append(max_sim)
        
        return torch.stack(max_similarities)
    
    def get_batch_confidence(self) -> torch.Tensor:
        """获取批量决策置信度"""
        confidences = []
        
        for ctx in self.contexts:
            # 使用最高概率作为置信度
            max_prob = torch.max(ctx.probability_distribution)
            confidences.append(max_prob)
        
        return torch.stack(confidences)
    
    def __len__(self) -> int:
        return len(self.contexts)
    
    def __getitem__(self, idx: int) -> AbstractDecisionContext:
        return self.contexts[idx]


def create_decision_context(
    strategy_vector: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    node_embedding: torch.Tensor,
    boundary_nodes_embeddings: torch.Tensor,
    state_embedding: torch.Tensor,
    temperature: float = 1.0
) -> AbstractDecisionContext:
    """
    创建抽象决策上下文的工厂函数
    
    Args:
        strategy_vector: 策略向量 [embedding_dim]
        candidate_embeddings: 候选分区嵌入 [num_candidates, embedding_dim]
        node_embedding: 目标节点嵌入 [embedding_dim]
        boundary_nodes_embeddings: 边界节点嵌入 [num_boundary, embedding_dim]
        state_embedding: 全局状态嵌入 [state_dim]
        temperature: 温度参数
    
    Returns:
        AbstractDecisionContext: 完整的决策上下文
    """
    
    # 计算相似度得分（余弦相似度）
    strategy_normalized = F.normalize(strategy_vector, dim=0)
    candidates_normalized = F.normalize(candidate_embeddings, dim=1)
    
    similarity_scores = torch.matmul(candidates_normalized, strategy_normalized)
    
    # 计算概率分布
    scaled_scores = similarity_scores / temperature
    probability_distribution = F.softmax(scaled_scores, dim=0)
    
    # 计算候选项排名
    candidate_rankings = torch.argsort(similarity_scores, descending=True)
    
    # 计算决策置信度（最高概率）
    decision_confidence = torch.max(probability_distribution).item()
    
    return AbstractDecisionContext(
        strategy_vector=strategy_vector,
        candidate_embeddings=candidate_embeddings,
        node_embedding=node_embedding,
        similarity_scores=similarity_scores,
        probability_distribution=probability_distribution,
        boundary_nodes_embeddings=boundary_nodes_embeddings,
        state_embedding=state_embedding,
        temperature=temperature,
        decision_confidence=decision_confidence,
        candidate_rankings=candidate_rankings
    )


def materialize_decision_context(
    context: AbstractDecisionContext,
    candidate_mappings: List[Tuple[int, int]],
    similarity_threshold: float = 0.1
) -> Tuple[int, int]:
    """
    将抽象决策上下文实体化为具体动作
    
    Args:
        context: 抽象决策上下文
        candidate_mappings: 候选项到具体动作的映射 [(node_id, partition_id), ...]
        similarity_threshold: 最低相似度阈值
    
    Returns:
        (node_id, partition_id): 具体的执行动作
    
    Raises:
        ValueError: 如果无法找到满足阈值的动作
    """
    
    # 获取最佳候选项
    best_candidate_idx, confidence = context.get_best_candidate()
    
    # 检查置信度
    if confidence < similarity_threshold:
        raise ValueError(f"最佳候选置信度 {confidence:.3f} 低于阈值 {similarity_threshold}")
    
    # 检查映射有效性
    if best_candidate_idx >= len(candidate_mappings):
        raise ValueError(f"候选索引 {best_candidate_idx} 超出映射范围 {len(candidate_mappings)}")
    
    return candidate_mappings[best_candidate_idx]


def compute_context_similarity(
    ctx1: AbstractDecisionContext,
    ctx2: AbstractDecisionContext
) -> float:
    """
    计算两个决策上下文的相似度
    
    用于分析跨拓扑的决策一致性
    """
    
    # 策略向量相似度
    strategy_sim = F.cosine_similarity(
        ctx1.strategy_vector.unsqueeze(0),
        ctx2.strategy_vector.unsqueeze(0)
    ).item()
    
    # 状态嵌入相似度
    state_sim = F.cosine_similarity(
        ctx1.state_embedding.unsqueeze(0),
        ctx2.state_embedding.unsqueeze(0)
    ).item()
    
    # 加权平均
    overall_similarity = 0.7 * strategy_sim + 0.3 * state_sim
    
    return overall_similarity


def validate_context_consistency(contexts: List[AbstractDecisionContext]) -> Dict[str, Any]:
    """
    验证多个决策上下文的一致性
    
    Returns:
        验证报告字典
    """
    
    report = {
        'total_contexts': len(contexts),
        'valid_contexts': 0,
        'consistency_issues': [],
        'embedding_dim_consistency': True,
        'probability_validity': True
    }
    
    if not contexts:
        report['consistency_issues'].append("空的上下文列表")
        return report
    
    # 检查嵌入维度一致性
    reference_dim = contexts[0].strategy_vector.size(0)
    
    for i, ctx in enumerate(contexts):
        try:
            # 内部一致性
            if ctx.validate_consistency():
                report['valid_contexts'] += 1
            else:
                report['consistency_issues'].append(f"上下文{i}内部一致性失败")
            
            # 维度一致性
            if ctx.strategy_vector.size(0) != reference_dim:
                report['embedding_dim_consistency'] = False
                report['consistency_issues'].append(f"上下文{i}嵌入维度不一致")
            
            # 概率有效性
            if not torch.allclose(ctx.probability_distribution.sum(), torch.tensor(1.0), atol=1e-5):
                report['probability_validity'] = False
                report['consistency_issues'].append(f"上下文{i}概率分布无效")
        
        except Exception as e:
            report['consistency_issues'].append(f"上下文{i}验证错误: {e}")
    
    return report 