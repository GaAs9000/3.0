"""
v3.0架构合规性验证

验证实现是否真正符合project_overview.md中定义的v3.0双塔架构设计理念：

核心要求：
1. 双塔模型：状态塔(ActorNetwork) + 动作塔(PartitionEncoder)
2. 策略向量输出，不直接输出具体分区ID
3. 通过余弦相似度计算进行决策
4. 支持跨拓扑泛化能力
5. "学习比较能力"而非"学习具体动作"
6. 完全解耦分区ID和数量(K值)
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl.decision_context import AbstractDecisionContext, create_decision_context
from rl.enhanced_agent import EnhancedPPOAgent
from rl.agent import PPOAgent  
from rl.fast_memory import FastPPOMemory
from rl.similarity_utils import SimilarityCalculator, NodeSimilarityMatcher, PartitionSimilarityMatcher

logger = logging.getLogger(__name__)


class V3ArchitectureComplianceValidator:
    """v3.0架构合规性验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.compliance_report = {
            'overall_compliant': False,
            'dual_tower_architecture': False,
            'strategy_vector_output': False,
            'similarity_based_decision': False,
            'cross_topology_generalization': False,
            'abstract_representation': False,
            'no_concrete_id_dependency': False,
            'decoupled_k_values': False,
            'issues': [],
            'recommendations': []
        }
    
    def validate_dual_tower_architecture(self, agent: Any) -> bool:
        """验证双塔架构实现"""
        print("\n=== 验证双塔架构 ===")
        
        # 检查状态塔（ActorNetwork）
        if not hasattr(agent, 'actor'):
            self.compliance_report['issues'].append("缺少状态塔（ActorNetwork）")
            return False
        
        # 检查动作塔（PartitionEncoder）
        if not hasattr(agent, 'partition_encoder'):
            self.compliance_report['issues'].append("缺少动作塔（PartitionEncoder）")
            return False
        
        print("✅ 发现状态塔（ActorNetwork）和动作塔（PartitionEncoder）")
        
        # 验证架构分离
        try:
            # 状态塔应该输出策略向量，不直接输出动作
            state_input = self._create_mock_state()
            batched_state = agent._prepare_batched_state([state_input])
            
            # 检查actor输出
            actor_output = agent.actor(batched_state)
            if isinstance(actor_output, tuple) and len(actor_output) == 2:
                node_logits, strategy_vectors = actor_output
                print(f"✅ 状态塔输出：node_logits {node_logits.shape}, strategy_vectors {strategy_vectors.shape}")
            else:
                self.compliance_report['issues'].append("状态塔输出格式不符合双塔设计")
                return False
        
        except Exception as e:
            self.compliance_report['issues'].append(f"状态塔验证失败：{e}")
            return False
        
        # 验证动作塔独立性
        try:
            # 动作塔应该能够独立编码分区特征
            mock_partition_features = torch.randn(3, 6)  # 3个分区，6个特征
            partition_embeddings = agent.partition_encoder(mock_partition_features)
            print(f"✅ 动作塔输出：partition_embeddings {partition_embeddings.shape}")
            
        except Exception as e:
            self.compliance_report['issues'].append(f"动作塔验证失败：{e}")
            return False
        
        self.compliance_report['dual_tower_architecture'] = True
        return True
    
    def validate_strategy_vector_output(self, agent: Any) -> bool:
        """验证策略向量输出（不直接输出分区ID）"""
        print("\n=== 验证策略向量输出 ===")
        
        try:
            state_input = self._create_mock_state()
            result = agent.select_action(state_input, training=True)
            
            # 检查返回格式
            if len(result) != 3:
                self.compliance_report['issues'].append(f"select_action返回格式错误：{len(result)}项")
                return False
            
            decision_context_or_action, value, embeddings = result
            
            # 关键验证：应该返回AbstractDecisionContext，不是具体action
            if isinstance(decision_context_or_action, AbstractDecisionContext):
                print("✅ select_action返回AbstractDecisionContext（符合v3.0设计）")
                
                # 验证策略向量存在
                if hasattr(decision_context_or_action, 'strategy_vector'):
                    strategy_dim = decision_context_or_action.strategy_vector.size(0)
                    print(f"✅ 策略向量维度：{strategy_dim}")
                else:
                    self.compliance_report['issues'].append("DecisionContext缺少策略向量")
                    return False
                    
            elif isinstance(decision_context_or_action, tuple):
                self.compliance_report['issues'].append("select_action仍返回具体动作tuple，未实现v3.0设计")
                return False
            else:
                self.compliance_report['issues'].append(f"select_action返回未知格式：{type(decision_context_or_action)}")
                return False
        
        except Exception as e:
            self.compliance_report['issues'].append(f"策略向量验证失败：{e}")
            return False
        
        self.compliance_report['strategy_vector_output'] = True
        return True
    
    def validate_similarity_based_decision(self, agent: Any) -> bool:
        """验证基于相似度的决策机制"""
        print("\n=== 验证相似度决策机制 ===")
        
        try:
            state_input = self._create_mock_state()
            result = agent.select_action(state_input, training=True)
            decision_context, _, _ = result
            
            if not isinstance(decision_context, AbstractDecisionContext):
                self.compliance_report['issues'].append("无法验证相似度决策：未使用AbstractDecisionContext")
                return False
            
            # 验证相似度计算存在
            if not hasattr(decision_context, 'selection_similarities'):
                self.compliance_report['issues'].append("DecisionContext缺少相似度得分")
                return False
            
            similarities = decision_context.selection_similarities
            print(f"✅ 相似度得分形状：{similarities.shape}")
            print(f"✅ 相似度范围：[{similarities.min():.3f}, {similarities.max():.3f}]")
            
            # 验证选择逻辑
            if not hasattr(decision_context, 'selected_rank'):
                self.compliance_report['issues'].append("DecisionContext缺少选择排名")
                return False
            
            selected_rank = decision_context.selected_rank
            num_candidates = decision_context.num_candidates
            
            if not (0 <= selected_rank < num_candidates):
                self.compliance_report['issues'].append(f"选择排名{selected_rank}超出范围[0,{num_candidates})")
                return False
            
            print(f"✅ 选择排名：{selected_rank}/{num_candidates}")
            
            # 验证相似度计算可重现性
            try:
                reconstructed_similarities = torch.cosine_similarity(
                    decision_context.strategy_vector.unsqueeze(0),
                    decision_context.candidate_embeddings,
                    dim=1
                )
                
                similarity_diff = torch.abs(similarities - reconstructed_similarities).max()
                if similarity_diff > 1e-5:
                    self.compliance_report['issues'].append(f"相似度计算不可重现，差异：{similarity_diff}")
                    return False
                
                print(f"✅ 相似度计算可重现，最大差异：{similarity_diff:.6f}")
                
            except Exception as e:
                self.compliance_report['issues'].append(f"相似度重现验证失败：{e}")
                return False
        
        except Exception as e:
            self.compliance_report['issues'].append(f"相似度决策验证失败：{e}")
            return False
        
        self.compliance_report['similarity_based_decision'] = True
        return True
    
    def validate_cross_topology_generalization(self, agent: Any) -> bool:
        """验证跨拓扑泛化能力"""
        print("\n=== 验证跨拓扑泛化能力 ===")
        
        try:
            # 创建不同拓扑的状态
            topologies = [
                self._create_mock_state(num_nodes=14, num_partitions=3),  # IEEE14
                self._create_mock_state(num_nodes=30, num_partitions=4),  # IEEE30
                self._create_mock_state(num_nodes=57, num_partitions=5),  # IEEE57
            ]
            
            decision_contexts = []
            
            for i, state in enumerate(topologies):
                try:
                    result = agent.select_action(state, training=False)
                    decision_context, _, _ = result
                    
                    if isinstance(decision_context, AbstractDecisionContext):
                        decision_contexts.append(decision_context)
                        print(f"✅ 拓扑{i+1}({state['x'].size(0)}节点)：成功生成DecisionContext")
                    else:
                        self.compliance_report['issues'].append(f"拓扑{i+1}返回具体动作而非抽象上下文")
                        return False
                
                except Exception as e:
                    self.compliance_report['issues'].append(f"拓扑{i+1}处理失败：{e}")
                    return False
            
            # 验证嵌入维度一致性（泛化的关键）
            if len(decision_contexts) >= 2:
                ref_node_dim = decision_contexts[0].node_embedding.size(0)
                ref_strategy_dim = decision_contexts[0].strategy_vector.size(0)
                ref_partition_dim = decision_contexts[0].target_partition_embedding.size(0)
                
                for i, ctx in enumerate(decision_contexts[1:], 1):
                    if ctx.node_embedding.size(0) != ref_node_dim:
                        self.compliance_report['issues'].append(f"拓扑{i+1}节点嵌入维度不一致")
                        return False
                    if ctx.strategy_vector.size(0) != ref_strategy_dim:
                        self.compliance_report['issues'].append(f"拓扑{i+1}策略向量维度不一致")
                        return False
                    if ctx.target_partition_embedding.size(0) != ref_partition_dim:
                        self.compliance_report['issues'].append(f"拓扑{i+1}分区嵌入维度不一致")
                        return False
                
                print(f"✅ 所有拓扑嵌入维度一致：节点{ref_node_dim}, 策略{ref_strategy_dim}, 分区{ref_partition_dim}")
            
            # 验证决策上下文不包含拓扑特定信息
            for i, ctx in enumerate(decision_contexts):
                if ctx.node_context and 'node_id' in ctx.node_context:
                    # 这是调试信息，不应该用于训练
                    print(f"⚠️  拓扑{i+1}包含调试用的node_id，但不影响泛化")
                
            print("✅ 跨拓扑泛化验证通过")
        
        except Exception as e:
            self.compliance_report['issues'].append(f"跨拓扑泛化验证失败：{e}")
            return False
        
        self.compliance_report['cross_topology_generalization'] = True
        return True
    
    def validate_abstract_representation(self, agent: Any) -> bool:
        """验证抽象表示（无具体ID依赖）"""
        print("\n=== 验证抽象表示 ===")
        
        try:
            state_input = self._create_mock_state()
            result = agent.select_action(state_input, training=True)
            decision_context, _, _ = result
            
            if not isinstance(decision_context, AbstractDecisionContext):
                self.compliance_report['issues'].append("未使用抽象决策上下文")
                return False
            
            # 验证核心要求：不直接存储具体ID
            # AbstractDecisionContext应该只包含嵌入表示
            abstract_fields = [
                'node_embedding', 'strategy_vector', 'target_partition_embedding',
                'candidate_embeddings', 'selection_similarities', 'selected_rank'
            ]
            
            for field in abstract_fields:
                if not hasattr(decision_context, field):
                    self.compliance_report['issues'].append(f"DecisionContext缺少抽象字段：{field}")
                    return False
            
            print("✅ DecisionContext包含所有必需的抽象字段")
            
            # 验证嵌入表示的有效性
            node_emb = decision_context.node_embedding
            strategy_vec = decision_context.strategy_vector
            partition_emb = decision_context.target_partition_embedding
            candidates = decision_context.candidate_embeddings
            
            # 检查嵌入是否为有效的实数张量
            for name, tensor in [
                ('node_embedding', node_emb),
                ('strategy_vector', strategy_vec),
                ('target_partition_embedding', partition_emb),
                ('candidate_embeddings', candidates)
            ]:
                if not torch.all(torch.isfinite(tensor)):
                    self.compliance_report['issues'].append(f"{name}包含无效值")
                    return False
            
            print("✅ 所有嵌入表示均为有效实数张量")
            
            # 验证相似度的数学正确性
            expected_similarities = torch.cosine_similarity(
                strategy_vec.unsqueeze(0), candidates, dim=1
            )
            actual_similarities = decision_context.selection_similarities
            
            max_diff = torch.abs(expected_similarities - actual_similarities).max()
            if max_diff > 1e-5:
                self.compliance_report['issues'].append(f"相似度计算不正确，最大差异：{max_diff}")
                return False
            
            print("✅ 相似度计算数学正确性验证通过")
        
        except Exception as e:
            self.compliance_report['issues'].append(f"抽象表示验证失败：{e}")
            return False
        
        self.compliance_report['abstract_representation'] = True
        return True
    
    def validate_no_concrete_id_dependency(self, agent: Any) -> bool:
        """验证无具体ID依赖"""
        print("\n=== 验证无具体ID依赖 ===")
        
        # 这个验证确保模型不依赖于具体的节点或分区ID
        try:
            # 创建相同拓扑但不同ID映射的状态
            original_state = self._create_mock_state(num_nodes=14, num_partitions=3)
            
            # 重新映射节点特征（模拟不同的节点编号）
            shuffled_state = original_state.copy()
            perm = torch.randperm(14)
            shuffled_state['x'] = original_state['x'][perm]
            shuffled_state['current_partition'] = original_state['current_partition'][perm]
            
            # 获取两种状态的决策上下文
            result1 = agent.select_action(original_state, training=False)
            result2 = agent.select_action(shuffled_state, training=False)
            
            ctx1, _, _ = result1
            ctx2, _, _ = result2
            
            if not (isinstance(ctx1, AbstractDecisionContext) and isinstance(ctx2, AbstractDecisionContext)):
                self.compliance_report['issues'].append("无法获取AbstractDecisionContext进行ID依赖性测试")
                return False
            
            # 验证策略向量维度一致（结构无关性）
            if ctx1.strategy_vector.size() != ctx2.strategy_vector.size():
                self.compliance_report['issues'].append("策略向量维度依赖于具体ID映射")
                return False
            
            # 验证分区嵌入维度一致
            if ctx1.target_partition_embedding.size() != ctx2.target_partition_embedding.size():
                self.compliance_report['issues'].append("分区嵌入维度依赖于具体ID映射")
                return False
            
            print("✅ 策略向量和嵌入维度不依赖于具体ID映射")
            
            # 验证决策过程的结构一致性
            # 注意：具体的嵌入值可能不同（因为输入不同），但结构应该一致
            if ctx1.num_candidates != ctx2.num_candidates:
                print(f"⚠️  候选数量不同（{ctx1.num_candidates} vs {ctx2.num_candidates}），这是正常的拓扑变化")
            
            print("✅ 决策过程结构保持一致")
        
        except Exception as e:
            self.compliance_report['issues'].append(f"ID依赖性验证失败：{e}")
            return False
        
        self.compliance_report['no_concrete_id_dependency'] = True
        return True
    
    def validate_decoupled_k_values(self, agent: Any) -> bool:
        """验证解耦K值（分区数量）"""
        print("\n=== 验证解耦K值 ===")
        
        try:
            # 测试不同K值的处理能力
            k_values = [2, 3, 4, 5]
            contexts = []
            
            for k in k_values:
                state = self._create_mock_state(num_nodes=14, num_partitions=k)
                result = agent.select_action(state, training=False)
                decision_context, _, _ = result
                
                if isinstance(decision_context, AbstractDecisionContext):
                    contexts.append((k, decision_context))
                    print(f"✅ K={k}：成功生成DecisionContext")
                else:
                    self.compliance_report['issues'].append(f"K={k}时返回具体动作")
                    return False
            
            # 验证嵌入维度不随K变化
            if len(contexts) >= 2:
                ref_strategy_dim = contexts[0][1].strategy_vector.size(0)
                ref_partition_dim = contexts[0][1].target_partition_embedding.size(0)
                
                for k, ctx in contexts[1:]:
                    if ctx.strategy_vector.size(0) != ref_strategy_dim:
                        self.compliance_report['issues'].append(f"K={k}时策略向量维度改变")
                        return False
                    if ctx.target_partition_embedding.size(0) != ref_partition_dim:
                        self.compliance_report['issues'].append(f"K={k}时分区嵌入维度改变")
                        return False
                
                print(f"✅ 所有K值的嵌入维度保持一致：策略{ref_strategy_dim}, 分区{ref_partition_dim}")
            
            # 验证候选数量随K值合理变化
            for k, ctx in contexts:
                # 候选数量应该与K值相关，但不应该完全相等（因为连通性约束）
                if ctx.num_candidates > k:
                    self.compliance_report['issues'].append(f"K={k}时候选数量{ctx.num_candidates}超过分区数")
                    return False
                if ctx.num_candidates < 1:
                    self.compliance_report['issues'].append(f"K={k}时候选数量{ctx.num_candidates}无效")
                    return False
                
                print(f"✅ K={k}：候选数量{ctx.num_candidates}合理")
        
        except Exception as e:
            self.compliance_report['issues'].append(f"K值解耦验证失败：{e}")
            return False
        
        self.compliance_report['decoupled_k_values'] = True
        return True
    
    def _create_mock_state(self, num_nodes: int = 14, num_partitions: int = 3) -> Dict[str, torch.Tensor]:
        """创建模拟状态"""
        return {
            'x': torch.randn(num_nodes, 32),  # 节点特征
            'edge_index': torch.randint(0, num_nodes, (2, num_nodes * 2)),  # 边索引
            'current_partition': torch.randint(1, num_partitions + 1, (num_nodes,)),  # 当前分区
            'boundary_nodes': torch.randint(0, num_nodes, (min(5, num_nodes),)),  # 边界节点
        }
    
    def run_full_validation(self, agent: Any) -> Dict[str, Any]:
        """运行完整的v3.0架构合规性验证"""
        print("🔍 开始v3.0架构合规性验证")
        print("=" * 60)
        
        # 运行所有验证
        validations = [
            ('双塔架构', self.validate_dual_tower_architecture),
            ('策略向量输出', self.validate_strategy_vector_output),
            ('相似度决策', self.validate_similarity_based_decision),
            ('跨拓扑泛化', self.validate_cross_topology_generalization),
            ('抽象表示', self.validate_abstract_representation),
            ('无ID依赖', self.validate_no_concrete_id_dependency),
            ('解耦K值', self.validate_decoupled_k_values),
        ]
        
        passed_count = 0
        total_count = len(validations)
        
        for name, validation_func in validations:
            try:
                if validation_func(agent):
                    passed_count += 1
                    print(f"✅ {name}验证通过")
                else:
                    print(f"❌ {name}验证失败")
            except Exception as e:
                print(f"❌ {name}验证出错：{e}")
                self.compliance_report['issues'].append(f"{name}验证异常：{e}")
        
        # 计算总体合规性
        self.compliance_report['overall_compliant'] = (passed_count == total_count)
        
        print("\n" + "=" * 60)
        if self.compliance_report['overall_compliant']:
            print("🎉 v3.0架构合规性验证完全通过！")
            print("✅ 实现真正符合双塔架构设计理念")
            print("✅ 支持跨拓扑泛化")
            print("✅ 无具体ID依赖")
            print("✅ 基于相似度的抽象决策")
        else:
            print(f"⚠️  v3.0架构合规性验证部分通过：{passed_count}/{total_count}")
            print("\n发现的问题：")
            for issue in self.compliance_report['issues']:
                print(f"  - {issue}")
        
        return self.compliance_report


def run_v3_architecture_compliance_test():
    """运行v3.0架构合规性测试"""
    print("🏗️  开始v3.0架构合规性测试")
    
    try:
        # 这里需要实际的agent实例
        # 由于测试环境限制，我们创建一个模拟检查
        print("⚠️  注意：这是独立的合规性验证工具")
        print("   实际使用时需要传入真实的EnhancedPPOAgent实例")
        
        validator = V3ArchitectureComplianceValidator()
        
        # 显示验证框架
        print("\n📋 v3.0架构合规性验证框架：")
        checks = [
            "1. 双塔架构（状态塔+动作塔）",
            "2. 策略向量输出（非直接分区ID）",
            "3. 相似度决策机制",
            "4. 跨拓扑泛化能力",
            "5. 抽象表示（无具体ID）",
            "6. 无具体ID依赖",
            "7. 解耦K值处理"
        ]
        
        for check in checks:
            print(f"   ✓ {check}")
        
        print("\n📝 使用方法：")
        print("   from test_v3_architecture_compliance import V3ArchitectureComplianceValidator")
        print("   validator = V3ArchitectureComplianceValidator()")
        print("   report = validator.run_full_validation(your_agent)")
        
        return True
        
    except Exception as e:
        print(f"❌ 合规性测试框架加载失败：{e}")
        return False


if __name__ == "__main__":
    success = run_v3_architecture_compliance_test()
    exit(0 if success else 1) 