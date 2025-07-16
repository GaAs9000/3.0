"""
Tensor一致性测试

验证v3.0架构修复了原始的tensor堆叠错误，确保：
1. DecisionContext批量处理的一致性
2. PPO训练中的tensor堆叠不会出错
3. 索引映射的正确性
4. 批量log_prob计算的稳定性
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any
import sys
import os

# 添加src路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl.decision_context import AbstractDecisionContext, DecisionContextBatch, create_decision_context
from rl.fast_memory import FastPPOMemory
from rl.agent import PPOAgent
from rl.compatibility import CompatibilityManager, CompatibilityConfig

class TestTensorConsistency:
    """Tensor一致性测试类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.device = torch.device('cpu')  # 使用CPU以便调试
        self.embedding_dim = 64
        self.strategy_dim = 32
        self.partition_dim = 32  # 策略向量和分区嵌入应该同维度
        
    def create_mock_decision_context(self, num_candidates: int = 3) -> AbstractDecisionContext:
        """创建模拟的决策上下文"""
        # 创建一致的测试数据
        strategy_vector = torch.randn(self.strategy_dim)
        candidate_embeddings = torch.randn(num_candidates, self.partition_dim)
        
        # 计算真实的相似度
        similarities = torch.cosine_similarity(
            strategy_vector.unsqueeze(0), 
            candidate_embeddings, 
            dim=1
        )
        
        # 选择相似度最高的作为目标
        selected_idx = torch.argmax(similarities).item()
        target_partition_embedding = candidate_embeddings[selected_idx].clone()
        
        # 计算排名：在降序排序中的位置
        sorted_indices = torch.argsort(similarities, descending=True)
        selected_rank = (sorted_indices == selected_idx).nonzero(as_tuple=True)[0].item()
        
        return AbstractDecisionContext(
            node_embedding=torch.randn(self.embedding_dim),
            strategy_vector=strategy_vector,
            target_partition_embedding=target_partition_embedding,
            candidate_embeddings=candidate_embeddings,
            selection_similarities=similarities,
            selected_rank=selected_rank,
            decision_timestamp=1642345678.0,
            num_candidates=num_candidates,
            node_context={'node_id': 0, 'available_partitions': list(range(num_candidates))}
        )
    
    def test_decision_context_consistency(self):
        """测试单个DecisionContext的内部一致性"""
        print("\n=== 测试DecisionContext内部一致性 ===")
        
        # 创建测试上下文
        ctx = self.create_mock_decision_context(num_candidates=5)
        
        # 验证基本属性
        assert ctx.node_embedding.size(0) == self.embedding_dim
        assert ctx.strategy_vector.size(0) == self.strategy_dim
        assert ctx.candidate_embeddings.size() == (5, self.partition_dim)
        assert ctx.selection_similarities.size(0) == 5
        assert ctx.num_candidates == 5
        
        # 验证内部一致性
        assert ctx.validate_consistency(), "DecisionContext内部一致性验证失败"
        
        print("✅ DecisionContext内部一致性测试通过")
    
    def test_decision_context_batch_consistency(self):
        """测试DecisionContextBatch的批量处理一致性"""
        print("\n=== 测试DecisionContextBatch批量处理一致性 ===")
        
        # 创建不同候选数量的上下文（模拟现实场景）
        contexts = [
            self.create_mock_decision_context(num_candidates=3),
            self.create_mock_decision_context(num_candidates=4),
            self.create_mock_decision_context(num_candidates=2),
            self.create_mock_decision_context(num_candidates=5),
        ]
        
        batch = DecisionContextBatch(contexts=contexts, batch_size=4)
        
        # 测试批量log_prob计算
        try:
            log_probs = batch.compute_batch_log_probs(temperature=1.0)
            assert log_probs.size(0) == 4, f"期望4个log_prob，实际得到{log_probs.size(0)}"
            assert torch.all(torch.isfinite(log_probs)), "log_prob包含无效值"
            print(f"✅ 批量log_prob计算成功：{log_probs}")
            
        except Exception as e:
            pytest.fail(f"批量log_prob计算失败：{e}")
        
        # 测试批量一致性验证
        consistency_results = batch.validate_batch_consistency()
        assert all(consistency_results), f"批量一致性验证失败：{consistency_results}"
        
        print("✅ DecisionContextBatch批量处理一致性测试通过")
    
    def test_memory_storage_consistency(self):
        """测试内存存储的一致性"""
        print("\n=== 测试内存存储一致性 ===")
        
        memory = FastPPOMemory(capacity=100, device=self.device)
        
        # 存储混合格式数据（v3.0和传统）
        contexts_data = []
        
        for i in range(10):
            state = {'x': torch.randn(14, 32), 'current_partition': torch.randint(1, 4, (14,))}
            
            if i % 2 == 0:
                # 存储v3.0格式
                ctx = self.create_mock_decision_context(num_candidates=3)
                memory.store(
                    state=state,
                    decision_context=ctx,
                    reward=float(i),
                    value=float(i * 2),
                    done=(i == 9)
                )
                contexts_data.append(('v3', ctx))
            else:
                # 存储传统格式
                action = (i % 14, (i % 3) + 1)
                memory.store(
                    state=state,
                    action=action,
                    reward=float(i),
                    log_prob=0.1,
                    value=float(i * 2),
                    done=(i == 9)
                )
                contexts_data.append(('legacy', action))
        
        # 验证存储
        assert memory.size == 10, f"期望存储10个样本，实际{memory.size}"
        assert memory.has_decision_contexts(), "应该包含决策上下文"
        
        # 获取v3.0格式的批次
        states, decision_contexts, rewards, values, dones = memory.get_decision_contexts_batch()
        
        # 验证v3.0批次的一致性
        assert len(decision_contexts) == 5, f"期望5个决策上下文，实际{len(decision_contexts)}"
        assert len(states) == 5, f"期望5个状态，实际{len(states)}"
        
        print(f"✅ v3.0格式存储了{len(decision_contexts)}个上下文")
        print(f"✅ 传统格式存储了{memory.size - len(decision_contexts)}个动作")
        
        print("✅ 内存存储一致性测试通过")
    
    def test_ppo_batch_processing(self):
        """测试PPO批量处理的tensor一致性"""
        print("\n=== 测试PPO批量处理tensor一致性 ===")
        
        # 创建模拟的PPO训练数据
        batch_size = 8
        contexts = []
        
        for i in range(batch_size):
            # 变化的候选数量，模拟实际训练中的情况
            num_candidates = np.random.randint(2, 6)
            ctx = self.create_mock_decision_context(num_candidates=num_candidates)
            contexts.append(ctx)
        
        # 创建DecisionContextBatch
        batch = DecisionContextBatch(contexts=contexts, batch_size=batch_size)
        
        # 测试批量log_prob计算（这是原来容易出错的地方）
        try:
            # 多次计算以测试稳定性
            for temp in [0.5, 1.0, 1.5, 2.0]:
                log_probs = batch.compute_batch_log_probs(temperature=temp)
                
                # 验证输出形状
                assert log_probs.shape == (batch_size,), f"温度{temp}时形状错误：{log_probs.shape}"
                
                # 验证数值有效性
                assert torch.all(torch.isfinite(log_probs)), f"温度{temp}时包含无效值"
                
                print(f"✅ 温度{temp}：log_probs形状{log_probs.shape}，范围[{log_probs.min():.3f}, {log_probs.max():.3f}]")
        
        except Exception as e:
            pytest.fail(f"PPO批量处理失败：{e}")
        
        print("✅ PPO批量处理tensor一致性测试通过")
    
    def test_original_tensor_stacking_error_scenario(self):
        """测试原始tensor堆叠错误的场景"""
        print("\n=== 测试原始tensor堆叠错误场景的修复 ===")
        
        # 模拟原始错误场景：不同大小的tensor列表
        # 在v3.0架构中，这应该被完全避免
        
        irregular_contexts = []
        
        # 创建一些"正常"的上下文
        for i in range(5):
            ctx = self.create_mock_decision_context(num_candidates=3)
            irregular_contexts.append(ctx)
        
        # 创建一些"异常"的上下文（不同候选数量）
        for candidates in [1, 2, 4, 6]:
            ctx = self.create_mock_decision_context(num_candidates=candidates)
            irregular_contexts.append(ctx)
        
        # 在原来的系统中，这种情况可能导致堆叠错误
        # 在v3.0中，应该能够正确处理
        try:
            batch = DecisionContextBatch(contexts=irregular_contexts, batch_size=len(irregular_contexts))
            
            # 关键测试：批量计算不应该失败
            log_probs = batch.compute_batch_log_probs()
            
            assert log_probs.shape == (len(irregular_contexts),), f"不规则批次形状错误：{log_probs.shape}"
            assert torch.all(torch.isfinite(log_probs)), "不规则批次包含无效值"
            
            print(f"✅ 不规则批次处理成功：{len(irregular_contexts)}个上下文，候选数量范围1-6")
            print(f"✅ log_probs: {log_probs}")
            
        except Exception as e:
            pytest.fail(f"不规则批次处理失败（原始错误场景）：{e}")
        
        print("✅ 原始tensor堆叠错误场景修复验证通过")
    
    def test_backward_compatibility_tensor_consistency(self):
        """测试向后兼容性中的tensor一致性"""
        print("\n=== 测试向后兼容性tensor一致性 ===")
        
        compat_mgr = CompatibilityManager()
        
        # 测试混合格式的处理
        mixed_actions = [
            (1, 2),  # 传统action tuple
            self.create_mock_decision_context(3),  # v3.0 context
            (3, 1),  # 另一个传统action
            self.create_mock_decision_context(4),  # 另一个v3.0 context
        ]
        
        # 测试格式检测的一致性
        for i, action in enumerate(mixed_actions):
            format_type = compat_mgr.detect_action_format(action)
            
            if i % 2 == 0:
                assert format_type == 'legacy_tuple', f"索引{i}应该是legacy_tuple，实际是{format_type}"
            else:
                assert format_type == 'v3_context', f"索引{i}应该是v3_context，实际是{format_type}"
        
        print("✅ 混合格式检测一致性通过")
        
        # 测试具体动作确保的一致性
        concrete_actions = []
        for action in mixed_actions:
            concrete = compat_mgr.ensure_concrete_action(action)
            concrete_actions.append(concrete)
        
        # 所有具体动作都应该是有效的tuple
        for i, concrete in enumerate(concrete_actions):
            assert isinstance(concrete, tuple), f"索引{i}的具体动作不是tuple：{type(concrete)}"
            assert len(concrete) == 2, f"索引{i}的具体动作长度不是2：{len(concrete)}"
            assert isinstance(concrete[0], int), f"索引{i}的node_id不是int：{type(concrete[0])}"
            assert isinstance(concrete[1], int), f"索引{i}的partition_id不是int：{type(concrete[1])}"
        
        print(f"✅ 具体动作确保一致性通过：{concrete_actions}")
        print("✅ 向后兼容性tensor一致性测试通过")
    
    def test_end_to_end_consistency(self):
        """端到端一致性测试"""
        print("\n=== 端到端一致性测试 ===")
        
        # 创建完整的工作流程
        memory = FastPPOMemory(capacity=50, device=self.device)
        
        # 模拟一个episode的数据收集
        episode_contexts = []
        
        for step in range(20):
            state = {
                'x': torch.randn(14, 32),
                'current_partition': torch.randint(1, 4, (14,)),
                'boundary_nodes': torch.randint(0, 14, (np.random.randint(2, 6),))
            }
            
            # 创建决策上下文
            num_candidates = np.random.randint(2, 5)
            ctx = self.create_mock_decision_context(num_candidates=num_candidates)
            
            # 存储到内存
            memory.store(
                state=state,
                decision_context=ctx,
                reward=np.random.random(),
                value=np.random.random(),
                done=(step == 19)
            )
            
            episode_contexts.append(ctx)
        
        # 获取批次进行"训练"
        states, contexts, rewards, values, dones = memory.get_decision_contexts_batch()
        
        # 验证批次完整性
        assert len(contexts) == 20, f"期望20个上下文，实际{len(contexts)}"
        assert len(rewards) == 20, f"期望20个奖励，实际{len(rewards)}"
        
        # 创建批次并计算log_probs
        batch = DecisionContextBatch(contexts=contexts, batch_size=len(contexts))
        log_probs = batch.compute_batch_log_probs()
        
        # 验证最终结果
        assert log_probs.shape == (20,), f"log_probs形状错误：{log_probs.shape}"
        assert torch.all(torch.isfinite(log_probs)), "log_probs包含无效值"
        
        # 模拟PPO计算（简化版）
        old_log_probs = torch.randn(20)
        advantages = torch.randn(20)
        
        try:
            # 这是原来容易出错的计算
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            
            assert torch.all(torch.isfinite(ratio)), "ratio包含无效值"
            assert torch.all(torch.isfinite(surr1)), "surr1包含无效值"
            
            print(f"✅ PPO计算成功：ratio范围[{ratio.min():.3f}, {ratio.max():.3f}]")
            
        except Exception as e:
            pytest.fail(f"端到端PPO计算失败：{e}")
        
        print("✅ 端到端一致性测试通过")


def run_tensor_consistency_tests():
    """运行所有tensor一致性测试"""
    print("🔬 开始Tensor一致性测试")
    print("=" * 60)
    
    test_suite = TestTensorConsistency()
    test_suite.setup_method()
    
    try:
        # 运行所有测试
        test_suite.test_decision_context_consistency()
        test_suite.test_decision_context_batch_consistency()
        test_suite.test_memory_storage_consistency()
        test_suite.test_ppo_batch_processing()
        test_suite.test_original_tensor_stacking_error_scenario()
        test_suite.test_backward_compatibility_tensor_consistency()
        test_suite.test_end_to_end_consistency()
        
        print("\n" + "=" * 60)
        print("🎉 所有Tensor一致性测试通过！")
        print("✅ v3.0架构成功修复了原始的tensor堆叠错误")
        print("✅ 批量处理稳定可靠")
        print("✅ 向后兼容性完整")
        print("✅ 端到端工作流程正常")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tensor_consistency_tests()
    exit(0 if success else 1) 