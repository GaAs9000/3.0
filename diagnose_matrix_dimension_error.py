#!/usr/bin/env python3
"""
矩阵维度不匹配错误诊断脚本

专门用于诊断和修复 'mat1 and mat2 shapes cannot be multiplied (1x256 and 12x256)' 错误
"""

import torch
import numpy as np
import sys
import os
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), 'code', 'src'))

from rl.environment import PowerGridPartitioningEnv
from rl.enhanced_agent import EnhancedPPOAgent
from gat import create_production_encoder
from data_processing import PowerGridDataProcessor
import pandapower.networks as pn
import pandapower as pp
from pandapower.converter import to_mpc

def load_test_case():
    """加载测试案例"""
    print("🔍 加载IEEE14测试案例...")
    
    # 加载网络
    net = pn.case14()
    
    # 运行潮流分析
    try:
        if getattr(net, "res_bus", None) is None or net.res_bus.empty:
            pp.runpp(net, init="flat", calculate_voltage_angles=False, numba=False)
    except Exception:
        pass
    
    # 转换为MATPOWER格式
    mpc_wrap = to_mpc(net)
    mpc = mpc_wrap["mpc"] if isinstance(mpc_wrap, dict) and "mpc" in mpc_wrap else mpc_wrap
    
    # 转换为异构数据
    processor = PowerGridDataProcessor()
    hetero_data = processor.graph_from_mpc(mpc)
    
    return hetero_data

def create_test_environment(hetero_data, num_partitions=3):
    """创建测试环境"""
    print(f"🔧 创建测试环境，分区数: {num_partitions}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建编码器
    encoder = create_production_encoder(hetero_data, {})
    encoder = encoder.to(device)
    
    with torch.no_grad():
        node_embeddings = encoder.encode_nodes(hetero_data)
    
    # 创建环境
    env = PowerGridPartitioningEnv(
        hetero_data=hetero_data,
        node_embeddings=node_embeddings,
        num_partitions=num_partitions,
        device=device
    )
    
    return env, device

def create_test_agent(env, device):
    """创建测试智能体"""
    print("🤖 创建测试智能体...")
    
    # 获取状态维度
    obs, _ = env.reset()
    state_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
    
    print(f"   状态维度: {state_dim}")
    
    # 创建智能体配置
    agent_config = {
        'node_embedding_dim': 128,
        'region_embedding_dim': 256,
        'strategy_vector_dim': 128,
        'hidden_dim': 256,
        'dropout': 0.1,
        'learning_rate': 3e-4,
        'clip_range': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01
    }
    
    print(f"   智能体配置: {agent_config}")
    
    agent = EnhancedPPOAgent(
        state_dim=state_dim,
        num_partitions=3,
        config=agent_config,
        device=device
    )
    
    return agent, obs

def diagnose_actor_network_dimensions(agent, obs):
    """诊断Actor网络维度问题"""
    print("\n🔍 诊断Actor网络维度...")
    
    try:
        # 检查智能体的维度配置
        print(f"   node_embedding_dim: {agent.node_embedding_dim}")
        print(f"   region_embedding_dim: {agent.region_embedding_dim}")
        print(f"   strategy_vector_dim: {agent.strategy_vector_dim}")
        
        # 检查Actor网络的配置
        actor = agent.actor
        print(f"   Actor类型: {type(actor).__name__}")
        
        if hasattr(actor, 'node_embedding_dim'):
            print(f"   Actor.node_embedding_dim: {actor.node_embedding_dim}")
            print(f"   Actor.region_embedding_dim: {actor.region_embedding_dim}")
            print(f"   Actor.strategy_vector_dim: {actor.strategy_vector_dim}")
        
        # 检查网络层的输入维度
        if hasattr(actor, 'node_selector'):
            node_selector_input_dim = actor.node_selector[0].in_features
            print(f"   node_selector输入维度: {node_selector_input_dim}")
            
        if hasattr(actor, 'strategy_generator'):
            strategy_generator_input_dim = actor.strategy_generator[0].in_features
            print(f"   strategy_generator输入维度: {strategy_generator_input_dim}")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Actor网络维度检查失败: {e}")
        traceback.print_exc()
        return False

def diagnose_batched_state_dimensions(agent, obs):
    """诊断批处理状态维度问题"""
    print("\n🔍 诊断批处理状态维度...")
    
    try:
        # 准备批处理状态
        batched_state = agent._prepare_batched_state([obs])
        
        print(f"   node_embeddings形状: {batched_state.node_embeddings.shape}")
        print(f"   region_embeddings形状: {batched_state.region_embeddings.shape}")
        print(f"   boundary_nodes形状: {batched_state.boundary_nodes.shape}")
        print(f"   boundary_nodes数量: {len(batched_state.boundary_nodes)}")
        
        if hasattr(batched_state, 'node_batch_idx'):
            print(f"   node_batch_idx形状: {batched_state.node_batch_idx.shape}")
        if hasattr(batched_state, 'region_batch_idx'):
            print(f"   region_batch_idx形状: {batched_state.region_batch_idx.shape}")
            
        return batched_state
        
    except Exception as e:
        print(f"   ❌ 批处理状态准备失败: {e}")
        traceback.print_exc()
        return None

def diagnose_actor_forward_pass(agent, batched_state):
    """诊断Actor前向传播过程"""
    print("\n🔍 诊断Actor前向传播过程...")
    
    try:
        actor = agent.actor
        bs = batched_state
        
        # 检查输入数据
        print(f"   输入检查:")
        print(f"     region_embeddings: {bs.region_embeddings.shape}")
        print(f"     boundary_nodes: {bs.boundary_nodes.shape}")
        print(f"     node_embeddings: {bs.node_embeddings.shape}")
        
        # 步骤1: 计算全局上下文
        print(f"   步骤1: 计算全局上下文...")
        from torch_scatter import scatter_mean
        
        global_context_per_batch = actor.global_encoder(
            scatter_mean(bs.region_embeddings, bs.region_batch_idx, dim=0)
        )
        print(f"     global_context_per_batch: {global_context_per_batch.shape}")
        
        # 步骤2: 获取边界节点信息
        print(f"   步骤2: 获取边界节点信息...")
        boundary_batch_idx = bs.node_batch_idx[bs.boundary_nodes]
        print(f"     boundary_batch_idx: {boundary_batch_idx.shape}")
        
        global_context = global_context_per_batch[boundary_batch_idx]
        print(f"     global_context: {global_context.shape}")
        
        # 步骤3: 获取边界节点嵌入
        print(f"   步骤3: 获取边界节点嵌入...")
        boundary_embeddings = bs.node_embeddings[bs.boundary_nodes]
        print(f"     boundary_embeddings: {boundary_embeddings.shape}")
        
        # 步骤4: 组合特征 - 这里可能是问题所在
        print(f"   步骤4: 组合特征...")
        print(f"     准备拼接: boundary_embeddings{boundary_embeddings.shape} + global_context{global_context.shape}")
        
        combined_features = torch.cat([boundary_embeddings, global_context], dim=1)
        print(f"     combined_features: {combined_features.shape}")
        
        # 步骤5: 通过网络层
        print(f"   步骤5: 通过网络层...")
        
        # 检查node_selector
        print(f"     node_selector输入维度期望: {actor.node_selector[0].in_features}")
        print(f"     combined_features实际维度: {combined_features.shape[1]}")
        
        if combined_features.shape[1] != actor.node_selector[0].in_features:
            print(f"     ❌ 维度不匹配！期望{actor.node_selector[0].in_features}，实际{combined_features.shape[1]}")
            return False
        
        node_logits = actor.node_selector(combined_features).squeeze(-1)
        print(f"     node_logits: {node_logits.shape}")
        
        # 检查strategy_generator
        strategy_vectors = actor.strategy_generator(combined_features)
        print(f"     strategy_vectors: {strategy_vectors.shape}")
        
        print(f"   ✅ Actor前向传播成功！")
        return True
        
    except Exception as e:
        print(f"   ❌ Actor前向传播失败: {e}")
        traceback.print_exc()
        return False

def diagnose_select_action_process(agent, obs):
    """诊断select_action过程"""
    print("\n🔍 诊断select_action过程...")
    
    try:
        # 逐步执行select_action
        with torch.no_grad():
            # 1. 准备批处理状态
            print("   步骤1: 准备批处理状态...")
            batched_state = agent._prepare_batched_state([obs])
            
            # 2. 获取价值估计
            print("   步骤2: 获取价值估计...")
            value = agent.critic(batched_state).item()
            print(f"     value: {value}")
            
            # 3. 检查边界节点
            print("   步骤3: 检查边界节点...")
            if batched_state.boundary_nodes.numel() == 0:
                print("     ❌ 没有边界节点！")
                return False
            
            # 4. 获取节点logits和策略向量 - 这里可能出错
            print("   步骤4: 获取节点logits和策略向量...")
            node_logits, strategy_vectors = agent.actor(batched_state)
            print(f"     node_logits: {node_logits.shape}")
            print(f"     strategy_vectors: {strategy_vectors.shape}")
            
            print(f"   ✅ select_action过程成功！")
            return True
            
    except Exception as e:
        print(f"   ❌ select_action过程失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主诊断函数"""
    print("🚀 开始矩阵维度不匹配错误诊断")
    print("=" * 60)
    
    try:
        # 1. 加载测试数据
        hetero_data = load_test_case()
        
        # 2. 创建测试环境
        env, device = create_test_environment(hetero_data)
        
        # 3. 创建测试智能体
        agent, obs = create_test_agent(env, device)
        
        # 4. 诊断Actor网络维度
        actor_ok = diagnose_actor_network_dimensions(agent, obs)
        
        # 5. 诊断批处理状态维度
        batched_state = diagnose_batched_state_dimensions(agent, obs)
        
        if batched_state is not None:
            # 6. 诊断Actor前向传播
            forward_ok = diagnose_actor_forward_pass(agent, batched_state)
            
            if forward_ok:
                # 7. 诊断完整的select_action过程
                select_action_ok = diagnose_select_action_process(agent, obs)
                
                if select_action_ok:
                    print("\n🎉 所有诊断都通过了！没有发现维度问题。")
                    return True
        
        print("\n❌ 发现了维度问题，需要修复。")
        return False
        
    except Exception as e:
        print(f"\n💥 诊断过程中发生严重错误: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
