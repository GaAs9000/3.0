#!/usr/bin/env python3
"""
验证修复后的双塔模型性能

测试修复维度不匹配问题后的双塔模型在不同规模电网上的表现
"""

import torch
import numpy as np
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'code', 'src'))

from rl.enhanced_environment import EnhancedPowerGridPartitioningEnv
from rl.enhanced_agent import EnhancedPPOAgent
from gat import create_production_encoder
from data_processing import PowerGridDataProcessor
import pandapower.networks as pn
import pandapower as pp
from pandapower.converter import to_mpc

def load_ieee_case(case_name):
    """加载IEEE标准案例"""
    case_map = {
        'ieee14': pn.case14,
        'ieee30': pn.case30,
        'ieee57': pn.case57,
        'ieee118': pn.case118
    }
    
    if case_name not in case_map:
        raise ValueError(f"不支持的案例: {case_name}")
    
    # 加载网络
    net = case_map[case_name]()
    
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

def test_single_case(case_name, expected_k):
    """测试单个案例"""
    print(f"🔍 测试 {case_name} (K={expected_k})...")
    
    try:
        # 加载数据
        hetero_data = load_ieee_case(case_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建编码器
        encoder = create_production_encoder(hetero_data, {})
        encoder = encoder.to(device)
        
        with torch.no_grad():
            node_embeddings = encoder.encode_nodes(hetero_data)
        
        # 创建增强环境（双塔架构需要）
        env = EnhancedPowerGridPartitioningEnv(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=expected_k,
            device=device
        )
        
        # 获取状态维度
        obs, _ = env.reset()
        state_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
        
        # 【修复】创建智能体配置，确保维度正确
        agent_config = {
            'node_embedding_dim': 128,  # 与实际节点嵌入维度一致
            'region_embedding_dim': 256,  # 与实际区域嵌入维度一致
            'strategy_vector_dim': 128,
            'hidden_dim': 256,
            'dropout': 0.1,
            'learning_rate': 3e-4,
            'clip_range': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'use_two_tower': True  # 明确启用双塔架构
        }
        
        agent = EnhancedPPOAgent(
            state_dim=state_dim,
            num_partitions=expected_k,
            config=agent_config,
            device=device
        )
        
        # 测试多个episode
        results = []
        for episode in range(3):
            if episode > 0:
                obs, _ = env.reset()
            
            episode_reward = 0
            episode_start = time.time()
            
            for step in range(5):  # 限制步数
                action, _, _, _ = agent.select_action(obs, training=False)
                if action is not None:
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if done or truncated:
                        break
                else:
                    break  # 没有可用动作
            
            episode_time = time.time() - episode_start
            connectivity = info.get('metrics', {}).get('connectivity', 0.0)
            
            results.append({
                'episode': episode,
                'reward': episode_reward,
                'time': episode_time,
                'connectivity': connectivity
            })
        
        # 计算统计信息
        avg_reward = np.mean([r['reward'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        avg_connectivity = np.mean([r['connectivity'] for r in results])
        
        print(f"  ✅ 节点数: {hetero_data['bus'].x.size(0)}")
        print(f"  ✅ 分区数: {expected_k}")
        print(f"  ✅ 平均奖励: {avg_reward:.3f}")
        print(f"  ✅ 平均时间: {avg_time:.4f}s")
        print(f"  ✅ 平均连通性: {avg_connectivity:.3f}")
        
        return {
            'case': case_name,
            'num_nodes': hetero_data['bus'].x.size(0),
            'num_partitions': expected_k,
            'avg_reward': avg_reward,
            'avg_time': avg_time,
            'avg_connectivity': avg_connectivity,
            'success': True
        }
        
    except Exception as e:
        print(f"  ❌ {case_name} 测试失败: {e}")
        return {
            'case': case_name,
            'success': False,
            'error': str(e)
        }

def test_k_value_adaptability():
    """测试K值适应性"""
    print("\n🔍 测试K值适应性...")
    
    # 使用IEEE30测试不同的K值
    hetero_data = load_ieee_case('ieee30')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建编码器
    encoder = create_production_encoder(hetero_data, {})
    encoder = encoder.to(device)
    
    with torch.no_grad():
        node_embeddings = encoder.encode_nodes(hetero_data)
    
    k_values = [3, 4, 5, 6]
    results = []
    
    for k in k_values:
        print(f"  测试K={k}...")
        
        try:
            # 创建增强环境（双塔架构需要）
            env = EnhancedPowerGridPartitioningEnv(
                hetero_data=hetero_data,
                node_embeddings=node_embeddings,
                num_partitions=k,
                device=device
            )
            
            # 获取状态维度
            obs, _ = env.reset()
            state_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
            
            # 创建智能体
            agent_config = {
                'node_embedding_dim': 128,
                'region_embedding_dim': 256,
                'strategy_vector_dim': 128,
                'hidden_dim': 256,
                'dropout': 0.1,
                'use_two_tower': True
            }
            
            agent = EnhancedPPOAgent(
                state_dim=state_dim,
                num_partitions=k,
                config=agent_config,
                device=device
            )
            
            # 测试一个episode
            episode_reward = 0
            for step in range(3):
                action, _, _, _ = agent.select_action(obs, training=False)
                if action is not None:
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if done or truncated:
                        break
                else:
                    break
            
            connectivity = info.get('metrics', {}).get('connectivity', 0.0)
            
            results.append({
                'k': k,
                'reward': episode_reward,
                'connectivity': connectivity,
                'success': True
            })
            
            print(f"    ✅ 奖励: {episode_reward:.3f}, 连通性: {connectivity:.3f}")
            
        except Exception as e:
            print(f"    ❌ K={k} 失败: {e}")
            results.append({
                'k': k,
                'success': False,
                'error': str(e)
            })
    
    return results

def main():
    """主测试函数"""
    print("🚀 验证修复后的双塔模型性能")
    print("=" * 60)
    
    # 测试不同规模的电网
    test_cases = [
        {'name': 'IEEE14', 'case': 'ieee14', 'expected_k': 3},
        {'name': 'IEEE30', 'case': 'ieee30', 'expected_k': 4},
        {'name': 'IEEE57', 'case': 'ieee57', 'expected_k': 5},
    ]
    
    cross_scale_results = []
    
    # 1. 跨规模测试
    print("📊 跨规模泛化测试:")
    for test_case in test_cases:
        result = test_single_case(test_case['case'], test_case['expected_k'])
        cross_scale_results.append(result)
    
    # 2. K值适应性测试
    k_adaptability_results = test_k_value_adaptability()
    
    # 3. 计算成功率
    cross_scale_success = sum(1 for r in cross_scale_results if r.get('success', False))
    k_adaptability_success = sum(1 for r in k_adaptability_results if r.get('success', False))
    
    total_tests = len(cross_scale_results) + len(k_adaptability_results)
    total_success = cross_scale_success + k_adaptability_success
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"✅ 跨规模测试成功率: {cross_scale_success}/{len(cross_scale_results)} ({cross_scale_success/len(cross_scale_results)*100:.1f}%)")
    print(f"✅ K值适应性成功率: {k_adaptability_success}/{len(k_adaptability_results)} ({k_adaptability_success/len(k_adaptability_results)*100:.1f}%)")
    print(f"🎯 总体成功率: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
    
    if total_success == total_tests:
        print("\n🎉 所有测试都通过了！双塔模型修复成功！")
        return True
    else:
        print(f"\n⚠️ 还有 {total_tests - total_success} 个测试失败，需要进一步调试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
