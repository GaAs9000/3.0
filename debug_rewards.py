#!/usr/bin/env python3
"""
调试奖励系统 - 检查为什么总是-2.0
"""

import sys
import torch
import numpy as np
sys.path.append('code/src')

from rl.reward import RewardFunction
from torch_geometric.data import HeteroData

def test_reward_calculation():
    """测试奖励计算的详细过程"""
    print("🔍 调试奖励计算过程...")
    
    # 创建简单的测试数据
    hetero_data = HeteroData()
    hetero_data['bus'].x = torch.randn(14, 12)
    hetero_data['bus', 'connects', 'bus'].edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    ])
    hetero_data['bus', 'connects', 'bus'].edge_attr = torch.randn(13, 9)
    
    # 测试配置
    config = {
        'adaptive_quality': {
            'plateau_detection': {
                'window_size': 15,
                'min_improvement_rate': 0.005,
                'stability_threshold': 0.8,
                'min_percentile': 0.7,
                'confidence_threshold': 0.8
            },
            'efficiency_reward': {
                'lambda': 0.5,
                'early_stop_confidence': 0.85
            },
            'quality_weights': {
                'cv_weight': 0.4,
                'coupling_weight': 0.3,
                'power_weight': 0.3
            }
        }
    }
    
    # 创建奖励函数
    reward_function = RewardFunction(hetero_data, config=config)
    
    # 测试不同的分区状态
    test_partitions = [
        torch.tensor([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]),  # 均匀分区
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),  # 不均匀分区
        torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]),  # 交替分区
    ]
    
    for i, partition in enumerate(test_partitions):
        print(f"\n--- 测试分区 {i+1} ---")
        print(f"分区分布: {partition.tolist()}")
        
        # 计算质量分数
        quality_score = reward_function._compute_quality_score(partition)
        print(f"质量分数: {quality_score:.4f}")
        
        # 计算核心指标
        metrics = reward_function._compute_core_metrics(partition)
        print(f"CV: {metrics['cv']:.4f}")
        print(f"耦合比: {metrics['coupling_ratio']:.4f}")
        print(f"功率不平衡: {metrics['power_imbalance_normalized']:.4f}")
        
        # 重置并计算即时奖励
        reward_function.reset_episode()
        
        # 第一步（应该返回0）
        reward1, plateau1 = reward_function.compute_incremental_reward(partition, (0, 2))
        print(f"第一步奖励: {reward1:.4f}")
        
        # 第二步（应该有实际奖励）
        reward2, plateau2 = reward_function.compute_incremental_reward(partition, (1, 3))
        print(f"第二步奖励: {reward2:.4f}")
        
        # 计算终局奖励
        final_reward, components = reward_function.compute_final_reward(partition)
        print(f"终局奖励: {final_reward:.4f}")
        print(f"终局组件: {components}")

def test_action_validity():
    """测试动作有效性检查"""
    print("\n🔍 测试动作有效性...")
    
    # 这里需要更复杂的测试，但先简单检查
    print("动作有效性检查需要完整的环境设置")

if __name__ == "__main__":
    test_reward_calculation()
    test_action_validity()
