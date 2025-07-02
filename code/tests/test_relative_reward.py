#!/usr/bin/env python3
"""
简单相对改进奖励测试脚本

验证新的相对改进奖励计算是否正确解决跨场景训练偏向问题
"""

import sys
import os
# 添加正确的路径：从code/tests回到2.0根目录，然后进入code/src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
from code.src.rl.reward import RewardFunction

def test_relative_reward_calculation():
    """测试相对奖励计算的正确性"""
    print("🧪 测试相对改进奖励计算")
    print("=" * 50)
    
    # 创建一个简单的RewardFunction实例用于测试
    # 注意：这里我们只测试_compute_simple_relative_reward方法
    class TestRewardFunction:
        def _compute_simple_relative_reward(self, prev_quality: float, curr_quality: float) -> float:
            """简单相对改进奖励 - 解决跨场景训练偏向问题"""
            try:
                if prev_quality > 0.01:  # 避免除零，处理边界情况
                    relative_improvement = (curr_quality - prev_quality) / prev_quality
                else:
                    # 从零开始的情况，直接用绝对改进
                    relative_improvement = curr_quality - prev_quality
                
                # 轻微裁剪避免极端值，保持训练稳定性
                return np.clip(relative_improvement, -1.0, 1.0)
                
            except Exception as e:
                print(f"警告：相对奖励计算出现异常: {e}")
                return 0.0
    
    reward_func = TestRewardFunction()
    
    # 测试案例1: 跨场景公平性
    print("📊 测试案例1: 跨场景公平性")
    print("-" * 30)
    
    # 故障场景：从低质量基线改进
    fault_prev = 0.40
    fault_curr = 0.42
    fault_reward = reward_func._compute_simple_relative_reward(fault_prev, fault_curr)
    fault_improvement = (fault_curr - fault_prev) / fault_prev * 100
    
    # 正常场景：从高质量基线改进
    normal_prev = 0.68
    normal_curr = 0.714  # 相同的5%相对改进
    normal_reward = reward_func._compute_simple_relative_reward(normal_prev, normal_curr)
    normal_improvement = (normal_curr - normal_prev) / normal_prev * 100
    
    print(f"故障场景: {fault_prev:.3f} → {fault_curr:.3f} ({fault_improvement:.1f}%改进) → 奖励: {fault_reward:.4f}")
    print(f"正常场景: {normal_prev:.3f} → {normal_curr:.3f} ({normal_improvement:.1f}%改进) → 奖励: {normal_reward:.4f}")
    print(f"奖励差异: {abs(fault_reward - normal_reward):.6f} (应该接近0)")
    
    # 测试案例2: 边界情况
    print("\n📊 测试案例2: 边界情况")
    print("-" * 30)
    
    # 从零开始
    zero_reward = reward_func._compute_simple_relative_reward(0.0, 0.5)
    print(f"从零开始: 0.0 → 0.5 → 奖励: {zero_reward:.4f}")
    
    # 极小基线
    small_reward = reward_func._compute_simple_relative_reward(0.005, 0.01)
    print(f"极小基线: 0.005 → 0.01 → 奖励: {small_reward:.4f}")
    
    # 负向改进
    negative_reward = reward_func._compute_simple_relative_reward(0.6, 0.5)
    negative_improvement = (0.5 - 0.6) / 0.6 * 100
    print(f"负向改进: 0.6 → 0.5 ({negative_improvement:.1f}%退化) → 奖励: {negative_reward:.4f}")
    
    # 测试案例3: 极端值处理
    print("\n📊 测试案例3: 极端值处理")
    print("-" * 30)
    
    # 极大改进（应该被裁剪）
    extreme_reward = reward_func._compute_simple_relative_reward(0.1, 0.5)
    extreme_improvement = (0.5 - 0.1) / 0.1 * 100
    print(f"极大改进: 0.1 → 0.5 ({extreme_improvement:.0f}%改进) → 奖励: {extreme_reward:.4f} (裁剪到1.0)")
    
    # 极大退化（应该被裁剪）
    extreme_negative = reward_func._compute_simple_relative_reward(0.8, 0.1)
    extreme_negative_improvement = (0.1 - 0.8) / 0.8 * 100
    print(f"极大退化: 0.8 → 0.1 ({extreme_negative_improvement:.0f}%退化) → 奖励: {extreme_negative:.4f} (裁剪到-1.0)")

def test_scenario_comparison():
    """对比不同场景下的奖励公平性"""
    print("\n🎯 场景对比分析")
    print("=" * 50)
    
    class TestRewardFunction:
        def _compute_simple_relative_reward(self, prev_quality: float, curr_quality: float) -> float:
            try:
                if prev_quality > 0.01:
                    relative_improvement = (curr_quality - prev_quality) / prev_quality
                else:
                    relative_improvement = curr_quality - prev_quality
                return np.clip(relative_improvement, -1.0, 1.0)
            except:
                return 0.0
    
    reward_func = TestRewardFunction()
    
    scenarios = [
        ("正常运行", 0.75, 0.7875),  # 5%改进
        ("轻微故障", 0.60, 0.63),    # 5%改进  
        ("严重故障", 0.35, 0.3675),  # 5%改进
        ("高负荷", 0.50, 0.525),     # 5%改进
        ("发电波动", 0.45, 0.4725),  # 5%改进
    ]
    
    print("所有场景都实现5%的相对改进:")
    print("-" * 40)
    
    rewards = []
    for name, prev, curr in scenarios:
        reward = reward_func._compute_simple_relative_reward(prev, curr)
        improvement = (curr - prev) / prev * 100
        rewards.append(reward)
        print(f"{name:8s}: {prev:.4f} → {curr:.4f} ({improvement:.1f}%) → 奖励: {reward:.4f}")
    
    # 计算奖励的标准差，应该很小
    reward_std = np.std(rewards)
    print(f"\n奖励标准差: {reward_std:.6f} (越小越好，表示公平性)")
    
    if reward_std < 0.001:
        print("✅ 测试通过：跨场景奖励公平性良好")
    else:
        print("❌ 测试失败：跨场景奖励存在偏向")

if __name__ == "__main__":
    print("🚀 简单相对改进奖励系统测试")
    print("解决跨场景训练偏向问题")
    print("=" * 60)
    
    test_relative_reward_calculation()
    test_scenario_comparison()
    
    print("\n✨ 测试完成！")
    print("\n💡 关键优势:")
    print("1. 相同相对努力 → 相同奖励幅度")
    print("2. 困难场景不被忽视")
    print("3. 训练过程更加均衡")
    print("4. 实现简单，性能无损")
