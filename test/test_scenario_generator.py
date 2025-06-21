#!/usr/bin/env python3
"""
场景生成器测试脚本
验证场景生成功能是否正常工作
"""

import sys
import numpy as np
from pathlib import Path

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rl.scenario_generator import ScenarioGenerator


def create_test_case():
    """创建一个简单的测试案例"""
    case_data = {
        'bus': np.array([
            [1, 3, 100, 50, 0, 0, 1, 1.0, 0, 1.0, 1, 1.1, 0.9],
            [2, 1, 80, 40, 0, 0, 1, 1.0, 0, 1.0, 1, 1.1, 0.9],
            [3, 1, 60, 30, 0, 0, 1, 1.0, 0, 1.0, 1, 1.1, 0.9],
            [4, 1, 40, 20, 0, 0, 1, 1.0, 0, 1.0, 1, 1.1, 0.9],
        ]),
        'branch': np.array([
            [1, 2, 0.01, 0.1, 0.02, 100, 100, 100, 0, 0, 1, -360, 360],
            [2, 3, 0.02, 0.15, 0.03, 100, 100, 100, 0, 0, 1, -360, 360],
            [3, 4, 0.01, 0.1, 0.02, 100, 100, 100, 0, 0, 1, -360, 360],
            [1, 4, 0.03, 0.2, 0.04, 100, 100, 100, 0, 0, 1, -360, 360],
        ]),
        'gen': np.array([
            [1, 150, 0, 100, -100, 1.0, 100, 1, 200, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 100, 0, 50, -50, 1.0, 100, 1, 150, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        'baseMVA': 100,
        'version': '2'
    }
    return case_data


def test_basic_generation():
    """测试基本场景生成功能"""
    print("=== 测试基本场景生成 ===")
    
    base_case = create_test_case()
    generator = ScenarioGenerator(base_case, seed=42)
    
    # 生成一个随机场景
    scenario = generator.generate_random_scene()
    
    print(f"原始负荷总和: {base_case['bus'][:, 2].sum():.2f} MW")
    print(f"扰动后负荷总和: {scenario['bus'][:, 2].sum():.2f} MW")
    
    # 检查线路状态
    orig_active = (base_case['branch'][:, 10] == 1).sum()
    new_active = (scenario['branch'][:, 10] == 1).sum()
    print(f"原始活跃线路数: {orig_active}")
    print(f"扰动后活跃线路数: {new_active}")
    
    return scenario


def test_specific_perturbations():
    """测试特定类型的扰动"""
    print("\n=== 测试特定扰动类型 ===")
    
    base_case = create_test_case()
    generator = ScenarioGenerator(base_case)
    
    # 测试N-1故障
    print("\n1. N-1故障测试:")
    n1_scenario = generator.generate_random_scene(perturb_types=['n-1'])
    n1_active = (n1_scenario['branch'][:, 10] == 1).sum()
    print(f"   活跃线路数: {n1_active} (应该比原始少1)")
    
    # 测试负荷波动
    print("\n2. 负荷波动测试:")
    load_scenario = generator.generate_random_scene(perturb_types=['load_gen_fluctuation'])
    load_ratio = load_scenario['bus'][:, 2].sum() / base_case['bus'][:, 2].sum()
    print(f"   负荷变化比例: {load_ratio:.3f} (应该在0.8-1.2之间)")
    
    # 测试组合扰动
    print("\n3. 组合扰动测试:")
    both_scenario = generator.generate_random_scene(perturb_types=['both'])
    both_active = (both_scenario['branch'][:, 10] == 1).sum()
    both_load_ratio = both_scenario['bus'][:, 2].sum() / base_case['bus'][:, 2].sum()
    print(f"   活跃线路数: {both_active}")
    print(f"   负荷变化比例: {both_load_ratio:.3f}")


def test_specific_contingency():
    """测试特定线路故障"""
    print("\n=== 测试特定线路故障 ===")
    
    base_case = create_test_case()
    generator = ScenarioGenerator(base_case)
    
    # 断开第2条线路
    scenario = generator.apply_specific_contingency(base_case, branch_idx=1)
    
    print("线路状态:")
    for i, status in enumerate(scenario['branch'][:, 10]):
        orig_status = base_case['branch'][i, 10]
        from_bus = int(scenario['branch'][i, 0])
        to_bus = int(scenario['branch'][i, 1])
        print(f"   线路 {i}: Bus {from_bus} - Bus {to_bus}, "
              f"状态: {int(orig_status)} → {int(status)}")


def test_batch_generation():
    """测试批量场景生成"""
    print("\n=== 测试批量场景生成 ===")
    
    base_case = create_test_case()
    generator = ScenarioGenerator(base_case)
    
    # 生成10个场景
    scenarios = generator.generate_batch_scenarios(num_scenarios=10)
    
    print(f"生成了 {len(scenarios)} 个场景")
    
    # 统计扰动类型
    n1_count = 0
    load_count = 0
    
    for i, scenario in enumerate(scenarios):
        # 检查N-1
        if (scenario['branch'][:, 10] == 1).sum() < (base_case['branch'][:, 10] == 1).sum():
            n1_count += 1
        
        # 检查负荷变化
        load_ratio = scenario['bus'][:, 2].sum() / base_case['bus'][:, 2].sum()
        if abs(load_ratio - 1.0) > 0.01:
            load_count += 1
    
    print(f"包含N-1故障的场景: {n1_count}")
    print(f"包含负荷波动的场景: {load_count}")


def test_reproducibility():
    """测试随机种子的可重复性"""
    print("\n=== 测试可重复性 ===")
    
    base_case = create_test_case()
    
    # 使用相同种子的两个生成器
    gen1 = ScenarioGenerator(base_case, seed=123)
    gen2 = ScenarioGenerator(base_case, seed=123)
    
    # 生成场景
    scenario1 = gen1.generate_random_scene()
    scenario2 = gen2.generate_random_scene()
    
    # 比较负荷
    load1 = scenario1['bus'][:, 2].sum()
    load2 = scenario2['bus'][:, 2].sum()
    
    # 比较线路状态
    branch1 = scenario1['branch'][:, 10].sum()
    branch2 = scenario2['branch'][:, 10].sum()
    
    print(f"场景1负荷: {load1:.2f}, 场景2负荷: {load2:.2f}")
    print(f"场景1活跃线路: {branch1}, 场景2活跃线路: {branch2}")
    print(f"是否相同: {load1 == load2 and branch1 == branch2}")


def main():
    """运行所有测试"""
    print("🧪 场景生成器测试开始\n")
    
    test_basic_generation()
    test_specific_perturbations()
    test_specific_contingency()
    test_batch_generation()
    test_reproducibility()
    
    print("\n✅ 所有测试完成!")


if __name__ == '__main__':
    main() 