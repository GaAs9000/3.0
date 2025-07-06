"""
ActionSpace和StateManager性能测试

验证稀疏张量实现的功能正确性和性能提升
"""

import torch
import time
import warnings
from torch_geometric.data import HeteroData
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rl.action_space import ActionSpace
from src.rl.state import StateManager


def create_test_graph(num_nodes=1000, edge_density=0.1, device='cuda'):
    """创建测试用的异构图"""
    torch.manual_seed(42)  # 固定随机种子
    
    hetero_data = HeteroData()
    
    # 创建两种类型的节点
    num_type1 = num_nodes // 2
    num_type2 = num_nodes - num_type1
    
    hetero_data['node_type1'].x = torch.randn(num_type1, 64, device=device)
    hetero_data['node_type2'].x = torch.randn(num_type2, 64, device=device)
    
    # 创建边
    num_edges = int(num_nodes * num_nodes * edge_density)
    
    # type1到type2的边
    src = torch.randint(0, num_type1, (num_edges//2,), device=device)
    dst = torch.randint(0, num_type2, (num_edges//2,), device=device)
    hetero_data['node_type1', 'connects', 'node_type2'].edge_index = torch.stack([src, dst])
    
    # type2到type1的边
    src = torch.randint(0, num_type2, (num_edges//2,), device=device)
    dst = torch.randint(0, num_type1, (num_edges//2,), device=device)
    hetero_data['node_type2', 'connects', 'node_type1'].edge_index = torch.stack([src, dst])
    
    return hetero_data


def create_random_partition(num_nodes, num_partitions, device='cuda'):
    """创建随机分区分配"""
    return torch.randint(1, num_partitions + 1, (num_nodes,), device=device)


def save_baseline_results(action_space, state_manager, current_partition, boundary_nodes):
    """保存基线结果用于功能验证"""
    # 保存valid_actions
    valid_actions = action_space.get_valid_actions(current_partition, boundary_nodes)
    
    # 保存action_mask
    action_mask = action_space.get_action_mask(current_partition, boundary_nodes)
    
    # 保存边界节点
    boundary_nodes_computed = state_manager.get_boundary_nodes()
    
    results = {
        'valid_actions': valid_actions,
        'action_mask': action_mask,
        'boundary_nodes': boundary_nodes_computed,
        'partition': current_partition
    }
    
    torch.save(results, 'baseline_results.pt')
    print("✅ 基线结果已保存")
    return results


def verify_results(action_space, state_manager, baseline_results):
    """验证新实现与基线结果的一致性"""
    current_partition = baseline_results['partition']
    
    # 重新计算边界节点
    state_manager.reset(current_partition)
    new_boundary_nodes = state_manager.get_boundary_nodes()
    
    # 比较边界节点
    baseline_boundary = baseline_results['boundary_nodes']
    if torch.equal(torch.sort(new_boundary_nodes)[0], torch.sort(baseline_boundary)[0]):
        print("✅ 边界节点计算正确")
    else:
        print("❌ 边界节点计算不一致")
        print(f"   基线: {baseline_boundary}")
        print(f"   新实现: {new_boundary_nodes}")
    
    # 比较valid_actions
    new_valid_actions = action_space.get_valid_actions(current_partition, new_boundary_nodes)
    baseline_valid_actions = baseline_results['valid_actions']
    
    if set(new_valid_actions) == set(baseline_valid_actions):
        print("✅ 有效动作计算正确")
    else:
        print("❌ 有效动作计算不一致")
        print(f"   基线数量: {len(baseline_valid_actions)}")
        print(f"   新实现数量: {len(new_valid_actions)}")
    
    # 比较action_mask
    new_action_mask = action_space.get_action_mask(current_partition, new_boundary_nodes)
    baseline_mask = baseline_results['action_mask']
    
    if torch.equal(new_action_mask, baseline_mask):
        print("✅ 动作掩码计算正确")
    else:
        print("❌ 动作掩码计算不一致")
        diff_count = (new_action_mask != baseline_mask).sum().item()
        print(f"   不同元素数量: {diff_count}")


def performance_test(num_nodes_list=[100, 500, 1000, 2000], num_iterations=10):
    """性能测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 性能测试 (设备: {device})")
    
    results = []
    
    for num_nodes in num_nodes_list:
        print(f"\n📊 测试规模: {num_nodes} 节点")
        
        # 创建测试数据
        hetero_data = create_test_graph(num_nodes, device=device)
        num_partitions = max(5, num_nodes // 100)
        
        # 创建node_embeddings
        node_embeddings = {
            'node_type1': torch.randn(hetero_data['node_type1'].x.shape[0], 64, device=device),
            'node_type2': torch.randn(hetero_data['node_type2'].x.shape[0], 64, device=device)
        }
        
        # 初始化
        action_space = ActionSpace(hetero_data, num_partitions, device)
        state_manager = StateManager(hetero_data, node_embeddings, device)
        
        # 创建随机分区
        current_partition = create_random_partition(num_nodes, num_partitions, device)
        state_manager.reset(current_partition)
        
        # 测试边界节点计算
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            state_manager._compute_boundary_nodes()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        print(f"   边界节点计算平均时间: {avg_time*1000:.2f} ms")
        
        # 测试有效动作计算
        boundary_nodes = state_manager.get_boundary_nodes()
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            valid_actions = action_space.get_valid_actions(current_partition, boundary_nodes)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        print(f"   有效动作计算平均时间: {avg_time*1000:.2f} ms")
        print(f"   边界节点数量: {len(boundary_nodes)}")
        print(f"   有效动作数量: {len(valid_actions)}")
        
        results.append({
            'num_nodes': num_nodes,
            'boundary_nodes': len(boundary_nodes),
            'valid_actions': len(valid_actions)
        })
    
    return results


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n🔧 测试向后兼容性")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hetero_data = create_test_graph(100, device=device)
    num_partitions = 5
    
    # 初始化
    action_space = ActionSpace(hetero_data, num_partitions, device)
    
    # 测试访问adjacency_list（应该产生警告）
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        adj_list = action_space.adjacency_list
        
        if len(w) > 0 and issubclass(w[0].category, DeprecationWarning):
            print("✅ 向后兼容性警告正确触发")
            print(f"   警告信息: {w[0].message}")
        else:
            print("❌ 未触发向后兼容性警告")
    
    # 验证adjacency_list格式
    if isinstance(adj_list, list) and all(isinstance(neighbors, set) for neighbors in adj_list):
        print("✅ adjacency_list格式正确")
    else:
        print("❌ adjacency_list格式不正确")


def main():
    """主测试函数"""
    print("=" * 60)
    print("ActionSpace和StateManager性能测试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    print("\n1️⃣ 创建测试数据...")
    hetero_data = create_test_graph(1000, device=device)
    num_partitions = 10
    
    # 创建node_embeddings
    node_embeddings = {
        'node_type1': torch.randn(hetero_data['node_type1'].x.shape[0], 64, device=device),
        'node_type2': torch.randn(hetero_data['node_type2'].x.shape[0], 64, device=device)
    }
    
    # 初始化
    action_space = ActionSpace(hetero_data, num_partitions, device)
    state_manager = StateManager(hetero_data, node_embeddings, device)
    
    # 创建测试分区
    current_partition = create_random_partition(1000, num_partitions, device)
    state_manager.reset(current_partition)
    boundary_nodes = state_manager.get_boundary_nodes()
    
    # 保存或加载基线结果
    import os
    if not os.path.exists('baseline_results.pt'):
        print("\n2️⃣ 保存基线结果...")
        baseline_results = save_baseline_results(action_space, state_manager, current_partition, boundary_nodes)
    else:
        print("\n2️⃣ 加载基线结果...")
        baseline_results = torch.load('baseline_results.pt')
        print("✅ 基线结果已加载")
    
    # 验证功能正确性
    print("\n3️⃣ 验证功能正确性...")
    verify_results(action_space, state_manager, baseline_results)
    
    # 性能测试
    print("\n4️⃣ 执行性能测试...")
    performance_test()
    
    # 向后兼容性测试
    print("\n5️⃣ 测试向后兼容性...")
    test_backward_compatibility()
    
    print("\n✅ 所有测试完成！")


if __name__ == "__main__":
    main() 