"""
接口一致性测试

验证优化后的ActionSpace和StateManager保持了相同的公共接口
"""

import torch
from torch_geometric.data import HeteroData
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rl.action_space import ActionSpace, ActionMask
from src.rl.state import StateManager


def create_simple_hetero_graph(device='cuda'):
    """创建简单的异构图用于测试"""
    hetero_data = HeteroData()
    
    hetero_data['type1'].x = torch.randn(10, 32, device=device)
    hetero_data['type2'].x = torch.randn(15, 32, device=device)
    
    # 添加边
    hetero_data['type1', 'connects', 'type2'].edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ], device=device)
    
    hetero_data['type2', 'connects', 'type1'].edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ], device=device)
    
    return hetero_data


def test_action_space_interface():
    """测试ActionSpace接口"""
    print("测试ActionSpace接口...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hetero_data = create_simple_hetero_graph(device)
    num_partitions = 3
    
    # 创建ActionSpace实例
    action_space = ActionSpace(hetero_data, num_partitions, device)
    
    # 测试必需的属性
    assert hasattr(action_space, 'device'), "缺少device属性"
    assert hasattr(action_space, 'hetero_data'), "缺少hetero_data属性"
    assert hasattr(action_space, 'num_partitions'), "缺少num_partitions属性"
    assert hasattr(action_space, 'total_nodes'), "缺少total_nodes属性"
    assert hasattr(action_space, 'adjacency_sparse'), "缺少adjacency_sparse属性"
    assert hasattr(action_space, 'neighbors_cache'), "缺少neighbors_cache属性"
    
    # 创建测试数据
    current_partition = torch.randint(1, num_partitions + 1, (25,), device=device)
    boundary_nodes = torch.tensor([0, 5, 10, 15, 20], device=device)
    
    # 测试公共方法
    print("  - get_valid_actions...")
    valid_actions = action_space.get_valid_actions(current_partition, boundary_nodes)
    assert isinstance(valid_actions, list), "get_valid_actions应返回列表"
    
    print("  - get_action_mask...")
    action_mask = action_space.get_action_mask(current_partition, boundary_nodes)
    assert action_mask.shape == (25, 3), f"action_mask形状不正确: {action_mask.shape}"
    
    print("  - is_valid_action...")
    if len(valid_actions) > 0:
        result = action_space.is_valid_action(valid_actions[0], current_partition, boundary_nodes)
        assert isinstance(result, bool), "is_valid_action应返回布尔值"
    
    print("  - sample_random_action...")
    random_action = action_space.sample_random_action(current_partition, boundary_nodes)
    assert random_action is None or isinstance(random_action, tuple), "sample_random_action返回类型不正确"
    
    print("  - get_action_space_size...")
    size = action_space.get_action_space_size()
    assert size == 25 * 3, f"动作空间大小不正确: {size}"
    
    print("✅ ActionSpace接口测试通过")


def test_state_manager_interface():
    """测试StateManager接口"""
    print("\n测试StateManager接口...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hetero_data = create_simple_hetero_graph(device)
    
    # 创建节点嵌入
    node_embeddings = {
        'type1': torch.randn(10, 64, device=device),
        'type2': torch.randn(15, 64, device=device)
    }
    
    # 创建StateManager实例
    state_manager = StateManager(hetero_data, node_embeddings, device)
    
    # 测试必需的属性
    assert hasattr(state_manager, 'device'), "缺少device属性"
    assert hasattr(state_manager, 'hetero_data'), "缺少hetero_data属性"
    assert hasattr(state_manager, 'total_nodes'), "缺少total_nodes属性"
    assert hasattr(state_manager, 'adjacency_sparse'), "缺少adjacency_sparse属性"
    assert hasattr(state_manager, 'neighbors_cache'), "缺少neighbors_cache属性"
    assert hasattr(state_manager, 'node_embeddings'), "缺少node_embeddings属性"
    assert hasattr(state_manager, 'embedding_dim'), "缺少embedding_dim属性"
    
    # 测试公共方法
    print("  - reset...")
    initial_partition = torch.randint(1, 4, (25,), device=device)
    state_manager.reset(initial_partition)
    
    print("  - get_boundary_nodes...")
    boundary_nodes = state_manager.get_boundary_nodes()
    assert isinstance(boundary_nodes, torch.Tensor), "get_boundary_nodes应返回张量"
    
    print("  - get_observation...")
    observation = state_manager.get_observation()
    assert isinstance(observation, dict), "get_observation应返回字典"
    assert 'node_embeddings' in observation, "缺少node_embeddings"
    assert 'region_embeddings' in observation, "缺少region_embeddings"
    assert 'boundary_features' in observation, "缺少boundary_features"
    assert 'current_partition' in observation, "缺少current_partition"
    assert 'boundary_nodes' in observation, "缺少boundary_nodes"
    
    print("  - update_partition...")
    if len(boundary_nodes) > 0:
        state_manager.update_partition(boundary_nodes[0].item(), 2)
    
    print("  - get_partition_info...")
    partition_info = state_manager.get_partition_info()
    assert isinstance(partition_info, dict), "get_partition_info应返回字典"
    assert 'partition_assignments' in partition_info, "缺少partition_assignments"
    assert 'partition_sizes' in partition_info, "缺少partition_sizes"
    assert 'num_partitions' in partition_info, "缺少num_partitions"
    assert 'boundary_nodes' in partition_info, "缺少boundary_nodes"
    
    print("✅ StateManager接口测试通过")


def test_action_mask_interface():
    """测试ActionMask接口"""
    print("\n测试ActionMask接口...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hetero_data = create_simple_hetero_graph(device)
    
    # 创建ActionMask实例
    action_mask = ActionMask(hetero_data, device)
    
    # 测试必需的属性
    assert hasattr(action_mask, 'device'), "缺少device属性"
    assert hasattr(action_mask, 'hetero_data'), "缺少hetero_data属性"
    assert hasattr(action_mask, 'total_nodes'), "缺少total_nodes属性"
    assert hasattr(action_mask, 'adjacency_sparse'), "缺少adjacency_sparse属性"
    assert hasattr(action_mask, 'neighbors_cache'), "缺少neighbors_cache属性"
    
    # 测试公共方法
    current_partition = torch.randint(1, 4, (25,), device=device)
    test_action = (0, 2)
    
    print("  - check_connectivity_violation...")
    violation_info = action_mask.check_connectivity_violation(current_partition, test_action)
    assert isinstance(violation_info, dict), "check_connectivity_violation应返回字典"
    assert 'violates_connectivity' in violation_info, "缺少violates_connectivity"
    assert 'violation_severity' in violation_info, "缺少violation_severity"
    assert 'isolated_nodes_count' in violation_info, "缺少isolated_nodes_count"
    assert 'affected_partition_size' in violation_info, "缺少affected_partition_size"
    
    print("✅ ActionMask接口测试通过")


def main():
    """主测试函数"""
    print("=" * 60)
    print("接口一致性测试")
    print("=" * 60)
    
    # 测试各个组件的接口
    test_action_space_interface()
    test_state_manager_interface()
    test_action_mask_interface()
    
    print("\n✅ 所有接口测试通过！")
    print("优化后的代码保持了完整的公共接口")


if __name__ == "__main__":
    main() 