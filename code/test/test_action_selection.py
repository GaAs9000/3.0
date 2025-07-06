import sys, pathlib
sys.modules.pop('code', None)  # 移除标准库 code 模块，避免命名冲突
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import torch
import pytest
from torch_geometric.data import HeteroData

# 导入模块
from code.src.rl.action_space import ActionSpace
from code.src.rl.agent import PPOAgent


def build_simple_env(num_nodes: int = 6, num_partitions: int = 3,
                     node_dim: int = 8, region_dim: int = 4,
                     device: torch.device = torch.device('cpu')):
    """构造一个极简异构图、ActionSpace 与 Agent，用于单元测试"""
    # ==== 构造 HeteroData ====
    data = HeteroData()
    data['bus'].x = torch.randn(num_nodes, node_dim)
    # 简单环形边
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    data['bus', 'connects', 'bus'].edge_index = torch.stack([src, dst])

    # ==== ActionSpace ====
    action_space = ActionSpace(data, num_partitions, device)

    # ==== Agent ====
    agent = PPOAgent(node_dim, region_dim, num_partitions, device=device)

    # ==== 初始分区 & 状态准备 ====
    current_partition = torch.randint(1, num_partitions + 1, (num_nodes,), device=device)
    boundary_nodes = torch.arange(num_nodes, device=device)  # 为简单起见全部视为边界节点

    state = {
        'node_embeddings': torch.randn(num_nodes, node_dim, device=device),
        'region_embeddings': torch.randn(num_partitions, region_dim, device=device),
        'boundary_nodes': boundary_nodes,
        'current_partition': current_partition,
        'action_mask': action_space.get_action_mask(current_partition, boundary_nodes),
    }
    return agent, action_space, state


def test_select_action_validity():
    torch.manual_seed(42)
    device = torch.device('cpu')
    agent, action_space, state = build_simple_env(device=device)

    for _ in range(1000):
        action, _, _ = agent.select_action(state, training=True)
        assert action is not None, "select_action 返回 None"
        assert action_space.is_valid_action(action, state['current_partition'], state['boundary_nodes']), (
            f"非法动作采样: {action}")


if __name__ == "__main__":
    pytest.main([__file__]) 