import time
import torch
import random
from torch_geometric.data import HeteroData
import sys, pathlib
sys.modules.pop('code', None)
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from code.src.rl.agent import PPOAgent
from code.src.rl.action_space import ActionSpace


def build_env(num_nodes: int = 32, num_partitions: int = 4,
              node_dim: int = 8, region_dim: int = 4,
              device: torch.device = torch.device('cpu')):
    """构建简易环境便于性能测试"""
    data = HeteroData()
    data['bus'].x = torch.randn(num_nodes, node_dim, device=device)
    src = torch.arange(num_nodes, device=device)
    dst = (src + 1) % num_nodes
    data['bus', 'connects', 'bus'].edge_index = torch.stack([src, dst])

    action_space = ActionSpace(data, num_partitions, device)
    agent = PPOAgent(node_dim, region_dim, num_partitions, device=device)

    current_partition = torch.randint(1, num_partitions + 1, (num_nodes,), device=device)
    boundary_nodes = torch.arange(num_nodes, device=device)
    state = {
        'node_embeddings': torch.randn(num_nodes, node_dim, device=device),
        'region_embeddings': torch.randn(num_partitions, region_dim, device=device),
        'boundary_nodes': boundary_nodes,
        'current_partition': current_partition,
        'action_mask': action_space.get_action_mask(current_partition, boundary_nodes),
    }
    return agent, state


def benchmark_select_action(steps: int = 2048, device: torch.device = torch.device('cpu')) -> float:
    torch.manual_seed(0)
    random.seed(0)
    agent, state = build_env(device=device)

    # 预热一次
    agent.select_action(state, training=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        agent.select_action(state, training=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_time = benchmark_select_action(device=device)
    print(f"Select_action 2048 steps time: {total_time:.4f} s -> {(total_time/2048*1e3):.3f} ms/step") 