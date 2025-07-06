import torch
import random
import sys, os, pathlib
# 确保可导入本项目 code 包，而不是Python内置 code 模块
sys.modules.pop('code', None)
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from code.src.rl.agent import PPOAgent, BatchedState


def generate_random_state(node_dim, region_dim, num_partitions, device):
    # 随机节点和区域数量
    num_nodes = random.randint(5, 12)
    num_regions = num_partitions

    node_embeddings = torch.randn(num_nodes, node_dim, device=device)
    region_embeddings = torch.randn(num_regions, region_dim, device=device)

    # 随机 current_partition (1..num_partitions)
    current_partition = torch.randint(1, num_partitions + 1, (num_nodes,), device=device)

    # 随机边界节点（至少1个）
    num_boundary = random.randint(1, min(4, num_nodes))
    boundary_indices = torch.tensor(random.sample(range(num_nodes), num_boundary), device=device)

    # 构造动作掩码（与agent旧逻辑一致）
    action_mask = torch.zeros(num_nodes, num_partitions, dtype=torch.bool, device=device)
    for node_idx in boundary_indices:
        cur_p = current_partition[node_idx].item()
        for p in range(num_partitions):
            if p + 1 != cur_p:
                action_mask[node_idx, p] = True

    return {
        'node_embeddings': node_embeddings,
        'region_embeddings': region_embeddings,
        'boundary_nodes': boundary_indices,
        'current_partition': current_partition,
        'action_mask': action_mask,
    }


def test_actor_critic_batch_consistency():
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_dim, region_dim, num_partitions = 8, 16, 4
    agent = PPOAgent(node_dim, region_dim, num_partitions, device=device, memory_capacity=16)

    agent.actor.eval()
    agent.critic.eval()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 生成若干随机state
    states = [generate_random_state(node_dim, region_dim, num_partitions, device) for _ in range(3)]

    # 单样本前向
    actor_single_outputs = []
    critic_single_outputs = []
    for s in states:
        a_logits, p_logits = agent.actor(
            s['node_embeddings'], s['region_embeddings'], s['boundary_nodes'], s['action_mask']
        )
        c_val = agent.critic(s['node_embeddings'], s['region_embeddings'], s['boundary_nodes'])
        actor_single_outputs.append((a_logits.detach(), p_logits.detach()))
        critic_single_outputs.append(c_val.detach())

    # 批处理前向
    bs: BatchedState = agent._prepare_batched_state(states)
    a_logits_b, p_logits_b = agent.actor(bs)
    c_vals_b = agent.critic(bs)

    # 逐样本比对
    offset = 0
    for i, s in enumerate(states):
        n_boundary = len(s['boundary_nodes'])
        node_logits_b = a_logits_b[offset: offset + n_boundary]
        part_logits_b = p_logits_b[offset: offset + n_boundary]
        node_logits_s, part_logits_s = actor_single_outputs[i]
        assert torch.allclose(node_logits_b, node_logits_s, atol=1e-6), "Node logits mismatch"
        assert torch.allclose(part_logits_b, part_logits_s, atol=1e-6), "Partition logits mismatch"
        assert torch.allclose(c_vals_b[i], critic_single_outputs[i], atol=1e-6), "Critic value mismatch"
        offset += n_boundary

    print("✅ Batch consistency test passed.")


if __name__ == "__main__":
    test_actor_critic_batch_consistency() 