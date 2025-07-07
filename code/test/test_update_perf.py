
print("--- test_update_perf.py script started ---")
import time
import torch
import random
import numpy as np
import sys
import pathlib

# Explicitly pop cached 'code' module to force re-evaluation
sys.modules.pop('code', None)

# Ensure the project root is in the Python path
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from code.src.rl.agent import PPOAgent
from torch_geometric.data import HeteroData

def populate_agent_memory(
    agent: PPOAgent,
    num_steps: int = 2048,
    num_nodes: int = 32,
    node_dim: int = 8,
    device: torch.device = torch.device('cpu')
):
    """Generates mock experience and stores it in the agent's memory."""
    for i in range(num_steps):
        # 1. Mock Graph State (simplified for speed)
        state = {
            'node_embeddings': torch.randn(num_nodes, node_dim, device=device),
            'region_embeddings': torch.randn(agent.num_partitions, agent.critic.state_encoder[0].in_features, device=device),
            'boundary_nodes': torch.arange(0, num_nodes, 2, device=device),
            'current_partition': torch.randint(1, agent.num_partitions + 1, (num_nodes,), device=device)
        }
        state['action_mask'] = torch.ones(num_nodes, agent.num_partitions, dtype=torch.bool, device=device)

        # 2. Mock other experience data
        # Action is a tuple of (node_idx, partition_idx)
        action = (torch.randint(0, num_nodes, (1,)).item(), torch.randint(1, agent.num_partitions + 1, (1,)).item())
        log_prob = -1.5
        value = 0.5
        reward = 0.1
        done = (i == num_steps - 1)
        
        # 3. Store experience
        agent.store_experience(state, action, reward, log_prob, value, done)


def benchmark_ppo_update(
    device: torch.device = torch.device('cpu'),
    num_steps: int = 2048,
    num_nodes: int = 256,
    num_partitions: int = 8,
    node_dim: int = 128,
    region_dim: int = 64
) -> float:
    """Benchmarks the PPO agent's update loop."""
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f"🚀 Setting up benchmark on device: {device}")
    print(f"Simulating {num_steps} steps with {num_nodes} nodes and {num_partitions} partitions.")

    # 1. Initialize Agent with correct parameters
    agent_config = {
        'lr_actor':3e-4,
        'lr_critic':1e-3,
        'gamma':0.99,
        'eps_clip':0.2,
        'k_epochs':4,
        'value_coef':0.5,
        'entropy_coef':0.01,
        'memory_capacity':num_steps
    }

    agent = PPOAgent(
        node_embedding_dim=node_dim,
        region_embedding_dim=region_dim,
        num_partitions=num_partitions,
        agent_config=agent_config,
        device=device
    )

    # 2. Populate agent's memory
    print("Populating agent memory...")
    populate_agent_memory(
        agent,
        num_steps=num_steps,
        num_nodes=num_nodes,
        node_dim=node_dim,
        device=device
    )
    print("Memory populated.")

    # 3. Warm-up run
    print("Performing warm-up run...")
    agent.update()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # After update, memory is cleared, so we need to populate it again
    populate_agent_memory(agent, num_steps, num_nodes, node_dim, device)
    print("Warm-up complete.")

    # 4. Actual benchmark
    print("Starting benchmark...")
    start_time = time.perf_counter()
    
    agent.update()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("Benchmark finished.")

    return end_time - start_time


if __name__ == "__main__":
    # Ensure CUDA is available, otherwise fall back to CPU
    if torch.cuda.is_available():
        selected_device = torch.device('cuda')
    else:
        print("⚠️ CUDA not available, running on CPU. Performance results will not reflect GPU capabilities.")
        selected_device = torch.device('cpu')

    # Run the benchmark
    total_time = benchmark_ppo_update(device=selected_device)
    
    print("\n" + "="*50)
    print(f"✅ PPO Update Phase Benchmark Complete")
    print(f"   Total time for one full update cycle: {total_time:.4f} seconds")
    print("="*50) 