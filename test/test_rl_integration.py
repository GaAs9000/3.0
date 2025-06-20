#!/usr/bin/env python3
"""
Integration Tests for Power Grid Partitioning RL System

This script provides comprehensive end-to-end testing of the RL system
including environment, agent, training, and evaluation components.
"""

import torch
import numpy as np
import os
import sys
import warnings
from pathlib import Path
import tempfile
import shutil

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import modules
from src.data_processing import PowerGridDataProcessor
from src.gat import create_hetero_graph_encoder
from src.rl.environment import PowerGridPartitioningEnv
from src.rl.agent import PPOAgent
from src.rl.training import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def create_test_power_grid():
    """Create a test power grid for integration testing"""
    # IEEE 14-bus test system (simplified)
    mpc = {
        'baseMVA': 100.0,
        'bus': np.array([
            # Bus data: [bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
            [1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.06, 0.0, 138, 1, 1.1, 0.9],      # Slack bus
            [2, 2, 21.7, 12.7, 0.0, 0.0, 1, 1.045, -4.98, 138, 1, 1.1, 0.9], # PV bus
            [3, 2, 94.2, 19.0, 0.0, 0.0, 1, 1.01, -12.72, 138, 1, 1.1, 0.9], # PV bus
            [4, 1, 47.8, -3.9, 0.0, 0.0, 1, 1.019, -10.33, 138, 1, 1.1, 0.9], # PQ bus
            [5, 1, 7.6, 1.6, 0.0, 0.0, 1, 1.02, -8.78, 138, 1, 1.1, 0.9],    # PQ bus
            [6, 2, 11.2, 7.5, 0.0, 0.0, 1, 1.07, -14.22, 138, 1, 1.1, 0.9],  # PV bus
            [7, 1, 0.0, 0.0, 0.0, 0.0, 1, 1.062, -13.37, 138, 1, 1.1, 0.9],  # PQ bus
            [8, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.09, -13.36, 138, 1, 1.1, 0.9],   # PV bus
            [9, 1, 29.5, 16.6, 0.0, 19.0, 1, 1.056, -14.94, 138, 1, 1.1, 0.9], # PQ bus
            [10, 1, 9.0, 5.8, 0.0, 0.0, 1, 1.051, -15.1, 138, 1, 1.1, 0.9],   # PQ bus
            [11, 1, 3.5, 1.8, 0.0, 0.0, 1, 1.057, -14.79, 138, 1, 1.1, 0.9],  # PQ bus
            [12, 1, 6.1, 1.6, 0.0, 0.0, 1, 1.055, -15.07, 138, 1, 1.1, 0.9],  # PQ bus
            [13, 1, 13.5, 5.8, 0.0, 0.0, 1, 1.05, -15.16, 138, 1, 1.1, 0.9],  # PQ bus
            [14, 1, 14.9, 5.0, 0.0, 0.0, 1, 1.036, -16.04, 138, 1, 1.1, 0.9], # PQ bus
        ]),
        'branch': np.array([
            # Branch data: [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]
            [1, 2, 0.01938, 0.05917, 0.0528, 100, 110, 120, 0, 0, 1, -360, 360],
            [1, 5, 0.05403, 0.22304, 0.0492, 100, 110, 120, 0, 0, 1, -360, 360],
            [2, 3, 0.04699, 0.19797, 0.0438, 100, 110, 120, 0, 0, 1, -360, 360],
            [2, 4, 0.05811, 0.17632, 0.034, 100, 110, 120, 0, 0, 1, -360, 360],
            [2, 5, 0.05695, 0.17388, 0.0346, 100, 110, 120, 0, 0, 1, -360, 360],
            [3, 4, 0.06701, 0.17103, 0.0128, 100, 110, 120, 0, 0, 1, -360, 360],
            [4, 5, 0.01335, 0.04211, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [4, 7, 0.0, 0.20912, 0.0, 100, 110, 120, 0.978, 0, 1, -360, 360],  # Transformer
            [4, 9, 0.0, 0.55618, 0.0, 100, 110, 120, 0.969, 0, 1, -360, 360],  # Transformer
            [5, 6, 0.0, 0.25202, 0.0, 100, 110, 120, 0.932, 0, 1, -360, 360],  # Transformer
            [6, 11, 0.09498, 0.1989, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [6, 12, 0.12291, 0.25581, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [6, 13, 0.06615, 0.13027, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [7, 8, 0.0, 0.17615, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [7, 9, 0.0, 0.11001, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [9, 10, 0.03181, 0.0845, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [9, 14, 0.12711, 0.27038, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [10, 11, 0.08205, 0.19207, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [12, 13, 0.22092, 0.19988, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
            [13, 14, 0.17093, 0.34802, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
        ]),
        'gen': np.array([
            # Generator data: [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin]
            [1, 232.4, -16.9, 10, -10, 1.06, 100, 1, 332.4, 0],
            [2, 40.0, 43.56, 50, -40, 1.045, 100, 1, 140.0, 0],
            [3, 0.0, 25.075, 40, 0, 1.01, 100, 1, 100.0, 0],
            [6, 0.0, 12.73, 24, -6, 1.07, 100, 1, 100.0, 0],
            [8, 0.0, 17.623, 24, -6, 1.09, 100, 1, 100.0, 0],
        ])
    }
    
    return mpc


def test_data_processing():
    """Test data processing pipeline"""
    print("🔧 Testing Data Processing...")
    
    # Create test data
    mpc = create_test_power_grid()
    
    # Process data
    processor = PowerGridDataProcessor(normalize=True, cache_dir='test_cache')
    hetero_data = processor.graph_from_mpc(mpc)
    
    # Validate data structure
    assert hasattr(hetero_data, 'x_dict'), "Missing node features"
    assert hasattr(hetero_data, 'edge_index_dict'), "Missing edge indices"
    assert hasattr(hetero_data, 'edge_attr_dict'), "Missing edge attributes"
    
    # Check node types
    expected_node_types = ['bus_pq', 'bus_pv', 'bus_slack']
    for node_type in expected_node_types:
        assert node_type in hetero_data.x_dict, f"Missing node type: {node_type}"
        
    print(f"✅ Data processing successful!")
    print(f"   Node types: {list(hetero_data.x_dict.keys())}")
    print(f"   Edge types: {len(hetero_data.edge_index_dict)}")
    
    return hetero_data


def test_gat_encoder(hetero_data):
    """Test GAT encoder with enhanced embeddings"""
    print("🧠 Testing GAT Encoder with Enhanced Embeddings...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hetero_data = hetero_data.to(device)

    # Create encoder
    encoder = create_hetero_graph_encoder(
        hetero_data,
        hidden_channels=32,
        gnn_layers=2,
        heads=4,
        output_dim=64
    ).to(device)

    # Test encoding with attention weights
    with torch.no_grad():
        node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data)

    # Validate embeddings
    assert isinstance(node_embeddings, dict), "Node embeddings should be a dictionary"
    assert isinstance(attention_weights, dict), "Attention weights should be a dictionary"

    total_nodes = 0
    for node_type, embeddings in node_embeddings.items():
        assert embeddings.shape[1] == 64, f"Wrong embedding dimension for {node_type}"
        total_nodes += embeddings.shape[0]

    print(f"✅ GAT encoder successful!")
    print(f"   Total nodes encoded: {total_nodes}")
    print(f"   Original embedding dimension: {64}")
    print(f"   Attention weights extracted: {len(attention_weights)} edge types")

    return node_embeddings, attention_weights


def test_rl_environment(hetero_data, node_embeddings, attention_weights):
    """Test RL environment with enhanced embeddings"""
    print("🌍 Testing RL Environment with Enhanced Embeddings...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment with enhanced embeddings
    env = PowerGridPartitioningEnv(
        hetero_data=hetero_data,
        node_embeddings=node_embeddings,
        num_partitions=3,
        max_steps=50,
        device=device,
        attention_weights=attention_weights
    )
    
    # Test reset
    obs, info = env.reset(seed=42)
    
    # Validate observation
    assert isinstance(obs, dict), "Observation should be a dictionary"
    required_keys = ['node_embeddings', 'region_embeddings', 'boundary_features', 
                    'current_partition', 'boundary_nodes']
    for key in required_keys:
        assert key in obs, f"Missing observation key: {key}"
        
    # Test step
    valid_actions = info['valid_actions']
    if len(valid_actions) > 0:
        action = valid_actions[0]
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        assert isinstance(reward, float), "Reward should be a float"
        assert isinstance(terminated, bool), "Terminated should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        
    print(f"✅ RL environment successful!")
    print(f"   Total nodes: {env.total_nodes}")
    print(f"   Partitions: {env.num_partitions}")
    print(f"   Valid actions: {len(valid_actions)}")
    
    return env


def test_ppo_agent(env, node_embeddings):
    """Test PPO agent with enhanced embeddings"""
    print("🤖 Testing PPO Agent with Enhanced Embeddings...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get enhanced embedding dimensions from the environment
    # The state manager stores concatenated embeddings, so we get the dimension from there
    node_embedding_dim = env.state_manager.embedding_dim
    region_embedding_dim = node_embedding_dim * 2  # Mean + max pooling
    
    # Create agent
    agent = PPOAgent(
        node_embedding_dim=node_embedding_dim,
        region_embedding_dim=region_embedding_dim,
        num_partitions=env.num_partitions,
        device=device
    )
    
    # Test action selection
    obs, info = env.reset(seed=42)
    action, log_prob, value = agent.select_action(obs, training=True)
    
    if action is not None:
        assert isinstance(action, tuple), "Action should be a tuple"
        assert len(action) == 2, "Action should have 2 elements"
        assert isinstance(log_prob, float), "Log prob should be a float"
        assert isinstance(value, float), "Value should be a float"
        
        # Test experience storage
        agent.store_experience(obs, action, 1.0, log_prob, value, False)
        
        # Test update (with minimal data)
        for _ in range(5):  # Add more experiences
            agent.store_experience(obs, action, 1.0, log_prob, value, False)
            
        update_stats = agent.update()
        assert isinstance(update_stats, dict), "Update stats should be a dictionary"
        
    print(f"✅ PPO agent successful!")
    print(f"   Node embedding dim: {node_embedding_dim}")
    print(f"   Region embedding dim: {region_embedding_dim}")
    
    return agent


def test_training_integration(env, agent):
    """Test training integration"""
    print("🏋️ Testing Training Integration...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = os.path.join(temp_dir, 'logs')
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        
        # Create trainer
        trainer = Trainer(
            agent=agent,
            env=env,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            save_interval=5,
            use_tensorboard=False  # Disable for testing
        )
        
        # Run short training
        history = trainer.train(
            num_episodes=10,
            max_steps_per_episode=20,
            update_interval=5
        )
        
        # Validate training history
        assert isinstance(history, dict), "History should be a dictionary"

        # Check if we have any recorded episodes
        has_episodes = False
        for key in ['episode_rewards', 'episode_reward']:
            if key in history and len(history[key]) > 0:
                has_episodes = True
                break

        if not has_episodes:
            # Print debug info
            print(f"Debug: History keys: {list(history.keys())}")
            for key, value in history.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")

        assert has_episodes, "No episodes recorded in history"
        
        # Test evaluation
        eval_stats = trainer.evaluate(num_episodes=3)
        assert isinstance(eval_stats, dict), "Eval stats should be a dictionary"
        assert 'mean_reward' in eval_stats, "Missing mean reward"
        
        # Clean up
        trainer.close()
        
    print(f"✅ Training integration successful!")

    # Get episode count from any available metric
    episode_count = 0
    mean_reward = 0.0
    for key in ['episode_rewards', 'episode_reward']:
        if key in history and len(history[key]) > 0:
            episode_count = len(history[key])
            mean_reward = np.mean(history[key])
            break

    print(f"   Episodes completed: {episode_count}")
    print(f"   Mean reward: {mean_reward:.3f}")


def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting Power Grid Partitioning RL Integration Tests")
    print("=" * 60)
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Test data processing
        hetero_data = test_data_processing()
        
        # Test GAT encoder with enhanced embeddings
        node_embeddings, attention_weights = test_gat_encoder(hetero_data)

        # Test RL environment with enhanced embeddings
        env = test_rl_environment(hetero_data, node_embeddings, attention_weights)
        
        # Test PPO agent
        agent = test_ppo_agent(env, node_embeddings)
        
        # Test training integration
        test_training_integration(env, agent)
        
        print("\n" + "=" * 60)
        print("🎉 All integration tests passed successfully!")
        print("✅ The RL system is ready for training on real power grid data.")
        
        # Clean up
        env.close()
        
        # Clean up test cache
        try:
            if os.path.exists('test_cache'):
                shutil.rmtree('test_cache')
        except PermissionError:
            print("Note: Could not remove test cache due to permissions (this is normal on Windows)")
            
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
