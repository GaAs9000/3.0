#!/usr/bin/env python3
"""
Generalization Evaluation Script for Power Grid Partitioning Agent

This script evaluates a trained PPO agent on its ability to generalize to:
1. Unseen power grid topologies.
2. Variable numbers of partitions (K-values) that may not have been used during training.

Key Functions:
- Loads a trained agent from a checkpoint.
- Runs evaluation on a specified grid case for a range of K-values.
- Reports key performance metrics for each K to assess generalization capability.
"""

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys
import pandas as pd

# Add project root to path robustly
# This allows the script to be run from anywhere
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


# Import necessary components from the training script and codebase
from train import DataManager, EnvironmentFactory, AgentFactory
from code.src.rl.agent import PPOAgent
from code.src.rl.environment import PowerGridPartitioningEnv

def print_header(title: str):
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def evaluate_agent_on_case(
    agent: PPOAgent,
    env: PowerGridPartitioningEnv,
    num_episodes: int = 5
) -> dict:
    """
    Evaluates the agent on a given environment for a number of episodes.

    Args:
        agent: The trained PPOAgent.
        env: The PowerGridPartitioningEnv instance.
        num_episodes: The number of episodes to run for evaluation.

    Returns:
        A dictionary with evaluation metrics.
    """
    episode_rewards = []
    final_quality_scores = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        last_info = {}
        
        while not done:
            # Pass the env instance to select_action for the new architecture
            action, _, _ = agent.select_action(state, env=env, training=False)
            if action is None:
                break
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            last_info = info

        episode_rewards.append(episode_reward)
        if 'metrics' in last_info and 'quality_score' in last_info['metrics']:
            final_quality_scores.append(last_info['metrics']['quality_score'])

    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_final_quality_score': np.mean(final_quality_scores) if final_quality_scores else 0,
    }

def main(args):
    """Main evaluation function."""
    print_header("Generalization Evaluation")

    # 1. Load Configuration from model checkpoint
    print(f"📂 Loading model and configuration from: {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    print("✅ Configuration loaded.")

    # 2. Load Test Case Data
    print(f"⚡ Loading test case: {args.case_name.upper()}")
    data_manager = DataManager(config)
    mpc = data_manager.load_power_grid_data(args.case_name)
    hetero_data = data_manager.process_data(mpc, config, device)
    print("✅ Test case data loaded and processed.")

    # 3. Initialize Agent from Checkpoint
    print("🤖 Initializing agent from checkpoint...")
    # Temporarily set a dummy num_partitions, as the agent.__init__ might still expect it
    # The new agent architecture doesn't use it for model creation anyway.
    config['environment']['num_partitions'] = 2 
    
    # We need a dummy environment to get the state dimensions for agent creation
    dummy_env = EnvironmentFactory.create_basic_environment(hetero_data, config)
    
    # Create agent using the factory
    agent = AgentFactory.create_agent(dummy_env, config, device)
    
    # Load state dicts
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor.eval()
    agent.critic.eval()
    print("✅ Agent initialized and weights loaded.")

    # 4. Run Evaluation Loop for each K
    print_header(f"Starting Evaluation on {args.case_name.upper()} for K = {args.k_values}")
    results = []
    
    for k in args.k_values:
        print(f"\n--- Evaluating for K = {k} ---")
        
        # Create a new environment for each K
        config['environment']['num_partitions'] = k
        env = EnvironmentFactory.create_basic_environment(hetero_data, config)
        
        # Run evaluation
        metrics = evaluate_agent_on_case(agent, env, args.num_episodes)
        
        result_row = {
            'K': k,
            'Avg Reward': metrics['avg_reward'],
            'Std Reward': metrics['std_reward'],
            'Avg Final Quality': metrics['avg_final_quality_score']
        }
        results.append(result_row)
        print(f"  - Avg Reward: {metrics['avg_reward']:.3f}")
        print(f"  - Avg Final Quality Score: {metrics['avg_final_quality_score']:.3f}")

    # 5. Display Final Report
    print_header("Evaluation Summary")
    report_df = pd.DataFrame(results)
    print(report_df.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GNN Agent Generalization")
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help="Path to the trained agent model checkpoint (.pth file)."
    )
    parser.add_argument(
        '--case-name',
        type=str,
        default='case3120sp',
        help="Name of the grid case to evaluate on. For strict generalization testing, choose a case not in the base training set (e.g., 'case3120sp', 'case_illinois200')."
    )
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[3, 4, 5, 6, 7, 8, 10, 12],
        help="A list of K-values (number of partitions) to test."
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help="Number of episodes to run for each K-value to average results."
    )
    
    args = parser.parse_args()
    main(args) 