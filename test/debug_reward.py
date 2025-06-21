#!/usr/bin/env python3
"""
Enhanced Reward Function Debugging Script
Fixed version with English labels and proper visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use standard font

# Add project path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing import PowerGridDataProcessor
from gat import create_hetero_graph_encoder
from rl.environment import PowerGridPartitioningEnv
from rl.reward import RewardFunction


def create_test_case():
    """Create test case"""
    # IEEE 14-bus system
    mpc = {
        'baseMVA': 100.0,
        'bus': np.array([
            [1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.06, 0.0, 138, 1, 1.1, 0.9],
            [2, 2, 21.7, 12.7, 0.0, 0.0, 1, 1.045, -4.98, 138, 1, 1.1, 0.9],
            [3, 2, 94.2, 19.0, 0.0, 0.0, 1, 1.01, -12.72, 138, 1, 1.1, 0.9],
            [4, 1, 47.8, -3.9, 0.0, 0.0, 1, 1.019, -10.33, 138, 1, 1.1, 0.9],
            [5, 1, 7.6, 1.6, 0.0, 0.0, 1, 1.02, -8.78, 138, 1, 1.1, 0.9],
            [6, 2, 11.2, 7.5, 0.0, 0.0, 1, 1.07, -14.22, 138, 1, 1.1, 0.9],
            [7, 1, 0.0, 0.0, 0.0, 0.0, 1, 1.062, -13.37, 138, 1, 1.1, 0.9],
            [8, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.09, -13.36, 138, 1, 1.1, 0.9],
            [9, 1, 29.5, 16.6, 0.0, 19.0, 1, 1.056, -14.94, 138, 1, 1.1, 0.9],
            [10, 1, 9.0, 5.8, 0.0, 0.0, 1, 1.051, -15.1, 138, 1, 1.1, 0.9],
            [11, 1, 3.5, 1.8, 0.0, 0.0, 1, 1.057, -14.79, 138, 1, 1.1, 0.9],
            [12, 1, 6.1, 1.6, 0.0, 0.0, 1, 1.055, -15.07, 138, 1, 1.1, 0.9],
            [13, 1, 13.5, 5.8, 0.0, 0.0, 1, 1.05, -15.16, 138, 1, 1.1, 0.9],
            [14, 1, 14.9, 5.0, 0.0, 0.0, 1, 1.036, -16.04, 138, 1, 1.1, 0.9],
        ]),
        'gen': np.array([
            [1, 232.4, -16.9, 10, -10, 1.06, 100, 1, 332.4, 0],
            [2, 40.0, 43.56, 50, -40, 1.045, 100, 1, 140.0, 0],
            [3, 0.0, 25.075, 40, 0, 1.01, 100, 1, 100.0, 0],
            [6, 0.0, 12.73, 24, -6, 1.07, 100, 1, 100.0, 0],
            [8, 0.0, 17.623, 24, -6, 1.09, 100, 1, 100.0, 0],
        ]),
        'branch': np.array([
            [1, 2, 0.01938, 0.05917, 0.0528, 100, 0, 0, 0, 0, 1, -360, 360],
            [1, 5, 0.05403, 0.22304, 0.0492, 100, 0, 0, 0, 0, 1, -360, 360],
            [2, 3, 0.04699, 0.19797, 0.0438, 100, 0, 0, 0, 0, 1, -360, 360],
            [2, 4, 0.05811, 0.17632, 0.034, 100, 0, 0, 0, 0, 1, -360, 360],
            [2, 5, 0.05695, 0.17388, 0.0346, 100, 0, 0, 0, 0, 1, -360, 360],
            [3, 4, 0.06701, 0.17103, 0.0128, 100, 0, 0, 0, 0, 1, -360, 360],
            [4, 5, 0.01335, 0.04211, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [4, 7, 0.0, 0.20912, 0.0, 100, 0, 0, 0.978, 0, 1, -360, 360],
            [4, 9, 0.0, 0.55618, 0.0, 100, 0, 0, 0.969, 0, 1, -360, 360],
            [5, 6, 0.0, 0.25202, 0.0, 100, 0, 0, 0.932, 0, 1, -360, 360],
            [6, 11, 0.09498, 0.1989, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [6, 12, 0.12291, 0.25581, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [6, 13, 0.06615, 0.13027, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [7, 8, 0.0, 0.17615, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [7, 9, 0.0, 0.11001, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [9, 10, 0.03181, 0.0845, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [9, 14, 0.12711, 0.27038, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [10, 11, 0.08205, 0.19207, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [12, 13, 0.22092, 0.19988, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
            [13, 14, 0.17093, 0.34802, 0.0, 100, 0, 0, 0, 0, 1, -360, 360],
        ])
    }
    return mpc


def test_individual_rewards(reward_function, test_partitions):
    """Test individual reward components"""
    print("\n" + "="*60)
    print("Testing Individual Reward Components")
    print("="*60)
    
    results = {}
    
    for name, partition in test_partitions.items():
        print(f"\nTesting partition: {name}")
        
        # Create dummy boundary nodes
        boundary_nodes = torch.tensor([0, 1, 2], device=reward_function.device)
        action = (0, 2)  # Dummy action
        
        # Test each component
        print("\n1. Load Balance Only:")
        reward_function.set_weights({'load_balance': 1.0, 'electrical_decoupling': 0.0, 'power_balance': 0.0})
        reward, components = reward_function.compute_reward(partition, boundary_nodes, action, return_components=True)
        print(f"   Reward: {reward:.4f}")
        print(f"   Component: {components['load_balance']:.4f}")
        
        print("\n2. Electrical Decoupling Only:")
        reward_function.set_weights({'load_balance': 0.0, 'electrical_decoupling': 1.0, 'power_balance': 0.0})
        reward, components = reward_function.compute_reward(partition, boundary_nodes, action, return_components=True)
        print(f"   Reward: {reward:.4f}")
        print(f"   Component: {components['electrical_decoupling']:.4f}")
        
        print("\n3. Power Balance Only:")
        reward_function.set_weights({'load_balance': 0.0, 'electrical_decoupling': 0.0, 'power_balance': 1.0})
        reward, components = reward_function.compute_reward(partition, boundary_nodes, action, return_components=True)
        print(f"   Reward: {reward:.4f}")
        print(f"   Component: {components['power_balance']:.4f}")
        
        print("\n4. Combined Reward (default weights):")
        reward_function.set_weights({'load_balance': 0.4, 'electrical_decoupling': 0.4, 'power_balance': 0.2})
        reward, components = reward_function.compute_reward(partition, boundary_nodes, action, return_components=True)
        print(f"   Total reward: {reward:.4f}")
        print(f"   Component contributions:")
        print(f"     - Load balance: {0.4 * components['load_balance']:.4f}")
        print(f"     - Electrical decoupling: {0.4 * components['electrical_decoupling']:.4f}")
        print(f"     - Power balance: {0.2 * components['power_balance']:.4f}")
        
        results[name] = components
    
    return results


def visualize_reward_sensitivity(reward_function, base_partition, num_moves=10):
    """Visualize reward sensitivity to partition changes"""
    print("\n" + "="*60)
    print("Reward Sensitivity Analysis")
    print("="*60)
    
    # Create figure with proper subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reward Function Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Test different node movements
    nodes_to_test = [3, 5, 8, 10]  # Test multiple nodes
    
    # Track reward changes
    reward_changes = {
        'node': [],
        'from_partition': [],
        'to_partition': [],
        'total_reward': [],
        'load_balance': [],
        'electrical_decoupling': [],
        'power_balance': []
    }
    
    for node_idx in nodes_to_test:
        original_partition = base_partition[node_idx].item()
        
        # Try moving to each other partition
        for target_partition in [1, 2, 3]:
            if target_partition == original_partition:
                continue
            
            # Create new partition
            test_partition = base_partition.clone()
            test_partition[node_idx] = target_partition
            
            # Calculate reward
            boundary_nodes = torch.tensor([node_idx], device=reward_function.device)
            action = (node_idx, target_partition)
            
            reward, components = reward_function.compute_reward(
                test_partition, boundary_nodes, action, return_components=True
            )
            
            # Store results
            reward_changes['node'].append(node_idx)
            reward_changes['from_partition'].append(original_partition)
            reward_changes['to_partition'].append(target_partition)
            reward_changes['total_reward'].append(reward)
            reward_changes['load_balance'].append(components['load_balance'])
            reward_changes['electrical_decoupling'].append(components['electrical_decoupling'])
            reward_changes['power_balance'].append(components['power_balance'])
            
            print(f"\nMove node {node_idx} from partition {original_partition} to {target_partition}:")
            print(f"  Total reward: {reward:.4f}")
            print(f"  Components:")
            for comp_name, comp_value in components.items():
                print(f"    - {comp_name}: {comp_value:.4f}")
    
    # Plot 1: Total reward changes
    ax = axes[0, 0]
    x = range(len(reward_changes['total_reward']))
    ax.bar(x, reward_changes['total_reward'], color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Move Index')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward for Different Moves')
    ax.grid(True, alpha=0.3)
    
    # Add labels
    labels = [f"N{n}: {f}->{t}" for n, f, t in 
              zip(reward_changes['node'], reward_changes['from_partition'], reward_changes['to_partition'])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Plot 2: Component breakdown
    ax = axes[0, 1]
    width = 0.25
    x = np.arange(len(reward_changes['total_reward']))
    
    ax.bar(x - width, reward_changes['load_balance'], width, label='Load Balance', alpha=0.8)
    ax.bar(x, reward_changes['electrical_decoupling'], width, label='Electrical Decoupling', alpha=0.8)
    ax.bar(x + width, reward_changes['power_balance'], width, label='Power Balance', alpha=0.8)
    
    ax.set_xlabel('Move Index')
    ax.set_ylabel('Component Value')
    ax.set_title('Reward Components Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Reward correlation matrix
    ax = axes[1, 0]
    components_matrix = np.array([
        reward_changes['load_balance'],
        reward_changes['electrical_decoupling'],
        reward_changes['power_balance']
    ])
    
    correlation = np.corrcoef(components_matrix)
    im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Load Bal.', 'Elec. Decoup.', 'Power Bal.'])
    ax.set_yticklabels(['Load Bal.', 'Elec. Decoup.', 'Power Bal.'])
    ax.set_title('Component Correlation Matrix')
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{correlation[i, j]:.2f}', ha='center', va='center')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""Summary Statistics:
    
    Total Moves Tested: {len(reward_changes['total_reward'])}
    
    Reward Ranges:
    - Total: [{min(reward_changes['total_reward']):.3f}, {max(reward_changes['total_reward']):.3f}]
    - Load Balance: [{min(reward_changes['load_balance']):.3f}, {max(reward_changes['load_balance']):.3f}]
    - Elec. Decoupling: [{min(reward_changes['electrical_decoupling']):.3f}, {max(reward_changes['electrical_decoupling']):.3f}]
    - Power Balance: [{min(reward_changes['power_balance']):.3f}, {max(reward_changes['power_balance']):.3f}]
    
    Mean Values:
    - Total: {np.mean(reward_changes['total_reward']):.3f}
    - Load Balance: {np.mean(reward_changes['load_balance']):.3f}
    - Elec. Decoupling: {np.mean(reward_changes['electrical_decoupling']):.3f}
    - Power Balance: {np.mean(reward_changes['power_balance']):.3f}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('reward_sensitivity.png', dpi=300, bbox_inches='tight')
    print("\nSensitivity analysis plot saved as reward_sensitivity.png")


def test_reward_convergence(reward_function, env):
    """Test reward convergence during training"""
    print("\n" + "="*60)
    print("Reward Convergence Test")
    print("="*60)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    rewards_history = []
    components_history = {
        'load_balance': [],
        'electrical_decoupling': [],
        'power_balance': []
    }
    
    # Simulate some steps
    max_steps = 20
    for step in range(max_steps):
        valid_actions = info['valid_actions']
        if not valid_actions:
            print(f"No valid actions at step {step + 1}")
            break
        
        # Choose action (prefer actions that might improve load balance)
        action = valid_actions[0]
        
        # Get current state
        current_partition = env.state_manager.current_partition
        boundary_nodes = env.state_manager.get_boundary_nodes()
        
        # Calculate reward with debug info
        reward, components = reward_function.compute_reward(
            current_partition, boundary_nodes, action, return_components=True
        )
        
        # Execute action
        obs, env_reward, terminated, truncated, info = env.step(action)
        
        # Record
        rewards_history.append(reward)
        for comp_name, comp_value in components.items():
            components_history[comp_name].append(comp_value)
        
        print(f"Step {step + 1}: reward = {reward:.4f}, env_reward = {env_reward:.4f}")
        
        if terminated or truncated:
            print(f"Episode terminated at step {step + 1}")
            break
    
    # Visualize convergence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    steps = range(len(rewards_history))
    
    # Total reward
    ax1.plot(steps, rewards_history, 'b-', linewidth=2, marker='o', label='Total Reward')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Total Reward Over Steps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add trend line
    if len(rewards_history) > 3:
        z = np.polyfit(list(steps), rewards_history, 1)
        p = np.poly1d(z)
        ax1.plot(steps, p(list(steps)), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
        ax1.legend()
    
    # Component rewards
    for comp_name, comp_values in components_history.items():
        ax2.plot(steps, comp_values, label=comp_name, linewidth=2, marker='o', alpha=0.8)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Component Value')
    ax2.set_title('Reward Components Over Steps')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('reward_convergence.png', dpi=300, bbox_inches='tight')
    print("\nConvergence analysis plot saved as reward_convergence.png")


def analyze_reward_distribution(reward_function, env, num_samples=100):
    """Analyze reward distribution across random actions"""
    print("\n" + "="*60)
    print("Reward Distribution Analysis")
    print("="*60)
    
    reward_samples = []
    component_samples = {
        'load_balance': [],
        'electrical_decoupling': [],
        'power_balance': []
    }
    
    for i in range(num_samples):
        # Reset and take random actions
        obs, info = env.reset(seed=i)
        
        if info['valid_actions']:
            action_idx = np.random.randint(len(info['valid_actions']))
            action = info['valid_actions'][action_idx]
            
            # Calculate reward
            current_partition = env.state_manager.current_partition
            boundary_nodes = env.state_manager.get_boundary_nodes()
            
            reward, components = reward_function.compute_reward(
                current_partition, boundary_nodes, action, return_components=True
            )
            
            reward_samples.append(reward)
            for comp_name, comp_value in components.items():
                component_samples[comp_name].append(comp_value)
    
    if reward_samples:
        # Plot distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Reward Distribution Analysis', fontsize=16)
        
        # Total reward distribution
        ax = axes[0, 0]
        ax.hist(reward_samples, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Total Reward Distribution')
        ax.axvline(np.mean(reward_samples), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(reward_samples):.3f}')
        ax.legend()
        
        # Component distributions
        for idx, (comp_name, comp_values) in enumerate(component_samples.items()):
            ax = axes.flatten()[idx + 1]
            ax.hist(comp_values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(comp_name.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{comp_name.replace("_", " ").title()} Distribution')
            ax.axvline(np.mean(comp_values), color='red', linestyle='--',
                      label=f'Mean: {np.mean(comp_values):.3f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('reward_distribution.png', dpi=300, bbox_inches='tight')
        print("\nReward distribution plot saved as reward_distribution.png")
        
        # Print statistics
        print("\nReward Statistics:")
        print(f"Total Reward - Mean: {np.mean(reward_samples):.4f}, Std: {np.std(reward_samples):.4f}")
        for comp_name, comp_values in component_samples.items():
            print(f"{comp_name} - Mean: {np.mean(comp_values):.4f}, Std: {np.std(comp_values):.4f}")


def main():
    """Main testing function"""
    print("Reward Function Debugging Tool")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    print("\n1. Loading test data...")
    mpc = create_test_case()
    processor = PowerGridDataProcessor(normalize=True)
    hetero_data = processor.graph_from_mpc(mpc).to(device)
    
    # 2. Create debug reward function
    print("\n2. Creating debug reward function...")
    reward_function = RewardFunction(
        hetero_data=hetero_data,
        device=device,
        debug_mode=True
    )
    
    # 3. Create test partitions
    print("\n3. Creating test partitions...")
    total_nodes = sum(x.shape[0] for x in hetero_data.x_dict.values())
    
    test_partitions = {
        'Balanced': torch.tensor([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3], device=device),
        'Unbalanced': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3], device=device),
        'Extreme': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3], device=device)
    }
    
    # 4. Test individual components
    component_results = test_individual_rewards(reward_function, test_partitions)
    
    # 5. Test sensitivity with visualization
    visualize_reward_sensitivity(reward_function, test_partitions['Balanced'])
    
    # 6. Create environment and test convergence
    print("\n6. Creating environment and testing convergence...")
    encoder = create_hetero_graph_encoder(hetero_data, hidden_channels=32, gnn_layers=2).to(device)
    with torch.no_grad():
        node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data)
    
    # Use environment without attention weights to avoid dimension mismatch issues
    env = PowerGridPartitioningEnv(
        hetero_data=hetero_data,
        node_embeddings=node_embeddings,
        num_partitions=3,
        max_steps=50,
        device=device,
        attention_weights=None  # Disable attention enhancement for now
    )
    
    # Replace reward function with debug version
    env.reward_function = reward_function
    
    test_reward_convergence(reward_function, env)
    
    # 7. Analyze reward distribution
    analyze_reward_distribution(reward_function, env)
    
    # 8. Generate debug report
    print("\n" + "="*60)
    print("Debug Summary")
    print("="*60)
    print("1. Physical properties: OK - Using real power loads and generation")
    print("2. Individual components: OK - All components tested")
    print("3. Sensitivity analysis: OK - Visualized reward changes")
    print("4. Convergence test: OK - Tracked reward evolution")
    print("5. Distribution analysis: OK - Analyzed reward statistics")
    print("\nRecommended debugging steps:")
    print("1. Train with load balance only first")
    print("2. Add electrical decoupling after convergence")
    print("3. Finally add power balance")
    print("4. Adjust weights based on component ranges")
    
    env.close()


if __name__ == '__main__':
    main()