#!/usr/bin/env python3
"""
Main Training Script for Power Grid Partitioning RL

This script provides a complete training pipeline for the power grid partitioning
reinforcement learning system with configurable hyperparameters and data loading.
"""

import torch
import numpy as np
import argparse
import os
import sys
import warnings
from pathlib import Path
import yaml
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import modules
from src.data_processing import PowerGridDataProcessor
from src.gat import create_hetero_graph_encoder
from src.rl.environment import PowerGridPartitioningEnv
from src.rl.agent import PPOAgent
from src.rl.training import Trainer

# Suppress warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        'data': {
            'case_name': 'ieee14',  # or path to MATPOWER file
            'normalize': True,
            'cache_dir': 'cache'
        },
        'environment': {
            'num_partitions': 3,
            'max_steps': 200,
            'reward_weights': {
                'balance': 1.0,
                'decoupling': 1.0,
                'internal_balance': 1.0
            }
        },
        'gat': {
            'hidden_channels': 64,
            'gnn_layers': 3,
            'heads': 4,
            'output_dim': 128,
            'dropout': 0.1
        },
        'agent': {
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'k_epochs': 4,
            'entropy_coef': 0.01,
            'value_coef': 0.5
        },
        'training': {
            'num_episodes': 1000,
            'max_steps_per_episode': 200,
            'update_interval': 10,
            'save_interval': 100,
            'eval_interval': 50,
            'log_dir': 'logs',
            'checkpoint_dir': 'checkpoints',
            'use_tensorboard': True
        },
        'system': {
            'device': 'auto',  # 'cpu', 'cuda', or 'auto'
            'seed': 42,
            'num_threads': 1
        }
    }


def load_power_grid_data(case_name: str) -> Dict[str, Any]:
    """Load power grid data"""
    if case_name == 'ieee14':
        # IEEE 14-bus test system
        return {
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
            'branch': np.array([
                [1, 2, 0.01938, 0.05917, 0.0528, 100, 110, 120, 0, 0, 1, -360, 360],
                [1, 5, 0.05403, 0.22304, 0.0492, 100, 110, 120, 0, 0, 1, -360, 360],
                [2, 3, 0.04699, 0.19797, 0.0438, 100, 110, 120, 0, 0, 1, -360, 360],
                [2, 4, 0.05811, 0.17632, 0.034, 100, 110, 120, 0, 0, 1, -360, 360],
                [2, 5, 0.05695, 0.17388, 0.0346, 100, 110, 120, 0, 0, 1, -360, 360],
                [3, 4, 0.06701, 0.17103, 0.0128, 100, 110, 120, 0, 0, 1, -360, 360],
                [4, 5, 0.01335, 0.04211, 0.0, 100, 110, 120, 0, 0, 1, -360, 360],
                [4, 7, 0.0, 0.20912, 0.0, 100, 110, 120, 0.978, 0, 1, -360, 360],
                [4, 9, 0.0, 0.55618, 0.0, 100, 110, 120, 0.969, 0, 1, -360, 360],
                [5, 6, 0.0, 0.25202, 0.0, 100, 110, 120, 0.932, 0, 1, -360, 360],
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
                [1, 232.4, -16.9, 10, -10, 1.06, 100, 1, 332.4, 0],
                [2, 40.0, 43.56, 50, -40, 1.045, 100, 1, 140.0, 0],
                [3, 0.0, 25.075, 40, 0, 1.01, 100, 1, 100.0, 0],
                [6, 0.0, 12.73, 24, -6, 1.07, 100, 1, 100.0, 0],
                [8, 0.0, 17.623, 24, -6, 1.09, 100, 1, 100.0, 0],
            ])
        }
    else:
        # Try to load from file
        if os.path.exists(case_name):
            # Implement MATPOWER file loading here
            raise NotImplementedError("MATPOWER file loading not implemented yet")
        else:
            raise ValueError(f"Unknown case: {case_name}")


def setup_device(device_config: str) -> torch.device:
    """Setup compute device"""
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
        
    print(f"Using device: {device}")
    return device


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Power Grid Partitioning RL Agent')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--case', type=str, default='ieee14', help='Power grid case name')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--partitions', type=int, help='Number of partitions')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--save-config', type=str, help='Save default config to file')
    
    args = parser.parse_args()
    
    # Save default config if requested
    if args.save_config:
        config = create_default_config()
        with open(args.save_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Default configuration saved to {args.save_config}")
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        
    # Override config with command line arguments
    if args.case:
        config['data']['case_name'] = args.case
    if args.episodes:
        config['training']['num_episodes'] = args.episodes
    if args.partitions:
        config['environment']['num_partitions'] = args.partitions
        
    print("🚀 Starting Power Grid Partitioning RL Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Case: {config['data']['case_name']}")
    print(f"  Partitions: {config['environment']['num_partitions']}")
    print(f"  Episodes: {config['training']['num_episodes']}")
    
    # Setup system
    torch.manual_seed(config['system']['seed'])
    np.random.seed(config['system']['seed'])
    torch.set_num_threads(config['system']['num_threads'])
    device = setup_device(config['system']['device'])
    
    try:
        # 1. Load and process data
        print("\n📊 Loading and processing power grid data...")
        mpc = load_power_grid_data(config['data']['case_name'])
        
        processor = PowerGridDataProcessor(
            normalize=config['data']['normalize'],
            cache_dir=config['data']['cache_dir']
        )
        hetero_data = processor.graph_from_mpc(mpc)
        hetero_data = hetero_data.to(device)
        
        print(f"✅ Data loaded: {hetero_data}")
        
        # 2. Create GAT encoder and get enhanced embeddings
        print("\n🧠 Creating GAT encoder and computing enhanced embeddings...")
        encoder = create_hetero_graph_encoder(
            hetero_data,
            hidden_channels=config['gat']['hidden_channels'],
            gnn_layers=config['gat']['gnn_layers'],
            heads=config['gat']['heads'],
            output_dim=config['gat']['output_dim']
        ).to(device)

        with torch.no_grad():
            # Step 2: Generate Enhanced Static Node Feature Embedding (H')
            node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data)

        total_nodes = sum(emb.shape[0] for emb in node_embeddings.values())
        print(f"✅ Original embeddings computed for {total_nodes} nodes")
        print(f"✅ Attention weights extracted for {len(attention_weights)} edge types")
        
        # 3. Create RL environment with enhanced embeddings
        print("\n🌍 Creating RL environment with enhanced embeddings...")
        env = PowerGridPartitioningEnv(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=config['environment']['num_partitions'],
            reward_weights=config['environment']['reward_weights'],
            max_steps=config['environment']['max_steps'],
            device=device,
            attention_weights=attention_weights
        )
        
        print(f"✅ Environment created: {env.total_nodes} nodes, {env.num_partitions} partitions")
        
        # 4. Create PPO agent with enhanced embedding dimensions
        print("\n🤖 Creating PPO agent with enhanced embeddings...")
        # Get enhanced embedding dimension (original + attention feature)
        node_embedding_dim = env.state_manager.embedding_dim
        region_embedding_dim = node_embedding_dim * 2

        print(f"   Enhanced node embedding dimension: {node_embedding_dim}")
        print(f"   Region embedding dimension: {region_embedding_dim}")
        
        agent = PPOAgent(
            node_embedding_dim=node_embedding_dim,
            region_embedding_dim=region_embedding_dim,
            num_partitions=env.num_partitions,
            lr_actor=config['agent']['lr_actor'],
            lr_critic=config['agent']['lr_critic'],
            gamma=config['agent']['gamma'],
            eps_clip=config['agent']['eps_clip'],
            k_epochs=config['agent']['k_epochs'],
            entropy_coef=config['agent']['entropy_coef'],
            value_coef=config['agent']['value_coef'],
            device=device
        )
        
        print(f"✅ Agent created with {sum(p.numel() for p in agent.actor.parameters()):,} parameters")
        
        # 5. Create trainer
        print("\n🏋️ Setting up trainer...")
        trainer = Trainer(
            agent=agent,
            env=env,
            log_dir=config['training']['log_dir'],
            checkpoint_dir=config['training']['checkpoint_dir'],
            save_interval=config['training']['save_interval'],
            eval_interval=config['training']['eval_interval'],
            use_tensorboard=config['training']['use_tensorboard']
        )
        
        # 6. Run training or evaluation
        if args.eval_only:
            print("\n📈 Running evaluation...")
            if args.resume:
                trainer.checkpoint_manager.load_checkpoint(agent, args.resume)
                print(f"Loaded model from {args.resume}")
            eval_stats = trainer.evaluate(num_episodes=20)
            print("\nEvaluation Results:")
            for key, value in eval_stats.items():
                print(f"  {key}: {value:.4f}")
        else:
            print("\n🚀 Starting training...")
            history = trainer.train(
                num_episodes=config['training']['num_episodes'],
                max_steps_per_episode=config['training']['max_steps_per_episode'],
                update_interval=config['training']['update_interval'],
                resume_from=args.resume
            )
            
            print("\nTraining completed!")
            print(f"Final mean reward: {np.mean(history['episode_rewards'][-10:]):.3f}")
            
            # Run final evaluation
            print("\n📈 Running final evaluation...")
            eval_stats = trainer.evaluate(num_episodes=20)
            print("\nFinal Evaluation Results:")
            for key, value in eval_stats.items():
                print(f"  {key}: {value:.4f}")
        
        # Clean up
        trainer.close()
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\n🎉 Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
