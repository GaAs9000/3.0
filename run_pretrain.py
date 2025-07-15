#!/usr/bin/env python3
"""
Main entry point for GNN pre-training.

This script orchestrates the pre-training process by:
1. Loading the main configuration.
2. Initializing the necessary components:
   - DataManager for handling base case data.
   - ScaleAwareSyntheticGenerator for creating diverse training data.
   - GNN Encoder model.
   - GNNPretrainer which encapsulates the training loop.
3. Starting the pre-training process.
"""

import yaml
import torch
import argparse
import sys
from pathlib import Path

# Add project root to path for module resolution
sys.path.append(str(Path(__file__).parent.parent))

from code.src.data_processing import PowerGridDataProcessor
from code.src.gat import create_production_encoder
from code.src.pretrain.gnn_pretrainer import GNNPretrainer, PretrainConfig
from code.src.rl.scale_aware_generator import ScaleAwareSyntheticGenerator, GridGenerationConfig
from train import DataManager # Re-use the DataManager from the main training script

def main(config_path: str):
    """
    Main function to run the GNN pre-training.

    Args:
        config_path: Path to the main YAML configuration file.
    """
    # 1. Load Configuration
    print("🚀 1. Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration loaded.")

    # 2. Initialize Components
    print("\n🚀 2. Initializing components...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DataManager for loading base cases
    data_manager = DataManager(config)
    
    # Scale-Aware Generator for creating diverse data
    multi_scale_cfg = config.get('multi_scale_generation', {})
    gen_cfg = GridGenerationConfig(**multi_scale_cfg.get('topology_variation', {}))
    scale_generator = ScaleAwareSyntheticGenerator(
        base_case_loader=data_manager.load_power_grid_data,
        config=gen_cfg,
        seed=config['system']['seed']
    )
    print("✅ ScaleAwareSyntheticGenerator initialized.")

    # A sample HeteroData object is needed to initialize the GNN encoder's architecture
    print("   - Creating a sample HeteroData object for model initialization...")
    sample_mpc = data_manager.load_power_grid_data('ieee30') # Use a medium-sized case
    processor = PowerGridDataProcessor(normalize=True, cache_dir='data/cache')
    sample_hetero_data = processor.graph_from_mpc(sample_mpc, config).to(device)
    print("✅ Sample HeteroData created.")
    
    # GNN Encoder
    gat_config = config.get('gat', {})
    encoder = create_production_encoder(sample_hetero_data, gat_config).to(device)
    print("✅ GNN Encoder created.")
    
    # Pre-trainer
    pretrain_cfg_dict = config.get('gnn_pretrain', {})
    pretrain_config = PretrainConfig(**pretrain_cfg_dict)
    pretrainer = GNNPretrainer(encoder, pretrain_config)
    print("✅ GNNPretrainer initialized.")

    # 3. Start Pre-training
    print("\n🚀 3. Starting GNN pre-training...")
    pretrainer.train(data_generator=scale_generator)
    print("\n✅ Pre-training finished successfully!")

    # 4. Final summary
    print("\n📊 Pre-training Summary:")
    print(f"   - Final model saved to: {pretrain_config.checkpoint_dir}/final_model.pt")
    print(f"   - Best model saved to: {pretrain_config.checkpoint_dir}/best_model.pt")
    print(f"   - TensorBoard logs can be found in: {pretrain_config.log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GNN Pre-training")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the main configuration file.'
    )
    args = parser.parse_args()
    main(args.config) 