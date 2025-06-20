# Power Grid Partitioning RL Module

This module implements a complete reinforcement learning system for power grid partitioning based on the MDP formulation described in the project documentation.

## Overview

The system follows a "top-down optimization" paradigm where:
1. **METIS** provides initial partitioning
2. **RL agent** performs iterative fine-tuning to optimize the partition

## Components

### Core Components

- **`environment.py`**: MDP environment implementing state space, action space, and transitions
- **`agent.py`**: PPO agent with actor-critic networks for two-stage action selection
- **`state.py`**: State management with node embeddings and boundary tracking
- **`action_space.py`**: Two-stage action space with masking constraints
- **`reward.py`**: Composite reward function (balance + decoupling + internal balance)
- **`utils.py`**: METIS initialization and partition evaluation utilities
- **`training.py`**: Training infrastructure with logging and checkpointing

### State Representation

The MDP state consists of:
- **Node Feature Embeddings (H)**: Static, pre-computed from GAT encoder
- **Node Assignment Labels (z_t)**: Dynamic partition assignments
- **Boundary Nodes**: Nodes with neighbors in different partitions
- **Region Embeddings**: Aggregated embeddings per partition

### Action Space

Two-stage action selection:
1. **Node Selection**: Choose a boundary node to move
2. **Partition Selection**: Choose target partition from neighboring partitions

Action masking enforces:
- Only boundary nodes can be moved
- Only moves to neighboring partitions are allowed
- Connectivity constraints (optional)

### Reward Function

Composite reward with three components:
- **R_balance**: Partition load balance (-Var(L₁, ..., Lₖ))
- **R_decoupling**: Electrical decoupling (-Σ|Y_uv| for coupling edges)
- **R_internal_balance**: Internal power balance (-Σ(P_gen - P_load)²)

## Usage

### Basic Training

```python
from src.rl import PowerGridPartitioningEnv, PPOAgent, Trainer

# Create environment
env = PowerGridPartitioningEnv(
    hetero_data=hetero_data,
    node_embeddings=node_embeddings,
    num_partitions=3
)

# Create agent
agent = PPOAgent(
    node_embedding_dim=128,
    region_embedding_dim=256,
    num_partitions=3
)

# Train
trainer = Trainer(agent, env)
history = trainer.train(num_episodes=1000)
```

### Using the Training Script

```bash
# Quick training with default settings
python train_rl.py --case ieee14 --episodes 100 --partitions 3

# Training with custom configuration
python train_rl.py --config config_rl.yaml

# Resume from checkpoint
python train_rl.py --resume checkpoints/best_model.pt

# Evaluation only
python train_rl.py --eval-only --resume checkpoints/best_model.pt
```

### Configuration

Use `config_rl.yaml` to configure:
- Data loading and preprocessing
- Environment parameters
- GAT encoder settings
- PPO hyperparameters
- Training settings

## Key Features

### Physics-Guided Learning
- Incorporates electrical impedance in attention mechanisms
- Reward function based on power system objectives
- Action masking for physical constraints

### Scalable Architecture
- Heterogeneous graph representation
- Efficient boundary node tracking
- Incremental state updates

### Robust Training
- PPO with action masking
- Comprehensive logging and checkpointing
- Curriculum learning support
- Evaluation metrics

### Integration
- Compatible with existing data processing pipeline
- Uses pre-trained GAT embeddings
- Supports multiple power grid formats

## Testing

Run integration tests:
```bash
python test/test_rl_integration.py
```

This validates the entire pipeline end-to-end with a test power grid.

## Performance Tips

1. **GPU Usage**: Set `device: cuda` in config for GPU acceleration
2. **Batch Updates**: Increase `update_interval` for more stable training
3. **Memory**: Use smaller embedding dimensions for large grids
4. **Convergence**: Monitor reward curves and adjust learning rates

## Troubleshooting

### Common Issues

1. **No Valid Actions**: Check boundary node computation and action masking
2. **Training Instability**: Reduce learning rates or increase clipping
3. **Memory Issues**: Reduce batch size or embedding dimensions
4. **Slow Training**: Enable GPU or reduce network complexity

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extension Points

The modular design allows easy extension:

- **Custom Reward Functions**: Modify `reward.py`
- **Advanced Action Spaces**: Extend `action_space.py`
- **Different Algorithms**: Replace PPO in `agent.py`
- **Curriculum Learning**: Implement in `training.py`

## References

See the main project documentation for the complete MDP formulation and mathematical details.
