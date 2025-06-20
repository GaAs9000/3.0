"""
Power Grid Partitioning MDP Environment

This module implements the core MDP environment for power grid partitioning
as specified in the project documentation. The environment follows the 
"top-down optimization" paradigm where METIS provides initial partitioning
and RL performs iterative fine-tuning.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any
from torch_geometric.data import HeteroData
import copy

from .state import StateManager
from .action_space import ActionSpace, ActionMask
from .reward import RewardFunction
from .utils import MetisInitializer, PartitionEvaluator


class PowerGridPartitioningEnv:
    """
    Power Grid Partitioning MDP Environment
    
    Implements the MDP formulation described in the documentation:
    - State: Node embeddings H (static) + Node assignments z_t (dynamic) + Boundary nodes
    - Action: Two-stage decision (node selection + partition selection)
    - Reward: Composite function (balance + decoupling + internal_balance)
    - Transitions: Single node reassignment per step
    """
    
    def __init__(self,
                 hetero_data: HeteroData,
                 node_embeddings: Dict[str, torch.Tensor],
                 num_partitions: int,
                 reward_weights: Dict[str, float] = None,
                 max_steps: int = 200,
                 device: torch.device = None):
        """
        Initialize the Power Grid Partitioning Environment
        
        Args:
            hetero_data: Heterogeneous graph data from data processing
            node_embeddings: Pre-computed node embeddings from GAT encoder
            num_partitions: Number of target partitions (K)
            reward_weights: Weights for reward components
            max_steps: Maximum steps per episode
            device: Torch device for computations
        """
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        self.num_partitions = num_partitions
        self.max_steps = max_steps
        
        # Initialize core components
        self.state_manager = StateManager(hetero_data, node_embeddings, device)
        self.action_space = ActionSpace(hetero_data, num_partitions, device)
        self.reward_function = RewardFunction(hetero_data, reward_weights, device)
        self.metis_initializer = MetisInitializer(hetero_data, device)
        self.evaluator = PartitionEvaluator(hetero_data, device)
        
        # Environment state
        self.current_step = 0
        self.episode_history = []
        self.is_terminated = False
        self.is_truncated = False
        
        # Cache frequently used data
        self._setup_cached_data()
        
    def _setup_cached_data(self):
        """Setup frequently accessed cached data"""
        # Total number of nodes across all types
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # Global node mapping (local indices to global indices)
        self.global_node_mapping = self.state_manager.get_global_node_mapping()
        
        # Edge information for reward computation
        self.edge_info = self._extract_edge_info()
        
    def _extract_edge_info(self) -> Dict[str, torch.Tensor]:
        """Extract edge information needed for reward computation"""
        edge_info = {}
        
        # Collect all edges and their attributes
        all_edges = []
        all_edge_attrs = []
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            
            # Convert local indices to global indices
            src_type, _, dst_type = edge_type
            src_global = self.state_manager.local_to_global(edge_index[0], src_type)
            dst_global = self.state_manager.local_to_global(edge_index[1], dst_type)
            
            global_edges = torch.stack([src_global, dst_global], dim=0)
            all_edges.append(global_edges)
            all_edge_attrs.append(edge_attr)
        
        edge_info['edge_index'] = torch.cat(all_edges, dim=1)
        edge_info['edge_attr'] = torch.cat(all_edge_attrs, dim=0)
        
        return edge_info
        
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observation: Initial state observation
            info: Additional information
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Initialize partition using METIS
        initial_partition = self.metis_initializer.initialize_partition(self.num_partitions)
        
        # Reset state manager with initial partition
        self.state_manager.reset(initial_partition)
        
        # Reset environment state
        self.current_step = 0
        self.episode_history = []
        self.is_terminated = False
        self.is_truncated = False
        
        # Get initial observation
        observation = self.state_manager.get_observation()
        
        # Compute initial metrics
        initial_metrics = self.evaluator.evaluate_partition(
            self.state_manager.current_partition
        )
        
        info = {
            'step': self.current_step,
            'metrics': initial_metrics,
            'partition': self.state_manager.current_partition.clone(),
            'boundary_nodes': self.state_manager.get_boundary_nodes(),
            'valid_actions': self.action_space.get_valid_actions(
                self.state_manager.current_partition,
                self.state_manager.get_boundary_nodes()
            )
        }
        
        return observation, info
        
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Tuple of (node_idx, target_partition)
            
        Returns:
            observation: Next state observation
            reward: Immediate reward
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        if self.is_terminated or self.is_truncated:
            raise RuntimeError("Cannot step in terminated/truncated environment. Call reset() first.")
            
        # Validate action
        if not self.action_space.is_valid_action(
            action, 
            self.state_manager.current_partition,
            self.state_manager.get_boundary_nodes()
        ):
            # Invalid action - return negative reward and terminate
            observation = self.state_manager.get_observation()
            reward = -10.0  # Large negative reward for invalid action
            self.is_terminated = True
            
            info = {
                'step': self.current_step,
                'invalid_action': True,
                'action': action
            }
            
            return observation, reward, True, False, info
            
        # Execute action
        node_idx, target_partition = action
        old_partition = self.state_manager.current_partition[node_idx].item()
        
        # Update state
        self.state_manager.update_partition(node_idx, target_partition)
        
        # Compute reward
        reward = self.reward_function.compute_reward(
            self.state_manager.current_partition,
            self.state_manager.get_boundary_nodes(),
            action
        )
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        terminated, truncated = self._check_termination()
        
        # Get next observation
        observation = self.state_manager.get_observation()
        
        # Compute current metrics
        current_metrics = self.evaluator.evaluate_partition(
            self.state_manager.current_partition
        )
        
        # Record step in history
        step_info = {
            'step': self.current_step,
            'action': action,
            'old_partition': old_partition,
            'new_partition': target_partition,
            'reward': reward,
            'metrics': current_metrics
        }
        self.episode_history.append(step_info)
        
        info = {
            'step': self.current_step,
            'metrics': current_metrics,
            'partition': self.state_manager.current_partition.clone(),
            'boundary_nodes': self.state_manager.get_boundary_nodes(),
            'valid_actions': self.action_space.get_valid_actions(
                self.state_manager.current_partition,
                self.state_manager.get_boundary_nodes()
            ) if not (terminated or truncated) else [],
            'episode_history': self.episode_history
        }
        
        self.is_terminated = terminated
        self.is_truncated = truncated
        
        return observation, reward, terminated, truncated, info
        
    def _check_termination(self) -> Tuple[bool, bool]:
        """
        Check if episode should be terminated or truncated
        
        Returns:
            terminated: Natural termination (convergence or no valid actions)
            truncated: Artificial termination (max steps reached)
        """
        # Check for truncation (max steps)
        if self.current_step >= self.max_steps:
            return False, True
            
        # Check for natural termination
        boundary_nodes = self.state_manager.get_boundary_nodes()
        valid_actions = self.action_space.get_valid_actions(
            self.state_manager.current_partition,
            boundary_nodes
        )
        
        # No valid actions remaining
        if len(valid_actions) == 0:
            return True, False
            
        # Convergence check (if enabled)
        if self._check_convergence():
            return True, False
            
        return False, False
        
    def _check_convergence(self, window_size: int = 10, threshold: float = 0.01) -> bool:
        """
        Check if the partition has converged based on recent reward history
        
        Args:
            window_size: Number of recent steps to consider
            threshold: Convergence threshold
            
        Returns:
            True if converged, False otherwise
        """
        if len(self.episode_history) < window_size:
            return False
            
        recent_rewards = [step['reward'] for step in self.episode_history[-window_size:]]
        reward_std = np.std(recent_rewards)
        
        return reward_std < threshold
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the current state of the environment
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', or 'ansi')
            
        Returns:
            Rendered output (depends on mode)
        """
        if mode == 'ansi':
            # Text-based rendering
            output = []
            output.append(f"Step: {self.current_step}/{self.max_steps}")
            output.append(f"Partitions: {self.num_partitions}")
            output.append(f"Total nodes: {self.total_nodes}")
            
            # Partition distribution
            partition_counts = torch.bincount(
                self.state_manager.current_partition, 
                minlength=self.num_partitions + 1
            )[1:]  # Skip partition 0
            output.append(f"Partition sizes: {partition_counts.tolist()}")
            
            # Boundary nodes
            boundary_nodes = self.state_manager.get_boundary_nodes()
            output.append(f"Boundary nodes: {len(boundary_nodes)}")
            
            return '\n'.join(output)
            
        elif mode == 'human':
            print(self.render('ansi'))
            return None
            
        else:
            raise NotImplementedError(f"Render mode '{mode}' not implemented")
            
    def close(self):
        """Clean up environment resources"""
        # Clear cached data
        if hasattr(self, 'edge_info'):
            del self.edge_info
        if hasattr(self, 'global_node_mapping'):
            del self.global_node_mapping
            
        # Clear component references
        self.state_manager = None
        self.action_space = None
        self.reward_function = None
        self.metis_initializer = None
        self.evaluator = None
        
    def get_action_mask(self) -> torch.Tensor:
        """
        Get action mask for current state
        
        Returns:
            Boolean tensor indicating valid actions
        """
        return self.action_space.get_action_mask(
            self.state_manager.current_partition,
            self.state_manager.get_boundary_nodes()
        )
        
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get detailed information about current state
        
        Returns:
            Dictionary with state information
        """
        return {
            'current_partition': self.state_manager.current_partition.clone(),
            'boundary_nodes': self.state_manager.get_boundary_nodes(),
            'step': self.current_step,
            'max_steps': self.max_steps,
            'num_partitions': self.num_partitions,
            'total_nodes': self.total_nodes,
            'is_terminated': self.is_terminated,
            'is_truncated': self.is_truncated
        }
