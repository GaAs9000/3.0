"""
Reward Function for Power Grid Partitioning MDP

This module implements the composite reward function as specified in the MDP formulation:
R_t = w1 * R_balance + w2 * R_decoupling + w3 * R_internal_balance

Where:
- R_balance: Partition load balance reward (-Var(L1, ..., LK))
- R_decoupling: Electrical decoupling reward (-Σ|Y_uv| for coupling edges)
- R_internal_balance: Internal power balance reward (-Σ(P_gen - P_load)²)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import HeteroData


class RewardFunction:
    """
    Composite reward function for power grid partitioning
    
    Implements the three-component reward function:
    1. Partition Load Balance: Minimizes variance in partition loads
    2. Electrical Decoupling: Minimizes coupling between partitions
    3. Internal Power Balance: Minimizes power imbalance within partitions
    """
    
    def __init__(self,
                 hetero_data: HeteroData,
                 reward_weights: Optional[Dict[str, float]] = None,
                 device: torch.device = None):
        """
        Initialize Reward Function
        
        Args:
            hetero_data: Heterogeneous graph data
            reward_weights: Weights for reward components
            device: Torch device for computations
        """
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        
        # Default reward weights
        self.reward_weights = reward_weights or {
            'balance': 1.0,
            'decoupling': 1.0,
            'internal_balance': 1.0
        }
        
        # Extract power system data
        self._setup_node_mappings()
        self._extract_power_data()
        self._extract_electrical_data()
        
    def _extract_power_data(self):
        """Extract power load and generation data from node features"""
        # Concatenate all node features
        all_node_features = []
        self.node_type_ranges = {}
        
        start_idx = 0
        for node_type, features in self.hetero_data.x_dict.items():
            num_nodes = features.shape[0]
            all_node_features.append(features)
            self.node_type_ranges[node_type] = (start_idx, start_idx + num_nodes)
            start_idx += num_nodes
            
        self.all_node_features = torch.cat(all_node_features, dim=0)
        
        # Extract power data (assuming standard feature order from data processing)
        # Features: ['Pd', 'Qd', 'Gs', 'Bs', 'Vm', 'Va', 'Vmax', 'Vmin', 'degree',
        #           'Pg', 'Qg', 'Pg_max', 'Pg_min', 'is_gen']
        
        self.load_active = self.all_node_features[:, 0]  # Pd
        self.load_reactive = self.all_node_features[:, 1]  # Qd
        
        if self.all_node_features.shape[1] > 9:  # Has generation data
            self.gen_active = self.all_node_features[:, 9]   # Pg
            self.gen_reactive = self.all_node_features[:, 10]  # Qg
        else:
            # No generation data - assume zero generation
            self.gen_active = torch.zeros_like(self.load_active)
            self.gen_reactive = torch.zeros_like(self.load_reactive)
            
    def _extract_electrical_data(self):
        """Extract electrical parameters from edge features"""
        # Collect all edge data with global node indices
        self.coupling_edges = []
        self.edge_admittances = []
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            
            # Convert to global indices
            src_global = self._local_to_global(edge_index[0], src_type)
            dst_global = self._local_to_global(edge_index[1], dst_type)
            
            # Store edge information
            global_edges = torch.stack([src_global, dst_global], dim=0)
            self.coupling_edges.append(global_edges)
            
            # Extract admittance (y column, typically index 4)
            if edge_attr.shape[1] > 4:
                admittances = edge_attr[:, 4]  # y column
            else:
                # Fallback: use inverse of impedance magnitude
                impedances = edge_attr[:, 3] if edge_attr.shape[1] > 3 else torch.ones(edge_attr.shape[0])
                admittances = 1.0 / (impedances + 1e-10)
                
            self.edge_admittances.append(admittances)
            
        # Concatenate all edges
        if self.coupling_edges:
            self.all_edges = torch.cat(self.coupling_edges, dim=1)
            self.all_admittances = torch.cat(self.edge_admittances, dim=0)
        else:
            self.all_edges = torch.empty(2, 0, device=self.device)
            self.all_admittances = torch.empty(0, device=self.device)
            
    def _setup_node_mappings(self):
        """Setup node type mappings"""
        self.node_types = list(self.hetero_data.x_dict.keys())
        self.local_to_global_map = {}
        
        global_idx = 0
        for node_type in self.node_types:
            num_nodes = self.hetero_data.x_dict[node_type].shape[0]
            global_indices = torch.arange(global_idx, global_idx + num_nodes, device=self.device)
            self.local_to_global_map[node_type] = global_indices
            global_idx += num_nodes
            
        self.total_nodes = global_idx
        
    def _local_to_global(self, local_indices: torch.Tensor, node_type: str) -> torch.Tensor:
        """Convert local indices to global indices"""
        return self.local_to_global_map[node_type][local_indices]
        
    def compute_reward(self,
                      current_partition: torch.Tensor,
                      boundary_nodes: torch.Tensor,
                      action: Optional[Tuple[int, int]] = None) -> float:
        """
        Compute composite reward for current state
        
        Args:
            current_partition: Current partition assignments [total_nodes]
            boundary_nodes: Current boundary nodes
            action: Last action taken (optional, for action-specific rewards)
            
        Returns:
            Composite reward value
        """
        # Compute individual reward components
        r_balance = self._compute_balance_reward(current_partition)
        r_decoupling = self._compute_decoupling_reward(current_partition)
        r_internal_balance = self._compute_internal_balance_reward(current_partition)
        
        # Weighted combination
        total_reward = (
            self.reward_weights['balance'] * r_balance +
            self.reward_weights['decoupling'] * r_decoupling +
            self.reward_weights['internal_balance'] * r_internal_balance
        )
        
        return total_reward.item()
        
    def _compute_balance_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        Compute partition load balance reward: -Var(L1, ..., LK)
        
        Args:
            current_partition: Current partition assignments
            
        Returns:
            Load balance reward (higher is better)
        """
        num_partitions = current_partition.max().item()
        partition_loads = torch.zeros(num_partitions, device=self.device)
        
        # Compute total load for each partition
        for partition_id in range(1, num_partitions + 1):
            partition_mask = (current_partition == partition_id)
            if partition_mask.any():
                partition_load = self.load_active[partition_mask].sum()
                partition_loads[partition_id - 1] = partition_load
                
        # Compute variance (negative for reward)
        load_variance = torch.var(partition_loads)
        return -load_variance
        
    def _compute_decoupling_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        Compute electrical decoupling reward: -Σ|Y_uv| for inter-partition edges
        
        Args:
            current_partition: Current partition assignments
            
        Returns:
            Decoupling reward (higher is better)
        """
        if self.all_edges.shape[1] == 0:
            return torch.tensor(0.0, device=self.device)
            
        # Find inter-partition edges (coupling edges)
        src_partitions = current_partition[self.all_edges[0]]
        dst_partitions = current_partition[self.all_edges[1]]
        
        # Mask for inter-partition edges
        coupling_mask = (src_partitions != dst_partitions)
        
        # Sum admittances of coupling edges
        if coupling_mask.any():
            coupling_admittances = self.all_admittances[coupling_mask]
            total_coupling = torch.sum(torch.abs(coupling_admittances))
        else:
            total_coupling = torch.tensor(0.0, device=self.device)
            
        # Negative for reward (minimize coupling)
        return -total_coupling
        
    def _compute_internal_balance_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        Compute internal power balance reward: -Σ(P_gen - P_load)² for each partition
        
        Args:
            current_partition: Current partition assignments
            
        Returns:
            Internal balance reward (higher is better)
        """
        num_partitions = current_partition.max().item()
        total_imbalance = torch.tensor(0.0, device=self.device)
        
        # Compute power imbalance for each partition
        for partition_id in range(1, num_partitions + 1):
            partition_mask = (current_partition == partition_id)
            
            if partition_mask.any():
                # Sum generation and load in this partition
                partition_gen = self.gen_active[partition_mask].sum()
                partition_load = self.load_active[partition_mask].sum()
                
                # Power imbalance (generation - load)
                imbalance = partition_gen - partition_load
                total_imbalance += imbalance ** 2
                
        # Negative for reward (minimize imbalance)
        return -total_imbalance
        
    def compute_detailed_metrics(self, current_partition: torch.Tensor) -> Dict[str, float]:
        """
        Compute detailed reward metrics for analysis
        
        Args:
            current_partition: Current partition assignments
            
        Returns:
            Dictionary with detailed metrics
        """
        # Individual reward components
        r_balance = self._compute_balance_reward(current_partition)
        r_decoupling = self._compute_decoupling_reward(current_partition)
        r_internal_balance = self._compute_internal_balance_reward(current_partition)
        
        # Additional metrics
        num_partitions = current_partition.max().item()
        partition_loads = torch.zeros(num_partitions, device=self.device)
        partition_sizes = torch.zeros(num_partitions, device=self.device)
        
        for partition_id in range(1, num_partitions + 1):
            partition_mask = (current_partition == partition_id)
            partition_loads[partition_id - 1] = self.load_active[partition_mask].sum()
            partition_sizes[partition_id - 1] = partition_mask.sum()
            
        # Coupling edges count
        if self.all_edges.shape[1] > 0:
            src_partitions = current_partition[self.all_edges[0]]
            dst_partitions = current_partition[self.all_edges[1]]
            coupling_edges = (src_partitions != dst_partitions).sum()
        else:
            coupling_edges = 0
            
        return {
            'balance_reward': r_balance.item(),
            'decoupling_reward': r_decoupling.item(),
            'internal_balance_reward': r_internal_balance.item(),
            'total_reward': (
                self.reward_weights['balance'] * r_balance +
                self.reward_weights['decoupling'] * r_decoupling +
                self.reward_weights['internal_balance'] * r_internal_balance
            ).item(),
            'load_variance': torch.var(partition_loads).item(),
            'load_cv': (torch.std(partition_loads) / torch.mean(partition_loads)).item() if torch.mean(partition_loads) > 0 else 0.0,
            'coupling_edges': coupling_edges.item() if hasattr(coupling_edges, 'item') else coupling_edges,
            'partition_sizes': partition_sizes.cpu().numpy().tolist(),
            'partition_loads': partition_loads.cpu().numpy().tolist(),
        }
        
    def set_reward_weights(self, new_weights: Dict[str, float]):
        """
        Update reward weights
        
        Args:
            new_weights: New reward weights
        """
        self.reward_weights.update(new_weights)
        
    def get_reward_weights(self) -> Dict[str, float]:
        """Get current reward weights"""
        return self.reward_weights.copy()
