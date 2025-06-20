"""
Action Space Management for Power Grid Partitioning MDP

This module implements the two-stage action space as specified in the MDP formulation:
1. Select a boundary node (from Bdry_t)
2. Select a target partition (from neighboring partitions)

Includes action masking for topological and physical constraints.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from torch_geometric.data import HeteroData


class ActionSpace:
    """
    Manages the two-stage action space for power grid partitioning
    
    Action Structure:
    - Stage 1: Select a boundary node i_t ∈ Bdry_t
    - Stage 2: Select target partition k_new ≠ k_old from neighboring partitions
    
    Action Representation: (node_idx, target_partition)
    """
    
    def __init__(self, 
                 hetero_data: HeteroData,
                 num_partitions: int,
                 device: torch.device):
        """
        Initialize Action Space
        
        Args:
            hetero_data: Heterogeneous graph data
            num_partitions: Number of target partitions
            device: Torch device for computations
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        self.num_partitions = num_partitions
        
        # Setup adjacency information
        self._setup_adjacency_info()
        
        # Cache for action validation
        self._action_cache = {}
        
    def _setup_adjacency_info(self):
        """Setup adjacency information for action validation"""
        # Get total number of nodes
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # Create global adjacency list
        self.adjacency_list = [set() for _ in range(self.total_nodes)]
        
        # Setup node type mappings
        self._setup_node_mappings()
        
        # Build adjacency list from heterogeneous edges
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            
            # Convert to global indices
            src_global = self._local_to_global(edge_index[0], src_type)
            dst_global = self._local_to_global(edge_index[1], dst_type)
            
            # Add to adjacency list (undirected)
            for src, dst in zip(src_global, dst_global):
                self.adjacency_list[src.item()].add(dst.item())
                self.adjacency_list[dst.item()].add(src.item())
                
    def _setup_node_mappings(self):
        """Setup mappings between local and global node indices"""
        self.node_types = list(self.hetero_data.x_dict.keys())
        self.local_to_global_map = {}
        
        global_idx = 0
        for node_type in self.node_types:
            num_nodes = self.hetero_data.x_dict[node_type].shape[0]
            global_indices = torch.arange(global_idx, global_idx + num_nodes, device=self.device)
            self.local_to_global_map[node_type] = global_indices
            global_idx += num_nodes
            
    def _local_to_global(self, local_indices: torch.Tensor, node_type: str) -> torch.Tensor:
        """Convert local indices to global indices"""
        return self.local_to_global_map[node_type][local_indices]
        
    def get_valid_actions(self, 
                         current_partition: torch.Tensor,
                         boundary_nodes: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Get all valid actions for current state
        
        Args:
            current_partition: Current partition assignments [total_nodes]
            boundary_nodes: Current boundary nodes [num_boundary]
            
        Returns:
            List of valid (node_idx, target_partition) tuples
        """
        valid_actions = []
        
        for node_idx in boundary_nodes:
            node_idx_int = node_idx.item()
            current_node_partition = current_partition[node_idx_int].item()
            
            # Get neighboring partitions
            neighboring_partitions = self._get_neighboring_partitions(
                node_idx_int, current_partition
            )
            
            # Add valid actions for each neighboring partition
            for target_partition in neighboring_partitions:
                if target_partition != current_node_partition:
                    valid_actions.append((node_idx_int, target_partition))
                    
        return valid_actions
        
    def _get_neighboring_partitions(self, 
                                   node_idx: int,
                                   current_partition: torch.Tensor) -> Set[int]:
        """
        Get partitions that are neighbors of the given node
        
        Args:
            node_idx: Global node index
            current_partition: Current partition assignments
            
        Returns:
            Set of neighboring partition IDs
        """
        neighboring_partitions = set()
        
        for neighbor_idx in self.adjacency_list[node_idx]:
            neighbor_partition = current_partition[neighbor_idx].item()
            neighboring_partitions.add(neighbor_partition)
            
        return neighboring_partitions
        
    def is_valid_action(self,
                       action: Tuple[int, int],
                       current_partition: torch.Tensor,
                       boundary_nodes: torch.Tensor) -> bool:
        """
        Check if an action is valid
        
        Args:
            action: (node_idx, target_partition) tuple
            current_partition: Current partition assignments
            boundary_nodes: Current boundary nodes
            
        Returns:
            True if action is valid, False otherwise
        """
        node_idx, target_partition = action
        
        # Check if node is in boundary nodes
        if node_idx not in boundary_nodes:
            return False
            
        # Check if target partition is valid
        if target_partition < 1 or target_partition > self.num_partitions:
            return False
            
        # Check if node is currently in different partition
        current_node_partition = current_partition[node_idx].item()
        if target_partition == current_node_partition:
            return False
            
        # Check if target partition is neighboring
        neighboring_partitions = self._get_neighboring_partitions(node_idx, current_partition)
        if target_partition not in neighboring_partitions:
            return False
            
        return True
        
    def get_action_mask(self,
                       current_partition: torch.Tensor,
                       boundary_nodes: torch.Tensor) -> torch.Tensor:
        """
        Get action mask for current state
        
        Args:
            current_partition: Current partition assignments
            boundary_nodes: Current boundary nodes
            
        Returns:
            Boolean tensor [total_nodes, num_partitions] indicating valid actions
        """
        # Initialize mask (all False)
        action_mask = torch.zeros(
            self.total_nodes, self.num_partitions, 
            dtype=torch.bool, device=self.device
        )
        
        # Set valid actions to True
        for node_idx in boundary_nodes:
            node_idx_int = node_idx.item()
            current_node_partition = current_partition[node_idx_int].item()
            
            # Get neighboring partitions
            neighboring_partitions = self._get_neighboring_partitions(
                node_idx_int, current_partition
            )
            
            # Mark valid target partitions
            for target_partition in neighboring_partitions:
                if target_partition != current_node_partition:
                    # Convert to 0-based indexing for mask
                    action_mask[node_idx_int, target_partition - 1] = True
                    
        return action_mask
        
    def sample_random_action(self,
                           current_partition: torch.Tensor,
                           boundary_nodes: torch.Tensor) -> Optional[Tuple[int, int]]:
        """
        Sample a random valid action
        
        Args:
            current_partition: Current partition assignments
            boundary_nodes: Current boundary nodes
            
        Returns:
            Random valid action or None if no valid actions
        """
        valid_actions = self.get_valid_actions(current_partition, boundary_nodes)
        
        if len(valid_actions) == 0:
            return None
            
        # Sample random action
        action_idx = torch.randint(0, len(valid_actions), (1,)).item()
        return valid_actions[action_idx]
        
    def get_action_space_size(self) -> int:
        """
        Get the maximum size of the action space
        
        Returns:
            Maximum number of possible actions
        """
        return self.total_nodes * self.num_partitions


class ActionMask:
    """
    Advanced action masking with topological and physical constraints
    """
    
    def __init__(self, 
                 hetero_data: HeteroData,
                 device: torch.device):
        """
        Initialize Action Mask
        
        Args:
            hetero_data: Heterogeneous graph data
            device: Torch device for computations
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        
        # Extract physical constraints from edge attributes
        self._setup_physical_constraints()
        
    def _setup_physical_constraints(self):
        """Setup physical constraints from edge attributes"""
        # Extract impedance information for constraint checking
        self.edge_impedances = {}
        
        for edge_type, edge_attr in self.hetero_data.edge_attr_dict.items():
            # Impedance magnitude is typically at index 3 in edge features
            if edge_attr.shape[1] > 3:
                impedances = edge_attr[:, 3]  # |Z| column
                self.edge_impedances[edge_type] = impedances
                
    def apply_connectivity_constraint(self,
                                    action_mask: torch.Tensor,
                                    current_partition: torch.Tensor,
                                    action: Tuple[int, int]) -> bool:
        """
        Check if action maintains partition connectivity
        
        Args:
            action_mask: Current action mask
            current_partition: Current partition assignments
            action: Proposed action
            
        Returns:
            True if action maintains connectivity, False otherwise
        """
        node_idx, target_partition = action
        
        # For now, implement basic connectivity check
        # This could be enhanced with more sophisticated graph connectivity algorithms
        
        # Check if moving this node would disconnect its current partition
        current_node_partition = current_partition[node_idx].item()
        
        # Count nodes in current partition
        current_partition_nodes = torch.where(current_partition == current_node_partition)[0]
        
        # If this is the only node in the partition, allow the move
        if len(current_partition_nodes) <= 1:
            return True
            
        # For more complex connectivity checking, we would need to:
        # 1. Temporarily remove the node from its current partition
        # 2. Check if the remaining nodes are still connected
        # 3. Check if the target partition remains connected after adding the node
        
        # For now, return True (basic implementation)
        return True
        
    def apply_forced_moves(self,
                          action_mask: torch.Tensor,
                          current_partition: torch.Tensor,
                          boundary_nodes: torch.Tensor,
                          adjacency_list: List[Set[int]]) -> torch.Tensor:
        """
        Apply forced move constraints
        
        If all neighbors of a boundary node are in the same partition,
        force the node to move to that partition.
        
        Args:
            action_mask: Current action mask
            current_partition: Current partition assignments
            boundary_nodes: Current boundary nodes
            adjacency_list: Graph adjacency list
            
        Returns:
            Updated action mask with forced moves
        """
        updated_mask = action_mask.clone()
        
        for node_idx in boundary_nodes:
            node_idx_int = node_idx.item()
            current_node_partition = current_partition[node_idx_int].item()
            
            # Get all neighbor partitions
            neighbor_partitions = set()
            for neighbor_idx in adjacency_list[node_idx_int]:
                neighbor_partition = current_partition[neighbor_idx].item()
                neighbor_partitions.add(neighbor_partition)
                
            # Remove current partition from neighbors
            neighbor_partitions.discard(current_node_partition)
            
            # If all neighbors are in a single partition, force move
            if len(neighbor_partitions) == 1:
                forced_partition = list(neighbor_partitions)[0]
                
                # Clear all other actions for this node
                updated_mask[node_idx_int, :] = False
                
                # Enable only the forced action
                updated_mask[node_idx_int, forced_partition - 1] = True
                
        return updated_mask
        
    def apply_physical_constraints(self,
                                 action_mask: torch.Tensor,
                                 current_partition: torch.Tensor,
                                 impedance_threshold: float = 1.0) -> torch.Tensor:
        """
        Apply physical constraints based on electrical properties
        
        Args:
            action_mask: Current action mask
            current_partition: Current partition assignments
            impedance_threshold: Threshold for impedance-based constraints
            
        Returns:
            Updated action mask with physical constraints
        """
        # This is a placeholder for more sophisticated physical constraints
        # Could include:
        # - Impedance-based connectivity requirements
        # - Power flow constraints
        # - Voltage stability requirements
        
        return action_mask
