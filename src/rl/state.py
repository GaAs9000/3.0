"""
State Management for Power Grid Partitioning MDP

This module manages the state representation as specified in the MDP formulation:
- Node Feature Embeddings (H): Static, pre-computed from GAT encoder
- Node Assignment Labels (z_t): Dynamic, changes with each action
- Boundary Nodes: Derived from current partition
- Region-Aggregated Embeddings: For policy input
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from torch_geometric.data import HeteroData


class StateManager:
    """
    Manages the MDP state representation for power grid partitioning
    
    State Components:
    1. Node Feature Embeddings (H) - Static matrix from GAT encoder
    2. Node Assignment Labels (z_t) - Dynamic partition assignments
    3. Boundary Nodes (Bdry_t) - Nodes with neighbors in different partitions
    4. Region-Aggregated Embeddings - Mean/max pooled embeddings per region
    """
    
    def __init__(self, 
                 hetero_data: HeteroData,
                 node_embeddings: Dict[str, torch.Tensor],
                 device: torch.device):
        """
        Initialize State Manager
        
        Args:
            hetero_data: Heterogeneous graph data
            node_embeddings: Pre-computed node embeddings from GAT encoder
            device: Torch device for computations
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        
        # Setup node mappings and embeddings
        self._setup_node_mappings()
        self._setup_node_embeddings(node_embeddings)
        self._setup_adjacency_info()
        
        # State variables
        self.current_partition = None
        self.boundary_nodes = None
        self.region_embeddings = None
        
    def _setup_node_mappings(self):
        """Setup mappings between local and global node indices"""
        self.node_types = list(self.hetero_data.x_dict.keys())
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # Create mapping from local indices to global indices
        self.local_to_global_map = {}
        self.global_to_local_map = {}
        
        global_idx = 0
        for node_type in self.node_types:
            num_nodes = self.hetero_data.x_dict[node_type].shape[0]
            
            # Local to global mapping for this type
            local_indices = torch.arange(num_nodes, device=self.device)
            global_indices = torch.arange(global_idx, global_idx + num_nodes, device=self.device)
            
            self.local_to_global_map[node_type] = global_indices
            
            # Global to local mapping
            for local_idx, global_idx_val in zip(local_indices, global_indices):
                self.global_to_local_map[global_idx_val.item()] = (node_type, local_idx.item())
                
            global_idx += num_nodes
            
    def _setup_node_embeddings(self, node_embeddings: Dict[str, torch.Tensor]):
        """Setup concatenated node embeddings matrix H"""
        # Concatenate all node embeddings into a single matrix
        embedding_list = []
        for node_type in self.node_types:
            embeddings = node_embeddings[node_type].to(self.device)
            embedding_list.append(embeddings)
            
        self.node_embeddings = torch.cat(embedding_list, dim=0)  # Shape: [total_nodes, embedding_dim]
        self.embedding_dim = self.node_embeddings.shape[1]
        
    def _setup_adjacency_info(self):
        """Setup adjacency information for boundary node computation"""
        # Create global adjacency list
        self.adjacency_list = [[] for _ in range(self.total_nodes)]
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            
            # Convert to global indices
            src_global = self.local_to_global(edge_index[0], src_type)
            dst_global = self.local_to_global(edge_index[1], dst_type)
            
            # Add to adjacency list
            for src, dst in zip(src_global, dst_global):
                self.adjacency_list[src.item()].append(dst.item())
                
    def local_to_global(self, local_indices: torch.Tensor, node_type: str) -> torch.Tensor:
        """Convert local indices to global indices for a given node type"""
        return self.local_to_global_map[node_type][local_indices]
        
    def global_to_local(self, global_idx: int) -> Tuple[str, int]:
        """Convert global index to (node_type, local_index)"""
        return self.global_to_local_map[global_idx]
        
    def reset(self, initial_partition: torch.Tensor):
        """
        Reset state with initial partition
        
        Args:
            initial_partition: Initial partition assignments [total_nodes]
        """
        self.current_partition = initial_partition.to(self.device)
        self._update_derived_state()
        
    def update_partition(self, node_idx: int, new_partition: int):
        """
        Update partition assignment for a single node
        
        Args:
            node_idx: Global node index
            new_partition: New partition assignment
        """
        old_partition = self.current_partition[node_idx].item()
        self.current_partition[node_idx] = new_partition
        
        # Efficiently update boundary nodes
        self._update_boundary_nodes_incremental(node_idx, old_partition, new_partition)
        
        # Update region embeddings for affected partitions
        self._update_region_embeddings_incremental(old_partition, new_partition)
        
    def _update_derived_state(self):
        """Update all derived state components"""
        self._compute_boundary_nodes()
        self._compute_region_embeddings()
        
    def _compute_boundary_nodes(self):
        """Compute boundary nodes from current partition"""
        boundary_set = set()
        
        for node_idx in range(self.total_nodes):
            node_partition = self.current_partition[node_idx].item()
            
            # Check if any neighbor is in a different partition
            for neighbor_idx in self.adjacency_list[node_idx]:
                neighbor_partition = self.current_partition[neighbor_idx].item()
                if neighbor_partition != node_partition:
                    boundary_set.add(node_idx)
                    break
                    
        self.boundary_nodes = torch.tensor(list(boundary_set), device=self.device)
        
    def _update_boundary_nodes_incremental(self, changed_node: int, old_partition: int, new_partition: int):
        """Incrementally update boundary nodes after a single node change"""
        # Convert to set for efficient operations
        if self.boundary_nodes is not None:
            try:
                boundary_set = set(self.boundary_nodes.cpu().numpy())
            except RuntimeError:
                boundary_set = set(self.boundary_nodes.cpu().tolist())
        else:
            boundary_set = set()
        
        # Check the changed node
        is_boundary = False
        for neighbor_idx in self.adjacency_list[changed_node]:
            neighbor_partition = self.current_partition[neighbor_idx].item()
            if neighbor_partition != new_partition:
                is_boundary = True
                break
                
        if is_boundary:
            boundary_set.add(changed_node)
        else:
            boundary_set.discard(changed_node)
            
        # Check all neighbors of the changed node
        for neighbor_idx in self.adjacency_list[changed_node]:
            neighbor_partition = self.current_partition[neighbor_idx].item()
            
            # Check if neighbor is now boundary
            neighbor_is_boundary = False
            for neighbor_neighbor_idx in self.adjacency_list[neighbor_idx]:
                neighbor_neighbor_partition = self.current_partition[neighbor_neighbor_idx].item()
                if neighbor_neighbor_partition != neighbor_partition:
                    neighbor_is_boundary = True
                    break
                    
            if neighbor_is_boundary:
                boundary_set.add(neighbor_idx)
            else:
                boundary_set.discard(neighbor_idx)
                
        self.boundary_nodes = torch.tensor(list(boundary_set), device=self.device)
        
    def _compute_region_embeddings(self):
        """Compute region-aggregated embeddings for all partitions"""
        num_partitions = self.current_partition.max().item()
        self.region_embeddings = {}
        
        for partition_id in range(1, num_partitions + 1):
            # Find nodes in this partition
            partition_mask = (self.current_partition == partition_id)
            partition_nodes = torch.where(partition_mask)[0]
            
            if len(partition_nodes) > 0:
                # Get embeddings for nodes in this partition
                partition_embeddings = self.node_embeddings[partition_nodes]
                
                # Compute mean and max pooling
                mean_embedding = torch.mean(partition_embeddings, dim=0)
                max_embedding = torch.max(partition_embeddings, dim=0)[0]
                
                # Concatenate mean and max
                region_embedding = torch.cat([mean_embedding, max_embedding], dim=0)
                self.region_embeddings[partition_id] = region_embedding
            else:
                # Empty partition - use zero embedding
                zero_embedding = torch.zeros(2 * self.embedding_dim, device=self.device)
                self.region_embeddings[partition_id] = zero_embedding
                
    def _update_region_embeddings_incremental(self, old_partition: int, new_partition: int):
        """Incrementally update region embeddings for affected partitions"""
        # For simplicity, recompute embeddings for affected partitions
        # This could be optimized further for very large graphs
        for partition_id in [old_partition, new_partition]:
            partition_mask = (self.current_partition == partition_id)
            partition_nodes = torch.where(partition_mask)[0]
            
            if len(partition_nodes) > 0:
                partition_embeddings = self.node_embeddings[partition_nodes]
                mean_embedding = torch.mean(partition_embeddings, dim=0)
                max_embedding = torch.max(partition_embeddings, dim=0)[0]
                region_embedding = torch.cat([mean_embedding, max_embedding], dim=0)
                self.region_embeddings[partition_id] = region_embedding
            else:
                zero_embedding = torch.zeros(2 * self.embedding_dim, device=self.device)
                self.region_embeddings[partition_id] = zero_embedding
                
    def get_observation(self) -> Dict[str, torch.Tensor]:
        """
        Get current state observation for the RL agent
        
        Returns:
            Dictionary containing state components
        """
        # Get region embeddings as a tensor
        num_partitions = len(self.region_embeddings)
        region_embedding_tensor = torch.stack([
            self.region_embeddings[i+1] for i in range(num_partitions)
        ], dim=0)
        
        # Boundary node features
        if len(self.boundary_nodes) > 0:
            boundary_features = self.node_embeddings[self.boundary_nodes]
        else:
            boundary_features = torch.empty(0, self.embedding_dim, device=self.device)
            
        observation = {
            'node_embeddings': self.node_embeddings,  # [total_nodes, embedding_dim]
            'region_embeddings': region_embedding_tensor,  # [num_partitions, 2*embedding_dim]
            'boundary_features': boundary_features,  # [num_boundary, embedding_dim]
            'current_partition': self.current_partition,  # [total_nodes]
            'boundary_nodes': self.boundary_nodes,  # [num_boundary]
        }
        
        return observation
        
    def get_boundary_nodes(self) -> torch.Tensor:
        """Get current boundary nodes"""
        return self.boundary_nodes if self.boundary_nodes is not None else torch.empty(0, dtype=torch.long, device=self.device)
        
    def get_global_node_mapping(self) -> Dict[str, torch.Tensor]:
        """Get mapping from node types to global indices"""
        return self.local_to_global_map
        
    def get_partition_info(self) -> Dict[str, torch.Tensor]:
        """Get detailed partition information"""
        num_partitions = self.current_partition.max().item()
        partition_sizes = torch.bincount(self.current_partition, minlength=num_partitions + 1)[1:]
        
        return {
            'partition_assignments': self.current_partition,
            'partition_sizes': partition_sizes,
            'num_partitions': num_partitions,
            'boundary_nodes': self.get_boundary_nodes()
        }
