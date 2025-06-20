"""
Utility functions for Power Grid Partitioning RL

This module provides utility functions including:
- METIS initialization for initial partitioning
- Partition quality evaluation
- State management helpers
- Visualization tools
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import HeteroData
import warnings

try:
    import metis
    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    warnings.warn("METIS not available. Using fallback spectral clustering for initialization.")

try:
    from sklearn.cluster import SpectralClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Using random initialization as fallback.")


class MetisInitializer:
    """
    METIS-based initialization for power grid partitioning
    
    Provides initial partition using METIS graph partitioning algorithm
    with node weights based on power loads.
    """
    
    def __init__(self, hetero_data: HeteroData, device: torch.device):
        """
        Initialize METIS partitioner
        
        Args:
            hetero_data: Heterogeneous graph data
            device: Torch device for computations
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        
        # Setup graph representation for METIS
        self._setup_graph_representation()
        
    def _setup_graph_representation(self):
        """Setup graph representation compatible with METIS"""
        # Get total number of nodes
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # Setup node mappings
        self._setup_node_mappings()
        
        # Extract node weights (power loads)
        self._extract_node_weights()
        
        # Build adjacency list for METIS
        self._build_adjacency_list()
        
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
        
    def _extract_node_weights(self):
        """Extract node weights based on power loads"""
        # Concatenate all node features
        all_features = []
        for node_type in self.node_types:
            features = self.hetero_data.x_dict[node_type]
            all_features.append(features)
            
        all_node_features = torch.cat(all_features, dim=0)
        
        # Extract load data (assuming Pd is at index 0)
        if all_node_features.shape[1] > 0:
            loads = all_node_features[:, 0]  # Pd column
            # Convert to positive integers for METIS (scale and add offset)
            loads_scaled = (loads * 1000 + 1000).clamp(min=1).int()
            try:
                self.node_weights = loads_scaled.cpu().numpy()
            except RuntimeError:
                # Fallback if numpy conversion fails
                self.node_weights = [int(x) for x in loads_scaled.cpu().tolist()]
        else:
            # Uniform weights if no load data
            self.node_weights = [1] * self.total_nodes
            
    def _build_adjacency_list(self):
        """Build adjacency list for METIS"""
        # Initialize adjacency list
        self.adjacency_list = [[] for _ in range(self.total_nodes)]
        
        # Add edges from heterogeneous graph
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            
            # Convert to global indices
            src_global = self._local_to_global(edge_index[0], src_type)
            dst_global = self._local_to_global(edge_index[1], dst_type)
            
            # Add to adjacency list
            try:
                src_list = src_global.cpu().numpy()
                dst_list = dst_global.cpu().numpy()
            except RuntimeError:
                src_list = src_global.cpu().tolist()
                dst_list = dst_global.cpu().tolist()

            for src, dst in zip(src_list, dst_list):
                self.adjacency_list[src].append(dst)
                self.adjacency_list[dst].append(src)  # Undirected graph
                
        # Remove duplicates and self-loops
        for i in range(self.total_nodes):
            self.adjacency_list[i] = list(set(self.adjacency_list[i]))
            if i in self.adjacency_list[i]:
                self.adjacency_list[i].remove(i)
                
    def initialize_partition(self, num_partitions: int) -> torch.Tensor:
        """
        Initialize partition using METIS or fallback methods
        
        Args:
            num_partitions: Number of target partitions
            
        Returns:
            Initial partition assignments [total_nodes]
        """
        if METIS_AVAILABLE and self.total_nodes > num_partitions:
            try:
                return self._metis_partition(num_partitions)
            except Exception as e:
                warnings.warn(f"METIS partitioning failed: {e}. Using fallback method.")
                
        if SKLEARN_AVAILABLE:
            return self._spectral_partition(num_partitions)
        else:
            return self._random_partition(num_partitions)
            
    def _metis_partition(self, num_partitions: int) -> torch.Tensor:
        """Partition using METIS algorithm"""
        # Convert adjacency list to METIS format
        xadj = [0]
        adjncy = []
        
        for neighbors in self.adjacency_list:
            adjncy.extend(neighbors)
            xadj.append(len(adjncy))
            
        # Run METIS partitioning
        if len(adjncy) == 0:
            # No edges - use random partition
            return self._random_partition(num_partitions)
            
        try:
            vwgt = self.node_weights if isinstance(self.node_weights, list) else self.node_weights.tolist()
            _, partition = metis.part_graph(
                nparts=num_partitions,
                xadj=xadj,
                adjncy=adjncy,
                vwgt=vwgt,
                recursive=True
            )
            
            # Convert to 1-based indexing and torch tensor
            partition_tensor = torch.tensor(partition, device=self.device) + 1
            return partition_tensor
            
        except Exception as e:
            warnings.warn(f"METIS failed: {e}. Using spectral clustering fallback.")
            return self._spectral_partition(num_partitions)
            
    def _spectral_partition(self, num_partitions: int) -> torch.Tensor:
        """Partition using spectral clustering"""
        # Build adjacency matrix
        adj_matrix = np.zeros((self.total_nodes, self.total_nodes))
        
        for i, neighbors in enumerate(self.adjacency_list):
            for j in neighbors:
                adj_matrix[i, j] = 1.0
                
        # Handle edge cases
        if np.sum(adj_matrix) == 0:
            return self._random_partition(num_partitions)
            
        try:
            clustering = SpectralClustering(
                n_clusters=num_partitions,
                affinity='precomputed',
                random_state=42
            )
            
            partition = clustering.fit_predict(adj_matrix)
            
            # Convert to 1-based indexing and torch tensor
            partition_tensor = torch.tensor(partition, device=self.device) + 1
            return partition_tensor
            
        except Exception as e:
            warnings.warn(f"Spectral clustering failed: {e}. Using random partition.")
            return self._random_partition(num_partitions)
            
    def _random_partition(self, num_partitions: int) -> torch.Tensor:
        """Random partition as final fallback"""
        partition = torch.randint(
            1, num_partitions + 1, 
            (self.total_nodes,), 
            device=self.device
        )
        return partition


class PartitionEvaluator:
    """
    Comprehensive partition quality evaluation
    
    Provides various metrics for evaluating partition quality including
    electrical, topological, and load balance metrics.
    """
    
    def __init__(self, hetero_data: HeteroData, device: torch.device):
        """
        Initialize Partition Evaluator
        
        Args:
            hetero_data: Heterogeneous graph data
            device: Torch device for computations
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        
        # Setup evaluation data
        self._setup_evaluation_data()
        
    def _setup_evaluation_data(self):
        """Setup data needed for evaluation"""
        # Setup node mappings
        self._setup_node_mappings()
        
        # Extract power data
        self._extract_power_data()
        
        # Extract electrical data
        self._extract_electrical_data()
        
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
        
    def _extract_power_data(self):
        """Extract power data from node features"""
        all_features = []
        for node_type in self.node_types:
            features = self.hetero_data.x_dict[node_type]
            all_features.append(features)
            
        self.all_node_features = torch.cat(all_features, dim=0)
        
        # Extract power data
        self.load_active = self.all_node_features[:, 0]  # Pd
        if self.all_node_features.shape[1] > 9:
            self.gen_active = self.all_node_features[:, 9]  # Pg
        else:
            self.gen_active = torch.zeros_like(self.load_active)
            
    def _extract_electrical_data(self):
        """Extract electrical data from edges"""
        self.all_edges = []
        self.all_admittances = []
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            
            # Convert to global indices
            src_global = self._local_to_global(edge_index[0], src_type)
            dst_global = self._local_to_global(edge_index[1], dst_type)
            
            global_edges = torch.stack([src_global, dst_global], dim=0)
            self.all_edges.append(global_edges)
            
            # Extract admittance
            if edge_attr.shape[1] > 4:
                admittances = edge_attr[:, 4]  # y column
            else:
                admittances = torch.ones(edge_attr.shape[0], device=self.device)
                
            self.all_admittances.append(admittances)
            
        if self.all_edges:
            self.edge_index = torch.cat(self.all_edges, dim=1)
            self.edge_admittances = torch.cat(self.all_admittances, dim=0)
        else:
            self.edge_index = torch.empty(2, 0, device=self.device)
            self.edge_admittances = torch.empty(0, device=self.device)
            
    def evaluate_partition(self, partition: torch.Tensor) -> Dict[str, float]:
        """
        Comprehensive partition evaluation
        
        Args:
            partition: Partition assignments [total_nodes]
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Basic partition info
        num_partitions = partition.max().item()
        partition_sizes = torch.bincount(partition, minlength=num_partitions + 1)[1:]
        
        # Load balance metrics
        partition_loads = torch.zeros(num_partitions, device=self.device)
        for i in range(1, num_partitions + 1):
            mask = (partition == i)
            if mask.any():
                partition_loads[i-1] = self.load_active[mask].sum()
                
        load_mean = torch.mean(partition_loads)
        load_std = torch.std(partition_loads)
        load_cv = (load_std / load_mean).item() if load_mean > 0 else 0.0
        
        # Coupling metrics
        if self.edge_index.shape[1] > 0:
            src_partitions = partition[self.edge_index[0]]
            dst_partitions = partition[self.edge_index[1]]
            coupling_mask = (src_partitions != dst_partitions)
            
            coupling_edges = coupling_mask.sum().item()
            total_coupling = self.edge_admittances[coupling_mask].sum().item() if coupling_mask.any() else 0.0
        else:
            coupling_edges = 0
            total_coupling = 0.0
            
        # Power balance metrics
        power_imbalances = []
        for i in range(1, num_partitions + 1):
            mask = (partition == i)
            if mask.any():
                gen = self.gen_active[mask].sum()
                load = self.load_active[mask].sum()
                imbalance = abs(gen - load).item()
                power_imbalances.append(imbalance)
                
        # Connectivity check (simplified)
        connectivity = self._check_connectivity(partition)
        
        try:
            partition_sizes_list = partition_sizes.cpu().numpy().tolist()
        except RuntimeError:
            partition_sizes_list = partition_sizes.cpu().tolist()

        metrics.update({
            'num_partitions': num_partitions,
            'partition_sizes': partition_sizes_list,
            'load_cv': load_cv,
            'load_variance': torch.var(partition_loads).item(),
            'coupling_edges': coupling_edges,
            'total_coupling': total_coupling,
            'power_imbalance_mean': np.mean(power_imbalances) if power_imbalances else 0.0,
            'power_imbalance_max': np.max(power_imbalances) if power_imbalances else 0.0,
            'connectivity': connectivity
        })
        
        return metrics
        
    def _check_connectivity(self, partition: torch.Tensor) -> float:
        """
        Check partition connectivity (simplified version)
        
        Args:
            partition: Partition assignments
            
        Returns:
            Connectivity score (1.0 if all partitions connected, lower otherwise)
        """
        # For now, return 1.0 (assume connected)
        # A full implementation would check graph connectivity within each partition
        return 1.0
