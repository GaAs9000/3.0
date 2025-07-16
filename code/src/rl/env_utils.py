#!/usr/bin/env python3
"""
Utility functions for the PowerGridPartitioningEnv, supporting v3.0 features.

This module isolates complex calculations, such as partition feature extraction,
to keep the main environment file cleaner and more focused on MDP logic.
"""
import torch
from typing import List, Dict, Any
from torch_geometric.utils import subgraph

# Assuming PartitionFeatures is in the models directory
from ..models.partition_encoder import PartitionFeatures
# Assuming StateManager is in the same directory
from .state import StateManager


def calculate_single_partition_features(
    partition_nodes: torch.Tensor,
    hetero_data: Any,
    total_nodes: int,
    total_load: float,
    total_generation: float,
    global_edge_index: torch.Tensor,
    state_manager: StateManager
) -> PartitionFeatures:
    """
    Calculates the standardized physical and topological features for a single partition.

    Args:
        partition_nodes: Tensor containing the global indices of nodes in the partition.
        hetero_data: The HeteroData object for the entire grid.
        total_nodes: Total number of nodes in the grid.
        total_load: Total load of the grid.
        total_generation: Total generation of the grid.
        global_edge_index: Edge index for the entire grid with global node indices.
        state_manager: The state manager instance for utility functions like global-to-local mapping.

    Returns:
        A PartitionFeatures dataclass instance.
    """
    num_nodes = len(partition_nodes)
    if num_nodes == 0:
        return PartitionFeatures(size_ratio=0, load_ratio=0, generation_ratio=0, internal_connectivity=0, boundary_ratio=0, power_imbalance=0)

    # --- Feature 1: size_ratio ---
    size_ratio = num_nodes / total_nodes if total_nodes > 0 else 0

    # --- Power-related features (load, generation, imbalance) ---
    part_load = 0.0
    part_generation = 0.0
    
    if 'bus' in hetero_data.x_dict:
        # Use state_manager to get local indices for 'bus' type nodes
        local_bus_indices, _ = state_manager.global_to_local_mask(partition_nodes, 'bus')
        if local_bus_indices.numel() > 0:
            bus_features = hetero_data.x_dict['bus'][local_bus_indices]
            if bus_features.shape[1] >= 2:
                part_load = bus_features[:, 0].sum().item()
                part_generation = bus_features[:, 1].sum().item()

    load_ratio = part_load / total_load if total_load > 0 else 0
    generation_ratio = part_generation / total_generation if total_generation > 0 else 0
    power_imbalance = (part_generation - part_load) / part_load if part_load > 0 else 0

    # --- Topology-related features (internal connectivity, boundary ratio) ---
    # Use torch_geometric's subgraph utility to find internal edges
    sub_edge_index, _ = subgraph(partition_nodes, global_edge_index, relabel_nodes=False)
    internal_edges = sub_edge_index.size(1)

    max_possible_edges = num_nodes * (num_nodes - 1) / 2
    internal_connectivity = internal_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    # Find boundary nodes within this partition
    src, dst = global_edge_index
    is_src_in = torch.isin(src, partition_nodes)
    is_dst_in = torch.isin(dst, partition_nodes)
    # An edge is a boundary edge if one end is in the partition and the other is not
    boundary_edges_mask = is_src_in ^ is_dst_in
    
    # Collect the nodes of this partition that are part of a boundary edge
    boundary_nodes_in_partition = torch.unique(torch.cat([
        src[boundary_edges_mask & is_src_in],
        dst[boundary_edges_mask & is_dst_in]
    ]))
    
    num_boundary_nodes = len(boundary_nodes_in_partition)
    boundary_ratio = num_boundary_nodes / num_nodes if num_nodes > 0 else 0

    return PartitionFeatures(
        size_ratio=size_ratio,
        load_ratio=load_ratio,
        generation_ratio=generation_ratio,
        internal_connectivity=internal_connectivity,
        boundary_ratio=boundary_ratio,
        power_imbalance=power_imbalance,
    )


def get_connectivity_safe_partitions(node_id: int, state_manager: StateManager) -> List[int]:
    """
    获取节点可以安全移动到的分区（保证连通性）
    
    使用图遍历验证移动后各分区的连通性，确保分区质量。

    Args:
        node_id: 要移动的节点全局ID
        state_manager: 状态管理器实例

    Returns:
        有效的目标分区ID列表
    """
    import networkx as nx
    
    current_partition_id = state_manager.current_partition[node_id].item()
    neighbor_nodes = state_manager.get_neighbors(node_id)
    
    if neighbor_nodes.numel() == 0:
        return []
    
    # 获取所有邻居的分区
    neighbor_partitions = torch.unique(state_manager.current_partition[neighbor_nodes])
    candidate_partitions = [
        p.item() for p in neighbor_partitions 
        if p.item() != current_partition_id and p.item() != 0
    ]
    
    # 构建图结构进行连通性验证
    try:
        # 获取图的边信息
        edge_index = state_manager.hetero_data['bus', 'connects', 'bus'].edge_index
        num_nodes = state_manager.hetero_data['bus'].x.size(0)
        
        # 创建NetworkX图
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        
        valid_partitions = []
        
        for target_partition in candidate_partitions:
            # 模拟移动节点
            temp_partition = state_manager.current_partition.clone()
            temp_partition[node_id] = target_partition
            
            # 验证源分区和目标分区的连通性
            is_valid = True
            
            # 检查源分区连通性（移除节点后）
            source_nodes = (temp_partition == current_partition_id).nonzero(dim=0).flatten()
            if len(source_nodes) > 0:
                source_subgraph = G.subgraph(source_nodes.cpu().numpy())
                if not nx.is_connected(source_subgraph):
                    is_valid = False
            
            # 检查目标分区连通性（添加节点后）
            if is_valid:
                target_nodes = (temp_partition == target_partition).nonzero(dim=0).flatten()
                if len(target_nodes) > 0:
                    target_subgraph = G.subgraph(target_nodes.cpu().numpy())
                    if not nx.is_connected(target_subgraph):
                        is_valid = False
            
            if is_valid:
                valid_partitions.append(target_partition)
        
        return valid_partitions
        
    except Exception as e:
        logger.warning(f"连通性检查失败，回退到基础验证: {e}")
        return candidate_partitions 