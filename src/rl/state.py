"""
电力网络分区MDP的状态管理

本模块管理MDP公式中指定的状态表示：
- 节点特征嵌入（H或H'）：静态的，由GAT编码器预计算，可能包含注意力增强信息
- 节点分配标签（z_t）：动态的，随每个动作变化
- 边界节点：从当前分区派生
- 区域聚合嵌入：用于策略输入
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from torch_geometric.data import HeteroData


class StateManager:
    """
    管理电力网络分区的MDP状态表示
    
    状态组件：
    1. 节点特征嵌入（H'）- 来自GAT编码器的静态矩阵，可能包含注意力增强信息
       - 如果提供了注意力权重：H' = concat(H, H_attn)
       - 如果没有注意力权重：H' = H（原始嵌入）
    2. 节点分配标签（z_t）- 动态分区分配
    3. 边界节点（Bdry_t）- 与不同分区中邻居相连的节点
    4. 区域聚合嵌入 - 每个区域的均值/最大池化嵌入
    """
    
    def __init__(self, 
                 hetero_data: HeteroData,
                 node_embeddings: Dict[str, torch.Tensor],
                 device: torch.device):
        """
        初始化状态管理器
        
        Args:
            hetero_data: 异构图数据
            node_embeddings: 来自GAT编码器的预计算节点嵌入
                           - 可能是原始嵌入H，也可能是增强嵌入H' = concat(H, H_attn)
                           - 具体取决于Environment是否提供了注意力权重
            device: 计算设备
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        
        # 设置节点映射和嵌入
        self._setup_node_mappings()
        self._setup_node_embeddings(node_embeddings)
        self._setup_adjacency_info()
        
        # 状态变量
        self.current_partition = None
        self.boundary_nodes = None
        self.region_embeddings = None
        
    def _setup_node_mappings(self):
        """设置局部和全局节点索引之间的映射"""
        self.node_types = list(self.hetero_data.x_dict.keys())
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # 创建从局部索引到全局索引的映射
        self.local_to_global_map = {}
        self.global_to_local_map = {}
        
        global_idx = 0
        for node_type in self.node_types:
            num_nodes = self.hetero_data.x_dict[node_type].shape[0]
            
            # 该类型的局部到全局映射
            local_indices = torch.arange(num_nodes, device=self.device)
            global_indices = torch.arange(global_idx, global_idx + num_nodes, device=self.device)
            
            self.local_to_global_map[node_type] = global_indices
            
            # 全局到局部映射
            for local_idx, global_idx_val in zip(local_indices, global_indices):
                self.global_to_local_map[global_idx_val.item()] = (node_type, local_idx.item())
                
            global_idx += num_nodes
            
    def _setup_node_embeddings(self, node_embeddings: Dict[str, torch.Tensor]):
        """
        设置连接的节点嵌入矩阵H'
        
        这里的node_embeddings可能是：
        1. 原始GAT嵌入H（如果Environment没有提供注意力权重）
        2. 增强嵌入H' = concat(H, H_attn)（如果Environment提供了注意力权重）
        
        无论哪种情况，我们都将其作为完整的节点特征矩阵使用
        """
        # 将所有节点嵌入连接成单个矩阵
        embedding_list = []
        for node_type in self.node_types:
            embeddings = node_embeddings[node_type].to(self.device)
            embedding_list.append(embeddings)
            
        self.node_embeddings = torch.cat(embedding_list, dim=0)  # 形状：[total_nodes, embedding_dim]
        self.embedding_dim = self.node_embeddings.shape[1]
        
        # 记录嵌入维度信息（用于调试）
        print(f"🔧 StateManager: 设置节点嵌入矩阵，形状 {self.node_embeddings.shape}")
        print(f"   - 总节点数: {self.total_nodes}")
        print(f"   - 嵌入维度: {self.embedding_dim}")
        print(f"   - 注意：此嵌入可能包含GAT原始嵌入 + 注意力增强信息")
        
    def _setup_adjacency_info(self):
        """设置用于边界节点计算的邻接信息"""
        # 创建全局邻接列表
        self.adjacency_list = [[] for _ in range(self.total_nodes)]
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            
            # 转换为全局索引
            src_global = self.local_to_global(edge_index[0], src_type)
            dst_global = self.local_to_global(edge_index[1], dst_type)
            
            # 添加到邻接列表
            for src, dst in zip(src_global, dst_global):
                self.adjacency_list[src.item()].append(dst.item())
                
    def local_to_global(self, local_indices: torch.Tensor, node_type: str) -> torch.Tensor:
        """将给定节点类型的局部索引转换为全局索引"""
        return self.local_to_global_map[node_type][local_indices]
        
    def global_to_local(self, global_idx: int) -> Tuple[str, int]:
        """将全局索引转换为(node_type, local_index)"""
        return self.global_to_local_map[global_idx]
        
    def reset(self, initial_partition: torch.Tensor):
        """
        使用初始分区重置状态
        
        Args:
            initial_partition: 初始分区分配 [total_nodes]
        """
        self.current_partition = initial_partition.to(self.device)
        self._update_derived_state()
        
    def update_partition(self, node_idx: int, new_partition: int):
        """
        更新单个节点的分区分配
        
        Args:
            node_idx: 全局节点索引
            new_partition: 新的分区分配
        """
        old_partition = self.current_partition[node_idx].item()
        self.current_partition[node_idx] = new_partition
        
        # 高效更新边界节点
        self._update_boundary_nodes_incremental(node_idx, old_partition, new_partition)
        
        # 更新受影响分区的区域嵌入
        self._update_region_embeddings_incremental(old_partition, new_partition)
        
    def _update_derived_state(self):
        """更新所有派生状态组件"""
        self._compute_boundary_nodes()
        self._compute_region_embeddings()
        
    def _compute_boundary_nodes(self):
        """从当前分区计算边界节点"""
        boundary_set = set()
        
        for node_idx in range(self.total_nodes):
            node_partition = self.current_partition[node_idx].item()
            
            # 检查是否有邻居在不同分区中
            for neighbor_idx in self.adjacency_list[node_idx]:
                neighbor_partition = self.current_partition[neighbor_idx].item()
                if neighbor_partition != node_partition:
                    boundary_set.add(node_idx)
                    break
                    
        self.boundary_nodes = torch.tensor(list(boundary_set), device=self.device)
        
    def _update_boundary_nodes_incremental(self, changed_node: int, old_partition: int, new_partition: int):
        """单个节点变化后增量更新边界节点"""
        # 转换为集合以进行高效操作
        if self.boundary_nodes is not None:
            try:
                boundary_set = set(self.boundary_nodes.cpu().numpy())
            except RuntimeError:
                boundary_set = set(self.boundary_nodes.cpu().tolist())
        else:
            boundary_set = set()
        
        # 检查变化的节点
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
            
        # 检查变化节点的所有邻居
        for neighbor_idx in self.adjacency_list[changed_node]:
            neighbor_partition = self.current_partition[neighbor_idx].item()
            
            # 检查邻居是否现在是边界节点
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
        """计算所有分区的区域聚合嵌入"""
        num_partitions = self.current_partition.max().item()
        self.region_embeddings = {}
        
        for partition_id in range(1, num_partitions + 1):
            # 找到该分区中的节点
            partition_mask = (self.current_partition == partition_id)
            partition_nodes = torch.where(partition_mask)[0]
            
            if len(partition_nodes) > 0:
                # 获取该分区中节点的嵌入
                partition_embeddings = self.node_embeddings[partition_nodes]
                
                # 计算均值和最大池化
                mean_embedding = torch.mean(partition_embeddings, dim=0)
                max_embedding = torch.max(partition_embeddings, dim=0)[0]
                
                # 连接均值和最大值
                region_embedding = torch.cat([mean_embedding, max_embedding], dim=0)
                self.region_embeddings[partition_id] = region_embedding
            else:
                # 空分区 - 使用零嵌入
                zero_embedding = torch.zeros(2 * self.embedding_dim, device=self.device)
                self.region_embeddings[partition_id] = zero_embedding
                
    def _update_region_embeddings_incremental(self, old_partition: int, new_partition: int):
        """增量更新受影响分区的区域嵌入"""
        # 为简单起见，重新计算受影响分区的嵌入
        # 对于非常大的图，这可以进一步优化
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
        获取RL智能体的当前状态观察
        
        Returns:
            包含状态组件的字典
        """
        # 将区域嵌入作为张量
        num_partitions = len(self.region_embeddings)
        region_embedding_tensor = torch.stack([
            self.region_embeddings[i+1] for i in range(num_partitions)
        ], dim=0)
        
        # 边界节点特征
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
        """获取当前边界节点"""
        return self.boundary_nodes if self.boundary_nodes is not None else torch.empty(0, dtype=torch.long, device=self.device)
        
    def get_global_node_mapping(self) -> Dict[str, torch.Tensor]:
        """获取从节点类型到全局索引的映射"""
        return self.local_to_global_map
        
    def get_partition_info(self) -> Dict[str, torch.Tensor]:
        """获取详细的分区信息"""
        num_partitions = self.current_partition.max().item()
        partition_sizes = torch.bincount(self.current_partition, minlength=num_partitions + 1)[1:]
        
        return {
            'partition_assignments': self.current_partition,
            'partition_sizes': partition_sizes,
            'num_partitions': num_partitions,
            'boundary_nodes': self.get_boundary_nodes()
        }
