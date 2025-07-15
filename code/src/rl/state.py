"""
电力网络分区MDP的状态管理

本模块管理MDP公式中指定的状态表示：
- 节点特征嵌入（H或H'）：静态的，由GAT编码器预计算，可能包含注意力增强信息
- 节点分配标签（z_t）：动态的，随每个动作变化
- 边界节点：从当前分区派生
- 区域聚合嵌入：用于策略输入
"""

import torch
import torch.nn as nn
import numpy as np
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from torch_geometric.data import HeteroData
import torch_sparse


class IntelligentRegionEmbedding:
    """
    智能区域嵌入生成器，优雅处理空分区
    """

    def __init__(self, embedding_dim, device, config=None):
        self.embedding_dim = embedding_dim
        self.device = device
        self.config = config or {}

        # 学习的空分区表示（可训练参数）
        self.empty_partition_embedding = nn.Parameter(
            torch.randn(2 * embedding_dim, device=device) * 0.1
        )

        # 分区历史信息
        self.partition_history = {}
        self.current_step = 0

        # 配置参数
        self.use_learned_empty_embedding = config.get('use_learned_empty_embedding', True)
        self.use_positional_encoding = config.get('use_positional_encoding', True)
        self.use_historical_decay = config.get('use_historical_decay', True)

    def compute_region_embedding(self, partition_id, partition_nodes, node_embeddings):
        """
        计算区域嵌入，智能处理空分区
        """
        if len(partition_nodes) > 0:
            # 非空分区：正常计算
            partition_embeddings = node_embeddings[partition_nodes]

            # 多种聚合方式
            mean_embedding = torch.mean(partition_embeddings, dim=0)
            max_embedding = torch.max(partition_embeddings, dim=0)[0]

            # 可选：添加更多统计信息
            if self.config.get('use_advanced_pooling', False):
                min_embedding = torch.min(partition_embeddings, dim=0)[0]
                std_embedding = torch.std(partition_embeddings, dim=0)

                region_embedding = torch.cat([
                    mean_embedding, max_embedding,
                    min_embedding, std_embedding
                ], dim=0)
            else:
                region_embedding = torch.cat([mean_embedding, max_embedding], dim=0)

            # 更新历史
            self._update_history(partition_id, region_embedding, 'active')

            return region_embedding

        else:
            # 空分区：智能处理
            return self._handle_empty_partition(partition_id)

    def _handle_empty_partition(self, partition_id):
        """
        智能处理空分区
        """
        # 策略1: 使用学习的空分区嵌入
        if self.use_learned_empty_embedding:
            base_embedding = self.empty_partition_embedding
        else:
            base_embedding = torch.zeros(2 * self.embedding_dim, device=self.device)

        # 策略2: 基于历史信息调整
        if self.use_historical_decay and partition_id in self.partition_history:
            history = self.partition_history[partition_id]

            # 如果分区最近是活跃的，使用衰减的历史嵌入
            if history['last_active_step'] is not None:
                steps_since_active = self.current_step - history['last_active_step']
                decay_factor = np.exp(-steps_since_active / 10.0)  # 指数衰减

                if decay_factor > 0.1 and history['last_embedding'] is not None:
                    # 混合历史嵌入和空嵌入
                    embedding = (decay_factor * history['last_embedding'] +
                               (1 - decay_factor) * base_embedding)

                    self._update_history(partition_id, embedding, 'decayed')
                    return embedding

        # 策略3: 添加位置编码信息
        if self.use_positional_encoding:
            position_encoding = self._get_partition_position_encoding(partition_id)
            embedding = base_embedding + 0.1 * position_encoding
        else:
            embedding = base_embedding

        self._update_history(partition_id, embedding, 'empty')
        return embedding

    def _get_partition_position_encoding(self, partition_id):
        """
        为分区ID生成位置编码
        """
        # 简单的正弦位置编码
        pe = torch.zeros(2 * self.embedding_dim, device=self.device)

        position = partition_id
        div_term = torch.exp(
            torch.arange(0, 2 * self.embedding_dim, 2).float() *
            (-np.log(10000.0) / (2 * self.embedding_dim))
        ).to(self.device)

        pe[0::2] = torch.sin(position * div_term)
        pe[1::2] = torch.cos(position * div_term)

        return pe

    def _update_history(self, partition_id, embedding, status):
        """
        更新分区历史
        """
        if partition_id not in self.partition_history:
            self.partition_history[partition_id] = {
                'last_embedding': None,
                'last_active_step': None,
                'status_history': []
            }

        history = self.partition_history[partition_id]

        if status == 'active':
            history['last_embedding'] = embedding.detach()
            history['last_active_step'] = self.current_step

        history['status_history'].append({
            'step': self.current_step,
            'status': status
        })

        # 限制历史长度
        if len(history['status_history']) > 100:
            history['status_history'] = history['status_history'][-100:]

    def update_step(self, step):
        """更新当前步数"""
        self.current_step = step

    def get_embedding_statistics(self):
        """获取嵌入统计信息"""
        stats = {
            'total_partitions': len(self.partition_history),
            'active_partitions': 0,
            'empty_partitions': 0,
            'decayed_partitions': 0
        }

        for history in self.partition_history.values():
            if history['status_history']:
                last_status = history['status_history'][-1]['status']
                if last_status == 'active':
                    stats['active_partitions'] += 1
                elif last_status == 'empty':
                    stats['empty_partitions'] += 1
                elif last_status == 'decayed':
                    stats['decayed_partitions'] += 1

        return stats


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
                 device: torch.device,
                 config: Dict[str, Any] = None):
        """
        初始化状态管理器

        Args:
            hetero_data: 异构图数据
            node_embeddings: 来自GAT编码器的预计算节点嵌入
                           - 可能是原始嵌入H，也可能是增强嵌入H' = concat(H, H_attn)
                           - 具体取决于Environment是否提供了注意力权重
            device: 计算设备
            config: 配置字典，用于控制输出详细程度
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        self.config = config

        # 获取调试配置
        debug_config = config.get('debug', {}) if config else {}
        self.training_output = debug_config.get('training_output', {})

        # 设置节点映射和嵌入
        self._setup_node_mappings()
        self._setup_node_embeddings(node_embeddings)
        self._setup_adjacency_info()

        # 创建智能区域嵌入生成器
        self.region_embedder = IntelligentRegionEmbedding(
            self.embedding_dim,
            self.device,
            config=config.get('region_embedding', {
                'use_learned_empty_embedding': True,
                'use_positional_encoding': True,
                'use_historical_decay': True,
                'use_advanced_pooling': False
            }) if config else {}
        )

        # 状态变量
        self.current_partition = None
        self.boundary_nodes = None
        self.region_embeddings = None
        self.current_step = 0
        
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
        show_state_manager_details = self.training_output.get('show_state_manager_details', True)
        only_show_errors = self.training_output.get('only_show_errors', False)

        if show_state_manager_details and not only_show_errors:
            print(f"StateManager: 设置节点嵌入矩阵，形状 {self.node_embeddings.shape}")
            print(f"   - 总节点数: {self.total_nodes}")
            print(f"   - 嵌入维度: {self.embedding_dim}")
            print(f"   - 注意：此嵌入可能包含GAT原始嵌入 + 注意力增强信息")
        
    def _setup_adjacency_info(self):
        """设置用于边界节点计算的邻接信息 - 使用稀疏张量优化"""
        # 收集所有边用于构建稀疏张量
        all_edges = []
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            
            # 转换为全局索引
            src_global = self.local_to_global(edge_index[0], src_type)
            dst_global = self.local_to_global(edge_index[1], dst_type)
            
            # 添加边（无向图需要双向边）
            edges = torch.stack([src_global, dst_global], dim=0)
            all_edges.append(edges)
            # 添加反向边
            all_edges.append(torch.stack([dst_global, src_global], dim=0))
        
        # 合并所有边
        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
            # 去重
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        
        # 创建稀疏张量
        edge_attr = torch.ones(edge_index.shape[1], device=self.device)
        self.adjacency_sparse = torch_sparse.SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_attr,
            sparse_sizes=(self.total_nodes, self.total_nodes)
        ).to(self.device).requires_grad_(False)
        
        # 缓存边索引用于向量化边界节点计算
        self.edge_index = edge_index
        
        # 预计算每个节点的邻居索引（用于增量更新）
        self._precompute_neighbors()
    
    def _precompute_neighbors(self):
        """预计算每个节点的邻居，用于快速查询"""
        self.neighbors_cache = {}
        
        # 从稀疏张量中提取每个节点的邻居
        row_ptr, col, _ = self.adjacency_sparse.csr()
        for node_idx in range(self.total_nodes):
            start_idx = row_ptr[node_idx]
            end_idx = row_ptr[node_idx + 1]
            neighbors = col[start_idx:end_idx]
            self.neighbors_cache[node_idx] = neighbors
        
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
        """从当前分区计算边界节点 - 向量化实现"""
        if hasattr(self, 'edge_index') and self.edge_index.shape[1] > 0:
            # 使用向量化方法
            src, dst = self.edge_index[0], self.edge_index[1]
            
            # 检查哪些边连接不同分区的节点
            diff_mask = self.current_partition[src] != self.current_partition[dst]
            
            # 找到所有涉及跨区连接的节点
            boundary_nodes = torch.unique(
                torch.cat([src[diff_mask], dst[diff_mask]])
            )
            
            self.boundary_nodes = boundary_nodes
        else:
            # 如果没有边，就没有边界节点
            self.boundary_nodes = torch.empty(0, dtype=torch.long, device=self.device)
        
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
        if changed_node in self.neighbors_cache:
            neighbors = self.neighbors_cache[changed_node]
            for neighbor_idx in neighbors:
                neighbor_partition = self.current_partition[neighbor_idx].item()
                if neighbor_partition != new_partition:
                    is_boundary = True
                    break
                
        if is_boundary:
            boundary_set.add(changed_node)
        else:
            boundary_set.discard(changed_node)
            
        # 检查变化节点的所有邻居
        if changed_node in self.neighbors_cache:
            neighbors = self.neighbors_cache[changed_node]
            for neighbor_idx in neighbors:
                neighbor_idx_int = neighbor_idx.item()
                neighbor_partition = self.current_partition[neighbor_idx_int].item()
                
                # 检查邻居是否现在是边界节点
                neighbor_is_boundary = False
                if neighbor_idx_int in self.neighbors_cache:
                    neighbor_neighbors = self.neighbors_cache[neighbor_idx_int]
                    for neighbor_neighbor_idx in neighbor_neighbors:
                        neighbor_neighbor_partition = self.current_partition[neighbor_neighbor_idx].item()
                        if neighbor_neighbor_partition != neighbor_partition:
                            neighbor_is_boundary = True
                            break
                        
                if neighbor_is_boundary:
                    boundary_set.add(neighbor_idx_int)
                else:
                    boundary_set.discard(neighbor_idx_int)
                
        self.boundary_nodes = torch.tensor(list(boundary_set), dtype=torch.long, device=self.device)
        
    def _compute_region_embeddings(self):
        """计算所有分区的区域聚合嵌入 - 使用智能嵌入生成器"""
        num_partitions = self.current_partition.max().item()
        self.region_embeddings = {}

        # 更新嵌入生成器的步数
        self.region_embedder.update_step(self.current_step)

        for partition_id in range(1, num_partitions + 1):
            # 找到该分区中的节点
            partition_mask = (self.current_partition == partition_id)
            partition_nodes = torch.where(partition_mask)[0]

            # 使用智能嵌入生成器
            self.region_embeddings[partition_id] = self.region_embedder.compute_region_embedding(
                partition_id, partition_nodes, self.node_embeddings
            )
                
    def _update_region_embeddings_incremental(self, old_partition: int, new_partition: int):
        """增量更新受影响分区的区域嵌入 - 使用智能嵌入生成器"""
        # 更新嵌入生成器的步数
        self.region_embedder.update_step(self.current_step)

        for partition_id in [old_partition, new_partition]:
            partition_mask = (self.current_partition == partition_id)
            partition_nodes = torch.where(partition_mask)[0]

            # 使用智能嵌入生成器
            self.region_embeddings[partition_id] = self.region_embedder.compute_region_embedding(
                partition_id, partition_nodes, self.node_embeddings
            )

    def update_step(self, step):
        """更新当前步数"""
        self.current_step = step
        if hasattr(self, 'region_embedder'):
            self.region_embedder.update_step(step)

    def get_embedding_statistics(self):
        """获取嵌入统计信息"""
        if hasattr(self, 'region_embedder'):
            return self.region_embedder.get_embedding_statistics()
        return {}

    def get_observation(self) -> Dict[str, torch.Tensor]:
        """
        获取RL智能体的当前状态观察

        Returns:
            包含状态组件的字典
        """
        # 将区域嵌入作为张量
        # 注意：region_embeddings的键是实际的分区ID，不一定是连续的1,2,3...
        # 因为某些分区可能在边界节点设置为0后变为空分区
        if self.region_embeddings:
            partition_ids = sorted(self.region_embeddings.keys())
            region_embedding_tensor = torch.stack([
                self.region_embeddings[pid] for pid in partition_ids
            ], dim=0)
        else:
            # 如果没有区域嵌入，创建空张量
            region_embedding_tensor = torch.empty(0, 2 * self.embedding_dim, device=self.device)
        
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
