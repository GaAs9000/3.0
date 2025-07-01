"""
电力网络分区MDP的动作空间管理

本模块实现MDP公式中指定的两阶段动作空间：
1. 选择边界节点（来自Bdry_t）
2. 选择目标分区（来自相邻分区）

包括用于拓扑和物理约束的mask function。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from torch_geometric.data import HeteroData


class ActionSpace:
    """
    管理电力网络分区的两阶段动作空间
    
    动作结构：
    - 阶段1：选择边界节点 i_t ∈ Bdry_t
    - 阶段2：选择目标分区 k_new ≠ k_old 来自相邻分区
    
    动作表示：(node_idx, target_partition)
    """
    
    def __init__(self, 
                 hetero_data: HeteroData,
                 num_partitions: int,
                 device: torch.device):
        """
        初始化动作空间
        
        Args:
            hetero_data: 异构图数据
            num_partitions: 目标分区数量
            device: 计算设备
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        self.num_partitions = num_partitions
        
        # 设置邻接信息
        self._setup_adjacency_info()
        
        # 动作验证缓存
        self._action_cache = {}
        
    def _setup_adjacency_info(self):
        """设置用于动作验证的邻接信息"""
        # 获取节点总数
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # 创建全局邻接列表
        self.adjacency_list = [set() for _ in range(self.total_nodes)]
        
        # 设置节点类型映射
        self._setup_node_mappings()
        
        # 从异构边构建邻接列表
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            
            # 转换为全局索引
            src_global = self._local_to_global(edge_index[0], src_type)
            dst_global = self._local_to_global(edge_index[1], dst_type)
            
            # 添加到邻接列表（无向图）
            for src, dst in zip(src_global, dst_global):
                self.adjacency_list[src.item()].add(dst.item())
                self.adjacency_list[dst.item()].add(src.item())
                
    def _setup_node_mappings(self):
        """设置局部和全局节点索引之间的映射"""
        self.node_types = list(self.hetero_data.x_dict.keys())
        self.local_to_global_map = {}
        
        global_idx = 0
        for node_type in self.node_types:
            num_nodes = self.hetero_data.x_dict[node_type].shape[0]
            global_indices = torch.arange(global_idx, global_idx + num_nodes, device=self.device)
            self.local_to_global_map[node_type] = global_indices
            global_idx += num_nodes
            
    def _local_to_global(self, local_indices: torch.Tensor, node_type: str) -> torch.Tensor:
        """将局部索引转换为全局索引"""
        return self.local_to_global_map[node_type][local_indices]
        
    def get_valid_actions(self,
                         current_partition: torch.Tensor,
                         boundary_nodes: torch.Tensor,
                         constraint_mode: str = 'hard') -> List[Tuple[int, int]]:
        """
        获取当前状态的所有有效动作 - 支持软约束模式

        Args:
            current_partition: 当前分区分配 [total_nodes]
            boundary_nodes: 当前边界节点 [num_boundary]
            constraint_mode: 约束模式 ('hard' 或 'soft')

        Returns:
            有效的(node_idx, target_partition)元组列表
        """
        valid_actions = []

        for node_idx in boundary_nodes:
            node_idx_int = node_idx.item()
            current_node_partition = current_partition[node_idx_int].item()

            # 获取相邻分区
            neighboring_partitions = self._get_neighboring_partitions(
                node_idx_int, current_partition
            )

            # 为每个相邻分区添加有效动作
            for target_partition in neighboring_partitions:
                if target_partition != current_node_partition:
                    action = (node_idx_int, target_partition)

                    # 在软约束模式下，不过滤连通性违规的动作
                    if constraint_mode == 'soft':
                        valid_actions.append(action)
                    else:
                        # 硬约束模式：检查连通性约束
                        if hasattr(self, 'mask_handler') and self.mask_handler:
                            violation_info = self.mask_handler.check_connectivity_violation(
                                current_partition, action
                            )
                            if not violation_info['violates_connectivity']:
                                valid_actions.append(action)
                        else:
                            # 如果没有mask_handler，默认添加动作
                            valid_actions.append(action)

        return valid_actions
        
    def _get_neighboring_partitions(self, 
                                   node_idx: int,
                                   current_partition: torch.Tensor) -> Set[int]:
        """
        获取给定节点的相邻分区
        
        Args:
            node_idx: 全局节点索引
            current_partition: 当前分区分配
            
        Returns:
            相邻分区ID的集合
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
        检查动作是否有效
        
        Args:
            action: (node_idx, target_partition)元组
            current_partition: 当前分区分配
            boundary_nodes: 当前边界节点
            
        Returns:
            如果动作有效返回True，否则返回False
        """
        node_idx, target_partition = action
        
        # 检查节点是否在边界节点中
        if node_idx not in boundary_nodes:
            return False
            
        # 检查目标分区是否有效
        if target_partition < 1 or target_partition > self.num_partitions:
            return False
            
        # 检查节点当前是否在不同分区中
        current_node_partition = current_partition[node_idx].item()
        if target_partition == current_node_partition:
            return False
            
        # 检查目标分区是否相邻
        neighboring_partitions = self._get_neighboring_partitions(node_idx, current_partition)
        if target_partition not in neighboring_partitions:
            return False
            
        return True
        
    def get_action_mask(self,
                       current_partition: torch.Tensor,
                       boundary_nodes: torch.Tensor) -> torch.Tensor:
        """
        获取当前状态的动作掩码
        
        Args:
            current_partition: 当前分区分配
            boundary_nodes: 当前边界节点
            
        Returns:
            布尔张量 [total_nodes, num_partitions] 表示有效动作
        """
        # 初始化掩码（全为False）
        action_mask = torch.zeros(
            self.total_nodes, self.num_partitions, 
            dtype=torch.bool, device=self.device
        )
        
        # 将有效动作设置为True
        for node_idx in boundary_nodes:
            node_idx_int = node_idx.item()
            current_node_partition = current_partition[node_idx_int].item()
            
            # 获取相邻分区
            neighboring_partitions = self._get_neighboring_partitions(
                node_idx_int, current_partition
            )
            
            # 标记有效的目标分区
            for target_partition in neighboring_partitions:
                if target_partition != current_node_partition:
                    # 转换为基于0的索引用于掩码
                    action_mask[node_idx_int, target_partition - 1] = True
                    
        return action_mask
        
    def sample_random_action(self,
                           current_partition: torch.Tensor,
                           boundary_nodes: torch.Tensor) -> Optional[Tuple[int, int]]:
        """
        采样一个随机有效动作
        
        Args:
            current_partition: 当前分区分配
            boundary_nodes: 当前边界节点
            
        Returns:
            随机有效动作或None（如果没有有效动作）
        """
        valid_actions = self.get_valid_actions(current_partition, boundary_nodes)
        
        if len(valid_actions) == 0:
            return None
            
        # 采样随机动作
        action_idx = torch.randint(0, len(valid_actions), (1,)).item()
        return valid_actions[action_idx]
        
    def get_action_space_size(self) -> int:
        """
        获取动作空间的最大大小
        
        Returns:
            可能动作的最大数量
        """
        return self.total_nodes * self.num_partitions

    def get_advanced_mask(self,
                         state: Dict[str, torch.Tensor],
                         boundary_nodes: torch.Tensor,
                         hetero_data: HeteroData,
                         apply_constraints: bool = True,
                         constraint_mode: str = 'hard') -> torch.Tensor:
        """
        获取带有高级约束的动作掩码

        支持硬约束和软约束两种模式：
        - 硬约束模式：直接从掩码中移除违规动作（传统模式）
        - 软约束模式：保留所有基础有效动作，违规信息用于奖励计算

        Args:
            state: 当前状态
            boundary_nodes: 当前边界节点
            hetero_data: 异构图数据
            apply_constraints: 是否应用高级约束
            constraint_mode: 约束模式 ('hard' 或 'soft')

        Returns:
            动作掩码
        """
        # 获取基础掩码
        base_mask = self.get_action_mask(state['partition_assignment'], boundary_nodes)

        if not apply_constraints:
            return base_mask

        # 初始化高级掩码处理器
        if not hasattr(self, 'mask_handler'):
            self.mask_handler = ActionMask(hetero_data, self.device)

        current_partition = state['partition_assignment']

        # 软约束模式：返回基础掩码，不进行额外过滤
        if constraint_mode == 'soft':
            return base_mask

        # 硬约束模式：应用传统的约束过滤
        advanced_mask = base_mask.clone()

        # 应用连通性约束（硬约束模式）
        for node_idx in boundary_nodes:
            for target_partition in range(self.num_partitions):
                if advanced_mask[node_idx, target_partition]:
                    action = (node_idx.item(), target_partition + 1)
                    if not self.mask_handler.apply_connectivity_constraint(
                        advanced_mask, current_partition, action):
                        advanced_mask[node_idx, target_partition] = False

        # 应用拓扑优化约束
        advanced_mask = self.mask_handler.apply_topology_optimization(
            advanced_mask, current_partition, boundary_nodes)

        # 应用大小平衡约束
        advanced_mask = self.mask_handler.apply_size_balance_constraint(
            advanced_mask, current_partition)

        return advanced_mask


class ActionMask:
    """
    用于计算分区的基础约束管理
    """
    
    def __init__(self, 
                 hetero_data: HeteroData,
                 device: torch.device):
        """
        初始化动作掩码
        
        Args:
            hetero_data: 异构图数据
            device: 计算设备
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        
        # 设置基础的图结构信息
        self._setup_graph_structure()
        
    def _setup_graph_structure(self):
        """设置图结构信息用于拓扑约束"""
        # 构建全局邻接关系（用于连通性检查）
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        self.adjacency_list = [[] for _ in range(self.total_nodes)]
        
        # 设置节点映射
        self._setup_node_mappings()
        
        # 构建邻接列表
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_global = self._local_to_global(edge_index[0], src_type)
            dst_global = self._local_to_global(edge_index[1], dst_type)
            
            for src, dst in zip(src_global, dst_global):
                self.adjacency_list[src.item()].append(dst.item())
                self.adjacency_list[dst.item()].append(src.item())
                
    def _setup_node_mappings(self):
        """设置节点类型映射"""
        self.node_types = list(self.hetero_data.x_dict.keys())
        self.local_to_global_map = {}
        
        global_idx = 0
        for node_type in self.node_types:
            num_nodes = self.hetero_data.x_dict[node_type].shape[0]
            global_indices = torch.arange(global_idx, global_idx + num_nodes, device=self.device)
            self.local_to_global_map[node_type] = global_indices
            global_idx += num_nodes
            
    def _local_to_global(self, local_indices: torch.Tensor, node_type: str) -> torch.Tensor:
        """将局部索引转换为全局索引"""
        return self.local_to_global_map[node_type][local_indices]
        
    def check_connectivity_violation(self,
                                   current_partition: torch.Tensor,
                                   action: Tuple[int, int]) -> Dict[str, Any]:
        """
        检查动作是否会违反连通性约束（仅检测，不阻止）

        目的：为软约束系统提供违规信息，用于奖励惩罚计算

        Args:
            current_partition: 当前分区分配
            action: 提议的动作

        Returns:
            包含违规信息的字典：
            - 'violates_connectivity': bool, 是否违反连通性
            - 'violation_severity': float, 违规严重程度 (0.0-1.0)
            - 'isolated_nodes_count': int, 会产生的孤立节点数量
            - 'affected_partition_size': int, 受影响分区的大小
        """
        node_idx, target_partition = action
        current_node_partition = current_partition[node_idx].item()

        # 检查移动节点后，原分区是否仍然连通
        current_partition_nodes = torch.where(current_partition == current_node_partition)[0]

        # 如果原分区只有这一个节点，移动后该分区消失，这是允许的
        if len(current_partition_nodes) <= 1:
            return {
                'violates_connectivity': False,
                'violation_severity': 0.0,
                'isolated_nodes_count': 0,
                'affected_partition_size': 1
            }

        # 如果有多个节点，需要检查移除该节点后剩余节点是否连通
        remaining_nodes = current_partition_nodes[current_partition_nodes != node_idx]
        if len(remaining_nodes) <= 1:
            return {
                'violates_connectivity': False,
                'violation_severity': 0.0,
                'isolated_nodes_count': 0,
                'affected_partition_size': len(current_partition_nodes)
            }

        # 检查剩余节点中的孤立节点
        isolated_nodes = []
        for remaining_node in remaining_nodes:
            has_connection = False
            for neighbor in self.adjacency_list[remaining_node.item()]:
                if neighbor in remaining_nodes:
                    has_connection = True
                    break
            if not has_connection:
                isolated_nodes.append(remaining_node.item())

        # 计算违规严重程度
        violation_severity = 0.0
        if len(isolated_nodes) > 0:
            # 基于孤立节点比例计算严重程度
            violation_severity = len(isolated_nodes) / len(remaining_nodes)

        return {
            'violates_connectivity': len(isolated_nodes) > 0,
            'violation_severity': violation_severity,
            'isolated_nodes_count': len(isolated_nodes),
            'affected_partition_size': len(current_partition_nodes)
        }

    def apply_connectivity_constraint(self,
                                    action_mask: torch.Tensor,
                                    current_partition: torch.Tensor,
                                    action: Tuple[int, int]) -> bool:
        """
        【已弃用】保留用于向后兼容，建议使用 check_connectivity_violation

        检查动作是否保持分区内部连通性
        """
        violation_info = self.check_connectivity_violation(current_partition, action)
        return not violation_info['violates_connectivity']
        
    def apply_topology_optimization(self,
                                  action_mask: torch.Tensor,
                                  current_partition: torch.Tensor,
                                  boundary_nodes: torch.Tensor) -> torch.Tensor:
        """
        应用拓扑优化约束
        
        目的：优化分区边界，减少分区间的连接数量，简化潮流计算
        
        Args:
            action_mask: 当前动作掩码
            current_partition: 当前分区分配
            boundary_nodes: 当前边界节点
            
        Returns:
            优化后的动作掩码
        """
        updated_mask = action_mask.clone()
        
        for node_idx in boundary_nodes:
            node_idx_int = node_idx.item()
            current_node_partition = current_partition[node_idx_int].item()
            
            # 统计该节点邻居的分区分布
            neighbor_partitions = {}
            for neighbor_idx in self.adjacency_list[node_idx_int]:
                neighbor_partition = current_partition[neighbor_idx].item()
                if neighbor_partition != current_node_partition:
                    neighbor_partitions[neighbor_partition] = neighbor_partitions.get(neighbor_partition, 0) + 1
                    
            # 如果某个邻居分区连接数明显更多，优先移动到该分区（减少跨分区连接）
            if neighbor_partitions:
                max_connections = max(neighbor_partitions.values())
                total_neighbors = sum(neighbor_partitions.values())
                
                # 如果某个分区的连接数超过80%，强烈建议移动到该分区
                for partition_id, connections in neighbor_partitions.items():
                    if connections < 0.8 * max_connections:
                        # 降低移动到连接较少分区的优先级（但不完全禁止）
                        # 这里可以选择保持原有掩码，不做强制限制
                        pass
                        
        return updated_mask
        
    def apply_size_balance_constraint(self,
                                    action_mask: torch.Tensor,
                                    current_partition: torch.Tensor,
                                    max_size_ratio: float = 3.0) -> torch.Tensor:
        """
        应用分区大小平衡约束
        
        目的：避免分区大小差异过大，确保计算负载相对均衡
        
        Args:
            action_mask: 当前动作掩码
            current_partition: 当前分区分配
            max_size_ratio: 最大分区大小比例
            
        Returns:
            应用大小约束后的动作掩码
        """
        updated_mask = action_mask.clone()
        
        # 计算各分区大小
        num_partitions = current_partition.max().item()
        partition_sizes = torch.bincount(current_partition, minlength=num_partitions + 1)[1:]
        
        if len(partition_sizes) == 0:
            return updated_mask
            
        avg_size = partition_sizes.float().mean()
        
        # 检查每个可能的动作是否会导致分区过大
        valid_actions = torch.where(updated_mask)
        for i in range(len(valid_actions[0])):
            node_idx = valid_actions[0][i].item()
            target_partition = valid_actions[1][i].item() + 1  # 转换为1-based
            
            # 检查目标分区移动后是否会过大
            target_size_after = partition_sizes[target_partition - 1] + 1
            if target_size_after > max_size_ratio * avg_size:
                updated_mask[node_idx, target_partition - 1] = False
                
        return updated_mask
