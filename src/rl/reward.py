"""
电力网络分区MDP的奖励函数

本模块实现MDP公式中指定的复合奖励函数：
R_t = w1 * R_balance + w2 * R_decoupling + w3 * R_internal_balance

其中：
- R_balance: 分区负载平衡奖励(-Var(L1, ..., LK))
- R_decoupling: 电气解耦奖励(-Σ|Y_uv| 对于耦合边)
- R_internal_balance: 内部功率平衡奖励(-Σ(P_gen - P_load)²)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import HeteroData


class RewardFunction:
    """
    计算分区的奖励函数
    
    主要关注：
    1. 负载平衡 - 确保计算负载在各分区间相对均衡
    2. 电气解耦 - 减少分区间的电气连接，简化计算复杂度  
    3. 内部功率平衡 - 减少分区间功率交换，便于潮流预测
    """
    
    def __init__(self,
                 hetero_data: HeteroData,
                 reward_weights: Dict[str, float] = None,
                 device: torch.device = None):
        """
        初始化奖励函数
        
        Args:
            hetero_data: 异构图数据
            reward_weights: 奖励组件权重
            device: 计算设备
        """
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        
        # 默认权重配置（专为计算分区优化）
        default_weights = {
            'load_balance': 0.4,      # 计算负载平衡
            'electrical_decoupling': 0.4,  # 电气解耦（减少跨分区连接）
            'power_balance': 0.2      # 功率平衡（减少功率交换）
        }
        
        self.weights = reward_weights or default_weights
        
        # 预处理数据结构
        self._setup_network_data()
        
    def _setup_network_data(self):
        """设置网络数据用于奖励计算"""
        # 提取边信息
        self.edges = []
        self.edge_powers = []
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            
            # 将本地索引转换为全局索引
            src_type, _, dst_type = edge_type
            global_edges = self._convert_to_global_indices(edge_index, src_type, dst_type)
            
            self.edges.append(global_edges)
            
            # 提取有功功率信息（用于功率平衡计算）
            if edge_attr.shape[1] > 1:
                power = edge_attr[:, 1]  # 假设有功功率在索引1
                self.edge_powers.append(power)
            else:
                power = torch.ones(edge_index.shape[1], device=self.device)
                self.edge_powers.append(power)
                
        if self.edges:
            self.all_edges = torch.cat(self.edges, dim=1)
            self.all_edge_powers = torch.cat(self.edge_powers, dim=0)
        else:
            self.all_edges = torch.empty((2, 0), device=self.device)
            self.all_edge_powers = torch.empty(0, device=self.device)
            
        # 节点数量
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # 节点负载数据（用于负载平衡计算）
        self._setup_node_loads()
        
    def _setup_node_loads(self):
        """设置节点负载数据"""
        # 合并所有节点特征来估计负载
        all_node_features = []
        
        for node_type, features in self.hetero_data.x_dict.items():
            all_node_features.append(features)
            
        if all_node_features:
            # 使用节点特征的L2范数作为负载指标
            combined_features = torch.cat(all_node_features, dim=0)
            self.node_loads = torch.norm(combined_features, dim=1)
        else:
            self.node_loads = torch.ones(self.total_nodes, device=self.device)
            
    def _convert_to_global_indices(self, edge_index: torch.Tensor, 
                                 src_type: str, dst_type: str) -> torch.Tensor:
        """将本地边索引转换为全局索引"""
        # 这里需要节点类型映射，简化实现
        # 实际实现中应该使用正确的映射逻辑
        return edge_index
        
    def compute_reward(self,
                      current_partition: torch.Tensor,
                      boundary_nodes: torch.Tensor,
                      action: Tuple[int, int]) -> float:
        """
        计算给定动作的综合奖励
        
        专为计算分区优化：
        - 平衡计算负载
        - 减少跨分区连接复杂度
        - 优化功率流分布
        
        Args:
            current_partition: 当前分区分配
            boundary_nodes: 边界节点
            action: 执行的动作
            
        Returns:
            综合奖励值
        """
        # 计算各组件奖励
        load_balance_reward = self._compute_load_balance_reward(current_partition)
        electrical_decoupling_reward = self._compute_electrical_decoupling_reward(current_partition)
        power_balance_reward = self._compute_power_balance_reward(current_partition)
        
        # 综合加权奖励
        total_reward = (
            self.weights['load_balance'] * load_balance_reward +
            self.weights['electrical_decoupling'] * electrical_decoupling_reward +
            self.weights['power_balance'] * power_balance_reward
        )
        
        return total_reward.item()
        
    def _compute_load_balance_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        计算负载平衡奖励
        
        目的：确保各分区的计算负载相对均衡，避免某些分区计算任务过重
        
        Args:
            current_partition: 当前分区分配
            
        Returns:
            负载平衡奖励（越平衡奖励越高）
        """
        num_partitions = current_partition.max().item()
        if num_partitions <= 1:
            return torch.tensor(1.0, device=self.device)
            
        # 计算每个分区的总负载
        partition_loads = torch.zeros(num_partitions, device=self.device)
        
        for partition_id in range(1, num_partitions + 1):
            mask = (current_partition == partition_id)
            if mask.any():
                partition_loads[partition_id - 1] = self.node_loads[mask].sum()
                
        # 计算负载分布的均匀程度
        mean_load = partition_loads.mean()
        load_std = partition_loads.std()
        
        # 奖励较低的标准差（更均匀的分布）
        if mean_load > 0:
            normalized_std = load_std / mean_load
            reward = torch.exp(-normalized_std)  # 标准差越小，奖励越高
        else:
            reward = torch.tensor(1.0, device=self.device)
            
        return reward
        
    def _compute_electrical_decoupling_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        计算电气解耦奖励
        
        目的：减少分区间的电气连接数量，简化潮流计算的复杂度
        
        Args:
            current_partition: 当前分区分配
            
        Returns:
            电气解耦奖励（跨分区连接越少奖励越高）
        """
        if self.all_edges.shape[1] == 0:
            return torch.tensor(1.0, device=self.device)
            
        # 计算跨分区连接数
        src_partitions = current_partition[self.all_edges[0]]
        dst_partitions = current_partition[self.all_edges[1]]
        
        cross_partition_edges = (src_partitions != dst_partitions).sum().float()
        total_edges = self.all_edges.shape[1]
        
        # 计算内部连接比例（用于奖励）
        internal_edge_ratio = 1.0 - (cross_partition_edges / total_edges)
        
        # 使用sigmoid函数平滑奖励
        reward = torch.sigmoid(5 * (internal_edge_ratio - 0.5))
        
        return reward
        
    def _compute_power_balance_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        计算功率平衡奖励
        
        目的：减少分区间的功率交换，使联络线潮流预测更加准确
        
        Args:
            current_partition: 当前分区分配
            
        Returns:
            功率平衡奖励（分区间功率交换越少奖励越高）
        """
        if self.all_edges.shape[1] == 0:
            return torch.tensor(1.0, device=self.device)
            
        # 计算跨分区功率流
        src_partitions = current_partition[self.all_edges[0]]
        dst_partitions = current_partition[self.all_edges[1]]
        
        cross_partition_mask = (src_partitions != dst_partitions)
        cross_partition_powers = self.all_edge_powers[cross_partition_mask]
        
        # 计算总的跨分区功率交换
        total_cross_power = cross_partition_powers.abs().sum()
        total_power = self.all_edge_powers.abs().sum()
        
        if total_power > 0:
            # 内部功率流比例
            internal_power_ratio = 1.0 - (total_cross_power / total_power)
            reward = torch.sigmoid(3 * (internal_power_ratio - 0.3))
        else:
            reward = torch.tensor(1.0, device=self.device)
            
        return reward
