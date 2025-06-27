"""
电力网络分区MDP的增强奖励函数（带调试功能）

修复了物理属性的使用，并添加了详细的调试功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import HeteroData


class RewardFunction:
    """
    【精准修复版】电力网络分区奖励函数
    - 修复了数值不稳定的问题
    - 确保所有奖励组件值域在 [0, 1]
    - 未引入新的奖励组件和课程学习，保持结构简单
    """

    def __init__(self,
                 hetero_data: HeteroData,
                 reward_weights: Dict[str, float] = None,
                 device: torch.device = None,
                 debug_mode: bool = False):
        """
        初始化奖励函数
        """
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        self.debug_mode = debug_mode

        default_weights = {
            'load_balance': 0.4,
            'electrical_decoupling': 0.4,
            'power_balance': 0.2
        }
        self.weights = reward_weights or default_weights

        self.debug_info = {}
        self._setup_network_data()
    
    def _setup_network_data(self):
        """设置网络数据用于奖励计算 (与之前版本相同)"""
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())

        # 提取物理数据 (负载、发电)
        all_features = []
        for node_type in self.hetero_data.x_dict.keys():
            all_features.append(self.hetero_data.x_dict[node_type])
        self.all_node_features = torch.cat(all_features, dim=0)

        # 假设 Pd 在第0列，Pg 在第9列 (根据您的 data_processing.py)
        self.node_loads = self.all_node_features[:, 0]
        if self.all_node_features.shape[1] > 9:
            self.node_generation = self.all_node_features[:, 9]
        else:
            self.node_generation = torch.zeros_like(self.node_loads)

        # 提取边数据 (导纳)
        self.edges = []
        self.edge_admittances = []
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            self.edges.append(edge_index)
            # 假设导纳 'y' 在第4列
            if edge_attr.shape[1] > 4:
                self.edge_admittances.append(edge_attr[:, 4])
            else:
                self.edge_admittances.append(torch.ones(edge_attr.shape[0], device=self.device))

        if self.edges:
            self.all_edges = torch.cat(self.edges, dim=1)
            self.all_edge_admittances = torch.cat(self.edge_admittances, dim=0)
        else:
            self.all_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.all_edge_admittances = torch.empty(0, device=self.device)

    def compute_reward(self,
                      current_partition: torch.Tensor,
                      boundary_nodes: torch.Tensor,
                      action: Tuple[int, int],
                      return_components: bool = False) -> float:
        """
        计算给定动作的综合奖励
        """
        self.debug_info = {}

        load_balance_reward = self._compute_load_balance_reward(current_partition)
        electrical_decoupling_reward = self._compute_electrical_decoupling_reward(current_partition)
        power_balance_reward = self._compute_power_balance_reward(current_partition)

        components = {
            'load_balance': load_balance_reward.item(),
            'electrical_decoupling': electrical_decoupling_reward.item(),
            'power_balance': power_balance_reward.item()
        }

        total_reward = (
            self.weights['load_balance'] * load_balance_reward +
            self.weights['electrical_decoupling'] * electrical_decoupling_reward +
            self.weights['power_balance'] * power_balance_reward
        )

        # 归一化总奖励，确保其在 [0, 1] 区间
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            total_reward = total_reward / total_weight

        if return_components:
            return total_reward.item(), components
        else:
            return total_reward.item()

    def _compute_load_balance_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        【精准修复】计算负载平衡奖励，值域稳定在 [0, 1]
        """
        num_partitions = current_partition.max().item()
        if num_partitions <= 1:
            return torch.tensor(1.0, device=self.device)

        partition_loads = torch.zeros(num_partitions, device=self.device)
        for i in range(1, num_partitions + 1):
            mask = (current_partition == i)
            if mask.any():
                # 【核心修复1】使用负载的绝对值求和，避免分区负载为负
                partition_loads[i-1] = self.node_loads[mask].abs().sum()

        mean_load = partition_loads.mean()
        load_std = partition_loads.std()

        # 加上极小值 epsilon 避免除零
        load_cv = load_std / (mean_load + 1e-9)

        # 【核心修复2】使用倒数形式，将 CV 映射到 [0, 1]
        # CV 越小，奖励越接近 1.0
        reward = 1.0 / (1.0 + 2.0 * load_cv)

        return reward
    def _compute_electrical_decoupling_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        【精准修复】计算电气解耦奖励，值域稳定在 [0, 1]
        """
        if self.all_edges.shape[1] == 0:
            return torch.tensor(1.0, device=self.device)

        src_partitions = current_partition[self.all_edges[0]]
        dst_partitions = current_partition[self.all_edges[1]]

        # 仅考虑已分配节点之间的跨区连接
        cross_partition_mask = (src_partitions != dst_partitions) & (src_partitions > 0) & (dst_partitions > 0)

        cross_partition_admittance = self.all_edge_admittances[cross_partition_mask].sum()
        total_admittance = self.all_edge_admittances.sum()

        if total_admittance == 0:
            return torch.tensor(1.0, device=self.device)

        # 使用线性映射，奖励与跨区耦合度成反比
        normalized_coupling = cross_partition_admittance / total_admittance
        reward = 1.0 - normalized_coupling

        return reward.clamp(0.0, 1.0) # 双重保险
    def _compute_power_balance_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        【精准修复】计算功率平衡奖励，值域稳定在 [0, 1]
        """
        num_partitions = current_partition.max().item()
        if num_partitions <= 1:
            return torch.tensor(1.0, device=self.device)

        power_imbalances = []
        for i in range(1, num_partitions + 1):
            mask = (current_partition == i)
            if mask.any():
                imbalance = (self.node_generation[mask] - self.node_loads[mask]).sum()
                power_imbalances.append(imbalance.abs())

        if not power_imbalances:
            return torch.tensor(1.0, device=self.device)

        total_imbalance = torch.stack(power_imbalances).sum()

        # 【核心修复】使用所有节点负载的绝对值之和作为基准，确保归一化分母稳定为正
        normalization_base = self.node_loads.abs().sum()
        if normalization_base == 0:
             return torch.tensor(1.0, device=self.device)

        normalized_imbalance = total_imbalance / normalization_base

        # 使用 exp 函数，其输入现在是稳定的非负数，不会爆炸
        reward = torch.exp(-3.0 * normalized_imbalance)

        return reward

