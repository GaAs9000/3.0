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
    计算分区的奖励函数（带调试功能）
    
    主要关注：
    1. 负载平衡 - 确保计算负载在各分区间相对均衡
    2. 电气解耦 - 减少分区间的电气连接，简化计算复杂度  
    3. 内部功率平衡 - 减少分区间功率交换，便于潮流预测
    """
    
    def __init__(self,
                 hetero_data: HeteroData,
                 reward_weights: Dict[str, float] = None,
                 device: torch.device = None,
                 debug_mode: bool = False):
        """
        初始化奖励函数
        
        Args:
            hetero_data: 异构图数据
            reward_weights: 奖励组件权重
            device: 计算设备
            debug_mode: 是否启用调试模式
        """
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        self.debug_mode = debug_mode
        
        # 默认权重配置（专为计算分区优化）
        default_weights = {
            'load_balance': 0.4,      # 计算负载平衡
            'electrical_decoupling': 0.4,  # 电气解耦（减少跨分区连接）
            'power_balance': 0.2      # 功率平衡（减少功率交换）
        }
        
        self.weights = reward_weights or default_weights
        
        # 调试信息存储
        self.debug_info = {}
        
        # 预处理数据结构
        self._setup_network_data()
        
        if self.debug_mode:
            self._print_data_summary()
    
    def _setup_network_data(self):
        """设置网络数据用于奖励计算 (已修复顺序)"""
        
        # 1. 【修复】首先计算节点总数，确保它在使用前已定义
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # 2. 然后再提取依赖于 total_nodes 的物理数据
        self._extract_physical_data()
        
        # 3. 最后提取边信息
        self._extract_edge_data()
        
        if self.debug_mode:
            print(f"🔍 网络数据设置完成:")
            print(f"    - 总节点数: {self.total_nodes}")
            print(f"    - 总负载: {self.node_loads.sum().item():.2f} MW")
            print(f"    - 总发电: {self.node_generation.sum().item():.2f} MW")
            print(f"    - 总边数: {self.all_edges.shape[1]}")
    
    def _extract_physical_data(self):
        """提取真实的物理数据（有功负载、无功负载、发电等）"""
        all_features = []
        
        # 收集所有节点特征
        for node_type in self.hetero_data.x_dict.keys():
            features = self.hetero_data.x_dict[node_type]
            all_features.append(features)
        
        self.all_node_features = torch.cat(all_features, dim=0)
        
        # 使用特征映射获取特征索引，避免硬编码
        node_type = list(self.hetero_data.x_dict.keys())[0]  # 获取第一个节点类型
        feature_map = getattr(self.hetero_data[node_type], 'feature_index_map', {})
        
        if self.debug_mode:
            print(f"📊 特征映射信息: {feature_map}")
            print(f"📊 节点特征维度: {self.all_node_features.shape}")
        
        # 基础负载特征 - 使用安全的索引访问
        pd_idx = feature_map.get('Pd', 0)
        qd_idx = feature_map.get('Qd', 1)
        self.node_loads = self.all_node_features[:, pd_idx]     # Pd (有功负载)
        self.node_reactive_loads = self.all_node_features[:, qd_idx]  # Qd (无功负载)
        
        # 发电机特征 - 安全访问
        pg_idx = feature_map.get('Pg', -1)
        qg_idx = feature_map.get('Qg', -1)
        is_gen_idx = feature_map.get('is_gen', -1)
        
        if (pg_idx >= 0 and qg_idx >= 0 and is_gen_idx >= 0 and 
            pg_idx < self.all_node_features.shape[1] and 
            qg_idx < self.all_node_features.shape[1] and 
            is_gen_idx < self.all_node_features.shape[1]):
            self.node_generation = self.all_node_features[:, pg_idx]         # Pg
            self.node_reactive_generation = self.all_node_features[:, qg_idx]  # Qg
            self.is_generator = self.all_node_features[:, is_gen_idx] > 0.5    # is_gen
        else:
            # 如果没有发电机特征或特征索引无效，使用默认值
            self.node_generation = torch.zeros_like(self.node_loads)
            self.node_reactive_generation = torch.zeros_like(self.node_loads)
            self.is_generator = torch.zeros(self.total_nodes, dtype=torch.bool, device=self.device)
            
            if self.debug_mode:
                print(f"⚠️ 警告：发电机特征不可用，使用默认值")
                print(f"   - Pg索引: {pg_idx}, Qg索引: {qg_idx}, is_gen索引: {is_gen_idx}")
                print(f"   - 特征维度: {self.all_node_features.shape[1]}")
        
        if self.debug_mode:
            print(f"\n📊 物理数据提取:")
            print(f"   - 有功负载范围: [{self.node_loads.min():.2f}, {self.node_loads.max():.2f}] MW")
            print(f"   - 无功负载范围: [{self.node_reactive_loads.min():.2f}, {self.node_reactive_loads.max():.2f}] MVar")
            print(f"   - 发电机节点数: {self.is_generator.sum().item()}")
    
    def _extract_edge_data(self):
        """提取边数据（电气参数）"""
        self.edges = []
        self.edge_admittances = []
        self.edge_resistances = []
        self.edge_reactances = []
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            
            # 将本地索引转换为全局索引
            src_type, _, dst_type = edge_type
            global_edges = self._convert_to_global_indices(edge_index, src_type, dst_type)
            
            self.edges.append(global_edges)
            
            # 使用边特征映射获取电气参数
            edge_feature_map = getattr(self.hetero_data[edge_type], 'edge_feature_index_map', {})
            
            # 安全地获取电气参数特征索引
            r_idx = edge_feature_map.get('r', 0)
            x_idx = edge_feature_map.get('x', 1)
            y_idx = edge_feature_map.get('y', 4)
            
            if (edge_attr.shape[1] > max(r_idx, x_idx, y_idx) and
                r_idx >= 0 and x_idx >= 0 and y_idx >= 0):
                resistance = edge_attr[:, r_idx]  # r
                reactance = edge_attr[:, x_idx]   # x
                admittance = edge_attr[:, y_idx]  # y
            else:
                # 默认值
                num_edges = edge_index.shape[1]
                resistance = torch.ones(num_edges, device=self.device) * 0.01
                reactance = torch.ones(num_edges, device=self.device) * 0.1
                admittance = 1.0 / torch.sqrt(resistance**2 + reactance**2)
                
                if self.debug_mode:
                    print(f"⚠️ 警告：边特征访问异常，使用默认电气参数")
                    print(f"   - 边特征维度: {edge_attr.shape[1]}")
                    print(f"   - r索引: {r_idx}, x索引: {x_idx}, y索引: {y_idx}")
            
            self.edge_resistances.append(resistance)
            self.edge_reactances.append(reactance)
            self.edge_admittances.append(admittance)
        
        if self.edges:
            self.all_edges = torch.cat(self.edges, dim=1)
            self.all_edge_admittances = torch.cat(self.edge_admittances, dim=0)
            self.all_edge_resistances = torch.cat(self.edge_resistances, dim=0)
            self.all_edge_reactances = torch.cat(self.edge_reactances, dim=0)
        else:
            self.all_edges = torch.empty((2, 0), device=self.device)
            self.all_edge_admittances = torch.empty(0, device=self.device)
            self.all_edge_resistances = torch.empty(0, device=self.device)
            self.all_edge_reactances = torch.empty(0, device=self.device)
    
    def _convert_to_global_indices(self, edge_index: torch.Tensor, 
                                 src_type: str, dst_type: str) -> torch.Tensor:
        """将本地边索引转换为全局索引"""
        # 简化实现 - 实际应用中需要正确的映射
        return edge_index
    
    def compute_reward(self,
                      current_partition: torch.Tensor,
                      boundary_nodes: torch.Tensor,
                      action: Tuple[int, int],
                      return_components: bool = False) -> float:
        """
        计算给定动作的综合奖励
        
        Args:
            current_partition: 当前分区分配
            boundary_nodes: 边界节点
            action: 执行的动作
            return_components: 是否返回各组件奖励（用于调试）
            
        Returns:
            综合奖励值或(奖励值, 组件字典)
        """
        # 清空调试信息
        self.debug_info = {}
        
        # 计算各组件奖励
        load_balance_reward = self._compute_load_balance_reward(current_partition)
        electrical_decoupling_reward = self._compute_electrical_decoupling_reward(current_partition)
        power_balance_reward = self._compute_power_balance_reward(current_partition)
        
        # 记录组件值
        components = {
            'load_balance': load_balance_reward.item(),
            'electrical_decoupling': electrical_decoupling_reward.item(),
            'power_balance': power_balance_reward.item()
        }
        
        # 综合加权奖励
        total_reward = (
            self.weights['load_balance'] * load_balance_reward +
            self.weights['electrical_decoupling'] * electrical_decoupling_reward +
            self.weights['power_balance'] * power_balance_reward
        )
        
        if self.debug_mode:
            self._print_reward_breakdown(components, total_reward.item())
        
        if return_components:
            return total_reward.item(), components
        else:
            return total_reward.item()
    
    def _compute_load_balance_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        计算负载平衡奖励（基于真实物理负载）
        """
        num_partitions = current_partition.max().item()
        if num_partitions <= 1:
            return torch.tensor(1.0, device=self.device)
        
        # 计算每个分区的总负载（使用真实的有功负载）
        partition_loads = torch.zeros(num_partitions, device=self.device)
        
        for partition_id in range(1, num_partitions + 1):
            mask = (current_partition == partition_id)
            if mask.any():
                partition_loads[partition_id - 1] = self.node_loads[mask].sum()
        
        # 计算负载分布的均匀程度
        mean_load = partition_loads.mean()
        load_std = partition_loads.std()
        load_cv = load_std / mean_load if mean_load > 0 else 0.0
        
        # 奖励较低的标准差（更均匀的分布）
        reward = torch.exp(-2.0 * load_cv)  # 调整系数以获得更好的梯度
        
        # 记录调试信息
        if self.debug_mode:
            self.debug_info['load_balance'] = {
                'partition_loads': partition_loads.cpu().numpy().tolist(),
                'mean_load': mean_load.item(),
                'std_load': load_std.item(),
                'cv': load_cv,
                'reward': reward.item()
            }
        
        return reward
    
    def _compute_electrical_decoupling_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        计算电气解耦奖励（基于真实导纳）
        """
        if self.all_edges.shape[1] == 0:
            return torch.tensor(1.0, device=self.device)
        
        # 计算跨分区连接
        src_partitions = current_partition[self.all_edges[0]]
        dst_partitions = current_partition[self.all_edges[1]]
        
        cross_partition_mask = (src_partitions != dst_partitions)
        cross_partition_edges = cross_partition_mask.sum().float()
        total_edges = self.all_edges.shape[1]
        
        # 计算跨分区的总导纳（电气耦合强度）
        cross_partition_admittance = self.all_edge_admittances[cross_partition_mask].sum()
        total_admittance = self.all_edge_admittances.sum()
        
        # 内部连接比例
        internal_edge_ratio = 1.0 - (cross_partition_edges / total_edges)
        internal_admittance_ratio = 1.0 - (cross_partition_admittance / total_admittance) if total_admittance > 0 else 1.0
        
        # 综合考虑连接数和电气耦合强度
        reward = 0.5 * torch.sigmoid(5 * (internal_edge_ratio - 0.5)) + \
                 0.5 * torch.sigmoid(5 * (internal_admittance_ratio - 0.5))
        
        if self.debug_mode:
            self.debug_info['electrical_decoupling'] = {
                'cross_partition_edges': int(cross_partition_edges.item()),
                'total_edges': total_edges,
                'cross_partition_admittance': cross_partition_admittance.item(),
                'total_admittance': total_admittance.item(),
                'internal_edge_ratio': internal_edge_ratio.item(),
                'internal_admittance_ratio': internal_admittance_ratio.item(),
                'reward': reward.item()
            }
        
        return reward
    
    def _compute_power_balance_reward(self, current_partition: torch.Tensor) -> torch.Tensor:
        """
        计算功率平衡奖励（基于真实发电和负载）
        """
        num_partitions = current_partition.max().item()
        if num_partitions <= 1:
            return torch.tensor(1.0, device=self.device)
        
        # 计算每个分区的功率不平衡
        power_imbalances = []
        
        for partition_id in range(1, num_partitions + 1):
            mask = (current_partition == partition_id)
            if mask.any():
                partition_load = self.node_loads[mask].sum()
                partition_generation = self.node_generation[mask].sum()
                imbalance = abs(partition_generation - partition_load)
                power_imbalances.append(imbalance)
        
        if not power_imbalances:
            return torch.tensor(1.0, device=self.device)
        
        # 转换为张量
        power_imbalances = torch.tensor(power_imbalances, device=self.device)
        total_imbalance = power_imbalances.sum()
        total_load = self.node_loads.sum()
        
        # 归一化不平衡
        normalized_imbalance = total_imbalance / total_load if total_load > 0 else 0.0
        
        # 奖励较小的不平衡
        reward = torch.exp(-3.0 * normalized_imbalance)
        
        if self.debug_mode:
            self.debug_info['power_balance'] = {
                'partition_imbalances': power_imbalances.cpu().numpy().tolist(),
                'total_imbalance': total_imbalance.item(),
                'normalized_imbalance': normalized_imbalance.item(),
                'reward': reward.item()
            }
        
        return reward
    
    def _print_data_summary(self):
        """打印数据摘要（调试用）"""
        print("\n" + "="*60)
        print("📊 奖励函数数据摘要")
        print("="*60)
        print(f"总节点数: {self.total_nodes}")
        print(f"总边数: {self.all_edges.shape[1] if hasattr(self, 'all_edges') else 0}")
        print(f"\n负载统计:")
        print(f"  - 总有功负载: {self.node_loads.sum().item():.2f} MW")
        print(f"  - 平均有功负载: {self.node_loads.mean().item():.2f} MW")
        print(f"  - 负载标准差: {self.node_loads.std().item():.2f} MW")
        print(f"\n发电统计:")
        print(f"  - 总有功发电: {self.node_generation.sum().item():.2f} MW")
        print(f"  - 发电机数量: {self.is_generator.sum().item()}")
        print(f"\n电气参数统计:")
        if hasattr(self, 'all_edge_admittances'):
            print(f"  - 平均导纳: {self.all_edge_admittances.mean().item():.4f}")
            print(f"  - 导纳范围: [{self.all_edge_admittances.min().item():.4f}, {self.all_edge_admittances.max().item():.4f}]")
        print("="*60)
    
    def _print_reward_breakdown(self, components: Dict[str, float], total: float):
        """打印奖励分解（调试用）"""
        print(f"\n🎯 奖励分解:")
        print(f"  负载平衡: {components['load_balance']:.4f} (权重: {self.weights['load_balance']})")
        print(f"  电气解耦: {components['electrical_decoupling']:.4f} (权重: {self.weights['electrical_decoupling']})")
        print(f"  功率平衡: {components['power_balance']:.4f} (权重: {self.weights['power_balance']})")
        print(f"  总奖励: {total:.4f}")
    
    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        return self.debug_info
    
    def set_weights(self, weights: Dict[str, float]):
        """动态设置奖励权重（用于调试不同组件）"""
        self.weights.update(weights)
        if self.debug_mode:
            print(f"\n⚙️ 更新奖励权重: {self.weights}")
