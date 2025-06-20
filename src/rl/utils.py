"""
电力网络分区强化学习的实用函数

本模块提供实用函数，包括：
- 用于初始分区的METIS初始化
- 分区质量评估
- 状态管理助手
- 可视化工具
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
    warnings.warn("METIS不可用。使用回退的谱聚类进行初始化。")

try:
    from sklearn.cluster import SpectralClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn不可用。使用随机初始化作为回退。")


class MetisInitializer:
    """
    基于METIS的电力网络分区初始化
    
    使用METIS图分区算法提供初始分区，
    节点权重基于功率负载。
    """
    
    def __init__(self, hetero_data: HeteroData, device: torch.device):
        """
        初始化METIS分区器
        
        Args:
            hetero_data: 异构图数据
            device: 计算设备
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        
        # 设置与METIS兼容的图表示
        self._setup_graph_representation()
        
    def _setup_graph_representation(self):
        """设置与METIS兼容的图表示"""
        # 获取节点总数
        self.total_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
        
        # 设置节点映射
        self._setup_node_mappings()
        
        # 提取节点权重（功率负载）
        self._extract_node_weights()
        
        # 为METIS构建邻接列表
        self._build_adjacency_list()
        
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
        
    def _extract_node_weights(self):
        """基于功率负载提取节点权重"""
        # 连接所有节点特征
        all_features = []
        for node_type in self.node_types:
            features = self.hetero_data.x_dict[node_type]
            all_features.append(features)
            
        all_node_features = torch.cat(all_features, dim=0)
        
        # 提取负载数据（假设Pd在索引0处）
        if all_node_features.shape[1] > 0:
            loads = all_node_features[:, 0]  # Pd列
            # 转换为METIS的正整数（缩放并添加偏移）
            loads_scaled = (loads * 1000 + 1000).clamp(min=1).int()
            try:
                self.node_weights = loads_scaled.cpu().numpy()
            except RuntimeError:
                # numpy转换失败时的回退
                self.node_weights = [int(x) for x in loads_scaled.cpu().tolist()]
        else:
            # 如果没有负载数据则使用均匀权重
            self.node_weights = [1] * self.total_nodes
            
    def _build_adjacency_list(self):
        """为METIS构建邻接列表"""
        # 初始化邻接列表
        self.adjacency_list = [[] for _ in range(self.total_nodes)]
        
        # 从异构图添加边
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            
            # 转换为全局索引
            src_global = self._local_to_global(edge_index[0], src_type)
            dst_global = self._local_to_global(edge_index[1], dst_type)
            
            # 添加到邻接列表
            try:
                src_list = src_global.cpu().numpy()
                dst_list = dst_global.cpu().numpy()
            except RuntimeError:
                src_list = src_global.cpu().tolist()
                dst_list = dst_global.cpu().tolist()

            for src, dst in zip(src_list, dst_list):
                self.adjacency_list[src].append(dst)
                self.adjacency_list[dst].append(src)  # 无向图
                
        # 移除重复项和自环
        for i in range(self.total_nodes):
            self.adjacency_list[i] = list(set(self.adjacency_list[i]))
            if i in self.adjacency_list[i]:
                self.adjacency_list[i].remove(i)
                
    def initialize_partition(self, num_partitions: int) -> torch.Tensor:
        """
        使用METIS或回退方法初始化分区
        
        Args:
            num_partitions: 目标分区数量
            
        Returns:
            初始分区分配 [total_nodes]
        """
        if METIS_AVAILABLE and self.total_nodes > num_partitions:
            try:
                return self._metis_partition(num_partitions)
            except Exception as e:
                warnings.warn(f"METIS分区失败：{e}。使用回退方法。")
                
        if SKLEARN_AVAILABLE:
            return self._spectral_partition(num_partitions)
        else:
            return self._random_partition(num_partitions)
            
    def _metis_partition(self, num_partitions: int) -> torch.Tensor:
        """使用METIS算法分区"""
        # 将邻接列表转换为METIS格式
        xadj = [0]
        adjncy = []
        
        for neighbors in self.adjacency_list:
            adjncy.extend(neighbors)
            xadj.append(len(adjncy))
            
        # 运行METIS分区
        if len(adjncy) == 0:
            # 没有边 - 使用随机分区
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
            
            # 转换为基于1的索引和torch张量
            partition_tensor = torch.tensor(partition, device=self.device) + 1
            return partition_tensor
            
        except Exception as e:
            warnings.warn(f"METIS失败：{e}。使用谱聚类回退。")
            return self._spectral_partition(num_partitions)
            
    def _spectral_partition(self, num_partitions: int) -> torch.Tensor:
        """使用谱聚类分区"""
        # 构建邻接矩阵
        adj_matrix = np.zeros((self.total_nodes, self.total_nodes))
        
        for i, neighbors in enumerate(self.adjacency_list):
            for j in neighbors:
                adj_matrix[i, j] = 1.0
                
        # 处理边界情况
        if np.sum(adj_matrix) == 0:
            return self._random_partition(num_partitions)
            
        try:
            clustering = SpectralClustering(
                n_clusters=num_partitions,
                affinity='precomputed',
                random_state=42
            )
            
            partition = clustering.fit_predict(adj_matrix)
            
            # 转换为基于1的索引和torch张量
            partition_tensor = torch.tensor(partition, device=self.device) + 1
            return partition_tensor
            
        except Exception as e:
            warnings.warn(f"谱聚类失败：{e}。使用随机分区。")
            return self._random_partition(num_partitions)
            
    def _random_partition(self, num_partitions: int) -> torch.Tensor:
        """作为最终回退的随机分区"""
        partition = torch.randint(
            1, num_partitions + 1, 
            (self.total_nodes,), 
            device=self.device
        )
        return partition


class PartitionEvaluator:
    """
    综合分区质量评估
    
    提供评估分区质量的各种指标，包括
    电气、拓扑和负载平衡指标。
    """
    
    def __init__(self, hetero_data: HeteroData, device: torch.device):
        """
        初始化分区评估器
        
        Args:
            hetero_data: 异构图数据
            device: 计算设备
        """
        self.device = device
        self.hetero_data = hetero_data.to(device)
        
        # 设置评估数据
        self._setup_evaluation_data()
        
    def _setup_evaluation_data(self):
        """设置评估所需的数据"""
        # 设置节点映射
        self._setup_node_mappings()
        
        # 提取功率数据
        self._extract_power_data()
        
        # 提取电气数据
        self._extract_electrical_data()
        
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
            
        self.total_nodes = global_idx
        
    def _local_to_global(self, local_indices: torch.Tensor, node_type: str) -> torch.Tensor:
        """将局部索引转换为全局索引"""
        return self.local_to_global_map[node_type][local_indices]
        
    def _extract_power_data(self):
        """从节点特征中提取功率数据"""
        all_features = []
        for node_type in self.node_types:
            features = self.hetero_data.x_dict[node_type]
            all_features.append(features)
            
        self.all_node_features = torch.cat(all_features, dim=0)
        
        # 提取功率数据
        self.load_active = self.all_node_features[:, 0]  # Pd
        if self.all_node_features.shape[1] > 9:
            self.gen_active = self.all_node_features[:, 9]  # Pg
        else:
            self.gen_active = torch.zeros_like(self.load_active)
            
    def _extract_electrical_data(self):
        """从边中提取电气数据"""
        self.all_edges = []
        self.all_admittances = []
        
        for edge_type, edge_index in self.hetero_data.edge_index_dict.items():
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            src_type, _, dst_type = edge_type
            
            # 转换为全局索引
            src_global = self._local_to_global(edge_index[0], src_type)
            dst_global = self._local_to_global(edge_index[1], dst_type)
            
            global_edges = torch.stack([src_global, dst_global], dim=0)
            self.all_edges.append(global_edges)
            
            # 提取导纳
            if edge_attr.shape[1] > 4:
                admittances = edge_attr[:, 4]  # y列
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
        综合分区评估
        
        Args:
            partition: 分区分配 [total_nodes]
            
        Returns:
            包含评估指标的字典
        """
        metrics = {}
        
        # 基本分区信息
        num_partitions = partition.max().item()
        partition_sizes = torch.bincount(partition, minlength=num_partitions + 1)[1:]
        
        # 负载平衡指标
        partition_loads = torch.zeros(num_partitions, device=self.device)
        for i in range(1, num_partitions + 1):
            mask = (partition == i)
            if mask.any():
                partition_loads[i-1] = self.load_active[mask].sum()
                
        load_mean = torch.mean(partition_loads)
        load_std = torch.std(partition_loads)
        load_cv = (load_std / load_mean).item() if load_mean > 0 else 0.0
        
        # 耦合指标
        if self.edge_index.shape[1] > 0:
            src_partitions = partition[self.edge_index[0]]
            dst_partitions = partition[self.edge_index[1]]
            coupling_mask = (src_partitions != dst_partitions)
            
            coupling_edges = coupling_mask.sum().item()
            total_coupling = self.edge_admittances[coupling_mask].sum().item() if coupling_mask.any() else 0.0
        else:
            coupling_edges = 0
            total_coupling = 0.0
            
        # 功率平衡指标
        power_imbalances = []
        for i in range(1, num_partitions + 1):
            mask = (partition == i)
            if mask.any():
                gen = self.gen_active[mask].sum()
                load = self.load_active[mask].sum()
                imbalance = abs(gen - load).item()
                power_imbalances.append(imbalance)
                
        # 连通性检查（简化版）
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
        检查分区连通性（简化版本）
        
        Args:
            partition: 分区分配
            
        Returns:
            连通性得分（如果所有分区连通为1.0，否则更低）
        """
        # 目前返回1.0（假设连通）
        # 完整实现将检查每个分区内的图连通性
        return 1.0
