#!/usr/bin/env python3
"""
Enhanced Power Grid Partitioning Environment

增强的电网分区环境，支持：
- 动态K值
- 连通性保证的动作空间
- 分区特征计算和缓存
- 与双塔架构的无缝集成
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any, Set
from torch_geometric.data import HeteroData
import networkx as nx
from collections import defaultdict
import logging

from .environment import PowerGridPartitioningEnv
from .state import StateManager
from .action_space import ActionSpace
from .decision_context import AbstractDecisionContext, materialize_decision_context

logger = logging.getLogger(__name__)


class EnhancedPowerGridPartitioningEnv(PowerGridPartitioningEnv):
    """
    增强的电网分区环境
    
    扩展原环境以支持：
    1. 连通性保证的动作查询
    2. 分区特征计算
    3. 动态K值支持
    4. 高效的增量更新
    """
    
    def __init__(self, *args, enable_features_cache: bool = True, **kwargs):
        """
        初始化增强环境
        
        Args:
            *args: 传递给父类的参数
            enable_features_cache: 是否启用分区特征缓存
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        
        # 初始化增强功能
        self.enable_features_cache = enable_features_cache
        self.partition_features_cache = {}
        self.connectivity_cache = {}
        
        # 构建NetworkX图用于连通性检查
        self._build_nx_graph()
        
        # 初始化分区信息
        self._update_partition_info()
        
        logger.info("EnhancedPowerGridPartitioningEnv initialized")
    
    def _build_nx_graph(self):
        """构建NetworkX图用于高效的连通性检查"""
        self.nx_graph = nx.Graph()
        
        # 添加节点
        num_nodes = self.hetero_data['bus'].x.size(0)
        self.nx_graph.add_nodes_from(range(num_nodes))
        
        # 添加边（从异构图的边索引）
        edge_index = self.hetero_data['bus', 'connects', 'bus'].edge_index
        
        # 过滤无效边索引
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & (edge_index[0] >= 0) & (edge_index[1] >= 0)
        invalid_count = (~valid_mask).sum().item()
        if invalid_count > 0:
            logger.warning(f"Filtered {invalid_count} invalid edges (out of range [0,{num_nodes-1}])")
            edge_index = edge_index[:, valid_mask]
        
        edges = edge_index.t().cpu().numpy()  # 使用过滤后的edge_index
        self.nx_graph.add_edges_from(edges)
        
        logger.info(f"Built NetworkX graph with {num_nodes} nodes and {len(edges)} edges")
    
    def get_connectivity_safe_partitions(self, node_id: int) -> List[int]:
        """
        获取节点可以移动到的所有保证连通性的分区

        Args:
            node_id: 边界节点ID

        Returns:
            可以安全移动到的分区ID列表
        """
        # 确保node_id是正确的类型
        node_id_int = int(node_id.item()) if torch.is_tensor(node_id) else int(node_id)

        # 🔧 修复：检查节点ID是否在有效范围内，并处理索引偏移
        max_nodes = self.state_manager.current_partition.size(0)
        if node_id_int >= max_nodes:
            # 如果节点ID从1开始，转换为从0开始的索引
            node_idx = node_id_int - 1
            if node_idx < 0 or node_idx >= max_nodes:
                # 如果仍然越界，返回空列表
                return []
        else:
            # 节点ID已经是从0开始的索引
            node_idx = node_id_int

        # 使用更高效的缓存键：只包含节点ID和其当前分区
        current_assignment = self.state_manager.current_partition[node_idx]

        # 获取节点的邻居分区（这是必要的计算）
        # 修复索引偏移问题
        max_node_id = max(self.nx_graph.nodes()) if self.nx_graph.nodes() else 0
        if node_id_int > max_node_id:
            logger.warning(f"Node {node_id_int} exceeds graph range [0,{max_node_id}], mapping to valid range")
            node_id_int = node_id_int % (max_node_id + 1)
        
        try:
            neighbors = list(self.nx_graph.neighbors(node_id_int))
        except Exception as e:
            logger.warning(f"Failed to get neighbors for node {node_id_int}: {e}. Graph has {len(self.nx_graph.nodes())} nodes (0-{max(self.nx_graph.nodes()) if self.nx_graph.nodes() else 0})")
            # 回退：返回除当前分区外的所有分区
            num_partitions = int(self.state_manager.current_partition.max().item())
            return [p for p in range(1, num_partitions + 1) if p != current_assignment]

        neighbor_partitions = set()
        for neighbor in neighbors:
            # 🔧 修复：处理邻居节点的索引偏移
            max_nodes = self.state_manager.current_partition.size(0)
            if neighbor >= max_nodes:
                # 如果邻居节点ID从1开始，转换为从0开始的索引
                neighbor_idx = neighbor - 1
                if neighbor_idx < 0 or neighbor_idx >= max_nodes:
                    continue  # 跳过无效的邻居节点
            else:
                # 邻居节点ID已经是从0开始的索引
                neighbor_idx = neighbor

            partition = self.state_manager.current_partition[neighbor_idx].item()
            if partition != current_assignment and partition > 0:
                neighbor_partitions.add(partition)

        # 如果没有邻居分区，返回所有其他分区（简化处理）
        if not neighbor_partitions:
            num_partitions = int(self.state_manager.current_partition.max().item())
            return [p for p in range(1, num_partitions + 1) if p != current_assignment]

        # 对每个邻居分区进行连通性测试
        safe_partitions = []
        for target_partition in neighbor_partitions:
            # 创建针对特定移动的缓存键
            move_key = (node_id_int, current_assignment, target_partition,
                       self.state_manager.current_step)  # 使用step作为版本号

            if move_key in self.connectivity_cache:
                if self.connectivity_cache[move_key]:
                    safe_partitions.append(target_partition)
            else:
                # 模拟节点移动进行连通性验证
                test_partition = self.state_manager.current_partition.clone()
                test_partition[node_id_int] = target_partition

                # 检查连通性
                is_safe = self._check_partition_connectivity(test_partition)
                self.connectivity_cache[move_key] = is_safe

                if is_safe:
                    safe_partitions.append(target_partition)

        # 如果没有安全分区，至少返回一个邻居分区（降级处理）
        if not safe_partitions and neighbor_partitions:
            safe_partitions = [list(neighbor_partitions)[0]]

        # 限制缓存大小
        if len(self.connectivity_cache) > 10000:
            # 清理旧的缓存条目
            self.connectivity_cache.clear()

        return safe_partitions
    
    def _check_partition_connectivity(self, partition: Dict[int, int]) -> bool:
        """
        检查分区方案是否保持所有分区的连通性
        
        Args:
            partition: 分区分配 {node_id: partition_id}
        
        Returns:
            是否所有分区都是连通的
        """
        # 按分区分组节点
        partition_nodes = defaultdict(set)
        # 处理张量类型的partition
        if torch.is_tensor(partition):
            for node in range(len(partition)):
                pid = partition[node].item()
                if pid > 0:  # 忽略未分配的节点
                    partition_nodes[pid].add(node)
        else:
            # 处理字典类型的partition
            for node, pid in partition.items():
                if pid > 0:  # 忽略未分配的节点
                    partition_nodes[pid].add(node)
        
        # 检查每个分区的连通性
        for pid, nodes in partition_nodes.items():
            if not self._is_connected_subgraph(nodes):
                return False
        
        return True
    
    def _is_connected_subgraph(self, nodes: Set[int]) -> bool:
        """
        检查节点集合是否形成连通子图
        
        Args:
            nodes: 节点集合
        
        Returns:
            是否连通
        """
        if not nodes:
            return True
        
        # 使用BFS检查连通性
        subgraph = self.nx_graph.subgraph(nodes)
        return nx.is_connected(subgraph)
    
    def get_partition_features(self, partition_id: int) -> Dict[str, float]:
        """
        计算分区的特征
        
        Args:
            partition_id: 分区ID
        
        Returns:
            分区特征字典
        """
        # 🔧 修复：确保partition_id是单个值而不是list
        if isinstance(partition_id, (list, tuple)):
            if len(partition_id) == 0:
                return self._empty_partition_features()
            partition_id = partition_id[0]  # 取第一个元素
        elif isinstance(partition_id, torch.Tensor):
            partition_id = partition_id.item()

        # 检查缓存
        if self.enable_features_cache and partition_id in self.partition_features_cache:
            return self.partition_features_cache[partition_id]
        
        # 获取分区节点
        # current_partition是一个张量，需要使用张量操作
        mask = (self.state_manager.current_partition == partition_id)
        partition_nodes = torch.where(mask)[0].cpu().tolist()
        
        if not partition_nodes:
            return self._empty_partition_features()
        
        # 计算特征
        features = self._compute_partition_features(partition_nodes, partition_id)
        
        # 缓存结果
        if self.enable_features_cache:
            self.partition_features_cache[partition_id] = features
        
        return features
    
    def _compute_partition_features(self, nodes: List[int], partition_id: int) -> Dict[str, float]:
        """
        计算分区的详细特征
        
        Args:
            nodes: 分区中的节点列表
            partition_id: 分区ID
        
        Returns:
            特征字典
        """
        # 基础统计
        num_nodes = len(nodes)
        total_nodes = self.hetero_data['bus'].x.size(0)
        
        # 负荷和发电信息
        node_features = self.hetero_data['bus'].x[nodes]
        total_load = node_features[:, 2].sum().item()  # PD
        total_generation = 0.0
        
        # 从gen数据中获取发电信息
        if 'gen' in self.hetero_data:
            gen_buses = self.hetero_data['gen'].bus_idx
            gen_powers = self.hetero_data['gen'].x[:, 0]  # PG
            for i, bus in enumerate(gen_buses):
                if bus.item() in nodes:
                    total_generation += gen_powers[i].item()
        
        # 内部边数
        internal_edges = 0
        edge_index = self.hetero_data['bus', 'connects', 'bus'].edge_index
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in nodes and dst in nodes:
                internal_edges += 1
        internal_edges //= 2  # 无向图，每条边计算了两次
        
        # 边界节点数
        boundary_nodes = 0
        for node in nodes:
            # 确保node是Python int类型，用于NetworkX接口
            node_int = int(node) if not isinstance(node, int) else node
            neighbors = list(self.nx_graph.neighbors(node_int))
            for neighbor in neighbors:
                if neighbor < len(self.state_manager.current_partition):
                    if self.state_manager.current_partition[neighbor].item() != partition_id:
                        boundary_nodes += 1
                        break
        
        # 计算比率
        size_ratio = num_nodes / total_nodes if total_nodes > 0 else 0
        
        # 获取全局统计
        grid_info = self._get_grid_info()
        load_ratio = total_load / grid_info['total_load'] if grid_info['total_load'] > 0 else 0
        generation_ratio = total_generation / grid_info['total_generation'] if grid_info['total_generation'] > 0 else 0
        
        # 内部连通性
        max_edges = num_nodes * (num_nodes - 1) / 2
        internal_connectivity = internal_edges / max_edges if max_edges > 0 else 0
        
        # 边界比率
        boundary_ratio = boundary_nodes / num_nodes if num_nodes > 0 else 0
        
        # 功率不平衡
        power_imbalance = (total_generation - total_load) / total_load if total_load > 0 else 0
        
        return {
            'nodes': nodes,
            'num_nodes': num_nodes,
            'load': total_load,
            'generation': total_generation,
            'internal_edges': internal_edges,
            'boundary_nodes': boundary_nodes,
            'size_ratio': size_ratio,
            'load_ratio': load_ratio,
            'generation_ratio': generation_ratio,
            'internal_connectivity': internal_connectivity,
            'boundary_ratio': boundary_ratio,
            'power_imbalance': power_imbalance
        }
    
    def _empty_partition_features(self) -> Dict[str, float]:
        """返回空分区的特征"""
        return {
            'nodes': [],
            'num_nodes': 0,
            'load': 0.0,
            'generation': 0.0,
            'internal_edges': 0,
            'boundary_nodes': 0,
            'size_ratio': 0.0,
            'load_ratio': 0.0,
            'generation_ratio': 0.0,
            'internal_connectivity': 0.0,
            'boundary_ratio': 0.0,
            'power_imbalance': 0.0
        }
    
    def _get_grid_info(self) -> Dict[str, float]:
        """获取电网全局信息"""
        if not hasattr(self, '_grid_info_cache'):
            # 计算全局统计
            # 使用 total_nodes 属性（已在父类中计算）
            total_nodes = self.total_nodes

            # 计算总负载
            total_load = 0.0
            if 'bus' in self.hetero_data.x_dict:
                bus_features = self.hetero_data.x_dict['bus']
                if bus_features.size(1) > 2:  # 确保有足够的特征维度
                    total_load = bus_features[:, 2].sum().item()

            # 计算总发电量
            total_generation = 0.0
            if 'gen' in self.hetero_data.x_dict:
                gen_features = self.hetero_data.x_dict['gen']
                if gen_features.size(1) > 0:  # 确保有特征
                    total_generation = gen_features[:, 0].sum().item()

            self._grid_info_cache = {
                'total_nodes': total_nodes,
                'total_load': total_load,
                'total_generation': total_generation
            }

        return self._grid_info_cache
    
    def _update_partition_info(self):
        """更新分区信息（在每次动作后调用）"""
        # 清理缓存
        self.partition_features_cache.clear()
        self.connectivity_cache.clear()

        # 重新计算所有分区的特征
        if self.enable_features_cache and self.state_manager.current_partition is not None:
            # current_partition是一个张量，不是字典，需要获取唯一值
            partition_ids = torch.unique(self.state_manager.current_partition).cpu().tolist()
            for pid in partition_ids:
                if pid > 0:  # 忽略未分配
                    self.get_partition_features(pid)
    
    def step(self, action: Union[Tuple[int, int], AbstractDecisionContext]) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict[str, Any]]:
        """
        执行环境步骤（增强版本，支持v3.0 AbstractDecisionContext）
        
        Args:
            action: 具体动作tuple或AbstractDecisionContext
        
        Returns:
            观察、奖励、终止标志、截断标志、信息字典
        """
        # v3.0架构：将AbstractDecisionContext转换为具体动作
        if isinstance(action, AbstractDecisionContext):
            materialized_action = self.materialize_action(action)
            if materialized_action is None:
                # 无法实体化，返回无效动作
                return self.state_manager.get_observation(), -1.0, True, False, {
                    'termination_reason': 'materialize_failed',
                    'error': 'Cannot materialize AbstractDecisionContext to concrete action'
                }
            concrete_action = materialized_action
        else:
            # 传统具体动作
            concrete_action = action
        
        # 执行父类step
        obs, reward, terminated, truncated, info = super().step(concrete_action)
        
        # 如果动作成功执行，更新分区信息
        if not terminated or info.get('termination_reason') != 'invalid_action':
            self._update_partition_info()
        
        # 添加增强信息到观察中
        obs = self._enhance_observation(obs)
        
        return obs, reward, terminated, truncated, info
    
    def materialize_action(self, context: AbstractDecisionContext, 
                          similarity_threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """
        将抽象决策上下文实体化为具体的执行动作
        
        这是v3.0架构的关键方法，实现跨拓扑泛化：
        通过嵌入相似度匹配，将抽象表示转换为当前环境的具体动作。
        
        Args:
            context: 抽象决策上下文
            similarity_threshold: 相似度阈值，低于此值认为无法匹配
        
        Returns:
            (node_id, partition_id) tuple，或None如果无法匹配
        """
        try:
            # 获取当前环境的节点和分区嵌入
            current_state = self.state_manager.get_observation()
            
            # 获取节点嵌入
            # 注意：这里需要从GNN编码器获取节点嵌入
            if 'node_embeddings' in current_state:
                current_node_embeddings = current_state['node_embeddings']
            else:
                # 如果观察中没有预计算的嵌入，需要通过GNN生成
                # 这可能需要调用状态管理器或GNN编码器
                logger.warning("当前观察中缺少node_embeddings，使用简化的特征映射")
                # 简化实现：使用节点特征作为嵌入代理
                current_node_embeddings = current_state.get('x', torch.randn(self.graph.number_of_nodes(), 
                                                                           context.node_embedding.size(0)))
            
            # 获取分区嵌入（需要为所有当前分区计算嵌入）
            current_partitions = torch.unique(current_state['current_partition'])
            partition_embeddings_list = []
            
            for pid in current_partitions:
                # 获取分区特征并编码为嵌入
                partition_features = self.get_partition_features(pid.item())
                
                # 转换为标准化特征向量（与PartitionEncoder期望的格式一致）
                expected_features = [
                    'size_ratio', 'load_ratio', 'generation_ratio',
                    'internal_connectivity', 'boundary_ratio', 'power_imbalance'
                ]
                
                feature_values = []
                for key in expected_features:
                    val = partition_features.get(key, 0.0)
                    if isinstance(val, (int, float)):
                        feature_values.append(float(val))
                    elif isinstance(val, torch.Tensor):
                        feature_values.append(val.item() if val.numel() == 1 else val.mean().item())
                    else:
                        feature_values.append(0.0)
                
                feature_tensor = torch.tensor(feature_values, dtype=torch.float32, device=self.device)
                partition_embeddings_list.append(feature_tensor)
            
            if partition_embeddings_list:
                current_partition_embeddings = torch.stack(partition_embeddings_list)
            else:
                logger.error("无法获取当前分区嵌入")
                return None
            
            # 使用decision_context模块的materialize函数
            # 创建候选映射：从当前分区ID列表
            candidate_mappings = [(0, pid.item()) for pid in current_partitions]  # 简化映射
            
            result = materialize_decision_context(
                context=context,
                candidate_mappings=candidate_mappings,
                similarity_threshold=similarity_threshold
            )
            
            if result is not None:
                node_id, partition_idx = result
                # 将分区索引转换为实际的分区ID
                actual_partition_id = current_partitions[partition_idx].item()
                return (node_id, actual_partition_id)
            else:
                logger.warning(f"无法将DecisionContext实体化：相似度低于阈值{similarity_threshold}")
                return None
                
        except Exception as e:
            logger.error(f"materialize_action失败: {e}")
            return None
    
    def _enhance_observation(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        增强观察，添加额外信息
        
        Args:
            obs: 原始观察
        
        Returns:
            增强的观察
        """
        # 添加连通性安全分区信息
        connectivity_info = {}
        boundary_nodes = self.state_manager.get_boundary_nodes()
        
        for node in boundary_nodes:
            # 确保node是Python int类型，用于字典键
            node_int = int(node.item()) if torch.is_tensor(node) else int(node)
            safe_partitions = self.get_connectivity_safe_partitions(node)
            connectivity_info[node_int] = safe_partitions
        
        obs['connectivity_safe_partitions'] = connectivity_info
        
        # 添加分区特征信息
        partition_info = {}
        # current_partition是一个张量，需要获取唯一值
        partition_ids = torch.unique(self.state_manager.current_partition).cpu().tolist()
        
        for pid in partition_ids:
            if pid > 0:
                partition_info[pid] = self.get_partition_features(pid)
        
        obs['partition_info'] = partition_info
        
        # 添加网格全局信息
        obs['grid_info'] = self._get_grid_info()
        
        # v3.0架构：添加嵌入信息以支持相似度匹配
        obs = self._add_embedding_info(obs)
        
        return obs
    
    def _add_embedding_info(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        为观察添加嵌入信息，支持v3.0架构的相似度匹配
        
        Args:
            obs: 原始观察字典
            
        Returns:
            增强的观察字典，包含节点和分区嵌入
        """
        try:
            # 添加节点嵌入信息
            # 注意：这里假设我们有一个GNN编码器可以生成节点嵌入
            # 实际实现中，这可能需要调用state_manager的GNN编码功能
            
            if hasattr(self.state_manager, 'get_node_embeddings'):
                # 如果state_manager支持直接获取节点嵌入
                node_embeddings = self.state_manager.get_node_embeddings()
                obs['node_embeddings'] = node_embeddings
            else:
                # 生成标准化的节点嵌入（基于节点特征）
                node_features = obs.get('x', self.state_manager.hetero_data['bus'].x)
                if node_features.size(1) >= 64:
                    obs['node_embeddings'] = node_features[:, :64]  # 截取前64维
                else:
                    # 扩展到64维（工程级实现）
                    pad_size = 64 - node_features.size(1)
                    padding = torch.zeros(node_features.size(0), pad_size, device=node_features.device)
                    obs['node_embeddings'] = torch.cat([node_features, padding], dim=1)
            
            # 添加当前分区的嵌入信息
            current_partitions = torch.unique(obs['current_partition'])
            partition_embeddings = self._compute_partition_embeddings(current_partitions)
            obs['partition_embeddings'] = partition_embeddings
            obs['partition_ids'] = current_partitions
            
            # 添加边界节点的嵌入索引，便于快速查找
            boundary_nodes = self.state_manager.get_boundary_nodes()
            obs['boundary_node_indices'] = boundary_nodes
            
        except Exception as e:
            logger.warning(f"添加嵌入信息失败：{e}，使用简化版本")
            # 即使失败也要提供基本的嵌入信息
            num_nodes = self.graph.number_of_nodes()
            obs['node_embeddings'] = torch.randn(num_nodes, 64, device=self.device)
            obs['partition_embeddings'] = torch.randn(1, 6, device=self.device)  # 至少有一个分区
            obs['partition_ids'] = torch.tensor([1], device=self.device)
            obs['boundary_node_indices'] = torch.tensor([], dtype=torch.long, device=self.device)
        
        return obs
    
    def _compute_partition_embeddings(self, partition_ids: torch.Tensor) -> torch.Tensor:
        """
        计算分区嵌入表示
        
        Args:
            partition_ids: 分区ID张量
            
        Returns:
            分区嵌入张量 [num_partitions, embedding_dim]
        """
        embeddings = []
        
        for pid in partition_ids:
            if pid.item() == 0:  # 跳过未分配的节点
                continue
                
            # 获取分区特征
            partition_features = self.get_partition_features(pid.item())
            
            # 转换为标准化特征向量
            expected_features = [
                'size_ratio', 'load_ratio', 'generation_ratio',
                'internal_connectivity', 'boundary_ratio', 'power_imbalance'
            ]
            
            feature_values = []
            for key in expected_features:
                val = partition_features.get(key, 0.0)
                if isinstance(val, (int, float)):
                    feature_values.append(float(val))
                elif isinstance(val, torch.Tensor):
                    feature_values.append(val.item() if val.numel() == 1 else val.mean().item())
                else:
                    feature_values.append(0.0)
            
            # 确保特征长度正确
            if len(feature_values) != 6:
                feature_values = feature_values[:6] + [0.0] * (6 - len(feature_values))
            
            feature_tensor = torch.tensor(feature_values, dtype=torch.float32, device=self.device)
            embeddings.append(feature_tensor)
        
        if embeddings:
            return torch.stack(embeddings)
        else:
            # 如果没有有效分区，返回空张量
            return torch.empty((0, 6), dtype=torch.float32, device=self.device)
    
    def get_current_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取当前环境的完整嵌入信息
        
        便捷方法，用于AbstractDecisionContext的实体化过程
        
        Returns:
            (node_embeddings, partition_embeddings, partition_ids)
        """
        current_obs = self.state_manager.get_observation()
        enhanced_obs = self._add_embedding_info(current_obs)
        
        return (
            enhanced_obs['node_embeddings'],
            enhanced_obs['partition_embeddings'], 
            enhanced_obs['partition_ids']
        )
    
    def find_similar_node(self, target_embedding: torch.Tensor, 
                         threshold: float = 0.8) -> Optional[int]:
        """
        通过嵌入相似度查找最相似的节点
        
        Args:
            target_embedding: 目标节点嵌入
            threshold: 相似度阈值
            
        Returns:
            最相似的节点ID，或None如果没有满足阈值的节点
        """
        node_embeddings, _, _ = self.get_current_embeddings()
        
        similarities = torch.cosine_similarity(
            target_embedding.unsqueeze(0),
            node_embeddings,
            dim=1
        )
        
        best_similarity, best_idx = torch.max(similarities, dim=0)
        
        if best_similarity.item() >= threshold:
            return best_idx.item()
        else:
            return None
    
    def find_similar_partition(self, target_embedding: torch.Tensor,
                              threshold: float = 0.8) -> Optional[int]:
        """
        通过嵌入相似度查找最相似的分区
        
        Args:
            target_embedding: 目标分区嵌入
            threshold: 相似度阈值
            
        Returns:
            最相似的分区ID，或None如果没有满足阈值的分区
        """
        _, partition_embeddings, partition_ids = self.get_current_embeddings()
        
        if partition_embeddings.size(0) == 0:
            return None
        
        similarities = torch.cosine_similarity(
            target_embedding.unsqueeze(0),
            partition_embeddings,
            dim=1
        )
        
        best_similarity, best_idx = torch.max(similarities, dim=0)
        
        if best_similarity.item() >= threshold:
            return partition_ids[best_idx].item()
        else:
            return None
    
    def reset(self, *args, **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        重置环境（增强版本）
        """
        obs, info = super().reset(*args, **kwargs)
        
        # 重建缓存
        self._update_partition_info()
        
        # 增强观察
        obs = self._enhance_observation(obs)
        
        return obs, info
    
    def get_action_mask_with_connectivity(self) -> torch.Tensor:
        """
        获取考虑连通性约束的动作掩码
        
        Returns:
            动作掩码张量 [num_nodes, num_partitions]
        """
        num_nodes = self.hetero_data['bus'].x.size(0)
        mask = torch.zeros(num_nodes, self.num_partitions, dtype=torch.bool, device=self.device)
        
        # 只为边界节点设置掩码
        boundary_nodes = self.state_manager.get_boundary_nodes()
        
        for node in boundary_nodes:
            # 确保node是Python int类型用于索引
            node_int = int(node.item()) if torch.is_tensor(node) else int(node)
            safe_partitions = self.get_connectivity_safe_partitions(node)
            for pid in safe_partitions:
                mask[node_int, pid - 1] = True  # 转换为0-indexed
        
        return mask
    
    def visualize_partition_features(self, partition_id: int):
        """
        可视化分区特征（用于调试）
        
        Args:
            partition_id: 分区ID
        """
        features = self.get_partition_features(partition_id)
        
        print(f"\n=== Partition {partition_id} Features ===")
        print(f"Nodes: {len(features['nodes'])} ({features['size_ratio']:.2%} of total)")
        print(f"Load: {features['load']:.2f} MW ({features['load_ratio']:.2%} of total)")
        print(f"Generation: {features['generation']:.2f} MW ({features['generation_ratio']:.2%} of total)")
        print(f"Power Balance: {features['power_imbalance']:.2f}")
        print(f"Internal Connectivity: {features['internal_connectivity']:.2%}")
        print(f"Boundary Ratio: {features['boundary_ratio']:.2%}")


def create_enhanced_environment(hetero_data: HeteroData,
                               node_embeddings: Dict[str, torch.Tensor],
                               num_partitions: int,
                               config: Dict[str, Any],
                               use_enhanced: bool = True,
                               gat_encoder: Optional[Any] = None) -> Union[EnhancedPowerGridPartitioningEnv, PowerGridPartitioningEnv]:
    """
    环境创建的工厂函数

    Args:
        hetero_data: 异构图数据
        node_embeddings: 预计算的节点嵌入
        num_partitions: 分区数
        config: 配置字典
        use_enhanced: 是否创建增强环境
        gat_encoder: [v3.0] 预训练的GAT编码器实例

    Returns:
        环境实例
    """
    env_config = config.get('environment', {})
    device = torch.device(config.get('system', {}).get('device', 'cpu'))

    state_manager = StateManager(
        hetero_data=hetero_data,
        node_embeddings=node_embeddings,
        num_partitions=num_partitions,
        config=config,
        gat_encoder=gat_encoder,  # 传递给状态管理器
        device=device
    )

    action_space = ActionSpace(
        state_manager=state_manager,
        config=config
    )

    env_class = EnhancedPowerGridPartitioningEnv if use_enhanced else PowerGridPartitioningEnv
    
    env = env_class(
        state_manager=state_manager,
        action_space=action_space,
        config=env_config
    )

    # 核心修复：将GAT编码器实例附加到env对象上，以便Agent可以访问
    if gat_encoder is not None:
        env.gat_encoder = gat_encoder

    return env