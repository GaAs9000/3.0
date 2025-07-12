#!/usr/bin/env python3
"""
Scale-Aware Synthetic Power Grid Generator

生产级的多尺度电网数据生成器，支持：
- 渐进式课程学习
- 拓扑变形（节点/边的增删）
- 负荷/发电扰动
- 物理约束验证
- 高效批量生成
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from dataclasses import dataclass
import random
import copy
import logging
from collections import defaultdict
import warnings

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class GridGenerationConfig:
    """电网生成配置"""
    topology_variation_prob: float = 0.2  # 拓扑变形概率
    node_addition_prob: float = 0.1      # 节点添加概率
    node_removal_prob: float = 0.05      # 节点删除概率
    edge_addition_prob: float = 0.1      # 边添加概率
    edge_removal_prob: float = 0.05      # 边删除概率
    load_variation_range: Tuple[float, float] = (0.7, 1.3)  # 负荷变化范围
    gen_variation_range: Tuple[float, float] = (0.8, 1.2)   # 发电变化范围
    ensure_connectivity: bool = True      # 确保连通性
    min_degree: int = 2                  # 最小节点度数
    max_degree: int = 6                  # 最大节点度数
    impedance_range: Tuple[float, float] = (0.01, 0.5)  # 阻抗范围
    

class ScaleAwareSyntheticGenerator:
    """
    多尺度电网合成数据生成器
    
    核心功能：
    1. 基于真实电网案例生成变体
    2. 支持拓扑结构的智能变形
    3. 负荷和发电的真实扰动模拟
    4. 确保生成电网的物理可行性
    5. 支持渐进式课程学习策略
    """
    
    def __init__(self, 
                 base_case_loader,  # 必需参数，移除默认值
                 config: Optional[GridGenerationConfig] = None,
                 seed: int = 42,
                 device: str = 'auto'):
        """
        Args:
            base_case_loader: 基础案例加载函数（如load_power_grid_data）
            config: 生成配置
            seed: 随机种子
            device: 计算设备
        """
        # 验证base_case_loader
        if not callable(base_case_loader):
            raise TypeError("base_case_loader must be a callable function")
        
        self.base_case_loader = base_case_loader
        self.config = config or GridGenerationConfig()
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        # 设备配置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 预定义的规模分类
        self.scale_categories = {
            'small': {
                'node_range': (10, 40),
                'base_cases': ['ieee14', 'ieee30', 'ieee39'],
                'k_range': [3, 4]
            },
            'medium': {
                'node_range': (40, 100),
                'base_cases': ['ieee57', 'ieee118'],
                'k_range': [4, 6, 8]
            },
            'large': {
                'node_range': (100, 500),
                'base_cases': ['ieee118', 'ieee300'],  # ieee300需要合成
                'k_range': [8, 12, 16]
            }
        }
        
        # 缓存基础案例
        self._base_case_cache = {}
        
        # 统计信息
        self.generation_stats = defaultdict(int)
        
    def generate_batch(self, 
                      stage_config: Dict[str, float],
                      batch_size: int = 32,
                      return_k_values: bool = True) -> List[Tuple[Dict, int]]:
        """
        根据阶段配置生成一批多样化的电网数据
        
        Args:
            stage_config: 规模分布配置 {'small': 0.5, 'medium': 0.3, 'large': 0.2}
            batch_size: 批量大小
            return_k_values: 是否返回推荐的K值
            
        Returns:
            List of (grid_data, k_value) tuples
        """
        batch_data = []
        
        # 根据权重分配各规模的样本数
        for scale_category, weight in stage_config.items():
            if scale_category not in self.scale_categories:
                logger.warning(f"Unknown scale category: {scale_category}")
                continue
                
            num_samples = int(weight * batch_size)
            if num_samples == 0:
                continue
                
            # 为该规模生成样本
            for _ in range(num_samples):
                try:
                    # 生成单个样本
                    grid_data, k_value = self._generate_single_sample(scale_category)
                    
                    if return_k_values:
                        batch_data.append((grid_data, k_value))
                    else:
                        batch_data.append(grid_data)
                        
                    # 更新统计
                    self.generation_stats[scale_category] += 1
                    
                except ValueError as e:
                    logger.error(f"Value error generating sample for {scale_category}: {e}")
                    continue
                except KeyError as e:
                    logger.error(f"Key error generating sample for {scale_category}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error generating sample for {scale_category}: {type(e).__name__}: {e}")
                    continue
                    
        # 如果由于舍入导致样本不足，补充
        while len(batch_data) < batch_size:
            # 使用最大权重的类别补充
            max_category = max(stage_config.keys(), key=lambda k: stage_config[k])
            try:
                grid_data, k_value = self._generate_single_sample(max_category)
                if return_k_values:
                    batch_data.append((grid_data, k_value))
                else:
                    batch_data.append(grid_data)
            except Exception as e:
                logger.error(f"Failed to generate additional sample: {e}")
                break
                
        return batch_data[:batch_size]  # 确保不超过batch_size
    
    def _generate_single_sample(self, scale_category: str) -> Tuple[Dict, int]:
        """生成单个电网样本"""
        category_info = self.scale_categories[scale_category]
        
        # 随机选择基础案例
        base_case_name = random.choice(category_info['base_cases'])
        
        # 加载或生成基础案例
        if base_case_name.startswith('ieee') and int(base_case_name[4:]) <= 118:
            # 使用真实案例
            base_case = self._load_base_case(base_case_name)
        else:
            # 合成大规模案例
            base_case = self._synthesize_large_case(category_info['node_range'])
            
        # 应用拓扑变形
        if random.random() < self.config.topology_variation_prob:
            grid_data = self._apply_topology_variations(base_case)
        else:
            grid_data = copy.deepcopy(base_case)
            
        # 应用负荷/发电扰动
        grid_data = self._apply_load_gen_variations(grid_data)
        
        # 验证物理可行性
        if not self._validate_grid_physics(grid_data):
            logger.warning("Generated grid failed physics validation, using base case")
            grid_data = base_case
            
        # 选择合适的K值
        k_value = random.choice(category_info['k_range'])
        
        return grid_data, k_value
    
    def _load_base_case(self, case_name: str) -> Dict:
        """加载基础电网案例"""
        if case_name not in self._base_case_cache:
            self._base_case_cache[case_name] = self.base_case_loader(case_name)
                
        return copy.deepcopy(self._base_case_cache[case_name])
    
    def _synthesize_large_case(self, node_range: Tuple[int, int]) -> Dict:
        """合成大规模电网案例"""
        num_nodes = random.randint(*node_range)
        
        # 生成基础拓扑（使用改进的Barabási-Albert模型）
        G = nx.barabasi_albert_graph(num_nodes, m=3, seed=self.seed)
        
        # 确保连通性
        if not nx.is_connected(G):
            # 连接所有连通分量
            components = list(nx.connected_components(G))
            for i in range(1, len(components)):
                # 连接每个分量到主分量
                node1 = random.choice(list(components[0]))
                node2 = random.choice(list(components[i]))
                G.add_edge(node1, node2)
                
        # 转换为电网格式
        grid_data = self._graph_to_grid_format(G, num_nodes)
        
        return grid_data
    
    def _graph_to_grid_format(self, G: nx.Graph, num_nodes: int) -> Dict:
        """将NetworkX图转换为电网数据格式"""
        # 初始化bus矩阵 (与MATPOWER格式兼容)
        bus = np.zeros((num_nodes, 13))
        
        # 设置基础参数
        for i in range(num_nodes):
            bus[i, 0] = i + 1  # 节点编号
            bus[i, 1] = 1      # 节点类型 (PQ节点)
            bus[i, 2] = np.random.uniform(5, 50)   # 有功负荷 (MW)
            bus[i, 3] = bus[i, 2] * 0.4            # 无功负荷 (MVar)
            bus[i, 7] = 1.0    # 电压幅值
            bus[i, 8] = 0.0    # 电压相角
            bus[i, 9] = 345    # 基准电压 (kV)
            bus[i, 10] = 1     # 区域
            bus[i, 11] = 1.1   # 最大电压
            bus[i, 12] = 0.9   # 最小电压
            
        # 随机选择一些节点作为发电机节点
        num_gens = max(1, num_nodes // 10)
        gen_nodes = np.random.choice(num_nodes, num_gens, replace=False)
        
        # 设置发电机节点
        for gen_node in gen_nodes:
            bus[gen_node, 1] = 2  # PV节点
            
        # 设置参考节点
        bus[gen_nodes[0], 1] = 3  # Slack节点
        
        # 初始化branch矩阵
        edges = list(G.edges())
        num_branches = len(edges)
        branch = np.zeros((num_branches, 13))
        
        for idx, (i, j) in enumerate(edges):
            branch[idx, 0] = i + 1  # from bus
            branch[idx, 1] = j + 1  # to bus
            branch[idx, 2] = np.random.uniform(0.01, 0.1)   # 电阻
            branch[idx, 3] = np.random.uniform(0.01, 0.3)   # 电抗
            branch[idx, 4] = np.random.uniform(0.0, 0.02)   # 电纳
            branch[idx, 5] = 100    # MVA rating
            branch[idx, 8] = 1.0    # 变压器变比
            branch[idx, 10] = 1     # 状态 (在线)
            
        # 初始化gen矩阵
        gen = np.zeros((num_gens, 21))
        for idx, gen_node in enumerate(gen_nodes):
            gen[idx, 0] = gen_node + 1  # 发电机所在节点
            gen[idx, 1] = np.random.uniform(50, 200)   # 有功出力
            gen[idx, 2] = gen[idx, 1] * 0.4            # 无功出力
            gen[idx, 3] = gen[idx, 1] * 0.5            # 最大无功
            gen[idx, 4] = -gen[idx, 1] * 0.5           # 最小无功
            gen[idx, 5] = 1.0   # 电压设定值
            gen[idx, 6] = 100   # MVA base
            gen[idx, 7] = 1     # 状态
            gen[idx, 8] = gen[idx, 1] * 1.2  # 最大有功
            gen[idx, 9] = 0     # 最小有功
            
        return {
            'bus': bus,
            'branch': branch,
            'gen': gen,
            'baseMVA': 100.0,
            'version': '2'
        }
    
    def _apply_topology_variations(self, base_case: Dict) -> Dict:
        """应用拓扑变形"""
        grid_data = copy.deepcopy(base_case)
        
        # 构建NetworkX图便于操作
        G = self._grid_to_networkx(grid_data)
        original_nodes = set(G.nodes())
        
        # 节点操作
        if random.random() < self.config.node_addition_prob:
            G, grid_data = self._add_nodes(G, grid_data)
            
        if random.random() < self.config.node_removal_prob and len(G.nodes()) > 10:
            G, grid_data = self._remove_nodes(G, grid_data)
            
        # 边操作
        if random.random() < self.config.edge_addition_prob:
            G, grid_data = self._add_edges(G, grid_data)
            
        if random.random() < self.config.edge_removal_prob:
            G, grid_data = self._remove_edges(G, grid_data)
            
        # 确保连通性
        if self.config.ensure_connectivity and not nx.is_connected(G):
            # 恢复连通性
            grid_data = self._restore_connectivity(G, grid_data, original_nodes)
            
        return grid_data
    
    def _grid_to_networkx(self, grid_data: Dict) -> nx.Graph:
        """将电网数据转换为NetworkX图"""
        G = nx.Graph()
        
        # 添加节点
        bus_data = grid_data['bus']
        for i in range(len(bus_data)):
            G.add_node(int(bus_data[i, 0]))
            
        # 添加边
        branch_data = grid_data['branch']
        for i in range(len(branch_data)):
            if branch_data[i, 10] == 1:  # 只添加在线的支路
                G.add_edge(int(branch_data[i, 0]), int(branch_data[i, 1]))
                
        return G
    
    def _add_nodes(self, G: nx.Graph, grid_data: Dict) -> Tuple[nx.Graph, Dict]:
        """添加新节点"""
        num_new_nodes = random.randint(1, 3)
        bus_data = grid_data['bus']
        branch_data = grid_data['branch']
        
        for _ in range(num_new_nodes):
            # 新节点编号
            new_node_id = int(bus_data[:, 0].max()) + 1
            
            # 创建新节点数据
            new_bus = np.zeros((1, bus_data.shape[1]))
            new_bus[0, 0] = new_node_id
            new_bus[0, 1] = 1  # PQ节点
            new_bus[0, 2] = np.random.uniform(5, 30)    # 负荷
            new_bus[0, 3] = new_bus[0, 2] * 0.4
            new_bus[0, 7:13] = bus_data[0, 7:13]  # 复制电压限制等
            
            # 添加到bus矩阵
            grid_data['bus'] = np.vstack([bus_data, new_bus])
            bus_data = grid_data['bus']
            
            # 连接到现有节点
            G.add_node(new_node_id)
            
            # 选择连接目标（优先连接度数较高的节点）
            node_degrees = dict(G.degree())
            if node_degrees:
                # 按度数加权选择
                nodes = list(node_degrees.keys())
                weights = [d + 1 for n, d in node_degrees.items() if n != new_node_id]
                if nodes and weights:
                    target_nodes = random.choices(nodes[:-1], weights=weights, 
                                                k=min(3, len(nodes)-1))
                    
                    for target in target_nodes:
                        # 创建新支路
                        new_branch = np.zeros((1, branch_data.shape[1]))
                        new_branch[0, 0] = new_node_id
                        new_branch[0, 1] = target
                        new_branch[0, 2] = np.random.uniform(0.01, 0.1)
                        new_branch[0, 3] = np.random.uniform(0.01, 0.3)
                        new_branch[0, 5] = 100
                        new_branch[0, 8] = 1.0
                        new_branch[0, 10] = 1
                        
                        grid_data['branch'] = np.vstack([branch_data, new_branch])
                        branch_data = grid_data['branch']
                        
                        G.add_edge(new_node_id, target)
                        
        return G, grid_data
    
    def _remove_nodes(self, G: nx.Graph, grid_data: Dict) -> Tuple[nx.Graph, Dict]:
        """移除节点（保持连通性）"""
        # 找出可以安全移除的节点（移除后图仍连通）
        removable_nodes = []
        for node in G.nodes():
            if len(G.nodes()) <= 10:  # 保持最小规模
                break
            G_temp = G.copy()
            G_temp.remove_node(node)
            if nx.is_connected(G_temp):
                removable_nodes.append(node)
                
        if not removable_nodes:
            return G, grid_data
            
        # 随机选择要移除的节点
        num_remove = min(len(removable_nodes), random.randint(1, 2))
        nodes_to_remove = random.sample(removable_nodes, num_remove)
        
        for node in nodes_to_remove:
            # 从图中移除
            G.remove_node(node)
            
            # 从电网数据中移除
            bus_mask = grid_data['bus'][:, 0] != node
            grid_data['bus'] = grid_data['bus'][bus_mask]
            
            # 移除相关支路
            branch_mask = (grid_data['branch'][:, 0] != node) & \
                         (grid_data['branch'][:, 1] != node)
            grid_data['branch'] = grid_data['branch'][branch_mask]
            
            # 移除相关发电机
            if 'gen' in grid_data:
                gen_mask = grid_data['gen'][:, 0] != node
                grid_data['gen'] = grid_data['gen'][gen_mask]
                
        return G, grid_data
    
    def _add_edges(self, G: nx.Graph, grid_data: Dict) -> Tuple[nx.Graph, Dict]:
        """添加新边（避免平行线路）"""
        num_new_edges = random.randint(1, 3)
        branch_data = grid_data['branch']
        
        for _ in range(num_new_edges):
            # 找出没有直接连接的节点对
            non_edges = list(nx.non_edges(G))
            if not non_edges:
                break
                
            # 优先连接度数较低的节点
            node_degrees = dict(G.degree())
            
            # 计算每条潜在边的权重（度数越低权重越高）
            edge_weights = []
            for u, v in non_edges:
                weight = 1.0 / (node_degrees[u] + node_degrees[v] + 1)
                edge_weights.append(weight)
                
            # 按权重选择
            if edge_weights:
                idx = random.choices(range(len(non_edges)), 
                                   weights=edge_weights, k=1)[0]
                u, v = non_edges[idx]
                
                # 添加到图
                G.add_edge(u, v)
                
                # 添加到支路数据
                new_branch = np.zeros((1, branch_data.shape[1]))
                new_branch[0, 0] = u
                new_branch[0, 1] = v
                new_branch[0, 2] = np.random.uniform(0.01, 0.1)
                new_branch[0, 3] = np.random.uniform(0.01, 0.3)
                new_branch[0, 5] = 100
                new_branch[0, 8] = 1.0
                new_branch[0, 10] = 1
                
                grid_data['branch'] = np.vstack([branch_data, new_branch])
                branch_data = grid_data['branch']
                
        return G, grid_data
    
    def _remove_edges(self, G: nx.Graph, grid_data: Dict) -> Tuple[nx.Graph, Dict]:
        """移除边（保持连通性）"""
        # 找出可以安全移除的边
        removable_edges = []
        for edge in G.edges():
            G_temp = G.copy()
            G_temp.remove_edge(*edge)
            if nx.is_connected(G_temp):
                removable_edges.append(edge)
                
        if not removable_edges:
            return G, grid_data
            
        # 随机选择要移除的边
        num_remove = min(len(removable_edges), random.randint(1, 2))
        edges_to_remove = random.sample(removable_edges, num_remove)
        
        for u, v in edges_to_remove:
            # 从图中移除
            G.remove_edge(u, v)
            
            # 从支路数据中移除
            branch_mask = ~((grid_data['branch'][:, 0] == u) & 
                          (grid_data['branch'][:, 1] == v) |
                          (grid_data['branch'][:, 0] == v) & 
                          (grid_data['branch'][:, 1] == u))
            grid_data['branch'] = grid_data['branch'][branch_mask]
            
        return G, grid_data
    
    def _restore_connectivity(self, G: nx.Graph, grid_data: Dict, 
                            original_nodes: set) -> Dict:
        """恢复图的连通性"""
        if nx.is_connected(G):
            return grid_data
            
        # 获取所有连通分量
        components = list(nx.connected_components(G))
        
        # 找出最大的分量作为主分量
        main_component = max(components, key=len)
        
        branch_data = grid_data['branch']
        
        # 连接其他分量到主分量
        for component in components:
            if component == main_component:
                continue
                
            # 找出最近的节点对
            min_dist = float('inf')
            best_pair = None
            
            for u in main_component:
                for v in component:
                    # 简单使用节点编号差作为"距离"
                    dist = abs(u - v)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (u, v)
                        
            if best_pair:
                u, v = best_pair
                G.add_edge(u, v)
                
                # 添加支路
                new_branch = np.zeros((1, branch_data.shape[1]))
                new_branch[0, 0] = u
                new_branch[0, 1] = v
                new_branch[0, 2] = np.random.uniform(0.02, 0.15)  # 稍大的阻抗
                new_branch[0, 3] = np.random.uniform(0.02, 0.4)
                new_branch[0, 5] = 100
                new_branch[0, 8] = 1.0
                new_branch[0, 10] = 1
                
                grid_data['branch'] = np.vstack([branch_data, new_branch])
                branch_data = grid_data['branch']
                
        return grid_data
    
    def _apply_load_gen_variations(self, grid_data: Dict) -> Dict:
        """应用负荷和发电扰动"""
        # 负荷扰动
        load_scale = np.random.uniform(*self.config.load_variation_range, 
                                     size=len(grid_data['bus']))
        grid_data['bus'][:, 2] *= load_scale  # PD
        grid_data['bus'][:, 3] *= load_scale  # QD
        
        # 发电扰动
        if 'gen' in grid_data and len(grid_data['gen']) > 0:
            gen_scale = np.random.uniform(*self.config.gen_variation_range, 
                                        size=len(grid_data['gen']))
            grid_data['gen'][:, 1] *= gen_scale  # PG
            
            # 确保发电满足负荷（简单调整）
            total_load = grid_data['bus'][:, 2].sum()
            total_gen = grid_data['gen'][:, 1].sum()
            
            if total_gen < total_load * 1.1:  # 留10%裕度
                # 等比例增加所有发电
                scale_factor = (total_load * 1.15) / total_gen
                grid_data['gen'][:, 1] *= scale_factor
                
        return grid_data
    
    def _validate_grid_physics(self, grid_data: Dict) -> bool:
        """验证电网物理可行性"""
        try:
            # 检查基本完整性
            if 'bus' not in grid_data or len(grid_data['bus']) == 0:
                return False
            if 'branch' not in grid_data or len(grid_data['branch']) == 0:
                return False
                
            # 检查节点编号连续性
            bus_ids = set(grid_data['bus'][:, 0].astype(int))
            
            # 检查支路连接的节点是否都存在
            for branch in grid_data['branch']:
                if branch[10] == 1:  # 在线支路
                    if int(branch[0]) not in bus_ids or int(branch[1]) not in bus_ids:
                        return False
                        
            # 检查至少有一个发电机
            if 'gen' not in grid_data or len(grid_data['gen']) == 0:
                # 如果没有发电机，添加一个
                gen = np.zeros((1, 21))
                gen[0, 0] = grid_data['bus'][0, 0]  # 第一个节点
                gen[0, 1] = grid_data['bus'][:, 2].sum() * 1.2  # 总负荷的120%
                gen[0, 7] = 1
                grid_data['gen'] = gen
                
            # 检查功率平衡（粗略）
            total_load = grid_data['bus'][:, 2].sum()
            total_gen_capacity = grid_data['gen'][:, 8].sum() if grid_data['gen'].shape[1] > 8 else grid_data['gen'][:, 1].sum() * 1.5
            
            if total_gen_capacity < total_load:
                return False
                
            # 检查参数范围
            if np.any(grid_data['branch'][:, 2] <= 0) or np.any(grid_data['branch'][:, 3] <= 0):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Physics validation error: {e}")
            return False
    
    def get_progressive_schedule(self, episode_num: int, 
                               total_episodes: int) -> Dict[str, float]:
        """
        获取渐进式训练计划
        
        Args:
            episode_num: 当前episode
            total_episodes: 总episode数
            
        Returns:
            规模分布字典
        """
        progress = episode_num / total_episodes
        
        if progress < 0.2:  # 前20%：专注小规模
            return {'small': 0.8, 'medium': 0.2, 'large': 0.0}
        elif progress < 0.5:  # 20%-50%：引入中等规模
            return {'small': 0.5, 'medium': 0.4, 'large': 0.1}
        elif progress < 0.8:  # 50%-80%：平衡发展
            return {'small': 0.3, 'medium': 0.4, 'large': 0.3}
        else:  # 最后20%：专注大规模
            return {'small': 0.2, 'medium': 0.3, 'large': 0.5}
    
    def get_k_value_schedule(self, scale_category: str, 
                           episode_num: int,
                           total_episodes: int) -> List[int]:
        """
        获取K值的渐进式计划
        
        Args:
            scale_category: 规模类别
            episode_num: 当前episode
            total_episodes: 总episode数
            
        Returns:
            K值候选列表
        """
        base_k_range = self.scale_categories[scale_category]['k_range']
        progress = episode_num / total_episodes
        
        if progress < 0.3:  # 早期：使用较小的K值
            return base_k_range[:len(base_k_range)//2 + 1]
        else:  # 后期：使用完整K值范围
            return base_k_range
    
    def print_statistics(self):
        """打印生成统计信息"""
        print("\n=== Generation Statistics ===")
        total = sum(self.generation_stats.values())
        for category, count in self.generation_stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{category}: {count} ({percentage:.1f}%)")
        print(f"Total: {total}")
