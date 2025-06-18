import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import silhouette_score
from torch_geometric.data import Data
from collections import deque, defaultdict
import random
import copy
from torch_scatter import scatter_add, scatter_mean

@dataclass
class PartitionMetrics:
    """分区评估指标数据类"""
    load_cv: float          # 负荷变异系数
    load_gini: float        # 负荷基尼系数
    total_coupling: float   # 总耦合度
    inter_region_lines: int # 跨区域线路数
    connectivity: float     # 连通性得分
    power_balance: float    # 功率平衡度
    efficiency: float       # 分区效率
    modularity: float       # 模块度


class PowerGridPartitionEnv:
    """
    电网分区强化学习环境
    
    实现特点：
    1. 多目标奖励函数（负荷均衡、解耦、连通性、功率平衡等）
    2. 物理约束检查（功率平衡、电压限制、N-1安全）
    3. 增量式分区（每步分配一个边界节点）
    4. 自适应奖励标准化
    5. 详细的指标追踪
    """
    
    def __init__(self, data: Data, embeddings: torch.Tensor, K: int = None,
                 device: str = 'cpu', enable_physics_constraints: bool = True,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        初始化环境
        
        参数:
            data: PyG数据对象
            embeddings: GAT编码后的节点嵌入
            K: 目标分区数
            device: 计算设备
            enable_physics_constraints: 是否启用物理约束
            reward_weights: 奖励权重字典
        """
        # 导入配置
        try:
            from config import NUM_REGIONS
            if K is None:
                K = NUM_REGIONS
        except ImportError:
            if K is None:
                K = 3  # 默认分区数量
        
        self.device = torch.device(device)
        self.data = data.to(self.device)
        self.embeddings = embeddings.to(self.device)
        self.K = K
        self.N = data.num_nodes
        self.enable_physics_constraints = enable_physics_constraints
        
        # 提取节点特征
        self.Pd = data.x[:, 0]  # 有功负荷
        self.Qd = data.x[:, 1]  # 无功负荷
        
        # 检查是否有发电数据
        if data.x.shape[1] > 8:  # 有发电机特征
            self.Pg = data.x[:, 8]   # 有功发电
            self.Qg = data.x[:, 9]   # 无功发电
            self.is_gen = data.x[:, 10] > 0.5
        else:
            self.Pg = torch.zeros_like(self.Pd)
            self.Qg = torch.zeros_like(self.Qd)
            self.is_gen = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        
        # 提取边特征
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        
        # 计算导纳（用于耦合度计算） - 直接使用预先计算好的导纳特征（第4列）
        self.admittance = self.edge_attr[:, 4]
        
        # 处理导纳中的NaN和inf值
        nan_mask = torch.isnan(self.admittance) | torch.isinf(self.admittance)
        if nan_mask.any():
            print(f"⚠️ 警告：发现 {nan_mask.sum().item()} 个异常导纳值，进行修复...")
            self.admittance = torch.where(nan_mask, torch.tensor(1e-10, device=self.device), self.admittance)
        
        # 构建邻接表（用于快速查询）
        self.adj_list = self._build_adjacency_list()
        
        # 预计算平均负载和度，用于奖励函数
        self.avg_load = self.Pd.mean().item()
        self.avg_degree = np.mean([len(adj) for adj in self.adj_list.values()])
        
        # 奖励权重（修改以强化平衡性）
        default_weights = {
            'balance': 0.4,      # 负荷均衡（提高权重）
            'coupling': 0.2,     # 解耦度
            'connectivity': 0.15, # 连通性
            'power_balance': 0.1, # 功率平衡
            'progress': 0.1,     # 进度奖励
            'efficiency': 0.05   # 效率奖励
        }
        self.reward_weights = reward_weights or default_weights
        
        # 奖励标准化参数（在线更新）
        self.reward_stats = {
            key: {'mean': 0.0, 'std': 1.0, 'count': 0}
            for key in ['cv', 'coupling', 'connectivity', 'balance']
        }
        
        # 动作历史（用于分析）
        self.action_history = []
        self.reward_history = []
        self.metrics_history = []
        
        # 重置环境
        self.reset()
    
    def _build_adjacency_list(self) -> Dict[int, List[int]]:
        """构建邻接表加速邻居查询"""
        adj_list = defaultdict(list)
        edge_array = self.edge_index.cpu().numpy()
        
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            adj_list[u].append(v)
        
        return dict(adj_list)
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """
        重置环境到初始状态
        
        初始化策略：
        1. 随机选择K个种子节点（考虑节点重要性）
        2. 种子节点优先选择发电节点或高负荷节点
        3. 种子节点之间保持一定距离
        """
        self.t = 0  # 时间步
        self.z = torch.zeros(self.N, dtype=torch.long, device=self.device)  # 节点分配
        
        # 智能选择种子节点
        seed_nodes = self._select_seed_nodes()
        for k, node in enumerate(seed_nodes):
            self.z[node] = k + 1
        
        # 更新环境状态
        self._update_state()
        
        # 清空历史
        self.action_history.clear()
        self.reward_history.clear()
        self.metrics_history.clear()
        
        return self.get_state()
    
    def _select_seed_nodes(self) -> List[int]:
        """
        智能选择种子节点
        
        策略：
        1. 优先选择发电节点
        2. 其次选择高负荷节点
        3. 确保种子之间有一定距离
        """
        # 计算节点重要性得分
        importance = self.Pd + self.Pg * 2.0  # 发电节点权重更高
        
        # 如果有发电节点，优先选择
        gen_nodes = torch.where(self.is_gen)[0]
        
        seeds = []
        
        # 首先从发电节点中选择
        if len(gen_nodes) > 0:
            # 选择最重要的发电节点
            gen_importance = importance[gen_nodes]
            first_gen = gen_nodes[torch.argmax(gen_importance)]
            seeds.append(first_gen.item())
        
        # 然后选择其他种子，确保分散
        while len(seeds) < self.K:
            if len(seeds) == 0:
                # 如果没有发电节点，选择最高负荷节点
                next_seed = torch.argmax(importance).item()
            else:
                # 计算到已有种子的最小距离
                min_distances = torch.full((self.N,), float('inf'), device=self.device)
                
                for seed in seeds:
                    distances = self._compute_graph_distances(seed)
                    min_distances = torch.minimum(min_distances, distances)
                
                # 排除已选种子
                for seed in seeds:
                    min_distances[seed] = -1
                
                # 选择距离最远且重要性高的节点
                scores = min_distances * importance
                next_seed = torch.argmax(scores).item()
            
            seeds.append(next_seed)
        
        return seeds[:self.K]
    
    def _compute_graph_distances(self, source: int) -> torch.Tensor:
        """计算从源节点到所有节点的图距离（BFS）"""
        distances = torch.full((self.N,), float('inf'), device=self.device)
        distances[source] = 0
        
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            if u in self.adj_list:
                for v in self.adj_list[u]:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        
        return distances
    
    def _update_state(self):
        """更新环境内部状态"""
        # 更新边界节点集合
        self.boundary_nodes = self._get_boundary_nodes()
        
        # 更新区域嵌入
        self.region_embeddings = self._compute_region_embeddings()
        
        # 更新全局上下文
        self.global_context = self._compute_global_context()
        
        # 计算当前指标
        self.current_metrics = self._compute_metrics()
    
    def _get_boundary_nodes(self) -> torch.Tensor:
        """获取边界节点（未分配但邻接已分配节点）"""
        boundary_set = set()
        
        # 遍历所有未分配节点
        unassigned = torch.where(self.z == 0)[0]
        
        for node in unassigned:
            node_idx = node.item()
            if node_idx in self.adj_list:
                # 检查是否有已分配的邻居
                for neighbor in self.adj_list[node_idx]:
                    if self.z[neighbor] > 0:
                        boundary_set.add(node_idx)
                        break
        
        return torch.tensor(list(boundary_set), dtype=torch.long, device=self.device)
    
    def _compute_region_embeddings(self) -> torch.Tensor:
        """
        计算区域嵌入（聚合区域内节点的嵌入）
        
        使用加权平均，权重基于节点重要性（负荷+发电）
        """
        region_embs = []
        
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            
            if mask.any():
                # 向量化的加权平均
                node_weights = self.Pd[mask] + self.Pg[mask] + 1e-6
                node_weights = node_weights / node_weights.sum()
                
                # 使用einsum进行高效的加权求和
                weighted_emb = torch.einsum('n,nd->d', node_weights, self.embeddings[mask])
                region_embs.append(weighted_emb)
            else:
                region_embs.append(torch.zeros_like(self.embeddings[0]))
        
        return torch.stack(region_embs)
    
    def _compute_global_context(self) -> torch.Tensor:
        """
        计算全局上下文向量
        
        包含：
        1. 已分配节点的平均嵌入
        2. 分区进度
        3. 当前负荷分布统计
        4. 平衡度指标
        """
        # 已分配节点的平均嵌入
        assigned_mask = (self.z > 0)
        if assigned_mask.any():
            mean_embedding = self.embeddings[assigned_mask].mean(dim=0)
        else:
            mean_embedding = torch.zeros_like(self.embeddings[0])
        
        # 分区进度
        progress = assigned_mask.float().mean()
        
        # 各区域负荷比例
        region_loads = []
        total_load = self.Pd.sum()
        
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            if mask.any():
                region_load = self.Pd[mask].sum() / (total_load + 1e-10)
            else:
                region_load = 0.0
            region_loads.append(region_load)
        
        # 当前平衡度（负的变异系数）
        if len(region_loads) > 0 and sum(region_loads) > 0:
            loads_tensor = torch.tensor(region_loads, device=self.device)
            balance_score = -loads_tensor.std() / (loads_tensor.mean() + 1e-10)
        else:
            balance_score = torch.tensor(0.0, device=self.device)
        
        # 组合所有信息
        context = torch.cat([
            mean_embedding,
            progress.unsqueeze(0),
            torch.tensor(region_loads, device=self.device),
            balance_score.unsqueeze(0)
        ])
        
        return context
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """获取当前状态字典"""
        return {
            'node_embeddings': self.embeddings,
            'z': self.z,
            'region_embeddings': self.region_embeddings,
            'global_context': self.global_context,
            'boundary_nodes': self.boundary_nodes,
            't': self.t,
            'node_features': self.data.x,  # 原始特征
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr
        }
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        获取当前有效动作列表
        
        有效动作：将边界节点分配到其邻接的区域
        如果启用物理约束，还需要通过约束检查
        
        修改：优先考虑较小的区域，避免极端不平衡
        """
        valid_actions = []
        
        # 计算各区域当前大小
        region_sizes = {}
        for k in range(1, self.K + 1):
            region_sizes[k] = (self.z == k).sum().item()
        
        # 计算目标平衡大小
        target_size = self.N // self.K
        max_size = target_size * 1.5  # 允许最大为目标大小的1.5倍
        
        for node in self.boundary_nodes:
            node_idx = node.item()
            
            # 获取该节点可以加入的区域
            valid_regions = set()
            
            if node_idx in self.adj_list:
                for neighbor in self.adj_list[node_idx]:
                    if self.z[neighbor] > 0:
                        region = self.z[neighbor].item()
                        # 检查区域大小限制
                        if region_sizes[region] < max_size:
                            valid_regions.add(region)
            
            # 如果没有邻接区域可加入，允许加入最小的区域
            if not valid_regions:
                min_region = min(region_sizes.keys(), key=lambda k: region_sizes[k])
                if region_sizes[min_region] < max_size:
                    valid_regions.add(min_region)
            
            # 检查每个可能的分配
            for region in valid_regions:
                # 物理约束检查
                if self.enable_physics_constraints:
                    if self._check_constraints(node_idx, region):
                        valid_actions.append((node_idx, region))
                else:
                    valid_actions.append((node_idx, region))
        
        return valid_actions
    
    def _check_constraints(self, node: int, region: int) -> bool:
        """
        检查物理约束
        
        包括:
        1. 功率平衡约束
        2. 区域大小约束
        3. 连通性约束
        """
        # 1. 检查区域大小（避免极端不平衡）
        current_size = (self.z == region).sum().item()
        max_size = self.N // self.K * 2  # 最大为平均大小的2倍
        if current_size >= max_size:
            return False
        
        # 2. 检查功率平衡（简化版）
        # 计算加入节点后的功率不平衡度
        mask = (self.z == region)
        future_mask = mask.clone()
        future_mask[node] = True
        
        future_p_gen = self.Pg[future_mask].sum()
        future_p_load = self.Pd[future_mask].sum()
        
        # 允许一定的不平衡（考虑区域间传输）
        imbalance_ratio = abs(future_p_gen - future_p_load) / (future_p_load + 1e-10)
        if imbalance_ratio > 0.5:  # 不平衡度超过50%
            return False
        
        # 3. 其他约束可以在这里添加
        
        return True
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作
        
        参数:
            action: (节点索引, 区域编号)
            
        返回:
            next_state: 下一状态
            reward: 即时奖励
            done: 是否结束
            info: 额外信息
        """
        node_idx, region = action
        
        # 记录动作前的指标
        prev_metrics = copy.deepcopy(self.current_metrics)
        
        # 执行动作
        self.z[node_idx] = region
        self.t += 1
        
        # 更新状态
        self._update_state()
        
        # 计算奖励
        reward = self._compute_reward(action, prev_metrics)
        
        # 检查是否完成
        done = (self.z == 0).sum() == 0
        
        # 记录历史
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.metrics_history.append(self.current_metrics)
        
        # 额外信息
        info = {
            'metrics': self.current_metrics,
            'prev_metrics': prev_metrics,
            'num_assigned': (self.z > 0).sum().item(),
            'num_boundary': len(self.boundary_nodes)
        }
        
        # 如果完成，计算最终奖励
        if done:
            final_bonus = self._compute_final_bonus()
            reward += final_bonus
            info['final_bonus'] = final_bonus
        
        return self.get_state(), reward, done, info
    
    def _compute_reward(self, action: Tuple[int, int], prev_metrics: PartitionMetrics) -> float:
        """
        计算即时奖励 - 结合稠密奖励和硬约束
        """
        node_idx, region = action
        curr_metrics = self.current_metrics
        
        rewards = {}
        
        # 1. 负荷均衡改进奖励
        cv_improvement = prev_metrics.load_cv - curr_metrics.load_cv
        rewards['balance'] = self._normalize_reward('cv', cv_improvement)
        
        # 2. 耦合度奖励
        coupling_change = curr_metrics.total_coupling - prev_metrics.total_coupling
        rewards['coupling'] = -self._normalize_reward('coupling', coupling_change)
        
        # 3. 连通性奖励
        # 这里的奖励是软引导，真正的约束在函数末尾的硬惩罚
        rewards['connectivity'] = curr_metrics.connectivity
        
        # 4. 邻居一致性奖励
        rewards['neighbor_consistency'] = self._compute_neighbor_consistency_reward(node_idx, region)
        
        # 5. 区域大小平衡奖励
        rewards['size_balance'] = self._compute_size_balance_reward(region)
        
        # 6. 关键节点奖励
        rewards['node_importance'] = self._compute_node_importance_reward(node_idx)

        # 加权求和
        total_reward = sum(
            self.reward_weights.get(key, 0) * value 
            for key, value in rewards.items()
        )
        
        # 强制连通性硬约束
        if curr_metrics.connectivity < 1.0:
            # 当出现不连通区域时，施加一个巨大的惩罚
            # 惩罚力度与不连通区域的数量成正比
            num_disconnected_regions = self.K * (1.0 - curr_metrics.connectivity)
            total_reward -= 1.0 * num_disconnected_regions  # 核心硬约束

        # 记录各分量用于调试
        self.reward_components = rewards

        return total_reward
    
    def _compute_neighbor_consistency_reward(self, node_idx: int, region: int) -> float:
        """计算邻居一致性奖励"""
        if node_idx not in self.adj_list:
            return 0.0
        
        neighbors = self.adj_list[node_idx]
        same_region_count = 0
        assigned_neighbors = 0
        
        for neighbor in neighbors:
            if self.z[neighbor] > 0:
                assigned_neighbors += 1
                if self.z[neighbor] == region:
                    same_region_count += 1
        
        if assigned_neighbors == 0:
            return 0.0 # 没有已分配的邻居，无所谓一致性
        
        consistency_ratio = same_region_count / assigned_neighbors
        # 将比例从[0, 1]映射到[-1, 1]，鼓励高一致性，惩罚不一致
        return consistency_ratio * 2 - 1

    def _compute_size_balance_reward(self, region: int) -> float:
        """计算区域大小平衡奖励，避免分区大小差异过大"""
        target_size = self.N / self.K
        current_size = (self.z == region).sum().item() + 1 # 加上即将分配的节点
        
        # 计算与理想大小的偏离度
        deviation = abs(current_size - target_size) / target_size
        
        # 使用指数衰减函数，偏离越小奖励越高
        return np.exp(-deviation * 3) # 乘3使惩罚更敏感

    def _compute_node_importance_reward(self, node_idx: int) -> float:
        """计算分配关键节点的奖励"""
        # 基于负荷和度数的综合重要性
        node_load = self.Pd[node_idx].item()
        node_degree = len(self.adj_list.get(node_idx, []))
        
        # 使用环境的平均值进行标准化
        avg_load = self.avg_load
        avg_degree = self.avg_degree
        
        importance = (node_load / (avg_load + 1e-10)) * 0.7 + \
                     (node_degree / (avg_degree + 1e-10)) * 0.3
        
        # 使用tanh将奖励中心化并限制在[-1, 1]
        return np.tanh(importance - 1.0)

    def _normalize_reward(self, key: str, value: float) -> float:
        """
        在线标准化奖励分量
        
        使用running statistics避免奖励scale问题
        """
        stats = self.reward_stats[key]
        
        # 更新统计量（指数移动平均）
        alpha = 0.01
        stats['mean'] = (1 - alpha) * stats['mean'] + alpha * value
        stats['std'] = (1 - alpha) * stats['std'] + alpha * abs(value - stats['mean'])
        stats['count'] += 1
        
        # 标准化
        if stats['count'] > 10 and stats['std'] > 1e-6:
            normalized = (value - stats['mean']) / stats['std']
        else:
            normalized = value
        
        # 使用tanh限制范围
        return np.tanh(normalized)
    
    def _compute_final_bonus(self) -> float:
        """
        计算完成分区后的最终奖励
        
        基于整体分区质量给予额外奖励
        """
        metrics = self.current_metrics
        
        # 综合评分
        quality_score = 0.0
        
        # 负荷均衡度（CV越小越好）
        if metrics.load_cv < 0.1:
            quality_score += 1.0
        elif metrics.load_cv < 0.2:
            quality_score += 0.5
        
        # 耦合度（越小越好）
        avg_coupling = metrics.total_coupling / max(metrics.inter_region_lines, 1)
        if avg_coupling < 0.5:
            quality_score += 0.5
        
        # 连通性（必须全部连通）
        if metrics.connectivity == 1.0:
            quality_score += 0.5
        
        # 功率平衡
        if metrics.power_balance < 0.1:
            quality_score += 0.5
        
        return quality_score
    
    def _compute_metrics(self) -> PartitionMetrics:
        """
        计算当前分区的所有评估指标 - 向量化版本
        """
        # GPU向量化计算负荷分布
        region_ids = self.z - 1  # 转换为0-based
        valid_mask = region_ids >= 0
        
        if valid_mask.any():
            # 使用scatter_add进行向量化聚合
            region_loads = scatter_add(self.Pd[valid_mask], region_ids[valid_mask],
                                     dim=0, dim_size=self.K)
            region_gens = scatter_add(self.Pg[valid_mask], region_ids[valid_mask],
                                    dim=0, dim_size=self.K)
            
            # 转换为numpy进行统计计算
            region_loads_np = region_loads.cpu().numpy()
            region_gens_np = region_gens.cpu().numpy()
            
            # 负荷均衡指标
            if region_loads_np.sum() > 0:
                load_cv = np.std(region_loads_np) / (np.mean(region_loads_np) + 1e-10)
                load_gini = self._compute_gini(region_loads_np)
            else:
                load_cv = 1.0
                load_gini = 1.0
        else:
            load_cv = 1.0
            load_gini = 1.0
            region_loads_np = np.array([])
            region_gens_np = np.array([])
        
        # 向量化计算耦合度
        edge_array = self.edge_index
        z_u = self.z[edge_array[0]]
        z_v = self.z[edge_array[1]]
        inter_mask = (z_u != z_v) & (z_u > 0) & (z_v > 0)
        
        inter_region_lines = int(inter_mask.sum().item())
        if inter_region_lines > 0:
            total_coupling = self.admittance[inter_mask].sum().item()
        else:
            total_coupling = 0.0
        
        # 连通性指标
        connectivity = self._check_connectivity()
        
        # 功率平衡指标
        power_balance = 0.0
        if len(region_loads_np) > 0:
            for i, (load, gen) in enumerate(zip(region_loads_np, region_gens_np)):
                if load > 0:
                    imbalance = abs(gen - load) / load
                    power_balance += imbalance
            power_balance /= self.K
        
        # 分区效率
        assigned_ratio = (self.z > 0).float().mean().item()
        efficiency = assigned_ratio
        
        # 模块度
        modularity = self._compute_modularity_vectorized()
        
        return PartitionMetrics(
            load_cv=load_cv,
            load_gini=load_gini,
            total_coupling=total_coupling,
            inter_region_lines=inter_region_lines,
            connectivity=connectivity,
            power_balance=power_balance,
            efficiency=efficiency,
            modularity=modularity
        )
    
    def _compute_gini(self, values: np.ndarray) -> float:
        """计算基尼系数"""
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        
        return (2 * index @ sorted_values) / (n * sorted_values.sum()) - (n + 1) / n
    
    def _check_connectivity(self) -> float:
        """
        检查各区域的连通性
        
        返回连通区域的比例
        """
        connected_regions = 0
        total_regions = 0
        
        for k in range(1, self.K + 1):
            mask = (self.z == k)
            if mask.sum() > 0:
                total_regions += 1
                
                # 使用BFS检查连通性
                region_nodes = torch.where(mask)[0].cpu().tolist()
                if self._is_connected_subgraph(region_nodes):
                    connected_regions += 1
        
        return connected_regions / max(total_regions, 1)
    
    def _is_connected_subgraph(self, nodes: List[int]) -> bool:
        """检查子图是否连通"""
        if len(nodes) <= 1:
            return True
        
        # BFS
        visited = {nodes[0]}
        queue = deque([nodes[0]])
        node_set = set(nodes)
        
        while queue:
            u = queue.popleft()
            
            if u in self.adj_list:
                for v in self.adj_list[u]:
                    if v in node_set and v not in visited:
                        visited.add(v)
                        queue.append(v)
        
        return len(visited) == len(nodes)
    
    def _compute_modularity_vectorized(self) -> float:
        """
        向量化计算网络模块度
        
        Q = 1/(2m) * Σ(A_ij - k_i*k_j/(2m)) * δ(c_i, c_j)
        """
        # 简化计算：基于边的比例
        total_edges = self.edge_index.shape[1] // 2

        # 向量化计算内部边
        edge_array = self.edge_index
        z_u = self.z[edge_array[0]]
        z_v = self.z[edge_array[1]]
        internal_mask = (z_u > 0) & (z_u == z_v)
        internal_edges = internal_mask[::2].sum().item()  # 只计算一半避免重复
        
        # 期望的内部边数
        expected_internal = 0.0
        for k in range(1, self.K + 1):
            region_size = (self.z == k).sum().item()
            if region_size > 0:
                expected_internal += (region_size / self.N) ** 2
        expected_internal *= total_edges
        
        modularity = (internal_edges - expected_internal) / max(total_edges, 1)
        
        return modularity


# Test function can be called from main if needed
def initialize_partition_env(data, embeddings, device):
    """Test function for partition environment"""
    print("\n🎮 测试强化学习环境...")

    # 创建环境（K=None 将自动使用配置文件中的设置）
    env = PowerGridPartitionEnv(
        data=data,
        embeddings=embeddings,
        K=None,  # 自动使用config.py中的NUM_REGIONS
        device=device
    )

    # 重置环境
    state = env.reset()
    print(f"✅ 环境初始化成功！")
    print(f"📊 初始边界节点数: {len(state['boundary_nodes'])}")
    print(f"📊 有效动作数: {len(env.get_valid_actions())}")

    # 测试一步
    valid_actions = env.get_valid_actions()
    if valid_actions:
        action = valid_actions[0]
        next_state, reward, done, info = env.step(action)
        print(f"\n🎯 执行动作: 节点{action[0]} → 区域{action[1]}")
        print(f"💰 获得奖励: {reward:.4f}")
        print(f"📈 当前指标: CV={info['metrics'].load_cv:.3f}, 耦合度={info['metrics'].total_coupling:.3f}")
    
    return env


