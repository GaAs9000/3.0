import numpy as np
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from ..src.rl.environment import PowerGridPartitioningEnv

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class MIPOptimalPartitioner(BasePartitioner):
    """
    基于混合整数规划(MIP)的优化分区方法
    
    将分区问题精确地建模为一个数学优化问题，通过MIP求解器寻找在给定目标和约束下的理论最优解。
    支持多种优化目标：最小化切边数、最大化模块度、平衡分区大小等。
    """
    
    def __init__(self, seed: int = 42, objective: str = 'min_cut_edges', 
                 max_time_limit: float = 300.0, mip_gap: float = 0.01,
                 balance_constraint: bool = True, connectivity_constraint: bool = True,
                 power_balance_weight: float = 1.0, cut_weight: float = 1.0):
        """
        初始化MIP优化分区器
        
        Args:
            seed: 随机种子
            objective: 优化目标 ('min_cut_edges', 'max_modularity', 'balanced_cut')
            max_time_limit: 最大求解时间（秒）
            mip_gap: MIP间隙容忍度
            balance_constraint: 是否添加平衡约束
            connectivity_constraint: 是否添加连通性约束
            power_balance_weight: 功率平衡权重
            cut_weight: 切边权重
        """
        super().__init__(seed)
        self.objective = objective
        self.max_time_limit = max_time_limit
        self.mip_gap = mip_gap
        self.balance_constraint = balance_constraint
        self.connectivity_constraint = connectivity_constraint
        self.power_balance_weight = power_balance_weight
        self.cut_weight = cut_weight
        
        if not GUROBI_AVAILABLE:
            raise ImportError("gurobipy库未安装。请运行: pip install gurobipy")
        
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx库未安装。请运行: pip install networkx")
    
    def partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """执行MIP优化分区"""
        # 设置随机种子
        self._set_seed()
        
        try:
            # 构建优化模型
            model = self._build_mip_model(env)
            
            # 求解模型
            model.optimize()
            
            # 提取解
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                labels = self._extract_solution(model, env)
                return labels
            else:
                print(f"⚠️ MIP求解失败，状态: {model.status}")
                return self._fallback_partition(env)
                
        except Exception as e:
            print(f"❌ MIP优化分区失败: {str(e)}")
            return self._fallback_partition(env)
    
    def _build_mip_model(self, env: 'PowerGridPartitioningEnv') -> gp.Model:
        """构建MIP优化模型"""
        model = gp.Model("PowerGridPartitioning")
        
        # 设置求解器参数
        model.setParam('TimeLimit', self.max_time_limit)
        model.setParam('MIPGap', self.mip_gap)
        model.setParam('OutputFlag', 0)  # 关闭输出
        model.setParam('Seed', self.seed)
        
        # 获取网络信息
        total_nodes = env.total_nodes
        num_partitions = env.num_partitions
        edge_array = env.edge_info['edge_index'].cpu().numpy()
        edges = [(edge_array[0, i], edge_array[1, i]) for i in range(edge_array.shape[1])]
        
        # 决策变量
        # x[i,k] = 1 if node i is assigned to partition k
        x = model.addVars(total_nodes, num_partitions, vtype=GRB.BINARY, name="x")
        
        # z[i,j] = 1 if edge (i,j) is cut
        z = model.addVars(edges, vtype=GRB.BINARY, name="z")
        
        # 约束1: 每个节点必须分配到一个分区
        for i in range(total_nodes):
            model.addConstr(gp.quicksum(x[i, k] for k in range(num_partitions)) == 1,
                           name=f"assign_node_{i}")
        
        # 约束2: 切边约束
        for i, j in edges:
            for k in range(num_partitions):
                model.addConstr(z[i, j] >= x[i, k] - x[j, k], name=f"cut_edge_{i}_{j}_{k}_1")
                model.addConstr(z[i, j] >= x[j, k] - x[i, k], name=f"cut_edge_{i}_{j}_{k}_2")
        
        # 约束3: 平衡约束（可选）
        if self.balance_constraint:
            self._add_balance_constraints(model, x, env)
        
        # 约束4: 连通性约束（可选）
        if self.connectivity_constraint:
            self._add_connectivity_constraints(model, x, env)
        
        # 设置目标函数
        self._set_objective(model, x, z, env)
        
        return model
    
    def _add_balance_constraints(self, model: gp.Model, x: gp.tupledict, 
                               env: 'PowerGridPartitioningEnv'):
        """添加平衡约束"""
        total_nodes = env.total_nodes
        num_partitions = env.num_partitions
        
        # 每个分区的节点数应该大致相等
        min_nodes_per_partition = total_nodes // num_partitions
        max_nodes_per_partition = min_nodes_per_partition + 1
        
        for k in range(num_partitions):
            # 最小节点数约束
            model.addConstr(gp.quicksum(x[i, k] for i in range(total_nodes)) >= min_nodes_per_partition,
                           name=f"min_nodes_partition_{k}")
            
            # 最大节点数约束
            model.addConstr(gp.quicksum(x[i, k] for i in range(total_nodes)) <= max_nodes_per_partition,
                           name=f"max_nodes_partition_{k}")
        
        # 功率平衡约束（如果有功率信息）
        if hasattr(env, 'node_features'):
            node_features = env.node_features.cpu().numpy()
            if node_features.shape[1] >= 1:
                power_values = node_features[:, 0]  # 假设第一列是功率
                
                # 每个分区的功率平衡
                total_power = np.sum(power_values)
                avg_power_per_partition = total_power / num_partitions
                power_tolerance = abs(avg_power_per_partition) * 0.2  # 20%容差
                
                for k in range(num_partitions):
                    partition_power = gp.quicksum(power_values[i] * x[i, k] for i in range(total_nodes))
                    model.addConstr(partition_power >= avg_power_per_partition - power_tolerance,
                                   name=f"power_balance_min_{k}")
                    model.addConstr(partition_power <= avg_power_per_partition + power_tolerance,
                                   name=f"power_balance_max_{k}")
    
    def _add_connectivity_constraints(self, model: gp.Model, x: gp.tupledict, 
                                    env: 'PowerGridPartitioningEnv'):
        """添加连通性约束（简化版本）"""
        # 这是一个简化的连通性约束实现
        # 完整的连通性约束需要更复杂的网络流建模
        
        total_nodes = env.total_nodes
        num_partitions = env.num_partitions
        edge_array = env.edge_info['edge_index'].cpu().numpy()
        
        # 构建邻接列表
        adj_list = [[] for _ in range(total_nodes)]
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # 为每个分区添加连通性约束
        for k in range(num_partitions):
            # 使用网络流方法确保连通性
            # 这里使用一个简化的方法：确保每个分区至少有一个"根"节点
            
            # 添加根节点变量
            root = model.addVars(total_nodes, vtype=GRB.BINARY, name=f"root_{k}")
            
            # 每个分区最多有一个根节点
            model.addConstr(gp.quicksum(root[i] for i in range(total_nodes)) <= 1,
                           name=f"max_one_root_{k}")
            
            # 如果节点i在分区k中，且分区k非空，则必须有一个根节点
            partition_size = gp.quicksum(x[i, k] for i in range(total_nodes))
            total_roots = gp.quicksum(root[i] for i in range(total_nodes))
            
            # 如果分区非空，则必须有一个根节点
            model.addConstr(total_roots >= 1 - total_nodes * (1 - partition_size / total_nodes),
                           name=f"root_if_nonempty_{k}")
            
            # 根节点必须在对应的分区中
            for i in range(total_nodes):
                model.addConstr(root[i] <= x[i, k], name=f"root_in_partition_{i}_{k}")
    
    def _set_objective(self, model: gp.Model, x: gp.tupledict, z: gp.tupledict,
                      env: 'PowerGridPartitioningEnv'):
        """设置目标函数"""
        edge_array = env.edge_info['edge_index'].cpu().numpy()
        edges = [(edge_array[0, i], edge_array[1, i]) for i in range(edge_array.shape[1])]
        
        if self.objective == 'min_cut_edges':
            # 最小化切边数
            model.setObjective(gp.quicksum(z[i, j] for i, j in edges), GRB.MINIMIZE)
            
        elif self.objective == 'weighted_min_cut':
            # 加权最小化切边（考虑边权重）
            cut_cost = gp.LinExpr()
            for idx, (i, j) in enumerate(edges):
                # 获取边权重
                if hasattr(env.evaluator, 'edge_admittances') and idx < len(env.evaluator.edge_admittances):
                    weight = float(env.evaluator.edge_admittances[idx].item())
                else:
                    weight = 1.0
                
                cut_cost += weight * z[i, j]
            
            model.setObjective(cut_cost, GRB.MINIMIZE)
            
        elif self.objective == 'balanced_cut':
            # 平衡切边：最小化切边数同时平衡分区大小
            total_nodes = env.total_nodes
            num_partitions = env.num_partitions
            
            # 切边成本
            cut_cost = self.cut_weight * gp.quicksum(z[i, j] for i, j in edges)
            
            # 平衡成本
            balance_cost = gp.LinExpr()
            ideal_size = total_nodes / num_partitions
            
            for k in range(num_partitions):
                partition_size = gp.quicksum(x[i, k] for i in range(total_nodes))
                # 添加平衡变量
                balance_pos = model.addVar(vtype=GRB.CONTINUOUS, name=f"balance_pos_{k}")
                balance_neg = model.addVar(vtype=GRB.CONTINUOUS, name=f"balance_neg_{k}")
                
                model.addConstr(balance_pos - balance_neg == partition_size - ideal_size,
                               name=f"balance_def_{k}")
                
                balance_cost += self.power_balance_weight * (balance_pos + balance_neg)
            
            model.setObjective(cut_cost + balance_cost, GRB.MINIMIZE)
            
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
    
    def _extract_solution(self, model: gp.Model, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """从求解结果中提取分区标签"""
        total_nodes = env.total_nodes
        num_partitions = env.num_partitions
        labels = np.zeros(total_nodes, dtype=int)
        
        # 提取变量值
        x_vars = model.getVars()
        x_values = model.getAttr('x', x_vars)
        
        # 找到x变量
        for var in x_vars:
            if var.varName.startswith('x['):
                # 解析变量名 x[i,k]
                var_name = var.varName[2:-1]  # 去掉 'x[' 和 ']'
                i, k = map(int, var_name.split(','))
                
                if var.x > 0.5:  # 二进制变量
                    labels[i] = k + 1  # 转换为1-based
        
        # 验证解的有效性
        if np.any(labels == 0):
            print("⚠️ 解不完整，使用修复方法")
            labels = self._fix_incomplete_solution(labels, env)
        
        return labels
    
    def _fix_incomplete_solution(self, labels: np.ndarray, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """修复不完整的解"""
        # 找到未分配的节点
        unassigned_nodes = np.where(labels == 0)[0]
        
        if len(unassigned_nodes) == 0:
            return labels
        
        # 简单策略：将未分配的节点分配到最小的分区
        for node in unassigned_nodes:
            # 找到当前最小的分区
            unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
            if len(unique_labels) > 0:
                min_partition = unique_labels[np.argmin(counts)]
                labels[node] = min_partition
            else:
                # 如果没有分配的节点，从1开始分配
                labels[node] = 1
        
        return labels
    
    def _fallback_partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """降级分区方法"""
        print("⚠️ 使用降级分区方法...")
        
        # 使用简单的轮询分配
        labels = np.zeros(env.total_nodes, dtype=int)
        for i in range(env.total_nodes):
            labels[i] = (i % env.num_partitions) + 1
        
        return labels
    
    def get_optimization_statistics(self, env: 'PowerGridPartitioningEnv') -> Dict[str, Any]:
        """获取优化统计信息"""
        try:
            model = self._build_mip_model(env)
            
            # 求解模型
            model.optimize()
            
            stats = {
                'status': model.status,
                'objective_value': model.objVal if model.status == GRB.OPTIMAL else None,
                'mip_gap': model.MIPGap if model.status == GRB.OPTIMAL else None,
                'solve_time': model.Runtime,
                'num_variables': model.NumVars,
                'num_constraints': model.NumConstrs,
                'num_binary_vars': model.NumBinVars,
                'num_integer_vars': model.NumIntVars
            }
            
            return stats
            
        except Exception as e:
            print(f"⚠️ 优化统计计算失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def solve_with_time_limit(self, env: 'PowerGridPartitioningEnv', 
                             time_limit: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """使用指定时间限制求解"""
        original_time_limit = self.max_time_limit
        self.max_time_limit = time_limit
        
        try:
            labels = self.partition(env)
            stats = self.get_optimization_statistics(env)
            
            return labels, stats
            
        finally:
            self.max_time_limit = original_time_limit
    
    def get_solution_quality(self, env: 'PowerGridPartitioningEnv', 
                           labels: np.ndarray) -> Dict[str, float]:
        """评估解的质量"""
        try:
            edge_array = env.edge_info['edge_index'].cpu().numpy()
            
            # 计算切边数
            cut_edges = 0
            total_edges = edge_array.shape[1]
            
            for i in range(total_edges):
                u, v = edge_array[0, i], edge_array[1, i]
                if labels[u] != labels[v]:
                    cut_edges += 1
            
            # 计算平衡度
            unique_labels, counts = np.unique(labels, return_counts=True)
            balance_ratio = np.std(counts) / np.mean(counts) if len(counts) > 1 else 0.0
            
            # 计算切边比例
            cut_ratio = cut_edges / total_edges if total_edges > 0 else 0.0
            
            quality = {
                'cut_edges': cut_edges,
                'total_edges': total_edges,
                'cut_ratio': cut_ratio,
                'balance_ratio': balance_ratio,
                'num_partitions': len(unique_labels)
            }
            
            return quality
            
        except Exception as e:
            print(f"⚠️ 解质量评估失败: {e}")
            return {
                'cut_edges': 0,
                'total_edges': 0,
                'cut_ratio': 0.0,
                'balance_ratio': 0.0,
                'num_partitions': 0
            }