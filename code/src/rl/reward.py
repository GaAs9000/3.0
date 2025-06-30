"""
电力网络分区MDP的奖励函数系统

基于势函数理论的双层激励结构：
- 即时奖励（reward）：基于势函数的增量奖励，提供密集反馈
- 终局奖励（Reward）：基于最终质量的综合评估，定义成功标准

核心设计原则：
1. 数值稳定性：所有计算都有NaN/inf保护
2. 理论基础：严格遵循势函数奖励塑造理论
3. 向后兼容：保留对旧系统的接口支持
"""

import torch
import numpy as np
import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
from torch_geometric.data import HeteroData


class DualLayerRewardFunction:
    """
    双层奖励函数系统
    
    实现基于势函数理论的双层激励结构：
    1. 即时奖励（reward）：每步的增量反馈，基于势函数差值
    2. 终局奖励（Reward）：回合结束时的质量评估，包含阈值奖励和终止折扣
    
    核心组件：
    - _compute_core_metrics(): 中心化的指标计算，避免重复计算
    - previous_metrics: 前一步指标缓存，支持增量计算
    - 数值稳定性保护：所有数学运算都有epsilon保护
    """
    
    def __init__(self, 
                 hetero_data: HeteroData,
                 config: Dict[str, Any] = None,
                 device: torch.device = None):
        """
        初始化双层奖励函数
        
        Args:
            hetero_data: 异构图数据
            config: 配置字典，包含权重、阈值等参数
            device: 计算设备
        """
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        self.config = config or {}
        
        # 从配置中获取权重和阈值
        self.weights = self._load_weights()
        self.thresholds = self._load_thresholds()
        
        # 前一步指标缓存
        self.previous_metrics = None
        
        # 预计算常用数据
        self._setup_cached_data()
        
        # 数值稳定性参数
        self.epsilon = 1e-9
        
    def _load_weights(self) -> Dict[str, float]:
        """从配置加载权重参数"""
        default_weights = {
            # 即时奖励权重
            'balance_weight': 1.0,
            'decoupling_weight': 1.0, 
            'power_weight': 1.0,
            # 终局奖励权重
            'final_balance_weight': 0.4,
            'final_decoupling_weight': 0.4,
            'final_power_weight': 0.2,
        }
        
        config_weights = self.config.get('reward_weights', {})
        dual_layer_config = config_weights.get('dual_layer_config', {})
        return {**default_weights, **dual_layer_config}
        
    def _load_thresholds(self) -> Dict[str, float]:
        """从配置加载阈值参数"""
        default_thresholds = {
            # 质量阈值（优秀/良好）
            'excellent_cv': 0.1,
            'good_cv': 0.2,
            'excellent_coupling': 0.3,
            'good_coupling': 0.5,
            'excellent_power': 10.0,
            'good_power': 50.0,
        }
        
        config_thresholds = self.config.get('thresholds', {})
        return {**default_thresholds, **config_thresholds}
        
    def _setup_cached_data(self):
        """预计算频繁使用的数据"""
        # 节点负载和发电数据
        if 'bus' in self.hetero_data.x_dict:
            bus_features = self.hetero_data.x_dict['bus']
            # 假设特征顺序：[Pd, Qd, Pg, Qg, ...]
            self.node_loads = bus_features[:, 0]  # 有功负载
            self.node_generation = bus_features[:, 2] if bus_features.shape[1] > 2 else torch.zeros_like(self.node_loads)
        else:
            # 如果没有bus数据，创建默认值
            num_nodes = sum(x.shape[0] for x in self.hetero_data.x_dict.values())
            self.node_loads = torch.ones(num_nodes, device=self.device)
            self.node_generation = torch.zeros(num_nodes, device=self.device)
            
        # 边信息（用于计算导纳）
        self._extract_edge_info()
        
    def _extract_edge_info(self):
        """提取边信息用于导纳计算"""
        if ('bus', 'connects', 'bus') in self.hetero_data.edge_index_dict:
            self.edge_index = self.hetero_data.edge_index_dict[('bus', 'connects', 'bus')]
            
            # 如果有边属性，提取导纳信息
            if ('bus', 'connects', 'bus') in self.hetero_data.edge_attr_dict:
                edge_attr = self.hetero_data.edge_attr_dict[('bus', 'connects', 'bus')]
                # 假设导纳在边属性的某个位置，这里需要根据实际数据调整
                self.edge_admittance = edge_attr[:, 0] if edge_attr.shape[1] > 0 else torch.ones(self.edge_index.shape[1], device=self.device)
            else:
                # 默认导纳为1
                self.edge_admittance = torch.ones(self.edge_index.shape[1], device=self.device)
        else:
            # 如果没有边信息，创建空的
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.edge_admittance = torch.empty(0, device=self.device)
            
    def _compute_core_metrics(self, partition: torch.Tensor) -> Dict[str, float]:
        """
        中心化的核心指标计算方法
        
        高效计算所有后续奖励所需的基础物理量，避免重复计算：
        - CV (变异系数): 负载平衡指标
        - coupling: 电气耦合度指标  
        - power_imbalance: 功率不平衡指标
        - 其他辅助指标
        
        Args:
            partition: 当前分区方案 [num_nodes]
            
        Returns:
            包含所有核心指标的字典
        """
        metrics = {}
        
        # 获取分区数量
        num_partitions = partition.max().item()
        if num_partitions <= 0:
            # 如果没有分区，返回最差指标
            return {
                'cv': 1.0,
                'coupling_ratio': 1.0, 
                'power_imbalance_normalized': 1.0,
                'num_partitions': 0
            }
            
        # 1. 计算负载平衡指标 (CV)
        partition_loads = torch.zeros(num_partitions, device=self.device)
        for i in range(1, num_partitions + 1):
            mask = (partition == i)
            if mask.any():
                partition_loads[i-1] = self.node_loads[mask].abs().sum()
                
        # 计算变异系数，添加数值稳定性保护
        mean_load = partition_loads.mean()
        std_load = partition_loads.std()
        
        # 多重保护：防止除零、NaN和inf
        if torch.isnan(mean_load) or torch.isinf(mean_load) or mean_load <= 0:
            cv = 1.0  # 最差情况
        elif torch.isnan(std_load) or torch.isinf(std_load):
            cv = 1.0
        else:
            cv = std_load / (mean_load + self.epsilon)
            cv = torch.clamp(cv, 0.0, 10.0)  # 限制CV在合理范围内
            
        metrics['cv'] = cv.item() if torch.is_tensor(cv) else cv
        
        # 2. 计算电气解耦指标
        if self.edge_index.shape[1] > 0:
            # 计算跨区线路的导纳和
            cross_partition_admittance = 0.0
            total_admittance = self.edge_admittance.sum().item()
            
            # 数值稳定性检查
            if torch.isnan(self.edge_admittance).any() or torch.isinf(self.edge_admittance).any():
                # 如果导纳数据有问题，使用默认值
                metrics['coupling_ratio'] = 0.5
                metrics['edge_decoupling_ratio'] = 0.5
            else:
                for i in range(self.edge_index.shape[1]):
                    node1, node2 = self.edge_index[:, i]
                    # 添加边界检查
                    if (node1 < len(partition) and node2 < len(partition) and 
                        partition[node1] != partition[node2] and 
                        partition[node1] > 0 and partition[node2] > 0):
                        admittance_val = self.edge_admittance[i].item()
                        if not (np.isnan(admittance_val) or np.isinf(admittance_val)):
                            cross_partition_admittance += admittance_val
                        
                # 计算耦合比率，添加保护
                if total_admittance <= 0 or np.isnan(total_admittance) or np.isinf(total_admittance):
                    coupling_ratio = 0.0
                else:
                    coupling_ratio = cross_partition_admittance / (total_admittance + self.epsilon)
                    coupling_ratio = np.clip(coupling_ratio, 0.0, 1.0)
                metrics['coupling_ratio'] = coupling_ratio
                
                # 计算拓扑解耦率
                cross_edges = 0
                total_edges = self.edge_index.shape[1]
                for i in range(total_edges):
                    node1, node2 = self.edge_index[:, i]
                    if (node1 < len(partition) and node2 < len(partition) and
                        partition[node1] != partition[node2] and 
                        partition[node1] > 0 and partition[node2] > 0):
                        cross_edges += 1
                        
                edge_decoupling_ratio = 1.0 - (cross_edges / (total_edges + self.epsilon))
                edge_decoupling_ratio = np.clip(edge_decoupling_ratio, 0.0, 1.0)
                metrics['edge_decoupling_ratio'] = edge_decoupling_ratio
        else:
            metrics['coupling_ratio'] = 0.0
            metrics['edge_decoupling_ratio'] = 1.0
            
        # 3. 计算功率平衡指标
        total_imbalance = 0.0
        
        # 检查节点数据的数值稳定性
        if (torch.isnan(self.node_generation).any() or torch.isinf(self.node_generation).any() or
            torch.isnan(self.node_loads).any() or torch.isinf(self.node_loads).any()):
            # 如果数据有问题，返回最差情况
            metrics['power_imbalance_normalized'] = 1.0
        else:
            for i in range(1, num_partitions + 1):
                mask = (partition == i)
                if mask.any():
                    partition_generation = self.node_generation[mask].sum()
                    partition_load = self.node_loads[mask].sum()
                    
                    # 检查每个分区的计算结果
                    if (torch.isnan(partition_generation) or torch.isinf(partition_generation) or
                        torch.isnan(partition_load) or torch.isinf(partition_load)):
                        continue  # 跳过有问题的分区
                        
                    imbalance = torch.abs(partition_generation - partition_load)
                    if not (torch.isnan(imbalance) or torch.isinf(imbalance)):
                        total_imbalance += imbalance.item()
                    
            # 归一化功率不平衡，添加多重保护
            total_load = self.node_loads.abs().sum().item()
            if total_load <= 0 or np.isnan(total_load) or np.isinf(total_load):
                power_imbalance_normalized = 1.0  # 最差情况
            else:
                power_imbalance_normalized = total_imbalance / (total_load + self.epsilon)
                power_imbalance_normalized = np.clip(power_imbalance_normalized, 0.0, 10.0)
                
            metrics['power_imbalance_normalized'] = power_imbalance_normalized
        
        # 4. 其他辅助指标
        metrics['num_partitions'] = num_partitions
        
        return metrics

    def compute_incremental_reward(self,
                                 current_partition: torch.Tensor,
                                 action: Tuple[int, int]) -> float:
        """
        计算基于势函数理论的即时奖励

        实现增量奖励（Delta Reward），等价于势函数奖励塑造：
        reward_t = γ * Φ(s_{t+1}) - Φ(s_t)

        其中势函数 Φ(s) = -(w1*CV + w2*coupling + w3*power_imbalance)

        Args:
            current_partition: 当前分区状态
            action: 执行的动作 (node_idx, partition_id)

        Returns:
            归一化的即时奖励值
        """
        # 计算当前状态的核心指标
        current_metrics = self._compute_core_metrics(current_partition)

        # 如果没有前一步指标，初始化并返回0
        if self.previous_metrics is None:
            self.previous_metrics = current_metrics
            return 0.0

        # 计算各指标的改善量（前一步 - 当前步），添加数值稳定性检查
        try:
            delta_balance = self.previous_metrics['cv'] - current_metrics['cv']
            delta_decoupling = self.previous_metrics['coupling_ratio'] - current_metrics['coupling_ratio']
            delta_power = self.previous_metrics['power_imbalance_normalized'] - current_metrics['power_imbalance_normalized']

            # 检查delta值的有效性
            if np.isnan(delta_balance) or np.isinf(delta_balance):
                delta_balance = 0.0
            if np.isnan(delta_decoupling) or np.isinf(delta_decoupling):
                delta_decoupling = 0.0
            if np.isnan(delta_power) or np.isinf(delta_power):
                delta_power = 0.0

            # 限制delta值在合理范围内
            delta_balance = np.clip(delta_balance, -1.0, 1.0)
            delta_decoupling = np.clip(delta_decoupling, -1.0, 1.0)
            delta_power = np.clip(delta_power, -1.0, 1.0)

            # 加权求和
            incremental_reward = (
                self.weights['balance_weight'] * delta_balance +
                self.weights['decoupling_weight'] * delta_decoupling +
                self.weights['power_weight'] * delta_power
            )

            # 检查加权求和结果
            if np.isnan(incremental_reward) or np.isinf(incremental_reward):
                incremental_reward = 0.0
            else:
                # 限制在合理范围内
                incremental_reward = np.clip(incremental_reward, -5.0, 5.0)

            # 使用tanh进行归一化，将奖励值稳定在[-1, 1]范围
            normalized_reward = np.tanh(incremental_reward * 5.0)  # 放大5倍增加敏感度

        except Exception as e:
            # 如果出现任何异常，返回0奖励
            print(f"警告：奖励计算出现异常: {e}")
            normalized_reward = 0.0

        # 更新前一步指标
        self.previous_metrics = current_metrics

        return normalized_reward

    def compute_final_reward(self,
                           final_partition: torch.Tensor,
                           termination_type: str = 'natural') -> Tuple[float, Dict[str, float]]:
        """
        计算终局奖励（Reward）

        基于最终分区质量的综合评估，包含：
        1. 三个核心奖励组件（按设计蓝图公式）
        2. 非线性阈值奖励
        3. 终止条件折扣

        Args:
            final_partition: 最终分区方案
            termination_type: 终止类型 ('natural', 'timeout', 'stuck')

        Returns:
            (总终局奖励, 奖励组件详情)
        """
        # 计算最终指标
        final_metrics = self._compute_core_metrics(final_partition)

        # 1. 计算三个核心奖励组件
        balance_reward = self._compute_balance_reward(final_metrics['cv'])
        decoupling_reward = self._compute_decoupling_reward(
            final_metrics['edge_decoupling_ratio'],
            final_metrics['coupling_ratio']
        )
        power_reward = self._compute_power_reward(final_metrics['power_imbalance_normalized'])

        # 2. 加权求和得到质量奖励
        quality_reward = (
            self.weights['final_balance_weight'] * balance_reward +
            self.weights['final_decoupling_weight'] * decoupling_reward +
            self.weights['final_power_weight'] * power_reward
        )

        # 3. 计算非线性阈值奖励
        threshold_bonus = self._compute_threshold_bonus(final_metrics)

        # 4. 应用终止条件折扣
        termination_discount = self._apply_termination_discount(termination_type)

        # 5. 计算最终奖励
        final_reward = (quality_reward + threshold_bonus) * termination_discount

        # 组件详情
        components = {
            'balance_reward': balance_reward,
            'decoupling_reward': decoupling_reward,
            'power_reward': power_reward,
            'quality_reward': quality_reward,
            'threshold_bonus': threshold_bonus,
            'termination_discount': termination_discount,
            'final_reward': final_reward,
            'metrics': final_metrics
        }

        return final_reward, components

    def _compute_balance_reward(self, cv: float) -> float:
        """
        计算负载平衡奖励

        公式：R_balance = exp(-2.0 * CV)
        CV越小，奖励越接近1.0
        """
        # 添加数值稳定性保护
        if np.isnan(cv) or np.isinf(cv):
            cv = 1.0  # 最差情况
        else:
            cv = max(0.0, min(cv, 10.0))  # 确保CV在合理范围内

        try:
            balance_reward = np.exp(-2.0 * cv)
            # 检查结果
            if np.isnan(balance_reward) or np.isinf(balance_reward):
                balance_reward = 0.0
            else:
                balance_reward = np.clip(balance_reward, 0.0, 1.0)
        except Exception:
            balance_reward = 0.0

        return balance_reward

    def _compute_decoupling_reward(self, edge_decoupling_ratio: float, coupling_ratio: float) -> float:
        """
        计算电气解耦奖励

        公式：R_decoupling = 0.5 * σ(5*(r_edge - 0.5)) + 0.5 * σ(5*(r_admittance - 0.5))
        其中 σ 是sigmoid函数，r_admittance = 1 - coupling_ratio
        """
        # 数值稳定性检查
        if np.isnan(edge_decoupling_ratio) or np.isinf(edge_decoupling_ratio):
            edge_decoupling_ratio = 0.5
        else:
            edge_decoupling_ratio = np.clip(edge_decoupling_ratio, 0.0, 1.0)

        if np.isnan(coupling_ratio) or np.isinf(coupling_ratio):
            coupling_ratio = 0.5
        else:
            coupling_ratio = np.clip(coupling_ratio, 0.0, 1.0)

        # 计算导纳解耦率
        admittance_decoupling_ratio = 1.0 - coupling_ratio

        try:
            # 应用sigmoid函数
            edge_component = self._sigmoid(5.0 * (edge_decoupling_ratio - 0.5))
            admittance_component = self._sigmoid(5.0 * (admittance_decoupling_ratio - 0.5))

            decoupling_reward = 0.5 * edge_component + 0.5 * admittance_component

            # 检查结果
            if np.isnan(decoupling_reward) or np.isinf(decoupling_reward):
                decoupling_reward = 0.5
            else:
                decoupling_reward = np.clip(decoupling_reward, 0.0, 1.0)

        except Exception:
            decoupling_reward = 0.5

        return decoupling_reward

    def _compute_power_reward(self, power_imbalance_normalized: float) -> float:
        """
        计算功率平衡奖励

        公式：R_power = exp(-3.0 * I_normalized)
        I_normalized越小，奖励越接近1.0
        """
        # 添加数值稳定性保护
        if np.isnan(power_imbalance_normalized) or np.isinf(power_imbalance_normalized):
            power_imbalance_normalized = 1.0  # 最差情况
        else:
            power_imbalance_normalized = max(0.0, min(power_imbalance_normalized, 10.0))

        try:
            # 限制指数参数避免溢出
            exp_arg = -3.0 * power_imbalance_normalized
            exp_arg = np.clip(exp_arg, -500, 0)  # 防止exp溢出

            power_reward = np.exp(exp_arg)

            # 检查结果
            if np.isnan(power_reward) or np.isinf(power_reward):
                power_reward = 0.0
            else:
                power_reward = np.clip(power_reward, 0.0, 1.0)

        except Exception:
            power_reward = 0.0

        return power_reward

    def _compute_threshold_bonus(self, metrics: Dict[str, float]) -> float:
        """
        计算非线性阈值奖励

        对达到不同质量阈值的指标给予不同量级的奖励，鼓励"卓越"
        """
        bonus = 0.0

        # 负载平衡阈值奖励
        cv = metrics['cv']
        if cv < self.thresholds['excellent_cv']:
            bonus += 10.0  # 卓越奖励
        elif cv < self.thresholds['good_cv']:
            bonus += 5.0   # 良好奖励

        # 电气解耦阈值奖励
        coupling_ratio = metrics['coupling_ratio']
        if coupling_ratio < self.thresholds['excellent_coupling']:
            bonus += 8.0
        elif coupling_ratio < self.thresholds['good_coupling']:
            bonus += 4.0

        # 功率平衡阈值奖励
        power_imbalance = metrics['power_imbalance_normalized'] * 100  # 转换为百分比
        if power_imbalance < self.thresholds['excellent_power']:
            bonus += 6.0
        elif power_imbalance < self.thresholds['good_power']:
            bonus += 3.0

        return bonus

    def _apply_termination_discount(self, termination_type: str) -> float:
        """
        根据终止条件应用不同的折扣或惩罚

        Args:
            termination_type: 'natural' (自然完成), 'timeout' (超时), 'stuck' (卡住)

        Returns:
            折扣系数
        """
        if termination_type == 'natural':
            return 1.0      # 自然完成，无折扣
        elif termination_type == 'timeout':
            return 0.7      # 超时完成，70%折扣
        elif termination_type == 'stuck':
            return 0.3      # 提前卡住，30%折扣
        else:
            return 0.5      # 未知情况，50%折扣

    def _sigmoid(self, x: float) -> float:
        """数值稳定的sigmoid函数"""
        # 防止数值溢出
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def reset_episode(self):
        """重置回合状态，清除前一步指标缓存"""
        self.previous_metrics = None

    def get_current_metrics(self, partition: torch.Tensor) -> Dict[str, float]:
        """获取当前分区的指标（用于调试和分析）"""
        return self._compute_core_metrics(partition)

    def update_weights(self, new_weights: Dict[str, float]):
        """动态更新奖励权重（用于智能自适应课程学习）"""
        try:
            # 更新即时奖励权重
            if 'balance_weight' in new_weights:
                self.weights['balance_weight'] = new_weights['balance_weight']
            if 'decoupling_weight' in new_weights:
                self.weights['decoupling_weight'] = new_weights['decoupling_weight']
            if 'power_weight' in new_weights:
                self.weights['power_weight'] = new_weights['power_weight']

            # 更新终局奖励权重（保持一致性）
            if 'balance_weight' in new_weights:
                self.weights['final_balance_weight'] = new_weights['balance_weight']
            if 'decoupling_weight' in new_weights:
                self.weights['final_decoupling_weight'] = new_weights['decoupling_weight']
            if 'power_weight' in new_weights:
                self.weights['final_power_weight'] = new_weights['power_weight']

            print(f"🎯 奖励权重已更新: balance={self.weights['balance_weight']:.2f}, "
                  f"decoupling={self.weights['decoupling_weight']:.2f}, "
                  f"power={self.weights['power_weight']:.2f}")

        except Exception as e:
            print(f"⚠️ 更新奖励权重失败: {e}")

    def get_current_weights(self) -> Dict[str, float]:
        """获取当前奖励权重"""
        return {
            'balance_weight': self.weights.get('balance_weight', 1.0),
            'decoupling_weight': self.weights.get('decoupling_weight', 1.0),
            'power_weight': self.weights.get('power_weight', 1.0)
        }

    # ==================== 向后兼容接口 ====================

    def compute_reward(self,
                      current_partition: torch.Tensor,
                      boundary_nodes: torch.Tensor,
                      action: Tuple[int, int],
                      return_components: bool = False) -> Union[float, Tuple[float, Dict]]:
        """
        向后兼容的奖励计算接口

        这个方法保持与原有EnhancedRewardFunction相同的接口，
        但内部使用新的DualLayerRewardFunction实现
        """
        # 计算即时奖励
        incremental_reward = self.compute_incremental_reward(current_partition, action)

        if return_components:
            # 如果需要返回组件，计算当前指标用于调试
            current_metrics = self._compute_core_metrics(current_partition)
            components = {
                'incremental_reward': incremental_reward,
                'current_metrics': current_metrics
            }
            return incremental_reward, components
        else:
            return incremental_reward


# 保留原有的RewardFunction类以确保向后兼容
class RewardFunction:
    """
    向后兼容的原始奖励函数类
    提供基本的奖励计算功能
    """

    def __init__(self, hetero_data: HeteroData, reward_weights: Dict[str, float] = None, device: torch.device = None):
        """简单的初始化"""
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        self.reward_weights = reward_weights or {}

    def compute_reward(self, current_partition: torch.Tensor, boundary_nodes: torch.Tensor,
                      action: Tuple[int, int], return_components: bool = False) -> Union[float, Tuple[float, Dict]]:
        """基本的奖励计算，返回0以保持兼容性"""
        if return_components:
            return 0.0, {}
        return 0.0
