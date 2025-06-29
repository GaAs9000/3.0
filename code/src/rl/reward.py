"""
电力网络分区MDP的双层奖励函数系统

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
        return {**default_weights, **config_weights}

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


class EnhancedRewardFunction:
    """
    【三阶段优化版】电力网络分区增强奖励函数

    第一阶段功能：
    - 稠密奖励函数：局部连通性、增量负载平衡、边界压缩
    - 奖励标准化机制：z-score标准化 + tanh压缩
    - 数值稳定性保证：epsilon保护、梯度裁剪

    第二阶段功能：
    - 探索奖励：状态新颖度、动作多样性
    - 势函数奖励塑造：理论最优的长期引导

    第三阶段功能：
    - 邻居一致性奖励：物理约束精细化
    - 动态自适应权重：训练进度感知的权重调整
    """

    def __init__(self,
                 hetero_data: HeteroData,
                 reward_weights: Dict[str, float] = None,
                 device: torch.device = None,
                 debug_mode: bool = False,
                 enable_dense_rewards: bool = True,
                 enable_exploration_bonus: bool = False,
                 enable_potential_shaping: bool = False,
                 enable_adaptive_weights: bool = False,
                 episode_count: int = 0):
        """
        初始化增强奖励函数

        Args:
            hetero_data: 异构图数据
            reward_weights: 奖励组件权重
            device: 计算设备
            debug_mode: 调试模式
            enable_dense_rewards: 启用稠密奖励（第一阶段）
            enable_exploration_bonus: 启用探索奖励（第二阶段）
            enable_potential_shaping: 启用势函数塑造（第二阶段）
            enable_adaptive_weights: 启用自适应权重（第三阶段）
            episode_count: 当前训练回合数（用于自适应权重）
        """
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        self.debug_mode = debug_mode
        self.episode_count = episode_count

        # 阶段控制开关
        self.enable_dense_rewards = enable_dense_rewards
        self.enable_exploration_bonus = enable_exploration_bonus
        self.enable_potential_shaping = enable_potential_shaping
        self.enable_adaptive_weights = enable_adaptive_weights

        # 基础奖励权重
        default_weights = {
            'load_balance': 0.4,
            'electrical_decoupling': 0.4,
            'power_balance': 0.2,
            # 第一阶段：稠密奖励权重
            'local_connectivity': 0.3,
            'incremental_balance': 0.3,
            'boundary_compression': 0.2,
            # 第二阶段：探索与塑造权重
            'exploration_bonus': 0.1,
            'potential_shaping': 0.2,
            # 第三阶段：物理约束权重
            'neighbor_consistency': 0.15
        }
        self.weights = reward_weights or default_weights

        # 奖励标准化统计
        self.reward_stats = defaultdict(lambda: {'mean': 0.0, 'std': 1.0, 'count': 0})
        self.normalization_window = 1000  # 滑动窗口大小

        # 探索相关状态（第二阶段）
        self.state_visits = defaultdict(int)
        self.action_history = deque(maxlen=50)  # 近期动作历史

        # 势函数相关（第二阶段）
        self.previous_potential = None

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
        【三阶段优化版】计算综合奖励

        根据启用的阶段功能，计算相应的奖励组件：
        - 基础奖励：负载平衡、电气解耦、功率平衡
        - 稠密奖励：局部连通性、增量平衡、边界压缩
        - 探索奖励：状态新颖度、动作多样性
        - 势函数塑造：长期价值引导
        - 物理约束：邻居一致性
        """
        self.debug_info = {}
        components = {}

        # ==================== 基础奖励组件 ====================
        load_balance_reward = self._compute_load_balance_reward(current_partition)
        electrical_decoupling_reward = self._compute_electrical_decoupling_reward(current_partition)
        power_balance_reward = self._compute_power_balance_reward(current_partition)

        components.update({
            'load_balance': load_balance_reward.item(),
            'electrical_decoupling': electrical_decoupling_reward.item(),
            'power_balance': power_balance_reward.item()
        })

        # ==================== 第一阶段：稠密奖励 ====================
        if self.enable_dense_rewards:
            dense_rewards = self._compute_dense_rewards(current_partition, boundary_nodes, action)
            components.update(dense_rewards)

        # ==================== 第二阶段：探索奖励 ====================
        if self.enable_exploration_bonus:
            exploration_rewards = self._compute_exploration_bonus(current_partition, action)
            components.update(exploration_rewards)

        # ==================== 第二阶段：势函数塑造 ====================
        if self.enable_potential_shaping:
            shaping_reward = self._compute_potential_based_shaping(current_partition)
            components['potential_shaping'] = shaping_reward

        # ==================== 第三阶段：物理约束 ====================
        if self.enable_adaptive_weights:
            neighbor_consistency = self._compute_neighbor_consistency_reward(current_partition)
            components['neighbor_consistency'] = neighbor_consistency

        # ==================== 奖励标准化与组合 ====================
        normalized_components = self._normalize_reward_components(components)

        # 获取当前权重（可能是自适应的）
        current_weights = self._get_adaptive_weights() if self.enable_adaptive_weights else self.weights

        # 计算加权总奖励
        total_reward = self._combine_rewards(normalized_components, current_weights)

        if return_components:
            return total_reward, normalized_components
        else:
            return total_reward

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

    # ==================== 第一阶段：稠密奖励函数 ====================

    def _compute_dense_rewards(self,
                              current_partition: torch.Tensor,
                              boundary_nodes: torch.Tensor,
                              action: Tuple[int, int]) -> Dict[str, float]:
        """
        【第一阶段核心】计算稠密奖励组件

        三个关键稠密奖励：
        1. 局部连通性奖励：直接奖励增强局部拓扑紧密性的动作
        2. 增量负载平衡：将全局平衡评估转化为单步动作的即时反馈
        3. 边界压缩奖励：将"完成任务"转化为"缩小边界"的可量化进度

        这些奖励解决了训练初期的"稀疏信号"问题，提供即时、清晰的反馈
        """
        dense_rewards = {}

        # 1. 局部连通性奖励
        local_connectivity = self._compute_local_connectivity_reward(current_partition, action)
        dense_rewards['local_connectivity'] = local_connectivity

        # 2. 增量负载平衡奖励
        incremental_balance = self._compute_incremental_balance_reward(current_partition, action)
        dense_rewards['incremental_balance'] = incremental_balance

        # 3. 边界压缩奖励
        boundary_compression = self._compute_boundary_compression_reward(boundary_nodes)
        dense_rewards['boundary_compression'] = boundary_compression

        return dense_rewards

    def _compute_local_connectivity_reward(self,
                                         current_partition: torch.Tensor,
                                         action: Tuple[int, int]) -> float:
        """
        局部连通性奖励：奖励那些能增强局部拓扑紧密性的动作

        核心思想：
        - 如果新分配的节点与同分区的邻居节点连接更紧密，给予奖励
        - 基于节点度数和邻居分区一致性计算
        - 提供即时的拓扑质量反馈
        """
        node_idx, partition_id = action

        if self.all_edges.shape[1] == 0:
            return 0.0

        # 找到该节点的所有邻居
        node_neighbors = []
        for i in range(self.all_edges.shape[1]):
            if self.all_edges[0, i] == node_idx:
                node_neighbors.append(self.all_edges[1, i].item())
            elif self.all_edges[1, i] == node_idx:
                node_neighbors.append(self.all_edges[0, i].item())

        if not node_neighbors:
            return 0.0

        # 计算邻居中已分配到同一分区的比例
        same_partition_neighbors = 0
        total_assigned_neighbors = 0

        for neighbor in node_neighbors:
            if neighbor < len(current_partition):
                neighbor_partition = current_partition[neighbor].item()
                if neighbor_partition > 0:  # 已分配
                    total_assigned_neighbors += 1
                    if neighbor_partition == partition_id:
                        same_partition_neighbors += 1

        if total_assigned_neighbors == 0:
            return 0.1  # 基础探索奖励

        # 连通性比例越高，奖励越大
        connectivity_ratio = same_partition_neighbors / total_assigned_neighbors

        # 使用平滑函数映射到 [0, 1]
        reward = connectivity_ratio ** 0.5  # 平方根使奖励更平滑

        return reward

    def _compute_incremental_balance_reward(self,
                                          current_partition: torch.Tensor,
                                          action: Tuple[int, int]) -> float:
        """
        增量负载平衡奖励：将对全局平衡的评估转化为对单步动作影响的即时反馈

        核心思想：
        - 计算动作前后负载变异系数的变化
        - 如果动作改善了负载平衡，给予正奖励
        - 如果动作恶化了负载平衡，给予负奖励
        """
        node_idx, partition_id = action

        # 计算当前负载平衡状态
        current_cv = self._compute_load_cv(current_partition)

        # 模拟执行动作后的状态
        simulated_partition = current_partition.clone()
        simulated_partition[node_idx] = partition_id

        # 计算动作后的负载平衡状态
        new_cv = self._compute_load_cv(simulated_partition)

        # 计算改善程度（CV越小越好，所以current - new > 0表示改善）
        improvement = current_cv - new_cv

        # 使用sigmoid函数将改善程度映射到 [-1, 1]
        reward = np.tanh(improvement * 10.0)  # 放大10倍使奖励更敏感

        # 映射到 [0, 1]
        reward = (reward + 1.0) / 2.0

        return reward

    def _compute_boundary_compression_reward(self, boundary_nodes: torch.Tensor) -> float:
        """
        边界压缩奖励：将"完成任务"转化为"缩小边界"的可量化进度奖励

        核心思想：
        - 边界节点数量越少，说明分区任务越接近完成
        - 提供持续的进度反馈，避免稀疏奖励问题
        - 使用反比例函数确保奖励随边界缩小而增加
        """
        num_boundary_nodes = len(boundary_nodes)
        total_nodes = self.total_nodes

        if num_boundary_nodes == 0:
            return 1.0  # 完成所有分区，最大奖励

        # 计算完成进度（已分配节点比例）
        assigned_nodes = total_nodes - num_boundary_nodes
        progress = assigned_nodes / total_nodes

        # 使用平方函数使后期进度更有价值
        reward = progress ** 1.5

        return reward

    def _compute_load_cv(self, partition: torch.Tensor) -> float:
        """
        计算负载变异系数的辅助函数
        """
        num_partitions = partition.max().item()
        if num_partitions <= 1:
            return 0.0

        partition_loads = torch.zeros(num_partitions, device=self.device)
        for i in range(1, num_partitions + 1):
            mask = (partition == i)
            if mask.any():
                partition_loads[i-1] = self.node_loads[mask].abs().sum()

        mean_load = partition_loads.mean()
        load_std = partition_loads.std()

        if mean_load == 0:
            return 0.0

        return (load_std / mean_load).item()

    # ==================== 奖励标准化机制 ====================

    def _normalize_reward_components(self, components: Dict[str, float]) -> Dict[str, float]:
        """
        【第一阶段核心】奖励标准化函数

        实现z-score标准化 + tanh压缩，确保：
        1. 各奖励组件尺度一致，避免梯度被单一分量主导
        2. 奖励值稳定在 [-1, 1] 区间，增强数值稳定性
        3. 使用滑动窗口统计，适应训练过程中的分布变化
        """
        normalized_components = {}

        for component_name, value in components.items():
            # 更新运行时统计
            self._update_reward_stats(component_name, value)

            # 获取当前统计信息
            stats = self.reward_stats[component_name]
            mean = stats['mean']
            std = max(stats['std'], 1e-8)  # 避免除零

            # Z-score标准化
            normalized_value = (value - mean) / std

            # Tanh压缩到 [-1, 1]
            compressed_value = np.tanh(normalized_value)

            normalized_components[component_name] = compressed_value

        return normalized_components

    def _update_reward_stats(self, component_name: str, value: float):
        """
        更新奖励组件的运行时统计信息
        使用指数移动平均来适应分布变化
        """
        stats = self.reward_stats[component_name]
        count = stats['count']

        if count == 0:
            # 首次观测
            stats['mean'] = value
            stats['std'] = 1.0
            stats['count'] = 1
        else:
            # 指数移动平均更新
            alpha = min(0.1, 1.0 / count)  # 自适应学习率

            # 更新均值
            old_mean = stats['mean']
            stats['mean'] = (1 - alpha) * old_mean + alpha * value

            # 更新标准差（使用偏差的指数移动平均）
            deviation = abs(value - old_mean)
            stats['std'] = (1 - alpha) * stats['std'] + alpha * deviation

            stats['count'] = min(count + 1, self.normalization_window)

    def _combine_rewards(self, normalized_components: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        组合标准化后的奖励组件
        """
        total_reward = 0.0
        total_weight = 0.0

        for component_name, value in normalized_components.items():
            if component_name in weights:
                weight = weights[component_name]
                total_reward += weight * value
                total_weight += weight

        # 归一化到权重总和
        if total_weight > 0:
            total_reward /= total_weight

        return total_reward

    # ==================== 第二阶段：智能探索奖励 ====================

    def _compute_exploration_bonus(self, current_partition: torch.Tensor, action: Tuple[int, int]) -> Dict[str, float]:
        """
        【第二阶段核心】计算智能探索奖励组件

        实现两个关键探索机制：
        1. 状态访问新颖度：基于状态哈希的访问频率，鼓励探索未访问状态
        2. 动作多样性：基于动作模式分析，鼓励尝试不同的分配策略

        这些机制帮助智能体跳出局部最优，提升探索效率
        """
        exploration_rewards = {}

        # 1. 状态新颖度奖励 - 高级实现
        novelty_bonus = self._compute_state_novelty_bonus(current_partition)
        exploration_rewards['novelty_bonus'] = novelty_bonus

        # 2. 动作多样性奖励 - 高级实现
        diversity_bonus = self._compute_action_diversity_bonus(action)
        exploration_rewards['diversity_bonus'] = diversity_bonus

        # 3. 探索深度奖励 - 鼓励深入探索
        depth_bonus = self._compute_exploration_depth_bonus(current_partition)
        exploration_rewards['exploration_depth'] = depth_bonus

        return exploration_rewards

    def _compute_state_novelty_bonus(self, current_partition: torch.Tensor) -> float:
        """
        计算状态新颖度奖励

        使用多层次状态表示：
        1. 粗粒度：分区分布模式
        2. 细粒度：具体节点分配
        3. 结构化：基于图拓扑的状态特征
        """
        # 生成多层次状态哈希
        coarse_hash = self._get_coarse_state_hash(current_partition)
        fine_hash = self._get_fine_state_hash(current_partition)
        structural_hash = self._get_structural_state_hash(current_partition)

        # 计算各层次的新颖度
        coarse_novelty = self._compute_hash_novelty(coarse_hash, 'coarse')
        fine_novelty = self._compute_hash_novelty(fine_hash, 'fine')
        structural_novelty = self._compute_hash_novelty(structural_hash, 'structural')

        # 加权组合新颖度分数
        novelty_score = (
            0.3 * coarse_novelty +      # 粗粒度：鼓励探索不同的分区模式
            0.4 * fine_novelty +        # 细粒度：鼓励具体的新分配
            0.3 * structural_novelty    # 结构化：鼓励拓扑上有意义的探索
        )

        return novelty_score

    def _compute_action_diversity_bonus(self, action: Tuple[int, int]) -> float:
        """
        计算动作多样性奖励

        分析动作模式：
        1. 节点选择多样性：是否总是选择相同类型的节点
        2. 分区选择多样性：是否总是分配到相同分区
        3. 时序模式多样性：是否存在重复的动作序列
        """
        # 更新动作历史
        self.action_history.append(action)

        if len(self.action_history) < 5:
            return 0.5  # 初期给予中等奖励

        # 1. 节点选择多样性
        recent_nodes = [act[0] for act in list(self.action_history)[-10:]]
        node_diversity = len(set(recent_nodes)) / len(recent_nodes)

        # 2. 分区选择多样性
        recent_partitions = [act[1] for act in list(self.action_history)[-10:]]
        partition_diversity = len(set(recent_partitions)) / len(recent_partitions)

        # 3. 时序模式多样性（检测重复模式）
        pattern_diversity = self._compute_pattern_diversity()

        # 组合多样性分数
        diversity_score = (
            0.4 * node_diversity +      # 节点选择多样性
            0.3 * partition_diversity +  # 分区选择多样性
            0.3 * pattern_diversity      # 模式多样性
        )

        return diversity_score

    def _compute_exploration_depth_bonus(self, current_partition: torch.Tensor) -> float:
        """
        计算探索深度奖励

        鼓励智能体进行"深度探索"而非"浅层随机"：
        1. 连续探索奖励：在同一区域持续探索
        2. 边界扩展奖励：从已知区域向未知区域扩展
        3. 完整性探索奖励：完成某个分区的完整探索
        """
        assigned_nodes = (current_partition > 0).sum().item()
        total_nodes = len(current_partition)
        progress = assigned_nodes / total_nodes

        # 深度探索奖励随进度非线性增长
        # 早期探索给予较高奖励，后期完成给予更高奖励
        if progress < 0.3:
            depth_bonus = progress * 2.0  # 早期探索激励
        elif progress < 0.7:
            depth_bonus = 0.6 + (progress - 0.3) * 0.5  # 中期稳定
        else:
            depth_bonus = 0.8 + (progress - 0.7) * 3.0  # 后期完成激励

        return min(depth_bonus, 1.0)  # 限制在[0,1]范围内

    # ==================== 探索奖励辅助函数 ====================

    def _get_coarse_state_hash(self, partition: torch.Tensor) -> str:
        """
        生成粗粒度状态哈希：基于分区大小分布
        """
        partition_counts = torch.bincount(partition[partition > 0])
        # 排序以确保相同分布的不同排列产生相同哈希
        sorted_counts = sorted(partition_counts.tolist())
        state_str = ','.join(map(str, sorted_counts))
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

    def _get_fine_state_hash(self, partition: torch.Tensor) -> str:
        """
        生成细粒度状态哈希：基于具体节点分配
        """
        # 使用节点分配的具体模式
        state_str = ','.join(map(str, partition.tolist()))
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

    def _get_structural_state_hash(self, partition: torch.Tensor) -> str:
        """
        生成结构化状态哈希：基于图拓扑特征
        """
        if self.all_edges.shape[1] == 0:
            return self._get_fine_state_hash(partition)

        # 计算跨分区边的模式
        cross_partition_pattern = []
        for i in range(self.all_edges.shape[1]):
            node1, node2 = self.all_edges[0, i].item(), self.all_edges[1, i].item()
            if (node1 < len(partition) and node2 < len(partition) and
                partition[node1] > 0 and partition[node2] > 0):
                if partition[node1] != partition[node2]:
                    cross_partition_pattern.append(1)
                else:
                    cross_partition_pattern.append(0)

        pattern_str = ''.join(map(str, cross_partition_pattern))
        return hashlib.md5(pattern_str.encode()).hexdigest()[:8]

    def _compute_hash_novelty(self, state_hash: str, hash_type: str) -> float:
        """
        计算状态哈希的新颖度分数
        """
        visit_key = f"{hash_type}_{state_hash}"
        visit_count = self.state_visits[visit_key]

        # 更新访问计数
        self.state_visits[visit_key] += 1

        # 新颖度分数：访问次数越少，新颖度越高
        # 使用对数衰减确保新颖度不会完全消失
        novelty = 1.0 / (1.0 + np.log(1.0 + visit_count))

        return novelty

    def _compute_pattern_diversity(self) -> float:
        """
        计算动作序列的模式多样性
        """
        if len(self.action_history) < 6:
            return 1.0  # 序列太短，给予高多样性分数

        # 检测最近动作中的重复模式
        recent_actions = list(self.action_history)[-10:]

        # 检测长度为2和3的重复模式
        pattern_counts = defaultdict(int)

        # 长度为2的模式
        for i in range(len(recent_actions) - 1):
            pattern = (recent_actions[i], recent_actions[i+1])
            pattern_counts[pattern] += 1

        # 长度为3的模式
        for i in range(len(recent_actions) - 2):
            pattern = (recent_actions[i], recent_actions[i+1], recent_actions[i+2])
            pattern_counts[pattern] += 1

        # 计算模式多样性：重复模式越少，多样性越高
        if not pattern_counts:
            return 1.0

        max_repeats = max(pattern_counts.values())
        diversity = 1.0 / (1.0 + max_repeats - 1)  # 减1因为每个模式至少出现1次

        return diversity

    # ==================== 第二阶段：势函数奖励塑造 ====================

    def _compute_potential_based_shaping(self, current_partition: torch.Tensor) -> float:
        """
        【第二阶段核心】基于势函数的奖励塑造

        实现理论最优的长期价值引导，基于Ng等人的势函数理论：
        F(s,a,s') = γ * Φ(s') - Φ(s)

        其中Φ(s)是状态势函数，保证奖励塑造不改变最优策略
        """
        # 计算当前状态的势能
        current_potential = self._compute_potential(current_partition)

        if self.previous_potential is None:
            self.previous_potential = current_potential
            return 0.0

        # 势能差作为塑造奖励（已经包含折扣因子γ）
        gamma = 0.99  # 折扣因子，应该与PPO的gamma一致
        shaping_reward = gamma * current_potential - self.previous_potential
        self.previous_potential = current_potential

        return shaping_reward

    def _compute_potential(self, partition: torch.Tensor) -> float:
        """
        【第二阶段核心】计算多维度状态势函数

        设计原则：
        1. 单调性：势函数值随着"好状态"增加而增加
        2. 有界性：势函数值在合理范围内
        3. 可微性：势函数平滑变化，提供稳定梯度
        4. 领域知识：结合电力系统的物理约束
        """
        # 1. 进度势能：鼓励完成分区任务
        progress_potential = self._compute_progress_potential(partition)

        # 2. 质量势能：鼓励高质量的分区
        quality_potential = self._compute_quality_potential(partition)

        # 3. 结构势能：鼓励拓扑上合理的分区
        structural_potential = self._compute_structural_potential(partition)

        # 4. 平衡势能：鼓励各分区间的平衡
        balance_potential = self._compute_balance_potential(partition)

        # 加权组合各维度势能
        total_potential = (
            0.3 * progress_potential +    # 进度权重
            0.3 * quality_potential +     # 质量权重
            0.2 * structural_potential +  # 结构权重
            0.2 * balance_potential       # 平衡权重
        )

        return total_potential

    def _compute_progress_potential(self, partition: torch.Tensor) -> float:
        """
        计算进度势能：基于分区完成度
        """
        assigned_nodes = (partition > 0).sum().item()
        total_nodes = len(partition)
        progress = assigned_nodes / total_nodes

        # 使用S型函数，早期和后期进度更有价值
        progress_potential = 2.0 / (1.0 + np.exp(-6.0 * (progress - 0.5)))

        return progress_potential

    def _compute_quality_potential(self, partition: torch.Tensor) -> float:
        """
        计算质量势能：基于分区质量指标
        """
        if (partition > 0).sum() <= 1:
            return 0.0

        # 负载平衡质量
        load_cv = self._compute_load_cv(partition)
        load_quality = 1.0 / (1.0 + 2.0 * load_cv)

        # 连通性质量（如果有边信息）
        connectivity_quality = self._compute_connectivity_quality(partition)

        # 组合质量势能
        quality_potential = 0.6 * load_quality + 0.4 * connectivity_quality

        return quality_potential

    def _compute_structural_potential(self, partition: torch.Tensor) -> float:
        """
        计算结构势能：基于图拓扑结构
        """
        if self.all_edges.shape[1] == 0:
            return 0.5  # 无边信息时返回中性值

        # 计算分区内边与跨分区边的比例
        intra_partition_edges = 0
        total_valid_edges = 0

        for i in range(self.all_edges.shape[1]):
            node1, node2 = self.all_edges[0, i].item(), self.all_edges[1, i].item()
            if (node1 < len(partition) and node2 < len(partition) and
                partition[node1] > 0 and partition[node2] > 0):
                total_valid_edges += 1
                if partition[node1] == partition[node2]:
                    intra_partition_edges += 1

        if total_valid_edges == 0:
            return 0.5

        # 分区内边比例越高，结构势能越高
        intra_ratio = intra_partition_edges / total_valid_edges
        structural_potential = intra_ratio

        return structural_potential

    def _compute_balance_potential(self, partition: torch.Tensor) -> float:
        """
        计算平衡势能：基于分区间的平衡性
        """
        num_partitions = partition.max().item()
        if num_partitions <= 1:
            return 0.0

        # 计算各分区的大小
        partition_sizes = []
        for i in range(1, num_partitions + 1):
            size = (partition == i).sum().item()
            if size > 0:
                partition_sizes.append(size)

        if len(partition_sizes) <= 1:
            return 0.0

        # 计算大小分布的均匀性
        mean_size = np.mean(partition_sizes)
        size_variance = np.var(partition_sizes)

        # 方差越小，平衡性越好
        balance_potential = 1.0 / (1.0 + size_variance / (mean_size + 1e-8))

        return balance_potential

    def _compute_connectivity_quality(self, partition: torch.Tensor) -> float:
        """
        计算连通性质量
        """
        if self.all_edges.shape[1] == 0:
            return 1.0

        # 简化的连通性评估：基于跨分区边的权重
        cross_partition_weight = 0.0
        total_weight = 0.0

        for i in range(self.all_edges.shape[1]):
            node1, node2 = self.all_edges[0, i].item(), self.all_edges[1, i].item()
            if (node1 < len(partition) and node2 < len(partition) and
                partition[node1] > 0 and partition[node2] > 0):

                edge_weight = 1.0  # 简化：所有边权重相等
                total_weight += edge_weight

                if partition[node1] != partition[node2]:
                    cross_partition_weight += edge_weight

        if total_weight == 0:
            return 1.0

        # 跨分区权重比例越小，连通性质量越高
        cross_ratio = cross_partition_weight / total_weight
        connectivity_quality = 1.0 - cross_ratio

        return connectivity_quality

    # ==================== 第三阶段：物理约束与精细化奖励 ====================

    def _compute_neighbor_consistency_reward(self, current_partition: torch.Tensor) -> float:
        """
        【第三阶段核心】邻居一致性奖励

        实现电力系统物理约束的精细化建模：
        1. 电气距离一致性：相邻节点应在同一分区
        2. 阻抗权重一致性：考虑线路阻抗的影响
        3. 功率流一致性：考虑功率传输的方向性
        4. 电压等级一致性：相同电压等级的节点优先聚类

        这些约束确保分区结果符合电力系统的物理特性
        """
        if self.all_edges.shape[1] == 0:
            return 1.0

        # 1. 基础邻居一致性
        basic_consistency = self._compute_basic_neighbor_consistency(current_partition)

        # 2. 阻抗加权一致性
        impedance_consistency = self._compute_impedance_weighted_consistency(current_partition)

        # 3. 电气距离一致性
        electrical_consistency = self._compute_electrical_distance_consistency(current_partition)

        # 4. 负载类型一致性
        load_type_consistency = self._compute_load_type_consistency(current_partition)

        # 加权组合各种一致性指标
        total_consistency = (
            0.3 * basic_consistency +      # 基础拓扑一致性
            0.3 * impedance_consistency +  # 阻抗权重一致性
            0.2 * electrical_consistency + # 电气距离一致性
            0.2 * load_type_consistency    # 负载类型一致性
        )

        return total_consistency

    def _compute_basic_neighbor_consistency(self, current_partition: torch.Tensor) -> float:
        """
        计算基础邻居一致性：直接相邻的节点在同一分区的比例
        """
        if self.all_edges.shape[1] == 0:
            return 1.0

        consistent_edges = 0
        total_valid_edges = 0

        for i in range(self.all_edges.shape[1]):
            node1, node2 = self.all_edges[0, i].item(), self.all_edges[1, i].item()
            if (node1 < len(current_partition) and node2 < len(current_partition) and
                current_partition[node1] > 0 and current_partition[node2] > 0):
                total_valid_edges += 1
                if current_partition[node1] == current_partition[node2]:
                    consistent_edges += 1

        if total_valid_edges == 0:
            return 1.0

        return consistent_edges / total_valid_edges

    def _compute_impedance_weighted_consistency(self, current_partition: torch.Tensor) -> float:
        """
        计算阻抗加权一致性：考虑线路阻抗的影响
        低阻抗线路连接的节点更应该在同一分区
        """
        if self.all_edges.shape[1] == 0 or self.all_edge_admittances.shape[0] == 0:
            return 1.0

        weighted_consistent = 0.0
        total_weight = 0.0

        for i in range(self.all_edges.shape[1]):
            node1, node2 = self.all_edges[0, i].item(), self.all_edges[1, i].item()
            if (node1 < len(current_partition) and node2 < len(current_partition) and
                current_partition[node1] > 0 and current_partition[node2] > 0):

                # 使用导纳作为权重（导纳越大，电气连接越紧密）
                if i < len(self.all_edge_admittances):
                    weight = self.all_edge_admittances[i].item()
                else:
                    weight = 1.0

                total_weight += weight

                if current_partition[node1] == current_partition[node2]:
                    weighted_consistent += weight

        if total_weight == 0:
            return 1.0

        return weighted_consistent / total_weight

    def _compute_electrical_distance_consistency(self, current_partition: torch.Tensor) -> float:
        """
        计算电气距离一致性：基于多跳邻居的一致性
        不仅考虑直接相邻，还考虑电气上接近的节点
        """
        # 简化实现：考虑2跳邻居的一致性
        if self.all_edges.shape[1] == 0:
            return 1.0

        # 构建邻接矩阵
        num_nodes = len(current_partition)
        adjacency = torch.zeros((num_nodes, num_nodes), device=self.device)

        for i in range(self.all_edges.shape[1]):
            node1, node2 = self.all_edges[0, i].item(), self.all_edges[1, i].item()
            if node1 < num_nodes and node2 < num_nodes:
                adjacency[node1, node2] = 1.0
                adjacency[node2, node1] = 1.0

        # 计算2跳邻居的一致性
        two_hop_adjacency = torch.matmul(adjacency, adjacency)

        consistent_pairs = 0
        total_pairs = 0

        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if (two_hop_adjacency[i, j] > 0 and
                    current_partition[i] > 0 and current_partition[j] > 0):
                    total_pairs += 1
                    if current_partition[i] == current_partition[j]:
                        consistent_pairs += 1

        if total_pairs == 0:
            return 1.0

        return consistent_pairs / total_pairs

    def _compute_load_type_consistency(self, current_partition: torch.Tensor) -> float:
        """
        计算负载类型一致性：相似负载特性的节点应在同一分区
        """
        # 基于节点特征中的负载信息计算一致性
        if not hasattr(self, 'node_loads') or len(self.node_loads) == 0:
            return 1.0

        num_partitions = current_partition.max().item()
        if num_partitions <= 1:
            return 1.0

        # 计算各分区内负载的变异系数
        partition_consistency = []

        for i in range(1, num_partitions + 1):
            mask = (current_partition == i)
            if mask.sum() > 1:  # 分区内至少有2个节点
                partition_loads = self.node_loads[mask]
                load_mean = partition_loads.mean()
                load_std = partition_loads.std()

                if load_mean > 0:
                    cv = load_std / load_mean
                    consistency = 1.0 / (1.0 + cv)  # CV越小，一致性越高
                    partition_consistency.append(consistency)

        if not partition_consistency:
            return 1.0

        return np.mean(partition_consistency)

    # ==================== 第三阶段：动态自适应权重系统 ====================

    def _get_adaptive_weights(self) -> Dict[str, float]:
        """
        【第三阶段核心】动态自适应权重系统

        实现训练进度感知的权重调整策略：
        1. 探索-利用平衡：训练初期重视探索，后期重视利用
        2. 粗糙-精细转换：从粗粒度奖励逐步转向精细化奖励
        3. 阶段性重点：不同训练阶段强调不同的优化目标
        4. 性能自适应：根据当前性能动态调整权重

        这种自适应机制确保训练过程的最优化
        """
        # 计算训练进度（0到1）
        progress = min(self.episode_count / 1000.0, 1.0)

        # 获取基础权重
        adaptive_weights = self.weights.copy()

        # 1. 探索-利用平衡调整
        adaptive_weights = self._adjust_exploration_exploitation_balance(adaptive_weights, progress)

        # 2. 粗糙-精细转换调整
        adaptive_weights = self._adjust_coarse_to_fine_transition(adaptive_weights, progress)

        # 3. 阶段性重点调整
        adaptive_weights = self._adjust_stage_specific_focus(adaptive_weights, progress)

        # 4. 归一化权重确保总和合理
        adaptive_weights = self._normalize_adaptive_weights(adaptive_weights)

        return adaptive_weights

    def _adjust_exploration_exploitation_balance(self, weights: Dict[str, float], progress: float) -> Dict[str, float]:
        """
        调整探索-利用平衡

        训练策略：
        - 初期（0-30%）：高探索，鼓励发现新状态和动作
        - 中期（30-70%）：平衡探索和利用
        - 后期（70-100%）：重利用，专注于优化已知好策略
        """
        # 探索权重衰减曲线
        if progress < 0.3:
            exploration_factor = 1.0  # 初期保持高探索
        elif progress < 0.7:
            exploration_factor = 1.0 - 0.5 * (progress - 0.3) / 0.4  # 中期线性衰减
        else:
            exploration_factor = 0.5 * (1.0 - (progress - 0.7) / 0.3)  # 后期快速衰减

        # 应用探索权重调整
        exploration_components = ['novelty_bonus', 'diversity_bonus', 'exploration_depth']
        for comp in exploration_components:
            if comp in weights:
                weights[comp] *= exploration_factor

        # 利用权重增长曲线（与探索相反）
        exploitation_factor = 1.0 + progress * 0.5  # 随进度增长

        # 应用利用权重调整
        exploitation_components = ['load_balance', 'electrical_decoupling', 'power_balance']
        for comp in exploitation_components:
            if comp in weights:
                weights[comp] *= exploitation_factor

        return weights

    def _adjust_coarse_to_fine_transition(self, weights: Dict[str, float], progress: float) -> Dict[str, float]:
        """
        调整粗糙-精细转换

        训练策略：
        - 初期：重视粗粒度奖励（边界压缩、进度奖励）
        - 中期：平衡粗细粒度奖励
        - 后期：重视精细化奖励（邻居一致性、物理约束）
        """
        # 粗粒度奖励权重衰减
        coarse_factor = 1.0 - 0.3 * progress  # 最多衰减30%

        coarse_components = ['boundary_compression', 'incremental_balance']
        for comp in coarse_components:
            if comp in weights:
                weights[comp] *= coarse_factor

        # 精细化奖励权重增长
        fine_factor = 1.0 + progress * 1.0  # 最多增长100%

        fine_components = ['neighbor_consistency', 'local_connectivity']
        for comp in fine_components:
            if comp in weights:
                weights[comp] *= fine_factor

        return weights

    def _adjust_stage_specific_focus(self, weights: Dict[str, float], progress: float) -> Dict[str, float]:
        """
        调整阶段性重点

        训练阶段划分：
        - 第一阶段（0-25%）：重点是稠密奖励和基础探索
        - 第二阶段（25-50%）：重点是智能探索和势函数引导
        - 第三阶段（50-75%）：重点是物理约束和质量优化
        - 第四阶段（75-100%）：重点是精细调优和稳定性
        """
        if progress < 0.25:
            # 第一阶段：稠密奖励重点
            stage_adjustments = {
                'local_connectivity': 1.2,
                'incremental_balance': 1.2,
                'boundary_compression': 1.3
            }
        elif progress < 0.5:
            # 第二阶段：探索和塑造重点
            stage_adjustments = {
                'potential_shaping': 1.3,
                'novelty_bonus': 1.1,
                'diversity_bonus': 1.1
            }
        elif progress < 0.75:
            # 第三阶段：物理约束重点
            stage_adjustments = {
                'neighbor_consistency': 1.4,
                'electrical_decoupling': 1.2,
                'power_balance': 1.2
            }
        else:
            # 第四阶段：质量优化重点
            stage_adjustments = {
                'load_balance': 1.3,
                'electrical_decoupling': 1.3,
                'neighbor_consistency': 1.2
            }

        # 应用阶段性调整
        for comp, factor in stage_adjustments.items():
            if comp in weights:
                weights[comp] *= factor

        return weights

    def _normalize_adaptive_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        归一化自适应权重

        确保权重总和在合理范围内，避免奖励尺度失控
        """
        # 计算当前权重总和
        total_weight = sum(weights.values())

        if total_weight <= 0:
            return weights

        # 目标权重总和（与原始权重总和相近）
        original_total = sum(self.weights.values())
        target_total = original_total

        # 归一化因子
        normalization_factor = target_total / total_weight

        # 应用归一化
        normalized_weights = {}
        for comp, weight in weights.items():
            normalized_weights[comp] = weight * normalization_factor

        return normalized_weights


# ==================== 向后兼容的原始类 ====================

class RewardFunction(EnhancedRewardFunction):
    """
    向后兼容的原始奖励函数类
    默认只启用基础奖励，保持与现有代码的兼容性
    """

    def __init__(self, *args, **kwargs):
        # 默认禁用所有增强功能，保持向后兼容
        kwargs.setdefault('enable_dense_rewards', False)
        kwargs.setdefault('enable_exploration_bonus', False)
        kwargs.setdefault('enable_potential_shaping', False)
        kwargs.setdefault('enable_adaptive_weights', False)
        super().__init__(*args, **kwargs)

