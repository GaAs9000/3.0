"""
自适应质量导向训练系统 - 奖励函数

实现基于势函数理论的自适应质量导向奖励系统：
1. 质量分数计算：统一的质量评估指标，支持跨网络适应性
2. 势函数主奖励：基于质量分数的相对改善，零固定阈值依赖
3. 平台期检测：智能检测质量改善平台期，支持早停机制
4. 效率奖励：在平台期激活的效率激励，鼓励快速收敛

核心特性：
- 完全相对化：自动适应不同网络的质量水平
- 平台期检测：基于改善率、稳定性和历史表现的综合判断
- 效率激励：仅在质量平台期激活，避免质量牺牲
- 数值稳定性：全面的NaN/inf保护和异常处理

作者：Augment Agent
日期：2025-07-01
"""

import torch
import numpy as np
import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
from torch_geometric.data import HeteroData
from .plateau_detector import QualityPlateauDetector, PlateauResult


class RewardFunction:
    """
    自适应质量导向奖励函数系统

    实现基于势函数理论的自适应质量导向激励结构：
    1. 质量分数计算：统一的质量评估指标，支持跨网络适应性
    2. 势函数主奖励：基于质量分数的相对改善，零固定阈值依赖
    3. 平台期检测：智能检测质量改善平台期，支持早停机制
    4. 效率奖励：在平台期激活的效率激励，鼓励快速收敛

    核心组件：
    - _compute_quality_score(): 统一质量分数计算
    - plateau_detector: 平台期检测器
    - previous_quality_score: 前一步质量分数缓存
    - 数值稳定性保护：所有数学运算都有epsilon保护
    """

    def __init__(self,
                 hetero_data: HeteroData,
                 config: Dict[str, Any] = None,
                 device: torch.device = None):
        """
        初始化自适应质量导向奖励函数

        Args:
            hetero_data: 异构图数据
            config: 配置字典，包含权重、阈值等参数
            device: 计算设备
        """
        self.device = device or torch.device('cpu')
        self.hetero_data = hetero_data.to(self.device)
        self.config = config or {}

        # 从配置中获取权重和自适应质量配置
        self.weights = self._load_weights()
        self.adaptive_quality_config = self._load_adaptive_quality_config()

        # 初始化平台期检测器
        self.plateau_detector = self._create_plateau_detector()

        # 前一步质量分数缓存
        self.previous_quality_score = None
        self.previous_metrics = None  # 保持向后兼容

        # 当前步数（用于效率奖励计算）
        self.current_step = 0
        self.max_steps = self.config.get('max_steps', 200)

        # 预计算常用数据
        self._setup_cached_data()

        # 数值稳定性参数
        self.epsilon = 1e-9
        
    def _load_weights(self) -> Dict[str, float]:
        """从配置加载权重参数"""
        default_weights = {
            # 质量分数权重（替代原有固定阈值）
            'cv_weight': 0.4,
            'coupling_weight': 0.3,
            'power_weight': 0.3,
            # 保留终局奖励权重以向后兼容
            'final_balance_weight': 0.4,
            'final_decoupling_weight': 0.4,
            'final_power_weight': 0.2,
        }

        adaptive_quality_config = self.config.get('adaptive_quality', {})
        quality_weights = adaptive_quality_config.get('quality_weights', {})

        # 合并配置，优先级：adaptive_quality > default
        return {**default_weights, **quality_weights}

    def _load_adaptive_quality_config(self) -> Dict[str, Any]:
        """从配置加载自适应质量配置"""
        default_config = {
            'plateau_detection': {
                'window_size': 15,
                'min_improvement_rate': 0.005,
                'stability_threshold': 0.8,
                'min_percentile': 0.7,
                'confidence_threshold': 0.8
            },
            'efficiency_reward': {
                'lambda': 0.5,
                'early_stop_confidence': 0.85
            }
        }

        config_adaptive = self.config.get('adaptive_quality', {})

        # 深度合并配置
        result = default_config.copy()
        if config_adaptive:
            if 'plateau_detection' in config_adaptive:
                result['plateau_detection'].update(config_adaptive['plateau_detection'])

            if 'efficiency_reward' in config_adaptive:
                result['efficiency_reward'].update(config_adaptive['efficiency_reward'])

        return result

    def _create_plateau_detector(self) -> QualityPlateauDetector:
        """创建平台期检测器"""
        plateau_config = self.adaptive_quality_config['plateau_detection']
        return QualityPlateauDetector(
            window_size=plateau_config['window_size'],
            min_improvement_rate=plateau_config['min_improvement_rate'],
            stability_threshold=plateau_config['stability_threshold'],
            min_percentile=plateau_config['min_percentile'],
            confidence_threshold=plateau_config['confidence_threshold']
        )
        
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

    def _compute_quality_score(self, partition: torch.Tensor) -> float:
        """
        计算统一质量分数

        基于技术方案中的公式：
        Q(s) = 1 - normalize(w₁·CV + w₂·coupling_ratio + w₃·power_imbalance)

        Args:
            partition: 当前分区方案 [num_nodes]

        Returns:
            质量分数 [0, 1]，越大越好
        """
        try:
            # 计算核心指标
            metrics = self._compute_core_metrics(partition)

            cv = metrics.get('cv', 1.0)
            coupling_ratio = metrics.get('coupling_ratio', 1.0)
            power_imbalance = metrics.get('power_imbalance_normalized', 1.0)

            # 归一化处理
            normalized_cv = cv / (1 + cv)  # 映射到 [0, 1)
            normalized_coupling = coupling_ratio  # 已经在 [0, 1]
            normalized_power = power_imbalance / (1 + power_imbalance)  # 映射到 [0, 1)

            # 加权组合（越小越好的指标）
            composite_badness = (
                self.weights['cv_weight'] * normalized_cv +
                self.weights['coupling_weight'] * normalized_coupling +
                self.weights['power_weight'] * normalized_power
            )

            # 归一化到权重总和
            total_weight = (
                self.weights['cv_weight'] +
                self.weights['coupling_weight'] +
                self.weights['power_weight']
            )

            if total_weight > 0:
                composite_badness = composite_badness / total_weight

            # 转换为质量分数（越大越好）
            quality_score = 1.0 - composite_badness

            # 数值稳定性保护
            if np.isnan(quality_score) or np.isinf(quality_score):
                quality_score = 0.0

            return np.clip(quality_score, 0.0, 1.0)

        except Exception as e:
            print(f"警告：质量分数计算出现异常: {e}")
            return 0.0
            
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
                                 action: Tuple[int, int]) -> Tuple[float, Optional[PlateauResult]]:
        """
        计算自适应质量导向即时奖励

        实现基于质量分数的势函数奖励：
        主奖励 = γ * Q(s_{t+1}) - Q(s_t)
        效率奖励 = λ * (max_steps - current_step) / max_steps (仅在平台期激活)

        Args:
            current_partition: 当前分区状态
            action: 执行的动作 (node_idx, partition_id)

        Returns:
            (总奖励, 平台期检测结果)
        """
        # 更新步数
        self.current_step += 1

        # 计算当前质量分数
        current_quality_score = self._compute_quality_score(current_partition)

        # 如果没有前一步质量分数，初始化并返回0
        if self.previous_quality_score is None:
            self.previous_quality_score = current_quality_score
            self.previous_metrics = self._compute_core_metrics(current_partition)  # 保持向后兼容

            # 更新平台期检测器
            plateau_result = self.plateau_detector.update(current_quality_score)
            return 0.0, plateau_result

        try:
            # 1. 计算主奖励（势函数奖励）
            gamma = 0.99  # 折扣因子
            main_reward = gamma * current_quality_score - self.previous_quality_score

            # 数值稳定性保护
            if np.isnan(main_reward) or np.isinf(main_reward):
                main_reward = 0.0
            else:
                main_reward = np.clip(main_reward, -1.0, 1.0)

            # 2. 平台期检测和效率奖励
            efficiency_reward = 0.0

            # 更新平台期检测器
            plateau_result = self.plateau_detector.update(current_quality_score)

            # 如果检测到平台期且置信度足够高，激活效率奖励
            if (plateau_result.plateau_detected and
                plateau_result.confidence > self.adaptive_quality_config['efficiency_reward']['early_stop_confidence']):

                lambda_efficiency = self.adaptive_quality_config['efficiency_reward']['lambda']
                efficiency_reward = lambda_efficiency * (self.max_steps - self.current_step) / self.max_steps
                efficiency_reward = max(0.0, efficiency_reward)  # 确保非负

            # 3. 总奖励
            total_reward = main_reward + efficiency_reward

            # 最终数值稳定性保护
            if np.isnan(total_reward) or np.isinf(total_reward):
                total_reward = 0.0
            else:
                total_reward = np.clip(total_reward, -2.0, 2.0)

        except Exception as e:
            print(f"警告：自适应质量导向奖励计算出现异常: {e}")
            total_reward = 0.0
            plateau_result = None

        # 更新前一步状态
        self.previous_quality_score = current_quality_score
        self.previous_metrics = self._compute_core_metrics(current_partition)  # 保持向后兼容

        return total_reward, plateau_result

    def reset_episode(self):
        """重置episode状态"""
        self.current_step = 0
        self.previous_quality_score = None
        self.previous_metrics = None

        self.plateau_detector.reset()

    def get_current_quality_score(self, partition: torch.Tensor) -> float:
        """获取当前质量分数（用于外部调用）"""
        return self._compute_quality_score(partition)

    def get_plateau_statistics(self) -> Dict[str, Any]:
        """获取平台期检测统计信息"""
        if self.plateau_detector is None:
            return {'enabled': False}

        stats = self.plateau_detector.get_statistics()
        stats['enabled'] = True
        return stats

    def should_early_stop(self, partition: torch.Tensor) -> Tuple[bool, float]:
        """
        判断是否应该早停

        Returns:
            (是否早停, 置信度)
        """
        # 更新质量分数并检测平台期
        quality_score = self._compute_quality_score(partition)
        plateau_result = self.plateau_detector.update(quality_score)

        early_stop_threshold = self.adaptive_quality_config['efficiency_reward']['early_stop_confidence']
        should_stop = (plateau_result.plateau_detected and
                      plateau_result.confidence > early_stop_threshold)

        return should_stop, plateau_result.confidence

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

        # 3. 阈值奖励已被自适应质量系统替代，设为0
        threshold_bonus = 0.0

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
        计算非线性阈值奖励 (已废弃，被自适应质量系统替代)

        保留此方法仅为向后兼容，实际返回0
        """
        # 固定阈值奖励已被自适应质量导向系统替代
        return 0.0

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





