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
import time
import warnings
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union
from torch_geometric.data import HeteroData
from .plateau_detector import QualityPlateauDetector, PlateauResult


class RobustMetricsCalculator:
    """
    鲁棒的指标计算器，提供智能的错误处理和诊断
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.error_history = []
        self.metrics_history = defaultdict(list)
        self.fallback_strategy = config.get('fallback_strategy', 'conservative')

    def calculate_coupling_metrics(self, edge_admittance, edge_index, partition):
        """
        计算耦合指标，带有完整的错误处理

        返回:
            dict: 包含指标值、置信度和来源信息
        """
        try:
            # 输入验证
            validation_result = self._validate_coupling_inputs(
                edge_admittance, edge_index, partition
            )

            if not validation_result['valid']:
                return self._handle_invalid_coupling_inputs(validation_result)

            # 正常计算
            cross_partition_admittance = 0.0
            total_admittance = edge_admittance.sum().item()

            for i in range(edge_index.shape[1]):
                node1, node2 = edge_index[:, i]
                if (node1 < len(partition) and node2 < len(partition) and
                    partition[node1] != partition[node2] and
                    partition[node1] > 0 and partition[node2] > 0):
                    cross_partition_admittance += edge_admittance[i].item()

            coupling_ratio = cross_partition_admittance / (total_admittance + 1e-10)
            edge_decoupling_ratio = 1.0 - coupling_ratio

            # 结果验证
            if not (0 <= coupling_ratio <= 1):
                return self._handle_out_of_range_coupling(coupling_ratio)

            # 记录成功的计算
            self._record_successful_calculation('coupling_ratio', coupling_ratio)
            self._record_successful_calculation('edge_decoupling_ratio', edge_decoupling_ratio)

            return {
                'coupling_ratio': coupling_ratio,
                'edge_decoupling_ratio': edge_decoupling_ratio,
                'confidence': 1.0,
                'source': 'calculated'
            }

        except Exception as e:
            return self._handle_coupling_calculation_error(e)

    def _validate_coupling_inputs(self, edge_admittance, edge_index, partition):
        """验证耦合计算的输入"""
        issues = []

        # 检查张量有效性
        if torch.isnan(edge_admittance).any():
            issues.append('edge_admittance contains NaN')
        if torch.isinf(edge_admittance).any():
            issues.append('edge_admittance contains Inf')

        # 检查维度匹配
        if edge_admittance.shape[0] != edge_index.shape[1]:
            issues.append(f'dimension mismatch: admittance {edge_admittance.shape[0]} vs edges {edge_index.shape[1]}')

        # 检查分区有效性
        if partition.min() < 0:
            issues.append('negative partition labels')

        # 检查导纳值合理性
        if (edge_admittance <= 0).any():
            issues.append('non-positive admittance values')

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def _handle_invalid_coupling_inputs(self, validation_result):
        """处理无效的耦合计算输入"""
        # 记录错误
        error_entry = {
            'metric': 'coupling_metrics',
            'timestamp': time.time(),
            'issues': validation_result['issues']
        }
        self.error_history.append(error_entry)

        # 根据策略选择处理方式
        if self.fallback_strategy == 'conservative':
            # 基于历史数据的保守估计
            coupling_estimate = self._get_historical_estimate('coupling_ratio', 0.3)
            edge_decoupling_estimate = 1.0 - coupling_estimate

            return {
                'coupling_ratio': coupling_estimate,
                'edge_decoupling_ratio': edge_decoupling_estimate,
                'confidence': 0.2,  # 低置信度
                'source': 'fallback_historical',
                'issues': validation_result['issues']
            }

        elif self.fallback_strategy == 'abort':
            # 抛出详细异常
            raise ValueError(
                f"无法计算耦合指标:\n" +
                "\n".join(f"  - {issue}" for issue in validation_result['issues'])
            )

        else:  # propagate_nan
            return {
                'coupling_ratio': float('nan'),
                'edge_decoupling_ratio': float('nan'),
                'confidence': 0.0,
                'source': 'error',
                'issues': validation_result['issues']
            }

    def _handle_out_of_range_coupling(self, coupling_ratio):
        """处理超出范围的耦合比率"""
        # 记录异常值
        warning_msg = f"耦合比率超出范围: {coupling_ratio:.3f}, 将被裁剪到[0,1]"
        warnings.warn(warning_msg)

        # 裁剪到合理范围
        clipped_ratio = np.clip(coupling_ratio, 0.0, 1.0)

        return {
            'coupling_ratio': clipped_ratio,
            'edge_decoupling_ratio': 1.0 - clipped_ratio,
            'confidence': 0.7,  # 中等置信度
            'source': 'clipped',
            'warning': warning_msg
        }

    def _handle_coupling_calculation_error(self, exception):
        """处理耦合计算异常"""
        error_entry = {
            'metric': 'coupling_metrics',
            'timestamp': time.time(),
            'exception': str(exception),
            'exception_type': type(exception).__name__
        }
        self.error_history.append(error_entry)

        # 使用历史数据回退
        coupling_estimate = self._get_historical_estimate('coupling_ratio', 0.4)

        return {
            'coupling_ratio': coupling_estimate,
            'edge_decoupling_ratio': 1.0 - coupling_estimate,
            'confidence': 0.1,  # 很低的置信度
            'source': 'exception_fallback',
            'exception': str(exception)
        }

    def _get_historical_estimate(self, metric_name, default_value):
        """基于历史数据获取估计值"""
        if metric_name in self.metrics_history:
            valid_history = [
                entry['value'] for entry in self.metrics_history[metric_name][-10:]
                if entry['confidence'] > 0.8
            ]
            if valid_history:
                return np.mean(valid_history)

        return default_value

    def _record_successful_calculation(self, metric_name, value):
        """记录成功的计算结果"""
        self.metrics_history[metric_name].append({
            'value': value,
            'confidence': 1.0,
            'timestamp': time.time()
        })

        # 限制历史长度
        if len(self.metrics_history[metric_name]) > 50:
            self.metrics_history[metric_name] = self.metrics_history[metric_name][-50:]


class DataIntegrityManager:
    """
    数据完整性管理器，确保数据质量和系统鲁棒性
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.missing_data_policy = config.get('missing_data_policy', 'adaptive')
        self.validation_rules = self._setup_validation_rules()
        self.completion_log = []

    def validate_hetero_data(self, hetero_data):
        """
        全面验证异构图数据
        """
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'data_quality_score': 1.0,
            'missing_features': []
        }

        # 检查必需的节点类型
        required_node_types = self.config.get('required_node_types', ['bus'])
        for node_type in required_node_types:
            if node_type not in hetero_data.x_dict:
                validation_report['errors'].append(
                    f"缺少必需的节点类型: {node_type}"
                )
                validation_report['valid'] = False

        # 检查节点特征完整性
        if 'bus' in hetero_data.x_dict:
            feature_report = self._validate_bus_features(hetero_data.x_dict['bus'])
            validation_report['warnings'].extend(feature_report['warnings'])
            validation_report['missing_features'].extend(feature_report['missing_features'])
            validation_report['data_quality_score'] *= feature_report['quality_score']

        # 检查边数据
        edge_report = self._validate_edge_data(hetero_data)
        validation_report['warnings'].extend(edge_report['warnings'])
        validation_report['data_quality_score'] *= edge_report['quality_score']

        return validation_report

    def handle_missing_data(self, hetero_data, validation_report):
        """
        根据策略处理缺失数据
        """
        if validation_report['valid'] and validation_report['data_quality_score'] > 0.8:
            # 数据质量良好，直接返回
            return hetero_data, 'original'

        if self.missing_data_policy == 'strict':
            # 严格模式：拒绝处理
            raise ValueError(
                "数据质量不满足要求:\n" +
                "\n".join(validation_report['errors']) +
                "\n建议：检查输入数据或使用 'adaptive' 策略"
            )

        elif self.missing_data_policy == 'adaptive':
            # 自适应模式：智能补全
            return self._intelligent_data_completion(hetero_data, validation_report)

        elif self.missing_data_policy == 'fallback':
            # 回退模式：使用备选数据源
            return self._use_fallback_data(hetero_data, validation_report)

    def _intelligent_data_completion(self, hetero_data, validation_report):
        """
        智能数据补全
        """
        completed_data = hetero_data.clone() if hasattr(hetero_data, 'clone') else hetero_data
        completion_log = []

        # 处理缺失的负荷数据
        if 'active_load' in validation_report['missing_features']:
            completed_data, log = self._estimate_missing_loads(completed_data)
            completion_log.extend(log)

        # 处理缺失的发电数据
        if 'active_generation' in validation_report['missing_features']:
            completed_data, log = self._estimate_missing_generation(completed_data)
            completion_log.extend(log)

        # 生成补全报告
        self._generate_completion_report(completion_log)

        return completed_data, 'completed'

    def _validate_bus_features(self, bus_features):
        """验证母线特征数据"""
        report = {
            'warnings': [],
            'missing_features': [],
            'quality_score': 1.0
        }

        expected_features = {
            0: 'active_load',      # 有功负载
            1: 'reactive_load',    # 无功负载
            2: 'active_generation', # 有功发电
        }

        # 检查特征维度
        if bus_features.shape[1] < 3:
            report['warnings'].append(
                f"母线特征维度不足: {bus_features.shape[1]} < 3"
            )
            report['quality_score'] *= 0.7

        # 检查每个特征的质量
        for col_idx, feature_name in expected_features.items():
            if col_idx < bus_features.shape[1]:
                feature_data = bus_features[:, col_idx]

                # 检查是否全为零（可能表示缺失）
                if torch.all(feature_data == 0):
                    report['missing_features'].append(feature_name)
                    report['warnings'].append(f"{feature_name}: 所有值为零，可能缺失")
                    report['quality_score'] *= 0.5

                # 检查NaN值
                nan_count = torch.isnan(feature_data).sum().item()
                if nan_count > 0:
                    report['warnings'].append(
                        f"{feature_name}: {nan_count}/{len(feature_data)} 个NaN值"
                    )
                    report['quality_score'] *= (1 - nan_count / len(feature_data))
            else:
                report['missing_features'].append(feature_name)
                report['quality_score'] *= 0.8

        return report

    def _validate_edge_data(self, hetero_data):
        """验证边数据"""
        report = {
            'warnings': [],
            'quality_score': 1.0
        }

        edge_type = ('bus', 'connects', 'bus')

        # 检查边索引
        if edge_type not in hetero_data.edge_index_dict:
            report['warnings'].append("缺少边索引数据")
            report['quality_score'] = 0.1
            return report

        # 检查边属性
        if edge_type not in hetero_data.edge_attr_dict:
            report['warnings'].append("缺少边属性数据")
            report['quality_score'] *= 0.5

        return report

    def _estimate_missing_loads(self, hetero_data):
        """基于统计模型估算缺失的负荷数据"""
        completion_log = []

        if 'bus' in hetero_data.x_dict:
            bus_features = hetero_data.x_dict['bus']
            num_buses = bus_features.shape[0]

            # 基于节点度数估算负荷
            node_degrees = self._calculate_node_degrees(hetero_data)

            # 典型负荷密度：10-50 MW per node
            base_load = 20.0  # MW

            # 基于度数的负荷分配
            estimated_loads = base_load * (node_degrees / node_degrees.mean())

            # 添加随机扰动
            noise = torch.randn_like(estimated_loads) * 5.0
            estimated_loads = torch.clamp(estimated_loads + noise, 5.0, 100.0)

            # 更新特征
            if bus_features.shape[1] > 0:
                bus_features[:, 0] = estimated_loads

            completion_log.append({
                'type': 'load_estimation',
                'method': 'degree_based',
                'count': num_buses,
                'mean_value': estimated_loads.mean().item()
            })

        return hetero_data, completion_log

    def _estimate_missing_generation(self, hetero_data):
        """估算缺失的发电数据"""
        completion_log = []

        if 'bus' in hetero_data.x_dict:
            bus_features = hetero_data.x_dict['bus']
            num_buses = bus_features.shape[0]

            # 假设20%的节点有发电机
            num_generators = max(1, num_buses // 5)

            # 随机选择发电机节点
            generator_indices = torch.randperm(num_buses)[:num_generators]

            # 估算发电容量
            total_load = bus_features[:, 0].sum() if bus_features.shape[1] > 0 else num_buses * 20.0
            total_generation = total_load * 1.2  # 120%的负荷覆盖

            generation_per_unit = total_generation / num_generators

            # 初始化发电数据
            if bus_features.shape[1] > 2:
                bus_features[:, 2] = 0.0  # 清零
                bus_features[generator_indices, 2] = generation_per_unit

            completion_log.append({
                'type': 'generation_estimation',
                'method': 'random_allocation',
                'generator_count': num_generators,
                'total_generation': total_generation.item() if hasattr(total_generation, 'item') else total_generation
            })

        return hetero_data, completion_log

    def _calculate_node_degrees(self, hetero_data):
        """计算节点度数"""
        edge_type = ('bus', 'connects', 'bus')

        if edge_type not in hetero_data.edge_index_dict:
            # 如果没有边数据，返回均匀度数
            num_nodes = hetero_data.x_dict['bus'].shape[0]
            return torch.ones(num_nodes) * 2.0

        edge_index = hetero_data.edge_index_dict[edge_type]
        num_nodes = hetero_data.x_dict['bus'].shape[0]

        degrees = torch.zeros(num_nodes)
        for i in range(edge_index.shape[1]):
            degrees[edge_index[0, i]] += 1
            degrees[edge_index[1, i]] += 1

        return degrees

    def _generate_completion_report(self, completion_log):
        """生成数据补全报告"""
        if not completion_log:
            return

        # 检查是否启用简洁模式
        try:
            from code.src.rich_output import rich_debug
            rich_debug("数据补全报告:", category="scenario")
            for entry in completion_log:
                rich_debug(f"  - {entry['type']}: {entry['method']}", category="scenario")
                if 'count' in entry:
                    rich_debug(f"    处理节点数: {entry['count']}", category="scenario")
                if 'mean_value' in entry:
                    rich_debug(f"    平均值: {entry['mean_value']:.2f}", category="scenario")
        except ImportError:
            # 备用标准输出
            print("📋 数据补全报告:")
            for entry in completion_log:
                print(f"  - {entry['type']}: {entry['method']}")
                if 'count' in entry:
                    print(f"    处理节点数: {entry['count']}")
                if 'mean_value' in entry:
                    print(f"    平均值: {entry['mean_value']:.2f}")

    def _setup_validation_rules(self):
        """设置验证规则"""
        return {
            'min_data_quality_score': self.config.get('min_data_quality_score', 0.5),
            'required_features': self.config.get('required_features', ['active_load']),
            'allow_missing_generation': self.config.get('allow_missing_generation', True)
        }


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
        self.config = config or {}

        # 数据完整性检查和处理
        self.data_manager = DataIntegrityManager(
            config=self.config.get('data_integrity', {
                'missing_data_policy': 'adaptive',
                'min_data_quality_score': 0.3
            })
        )

        # 验证和处理数据
        validation_report = self.data_manager.validate_hetero_data(hetero_data)

        if validation_report['data_quality_score'] < 0.2:
            raise ValueError(
                f"数据质量分数 {validation_report['data_quality_score']:.2f} 过低\n"
                f"错误: {validation_report['errors']}\n"
                f"警告: {validation_report['warnings']}"
            )

        # 处理缺失数据
        self.hetero_data, data_source = self.data_manager.handle_missing_data(
            hetero_data, validation_report
        )
        self.hetero_data = self.hetero_data.to(self.device)

        # 记录数据来源
        if data_source != 'original':
            warning_msg = (
                f"使用 {data_source} 数据源，"
                f"质量分数: {validation_report['data_quality_score']:.2f}"
            )
            warnings.warn(warning_msg)

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

        # 鲁棒指标计算器
        self.metrics_calculator = RobustMetricsCalculator(
            config=self.config.get('error_handling', {
                'fallback_strategy': 'conservative'
            })
        )

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
        """预计算频繁使用的数据 - 改进的数据处理"""
        # 安全提取节点负载和发电数据
        self._safe_extract_power_data()

        # 边信息（用于计算导纳）
        self._extract_edge_info()

    def _safe_extract_power_data(self):
        """安全提取功率数据，避免创建虚假默认数据"""
        if 'bus' not in self.hetero_data.x_dict:
            raise ValueError(
                "缺少必需的'bus'节点数据。"
                "数据完整性检查应该在此之前捕获此问题。"
            )

        bus_features = self.hetero_data.x_dict['bus']
        num_nodes = bus_features.shape[0]

        # 提取有功负载（第0列）
        if bus_features.shape[1] > 0:
            self.node_loads = bus_features[:, 0]

            # 验证负载数据的合理性
            if torch.all(self.node_loads == 0):
                warnings.warn(
                    "所有节点负载为零，这可能表示数据缺失。"
                    "建议检查数据源或使用数据补全功能。"
                )
        else:
            raise ValueError("母线特征数据为空，无法提取负载信息")

        # 提取有功发电（第2列，如果存在）
        if bus_features.shape[1] > 2:
            self.node_generation = bus_features[:, 2]
        else:
            # 智能估算发电数据而不是简单使用零值
            self.node_generation = self._estimate_missing_generation_data()
            warnings.warn(
                "缺少发电数据，使用智能估算值。建议提供完整的发电数据以提高准确性。"
            )

        # 数据质量检查
        self._validate_power_data_quality()

    def _validate_power_data_quality(self):
        """验证功率数据质量"""
        quality_issues = []

        # 检查负载数据
        load_nan_count = torch.isnan(self.node_loads).sum().item()
        if load_nan_count > 0:
            quality_issues.append(f"负载数据包含 {load_nan_count} 个NaN值")

        load_inf_count = torch.isinf(self.node_loads).sum().item()
        if load_inf_count > 0:
            quality_issues.append(f"负载数据包含 {load_inf_count} 个Inf值")

        # 检查发电数据
        gen_nan_count = torch.isnan(self.node_generation).sum().item()
        if gen_nan_count > 0:
            quality_issues.append(f"发电数据包含 {gen_nan_count} 个NaN值")

        # 检查功率平衡
        total_load = self.node_loads.sum().item()
        total_generation = self.node_generation.sum().item()

        if total_load > 0 and total_generation > 0:
            imbalance_ratio = abs(total_generation - total_load) / total_load
            if imbalance_ratio > 0.5:  # 50%以上的不平衡
                quality_issues.append(
                    f"功率严重不平衡: 负载={total_load:.1f}, 发电={total_generation:.1f}, "
                    f"不平衡率={imbalance_ratio:.1%}"
                )

        # 记录质量问题
        if quality_issues:
            warning_msg = "功率数据质量问题:\n" + "\n".join(f"  - {issue}" for issue in quality_issues)
            warnings.warn(warning_msg)

            # 记录到数据管理器
            if hasattr(self.data_manager, 'completion_log'):
                self.data_manager.completion_log.append({
                    'type': 'power_data_quality_check',
                    'issues': quality_issues,
                    'total_load': total_load,
                    'total_generation': total_generation
                })

    def _estimate_missing_generation_data(self) -> torch.Tensor:
        """
        智能估算缺失的发电数据

        使用多种策略来估算发电数据，而不是简单的零向量：
        1. 基于负载分布的比例估算
        2. 基于节点类型的启发式估算
        3. 基于网络拓扑的估算

        Returns:
            估算的发电数据张量
        """
        num_nodes = self.node_loads.shape[0]

        # 策略1: 基于负载分布估算
        total_load = torch.sum(torch.abs(self.node_loads))

        if total_load > 0:
            # 假设总发电量略大于总负载（考虑损耗）
            total_generation_estimate = total_load * 1.05  # 5%的损耗余量

            # 基于负载分布分配发电
            load_weights = torch.abs(self.node_loads) / total_load

            # 识别可能的发电节点（高负载节点更可能有发电机）
            load_percentile_80 = torch.quantile(torch.abs(self.node_loads), 0.8)
            potential_gen_mask = torch.abs(self.node_loads) >= load_percentile_80

            # 初始化发电数据
            estimated_generation = torch.zeros_like(self.node_loads)

            if potential_gen_mask.any():
                # 在潜在发电节点中分配发电量
                gen_weights = load_weights[potential_gen_mask]
                gen_weights = gen_weights / gen_weights.sum()

                estimated_generation[potential_gen_mask] = total_generation_estimate * gen_weights
            else:
                # 如果没有明显的发电节点，均匀分配到负载较大的节点
                top_load_indices = torch.topk(torch.abs(self.node_loads),
                                            min(3, num_nodes)).indices
                estimated_generation[top_load_indices] = total_generation_estimate / len(top_load_indices)

        else:
            # 如果没有负载数据，使用基于拓扑的估算
            estimated_generation = self._estimate_generation_from_topology()

        # 策略2: 基于节点特征的调整（如果有节点类型信息）
        if hasattr(self.hetero_data, 'x_dict') and 'bus' in self.hetero_data.x_dict:
            bus_features = self.hetero_data.x_dict['bus']

            # 检查是否有节点类型信息（通常在特征的后几列）
            if bus_features.shape[1] > 10:  # 假设有足够的特征列
                # 尝试识别PV节点和平衡节点（它们通常有发电机）
                # 这是一个启发式方法，基于特征模式
                for i in range(max(0, bus_features.shape[1] - 5), bus_features.shape[1]):
                    feature_col = bus_features[:, i]

                    # 检查是否是二进制特征（可能表示节点类型）
                    unique_values = torch.unique(feature_col)
                    if len(unique_values) <= 3 and torch.all((unique_values >= 0) & (unique_values <= 3)):
                        # 可能是节点类型特征
                        # PV节点(2)和平衡节点(3)通常有发电机
                        gen_node_mask = (feature_col == 2) | (feature_col == 3)

                        if gen_node_mask.any():
                            # 调整这些节点的发电量
                            boost_factor = 1.5
                            estimated_generation[gen_node_mask] *= boost_factor

        # 策略3: 数值合理性检查和调整
        estimated_generation = self._validate_and_adjust_generation(estimated_generation)

        return estimated_generation

    def _estimate_generation_from_topology(self) -> torch.Tensor:
        """基于网络拓扑估算发电分布"""
        num_nodes = self.node_loads.shape[0]

        if hasattr(self, 'edge_index') and self.edge_index.shape[1] > 0:
            # 计算节点度数
            node_degrees = torch.zeros(num_nodes, device=self.device)

            for i in range(self.edge_index.shape[1]):
                node1, node2 = self.edge_index[:, i]
                if node1 < num_nodes:
                    node_degrees[node1] += 1
                if node2 < num_nodes:
                    node_degrees[node2] += 1

            # 高度数节点更可能是发电节点
            degree_percentile_70 = torch.quantile(node_degrees, 0.7)
            high_degree_mask = node_degrees >= degree_percentile_70

            estimated_generation = torch.zeros_like(self.node_loads)

            if high_degree_mask.any():
                # 基于度数分配发电量
                degree_weights = node_degrees[high_degree_mask]
                degree_weights = degree_weights / degree_weights.sum()

                # 假设适度的总发电量
                total_gen_estimate = num_nodes * 0.1  # 每个节点平均0.1 p.u.
                estimated_generation[high_degree_mask] = total_gen_estimate * degree_weights

            return estimated_generation
        else:
            # 没有拓扑信息，使用最保守的估算
            return torch.zeros(num_nodes, device=self.device)

    def _validate_and_adjust_generation(self, generation: torch.Tensor) -> torch.Tensor:
        """验证和调整发电数据的合理性"""
        # 1. 确保非负值
        generation = torch.clamp(generation, min=0.0)

        # 2. 检查总发电量是否合理
        total_load = torch.sum(torch.abs(self.node_loads))
        total_generation = torch.sum(generation)

        if total_load > 0:
            # 发电量应该在负载的90%-120%之间
            min_gen = total_load * 0.9
            max_gen = total_load * 1.2

            if total_generation < min_gen:
                # 发电量太少，按比例增加
                scale_factor = min_gen / (total_generation + 1e-9)
                generation = generation * scale_factor
            elif total_generation > max_gen:
                # 发电量太多，按比例减少
                scale_factor = max_gen / total_generation
                generation = generation * scale_factor

        # 3. 限制单个节点的发电量
        max_single_gen = torch.sum(generation) * 0.4  # 单个节点最多40%的总发电量
        generation = torch.clamp(generation, max=max_single_gen)

        # 4. 确保数值稳定性
        generation = torch.where(torch.isnan(generation),
                               torch.zeros_like(generation),
                               generation)
        generation = torch.where(torch.isinf(generation),
                               torch.zeros_like(generation),
                               generation)

        return generation

    def _extract_edge_info(self):
        """智能提取边信息用于导纳计算"""
        if ('bus', 'connects', 'bus') in self.hetero_data.edge_index_dict:
            self.edge_index = self.hetero_data.edge_index_dict[('bus', 'connects', 'bus')]

            # 智能提取导纳数据
            self.edge_admittance, admittance_source = self._extract_admittance_data()

            # 记录数据来源
            if hasattr(self, 'logger'):
                if admittance_source != 'extracted':
                    self.logger.warning(f"导纳数据来源: {admittance_source}")
            elif admittance_source != 'extracted':
                print(f"⚠️ 导纳数据来源: {admittance_source}")

        else:
            # 如果没有边信息，创建空的
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.edge_admittance = torch.empty(0, device=self.device)

    def _extract_admittance_data(self):
        """
        智能提取导纳数据，按优先级尝试多种方法

        返回:
            admittance: 导纳张量
            source: 数据来源标识
        """
        edge_type = ('bus', 'connects', 'bus')

        # 方法1: 从边属性直接提取
        if edge_type in self.hetero_data.edge_attr_dict:
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]

            # 尝试识别导纳数据
            admittance = self._identify_admittance_from_attributes(edge_attr)
            if admittance is not None:
                return admittance, 'extracted'

        # 方法2: 从阻抗计算导纳
        if edge_type in self.hetero_data.edge_attr_dict:
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            admittance = self._calculate_admittance_from_impedance(edge_attr)
            if admittance is not None:
                return admittance, 'calculated'

        # 方法3: 基于物理模型估算
        admittance = self._estimate_admittance_from_topology()
        if admittance is not None:
            return admittance, 'estimated'

        # 方法4: 保守默认值（带警告）
        return self._get_conservative_admittance_defaults(), 'default'

    def _identify_admittance_from_attributes(self, edge_attr):
        """
        从边属性中智能识别导纳数据

        根据data_processing.py中的特征顺序：
        [r, x, b, z_magnitude, y, rateA, angle_diff, is_transformer, status]
        导纳y在第4列（索引4）
        """
        if edge_attr.shape[1] <= 4:
            return None

        # 提取导纳列（第4列）
        admittance_candidate = edge_attr[:, 4]

        # 检查导纳值的合理性
        # 导纳应该是正值，且在合理范围内
        if self._validate_admittance_values(admittance_candidate):
            # 取绝对值确保为正
            return torch.abs(admittance_candidate)

        # 如果第4列不合理，尝试其他可能的列
        for col_idx in [3, 5]:  # z_magnitude, rateA
            if edge_attr.shape[1] > col_idx:
                candidate = edge_attr[:, col_idx]
                if self._validate_admittance_values(candidate):
                    return torch.abs(candidate)

        return None

    def _calculate_admittance_from_impedance(self, edge_attr):
        """
        从阻抗参数计算导纳

        特征顺序：[r, x, b, z_magnitude, y, ...]
        """
        if edge_attr.shape[1] < 2:
            return None

        # 提取电阻和电抗
        r = edge_attr[:, 0]  # 电阻
        x = edge_attr[:, 1]  # 电抗

        # 计算阻抗模长
        z_squared = r**2 + x**2

        # 避免除零，设置最小阻抗值
        min_impedance = 1e-6
        z_magnitude = torch.sqrt(z_squared.clamp(min=min_impedance**2))

        # 计算导纳 Y = 1/Z
        admittance = 1.0 / z_magnitude

        # 验证结果
        if self._validate_admittance_values(admittance):
            return admittance

        return None

    def _estimate_admittance_from_topology(self):
        """
        基于网络拓扑估算导纳
        """
        num_edges = self.edge_index.shape[1]
        if num_edges == 0:
            return torch.empty(0, device=self.device)

        # 计算节点度数
        node_degrees = self._calculate_node_degrees()

        # 基于连接的节点度数估算导纳
        admittance_estimates = torch.zeros(num_edges, device=self.device)

        for i in range(num_edges):
            node1, node2 = self.edge_index[:, i]

            # 基于节点度数的启发式估算
            # 高度数节点间的连接通常有更高的导纳
            avg_degree = (node_degrees[node1] + node_degrees[node2]) / 2.0

            # 典型导纳范围：0.5-5.0，基于度数调整
            base_admittance = 2.0
            degree_factor = torch.clamp(avg_degree / 3.0, 0.5, 2.0)

            admittance_estimates[i] = base_admittance * degree_factor

        # 添加随机扰动以避免完全相同的值
        noise = torch.randn_like(admittance_estimates) * 0.1
        admittance_estimates = torch.clamp(admittance_estimates + noise, 0.5, 5.0)

        return admittance_estimates

    def _calculate_node_degrees(self):
        """计算节点度数"""
        num_nodes = max(self.edge_index.max().item() + 1,
                       len(self.hetero_data.x_dict.get('bus', torch.empty(0))))

        degrees = torch.zeros(num_nodes, device=self.device)

        for i in range(self.edge_index.shape[1]):
            node1, node2 = self.edge_index[:, i]
            degrees[node1] += 1
            degrees[node2] += 1

        return degrees

    def _get_conservative_admittance_defaults(self):
        """
        获取保守的默认导纳值，并记录详细警告
        """
        num_edges = self.edge_index.shape[1]

        # 记录详细警告
        warning_msg = (
            f"无法提取或计算导纳数据，使用保守默认值。"
            f"边数: {num_edges}。"
            f"这可能影响电气解耦计算的准确性。"
            f"建议检查输入数据格式或提供导纳信息。"
        )

        if hasattr(self, 'logger'):
            self.logger.warning(warning_msg)
        else:
            print(f"⚠️ {warning_msg}")

        if num_edges == 0:
            return torch.empty(0, device=self.device)

        # 使用基于网络规模的合理默认值
        # 典型输电线路导纳范围：1-3 S
        mean_admittance = 2.0
        std_admittance = 0.3

        # 生成正态分布的导纳值
        admittance = torch.normal(
            mean=mean_admittance,
            std=std_admittance,
            size=(num_edges,),
            device=self.device
        )

        # 确保在合理范围内
        admittance = torch.clamp(torch.abs(admittance), 0.8, 4.0)

        return admittance

    def _validate_admittance_values(self, admittance):
        """
        验证导纳值的物理合理性
        """
        # 检查NaN和Inf
        if torch.isnan(admittance).any() or torch.isinf(admittance).any():
            return False

        # 检查是否全为零
        if torch.all(admittance == 0):
            return False

        # 取绝对值进行范围检查
        abs_admittance = torch.abs(admittance)

        # 检查物理合理范围（放宽范围以适应不同的标幺化）
        # 允许更大的范围：0.001-1000
        if (abs_admittance < 0.001).any() or (abs_admittance > 1000).any():
            return False

        # 检查是否有足够的非零值
        non_zero_count = torch.sum(abs_admittance > 0.001).item()
        if non_zero_count < len(admittance) * 0.5:  # 至少50%的值应该是有意义的
            return False

        return True

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

    def _compute_simple_relative_reward(self, prev_quality: float, curr_quality: float) -> float:
        """
        简单相对改进奖励 - 解决跨场景训练偏向问题

        核心思想：相同的相对努力应该获得相同的奖励幅度
        - 故障场景: (0.42-0.40)/0.40 = 5%改进 → 奖励+0.05
        - 正常场景: (0.714-0.68)/0.68 = 5%改进 → 奖励+0.05

        这确保算法不会偏向简单场景，在困难场景(故障、高负荷)下也能获得公平激励

        Args:
            prev_quality: 前一步质量分数
            curr_quality: 当前质量分数

        Returns:
            相对改进奖励 [-1.0, 1.0]
        """
        try:
            if prev_quality > 0.01:  # 避免除零，处理边界情况
                relative_improvement = (curr_quality - prev_quality) / prev_quality
            else:
                # 从零开始的情况，直接用绝对改进
                relative_improvement = curr_quality - prev_quality

            # 轻微裁剪避免极端值，保持训练稳定性
            return np.clip(relative_improvement, -1.0, 1.0)

        except Exception as e:
            print(f"警告：相对奖励计算出现异常: {e}")
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
        
        # 2. 计算电气解耦指标 - 使用鲁棒计算器
        if self.edge_index.shape[1] > 0:
            # 使用鲁棒指标计算器
            coupling_result = self.metrics_calculator.calculate_coupling_metrics(
                self.edge_admittance, self.edge_index, partition
            )

            # 提取结果
            metrics['coupling_ratio'] = coupling_result['coupling_ratio']
            metrics['edge_decoupling_ratio'] = coupling_result['edge_decoupling_ratio']

            # 记录置信度和来源信息（用于调试）
            if hasattr(self, 'logger'):
                if coupling_result['confidence'] < 0.8:
                    self.logger.warning(
                        f"耦合指标置信度较低: {coupling_result['confidence']:.2f}, "
                        f"来源: {coupling_result['source']}"
                    )
            elif coupling_result['confidence'] < 0.8:
                print(f"⚠️ 耦合指标置信度较低: {coupling_result['confidence']:.2f}, "
                      f"来源: {coupling_result['source']}")
        else:
            # 没有边的情况
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

        实现基于相对改进的奖励系统，解决跨场景训练偏向问题：
        主奖励 = 相对改进奖励 = (Q(s_{t+1}) - Q(s_t)) / Q(s_t)
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
            # 1. 计算主奖励（相对改进奖励）
            main_reward = self._compute_simple_relative_reward(
                self.previous_quality_score,
                current_quality_score
            )

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
        计算电气解耦奖励 - 改进的错误处理

        公式：R_decoupling = 0.5 * σ(5*(r_edge - 0.5)) + 0.5 * σ(5*(r_admittance - 0.5))
        其中 σ 是sigmoid函数，r_admittance = 1 - coupling_ratio
        """
        # 智能数值稳定性检查
        edge_decoupling_ratio = self._robust_value_check(
            edge_decoupling_ratio, 'edge_decoupling_ratio',
            expected_range=(0.0, 1.0), default_strategy='historical'
        )

        coupling_ratio = self._robust_value_check(
            coupling_ratio, 'coupling_ratio',
            expected_range=(0.0, 1.0), default_strategy='historical'
        )

        # 计算导纳解耦率
        admittance_decoupling_ratio = 1.0 - coupling_ratio

        try:
            # 应用sigmoid函数
            edge_component = self._sigmoid(5.0 * (edge_decoupling_ratio - 0.5))
            admittance_component = self._sigmoid(5.0 * (admittance_decoupling_ratio - 0.5))

            decoupling_reward = 0.5 * edge_component + 0.5 * admittance_component

            # 检查结果
            if np.isnan(decoupling_reward) or np.isinf(decoupling_reward):
                # 使用历史数据回退
                decoupling_reward = self._get_historical_reward_estimate('decoupling_reward', 0.5)
            else:
                decoupling_reward = np.clip(decoupling_reward, 0.0, 1.0)
                # 记录成功的计算
                self._record_reward_calculation('decoupling_reward', decoupling_reward)

        except Exception as e:
            # 记录异常并使用智能回退
            self._record_calculation_error('decoupling_reward', e)
            decoupling_reward = self._get_historical_reward_estimate('decoupling_reward', 0.4)

        return decoupling_reward

    def _robust_value_check(self, value, value_name, expected_range, default_strategy='historical'):
        """
        鲁棒的数值检查，支持多种回退策略
        """
        # 检查NaN和Inf
        if np.isnan(value) or np.isinf(value):
            if default_strategy == 'historical':
                return self._get_historical_value_estimate(value_name, expected_range[0] + 0.5 * (expected_range[1] - expected_range[0]))
            else:
                return expected_range[0] + 0.5 * (expected_range[1] - expected_range[0])  # 中点

        # 检查范围
        if not (expected_range[0] <= value <= expected_range[1]):
            # 记录超出范围的警告
            if hasattr(self, 'logger'):
                self.logger.warning(f"{value_name} 超出范围 {expected_range}: {value:.3f}")

            # 裁剪到合理范围
            return np.clip(value, expected_range[0], expected_range[1])

        return value

    def _get_historical_value_estimate(self, value_name, default_value):
        """获取历史数值估计"""
        if hasattr(self.metrics_calculator, 'metrics_history') and value_name in self.metrics_calculator.metrics_history:
            valid_history = [
                entry['value'] for entry in self.metrics_calculator.metrics_history[value_name][-5:]
                if entry['confidence'] > 0.7
            ]
            if valid_history:
                return np.mean(valid_history)

        return default_value

    def _get_historical_reward_estimate(self, reward_name, default_value):
        """获取历史奖励估计"""
        if hasattr(self, 'reward_history') and reward_name in self.reward_history:
            recent_rewards = self.reward_history[reward_name][-5:]
            if recent_rewards:
                return np.mean(recent_rewards)

        return default_value

    def _record_reward_calculation(self, reward_name, value):
        """记录成功的奖励计算"""
        if not hasattr(self, 'reward_history'):
            self.reward_history = defaultdict(list)

        self.reward_history[reward_name].append(value)

        # 限制历史长度
        if len(self.reward_history[reward_name]) > 20:
            self.reward_history[reward_name] = self.reward_history[reward_name][-20:]

    def _record_calculation_error(self, calculation_name, exception):
        """记录计算错误"""
        if not hasattr(self, 'calculation_errors'):
            self.calculation_errors = []

        self.calculation_errors.append({
            'name': calculation_name,
            'exception': str(exception),
            'timestamp': time.time()
        })

        # 限制错误历史长度
        if len(self.calculation_errors) > 50:
            self.calculation_errors = self.calculation_errors[-50:]

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





