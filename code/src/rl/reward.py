"""
自适应质量导向训练系统 - 奖励函数

实现基于势函数理论的自适应质量导向奖励系统：
1. 质量分数计算：统一的质量评估指标，支持跨网络适应性
2. 势函数主奖励：基于质量分数的相对改善，零固定阈值依赖
3. 平台期检测：智能检测质量改善平台期，支持早停机制
4. 效率奖励：在平台期激活的效率激励，鼓励快速收敛
5. 场景感知：按场景分类历史数据，支持相对改进奖励

核心特性：
- 完全相对化：自动适应不同网络的质量水平
- 平台期检测：基于改善率、稳定性和历史表现的综合判断
- 效率激励：仅在质量平台期激活，避免质量牺牲
- 场景感知：同类场景内进行比较，支持跨场景泛化
- 相对改进奖励：确保相同相对努力获得相同奖励幅度
- 数值稳定性：全面的NaN/inf保护和异常处理

作者：Augment Agent  
日期：2025-01-15
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
from .scenario_context import ScenarioContext
from .scenario_aware_tracker import ScenarioAwareHistoryTracker
from .scenario_aware_plateau_detector import ScenarioAwarePlateauDetector
from .relative_reward_calculator import RelativeImprovementReward, HybridRewardFunction
from .utils import IntelligentAdmittanceExtractor


class RobustMetricsCalculator:
    """
    计算核心物理指标，内置了丰富的错误处理和回退机制。

    该类的设计目标是提供一个极其健壮的指标计算服务，即使在输入
    数据有缺失或计算过程中出现意外错误时，也能够通过预设的策略
    （如使用历史均值）提供一个合理的估价值，从而保证上层应用的稳定运行。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化鲁棒指标计算器。

        Args:
            config (Optional[Dict[str, Any]]): 
                配置字典，可包含 `fallback_strategy` 等键。
        """
        self.config = config or {}
        self.error_history = []
        self.metrics_history = defaultdict(list)
        self.fallback_strategy = config.get('fallback_strategy', 'conservative')

    def calculate_coupling_metrics(self, 
                                 edge_admittance: torch.Tensor, 
                                 edge_index: torch.Tensor, 
                                 partition: torch.Tensor,
                                 total_admittance: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        以向量化的方式高效计算电气耦合度指标。

        Args:
            edge_admittance (torch.Tensor): 边的导纳值张量。
            edge_index (torch.Tensor): 描述边连接的张量, shape `[2, num_edges]`。
            partition (torch.Tensor): 每个节点的所属分区标签。
            total_admittance (Optional[torch.Tensor]): 预先计算好的总导纳，用于性能优化

        Returns:
            Dict[str, Any]: 包含耦合指标、置信度和来源信息的字典。
        """
        try:
            # 输入验证
            validation_result = self._validate_coupling_inputs(
                edge_admittance, edge_index, partition
            )

            if not validation_result['valid']:
                return self._handle_invalid_coupling_inputs(validation_result)

            # 高效的向量化计算
            if total_admittance is None:
                total_admittance = edge_admittance.sum()

            # 获取边的源节点和目标节点
            src_nodes, dst_nodes = edge_index

            # 获取节点所在的分区
            src_partitions = partition[src_nodes]
            dst_partitions = partition[dst_nodes]

            # 创建跨区边的掩码
            cross_partition_mask = (src_partitions != dst_partitions) & \
                                   (src_partitions > 0) & \
                                   (dst_partitions > 0)

            # 计算跨区导纳总和
            cross_partition_admittance = edge_admittance[cross_partition_mask].sum()

            # 计算耦合比率
            coupling_ratio = cross_partition_admittance / (total_admittance + 1e-10)
            edge_decoupling_ratio = 1.0 - coupling_ratio

            # 结果验证
            if not (0 <= coupling_ratio <= 1):
                return self._handle_out_of_range_coupling(coupling_ratio.item())

            # 记录成功的计算
            self._record_successful_calculation('coupling_ratio', coupling_ratio.item())
            self._record_successful_calculation('edge_decoupling_ratio', edge_decoupling_ratio.item())

            return {
                'coupling_ratio': coupling_ratio.item(),
                'edge_decoupling_ratio': edge_decoupling_ratio.item(),
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
            # 回退模式：使用智能补全作为备选方案
            return self._intelligent_data_completion(hetero_data, validation_report)

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
        from utils_common import safe_rich_debug
        safe_rich_debug("数据补全报告:", "scenario")
        for entry in completion_log:
            safe_rich_debug(f"  - {entry['type']}: {entry['method']}", "scenario")
            if 'count' in entry:
                safe_rich_debug(f"    处理节点数: {entry['count']}", "scenario")
            if 'mean_value' in entry:
                safe_rich_debug(f"    平均值: {entry['mean_value']:.2f}", "scenario")

    def _setup_validation_rules(self):
        """设置验证规则"""
        return {
            'min_data_quality_score': self.config.get('min_data_quality_score', 0.5),
            'required_features': self.config.get('required_features', ['active_load']),
            'allow_missing_generation': self.config.get('allow_missing_generation', True)
        }


class RewardFunction:
    """
    自适应质量导向奖励函数系统。

    该类是整个强化学习训练的核心，它不仅计算每一步的即时奖励，还集成了
    复杂的自适应机制，如平台期检测、场景感知和数据完整性管理。
    其设计哲学是，奖励信号不仅要引导智能体走向最优解，还必须能够适应
    各种复杂的、非理想的训练场景。
    """

    def __init__(self,
                 hetero_data: HeteroData,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None,
                 is_normalized: bool = True):
        """
        初始化自适应质量导向奖励函数。

        Args:
            hetero_data (HeteroData): 
                PyG格式的异构图数据，包含了电网的完整拓扑和节点/边特征。
            config (Optional[Dict[str, Any]]): 
                一个配置字典，用于控制奖励权重、平台期检测参数、场景感知策略等。
            device (Optional[torch.device]): 
                计算所用的设备 (e.g., 'cpu', 'cuda')。
            is_normalized (bool): 
                是否使用归一化数据。
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

        # 场景感知奖励系统配置
        scenario_aware_config = self.config.get('scenario_aware_reward', {})
        
        # 创建场景感知组件
        self.history_tracker = ScenarioAwareHistoryTracker(scenario_aware_config)
        self.plateau_detector = ScenarioAwarePlateauDetector(
            self.history_tracker, 
            scenario_aware_config.get('detection_config', {})
        )
        
        # 创建相对奖励计算器
        relative_config = scenario_aware_config.get('relative_reward', {})
        self.relative_reward_calculator = HybridRewardFunction(
            self.history_tracker, 
            relative_config
        )

        # 前一步质量分数缓存
        self.previous_quality_score = None
        self.previous_metrics = None  # 保持向后兼容

        # 当前步数（用于效率奖励计算）
        self.current_step = 0
        self.max_steps = self.config.get('max_steps', 200)
        self.current_scenario_context = None  # 当前场景上下文

        # 鲁棒指标计算器
        self.metrics_calculator = RobustMetricsCalculator(
            config=self.config.get('error_handling', {
                'fallback_strategy': 'conservative'
            })
        )

        # 预计算常用数据
        self._setup_cached_data(is_normalized=is_normalized)

        # 数值稳定性参数
        self.epsilon = 1e-9
        
    def _load_weights(self) -> Dict[str, float]:
        """从配置中加载并合并奖励权重。"""
        default_weights = {
            # 质量分数权重（统一命名规范）
            'load_b': 0.4,
            'decoupling': 0.3,
            'power_b': 0.3,
            # 终局奖励权重（统一命名规范）
            'final_load_b': 0.4,
            'final_decoupling': 0.4,
            'final_power_b': 0.2,
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
        """创建平台期检测器实例。"""
        plateau_config = self.adaptive_quality_config['plateau_detection']
        return QualityPlateauDetector(
            window_size=plateau_config['window_size'],
            min_improvement_rate=plateau_config['min_improvement_rate'],
            stability_threshold=plateau_config['stability_threshold'],
            min_percentile=plateau_config['min_percentile'],
            confidence_threshold=plateau_config['confidence_threshold']
        )
        
    def _setup_cached_data(self, is_normalized: bool = True):
        """预计算频繁使用的数据 - 改进的数据处理"""
        # 安全提取节点负载和发电数据
        self._safe_extract_power_data()

        # 边信息（用于计算导纳）
        self._extract_edge_info(is_normalized=is_normalized)

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
            # 智能估算发电数据
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

    def _extract_edge_info(self, is_normalized: bool = True):
        """智能提取边信息用于导纳计算"""
        if ('bus', 'connects', 'bus') in self.hetero_data.edge_index_dict:
            self.edge_index = self.hetero_data.edge_index_dict[('bus', 'connects', 'bus')]

            # 智能提取导纳数据
            self.edge_admittance, admittance_source = self._extract_admittance_data(is_normalized=is_normalized)
            self.total_admittance = self.edge_admittance.sum() # <--- 预计算并缓存

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

    def _extract_admittance_data(self, is_normalized: bool = True):
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
            admittance_extractor = IntelligentAdmittanceExtractor(self.device, is_normalized=is_normalized)
            admittance = admittance_extractor._identify_admittance_from_attributes(edge_attr)
            if admittance is not None:
                return admittance, 'extracted'

        # 方法2: 从阻抗计算导纳
        if edge_type in self.hetero_data.edge_attr_dict:
            edge_attr = self.hetero_data.edge_attr_dict[edge_type]
            admittance_extractor = IntelligentAdmittanceExtractor(self.device, is_normalized=is_normalized)
            admittance = admittance_extractor._calculate_admittance_from_impedance(edge_attr)
            if admittance is not None:
                return admittance, 'calculated'

        # 方法3: 基于物理模型估算
        admittance = self._estimate_admittance_from_topology()
        if admittance is not None:
            return admittance, 'estimated'

        # 方法4: 保守默认值（带警告）
        return self._get_conservative_admittance_defaults(), 'default'

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

    def _compute_quality_score(self, partition: torch.Tensor) -> float:
        """
        根据当前分区方案，计算一个归一化的统一质量分数。

        该分数综合了负载均衡(CV)、电气解耦(Coupling)和功率平衡(Power Imbalance)
        三个核心维度，是评估分区方案质量的核心标量。

        Args:
            partition (torch.Tensor): 
                包含每个节点所属分区的张量。Shape: `[num_nodes]`

        Returns:
            float: 质量分数，范围在 [0, 1] 之间，值越大表示质量越好。
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
                self.weights['load_b'] * normalized_cv +
                self.weights['decoupling'] * normalized_coupling +
                self.weights['power_b'] * normalized_power
            )

            # 归一化到权重总和
            total_weight = (
                self.weights['load_b'] +
                self.weights['decoupling'] +
                self.weights['power_b']
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
        以向量化的方式，中心化地计算所有核心物理指标。
        
        该函数是性能关键路径，它通过高效的PyTorch张量操作，避免了任何
        在Python层面的循环，一次性计算出所有下游奖励函数所需的基础物理量。
        
        Args:
            partition (torch.Tensor): 
                包含每个节点所属分区的张量。Shape: `[num_nodes]`
            
        Returns:
            Dict[str, float]: 包含所有核心指标的字典 (e.g., 'cv', 'coupling_ratio')。
        """
        metrics = {}
        
        # 获取分区数量，确保分区索引是连续的，从0开始
        unique_partitions = torch.unique(partition)
        # 排除无效分区（如0）
        unique_partitions = unique_partitions[unique_partitions > 0]
        if unique_partitions.numel() == 0:
            return {'cv': 1.0, 'coupling_ratio': 1.0, 'power_imbalance_normalized': 1.0, 'num_partitions': 0}
        
        num_partitions = unique_partitions.max().item()
        
        # 创建一个从1开始的分区到从0开始的连续索引的映射
        # 这对于scatter操作至关重要，因为它需要连续的索引
        partition_map = torch.zeros(num_partitions + 1, dtype=torch.long, device=self.device)
        partition_map[unique_partitions] = torch.arange(unique_partitions.numel(), device=self.device)
        
        # 将原始分区标签映射到连续索引
        mapped_partition = partition_map[partition.long()]
        
        # 只处理有效分区 (>0) 的节点
        valid_mask = partition > 0
        
        # 1. 高效计算负载平衡指标 (CV)
        # 使用scatter_add计算每个分区的负载总和
        partition_loads = torch.zeros(unique_partitions.numel(), device=self.device).scatter_add_(
            0, mapped_partition[valid_mask], self.node_loads[valid_mask].abs()
        )
        
        # 计算变异系数
        mean_load = partition_loads.mean()
        std_load = partition_loads.std()
        
        if torch.isnan(mean_load) or torch.isinf(mean_load) or mean_load <= self.epsilon:
            cv = 1.0
        else:
            cv = (std_load / (mean_load + self.epsilon)).item()
        metrics['cv'] = np.clip(cv, 0.0, 10.0)

        # 2. 计算电气解耦指标 (已向量化)
        if self.edge_index.shape[1] > 0:
            coupling_result = self.metrics_calculator.calculate_coupling_metrics(
                self.edge_admittance, self.edge_index, partition, self.total_admittance
            )
            metrics.update(coupling_result)
        else:
            metrics['coupling_ratio'] = 0.0
            metrics['edge_decoupling_ratio'] = 1.0
            
        # 3. 高效计算功率平衡指标
        # 使用scatter_add同时计算每个分区的发电和负载
        partition_generation = torch.zeros(unique_partitions.numel(), device=self.device).scatter_add_(
            0, mapped_partition[valid_mask], self.node_generation[valid_mask]
        )
        # partition_loads 已在上文计算过，无需重复
        
        # 计算总不平衡
        total_imbalance = torch.abs(partition_generation - partition_loads).sum().item()
        
        # 归一化
        total_load = self.node_loads.abs().sum().item()
        if total_load > self.epsilon:
            power_imbalance_normalized = total_imbalance / total_load
        else:
            power_imbalance_normalized = 1.0
        metrics['power_imbalance_normalized'] = np.clip(power_imbalance_normalized, 0.0, 10.0)
        
        # 4. 其他辅助指标
        metrics['num_partitions'] = unique_partitions.numel()
        
        # 清理 NaN/Inf 值，确保返回结果的健壮性
        for key, value in metrics.items():
            if isinstance(value, (float, int)) and (np.isnan(value) or np.isinf(value)):
                metrics[key] = 1.0 if key in ['cv', 'coupling_ratio', 'power_imbalance_normalized'] else 0.0
        
        return metrics

    def compute_incremental_reward(self, 
                                  current_partition: torch.Tensor, 
                                  scenario_context: Optional[ScenarioContext] = None) -> Tuple[float, Dict[str, Any]]:
        """
        计算增量奖励 - 适配升级版HybridRewardFunction
        
        Args:
            current_partition: 当前分区配置
            scenario_context: 场景上下文（如果有）
            
        Returns:
            Tuple[float, Dict]: (奖励值, 平台期检测结果和奖励详情)
        """
        try:
            # 1. 计算当前质量分数
            current_quality_score = self._compute_quality_score(current_partition)
            
            # 2. 获取场景上下文（如果未提供）
            if scenario_context is None:
                # 创建默认场景上下文（正常运行状态）
                scenario_context = ScenarioContext()
            
            # 3. 使用升级版HybridRewardFunction计算奖励
            if self.relative_reward_calculator is not None:
                # 设置当前episode数（用于探索奖励衰减）
                self.relative_reward_calculator.set_episode(self.current_step // 100)  # 粗略的episode估计
                
                # 计算主要奖励
                main_reward, reward_details = self.relative_reward_calculator.compute_reward(
                    current_quality_score,
                    self.previous_quality_score,
                    scenario_context
                )
            else:
                # 如果相对奖励计算器未初始化，抛出错误
                raise RuntimeError("相对奖励计算器未正确初始化，无法计算奖励")
            
            # 4. 平台期检测
            plateau_result = None
            if self.plateau_detector:
                try:
                    plateau_result = self.plateau_detector.detect_plateau(
                        current_quality_score, scenario_context
                    )
                except Exception as e:
                    print(f"警告：平台期检测失败: {e}")
                    # 创建默认平台期结果
                    from .scenario_aware_plateau_detector import PlateauResult
                    plateau_result = PlateauResult(
                        plateau_detected=False,
                        confidence=0.0,
                        improvement_rate=1.0,
                        stability_score=0.0,
                        historical_percentile=0.0,
                        details={'error': str(e)}
                    )
            
            # 5. 更新历史
            self.previous_quality_score = current_quality_score
            self.previous_metrics = self._compute_core_metrics(current_partition)  # 保持向后兼容
            
            # 6. 组合完整的信息返回
            complete_info = {
                'quality_score': current_quality_score,
                'reward_details': reward_details,
                'scenario_context': scenario_context.to_dict(),
                'plateau_result': plateau_result._asdict() if plateau_result else None,
                'step': self.current_step
            }
            
            # 7. 添加奖励组件信息（用于TensorBoard记录）
            complete_info.update({
                'main_reward': reward_details['main_reward'],
                'base_reward': reward_details['base_reward'],
                'relative_reward': reward_details['relative_reward'],
                'improvement_bonus': reward_details['improvement_bonus'],
                'exploration_reward': reward_details['exploration_reward'],
                'total_reward': reward_details['total_reward'],
                'baseline_quality': reward_details['baseline_info']['baseline_quality'],
                'baseline_source': reward_details['baseline_info']['baseline_source'],
                'query_level': reward_details['baseline_info']['query_level']
            })
            
            return main_reward, complete_info
            
        except Exception as e:
            print(f"警告：增量奖励计算出现异常: {e}")
            # 返回保守的默认奖励
            return 0.0, {
                'error': str(e),
                'quality_score': 0.0,
                'reward_details': {'total_reward': 0.0},
                'scenario_context': scenario_context.to_dict() if scenario_context else {},
                'step': self.current_step
            }

    def reset_episode(self, scenario_context: Optional[ScenarioContext] = None):
        """重置episode状态"""
        self.current_step = 0
        self.previous_quality_score = None
        self.previous_metrics = None
        self.current_scenario_context = scenario_context

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
        计算回合结束时的终局奖励 (Final Reward)。

        该奖励在回合结束时被调用，用于对最终的分区质量给出一个综合性的评估。

        Args:
            final_partition (torch.Tensor): 
                最终的分区方案。Shape: `[num_nodes]`
            termination_type (str): 
                回合的终止类型 (e.g., 'natural', 'timeout', 'stuck')。

        Returns:
            Tuple[float, Dict[str, float]]:
                - total_reward: 计算出的总终局奖励。
                - components: 包含各个奖励分量详情的字典。
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
            self.weights['final_load_b'] * balance_reward +
            self.weights['final_decoupling'] * decoupling_reward +
            self.weights['final_power_b'] * power_reward
        )

        # 3. 阈值奖励系统已被自适应质量系统完全替代

        # 4. 应用终止条件折扣
        termination_discount = self._apply_termination_discount(termination_type)

        # 5. 计算最终奖励
        final_reward = quality_reward * termination_discount

        # 组件详情
        components = {
            'balance_reward': balance_reward,
            'decoupling_reward': decoupling_reward,
            'power_reward': power_reward,
            'quality_reward': quality_reward,
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
        """
        获取当前分区的指标（用于调试和分析）

        注意：为了训练效率，这个方法现在只返回完整的评估指标。
        训练过程中应该直接使用 get_current_quality_score() 方法。

        Args:
            partition: 当前分区方案

        Returns:
            完整的指标字典
        """
        return self._compute_core_metrics(partition)

    def update_weights(self, new_weights: Dict[str, float]):
        """动态更新奖励权重（用于智能自适应课程学习）"""
        try:
            # 更新即时奖励权重
            if 'load_b' in new_weights:
                self.weights['load_b'] = new_weights['load_b']
            if 'decoupling' in new_weights:
                self.weights['decoupling'] = new_weights['decoupling']
            if 'power_b' in new_weights:
                self.weights['power_b'] = new_weights['power_b']

            # 更新终局奖励权重（保持一致性）
            if 'load_b' in new_weights:
                self.weights['final_load_b'] = new_weights['load_b']
            if 'decoupling' in new_weights:
                self.weights['final_decoupling'] = new_weights['decoupling']
            if 'power_b' in new_weights:
                self.weights['final_power_b'] = new_weights['power_b']

            print(f"🎯 奖励权重已更新: load_b={self.weights['load_b']:.2f}, "
                  f"decoupling={self.weights['decoupling']:.2f}, "
                  f"power_b={self.weights['power_b']:.2f}")

        except Exception as e:
            print(f"⚠️ 更新奖励权重失败: {e}")

    def get_current_weights(self) -> Dict[str, float]:
        """获取当前奖励权重"""
        return {
            'load_b': self.weights.get('load_b', 1.0),
            'decoupling': self.weights.get('decoupling', 1.0),
            'power_b': self.weights.get('power_b', 1.0)
        }





