#!/usr/bin/env python3
"""
评估配置文件

定义所有评估相关的参数和配置：
- Baseline对比配置
- 泛化测试配置
- 指标权重配置
- 可视化配置
"""

from typing import Dict, List, Any

# 评估配置
EVALUATION_CONFIG = {
    # Baseline对比测试配置
    'baseline_comparison': {
        'enabled': True,
        'test_networks': ['ieee14', 'ieee30', 'ieee57'],
        'scenarios': ['normal', 'high_load', 'unbalanced', 'fault'],
        'runs_per_test': 10,
        'baseline_methods': {
            'random': {
                'enabled': True,
                'seed': 42
            },
            'degree_based': {
                'enabled': True,
                'balance_strategy': 'total_degree'
            },
            'spectral': {
                'enabled': True,
                'random_state': 42
            }
        }
    },
    
    # 泛化测试配置
    'generalization_test': {
        'enabled': True,
        'train_network': 'ieee14',
        'test_networks': ['ieee30', 'ieee57', 'ieee118'],
        'runs_per_network': 20,
        'cross_network_evaluation': True
    },
    
    # 指标权重配置
    'metrics_weights': {
        'inter_region_balance': 0.3,  # 分区间平衡权重
        'intra_region_balance': 0.3,  # 区域内平衡权重（新增）
        'decoupling': 0.4  # 电气解耦权重
    },
    
    # 性能阈值配置
    'thresholds': {
        'excellent_generalization': 10,  # <10%性能下降为优秀
        'good_generalization': 20,       # 10-20%性能下降为良好
        'excellent_improvement': 15,     # >15%提升为优秀
        'good_improvement': 5            # 5-15%提升为良好
    },
    
    # 可视化配置
    'visualization': {
        'enabled': True,
        'save_figures': True,
        'figure_format': 'png',
        'dpi': 300,
        'output_dir': 'evaluation_results',
        'color_scheme': 'Set3',
        'figure_size': (16, 12)
    },
    
    # 报告生成配置
    'reporting': {
        'enabled': True,
        'output_format': ['txt', 'html'],
        'include_detailed_metrics': True,
        'include_visualizations': True,
        'auto_open_report': False
    },
    
    # 场景生成配置
    'scenario_generation': {
        'high_load_multiplier': 1.5,      # 高负载场景负载倍数
        'unbalanced_ratio': 0.3,          # 不平衡场景高负载节点比例
        'unbalanced_high_multiplier': 2.0, # 不平衡场景高负载倍数
        'unbalanced_low_multiplier': 0.8,  # 不平衡场景低负载倍数
        'fault_edges_to_remove': 1,       # 故障场景移除边数
        'random_seed': 42                 # 随机种子
    },
    
    # 性能优化配置
    'performance': {
        'max_parallel_runs': 4,           # 最大并行运行数
        'enable_caching': True,           # 启用结果缓存
        'cache_dir': 'evaluation_cache',  # 缓存目录
        'timeout_per_run': 300            # 每次运行超时时间（秒）
    }
}

# 网络特定配置
NETWORK_CONFIGS = {
    'ieee14': {
        'num_partitions': 3,
        'expected_nodes': 14,
        'complexity_level': 'low'
    },
    'ieee30': {
        'num_partitions': 3,  # 修改为3分区以匹配预训练模型
        'expected_nodes': 30,
        'complexity_level': 'medium'
    },
    'ieee57': {
        'num_partitions': 3,  # 修改为3分区以匹配预训练模型
        'expected_nodes': 57,
        'complexity_level': 'medium'
    },
    'ieee118': {
        'num_partitions': 6,
        'expected_nodes': 118,
        'complexity_level': 'high'
    }
}

# 指标定义和描述
METRICS_DEFINITIONS = {
    'inter_region_cv': {
        'name': '分区间平衡度',
        'description': '各分区总负载的变异系数，越小越好',
        'unit': 'CV值',
        'optimal_range': [0.0, 0.3],
        'display_format': '.3f'
    },
    'intra_region_cv': {
        'name': '区域内平衡度',
        'description': '各分区内部节点负载的平均变异系数，越小越好',
        'unit': 'CV值',
        'optimal_range': [0.0, 0.5],
        'display_format': '.3f'
    },
    'decoupling': {
        'name': '电气解耦度',
        'description': '分区内部边占总边数的比例，越大越好',
        'unit': '比例',
        'optimal_range': [0.7, 1.0],
        'display_format': '.3f'
    },
    'comprehensive_score': {
        'name': '综合质量分数',
        'description': '基于三个核心指标的加权综合分数，越大越好',
        'unit': '分数',
        'optimal_range': [0.6, 1.0],
        'display_format': '.3f'
    }
}

# 评估模式配置
EVALUATION_MODES = {
    'quick': {
        'description': '快速评估模式',
        'baseline_comparison': {
            'enabled': True,
            'test_networks': ['ieee14', 'ieee30'],
            'scenarios': ['normal', 'high_load'],
            'runs_per_test': 5
        },
        'generalization_test': {
            'enabled': True,
            'test_networks': ['ieee30'],
            'runs_per_network': 10
        }
    },
    'standard': {
        'description': '标准评估模式',
        'baseline_comparison': {
            'enabled': True,
            'test_networks': ['ieee14', 'ieee30', 'ieee57'],
            'scenarios': ['normal', 'high_load', 'unbalanced'],
            'runs_per_test': 10
        },
        'generalization_test': {
            'enabled': True,
            'test_networks': ['ieee30', 'ieee57'],
            'runs_per_network': 20
        }
    },
    'comprehensive': {
        'description': '全面评估模式',
        'baseline_comparison': {
            'enabled': True,
            'test_networks': ['ieee14', 'ieee30', 'ieee57'],
            'scenarios': ['normal', 'high_load', 'unbalanced', 'fault'],
            'runs_per_test': 15
        },
        'generalization_test': {
            'enabled': True,
            'test_networks': ['ieee30', 'ieee57', 'ieee118'],
            'runs_per_network': 30
        }
    }
}

def get_evaluation_config(mode: str = 'standard') -> Dict[str, Any]:
    """
    获取指定模式的评估配置
    
    Args:
        mode: 评估模式 ('quick', 'standard', 'comprehensive')
        
    Returns:
        评估配置字典
    """
    base_config = EVALUATION_CONFIG.copy()
    
    if mode in EVALUATION_MODES:
        mode_config = EVALUATION_MODES[mode]
        
        # 更新baseline对比配置
        if 'baseline_comparison' in mode_config:
            base_config['baseline_comparison'].update(mode_config['baseline_comparison'])
        
        # 更新泛化测试配置
        if 'generalization_test' in mode_config:
            base_config['generalization_test'].update(mode_config['generalization_test'])
    
    return base_config

def get_network_config(network_name: str) -> Dict[str, Any]:
    """
    获取指定网络的配置
    
    Args:
        network_name: 网络名称
        
    Returns:
        网络配置字典
    """
    return NETWORK_CONFIGS.get(network_name, {
        'num_partitions': 4,
        'expected_nodes': 30,
        'complexity_level': 'medium'
    })

def get_metric_definition(metric_name: str) -> Dict[str, Any]:
    """
    获取指标定义
    
    Args:
        metric_name: 指标名称
        
    Returns:
        指标定义字典
    """
    return METRICS_DEFINITIONS.get(metric_name, {
        'name': metric_name,
        'description': '未定义指标',
        'unit': '未知',
        'optimal_range': [0.0, 1.0],
        'display_format': '.3f'
    })
