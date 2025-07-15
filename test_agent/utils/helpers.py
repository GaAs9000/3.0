#!/usr/bin/env python3
"""
结果分析工具函数
提供通用的辅助功能
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


def load_model_checkpoint(model_path: str, device: torch.device) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
        
    Returns:
        检查点字典
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # 验证必要字段
    required_fields = ['actor_state_dict', 'critic_state_dict', 'model_info']
    for field in required_fields:
        if field not in checkpoint:
            raise ValueError(f"模型文件缺少必要字段: {field}")
    
    return checkpoint


def calculate_improvement_metrics(baseline_metrics: Dict[str, float],
                                agent_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    计算改进指标
    
    Args:
        baseline_metrics: 基准方法指标
        agent_metrics: Agent指标
        
    Returns:
        改进百分比字典
    """
    improvements = {}
    
    # CV改进（越小越好）
    if baseline_metrics.get('cv', 0) > 0:
        improvements['cv'] = (baseline_metrics['cv'] - agent_metrics['cv']) / baseline_metrics['cv'] * 100
    else:
        improvements['cv'] = 0
    
    # 耦合度改进（越小越好）
    if baseline_metrics.get('coupling', 0) > 0:
        improvements['coupling'] = (baseline_metrics['coupling'] - agent_metrics['coupling']) / baseline_metrics['coupling'] * 100
    else:
        improvements['coupling'] = 0
    
    # 成功率改进（越大越好）
    baseline_success = 1.0 if baseline_metrics.get('success', False) else 0.0
    agent_success = 1.0 if agent_metrics.get('success', False) else 0.0
    improvements['success'] = (agent_success - baseline_success) * 100
    
    return improvements


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    格式化指标表格
    
    Args:
        metrics: 指标字典
        
    Returns:
        格式化的表格字符串
    """
    table = "| 指标 | 数值 |\n"
    table += "|------|------|\n"
    
    for key, value in metrics.items():
        if isinstance(value, bool):
            table += f"| {key} | {'✓' if value else '✗'} |\n"
        elif isinstance(value, float):
            table += f"| {key} | {value:.4f} |\n"
        else:
            table += f"| {key} | {value} |\n"
    
    return table


def save_results_summary(results: Dict[str, Any], output_path: str):
    """
    保存结果摘要
    
    Args:
        results: 结果字典
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON格式
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 保存文本摘要
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("电力网络分区结果摘要\n")
        f.write("=" * 50 + "\n\n")
        
        # 写入各场景结果
        if 'scenarios' in results:
            for scenario, data in results['scenarios'].items():
                f.write(f"\n场景: {scenario}\n")
                f.write("-" * 30 + "\n")
                
                if 'agent' in data:
                    f.write("\nAgent指标:\n")
                    f.write(format_metrics_table(data['agent']))
                
                if 'baseline' in data:
                    f.write("\nBaseline指标:\n")
                    f.write(format_metrics_table(data['baseline']))
                
                if 'improvement' in data:
                    f.write("\n改进百分比:\n")
                    f.write(format_metrics_table(data['improvement']))
    
    print(f"📄 结果已保存:")
    print(f"   JSON: {json_path}")
    print(f"   文本: {txt_path}")


def get_network_statistics(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取网络统计信息
    
    Args:
        case_data: 电网案例数据
        
    Returns:
        统计信息字典
    """
    stats = {
        'num_buses': len(case_data['bus']),
        'num_branches': len(case_data['branch']),
        'num_generators': len(case_data['gen']) if 'gen' in case_data else 0,
        'total_load': float(np.sum(case_data['bus'][:, 2])),  # PD列
        'total_generation': float(np.sum(case_data['gen'][:, 1])) if 'gen' in case_data else 0
    }
    
    # 计算网络密度
    max_edges = stats['num_buses'] * (stats['num_buses'] - 1) / 2
    stats['density'] = stats['num_branches'] / max_edges if max_edges > 0 else 0
    
    return stats


def setup_output_directory(base_dir: str = 'evaluation') -> Path:
    """
    设置输出目录
    
    Args:
        base_dir: 基础目录名
        
    Returns:
        时间戳目录路径
    """
    from datetime import datetime
    
    base_path = Path(base_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = base_path / timestamp
    
    # 创建所有必要的子目录
    subdirs = ['comparisons', 'visualizations', 'reports', 'cache']
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return output_dir


def validate_partition(partition: np.ndarray, num_partitions: int) -> Tuple[bool, List[str]]:
    """
    验证分区结果的有效性
    
    Args:
        partition: 分区数组
        num_partitions: 期望的分区数
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查分区编号范围
    unique_partitions = np.unique(partition[partition > 0])
    if len(unique_partitions) != num_partitions:
        errors.append(f"分区数不匹配: 期望{num_partitions}, 实际{len(unique_partitions)}")
    
    # 检查是否所有分区都被使用
    for p in range(1, num_partitions + 1):
        if p not in partition:
            errors.append(f"分区{p}未被使用")
    
    # 检查是否有未分配的节点
    unassigned = np.sum(partition == 0)
    if unassigned > 0:
        errors.append(f"有{unassigned}个节点未分配")
    
    is_valid = len(errors) == 0
    return is_valid, errors