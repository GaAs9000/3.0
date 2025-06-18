import torch
import numpy as np
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from env import PowerGridPartitionEnv


def evaluate_partition_method(env: 'PowerGridPartitionEnv', partition: np.ndarray) -> Dict[str, float]:
    """
    评估分区方案
    
    参数:
        env: 环境实例
        partition: 分区结果
        
    返回:
        评估指标字典
    """
    # 应用分区
    env.z = torch.tensor(partition, dtype=torch.long, device=env.device)
    env._update_state()
    
    # 获取指标
    metrics = env._compute_metrics()
    
    return {
        'load_cv': metrics.load_cv,
        'load_gini': metrics.load_gini,
        'total_coupling': metrics.total_coupling,
        'inter_region_lines': metrics.inter_region_lines,
        'connectivity': metrics.connectivity,
        'power_balance': metrics.power_balance,
        'modularity': metrics.modularity
    } 