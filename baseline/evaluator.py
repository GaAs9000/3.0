import torch
import numpy as np
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from env import PowerGridPartitionEnv


def evaluate_partition_method(env: 'PowerGridPartitionEnv', partition: np.ndarray) -> Dict[str, float]:
    """
    评估分区方案（已更新以兼容新的环境API）

    参数:
        env: 环境实例
        partition: 分区结果

    返回:
        评估指标字典
    """
    # 转换分区为torch张量
    partition_tensor = torch.tensor(partition, dtype=torch.long, device=env.device)

    # 使用新的评估器API
    metrics = env.evaluator.evaluate_partition(partition_tensor)

    return {
        'load_cv': metrics.get('load_cv', 0.0),
        'load_gini': metrics.get('load_gini', 0.0),
        'total_coupling': metrics.get('total_coupling', 0.0),
        'inter_region_lines': metrics.get('coupling_edges', 0),
        'connectivity': metrics.get('connectivity', 1.0),
        'power_balance': metrics.get('power_imbalance_mean', 0.0),
        'modularity': metrics.get('modularity', 0.0)
    }