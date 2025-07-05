import torch
import numpy as np
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..src.rl.environment import PowerGridPartitioningEnv


def evaluate_partition_method(env: 'PowerGridPartitioningEnv', partition: np.ndarray) -> Dict[str, float]:
    """
    评估分区方案（已更新以使用统一的新奖励系统）

    参数:
        env: 环境实例
        partition: 分区结果

    返回:
        评估指标字典
    """
    # 转换分区为torch张量
    partition_tensor = torch.tensor(partition, dtype=torch.long, device=env.device)

    # 【修复】使用统一的新奖励系统API，确保与RL方法评估标准一致
    if hasattr(env, 'reward_function') and env.reward_function is not None:
        # 使用新奖励系统
        metrics = env.reward_function.get_current_metrics(partition_tensor)

        # 适配到基线比较期望的格式
        return {
            'load_cv': metrics.get('cv', 0.0),
            'load_gini': metrics.get('load_gini', 0.0),  # 如果新系统没有，保持默认值
            'total_coupling': metrics.get('coupling_ratio', 0.0),
            'inter_region_lines': int(metrics.get('coupling_ratio', 0.0) * metrics.get('total_edges', 100)),
            'connectivity': metrics.get('connectivity', 1.0),
            'power_b': metrics.get('power_imbalance_normalized', 0.0),
            'modularity': metrics.get('modularity', 0.0)  # 如果新系统没有，保持默认值
        }
    else:
        # 向后兼容：如果没有新奖励系统，使用旧评估器
        metrics = env.evaluator.evaluate_partition(partition_tensor)

        return {
            'load_cv': metrics.get('load_cv', 0.0),
            'load_gini': metrics.get('load_gini', 0.0),
            'total_coupling': metrics.get('total_coupling', 0.0),
            'inter_region_lines': metrics.get('coupling_edges', 0),
            'connectivity': metrics.get('connectivity', 1.0),
            'power_b': metrics.get('power_imbalance_mean', 0.0),
            'modularity': metrics.get('modularity', 0.0)
        }