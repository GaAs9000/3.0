import numpy as np
from typing import TYPE_CHECKING
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from env import PowerGridPartitionEnv


class RandomPartitioner(BasePartitioner):
    """
    随机分区方法
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
    
    def partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """执行随机分区"""
        # 确保使用固定种子
        self._set_seed()
        labels = np.random.randint(1, env.K + 1, size=env.N)
        return labels 