"""
主函数文件
"""
import numpy as np
import torch
import random
from typing import TYPE_CHECKING

# 延迟导入避免循环依赖

if TYPE_CHECKING:
    from env import PowerGridPartitionEnv


def set_baseline_seed(seed: int = 42):
    """为基线方法设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BasePartitioner:
    """
    基础分区器类
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """执行分区 - 子类需要重写此方法"""
        raise NotImplementedError("子类必须实现partition方法")
    
    def _set_seed(self):
        """设置随机种子"""
        set_baseline_seed(self.seed)

# 为了保持完全向后兼容，创建原有的BaselinePartitioner类
class BaselinePartitioner:
    """
    基线分区方法
    """
    
    def __init__(self, method: str = 'spectral', seed: int = 42):
        self.method = method
        self.seed = seed
        self._partitioner = None  # 延迟初始化
        
    def _get_partitioner(self):
        """延迟初始化分区器以避免循环导入"""
        if self._partitioner is None:
            if self.method == 'spectral':
                from .spectral_clustering import SpectralPartitioner
                self._partitioner = SpectralPartitioner(self.seed)
            elif self.method == 'kmeans':
                from .kmeans_clustering import KMeansPartitioner
                self._partitioner = KMeansPartitioner(self.seed)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        return self._partitioner
    
    def partition(self, env):
        """执行分区"""
        partitioner = self._get_partitioner()
        return partitioner.partition(env)

# 导出基础类和函数
__all__ = [
    'BasePartitioner',
    'BaselinePartitioner',
    'set_baseline_seed'
]

