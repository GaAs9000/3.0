import numpy as np
import gc
import os
from sklearn.cluster import MiniBatchKMeans
from typing import TYPE_CHECKING
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from env import PowerGridPartitionEnv


class KMeansPartitioner(BasePartitioner):
    """
    K-means分区方法（基于节点嵌入）
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
    
    def partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """执行K-means分区"""
        # 再次确保种子设置
        self._set_seed()
        
        # 强制设置单线程环境变量，避免多线程冲突
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        
        try:
            # 确保使用CPU版本的嵌入，避免CUDA内存问题（已更新以兼容新环境API）
            embeddings = env.node_embeddings.detach().cpu().numpy().copy()

            # 检查和清理数据
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                print("⚠️ 警告：嵌入中存在异常值，进行清理...")
                embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

            # 使用MiniBatchKMeans，它更稳定且内存友好
            kmeans = MiniBatchKMeans(
                n_clusters=env.num_partitions,
                n_init=3,
                max_iter=50,
                random_state=self.seed,  # 使用实例的种子
                batch_size=min(100, len(embeddings))
            )
            
            labels = kmeans.fit_predict(embeddings)
            
            # 清理内存
            del embeddings, kmeans
            gc.collect()
            
            return labels + 1
            
        except Exception as e:
            print(f"⚠️ K-means聚类失败: {str(e)}，使用随机分区...")
            # 降级到随机分区
            from .random_partition import RandomPartitioner
            random_partitioner = RandomPartitioner(self.seed)
            return random_partitioner.partition(env) 