import numpy as np
from sklearn.cluster import SpectralClustering
from typing import TYPE_CHECKING
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from env import PowerGridPartitionEnv


class SpectralPartitioner(BasePartitioner):
    """
    谱聚类分区方法
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(seed)
    
    def partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """执行谱聚类分区"""
        # 每次执行前设置随机种子
        self._set_seed()
        
        try:
            # 首先尝试使用 METIS 进行分区
            return self._metis_partition(env)
        except Exception as e:
            print(f"⚠️ METIS 分区失败: {str(e)}，使用谱聚类...")
            return self._spectral_clustering_partition(env)
    
    def _metis_partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """使用 PyMetis 进行图分区"""
        try:
            import pymetis
        except ImportError:
            raise ImportError("PyMetis 未安装或不可用")
        
        # 构建邻接列表（PyMetis 需要的格式）
        adjacency_list = self._build_adjacency_list(env)
        
        # 使用 PyMetis 进行分区
        n_cuts, membership = pymetis.part_graph(env.K, adjacency=adjacency_list)
        
        # PyMetis 返回 0-based 标签，转换为 1-based
        labels = np.array(membership) + 1
        
        print(f"✅ PyMetis 分区成功：切边数 = {n_cuts}")
        return labels
    
    def _build_adjacency_list(self, env: 'PowerGridPartitionEnv') -> list:
        """构建 PyMetis 需要的邻接列表格式"""
        # 获取边信息
        edge_array = env.edge_index.cpu().numpy()
        
        # 初始化邻接列表
        adjacency_list = [[] for _ in range(env.N)]
        
        # 构建邻接列表
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            # PyMetis 需要无向图，每条边都要添加两次
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)
        
        # 转换为 numpy 数组格式（PyMetis 要求）
        adjacency_list = [np.array(adj, dtype=np.int32) for adj in adjacency_list]
        
        return adjacency_list
    
    def _spectral_clustering_partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """使用 scikit-learn 的谱聚类作为备选方案"""
        # 构建邻接矩阵
        adj_matrix = np.zeros((env.N, env.N))
        edge_array = env.edge_index.cpu().numpy()
        
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            # 使用导纳作为权重
            weight = env.admittance[i].item()
            
            # 检查权重是否为NaN或无穷大
            if np.isnan(weight) or np.isinf(weight):
                weight = 1e-10  # 使用极小的正值代替
            
            adj_matrix[u, v] = weight
            adj_matrix[v, u] = weight
        
        # 全面的NaN和inf处理
        adj_matrix = np.nan_to_num(adj_matrix, nan=1e-10, posinf=1.0, neginf=0.0)
        
        # 确保矩阵是对称的
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        
        # 确保所有值为非负（谱聚类要求非负权重）
        adj_matrix = np.abs(adj_matrix)
        
        # 确保对角线上至少有一个小的正值（避免奇异矩阵）
        np.fill_diagonal(adj_matrix, np.maximum(adj_matrix.diagonal(), 1e-10))
        
        # 最终验证矩阵的有效性
        if np.any(np.isnan(adj_matrix)) or np.any(np.isinf(adj_matrix)):
            print("⚠️ 警告：邻接矩阵仍有异常值，使用单位矩阵作为备选方案...")
            adj_matrix = np.eye(env.N) * 1e-10 + np.ones((env.N, env.N)) * 1e-12
        
        # 谱聚类
        clustering = SpectralClustering(
            n_clusters=env.K,
            affinity='precomputed',
            n_init=10,
            random_state=self.seed,  # 使用实例的种子
            assign_labels='discretize'  # 使用更稳定的标签分配方法
        )
        
        labels = clustering.fit_predict(adj_matrix)
        
        return labels + 1  # 转换为1-based 