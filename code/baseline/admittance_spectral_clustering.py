import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from typing import TYPE_CHECKING
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from ..src.rl.environment import PowerGridPartitioningEnv


class AdmittanceSpectralPartitioner(BasePartitioner):
    """
    基于导纳的谱聚类分区方法
    
    利用线路导纳构建加权拉普拉斯矩阵，通过谱分析进行聚类。
    特别适用于电力网络，能够利用电气特性进行分区。
    """
    
    def __init__(self, seed: int = 42, gamma: float = 1.0, n_neighbors: int = 10, 
                 eigen_solver: str = 'arpack', assign_labels: str = 'cluster_qr'):
        """
        初始化基于导纳的谱聚类分区器
        
        Args:
            seed: 随机种子
            gamma: RBF核参数，用于构建相似度矩阵
            n_neighbors: 近邻数量（用于稀疏化）
            eigen_solver: 特征值求解器 ('arpack', 'lobpcg', 'amg')
            assign_labels: 标签分配方法 ('kmeans', 'discretize', 'cluster_qr')
        """
        super().__init__(seed)
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.eigen_solver = eigen_solver
        self.assign_labels = assign_labels
    
    def partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """执行基于导纳的谱聚类分区"""
        # 设置随机种子
        self._set_seed()
        
        try:
            # 构建基于导纳的权重矩阵
            W = self._build_admittance_weight_matrix(env)
            
            # 执行谱聚类
            labels = self._perform_spectral_clustering(W, env.num_partitions)
            
            return labels + 1  # 转换为1-based标签
            
        except Exception as e:
            print(f"❌ 基于导纳的谱聚类失败: {str(e)}")
            # 降级到简单分区
            return self._fallback_partition(env)
    
    def _build_admittance_weight_matrix(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """构建基于导纳的权重矩阵"""
        total_nodes = env.total_nodes
        edge_array = env.edge_info['edge_index'].cpu().numpy()
        
        # 初始化权重矩阵
        W = np.zeros((total_nodes, total_nodes))
        
        # 填充权重矩阵
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            
            # 获取导纳权重
            if hasattr(env.evaluator, 'edge_admittances') and i < len(env.evaluator.edge_admittances):
                admittance = float(env.evaluator.edge_admittances[i].item())
            else:
                # 使用估算的导纳
                admittance = self._estimate_admittance(u, v, env)
            
            # 处理异常值
            if np.isnan(admittance) or np.isinf(admittance) or admittance <= 0:
                admittance = 1e-6
            
            # 构建相似度权重（导纳越大，电气距离越近）
            weight = abs(admittance)
            
            W[u, v] = weight
            W[v, u] = weight
        
        # 增强对角线以提高数值稳定性
        np.fill_diagonal(W, W.diagonal() + 1e-6)
        
        # 应用RBF核进行平滑
        W = self._apply_rbf_kernel(W)
        
        # 可选：稀疏化处理
        if self.n_neighbors > 0:
            W = self._sparsify_matrix(W)
        
        return W
    
    def _estimate_admittance(self, u: int, v: int, env: 'PowerGridPartitioningEnv') -> float:
        """估算线路导纳"""
        total_nodes = env.total_nodes
        
        # 基于网络规模和节点距离的启发式估算
        if total_nodes <= 30:
            base_admittance = 5.0
        elif total_nodes <= 100:
            base_admittance = 3.0
        else:
            base_admittance = 2.0
        
        # 基于节点索引距离的调整
        index_diff = abs(u - v)
        if index_diff == 1:
            distance_factor = 1.5  # 相邻节点导纳更高
        elif index_diff <= 5:
            distance_factor = 1.2  # 近邻节点
        else:
            distance_factor = 1.0  # 远距离节点
        
        return base_admittance * distance_factor
    
    def _apply_rbf_kernel(self, W: np.ndarray) -> np.ndarray:
        """应用RBF核进行权重平滑"""
        if self.gamma <= 0:
            return W
        
        # 计算欧几里得距离矩阵
        n = W.shape[0]
        D = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # 使用导纳差异作为距离度量
                dist = abs(W[i, j] - W[j, i])
                D[i, j] = D[j, i] = dist
        
        # 应用RBF核
        W_rbf = np.exp(-self.gamma * D)
        
        # 与原始权重矩阵结合
        W_combined = W * W_rbf
        
        return W_combined
    
    def _sparsify_matrix(self, W: np.ndarray) -> np.ndarray:
        """稀疏化权重矩阵，保留每个节点的k个最近邻"""
        n = W.shape[0]
        W_sparse = np.zeros_like(W)
        
        for i in range(n):
            # 找到第i个节点的k个最近邻
            row = W[i, :]
            # 排除自身
            row[i] = 0
            
            # 找到最大的k个值的索引
            k = min(self.n_neighbors, n - 1)
            top_k_indices = np.argpartition(row, -k)[-k:]
            
            # 保留这些连接
            W_sparse[i, top_k_indices] = W[i, top_k_indices]
            W_sparse[top_k_indices, i] = W[top_k_indices, i]
        
        # 确保对称性
        W_sparse = (W_sparse + W_sparse.T) / 2
        
        return W_sparse
    
    def _perform_spectral_clustering(self, W: np.ndarray, n_clusters: int) -> np.ndarray:
        """执行谱聚类"""
        # 数据预处理
        W = self._preprocess_weight_matrix(W)
        
        # 创建谱聚类对象
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels=self.assign_labels,
            random_state=self.seed,
            eigen_solver=self.eigen_solver,
            n_init=10
        )
        
        # 执行聚类
        labels = spectral.fit_predict(W)
        
        return labels
    
    def _preprocess_weight_matrix(self, W: np.ndarray) -> np.ndarray:
        """预处理权重矩阵"""
        # 处理NaN和inf
        W = np.nan_to_num(W, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 确保非负
        W = np.abs(W)
        
        # 确保对称性
        W = (W + W.T) / 2
        
        # 确保对角线为正
        np.fill_diagonal(W, np.maximum(W.diagonal(), 1e-10))
        
        # 归一化（可选）
        W = normalize(W, norm='l1', axis=1)
        
        return W
    
    def _fallback_partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """降级分区方法"""
        print("⚠️ 使用降级分区方法...")
        
        # 简单的轮询分配
        labels = np.zeros(env.total_nodes, dtype=int)
        for i in range(env.total_nodes):
            labels[i] = (i % env.num_partitions) + 1
        
        return labels
    
    def get_spectral_gap(self, env: 'PowerGridPartitioningEnv') -> float:
        """计算谱间隙（用于评估聚类质量）"""
        try:
            W = self._build_admittance_weight_matrix(env)
            W = self._preprocess_weight_matrix(W)
            
            # 计算拉普拉斯矩阵
            D = np.diag(np.sum(W, axis=1))
            L = D - W
            
            # 计算特征值
            eigenvalues = np.linalg.eigvals(L)
            eigenvalues = np.sort(np.real(eigenvalues))
            
            # 计算谱间隙（第二小特征值）
            if len(eigenvalues) > 1:
                spectral_gap = eigenvalues[1] - eigenvalues[0]
            else:
                spectral_gap = 0.0
            
            return spectral_gap
            
        except Exception as e:
            print(f"⚠️ 计算谱间隙失败: {e}")
            return 0.0
    
    def get_conductance(self, env: 'PowerGridPartitioningEnv', labels: np.ndarray) -> float:
        """计算分区的导度（用于评估分区质量）"""
        try:
            W = self._build_admittance_weight_matrix(env)
            
            # 计算每个分区的导度
            conductances = []
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                # 找到属于该分区的节点
                nodes_in_partition = np.where(labels == label)[0]
                
                # 计算分区内部权重和
                internal_weight = 0.0
                for i in nodes_in_partition:
                    for j in nodes_in_partition:
                        if i != j:
                            internal_weight += W[i, j]
                
                # 计算分区边界权重和
                boundary_weight = 0.0
                for i in nodes_in_partition:
                    for j in range(len(labels)):
                        if labels[j] != label:
                            boundary_weight += W[i, j]
                
                # 计算导度
                total_weight = internal_weight + boundary_weight
                if total_weight > 0:
                    conductance = boundary_weight / total_weight
                else:
                    conductance = 0.0
                
                conductances.append(conductance)
            
            # 返回平均导度
            return np.mean(conductances)
            
        except Exception as e:
            print(f"⚠️ 计算导度失败: {e}")
            return 1.0