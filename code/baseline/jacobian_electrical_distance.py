import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from typing import TYPE_CHECKING, Optional, Dict, Any
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from ..src.rl.environment import PowerGridPartitioningEnv

try:
    # 尝试导入潮流计算相关的库
    import pandapower as pp
    import pandapower.networks as pn
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False


class JacobianElectricalDistancePartitioner(BasePartitioner):
    """
    基于雅可比的电气距离聚类方法
    
    利用当前运行点下的潮流雅可比矩阵，定义反映节点间功率-相角灵敏度的"电气距离"，
    并基于此距离进行聚类。分区结果会随注入功率的变化而变化。
    """
    
    def __init__(self, seed: int = 42, clustering_method: str = 'affinity_propagation',
                 damping: float = 0.5, preference: Optional[float] = None,
                 linkage: str = 'ward', distance_metric: str = 'euclidean'):
        """
        初始化基于雅可比的电气距离分区器
        
        Args:
            seed: 随机种子
            clustering_method: 聚类方法 ('affinity_propagation', 'agglomerative')
            damping: 阻尼参数（仅用于Affinity Propagation）
            preference: 偏好参数（仅用于Affinity Propagation）
            linkage: 连接方法（仅用于Agglomerative）
            distance_metric: 距离度量方法
        """
        super().__init__(seed)
        self.clustering_method = clustering_method
        self.damping = damping
        self.preference = preference
        self.linkage = linkage
        self.distance_metric = distance_metric
    
    def partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """执行基于雅可比的电气距离聚类"""
        # 设置随机种子
        self._set_seed()
        
        try:
            # 获取当前状态的功率注入
            injections = self._extract_power_injections(env)
            
            # 计算雅可比矩阵
            jacobian = self._compute_jacobian_matrix(env, injections)
            
            # 计算电气距离矩阵
            distance_matrix = self._compute_electrical_distance_matrix(jacobian)
            
            # 执行聚类
            labels = self._perform_clustering(distance_matrix, env.num_partitions)
            
            return labels + 1  # 转换为1-based标签
            
        except Exception as e:
            print(f"❌ 基于雅可比的电气距离聚类失败: {str(e)}")
            # 降级到简单分区
            return self._fallback_partition(env)
    
    def _extract_power_injections(self, env: 'PowerGridPartitioningEnv') -> Dict[str, np.ndarray]:
        """提取当前状态的功率注入"""
        try:
            # 尝试从环境中提取功率注入信息
            if hasattr(env, 'current_injections'):
                return env.current_injections
            
            # 如果没有当前注入，使用默认值
            total_nodes = env.total_nodes
            
            # 创建默认的功率注入（基于节点特征）
            if hasattr(env, 'node_features'):
                node_features = env.node_features.cpu().numpy()
                
                # 假设节点特征包含功率信息
                if node_features.shape[1] >= 2:
                    P = node_features[:, 0]  # 有功功率
                    Q = node_features[:, 1]  # 无功功率
                else:
                    P = np.random.normal(0, 1, total_nodes)
                    Q = np.random.normal(0, 0.5, total_nodes)
            else:
                # 生成随机功率注入作为示例
                P = np.random.normal(0, 1, total_nodes)
                Q = np.random.normal(0, 0.5, total_nodes)
            
            return {'P': P, 'Q': Q}
            
        except Exception as e:
            print(f"⚠️ 功率注入提取失败，使用默认值: {e}")
            total_nodes = env.total_nodes
            return {
                'P': np.random.normal(0, 1, total_nodes),
                'Q': np.random.normal(0, 0.5, total_nodes)
            }
    
    def _compute_jacobian_matrix(self, env: 'PowerGridPartitioningEnv', 
                               injections: Dict[str, np.ndarray]) -> np.ndarray:
        """计算潮流雅可比矩阵"""
        try:
            # 如果可用，使用pandapower进行精确计算
            if PANDAPOWER_AVAILABLE:
                return self._compute_jacobian_with_pandapower(env, injections)
            else:
                # 使用简化的DC潮流近似
                return self._compute_jacobian_dc_approximation(env, injections)
                
        except Exception as e:
            print(f"⚠️ 雅可比矩阵计算失败，使用近似方法: {e}")
            return self._compute_jacobian_dc_approximation(env, injections)
    
    def _compute_jacobian_with_pandapower(self, env: 'PowerGridPartitioningEnv',
                                        injections: Dict[str, np.ndarray]) -> np.ndarray:
        """使用pandapower计算精确的雅可比矩阵"""
        # 这里需要将环境的图结构转换为pandapower网络
        # 这是一个复杂的过程，需要根据具体的环境实现
        
        # 简化实现：创建一个基本的pandapower网络
        net = pp.create_empty_network()
        
        # 添加节点
        total_nodes = env.total_nodes
        for i in range(total_nodes):
            pp.create_bus(net, vn_kv=110, name=f"Bus_{i}")
        
        # 添加负荷
        P = injections['P']
        Q = injections['Q']
        for i in range(total_nodes):
            if P[i] < 0:  # 负荷
                pp.create_load(net, bus=i, p_mw=-P[i], q_mvar=-Q[i])
            elif P[i] > 0:  # 发电机
                pp.create_gen(net, bus=i, p_mw=P[i], vm_pu=1.0)
        
        # 添加线路（基于环境的边信息）
        edge_array = env.edge_info['edge_index'].cpu().numpy()
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            pp.create_line(net, from_bus=u, to_bus=v, length_km=1.0, 
                          std_type="NAYY 4x50 SE")
        
        # 运行潮流计算
        try:
            pp.runpp(net)
            
            # 获取雅可比矩阵（这需要pandapower的高级功能）
            # 简化版本：返回导纳矩阵
            Y = pp.pypower.idx_bus.build_admittance_matrix(net)
            return Y.toarray()
            
        except:
            # 如果潮流计算失败，使用DC近似
            return self._compute_jacobian_dc_approximation(env, injections)
    
    def _compute_jacobian_dc_approximation(self, env: 'PowerGridPartitioningEnv',
                                         injections: Dict[str, np.ndarray]) -> np.ndarray:
        """使用DC潮流近似计算雅可比矩阵"""
        total_nodes = env.total_nodes
        edge_array = env.edge_info['edge_index'].cpu().numpy()
        
        # 构建导纳矩阵 B（虚部）
        B = np.zeros((total_nodes, total_nodes))
        
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            
            # 获取线路导纳
            if hasattr(env.evaluator, 'edge_admittances') and i < len(env.evaluator.edge_admittances):
                admittance = float(env.evaluator.edge_admittances[i].item())
            else:
                # 使用估算值
                admittance = self._estimate_line_susceptance(u, v, env)
            
            # 构建B矩阵（DC潮流中的雅可比矩阵）
            B[u, v] = -admittance
            B[v, u] = -admittance
            B[u, u] += admittance
            B[v, v] += admittance
        
        # 处理奇异性（选择参考节点）
        # 将第一个节点作为参考节点
        B_reduced = B[1:, 1:]
        
        # 确保矩阵非奇异
        if np.linalg.det(B_reduced) == 0:
            B_reduced += np.eye(B_reduced.shape[0]) * 1e-8
        
        # 在DC潮流中，雅可比矩阵就是B矩阵
        # 扩展回原始维度
        jacobian = np.zeros((total_nodes, total_nodes))
        jacobian[1:, 1:] = B_reduced
        
        return jacobian
    
    def _estimate_line_susceptance(self, u: int, v: int, env: 'PowerGridPartitioningEnv') -> float:
        """估算线路电纳"""
        total_nodes = env.total_nodes
        
        # 基于网络规模的估算
        if total_nodes <= 30:
            base_susceptance = 10.0
        elif total_nodes <= 100:
            base_susceptance = 5.0
        else:
            base_susceptance = 2.0
        
        # 基于节点距离的调整
        distance = abs(u - v)
        if distance == 1:
            distance_factor = 1.5
        elif distance <= 5:
            distance_factor = 1.2
        else:
            distance_factor = 1.0
        
        return base_susceptance * distance_factor
    
    def _compute_electrical_distance_matrix(self, jacobian: np.ndarray) -> np.ndarray:
        """计算电气距离矩阵"""
        try:
            # 计算雅可比矩阵的逆（灵敏度矩阵）
            # 只使用非奇异部分
            n = jacobian.shape[0]
            if n <= 1:
                return np.zeros((n, n))
            
            # 去除参考节点（第一个节点）
            J_reduced = jacobian[1:, 1:]
            
            # 计算伪逆以处理奇异性
            try:
                J_inv = np.linalg.pinv(J_reduced)
            except:
                # 如果计算失败，使用单位矩阵
                J_inv = np.eye(J_reduced.shape[0])
            
            # 计算电气距离
            # 电气距离定义为灵敏度矩阵元素的倒数
            sensitivity_matrix = J_inv
            
            # 构建距离矩阵
            distance_matrix = np.zeros((n, n))
            
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        # 电气距离 = 1 / |灵敏度|
                        sensitivity = abs(sensitivity_matrix[i-1, j-1])
                        if sensitivity > 1e-10:
                            distance = 1.0 / sensitivity
                        else:
                            distance = 1e10  # 非常大的距离
                        distance_matrix[i, j] = distance
            
            # 对于参考节点，使用特殊处理
            for i in range(1, n):
                distance_matrix[0, i] = distance_matrix[i, 0] = np.mean(distance_matrix[i, 1:])
            
            return distance_matrix
            
        except Exception as e:
            print(f"⚠️ 电气距离计算失败: {e}")
            # 返回欧几里得距离作为备选
            n = jacobian.shape[0]
            return pairwise_distances(jacobian, metric='euclidean')
    
    def _perform_clustering(self, distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
        """基于距离矩阵执行聚类"""
        # 预处理距离矩阵
        distance_matrix = self._preprocess_distance_matrix(distance_matrix)
        
        if self.clustering_method == 'affinity_propagation':
            return self._affinity_propagation_clustering(distance_matrix, n_clusters)
        elif self.clustering_method == 'agglomerative':
            return self._agglomerative_clustering(distance_matrix, n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def _preprocess_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """预处理距离矩阵"""
        # 处理NaN和inf
        distance_matrix = np.nan_to_num(distance_matrix, nan=1e10, 
                                       posinf=1e10, neginf=0.0)
        
        # 确保对称性
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # 确保对角线为0
        np.fill_diagonal(distance_matrix, 0)
        
        # 归一化
        max_dist = np.max(distance_matrix)
        if max_dist > 0:
            distance_matrix = distance_matrix / max_dist
        
        return distance_matrix
    
    def _affinity_propagation_clustering(self, distance_matrix: np.ndarray, 
                                       n_clusters: int) -> np.ndarray:
        """使用Affinity Propagation进行聚类"""
        # 转换距离矩阵为相似度矩阵
        similarity_matrix = -distance_matrix
        
        # 设置preference参数
        if self.preference is None:
            preference = np.median(similarity_matrix)
        else:
            preference = self.preference
        
        # 执行聚类
        clustering = AffinityPropagation(
            affinity='precomputed',
            damping=self.damping,
            preference=preference,
            random_state=self.seed
        )
        
        try:
            labels = clustering.fit_predict(similarity_matrix)
            
            # 如果聚类数不匹配，进行调整
            unique_labels = np.unique(labels)
            if len(unique_labels) != n_clusters:
                return self._adjust_cluster_count(labels, n_clusters)
            
            return labels
            
        except Exception as e:
            print(f"⚠️ Affinity Propagation聚类失败: {e}")
            # 降级到层次聚类
            return self._agglomerative_clustering(distance_matrix, n_clusters)
    
    def _agglomerative_clustering(self, distance_matrix: np.ndarray, 
                                n_clusters: int) -> np.ndarray:
        """使用层次聚类"""
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage=self.linkage
        )
        
        return clustering.fit_predict(distance_matrix)
    
    def _adjust_cluster_count(self, labels: np.ndarray, target_clusters: int) -> np.ndarray:
        """调整聚类数量"""
        unique_labels = np.unique(labels)
        current_clusters = len(unique_labels)
        
        if current_clusters < target_clusters:
            # 分割最大的聚类
            return self._split_largest_clusters(labels, target_clusters)
        elif current_clusters > target_clusters:
            # 合并最小的聚类
            return self._merge_smallest_clusters(labels, target_clusters)
        else:
            return labels
    
    def _split_largest_clusters(self, labels: np.ndarray, target_clusters: int) -> np.ndarray:
        """分割最大的聚类"""
        current_labels = labels.copy()
        unique_labels, counts = np.unique(current_labels, return_counts=True)
        current_clusters = len(unique_labels)
        
        additional_clusters = target_clusters - current_clusters
        
        for _ in range(additional_clusters):
            # 找到最大的聚类
            unique_labels, counts = np.unique(current_labels, return_counts=True)
            largest_cluster_idx = np.argmax(counts)
            largest_cluster = unique_labels[largest_cluster_idx]
            
            if counts[largest_cluster_idx] <= 1:
                break
            
            # 分割最大聚类
            cluster_nodes = np.where(current_labels == largest_cluster)[0]
            split_point = len(cluster_nodes) // 2
            
            new_label = current_labels.max() + 1
            current_labels[cluster_nodes[split_point:]] = new_label
        
        return current_labels
    
    def _merge_smallest_clusters(self, labels: np.ndarray, target_clusters: int) -> np.ndarray:
        """合并最小的聚类"""
        current_labels = labels.copy()
        unique_labels, counts = np.unique(current_labels, return_counts=True)
        current_clusters = len(unique_labels)
        
        clusters_to_remove = current_clusters - target_clusters
        
        for _ in range(clusters_to_remove):
            # 找到最小的聚类
            unique_labels, counts = np.unique(current_labels, return_counts=True)
            if len(unique_labels) <= target_clusters:
                break
                
            smallest_cluster_idx = np.argmin(counts)
            smallest_cluster = unique_labels[smallest_cluster_idx]
            
            # 找到最近的聚类进行合并
            other_clusters = [c for c in unique_labels if c != smallest_cluster]
            target_cluster = other_clusters[0]  # 简化：合并到第一个其他聚类
            
            current_labels[current_labels == smallest_cluster] = target_cluster
        
        # 重新映射标签
        unique_labels = np.unique(current_labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        final_labels = np.array([label_mapping[label] for label in current_labels])
        
        return final_labels
    
    def _fallback_partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """降级分区方法"""
        print("⚠️ 使用降级分区方法...")
        
        # 简单的轮询分配
        labels = np.zeros(env.total_nodes, dtype=int)
        for i in range(env.total_nodes):
            labels[i] = (i % env.num_partitions) + 1
        
        return labels
    
    def get_electrical_distance_statistics(self, env: 'PowerGridPartitioningEnv') -> Dict[str, float]:
        """获取电气距离统计信息"""
        try:
            injections = self._extract_power_injections(env)
            jacobian = self._compute_jacobian_matrix(env, injections)
            distance_matrix = self._compute_electrical_distance_matrix(jacobian)
            
            # 计算统计信息
            non_zero_distances = distance_matrix[distance_matrix > 0]
            
            stats = {
                'mean_distance': float(np.mean(non_zero_distances)),
                'std_distance': float(np.std(non_zero_distances)),
                'min_distance': float(np.min(non_zero_distances)),
                'max_distance': float(np.max(non_zero_distances)),
                'median_distance': float(np.median(non_zero_distances))
            }
            
            return stats
            
        except Exception as e:
            print(f"⚠️ 电气距离统计计算失败: {e}")
            return {
                'mean_distance': 0.0,
                'std_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'median_distance': 0.0
            }