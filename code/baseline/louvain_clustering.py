import numpy as np
import networkx as nx
from typing import TYPE_CHECKING
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from ..src.rl.environment import PowerGridPartitioningEnv

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


class LouvainPartitioner(BasePartitioner):
    """
    Louvain社区发现分区方法
    
    基于模块度优化的社区发现算法，通过迭代优化模块度函数来发现网络中的社区结构。
    适用于大规模网络的快速社区发现。
    """
    
    def __init__(self, seed: int = 42, resolution: float = 1.0):
        """
        初始化Louvain分区器
        
        Args:
            seed: 随机种子，用于保证结果可重复性
            resolution: 分辨率参数，控制社区的大小。值越大，社区越小
        """
        super().__init__(seed)
        self.resolution = resolution
        
        if not LOUVAIN_AVAILABLE:
            raise ImportError("python-louvain库未安装。请运行: pip install python-louvain")
    
    def partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """执行Louvain社区发现分区"""
        # 设置随机种子
        self._set_seed()
        
        try:
            # 构建NetworkX图
            G = self._build_networkx_graph(env)
            
            # 如果网络太小，使用简单的分区方法
            if G.number_of_nodes() <= env.num_partitions:
                return self._simple_partition(G, env.num_partitions)
            
            # 运行Louvain算法
            # 为了保证实验可重复性，统一使用random_state=seed
            partition = community_louvain.best_partition(
                G, 
                resolution=self.resolution, 
                random_state=self.seed
            )
            
            # 转换分区结果
            labels = self._convert_partition_to_labels(partition, env)
            
            return labels
            
        except Exception as e:
            print(f"❌ Louvain分区失败: {str(e)}")
            # 降级到简单的分区方法
            return self._fallback_partition(env)
    
    def _build_networkx_graph(self, env: 'PowerGridPartitioningEnv') -> nx.Graph:
        """构建NetworkX图，使用导纳作为边权重"""
        G = nx.Graph()
        
        # 获取边信息
        edge_array = env.edge_info['edge_index'].cpu().numpy()
        total_nodes = env.total_nodes
        
        # 添加所有节点
        G.add_nodes_from(range(total_nodes))
        
        # 添加边和权重
        for i in range(edge_array.shape[1]):
            u, v = edge_array[0, i], edge_array[1, i]
            
            # 获取边权重（导纳绝对值）
            if hasattr(env.evaluator, 'edge_admittances') and i < len(env.evaluator.edge_admittances):
                weight = float(env.evaluator.edge_admittances[i].item())
            else:
                # 使用估算的权重
                weight = self._estimate_edge_weight(u, v, total_nodes)
            
            # 处理异常权重
            if np.isnan(weight) or np.isinf(weight) or weight <= 0:
                weight = 1.0
            
            G.add_edge(u, v, weight=weight)
        
        return G
    
    def _estimate_edge_weight(self, u: int, v: int, total_nodes: int) -> float:
        """估算边权重（与谱聚类中相同的策略）"""
        # 基于网络规模的基础权重
        if total_nodes <= 30:
            base_weight = 3.0
        elif total_nodes <= 100:
            base_weight = 2.0
        else:
            base_weight = 1.5
        
        # 基于节点索引的调整
        index_diff = abs(u - v)
        if index_diff == 1:
            weight_factor = 1.2
        elif index_diff <= 5:
            weight_factor = 1.1
        else:
            weight_factor = 1.0
        
        return base_weight * weight_factor
    
    def _convert_partition_to_labels(self, partition: dict, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """
        将Louvain分区结果转换为标签数组
        
        Args:
            partition: Louvain算法返回的分区字典 {node_id: community_id}
            env: 环境对象
            
        Returns:
            标签数组，从1开始编号
        """
        # 获取原始社区标签
        original_labels = np.array([partition[i] for i in range(env.total_nodes)])
        
        # 获取唯一的社区ID
        unique_communities = np.unique(original_labels)
        num_communities = len(unique_communities)
        
        # 如果社区数量等于目标分区数，直接返回
        if num_communities == env.num_partitions:
            # 重新映射到1开始的连续标签
            label_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_communities)}
            return np.array([label_mapping[label] for label in original_labels])
        
        # 如果社区数量小于目标分区数，使用细分策略
        elif num_communities < env.num_partitions:
            return self._subdivide_communities(original_labels, unique_communities, env)
        
        # 如果社区数量大于目标分区数，使用合并策略
        else:
            return self._merge_communities(original_labels, unique_communities, env)
    
    def _subdivide_communities(self, labels: np.ndarray, communities: np.ndarray, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """细分社区以达到目标分区数"""
        print(f"🔄 Louvain发现{len(communities)}个社区，目标{env.num_partitions}个分区，执行细分...")
        
        # 找到最大的社区进行细分
        current_labels = labels.copy()
        current_num_partitions = len(communities)
        
        # 重新映射到1开始的连续标签
        label_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(communities)}
        current_labels = np.array([label_mapping[label] for label in current_labels])
        
        # 需要额外的分区数
        additional_partitions = env.num_partitions - current_num_partitions
        
        for _ in range(additional_partitions):
            # 找到最大的社区
            unique_labels, counts = np.unique(current_labels, return_counts=True)
            largest_community_idx = np.argmax(counts)
            largest_community = unique_labels[largest_community_idx]
            
            # 如果最大社区只有一个节点，停止细分
            if counts[largest_community_idx] <= 1:
                break
            
            # 分割最大社区
            largest_community_nodes = np.where(current_labels == largest_community)[0]
            split_point = len(largest_community_nodes) // 2
            
            # 创建新的标签
            new_label = current_labels.max() + 1
            current_labels[largest_community_nodes[split_point:]] = new_label
        
        return current_labels
    
    def _merge_communities(self, labels: np.ndarray, communities: np.ndarray, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """合并社区以达到目标分区数"""
        print(f"🔄 Louvain发现{len(communities)}个社区，目标{env.num_partitions}个分区，执行合并...")
        
        # 计算每个社区的大小
        community_sizes = []
        for community in communities:
            size = np.sum(labels == community)
            community_sizes.append((community, size))
        
        # 按大小排序，保留最大的几个社区
        community_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 保留前num_partitions个最大社区
        kept_communities = [comm[0] for comm in community_sizes[:env.num_partitions]]
        
        # 将剩余的小社区合并到最近的大社区
        merged_labels = labels.copy()
        
        for community in communities:
            if community not in kept_communities:
                # 找到最接近的保留社区（这里简化为最小的保留社区）
                target_community = kept_communities[-1]  # 最小的保留社区
                merged_labels[labels == community] = target_community
        
        # 重新映射到1开始的连续标签
        unique_labels = np.unique(merged_labels)
        label_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_labels)}
        final_labels = np.array([label_mapping[label] for label in merged_labels])
        
        return final_labels
    
    def _simple_partition(self, G: nx.Graph, num_partitions: int) -> np.ndarray:
        """简单的分区方法，用于小网络"""
        nodes = list(G.nodes())
        labels = np.zeros(len(nodes), dtype=int)
        
        # 简单的轮询分配
        for i, node in enumerate(nodes):
            labels[node] = (i % num_partitions) + 1
        
        return labels
    
    def _fallback_partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """降级分区方法"""
        print("⚠️ 使用降级分区方法...")
        
        # 简单的轮询分配
        labels = np.zeros(env.total_nodes, dtype=int)
        for i in range(env.total_nodes):
            labels[i] = (i % env.num_partitions) + 1
        
        return labels
    
    def get_modularity(self, env: 'PowerGridPartitioningEnv', labels: np.ndarray) -> float:
        """计算分区的模块度"""
        try:
            G = self._build_networkx_graph(env)
            
            # 将标签转换为社区字典
            partition = {i: labels[i] for i in range(len(labels))}
            
            # 计算模块度
            modularity = community_louvain.modularity(partition, G)
            
            return modularity
            
        except Exception as e:
            print(f"⚠️ 计算模块度失败: {e}")
            return 0.0