import numpy as np
import torch
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from typing import Dict, List, Optional
import pandas as pd
import gc
import random
import os

# Import types that will be defined in other modules
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env import PowerGridPartitionEnv

def set_baseline_seed(seed: int = 42):
    """为基线方法设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BaselinePartitioner:
    """
    基线分区方法（用于对比）
    """
    
    def __init__(self, method: str = 'spectral', seed: int = 42):
        self.method = method
        self.seed = seed
    
    def partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """执行分区"""
        # 每次执行前设置随机种子
        set_baseline_seed(self.seed)
        
        if self.method == 'spectral':
            return self._spectral_partition(env)
        elif self.method == 'kmeans':
            return self._kmeans_partition(env)
        elif self.method == 'random':
            return self._random_partition(env)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _spectral_partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """谱聚类分区"""
        # 再次确保种子设置
        set_baseline_seed(self.seed)
        
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
    
    def _kmeans_partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """K-means分区（基于节点嵌入）"""
        import gc
        import os
        
        # 再次确保种子设置
        set_baseline_seed(self.seed)
        
        # 强制设置单线程环境变量，避免多线程冲突
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        
        try:
            # 确保使用CPU版本的嵌入，避免CUDA内存问题
            embeddings = env.embeddings.detach().cpu().numpy().copy()
            
            # 检查和清理数据
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                print("⚠️ 警告：嵌入中存在异常值，进行清理...")
                embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 简化的K-means实现，避免段错误
            from sklearn.cluster import MiniBatchKMeans
            
            # 使用MiniBatchKMeans，它更稳定且内存友好
            kmeans = MiniBatchKMeans(
                n_clusters=env.K,
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
            return self._random_partition(env)
    
    def _random_partition(self, env: 'PowerGridPartitionEnv') -> np.ndarray:
        """随机分区"""
        # 确保使用固定种子
        set_baseline_seed(self.seed)
        labels = np.random.randint(1, env.K + 1, size=env.N)
        return labels


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


def compare_methods(env: 'PowerGridPartitionEnv', agent, seed: int = 42) -> pd.DataFrame:
    """
    比较不同分区方法
    """
    import gc
    import torch
    
    # 在对比开始前设置全局随机种子
    set_baseline_seed(seed)
    
    results = []
    
    try:
        # 1. RL方法
        print("\n🤖 评估RL方法...")
        env.reset()
        
        # 使用训练好的智能体进行分区
        state = env.get_state()
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            action_value = agent.select_action(state, valid_actions, training=False)
            if action_value is None:
                break
            
            action, _ = action_value
            state, _, done, _ = env.step(action)
        
        rl_metrics = evaluate_partition_method(env, env.z.cpu().numpy())
        rl_metrics['method'] = 'RL (PPO)'
        results.append(rl_metrics)
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"⚠️ RL方法评估失败: {str(e)}")
        results.append({'method': 'RL (PPO)', 'load_cv': 999, 'total_coupling': 999, 
                       'connectivity': 0, 'modularity': 0})

    try:
        # 2. 谱聚类
        print("📊 评估谱聚类...")
        spectral = BaselinePartitioner('spectral', seed=seed)
        spectral_partition = spectral.partition(env)
        spectral_metrics = evaluate_partition_method(env, spectral_partition)
        spectral_metrics['method'] = 'Spectral Clustering'
        results.append(spectral_metrics)
        gc.collect()

    except Exception as e:
        print(f"⚠️ 谱聚类评估失败: {str(e)}")
        results.append({'method': 'Spectral Clustering', 'load_cv': 999, 'total_coupling': 999,
                       'connectivity': 0, 'modularity': 0})

    try:
        # 3. K-means
        print("📊 评估K-means...")
        kmeans = BaselinePartitioner('kmeans', seed=seed)
        kmeans_partition = kmeans.partition(env)
        kmeans_metrics = evaluate_partition_method(env, kmeans_partition)
        kmeans_metrics['method'] = 'K-means'
        results.append(kmeans_metrics)
        gc.collect()

    except Exception as e:
        print(f"⚠️ K-means评估失败: {str(e)}")
        results.append({'method': 'K-means', 'load_cv': 999, 'total_coupling': 999,
                       'connectivity': 0, 'modularity': 0})

    try:
        # 4. 随机分区
        print("🎲 评估随机分区...")
        random_partitioner = BaselinePartitioner('random', seed=seed)
        random_partition = random_partitioner.partition(env)
        random_metrics = evaluate_partition_method(env, random_partition)
        random_metrics['method'] = 'Random'
        results.append(random_metrics)
        gc.collect()

    except Exception as e:
        print(f"⚠️ 随机分区评估失败: {str(e)}")
        results.append({'method': 'Random', 'load_cv': 999, 'total_coupling': 999,
                       'connectivity': 0, 'modularity': 0})
    
    # 创建结果DataFrame，确保至少有一些结果
    if not results:
        results = [{'method': 'No Method Succeeded', 'load_cv': 999, 'total_coupling': 999,
                   'connectivity': 0, 'modularity': 0}]
    
    df = pd.DataFrame(results)
    df = df.set_index('method')
    
    return df


def run_baseline_comparison(env, agent, seed: int = 42):
    """Test function for baseline methods comparison"""
    # 设置随机种子确保可重现性
    set_baseline_seed(seed)
    
    # 执行对比
    print(f"\n📊 执行方法对比（随机种子：{seed}）...")
    comparison_df = compare_methods(env, agent, seed=seed)

    # 显示结果
    print("\n📋 分区方法对比结果:")
    print(comparison_df.round(4))

    # 计算综合得分（越小越好的指标取负值）
    weights = {
        'load_cv': -0.3,
        'total_coupling': -0.25,
        'connectivity': 0.2,
        'modularity': 0.15,
        'power_balance': -0.1
    }

    scores = []
    for method in comparison_df.index:
        score = sum(comparison_df.loc[method, metric] * weight 
                   for metric, weight in weights.items() 
                   if metric in comparison_df.columns)
        scores.append(score)

    comparison_df['overall_score'] = scores
    comparison_df = comparison_df.sort_values('overall_score', ascending=False)

    print("\n🏆 综合评分排名:")
    print(comparison_df[['overall_score']].round(4))
    
    return comparison_df

