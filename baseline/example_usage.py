"""
Baseline方法使用示例
展示如何使用拆分后的模块
"""

# 方式1: 使用原有的接口（向后兼容）
from baseline import BaselinePartitioner, run_baseline_comparison

# 方式2: 使用新的模块化接口
from baseline import SpectralPartitioner, KMeansPartitioner, RandomPartitioner
from baseline import evaluate_partition_method, compare_methods

# 方式3: 直接导入特定模块
from baseline.spectral_clustering import SpectralPartitioner
from baseline.kmeans_clustering import KMeansPartitioner
from baseline.random_partition import RandomPartitioner
from baseline.comparison import run_baseline_comparison


def example_usage_old_interface(env, agent):
    """使用原有接口的示例"""
    print("=== 使用原有接口 ===")
    
    # 创建不同的基线分区器
    spectral = BaselinePartitioner('spectral', seed=42)
    kmeans = BaselinePartitioner('kmeans', seed=42)
    random = BaselinePartitioner('random', seed=42)
    
    # 执行分区
    spectral_result = spectral.partition(env)
    kmeans_result = kmeans.partition(env)
    random_result = random.partition(env)
    
    print(f"谱聚类结果: {spectral_result[:10]}...")
    print(f"K-means结果: {kmeans_result[:10]}...")
    print(f"随机分区结果: {random_result[:10]}...")
    
    # 运行完整对比
    comparison_results = run_baseline_comparison(env, agent, seed=42)
    return comparison_results


def example_usage_new_interface(env, agent):
    """使用新模块化接口的示例"""
    print("=== 使用新模块化接口 ===")
    
    # 创建不同的分区器
    spectral = SpectralPartitioner(seed=42)
    kmeans = KMeansPartitioner(seed=42)
    random = RandomPartitioner(seed=42)
    
    # 执行分区
    spectral_result = spectral.partition(env)
    kmeans_result = kmeans.partition(env)
    random_result = random.partition(env)
    
    print(f"谱聚类结果: {spectral_result[:10]}...")
    print(f"K-means结果: {kmeans_result[:10]}...")
    print(f"随机分区结果: {random_result[:10]}...")
    
    # 评估单个方法
    spectral_metrics = evaluate_partition_method(env, spectral_result)
    print(f"谱聚类指标: {spectral_metrics}")
    
    # 运行完整对比
    comparison_results = compare_methods(env, agent, seed=42)
    return comparison_results


def example_individual_methods(env):
    """单独使用各个方法的示例"""
    print("=== 单独使用各个方法 ===")
    
    # 只使用谱聚类
    spectral = SpectralPartitioner(seed=42)
    result = spectral.partition(env)
    metrics = evaluate_partition_method(env, result)
    print(f"谱聚类 - 负载CV: {metrics['load_cv']:.4f}, 连通性: {metrics['connectivity']:.4f}")
    
    # 只使用K-means
    kmeans = KMeansPartitioner(seed=42)
    result = kmeans.partition(env)
    metrics = evaluate_partition_method(env, result)
    print(f"K-means - 负载CV: {metrics['load_cv']:.4f}, 连通性: {metrics['connectivity']:.4f}")
    
    # 只使用随机分区
    random = RandomPartitioner(seed=42)
    result = random.partition(env)
    metrics = evaluate_partition_method(env, result)
    print(f"随机分区 - 负载CV: {metrics['load_cv']:.4f}, 连通性: {metrics['connectivity']:.4f}")


if __name__ == "__main__":
    # 这里需要实际的env和agent实例
    # env = PowerGridPartitionEnv(...)
    # agent = trained_agent
    
    print("这是一个使用示例文件")
    print("请在实际环境中导入相应的环境和智能体实例")
    
    # 示例调用（需要实际的env和agent）
    # example_usage_old_interface(env, agent)
    # example_usage_new_interface(env, agent)
    # example_individual_methods(env) 