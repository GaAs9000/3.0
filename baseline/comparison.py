import pandas as pd
import torch
import gc
from typing import TYPE_CHECKING

from .baseline import set_baseline_seed
from .spectral_clustering import SpectralPartitioner
from .kmeans_clustering import KMeansPartitioner
from .random_partition import RandomPartitioner
from .evaluator import evaluate_partition_method

if TYPE_CHECKING:
    from env import PowerGridPartitionEnv


def compare_methods(env: 'PowerGridPartitionEnv', agent, seed: int = 42) -> pd.DataFrame:
    """
    比较不同分区方法
    """
    # 在对比开始前设置全局随机种子
    set_baseline_seed(seed)
    
    results = []
    
    try:
        # 1. RL方法
        print("\n🤖 评估RL方法...")
        obs_dict, _ = env.reset()

        # 使用训练好的智能体进行分区
        state = obs_dict
        done = False

        while not done:
            # 使用智能体选择动作（智能体内部会处理动作掩码）
            action, _, _ = agent.select_action(state, training=False)
            if action is None:
                break

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_obs
        
        rl_metrics = evaluate_partition_method(env, env.state_manager.current_partition.cpu().numpy())
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
        spectral = SpectralPartitioner(seed=seed)
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
        kmeans = KMeansPartitioner(seed=seed)
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
        random_partitioner = RandomPartitioner(seed=seed)
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
    """执行基线方法对比的主函数"""
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