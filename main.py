#!/usr/bin/env python3
"""
主程序：智能电网分区系统演示

这个脚本运行完整的电网分区系统，包括：
1. 数据处理和图构建
2. 物理引导的GAT编码器
3. 强化学习环境和智能体
4. 训练过程
5. 结果可视化和对比
"""

import torch
import os
import warnings
import gc
import sys
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

# 忽略警告
warnings.filterwarnings('ignore')

# 设置更安全的内存管理
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 如果使用CUDA，设置内存分配策略
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 导入随机种子设置函数
from utils import set_seed

from src.data_processing import PowerGridDataProcessor
from src.gat import create_hetero_graph_encoder
from src.rl.agent import DQNAgent, ActorCriticAgent
from src.rl.env import PowerGridEnv
from src.rl.replay_buffer import ReplayBuffer

def main(training_mode='quick'):
    """
    主函数：运行完整的电网分区系统演示
    
    参数:
        training_mode: 训练模式 ('quick' 或 'full')
    """
    
    print("\n" + "="*80)
    print("🎉 智能电网分区系统 - 完整演示")
    print("="*80)
    print(f"📊 训练模式: {training_mode.upper()}")
    
    # 设置全局随机种子确保可重现性
    set_seed(42)
    print("🔧 已设置随机种子为 42，确保实验可重现性")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)  # TensorBoard日志目录
    
    try:
        # 1. 数据处理
        print("\n" + "="*50)
        print("1️⃣ 数据处理阶段")
        print("="*50)
        
        processor = PowerGridDataProcessor()
        data = processor.graph_from_mpc(mpc)
        data = ToUndirected()(data)
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. GAT编码器
        print("\n" + "="*50)
        print("2️⃣ 神经网络编码器")
        print("="*50)
        
        encoder = create_hetero_graph_encoder(
            data,
            hidden_channels=64,
            gnn_layers=3,
            heads=4,
            output_dim=128
        ).to(device)
        
        # 为了演示，我们可以进行一次前向传播
        with torch.no_grad():
            embeddings = encoder.encode_nodes(data.to(device))

        print(f"✅ GAT Encoder initialized successfully.")
        for node_type, emb in embeddings.items():
            print(f"   - {node_type} embedding shape: {emb.shape}")
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 3. 强化学习环境
        print("\n" + "="*50)
        print("3️⃣ 强化学习环境")
        print("="*50)
        
        from metrics import initialize_partition_env
        env = initialize_partition_env(data, embeddings, device)
        
        # 4. PPO智能体
        print("\n" + "="*50)
        print("4️⃣ PPO智能体")
        print("="*50)
        
        from agent import initialize_ppo_agent
        agent = initialize_ppo_agent(embeddings, env, device)
        
        # 5. 课程学习环境
        print("\n" + "="*50)
        print("5️⃣ 课程学习环境")
        print("="*50)
        
        from env import initialize_curriculum_env
        curriculum_env = initialize_curriculum_env(env)
        
        # 6. 训练过程
        print("\n" + "="*50)
        print("6️⃣ 训练过程")
        print("="*50)
        
        # 根据参数选择训练模式
        if training_mode == 'quick':
            from training import quick_training
            print("🚀 执行快速训练 ...")
            history = quick_training(agent, curriculum_env)
        elif training_mode == 'full':
            from training import full_training
            print("🎯 执行完整训练 ...")
            history = full_training(agent, curriculum_env)
        else:
            # 默认使用快速训练
            from training import quick_training
            print("使用快速训练...")
            history = quick_training(agent, curriculum_env)
        
        # 评估RL模型并保存其分区结果，用于最终展示
        from agent import evaluate_agent
        print("\n" + "="*50)
        print("📈 评估最终的RL智能体性能")
        print("="*50)
        rl_final_metrics, rl_final_env_state = evaluate_agent(agent, env, return_env_state=True)
        print("✅ RL智能体评估完成，分区状态已保存")
        
        # 7. 基线方法对比
        print("\n" + "="*50)
        print("7️⃣ 基线方法对比")
        print("="*50)
        
        # 在基线对比前再次设置随机种子，确保结果一致性
        set_seed(42)
        print("🔧 为基线对比重新设置随机种子，确保结果可重现")
        
        from baseline import run_baseline_comparison
        comparison_df = run_baseline_comparison(env, agent, seed=42)
        
        # 8. 结果可视化
        print("\n" + "="*50)
        print("8️⃣ 结果可视化")
        print("="*50)
        
        try:
            from visualization import run_basic_visualization
            # 使用保存的RL环境状态进行可视化
            run_basic_visualization(rl_final_env_state, history)
            print("✅ 基础可视化完成")
        except Exception as e:
            print(f"⚠️ 基础可视化失败: {e}")
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 如果有可视化依赖，尝试交互式可视化
        try:
            from visualization import run_interactive_visualization
            # 使用保存的RL环境状态进行可视化
            run_interactive_visualization(rl_final_env_state, comparison_df)
            print("✅ 交互式可视化完成")
        except ImportError as e:
            print(f"⚠️ 跳过交互式可视化 (缺少依赖: {e})")
        except Exception as e:
            print(f"⚠️ 交互式可视化出错: {e}")
        
        # 9. 总结
        print("\n" + "="*80)
        print("🎉 演示完成！项目成果总结")
        print("="*80)
        
        # 使用评估RL模型时得到的指标
        final_metrics = rl_final_metrics
        print(f"\n📊 最终RL模型分区指标：")
        print(f"   • 负荷变异系数 (CV): {final_metrics.load_cv:.4f}")
        print(f"   • 总耦合度: {final_metrics.total_coupling:.4f}")
        print(f"   • 连通性: {final_metrics.connectivity:.2f}")
        print(f"   • 功率平衡度: {final_metrics.power_balance:.4f}")
        print(f"   • 模块度: {final_metrics.modularity:.4f}")
        
        print(f"\n📈 训练统计：")
        print(f"   • 训练回合数: {len(history['episode_rewards'])}")
        print(f"   • 平均奖励: {sum(history['episode_rewards'])/len(history['episode_rewards']):.3f}")
        print(f"   • 最佳回合奖励: {max(history['episode_rewards']):.3f}")
        
        print(f"\n🏆 方法对比排名：")
        top_method = comparison_df.index[0]
        top_score = comparison_df.iloc[0]['overall_score']
        print(f"   • 最佳方法: {top_method}")
        print(f"   • 综合得分: {top_score:.4f}")
        
        print(f"\n📁 输出文件：")
        print(f"   • 模型保存: models/")
        print(f"   • 图表保存: figures/")
        print(f"   • TensorBoard日志: runs/")
        print(f"   • 训练检查点: models/training_final.pth")
          
        print(f"\n🔧 使用提示：")
        print(f"   • 查看训练曲线: tensorboard --logdir=runs")
        print(f"   • 完整训练: python run_training.py --mode full")
        print(f"   • 恢复训练: python run_training.py --mode full --resume models/training_checkpoint_ep100.pth")
        print(f"   • 所有基线方法现在使用相同随机种子，结果可重现")
        
        # 最终清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        print(f"\n🔧 建议解决方案:")
        print(f"   1. 检查所有依赖库是否正确安装")
        print(f"   2. 确保有足够的内存和计算资源")
        print(f"   3. 检查Python版本兼容性")
        
        # 清理内存，即使出错也要释放资源
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # 打印详细错误信息（用于调试）
        import traceback
        print(f"\n🐛 详细错误信息:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='智能电网分区系统完整演示')
    parser.add_argument('--mode', type=str, choices=['quick', 'full'], default='quick',
                      help='训练模式: quick(快速演示) 或 full(完整训练)')
    
    args = parser.parse_args()
    
    # 运行主程序
    success = main(training_mode=args.mode)
    
    if success:
        print(f"\n🎊 程序成功完成！")
        if args.mode == 'full':
            print(f"💡 提示: 使用 'tensorboard --logdir=runs' 查看训练过程")
    else:
        print(f"\n❌ 程序执行失败，请检查错误信息。")