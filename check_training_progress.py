#!/usr/bin/env python3
"""
训练进度检查脚本
快速查看当前训练状态和统计信息
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """查找最新的检查点文件"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
        
    checkpoint_files = list(checkpoint_path.glob("training_stats_episode_*.json"))
    if not checkpoint_files:
        return None
        
    latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    return latest_file

def find_latest_results(results_dir: str = "experiments") -> Optional[Path]:
    """查找最新的训练结果文件"""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None
        
    result_files = list(results_path.glob("*_results_*.json"))
    if not result_files:
        return None
        
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    return latest_file

def load_json_file(file_path: Path) -> Dict:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 加载文件失败 {file_path}: {e}")
        return {}

def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"

def print_training_status(checkpoint_data: Dict):
    """打印训练状态"""
    print("📊 当前训练状态")
    print("=" * 50)
    
    if 'total_episodes' in checkpoint_data:
        print(f"总回合数: {checkpoint_data['total_episodes']}")
    
    if 'best_reward' in checkpoint_data:
        print(f"最佳奖励: {checkpoint_data['best_reward']:.4f}")
    
    if 'mean_reward' in checkpoint_data:
        print(f"平均奖励: {checkpoint_data['mean_reward']:.4f}")
    
    if 'std_reward' in checkpoint_data:
        print(f"奖励标准差: {checkpoint_data['std_reward']:.4f}")
    
    if 'mean_length' in checkpoint_data:
        print(f"平均回合长度: {checkpoint_data['mean_length']:.1f}")
    
    if 'success_rate' in checkpoint_data:
        print(f"成功率: {checkpoint_data['success_rate']:.3f}")
    
    if 'mean_load_cv' in checkpoint_data:
        print(f"平均负载CV: {checkpoint_data['mean_load_cv']:.4f}")
    
    if 'training_time' in checkpoint_data:
        print(f"训练时间: {format_time(checkpoint_data['training_time'])}")
    
    if 'mean_actor_loss' in checkpoint_data:
        print(f"平均Actor损失: {checkpoint_data['mean_actor_loss']:.6f}")
    
    if 'mean_critic_loss' in checkpoint_data:
        print(f"平均Critic损失: {checkpoint_data['mean_critic_loss']:.6f}")

def print_final_results(results_data: Dict):
    """打印最终结果"""
    print("\n🎯 最终训练结果")
    print("=" * 50)
    
    if 'mode' in results_data:
        print(f"训练模式: {results_data['mode']}")
    
    if 'success' in results_data:
        status = "✅ 成功" if results_data['success'] else "❌ 失败"
        print(f"训练状态: {status}")
    
    if 'best_reward' in results_data:
        print(f"最佳奖励: {results_data['best_reward']:.4f}")
    
    # 配置信息
    if 'config' in results_data:
        config = results_data['config']
        print(f"\n📋 配置信息:")
        if 'data' in config and 'case_name' in config['data']:
            print(f"  案例: {config['data']['case_name']}")
        if 'environment' in config and 'num_partitions' in config['environment']:
            print(f"  分区数: {config['environment']['num_partitions']}")
        if 'training' in config and 'num_episodes' in config['training']:
            print(f"  训练回合: {config['training']['num_episodes']}")
    
    # 评估统计
    if 'eval_stats' in results_data:
        eval_stats = results_data['eval_stats']
        print(f"\n📈 评估统计:")
        if 'avg_reward' in eval_stats:
            print(f"  平均奖励: {eval_stats['avg_reward']:.4f}")
        if 'success_rate' in eval_stats:
            print(f"  成功率: {eval_stats['success_rate']:.3f}")
    
    # 训练历史
    if 'history' in results_data:
        history = results_data['history']
        if 'episode_rewards' in history:
            rewards = history['episode_rewards']
            print(f"\n📊 训练历史:")
            print(f"  总回合数: {len(rewards)}")
            if rewards:
                print(f"  最终奖励: {rewards[-1]:.4f}")
                print(f"  最大奖励: {max(rewards):.4f}")
                print(f"  最小奖励: {min(rewards):.4f}")

def check_tensorboard_logs(log_dir: str):
    """检查TensorBoard日志"""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"⚠️ 日志目录不存在: {log_dir}")
        return
    
    # 查找训练日志目录
    training_dirs = [d for d in log_path.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    if training_dirs:
        print(f"\n📊 TensorBoard日志:")
        print(f"  日志目录: {log_dir}")
        print(f"  可用日志: {len(training_dirs)} 个")
        
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        print(f"  最新日志: {latest_dir.name}")
        print(f"  启动命令: tensorboard --logdir={log_dir}")
    else:
        print(f"⚠️ 未找到TensorBoard日志")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查训练进度')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                       help='检查点目录')
    parser.add_argument('--results-dir', type=str, default='experiments', 
                       help='结果目录')
    parser.add_argument('--log-dir', type=str, default='logs', 
                       help='日志目录')
    parser.add_argument('--watch', action='store_true', 
                       help='持续监控模式')
    parser.add_argument('--interval', type=int, default=10, 
                       help='监控间隔（秒）')
    
    args = parser.parse_args()
    
    def check_once():
        """执行一次检查"""
        print(f"\n🔍 训练进度检查 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 检查最新的检查点
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            print(f"📁 最新检查点: {latest_checkpoint.name}")
            checkpoint_data = load_json_file(latest_checkpoint)
            if checkpoint_data:
                print_training_status(checkpoint_data)
        else:
            print("⚠️ 未找到训练检查点")
        
        # 检查最终结果
        latest_results = find_latest_results(args.results_dir)
        if latest_results:
            print(f"\n📁 最新结果: {latest_results.name}")
            results_data = load_json_file(latest_results)
            if results_data:
                print_final_results(results_data)
        else:
            print("\n⚠️ 未找到训练结果")
        
        # 检查TensorBoard日志
        check_tensorboard_logs(args.log_dir)
    
    if args.watch:
        print(f"👀 持续监控模式 (间隔: {args.interval}秒)")
        print("按 Ctrl+C 停止监控")
        try:
            while True:
                check_once()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 监控已停止")
    else:
        check_once()

if __name__ == "__main__":
    main()
