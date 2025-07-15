#!/usr/bin/env python3
"""
电力网络分区智能体测试系统 - 清理版本
只使用evaluators/comprehensive_evaluator.py
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'code' / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from evaluators.comprehensive_evaluator import ComprehensiveEvaluator
    ComprehensiveEvaluator_available = True
    print("✅ ComprehensiveEvaluator导入成功")
except ImportError as e:
    print(f"⚠️ 警告: 无法导入ComprehensiveEvaluator: {e}")
    ComprehensiveEvaluator = None
    ComprehensiveEvaluator_available = False


def run_baseline_comparison(config_path: Optional[str] = None, 
                          network: str = 'ieee14', 
                          mode: str = 'quick',
                          model_path: Optional[str] = None, 
                          custom_scenarios: Optional[List[str]] = None,
                          custom_runs: Optional[int] = None):
    """运行Baseline对比测试"""
    print(f"🔍 启动Baseline对比测试")
    print(f"   测试网络: {network.upper()}")
    print(f"   评估模式: {mode}")
    if model_path:
        print(f"   预训练模型: {model_path}")
    print("=" * 50)

    if not ComprehensiveEvaluator_available:
        print("❌ 错误: ComprehensiveEvaluator不可用，请检查依赖")
        return False

    try:
        # 创建带时间戳的输出目录
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        output_dir = Path(__file__).parent.parent / 'evaluation' / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'comparisons').mkdir(exist_ok=True)
        (output_dir / 'reports').mkdir(exist_ok=True)
        
        # 创建配置
        config = {
            'system': {'device': 'auto'},
            'data': {
                'normalize': True,
                'cache_dir': 'data/latest/cache'
            },
            'environment': {
                'num_partitions': 3,
                'max_steps': 200,
                'reward_weights': {'balance': 1.0, 'connectivity': 0.5, 'size': 0.3}
            },
            'gat': {
                'hidden_channels': 64,
                'gnn_layers': 3,
                'heads': 4,
                'output_dim': 128,
                'dropout': 0.1,
                'edge_dim': 9,
                'physics_enhanced': True,
                'temperature': 1.0,
                'physics_weight': 1.0
            },
            'agent': {
                'learning_rate': 3e-4,
                'gamma': 0.99
            }
        }
        
        # 创建ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator(config, output_dir)
        
        # 运行对比测试
        print("🚀 开始运行评估...")
        results = evaluator.evaluate_all_scenarios(model_path, network, custom_runs or 3)

        print(f"\n✅ Baseline对比测试完成！")
        print(f"   结果已保存到: {output_dir}")
        print(f"   包含所有方法的2×4可视化图已生成")
        
        return True

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='电力网络分区智能体测试系统')
    
    subparsers = parser.add_subparsers(dest='command', help='测试模式')
    
    # Baseline对比测试
    baseline_parser = subparsers.add_parser('baseline', help='Baseline对比测试')
    baseline_parser.add_argument('--config', type=str, help='配置文件路径')
    baseline_parser.add_argument('--mode', choices=['quick', 'standard', 'comprehensive'], 
                               default='quick', help='评估模式')
    baseline_parser.add_argument('--model', type=str, required=True, help='预训练模型路径')
    baseline_parser.add_argument('--scenarios', nargs='+', 
                               choices=['normal', 'high_load', 'unbalanced', 'fault'],
                               help='测试场景')
    baseline_parser.add_argument('--runs', type=int, help='每个场景的运行次数')
    baseline_parser.add_argument('--network', choices=['ieee14', 'ieee30', 'ieee57', 'ieee118'],
                               default='ieee14', help='测试网络')
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print(f"🎯 电力网络分区智能体测试系统")
    print("=" * 60)
    print(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.command == 'baseline':
        success = run_baseline_comparison(
            config_path=args.config,
            network=args.network,
            mode=args.mode,
            model_path=args.model,
            custom_scenarios=args.scenarios,
            custom_runs=args.runs
        )
        
        if success:
            print("\n🎉 测试成功完成！")
        else:
            print("\n❌ 测试失败")
            sys.exit(1)
    else:
        print(f"❌ 未知命令: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()