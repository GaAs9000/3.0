#!/usr/bin/env python3
"""
电力网络分区智能体综合评估系统 - 主入口

集成所有评估功能：
- Baseline方法对比测试
- 跨网络泛化能力评估
- 自动化报告生成
- 可视化分析
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# 添加路径
sys.path.append(str(Path(__file__).parent / 'code' / 'src'))
sys.path.append(str(Path(__file__).parent / 'code'))
sys.path.append(str(Path(__file__).parent / 'code' / 'test'))

from comprehensive_evaluator import ComprehensiveAgentEvaluator


def run_baseline_comparison(config_path: Optional[str] = None, network: str = 'ieee57', mode: str = 'standard', model_path: Optional[str] = None):
    """运行baseline对比测试"""
    print(f"🔍 启动Baseline对比测试")
    print(f"   测试网络: {network.upper()}")
    print(f"   评估模式: {mode}")
    if model_path:
        print(f"   预训练模型: {model_path}")
    print("=" * 50)

    try:
        # 创建评估器
        evaluator = ComprehensiveAgentEvaluator(config_path, model_path)

        # 运行对比测试
        results = evaluator.run_baseline_comparison(network)

        if results['success']:
            print(f"\n✅ Baseline对比测试完成！")

            # 生成可视化
            evaluator.create_comparison_visualization(results['results'])

            # 生成报告
            report = evaluator.generate_evaluation_report(comparison_results=results)

            # 打印简要总结
            summary = results['summary']
            avg_improvement = summary['overall_improvement']['average_improvement']
            print(f"\n🏆 测试总结:")
            print(f"   平均性能提升: {avg_improvement:.1f}%")

            for scenario, scenario_summary in summary['best_scenarios'].items():
                improvement = scenario_summary['improvement_pct']
                status = "🎉 优秀" if improvement > 15 else "✅ 良好" if improvement > 5 else "⚠️ 有限"
                print(f"   {scenario}: {improvement:.1f}% - {status}")

            return True
        else:
            print(f"❌ 测试失败: {results.get('error', '未知错误')}")
            return False

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_generalization_test(config_path: Optional[str] = None, train_network: str = 'ieee14', mode: str = 'standard', model_path: Optional[str] = None):
    """运行泛化能力测试"""
    print(f"🌐 启动泛化能力测试")
    print(f"   训练网络: {train_network.upper()}")
    print(f"   评估模式: {mode}")
    if model_path:
        print(f"   预训练模型: {model_path}")
    print("=" * 50)

    try:
        # 创建评估器
        evaluator = ComprehensiveAgentEvaluator(config_path, model_path)

        # 运行泛化测试
        results = evaluator.run_generalization_test(train_network)

        if results['success']:
            print(f"\n✅ 泛化能力测试完成！")

            # 生成可视化
            evaluator.create_generalization_visualization(results['results'], train_network)

            # 生成报告
            report = evaluator.generate_evaluation_report(generalization_results=results)

            # 打印简要总结
            summary = results['summary']
            overall = summary['overall_generalization']
            avg_degradation = overall['average_degradation']
            status = overall['status']

            print(f"\n🏆 测试总结:")
            print(f"   训练网络性能: {summary['train_score']:.3f}")
            print(f"   平均性能下降: {avg_degradation:.1f}%")
            print(f"   泛化评级: {status}")

            for network, result in summary['test_results'].items():
                print(f"   {network.upper()}: {result['score']:.3f} ({result['degradation_pct']:.1f}% 下降) - {result['status']}")

            return True
        else:
            print(f"❌ 测试失败: {results.get('error', '未知错误')}")
            return False

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_evaluation(config_path: Optional[str] = None, mode: str = 'standard', model_path: Optional[str] = None):
    """运行完整评估（包含baseline对比和泛化测试）"""
    print(f"🎯 启动完整评估")
    print(f"   评估模式: {mode}")
    if model_path:
        print(f"   预训练模型: {model_path}")
    else:
        print("❌ 错误：必须提供预训练模型路径")
        print("💡 使用方法: python test.py --comprehensive --model <模型路径>")
        return False
    print("=" * 50)

    try:
        # 创建评估器
        evaluator = ComprehensiveAgentEvaluator(config_path, model_path)

        # 获取评估配置
        eval_config = evaluator.get_evaluation_config(mode)

        comparison_results = None
        generalization_results = None

        # 1. Baseline对比测试
        if eval_config['baseline_comparison']['enabled']:
            print(f"\n📊 第一阶段: Baseline对比测试")
            test_networks = eval_config['baseline_comparison']['test_networks']

            for network in test_networks[:1]:  # 只测试第一个网络以节省时间
                print(f"\n   测试网络: {network.upper()}")
                results = evaluator.run_baseline_comparison(network)
                if results['success']:
                    comparison_results = results
                    print(f"   ✅ {network.upper()} 对比测试完成")
                    break

        # 2. 泛化能力测试
        if eval_config['generalization_test']['enabled']:
            print(f"\n🌐 第二阶段: 泛化能力测试")
            train_network = eval_config['generalization_test']['train_network']

            results = evaluator.run_generalization_test(train_network)
            if results['success']:
                generalization_results = results
                print(f"   ✅ 泛化测试完成")

        # 3. 生成综合报告和可视化
        if comparison_results or generalization_results:
            print(f"\n📝 第三阶段: 生成综合报告")

            # 生成可视化
            if comparison_results:
                evaluator.create_comparison_visualization(comparison_results['results'])

            if generalization_results:
                evaluator.create_generalization_visualization(
                    generalization_results['results'],
                    generalization_results['train_network']
                )

            # 生成综合报告
            report = evaluator.generate_evaluation_report(
                comparison_results=comparison_results,
                generalization_results=generalization_results
            )

            print(f"\n🎉 完整评估完成！")
            print(f"📁 结果保存在: evaluation_results/ 目录")

            return True
        else:
            print(f"❌ 所有测试都失败了")
            return False

    except Exception as e:
        print(f"❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_available_modes():
    """打印可用的评估模式"""
    print("📋 可用的评估模式:")
    
    modes = {
        'quick': {
            'description': '快速评估模式',
            'baseline_comparison': {
                'enabled': True,
                'test_networks': ['ieee14', 'ieee30'],
                'scenarios': ['normal', 'high_load'],
                'runs_per_test': 5
            },
            'generalization_test': {
                'enabled': True,
                'test_networks': ['ieee30'],
                'runs_per_network': 10
            }
        },
        'standard': {
            'description': '标准评估模式',
            'baseline_comparison': {
                'enabled': True,
                'test_networks': ['ieee14', 'ieee30', 'ieee57'],
                'scenarios': ['normal', 'high_load', 'unbalanced'],
                'runs_per_test': 10
            },
            'generalization_test': {
                'enabled': True,
                'test_networks': ['ieee30', 'ieee57'],
                'runs_per_network': 20
            }
        },
        'comprehensive': {
            'description': '全面评估模式',
            'baseline_comparison': {
                'enabled': True,
                'test_networks': ['ieee14', 'ieee30', 'ieee57'],
                'scenarios': ['normal', 'high_load', 'unbalanced', 'fault'],
                'runs_per_test': 15
            },
            'generalization_test': {
                'enabled': True,
                'test_networks': ['ieee30', 'ieee57', 'ieee118'],
                'runs_per_network': 30
            }
        }
    }
    
    for mode_name, mode_config in modes.items():
        print(f"   {mode_name}: {mode_config['description']}")

        # Baseline对比配置
        baseline_config = mode_config.get('baseline_comparison', {})
        if baseline_config.get('enabled', False):
            networks = baseline_config.get('test_networks', [])
            scenarios = baseline_config.get('scenarios', [])
            runs = baseline_config.get('runs_per_test', 0)
            print(f"     - Baseline对比: {len(networks)}个网络, {len(scenarios)}种场景, 每个{runs}次运行")

        # 泛化测试配置
        gen_config = mode_config.get('generalization_test', {})
        if gen_config.get('enabled', False):
            test_networks = gen_config.get('test_networks', [])
            runs = gen_config.get('runs_per_network', 0)
            print(f"     - 泛化测试: {len(test_networks)}个测试网络, 每个{runs}次运行")

        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='电力网络分区智能体综合评估系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python test.py --baseline --network ieee30 --model data/checkpoints/models/agent_ieee57_adaptive_best.pth
  python test.py --generalization --train-network ieee14 --model data/checkpoints/models/agent_ieee57_adaptive_best.pth
  python test.py --comprehensive --mode standard --model data/checkpoints/models/agent_ieee57_adaptive_best.pth
  python test.py --list-modes                                  # 查看可用模式
        """
    )

    # 测试类型选择
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument('--baseline', action='store_true', help='运行Baseline对比测试')
    test_group.add_argument('--generalization', action='store_true', help='运行泛化能力测试')
    test_group.add_argument('--comprehensive', action='store_true', help='运行完整评估')
    test_group.add_argument('--list-modes', action='store_true', help='列出可用的评估模式')

    # 配置参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['quick', 'standard', 'comprehensive'],
                       help='评估模式 (默认: standard)')
    parser.add_argument('--model', type=str, help='预训练模型路径 (除--list-modes外必需)')

    # Baseline测试参数
    parser.add_argument('--network', type=str, default='ieee30',
                       choices=['ieee14', 'ieee30', 'ieee57', 'ieee118'],
                       help='Baseline测试网络 (默认: ieee30)')

    # 泛化测试参数
    parser.add_argument('--train-network', type=str, default='ieee14',
                       choices=['ieee14', 'ieee30', 'ieee57'],
                       help='泛化测试的训练网络 (默认: ieee14)')

    args = parser.parse_args()

    try:
        print("🎯 电力网络分区智能体综合评估系统")
        print("=" * 60)
        print(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        if args.list_modes:
            print_available_modes()
            return 0

        # 检查模型参数
        if not args.model:
            print("❌ 错误：必须提供预训练模型路径")
            print("💡 使用方法: python test.py <测试类型> --model <模型路径>")
            print("📋 查看帮助: python test.py --help")
            return 1

        success = False

        if args.baseline:
            success = run_baseline_comparison(args.config, args.network, args.mode, args.model)

        elif args.generalization:
            success = run_generalization_test(args.config, args.train_network, args.mode, args.model)

        elif args.comprehensive:
            success = run_comprehensive_evaluation(args.config, args.mode, args.model)

        if success:
            print(f"\n🎉 评估完成！")
            print(f"📁 详细结果请查看: evaluation_results/ 目录")
            print(f"💡 提示: 可以使用不同的 --mode 参数来调整评估详细程度")
            return 0
        else:
            print(f"\n❌ 评估失败")
            return 1

    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断评估")
        return 1
    except Exception as e:
        print(f"\n❌ 评估系统发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
