#!/usr/bin/env python3
"""
统一的电力网络分区测试系统
集成原test.py的功能 + 新的可视化分析功能
"""

import argparse
import sys
import os
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'code' / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'code'))
sys.path.append(str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='电力网络分区智能体统一测试系统')
    
    # 主要功能选择
    parser.add_argument('--baseline', action='store_true',
                       help='运行baseline对比测试 (原test.py功能)')
    parser.add_argument('--generalization', action='store_true', 
                       help='运行泛化能力测试 (原test.py功能)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='运行完整评估 (原test.py功能)')
    parser.add_argument('--partition-comparison', action='store_true',
                       help='运行分区对比分析 (新可视化功能)')
    parser.add_argument('--enhanced-visualization', action='store_true',
                       help='运行增强可视化分析 (新功能)')
    
    # 通用参数
    parser.add_argument('--model', type=str, required=True,
                       help='预训练模型路径')
    parser.add_argument('--network', type=str, default='ieee57',
                       help='测试网络 (默认: ieee57)')
    parser.add_argument('--runs', type=int, default=10,
                       help='每个场景的运行次数 (默认: 10)')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式')
    
    # 可视化相关参数
    parser.add_argument('--save-dir', type=str, default='evaluation',
                       help='结果保存目录 (默认: evaluation)')
    parser.add_argument('--show-plots', action='store_true',
                       help='显示生成的图表')
    
    args = parser.parse_args()
    
    # 验证参数
    if not any([args.baseline, args.generalization, args.comprehensive, 
               args.partition_comparison, args.enhanced_visualization]):
        print("❌ 错误：必须选择至少一个测试模式")
        parser.print_help()
        return
    
    print("🎯 电力网络分区智能体统一测试系统")
    print("=" * 60)
    
    # 原test.py功能
    if args.baseline or args.generalization or args.comprehensive:
        print("🔍 运行原test.py功能...")
        from test import main as legacy_test_main
        
        # 构造legacy test的参数
        legacy_args = []
        if args.baseline:
            legacy_args.extend(['--baseline'])
        if args.generalization:
            legacy_args.extend(['--generalization'])
        if args.comprehensive:
            legacy_args.extend(['--comprehensive'])
        
        legacy_args.extend(['--model', args.model])
        if args.network != 'ieee57':
            legacy_args.extend(['--network', args.network])
        if args.runs != 10:
            legacy_args.extend(['--runs', str(args.runs)])
        if args.quick:
            legacy_args.extend(['--quick'])
        
        # 临时修改sys.argv来调用legacy test
        original_argv = sys.argv.copy()
        sys.argv = ['test.py'] + legacy_args
        
        try:
            legacy_test_main()
        except SystemExit:
            pass  # 忽略SystemExit
        finally:
            sys.argv = original_argv
    
    # 新的可视化功能
    if args.partition_comparison or args.enhanced_visualization:
        print("\n🎨 运行新的可视化分析功能...")
        
        # 导入新的分析器
        sys.path.append(str(Path(__file__).parent.parent))
        from result import ResultAnalyzer
        
        analyzer = ResultAnalyzer()
        
        if args.partition_comparison:
            print("🔬 分区对比分析...")
            results = analyzer.run_partition_comparison(
                model_path=args.model,
                network_name=args.network,
                num_runs=args.runs,
                output_dir=args.save_dir
            )
            print(f"✅ 分区对比完成！结果保存在: {results.get('output_dir', args.save_dir)}")
        
        if args.enhanced_visualization:
            print("📊 增强可视化分析...")
            results = analyzer.run_enhanced_visualization(
                model_path=args.model,
                network_name=args.network,
                num_runs=args.runs,
                output_dir=args.save_dir
            )
            print(f"✅ 增强可视化完成！结果保存在: {results.get('output_dir', args.save_dir)}")
    
    print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    main()