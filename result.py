#!/usr/bin/env python3
"""
电力网络分区强化学习系统 - 结果分析主程序
替代test.py，提供更简洁、更强大的测试和可视化功能
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'code' / 'src'))
sys.path.append(str(Path(__file__).parent / 'code'))

# 导入必要的模块
from test_agent.evaluators.comprehensive_evaluator import ComprehensiveEvaluator
from test_agent.visualizers.partition_comparator import PartitionComparator
from test_agent.utils.helpers import setup_output_directory


class ResultAnalyzer:
    """结果分析主类"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """初始化结果分析器"""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        # 创建时间戳输出目录
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.output_dir = Path('evaluation') / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_dir / 'comparisons').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
    def run_partition_comparison(self, 
                               model_path: str,
                               network: str = 'ieee57',
                               num_runs: int = 10) -> Dict[str, Any]:
        """
        运行分区对比分析
        为每个场景生成agent最佳性能的对比图
        """
        print("\n" + "="*60)
        print("🔬 分区对比分析")
        print("="*60)
        
        # 创建评估器（传入输出目录）
        evaluator = ComprehensiveEvaluator(self.config, self.output_dir)
        
        # 运行评估
        results = evaluator.evaluate_all_scenarios(
            model_path=model_path,
            network_name=network,
            num_runs=num_runs
        )
        
        return results
    
    def run_enhanced_visualization(self,
                                 model_path: str,
                                 network: str = 'ieee57',
                                 enable_3d: bool = True,
                                 enable_attention: bool = True) -> Dict[str, str]:
        """
        运行增强可视化分析
        """
        print("\n" + "="*60)
        print("🎨 增强可视化分析")
        print("="*60)
        
        # 准备测试数据
        test_results = self._run_basic_test(model_path, network)
        
        # 模型信息
        checkpoint = torch.load(model_path, map_location=self.device)
        model_info = {
            'network_type': network.upper(),
            'training_episodes': checkpoint.get('training_stats', {}).get('total_episodes', 0),
            'best_reward': checkpoint.get('best_reward', 0)
        }
        
        # 配置可视化
        viz_config = {
            'output_dir': str(self.output_dir / 'visualizations'),
            'advanced': {
                'enable_3d': enable_3d,
                'enable_interactive': True,
                'enable_attention': enable_attention
            }
        }
        
        # 创建可视化器
        visualizer = EnhancedTestVisualizer(viz_config)
        
        # 生成可视化
        viz_files = visualizer.visualize_test_results(
            test_results=test_results,
            model_info=model_info,
            env_data=test_results.get('env_data')
        )
        
        print(f"\n✅ 生成了 {len(viz_files)} 个可视化文件")
        
        return viz_files
    
    def run_comprehensive_analysis(self,
                                 model_path: str,
                                 network: str = 'ieee57',
                                 num_runs: int = 10) -> Dict[str, Any]:
        """
        运行完整的综合分析
        包括分区对比和所有增强可视化
        """
        print("\n" + "="*60)
        print("🚀 综合分析模式")
        print("="*60)
        
        all_results = {}
        
        # 1. 分区对比分析
        print("\n[1/3] 分区对比分析...")
        partition_results = self.run_partition_comparison(model_path, network, num_runs)
        all_results['partition_comparison'] = partition_results
        
        # 2. 增强可视化
        print("\n[2/3] 增强可视化分析...")
        viz_files = self.run_enhanced_visualization(model_path, network)
        all_results['visualizations'] = viz_files
        
        # 3. 生成综合报告
        print("\n[3/3] 生成综合报告...")
        report_path = self._generate_comprehensive_report(all_results, model_path, network)
        all_results['report'] = report_path
        
        print("\n" + "="*60)
        print("✅ 综合分析完成！")
        print(f"📄 报告路径: {report_path}")
        print("="*60)
        
        return all_results
    
    def _run_basic_test(self, model_path: str, network: str) -> Dict[str, Any]:
        """运行基础测试获取数据"""
        from train import load_power_grid_data, create_environment_from_config
        from code.src.data_processing import PowerGridDataProcessor
        from code.src.gat import create_production_encoder
        from code.src.rl.agent import PPOAgent
        
        # 加载数据
        case_data = load_power_grid_data(network)
        
        # 创建环境
        env = create_environment_from_config(case_data, self.config)
        
        # 创建数据处理器
        processor = PowerGridDataProcessor(
            normalize=self.config['data']['normalize'],
            cache_dir=self.config['data']['cache_dir']
        )
        
        # 处理数据
        hetero_data = processor.process_power_grid_data(
            case_data, 
            self.config['data']['case_name']
        )
        
        # 创建编码器
        encoder = create_production_encoder(
            hetero_data,
            self.config['gat'],
            self.device
        )
        
        # 获取嵌入和注意力权重
        with torch.no_grad():
            node_embeddings, attention_weights = encoder.encode_nodes_with_attention(
                hetero_data, self.config
            )
        
        # 准备环境数据
        env_data = {
            'bus_data': [
                {
                    'type': 'load' if bus[2] < 0 else 'generator' if i in case_data.get('gen', {}).get('bus', []) else 'bus',
                    'p': float(bus[2]),
                    'q': float(bus[3]),
                    'v': float(bus[7]) if len(bus) > 7 else 1.0
                }
                for i, bus in enumerate(case_data['bus'])
            ],
            'branch_data': [
                {
                    'from': int(branch[0]) - 1,
                    'to': int(branch[1]) - 1,
                    'r': float(branch[2]),
                    'x': float(branch[3]),
                    'rating': float(branch[5]) if len(branch) > 5 else 100.0
                }
                for branch in case_data['branch']
                if branch[10] == 1  # 只包含运行中的线路
            ]
        }
        
        # 运行一次测试获取分区结果
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建agent
        agent = PPOAgent(
            node_embedding_dim=checkpoint['model_info']['node_embedding_dim'],
            region_embedding_dim=checkpoint['model_info']['region_embedding_dim'],
            num_partitions=checkpoint['model_info']['num_partitions'],
            device=self.device
        )
        
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.eval()
        
        # 运行一个episode
        state = env.reset()
        done = False
        rewards = []
        
        while not done:
            with torch.no_grad():
                action, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            state = next_state
        
        # 收集结果
        test_results = {
            'num_episodes': 1,
            'avg_reward': sum(rewards),
            'episode_rewards': rewards,
            'best_partition': env.get_current_partition(),
            'node_embeddings': node_embeddings['bus'] if 'bus' in node_embeddings else None,
            'attention_weights': attention_weights,
            'env_data': env_data,
            'computation_times': [10.0] * len(rewards),  # 示例数据
            'success_rate': 1.0 if info.get('success', False) else 0.0,
            'best_cv': info.get('cv', 0.5)
        }
        
        return test_results
    
    def _generate_comprehensive_report(self, 
                                     results: Dict[str, Any],
                                     model_path: str,
                                     network: str) -> str:
        """生成综合报告"""
        report = {
            'model': model_path,
            'network': network,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'summary': {
                'partition_comparison_scenarios': len(results.get('partition_comparison', {})),
                'visualizations_generated': len(results.get('visualizations', {})),
            },
            'details': results
        }
        
        # 保存JSON报告
        report_path = self.output_dir / 'reports' / 'comprehensive_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 生成HTML报告
        html_report = self._create_html_report(results, model_path, network)
        html_path = self.output_dir / 'reports' / 'comprehensive_report.html'
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        return str(html_path)
    
    def _create_html_report(self, results: Dict[str, Any], model_path: str, network: str) -> str:
        """创建HTML报告"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>电力网络分区 - 综合分析报告</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E86AB;
            text-align: center;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 20px;
        }}
        h2 {{
            color: #A23B72;
            margin-top: 30px;
        }}
        .info-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #2E86AB;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 15px;
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔋 电力网络分区 - 综合分析报告</h1>
        
        <div class="info-box">
            <h3>测试信息</h3>
            <p><strong>模型:</strong> {Path(model_path).name}</p>
            <p><strong>网络:</strong> {network.upper()}</p>
            <p><strong>时间:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>📊 分区对比结果</h2>
        <p>以下展示了不同场景下Agent（最佳性能）与Baseline的分区对比：</p>
        
        <div class="grid">
"""
        
        # 添加分区对比图片
        scenarios = ['normal', 'n-1', 'load_fluctuation', 'both']
        for scenario in scenarios:
            img_path = f"partition_comparison_{scenario.replace('_', ' ').lower()}.png"
            html += f"""
            <div class="card">
                <h3>{scenario.replace('_', ' ').title()}</h3>
                <img src="{img_path}" alt="{scenario} comparison">
            </div>
"""
        
        html += """
        </div>
        
        <h2>🎨 增强可视化</h2>
        <p>高级分析和可视化结果：</p>
        
        <div class="grid">
"""
        
        # 添加增强可视化链接
        viz_types = {
            'interactive_network': '交互式网络图',
            '3d_network': '3D网络可视化',
            'attention_heatmap': '注意力热图',
            'performance_radar': '性能雷达图'
        }
        
        for viz_key, viz_name in viz_types.items():
            if viz_key in results.get('visualizations', {}):
                html += f"""
            <div class="card">
                <h3>{viz_name}</h3>
                <p><a href="{Path(results['visualizations'][viz_key]).name}" target="_blank">查看 {viz_name} →</a></p>
            </div>
"""
        
        html += f"""
        </div>
        
        <div class="timestamp">
            报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        return html


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='电力网络分区强化学习 - 结果分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行分区对比分析
  python result.py --partition-comparison --model path/to/model.pth
  
  # 运行增强可视化
  python result.py --enhanced-viz --model path/to/model.pth
  
  # 运行综合分析（推荐）
  python result.py --comprehensive --model path/to/model.pth
  
  # 指定网络和运行次数
  python result.py --comprehensive --model model.pth --network ieee118 --runs 20
        """
    )
    
    # 分析模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--partition-comparison', action='store_true',
                           help='运行分区对比分析（生成baseline vs agent对比图）')
    mode_group.add_argument('--enhanced-viz', action='store_true',
                           help='运行增强可视化分析')
    mode_group.add_argument('--comprehensive', action='store_true',
                           help='运行综合分析（包含所有功能）')
    
    # 必需参数
    parser.add_argument('--model', type=str, required=True,
                       help='训练好的模型路径')
    
    # 可选参数
    parser.add_argument('--network', type=str, default='ieee57',
                       choices=['ieee14', 'ieee30', 'ieee57', 'ieee118'],
                       help='测试网络 (默认: ieee57)')
    parser.add_argument('--runs', type=int, default=10,
                       help='每个场景的运行次数 (默认: 10)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    
    # 可视化选项
    parser.add_argument('--no-3d', action='store_true',
                       help='禁用3D可视化')
    parser.add_argument('--no-attention', action='store_true',
                       help='禁用注意力热图')
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 打印启动信息
    print("\n" + "="*60)
    print("🔋 电力网络分区强化学习 - 结果分析系统")
    print("="*60)
    
    # 检查模型文件
    if not Path(args.model).exists():
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 创建分析器
    analyzer = ResultAnalyzer(args.config)
    
    # 根据模式运行
    try:
        if args.partition_comparison:
            results = analyzer.run_partition_comparison(
                model_path=args.model,
                network=args.network,
                num_runs=args.runs
            )
            
        elif args.enhanced_viz:
            results = analyzer.run_enhanced_visualization(
                model_path=args.model,
                network=args.network,
                enable_3d=not args.no_3d,
                enable_attention=not args.no_attention
            )
            
        elif args.comprehensive:
            results = analyzer.run_comprehensive_analysis(
                model_path=args.model,
                network=args.network,
                num_runs=args.runs
            )
        
        print("\n✅ 分析完成！")
        print(f"📁 结果保存在: {analyzer.output_dir}/")
        print(f"   - 分区对比图: comparisons/")
        print(f"   - 增强可视化: visualizations/")
        print(f"   - 分析报告: reports/")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()