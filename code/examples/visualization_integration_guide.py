#!/usr/bin/env python3
"""
增强测试可视化使用示例
展示如何在test.py中集成新的可视化功能
"""

import numpy as np
import sys
from pathlib import Path

# 添加必要的路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# 导入增强可视化模块
from enhanced_test_visualizer import enhance_test_visualization


def integrate_enhanced_visualization_in_test():
    """
    示例：如何在test.py中集成增强可视化
    
    在test.py的适当位置添加以下代码
    """
    
    # 在test.py中的ComprehensiveAgentEvaluator类中添加方法
    example_code = '''
    def run_evaluation_with_enhanced_viz(self, model_path: str, network: str = 'ieee57'):
        """运行评估并生成增强可视化"""
        
        # 1. 运行标准评估
        print("🔬 运行模型评估...")
        test_results = self._run_standard_evaluation(model_path, network)
        
        # 2. 收集额外的可视化数据
        print("📊 收集可视化数据...")
        
        # 获取最佳分区结果
        best_partition, best_metrics = self._get_best_partition_result()
        test_results['best_partition'] = best_partition
        
        # 获取节点嵌入（从GAT编码器）
        if hasattr(self, 'encoder') and self.encoder is not None:
            with torch.no_grad():
                node_embeddings, attention_weights = self.encoder.encode_nodes_with_attention(
                    self.hetero_data, self.config
                )
                test_results['node_embeddings'] = node_embeddings['bus']
                test_results['attention_weights'] = attention_weights
        
        # 获取环境数据
        test_results['env_data'] = {
            'bus_data': self._extract_bus_data(),
            'branch_data': self._extract_branch_data()
        }
        
        # 收集算法对比数据
        test_results['algorithm_comparison'] = self._run_baseline_comparisons()
        
        # 收集场景性能数据
        test_results['scenario_performance'] = self._evaluate_scenarios()
        
        # 3. 调用增强可视化
        print("🎨 生成增强可视化...")
        
        # 模型信息
        model_info = {
            'network_type': network.upper(),
            'model_path': model_path,
            'training_episodes': self._get_model_training_info(model_path).get('episodes', 0),
            'best_reward': self._get_model_training_info(model_path).get('best_reward', 0)
        }
        
        # 配置可视化参数
        viz_config = {
            'output_dir': f'evaluation/{network}_enhanced_viz',
            'advanced': {
                'enable_3d': True,
                'enable_interactive': True,
                'enable_animation': True
            },
            'chart_factory': {
                'color_scheme': 'professional'
            }
        }
        
        # 生成可视化
        from enhanced_test_visualizer import enhance_test_visualization
        visualization_files = enhance_test_visualization(
            test_results, 
            model_info, 
            viz_config
        )
        
        # 4. 打印结果摘要
        print("\\n📈 评估结果摘要:")
        print(f"   平均奖励: {test_results.get('avg_reward', 0):.4f}")
        print(f"   成功率: {test_results.get('success_rate', 0)*100:.1f}%")
        print(f"   最佳CV值: {test_results.get('best_cv', 0):.4f}")
        print(f"   平均计算时间: {np.mean(test_results.get('computation_times', [0])):.2f} ms")
        
        print("\\n🎨 生成的可视化:")
        for viz_name, viz_path in visualization_files.items():
            print(f"   {viz_name}: {viz_path}")
        
        return test_results, visualization_files
    '''
    
    print("=== 在test.py中添加以下代码 ===")
    print(example_code)
    print("================================")


def add_cli_arguments_for_visualization():
    """
    示例：为test.py添加可视化相关的命令行参数
    """
    
    cli_code = '''
    # 在test.py的create_parser()函数中添加
    
    # 可视化选项
    parser.add_argument('--enhanced-viz', action='store_true',
                      help='启用增强可视化功能')
    
    parser.add_argument('--viz-3d', action='store_true',
                      help='生成3D网络可视化')
    
    parser.add_argument('--viz-interactive', action='store_true',
                      help='生成交互式网络图')
    
    parser.add_argument('--viz-attention', action='store_true',
                      help='生成GAT注意力热图')
    
    parser.add_argument('--viz-live-monitor', action='store_true',
                      help='启动实时监控仪表板')
    
    parser.add_argument('--viz-port', type=int, default=8050,
                      help='实时监控仪表板端口 (默认: 8050)')
    
    parser.add_argument('--viz-output-dir', type=str, 
                      default='evaluation/enhanced_viz',
                      help='可视化输出目录')
    '''
    
    print("\n=== 添加命令行参数 ===")
    print(cli_code)
    print("====================")


def example_main_integration():
    """
    示例：在test.py的main函数中集成
    """
    
    main_code = '''
    def main():
        parser = create_parser()
        args = parser.parse_args()
        
        # ... 现有代码 ...
        
        # 增强可视化分支
        if args.enhanced_viz:
            print("\\n🎨 使用增强可视化模式")
            
            # 创建评估器
            evaluator = ComprehensiveAgentEvaluator()
            
            # 运行带增强可视化的评估
            test_results, viz_files = evaluator.run_evaluation_with_enhanced_viz(
                model_path=args.model,
                network=args.network
            )
            
            # 如果启用实时监控
            if args.viz_live_monitor:
                from enhanced_test_visualizer import EnhancedTestVisualizer
                visualizer = EnhancedTestVisualizer()
                dashboard_url = visualizer.create_live_monitoring_dashboard(
                    port=args.viz_port
                )
                print(f"\\n🌐 实时监控仪表板: {dashboard_url}")
                print("   按Ctrl+C停止服务器")
                
                # 保持服务器运行
                try:
                    import time
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\\n停止监控服务器")
            
            return
        
        # ... 原有的测试逻辑 ...
    '''
    
    print("\n=== 在main函数中集成 ===")
    print(main_code)
    print("=====================")


def show_usage_examples():
    """展示使用示例"""
    
    print("\n📚 增强可视化使用示例")
    print("=" * 50)
    
    print("\n1️⃣ 基础增强可视化:")
    print("   python test.py --enhanced-viz --model path/to/model.pth --network ieee57")
    
    print("\n2️⃣ 生成所有可视化类型:")
    print("   python test.py --enhanced-viz --viz-3d --viz-interactive --viz-attention \\")
    print("                  --model path/to/model.pth --network ieee57")
    
    print("\n3️⃣ 启动实时监控:")
    print("   python test.py --enhanced-viz --viz-live-monitor --viz-port 8080 \\")
    print("                  --model path/to/model.pth --network ieee57")
    
    print("\n4️⃣ 自定义输出目录:")
    print("   python test.py --enhanced-viz --viz-output-dir my_results/viz \\")
    print("                  --model path/to/model.pth --network ieee57")
    
    print("\n5️⃣ 批量测试多个网络:")
    print("   for network in ieee14 ieee30 ieee57 ieee118; do")
    print("       python test.py --enhanced-viz --model model.pth --network $network")
    print("   done")


def create_visualization_config_template():
    """创建可视化配置模板"""
    
    config = {
        "visualization": {
            "enabled": True,
            "output_formats": ["html", "png", "svg"],
            "themes": {
                "color_scheme": "professional",
                "font_family": "Arial, sans-serif",
                "background_color": "#ffffff"
            },
            "network_graph": {
                "layout": "force_directed",
                "node_size_factor": 1.0,
                "edge_width_factor": 1.0,
                "show_labels": True,
                "physics_enabled": True
            },
            "3d_visualization": {
                "enabled": True,
                "camera_angle": {
                    "eye": {"x": 1.5, "y": 1.5, "z": 1.5}
                },
                "use_embeddings": True
            },
            "attention_heatmap": {
                "colorscale": "Viridis",
                "show_values": False,
                "aggregate_heads": True
            },
            "performance_charts": {
                "show_confidence_intervals": True,
                "smoothing_window": 10,
                "comparison_baseline": "spectral"
            },
            "animation": {
                "fps": 30,
                "duration_per_frame": 500,
                "transition_duration": 300
            },
            "dashboard": {
                "auto_refresh": True,
                "refresh_interval": 2000,
                "show_metrics": [
                    "reward", "cv", "coupling", "success_rate",
                    "computation_time", "episode_length"
                ]
            }
        }
    }
    
    print("\n=== 可视化配置模板 (viz_config.yaml) ===")
    import yaml
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("======================================")


if __name__ == "__main__":
    print("🎨 增强测试可视化集成指南")
    print("=" * 60)
    
    # 展示集成步骤
    print("\n📋 集成步骤:")
    print("1. 安装新的可视化依赖")
    print("2. 在test.py中添加增强可视化方法")
    print("3. 添加命令行参数")
    print("4. 在main函数中集成")
    print("5. 运行测试")
    
    # 展示代码示例
    integrate_enhanced_visualization_in_test()
    add_cli_arguments_for_visualization()
    example_main_integration()
    
    # 展示使用示例
    show_usage_examples()
    
    # 创建配置模板
    create_visualization_config_template()
    
    print("\n✅ 集成完成！现在可以使用增强的可视化功能了。")