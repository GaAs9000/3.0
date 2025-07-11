#!/usr/bin/env python3
"""
电力网络分区智能体现代化测试系统 - 主入口

基于pytest的现代化测试框架，支持：
- 单元测试 (Unit Tests)
- 集成测试 (Integration Tests)
- 性能测试 (Performance Tests)
- 端到端测试 (E2E Tests)
- 测试覆盖率报告
- CI/CD集成
- 自动化报告生成
- 可视化分析
"""

import argparse
import sys
import time
import subprocess
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any

# 添加路径
sys.path.append(str(Path(__file__).parent / 'code' / 'src'))
sys.path.append(str(Path(__file__).parent / 'code'))
sys.path.append(str(Path(__file__).parent / 'code' / 'test'))

try:
    from comprehensive_evaluator import ComprehensiveAgentEvaluator
except ImportError:
    print("⚠️ 警告: 无法导入comprehensive_evaluator，某些功能可能不可用")
    ComprehensiveAgentEvaluator = None


class RealTimePerformanceTester:
    """实时性能测试器 - Agent vs Baseline算法执行时间对比测试"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results_dir = self.project_root / 'performance_results'
        self.results_dir.mkdir(exist_ok=True)

        # 添加路径以导入baseline模块
        sys.path.append(str(self.project_root / 'code'))
        sys.path.append(str(self.project_root / 'code' / 'baseline'))

    def run_engineering_performance_test(self, model_path: str, network: str = 'ieee57', num_runs: int = 100) -> Dict[str, Any]:
        """运行工程级性能测试 - 包含Agent vs Baseline性能对比"""
        print(f"🏭 工程级Agent vs Baseline实时性能对比测试")
        print(f"   网络: {network.upper()}")
        print(f"   模型: {model_path}")
        print(f"   运行次数: {num_runs}")
        print("=" * 60)

        try:
            # 1. 运行Agent性能测试
            print("🤖 第一阶段: Agent性能测试")
            agent_results = self._run_agent_performance_test(model_path, network, num_runs)

            if not agent_results['success']:
                return agent_results

            # 2. 运行Baseline算法性能测试
            print("\n📊 第二阶段: Baseline算法性能测试")
            baseline_results = self._run_baseline_performance_test(network, num_runs)

            # 3. 生成对比报告
            print("\n📈 第三阶段: 生成性能对比报告")
            comparison_results = self._generate_performance_comparison(
                agent_results, baseline_results, model_path, network, num_runs
            )

            # 4. 保存结果
            self._save_comparison_results(comparison_results, network)

            return comparison_results

        except Exception as e:
            print(f"❌ 性能对比测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _run_agent_performance_test(self, model_path: str, network: str, num_runs: int) -> Dict[str, Any]:
        """运行Agent性能测试 - 直接在内部实现避免外部脚本调用问题"""
        try:
            print("🚀 启动Agent性能测试...")

            # 导入必要的模块
            from train import load_power_grid_data, create_environment_from_config
            from data_processing import PowerGridDataProcessor
            from gat import create_production_encoder
            from rl.agent import PPOAgent
            import yaml
            import psutil
            import os

            # 加载配置
            config_path = self.project_root / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                config = {
                    'system': {'device': 'cpu', 'seed': 42},
                    'data': {'normalize': True},
                    'environment': {
                        'num_partitions': 4,
                        'max_steps': 200,
                        'reward_weights': {'balance': 1.0, 'connectivity': 0.5, 'size': 0.3}
                    },
                    'gat': {'hidden_dim': 64, 'num_heads': 4, 'num_layers': 2},
                    'agent': {'learning_rate': 3e-4, 'gamma': 0.99}
                }

            print("   初始化Agent测试环境...")

            # 加载网络数据并创建环境
            mpc_data = load_power_grid_data(network)
            processor = PowerGridDataProcessor()
            hetero_data = processor.graph_from_mpc(mpc_data, config=config)
            encoder = create_production_encoder(hetero_data, config.get('gat', {}))
            hetero_data = hetero_data.to(torch.device('cpu'))

            with torch.no_grad():
                node_embeddings_dict, attention_weights = encoder.encode_nodes_with_attention(hetero_data, config)

            env, _ = create_environment_from_config(
                config, hetero_data, node_embeddings_dict, attention_weights, mpc_data,
                torch.device('cpu'), is_normalized=config['data'].get('normalize', True)
            )

            # 获取环境信息
            init_state, init_info = env.reset()
            node_dim = init_state['node_embeddings'].shape[1]
            region_dim = init_state['region_embeddings'].shape[1]
            num_partitions = init_state['region_embeddings'].shape[0]
            num_nodes = init_state['node_embeddings'].shape[0]

            print(f"   环境规模: {num_nodes}节点, {num_partitions}分区")

            # 创建智能体
            agent = PPOAgent(
                node_embedding_dim=node_dim,
                region_embedding_dim=region_dim,
                num_partitions=num_partitions,
                agent_config=config.get('agent', {}),
                device=torch.device('cpu')
            )

            # 加载预训练模型
            print("   加载预训练模型...")
            checkpoint = torch.load(model_path, map_location='cpu')

            if 'actor_state_dict' not in checkpoint:
                raise ValueError(f"不支持的模型格式: {model_path}")

            # 加载模型状态
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.actor.eval()
            agent.critic.eval()

            print("   开始Agent性能测试...")

            # 系统预热
            for i in range(10):
                state, _ = env.reset()
                action, _, _ = agent.select_action(state, training=False)
                if action is not None:
                    env.step(action)

            # 性能测试
            agent_times = []
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            successful_runs = 0
            for i in range(num_runs):
                try:
                    state, _ = env.reset()

                    # 测量Agent决策时间
                    start_time = time.perf_counter()
                    action, _, _ = agent.select_action(state, training=False)
                    end_time = time.perf_counter()

                    if action is not None:
                        execution_time = (end_time - start_time) * 1000  # 毫秒
                        agent_times.append(execution_time)
                        successful_runs += 1

                except Exception as e:
                    print(f"       第{i+1}次运行失败: {e}")
                    continue

            if not agent_times:
                return {'success': False, 'error': 'Agent测试全部失败'}

            # 统计分析
            agent_times = np.array(agent_times)
            current_memory = process.memory_info().rss / 1024 / 1024

            results = {
                'success': True,
                'data': {
                    'test_info': {
                        'network': network,
                        'model_path': model_path,
                        'num_runs': num_runs,
                        'successful_runs': successful_runs,
                        'num_nodes': num_nodes,
                        'num_partitions': num_partitions,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'performance_metrics': {
                        'agent_decision': {
                            'mean_ms': float(np.mean(agent_times)),
                            'std_ms': float(np.std(agent_times)),
                            'min_ms': float(np.min(agent_times)),
                            'max_ms': float(np.max(agent_times)),
                            'p50_ms': float(np.percentile(agent_times, 50)),
                            'p95_ms': float(np.percentile(agent_times, 95)),
                            'p99_ms': float(np.percentile(agent_times, 99)),
                            'p999_ms': float(np.percentile(agent_times, 99.9))
                        },
                        'memory': {
                            'increase_mb': float(current_memory - initial_memory)
                        }
                    }
                }
            }

            print(f"   ✅ Agent测试完成: 平均 {np.mean(agent_times):.2f}ms")
            return results

        except Exception as e:
            print(f"❌ Agent性能测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _get_latest_performance_result(self, network: str) -> Dict[str, Any]:
        """获取最新的性能测试结果"""
        try:
            import glob
            import json

            # 查找最新的结果文件
            pattern = str(self.results_dir / f"engineering_benchmark_{network}_*.json")
            files = glob.glob(pattern)

            if not files:
                return None

            # 获取最新文件
            latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)

            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            print(f"⚠️ 无法读取Agent性能结果: {e}")
            return None



    def _run_baseline_performance_test(self, network: str, num_runs: int) -> Dict[str, Any]:
        """运行Baseline算法性能测试"""
        try:
            # 导入必要的模块
            from train import load_power_grid_data, create_environment_from_config
            from data_processing import PowerGridDataProcessor
            from gat import create_production_encoder
            from baseline import SpectralPartitioner, KMeansPartitioner
            import yaml
            import time
            import psutil
            import os

            print("   初始化测试环境...")

            # 加载配置
            config_path = self.project_root / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                config = {
                    'system': {'device': 'cpu', 'seed': 42},
                    'data': {'normalize': True},
                    'environment': {
                        'num_partitions': 4,
                        'max_steps': 200,
                        'reward_weights': {'balance': 1.0, 'connectivity': 0.5, 'size': 0.3}
                    },
                    'gat': {'hidden_dim': 64, 'num_heads': 4, 'num_layers': 2}
                }

            # 加载网络数据并创建环境
            mpc_data = load_power_grid_data(network)
            processor = PowerGridDataProcessor()
            hetero_data = processor.graph_from_mpc(mpc_data, config=config)
            encoder = create_production_encoder(hetero_data, config.get('gat', {}))
            hetero_data = hetero_data.to(torch.device('cpu'))

            with torch.no_grad():
                node_embeddings_dict, attention_weights = encoder.encode_nodes_with_attention(hetero_data, config)

            env, _ = create_environment_from_config(
                config, hetero_data, node_embeddings_dict, attention_weights, mpc_data,
                torch.device('cpu'), is_normalized=config['data'].get('normalize', True)
            )

            print("   开始Baseline算法性能测试...")

            # 测试不同的baseline算法
            baseline_methods = {
                'Spectral Clustering': SpectralPartitioner(seed=42),
                'K-means Clustering': KMeansPartitioner(seed=42)
            }

            baseline_results = {}
            process = psutil.Process(os.getpid())

            for method_name, partitioner in baseline_methods.items():
                print(f"     测试 {method_name}...")

                method_times = []
                memory_usage = []
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB

                successful_runs = 0
                for i in range(num_runs):
                    try:
                        # 重置环境
                        env.reset()

                        # 测量算法执行时间
                        start_time = time.perf_counter()
                        partition_result = partitioner.partition(env)
                        end_time = time.perf_counter()

                        # 记录时间 (毫秒)
                        execution_time = (end_time - start_time) * 1000
                        method_times.append(execution_time)
                        successful_runs += 1

                        # 每50次测量内存使用
                        if i % 50 == 0:
                            current_memory = process.memory_info().rss / 1024 / 1024
                            memory_usage.append(current_memory - initial_memory)

                    except Exception as e:
                        print(f"       第{i+1}次运行失败: {e}")
                        continue

                if method_times:
                    method_times = np.array(method_times)
                    baseline_results[method_name] = {
                        'successful_runs': successful_runs,
                        'mean_ms': float(np.mean(method_times)),
                        'std_ms': float(np.std(method_times)),
                        'min_ms': float(np.min(method_times)),
                        'max_ms': float(np.max(method_times)),
                        'p50_ms': float(np.percentile(method_times, 50)),
                        'p95_ms': float(np.percentile(method_times, 95)),
                        'p99_ms': float(np.percentile(method_times, 99)),
                        'peak_memory_mb': float(np.max(memory_usage)) if memory_usage else 0,
                        'raw_times': method_times.tolist()
                    }
                    print(f"       ✅ {method_name}: 平均 {np.mean(method_times):.2f}ms")
                else:
                    baseline_results[method_name] = {
                        'successful_runs': 0,
                        'error': 'All runs failed'
                    }
                    print(f"       ❌ {method_name}: 所有运行都失败")

            return {
                'success': True,
                'data': {
                    'network': network,
                    'num_runs': num_runs,
                    'methods': baseline_results,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }

        except Exception as e:
            print(f"   ❌ Baseline性能测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _generate_performance_comparison(self, agent_results: Dict[str, Any], baseline_results: Dict[str, Any],
                                       model_path: str, network: str, num_runs: int) -> Dict[str, Any]:
        """生成Agent vs Baseline性能对比报告"""
        try:
            print("   分析性能对比数据...")

            comparison = {
                'success': True,
                'test_info': {
                    'network': network,
                    'model_path': model_path,
                    'num_runs': num_runs,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'agent_performance': {},
                'baseline_performance': {},
                'comparison_analysis': {},
                'summary': {}
            }

            # 提取Agent性能数据
            if agent_results.get('success') and 'data' in agent_results:
                agent_data = agent_results['data']
                if 'performance_metrics' in agent_data:
                    agent_perf = agent_data['performance_metrics']['agent_decision']
                    comparison['agent_performance'] = {
                        'method': 'RL Agent (PPO)',
                        'mean_ms': agent_perf['mean_ms'],
                        'std_ms': agent_perf['std_ms'],
                        'p50_ms': agent_perf['p50_ms'],
                        'p95_ms': agent_perf['p95_ms'],
                        'p99_ms': agent_perf['p99_ms'],
                        'max_ms': agent_perf['max_ms']
                    }

            # 提取Baseline性能数据
            if baseline_results.get('success') and 'data' in baseline_results:
                comparison['baseline_performance'] = baseline_results['data']['methods']

            # 生成对比分析
            if comparison['agent_performance'] and comparison['baseline_performance']:
                comparison['comparison_analysis'] = self._analyze_performance_comparison(
                    comparison['agent_performance'], comparison['baseline_performance']
                )

                # 生成总结
                comparison['summary'] = self._generate_performance_summary(comparison['comparison_analysis'])

            # 打印对比结果
            self._print_performance_comparison(comparison)

            return comparison

        except Exception as e:
            print(f"   ❌ 性能对比分析失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_results': agent_results,
                'baseline_results': baseline_results
            }

    def _analyze_performance_comparison(self, agent_perf: Dict[str, Any], baseline_perf: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能对比数据"""
        analysis = {}

        agent_mean = agent_perf['mean_ms']
        agent_p99 = agent_perf['p99_ms']

        for method_name, method_data in baseline_perf.items():
            if 'mean_ms' not in method_data:
                continue

            baseline_mean = method_data['mean_ms']
            baseline_p99 = method_data['p99_ms']

            # 计算性能比较
            mean_speedup = baseline_mean / agent_mean if agent_mean > 0 else float('inf')
            p99_speedup = baseline_p99 / agent_p99 if agent_p99 > 0 else float('inf')

            analysis[method_name] = {
                'baseline_mean_ms': baseline_mean,
                'baseline_p99_ms': baseline_p99,
                'mean_speedup_ratio': mean_speedup,  # >1 表示Agent更快
                'p99_speedup_ratio': p99_speedup,
                'mean_performance': 'Agent更快' if mean_speedup > 1 else 'Baseline更快',
                'p99_performance': 'Agent更快' if p99_speedup > 1 else 'Baseline更快',
                'mean_improvement_pct': (mean_speedup - 1) * 100 if mean_speedup > 1 else (1 - mean_speedup) * -100,
                'p99_improvement_pct': (p99_speedup - 1) * 100 if p99_speedup > 1 else (1 - p99_speedup) * -100
            }

        return analysis

    def _generate_performance_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能总结"""
        if not analysis:
            return {'status': 'No valid comparison data'}

        # 统计Agent相对于各baseline的表现
        faster_count = sum(1 for data in analysis.values() if data['mean_speedup_ratio'] > 1)
        total_methods = len(analysis)

        avg_speedup = np.mean([data['mean_speedup_ratio'] for data in analysis.values()])
        best_speedup = max([data['mean_speedup_ratio'] for data in analysis.values()])
        worst_speedup = min([data['mean_speedup_ratio'] for data in analysis.values()])

        # 检查是否超过Spectral Clustering（关键指标）
        spectral_speedup = analysis.get('Spectral Clustering', {}).get('mean_speedup_ratio', 0)
        beats_spectral = spectral_speedup > 1

        return {
            'agent_faster_than_baselines': f"{faster_count}/{total_methods}",
            'average_speedup_ratio': avg_speedup,
            'best_speedup_ratio': best_speedup,
            'worst_speedup_ratio': worst_speedup,
            'beats_spectral_clustering': beats_spectral,
            'spectral_speedup_ratio': spectral_speedup,
            'overall_performance': self._evaluate_overall_performance(beats_spectral, spectral_speedup, avg_speedup),
            'recommendation': self._get_performance_recommendation(beats_spectral, spectral_speedup, faster_count, total_methods)
        }

    def _evaluate_overall_performance(self, beats_spectral: bool, spectral_speedup: float, avg_speedup: float) -> str:
        """评估整体性能 - 只有超过Spectral Clustering才算优秀"""
        if beats_spectral:
            if spectral_speedup > 2.0:
                return 'Agent表现优秀'
            elif spectral_speedup > 1.5:
                return 'Agent表现良好'
            else:
                return 'Agent表现一般'
        else:
            return 'Agent需要优化'

    def _get_performance_recommendation(self, beats_spectral: bool, spectral_speedup: float, faster_count: int, total_methods: int) -> str:
        """获取性能建议 - 基于是否超过Spectral Clustering"""
        if beats_spectral:
            if spectral_speedup > 3.0:
                return "🟢 Agent性能优秀，显著超过Spectral Clustering，适合实时部署"
            elif spectral_speedup > 2.0:
                return "🟡 Agent性能良好，超过Spectral Clustering，可考虑实时部署"
            elif spectral_speedup > 1.5:
                return "🟠 Agent性能一般，略超过Spectral Clustering，建议进一步优化"
            else:
                return "🟠 Agent勉强超过Spectral Clustering，建议优化以获得更大优势"
        else:
            return "🔴 Agent未能超过Spectral Clustering，需要显著改进才能用于实时部署"

    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """生成测试报告"""
        report_path = self.results_dir / 'test_report.json'

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'summary': {
                'total_tests': len(results),
                'passed': sum(1 for r in results.values() if r.get('status') == 'success'),
                'failed': sum(1 for r in results.values() if r.get('status') == 'failed'),
                'errors': sum(1 for r in results.values() if r.get('status') == 'error'),
                'timeouts': sum(1 for r in results.values() if r.get('status') == 'timeout')
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📊 测试报告已保存: {report_path}")
        return str(report_path)

    def _print_performance_comparison(self, comparison: Dict[str, Any]):
        """打印性能对比结果"""
        if not comparison.get('success'):
            print("❌ 无法生成性能对比报告")
            return

        print(f"\n🏆 Agent vs Baseline 性能对比结果")
        print("=" * 80)

        info = comparison['test_info']
        print(f"网络: {info['network'].upper()}")
        print(f"测试次数: {info['num_runs']}")
        print(f"时间: {info['timestamp']}")
        print()

        # Agent性能
        if comparison['agent_performance']:
            agent = comparison['agent_performance']
            print(f"🤖 RL Agent (PPO) 性能:")
            print(f"   平均时间: {agent['mean_ms']:.2f} ± {agent['std_ms']:.2f} ms")
            print(f"   中位数: {agent['p50_ms']:.2f} ms")
            print(f"   95%分位: {agent['p95_ms']:.2f} ms")
            print(f"   99%分位: {agent['p99_ms']:.2f} ms")
            print()

        # Baseline性能
        if comparison['baseline_performance']:
            print(f"📊 Baseline算法性能:")
            for method_name, method_data in comparison['baseline_performance'].items():
                if 'mean_ms' in method_data:
                    print(f"   {method_name}:")
                    print(f"     平均时间: {method_data['mean_ms']:.2f} ± {method_data['std_ms']:.2f} ms")
                    print(f"     99%分位: {method_data['p99_ms']:.2f} ms")
                    print(f"     成功率: {method_data['successful_runs']}/{info['num_runs']}")
                else:
                    print(f"   {method_name}: ❌ 测试失败")
            print()

        # 对比分析
        if comparison['comparison_analysis']:
            print(f"📈 性能对比分析:")
            agent_perf = comparison['agent_performance']

            for method_name, analysis in comparison['comparison_analysis'].items():
                speedup = analysis['mean_speedup_ratio']
                improvement = analysis['mean_improvement_pct']

                if speedup > 1:
                    status = "🟢 Agent更快"
                    desc = f"快 {speedup:.1f}x ({improvement:+.1f}%)"
                else:
                    status = "🔴 Baseline更快"
                    desc = f"慢 {1/speedup:.1f}x ({improvement:+.1f}%)"

                print(f"   vs {method_name}: {status}")
                print(f"     Agent: {agent_perf['mean_ms']:.2f}ms vs Baseline: {analysis['baseline_mean_ms']:.2f}ms")
                print(f"     性能差异: {desc}")
            print()

        # 总结
        if comparison['summary']:
            summary = comparison['summary']
            print(f"🎯 总体评估:")
            print(f"   Agent优于Baseline: {summary['agent_faster_than_baselines']}")
            print(f"   平均加速比: {summary['average_speedup_ratio']:.2f}x")
            print(f"   最佳加速比: {summary['best_speedup_ratio']:.2f}x")

            # 突出显示与Spectral Clustering的对比（关键指标）
            if 'beats_spectral_clustering' in summary:
                spectral_status = "✅ 是" if summary['beats_spectral_clustering'] else "❌ 否"
                spectral_ratio = summary.get('spectral_speedup_ratio', 0)
                print(f"   🔑 超过Spectral Clustering: {spectral_status} ({spectral_ratio:.2f}x)")

            print(f"   整体表现: {summary['overall_performance']}")
            print(f"   建议: {summary['recommendation']}")

    def _save_comparison_results(self, comparison: Dict[str, Any], network: str):
        """保存性能对比结果"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"agent_vs_baseline_comparison_{network}_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"\n💾 性能对比结果已保存: {filepath}")

        # 同时保存简化的CSV格式便于分析
        if comparison.get('success') and comparison.get('comparison_analysis'):
            csv_filename = f"performance_summary_{network}_{timestamp}.csv"
            csv_filepath = self.results_dir / csv_filename

            try:
                import pandas as pd

                # 准备CSV数据
                csv_data = []
                agent_perf = comparison['agent_performance']

                # Agent数据
                csv_data.append({
                    'Method': 'RL Agent (PPO)',
                    'Mean_ms': agent_perf['mean_ms'],
                    'Std_ms': agent_perf['std_ms'],
                    'P99_ms': agent_perf['p99_ms'],
                    'Speedup_vs_Agent': 1.0,
                    'Performance_Status': 'Reference'
                })

                # Baseline数据
                for method_name, analysis in comparison['comparison_analysis'].items():
                    csv_data.append({
                        'Method': method_name,
                        'Mean_ms': analysis['baseline_mean_ms'],
                        'Std_ms': comparison['baseline_performance'][method_name].get('std_ms', 0),
                        'P99_ms': analysis['baseline_p99_ms'],
                        'Speedup_vs_Agent': 1.0 / analysis['mean_speedup_ratio'],  # Baseline相对于Agent的速度
                        'Performance_Status': analysis['mean_performance']
                    })

                df = pd.DataFrame(csv_data)
                df.to_csv(csv_filepath, index=False)
                print(f"📊 CSV摘要已保存: {csv_filepath}")

            except ImportError:
                print("⚠️ pandas未安装，跳过CSV格式保存")
            except Exception as e:
                print(f"⚠️ CSV保存失败: {e}")



def run_baseline_comparison(config_path: Optional[str] = None, network: str = 'ieee57', mode: str = 'standard', model_path: Optional[str] = None, custom_scenarios: Optional[List[str]] = None, custom_runs: Optional[int] = None):
    """运行baseline对比测试"""
    print(f"🔍 启动Baseline对比测试")
    print(f"   测试网络: {network.upper()}")
    print(f"   评估模式: {mode}")
    if model_path:
        print(f"   预训练模型: {model_path}")
    print("=" * 50)

    if ComprehensiveAgentEvaluator is None:
        print("❌ 错误: ComprehensiveAgentEvaluator不可用，请检查依赖")
        return False

    try:
        # 创建评估器
        evaluator = ComprehensiveAgentEvaluator(config_path, model_path)

        # 运行对比测试
        results = evaluator.run_baseline_comparison(network, custom_scenarios, custom_runs)

        if results['success']:
            print(f"\n✅ Baseline对比测试完成！")

            # 生成可视化
            evaluator.create_comparison_visualization(results['results'])

            # 生成报告
            evaluator.generate_evaluation_report(comparison_results=results)

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


def run_generalization_test(config_path: Optional[str] = None, train_network: str = 'ieee14', mode: str = 'standard', model_path: Optional[str] = None, custom_runs: Optional[int] = None):
    """运行泛化能力测试"""
    print(f"🌐 启动泛化能力测试")
    print(f"   训练网络: {train_network.upper()}")
    print(f"   评估模式: {mode}")
    if model_path:
        print(f"   预训练模型: {model_path}")
    print("=" * 50)

    if ComprehensiveAgentEvaluator is None:
        print("❌ 错误: ComprehensiveAgentEvaluator不可用，请检查依赖")
        return False

    try:
        # 创建评估器
        evaluator = ComprehensiveAgentEvaluator(config_path, model_path)

        # 运行泛化测试
        results = evaluator.run_generalization_test(train_network, custom_runs=custom_runs)

        if results['success']:
            print(f"\n✅ 泛化能力测试完成！")

            # 生成可视化
            evaluator.create_generalization_visualization(results['results'], train_network)

            # 生成报告
            evaluator.generate_evaluation_report(generalization_results=results)

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

    if ComprehensiveAgentEvaluator is None:
        print("❌ 错误: ComprehensiveAgentEvaluator不可用，请检查依赖")
        return False

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
            evaluator.generate_evaluation_report(
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
        description='电力网络分区智能体现代化测试系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 传统评估测试
  python test.py --baseline --network ieee30 --model models/agent_ieee57_adaptive_best.pth
  python test.py --generalization --train-network ieee14 --model models/agent_ieee57_adaptive_best.pth
  python test.py --comprehensive --mode standard --model models/agent_ieee57_adaptive_best.pth

  # 实时性能测试（重点功能 - Agent vs Baseline算法性能对比）
  python test.py --benchmark --model models/agent_ieee57_adaptive_best.pth --network ieee57 --runs 100

  # 性能测试说明：
  # - 测试Agent决策时间 vs Baseline算法（谱聚类、K-means）执行时间
  # - 生成详细的性能对比报告和可视化结果
  # - 评估Agent在实时电网控制场景下的性能优势

  # 其他
  python test.py --list-modes                                  # 查看可用模式
        """
    )

    # 测试类型选择
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument('--baseline', action='store_true', help='运行Baseline对比测试')
    test_group.add_argument('--generalization', action='store_true', help='运行泛化能力测试')
    test_group.add_argument('--comprehensive', action='store_true', help='运行完整评估')
    test_group.add_argument('--benchmark', action='store_true', help='运行实时性能测试（Agent vs Baseline算法执行时间对比，包含谱聚类和K-means）')
    test_group.add_argument('--list-modes', action='store_true', help='列出可用的评估模式')

    # 配置参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['quick', 'standard', 'comprehensive'],
                       help='评估模式 (默认: standard)')
    parser.add_argument('--model', type=str, help='预训练模型路径 (传统测试必需)')



    # 自定义测试参数
    parser.add_argument('--scenarios', type=str, nargs='+',
                       choices=['normal', 'high_load', 'unbalanced', 'fault'],
                       help='(自定义) 指定一个或多个测试场景')
    parser.add_argument('--runs', type=int, help='(自定义) 指定每个测试的运行次数')

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
        print("🎯 电力网络分区智能体现代化测试系统")
        print("=" * 60)
        print(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 处理不需要模型的命令
        if args.list_modes:
            print_available_modes()
            return 0



        # 处理实时性能测试
        if args.benchmark:
            if not args.model:
                print("❌ 错误：性能测试需要提供模型路径")
                print("💡 使用方法: python test.py --benchmark --model <模型路径>")
                return 1

            performance_tester = RealTimePerformanceTester()

            print("⚡ 运行Agent vs Baseline实时性能对比测试")
            results = performance_tester.run_engineering_performance_test(
                model_path=args.model,
                network=args.network,
                num_runs=args.runs if args.runs else 100
            )

            if results['success']:
                print(f"\n🎉 Agent vs Baseline性能对比测试完成！")
                print(f"📁 详细结果请查看: performance_results/ 目录")
                print(f"📊 性能对比报告: agent_vs_baseline_comparison_{args.network}_*.json")
                print(f"📈 CSV摘要文件: performance_summary_{args.network}_*.csv")
                return 0
            else:
                print(f"\n❌ 性能对比测试失败: {results.get('error', '未知错误')}")
                return 1

        # 检查传统测试的模型参数
        if not args.model and (args.baseline or args.generalization or args.comprehensive):
            print("❌ 错误：传统测试必须提供预训练模型路径")
            print("💡 使用方法: python test.py <测试类型> --model <模型路径>")
            print("📋 查看帮助: python test.py --help")
            return 1

        success = False

        if args.baseline:
            success = run_baseline_comparison(
                config_path=args.config,
                network=args.network,
                mode=args.mode,
                model_path=args.model,
                custom_scenarios=args.scenarios,
                custom_runs=args.runs
            )

        elif args.generalization:
            success = run_generalization_test(
                config_path=args.config,
                train_network=args.train_network,
                mode=args.mode,
                model_path=args.model,
                custom_runs=args.runs
            )

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
