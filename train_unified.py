#!/usr/bin/env python3
"""
电力网络分区强化学习统一训练系统

整合了所有训练模式：
- 标准训练 (IEEE 14, 30, 57节点)
- 大规模训练 (IEEE 118节点)
- 并行训练支持
- 场景生成训练
- 课程学习训练
- 基线方法对比
"""

import torch
import numpy as np
import argparse
import yaml
import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.append(str(Path(__file__).parent / 'src'))

# 禁用警告
warnings.filterwarnings('ignore')

# 动态导入检查
def check_dependencies():
    """检查可选依赖"""
    deps = {
        'stable_baselines3': False,
        'tensorboard': False,
        'plotly': False
    }
    
    try:
        import stable_baselines3
        deps['stable_baselines3'] = True
    except ImportError:
        pass
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        deps['tensorboard'] = True
    except ImportError:
        pass
        
    try:
        import plotly.graph_objects as go
        deps['plotly'] = True
    except ImportError:
        pass
    
    return deps


class UnifiedTrainingSystem:
    """统一训练系统 - 整合所有训练模式"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化统一训练系统"""
        self.deps = check_dependencies()
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.setup_directories()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        return {
            'system': {
                'name': 'unified_power_grid_partitioning',
                'version': '2.0',
                'device': 'auto',
                'seed': 42,
                'num_threads': 1
            },
            'data': {
                'case_name': 'ieee14',
                'normalize': True,
                'cache_dir': 'cache'
            },
            'training': {
                'mode': 'standard',  # standard, parallel, curriculum, large_scale
                'num_episodes': 1000,
                'max_steps_per_episode': 200,
                'update_interval': 10,
                'save_interval': 100,
                'eval_interval': 50
            },
            'environment': {
                'num_partitions': 3,
                'max_steps': 200,
                'reward_weights': {
                    'load_balance': 0.4,
                    'electrical_decoupling': 0.4,
                    'power_balance': 0.2
                }
            },
            'gat': {
                'hidden_channels': 64,
                'gnn_layers': 3,
                'heads': 4,
                'output_dim': 128,
                'dropout': 0.1
            },
            'agent': {
                'type': 'ppo',  # ppo, sb3_ppo
                'lr_actor': 3e-4,
                'lr_critic': 1e-3,
                'gamma': 0.99,
                'eps_clip': 0.2,
                'k_epochs': 4,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'hidden_dim': 256
            },
            'parallel_training': {
                'enabled': False,
                'num_cpus': 12,
                'total_timesteps': 5_000_000,
                'scenario_generation': True
            },
            'scenario_generation': {
                'enabled': False,
                'perturb_prob': 0.8,
                'perturb_types': ['n-1', 'load_gen_fluctuation', 'both', 'none'],
                'scale_range': [0.8, 1.2]
            },
            'curriculum': {
                'enabled': False,
                'start_partitions': 2,
                'end_partitions': 5,
                'episodes_per_stage': 200
            },
            'evaluation': {
                'num_episodes': 20,
                'include_baselines': True,
                'baseline_methods': ['spectral', 'kmeans', 'random']
            },
            'visualization': {
                'enabled': True,
                'save_figures': True,
                'figures_dir': 'figures',
                'interactive': True
            },
            'logging': {
                'use_tensorboard': True,
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',
                'console_log_interval': 10,
                'metrics_save_interval': 50
            }
        }
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_config = self.config['system'].get('device', 'auto')
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        print(f"🔧 使用设备: {device}")
        return device
    
    def setup_directories(self):
        """创建必要的目录"""
        dirs = [
            self.config['data']['cache_dir'],
            self.config['logging']['log_dir'], 
            self.config['logging']['checkpoint_dir'],
            self.config['visualization']['figures_dir'],
            'models', 'output', 'experiments'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_training_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取不同训练模式的配置"""
        base_config = self.config.copy()
        
        configs = {
            'quick': {
                **base_config,
                'training': {
                    **base_config['training'],
                    'num_episodes': 100,
                    'max_steps_per_episode': 50,
                    'update_interval': 5
                }
            },
            'standard': base_config,
            'full': {
                **base_config,
                'training': {
                    **base_config['training'],
                    'num_episodes': 2000,
                    'max_steps_per_episode': 500,
                    'update_interval': 20
                }
            },
            'ieee118': {
                **base_config,
                'data': {
                    **base_config['data'],
                    'case_name': 'ieee118'
                },
                'environment': {
                    **base_config['environment'],
                    'num_partitions': 8,
                    'max_steps': 500
                },
                'gat': {
                    **base_config['gat'],
                    'hidden_channels': 128,
                    'gnn_layers': 4,
                    'heads': 8,
                    'output_dim': 256
                },
                'training': {
                    **base_config['training'],
                    'num_episodes': 5000,
                    'max_steps_per_episode': 500
                },
                'parallel_training': {
                    **base_config['parallel_training'],
                    'enabled': True
                },
                'scenario_generation': {
                    **base_config['scenario_generation'],
                    'enabled': True
                }
            },
            'parallel': {
                **base_config,
                'parallel_training': {
                    **base_config['parallel_training'],
                    'enabled': True
                },
                'scenario_generation': {
                    **base_config['scenario_generation'],
                    'enabled': True
                }
            },
            'curriculum': {
                **base_config,
                'curriculum': {
                    **base_config['curriculum'],
                    'enabled': True
                }
            }
        }
        
        return configs
    
    def run_training(self, mode: str = 'standard', **kwargs) -> Dict[str, Any]:
        """运行训练"""
        print(f"\n🚀 开始{mode.upper()}模式训练")
        print("=" * 60)
        
        # 获取模式配置
        configs = self.get_training_configs()
        if mode not in configs:
            print(f"⚠️ 未知训练模式: {mode}，使用标准模式")
            mode = 'standard'
        
        config = configs[mode]
        
        # 应用命令行参数覆盖
        for key, value in kwargs.items():
            if '.' in key:
                # 支持嵌套配置，如 training.num_episodes
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        # 设置随机种子
        torch.manual_seed(config['system']['seed'])
        np.random.seed(config['system']['seed'])
        
        try:
            if mode == 'parallel' or config['parallel_training']['enabled']:
                return self._run_parallel_training(config)
            elif mode == 'curriculum' or config['curriculum']['enabled']:
                return self._run_curriculum_training(config)
            else:
                return self._run_standard_training(config)
                
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _run_standard_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行标准训练"""
        print("📊 标准训练模式")
        
        # 导入必要模块
        from data_processing import PowerGridDataProcessor
        from gat import create_hetero_graph_encoder
        from rl.environment import PowerGridPartitioningEnv
        from rl.agent import PPOAgent
        from src.train_rl import UnifiedTrainer, TrainingLogger, CheckpointManager
        
        # 1. 数据处理
        print("\n1️⃣ 数据处理...")
        processor = PowerGridDataProcessor(
            normalize=config['data']['normalize'],
            cache_dir=config['data']['cache_dir']
        )
        
        # 加载数据
        if config['data']['case_name'] == 'ieee118':
            from src.train_parallel_118bus import load_case118_data
            mpc = load_case118_data()
        else:
            from src.train_rl import load_power_grid_data
            mpc = load_power_grid_data(config['data']['case_name'])
        
        hetero_data = processor.graph_from_mpc(mpc).to(self.device)
        print(f"✅ 数据加载完成: {hetero_data}")
        
        # 2. GAT编码器
        print("\n2️⃣ GAT编码器...")
        gat_config = config['gat']
        encoder = create_hetero_graph_encoder(
            hetero_data,
            hidden_channels=gat_config['hidden_channels'],
            gnn_layers=gat_config['gnn_layers'],
            heads=gat_config['heads'],
            output_dim=gat_config['output_dim']
        ).to(self.device)
        
        with torch.no_grad():
            node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data)
        
        print(f"✅ 编码器初始化完成")
        
        # 3. 环境
        print("\n3️⃣ 强化学习环境...")
        env_config = config['environment']
        env = PowerGridPartitioningEnv(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=env_config['num_partitions'],
            reward_weights=env_config['reward_weights'],
            max_steps=env_config['max_steps'],
            device=self.device,
            attention_weights=attention_weights
        )
        
        print(f"✅ 环境创建完成: {env.total_nodes}节点, {env.num_partitions}分区")
        
        # 4. 智能体
        print("\n4️⃣ PPO智能体...")
        agent_config = config['agent']
        node_embedding_dim = env.state_manager.embedding_dim
        region_embedding_dim = node_embedding_dim * 2
        
        agent = PPOAgent(
            node_embedding_dim=node_embedding_dim,
            region_embedding_dim=region_embedding_dim,
            num_partitions=env.num_partitions,
            lr_actor=agent_config['lr_actor'],
            lr_critic=agent_config['lr_critic'],
            gamma=agent_config['gamma'],
            eps_clip=agent_config['eps_clip'],
            k_epochs=agent_config['k_epochs'],
            entropy_coef=agent_config['entropy_coef'],
            value_coef=agent_config['value_coef'],
            device=self.device
        )
        
        print(f"✅ 智能体创建完成")
        
        # 5. 训练
        print("\n5️⃣ 开始训练...")
        trainer = UnifiedTrainer(agent=agent, env=env, config=config)
        
        training_config = config['training']
        history = trainer.train(
            num_episodes=training_config['num_episodes'],
            max_steps_per_episode=training_config['max_steps_per_episode'],
            update_interval=training_config['update_interval']
        )
        
        # 6. 评估
        print("\n6️⃣ 评估...")
        eval_stats = trainer.evaluate()
        
        # 7. 基线对比
        baseline_results = None
        if config['evaluation']['include_baselines']:
            print("\n7️⃣ 基线方法对比...")
            try:
                from baseline import run_baseline_comparison
                baseline_results = run_baseline_comparison(env, agent, seed=42)
                print("✅ 基线对比完成")
            except Exception as e:
                print(f"⚠️ 基线对比失败: {e}")
        
        # 8. 可视化
        if config['visualization']['enabled']:
            print("\n8️⃣ 生成可视化...")
            try:
                trainer.run_final_visualization()
                if baseline_results is not None and config['visualization']['interactive']:
                    from visualization import run_interactive_visualization
                    run_interactive_visualization(env, baseline_results)
            except Exception as e:
                print(f"⚠️ 可视化失败: {e}")
        
        trainer.close()
        
        return {
            'success': True,
            'mode': 'standard',
            'config': config,
            'history': history,
            'eval_stats': eval_stats,
            'baseline_results': baseline_results,
            'best_reward': trainer.logger.best_reward
        }
    
    def _run_parallel_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行并行训练"""
        print("🌐 并行训练模式")
        
        if not self.deps['stable_baselines3']:
            print("❌ 并行训练需要stable-baselines3，请安装：pip install stable-baselines3")
            return {'success': False, 'error': 'Missing stable-baselines3'}
        
        # 使用118节点系统的并行训练逻辑
        from src.train_parallel_118bus import main as run_parallel_118
        
        # 模拟命令行参数
        import sys
        original_argv = sys.argv.copy()
        
        parallel_config = config['parallel_training']
        sys.argv = [
            'train_parallel_118bus.py',
            '--num-cpus', str(parallel_config['num_cpus']),
            '--total-timesteps', str(parallel_config['total_timesteps'])
        ]
        
        if not parallel_config['scenario_generation']:
            sys.argv.append('--no-scenario')
        
        try:
            result = run_parallel_118()
            return {
                'success': True,
                'mode': 'parallel',
                'config': config
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            sys.argv = original_argv
    
    def _run_curriculum_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行课程学习训练"""
        print("📚 课程学习训练模式")
        
        curriculum_config = config['curriculum']
        start_partitions = curriculum_config['start_partitions']
        end_partitions = curriculum_config['end_partitions']
        episodes_per_stage = curriculum_config['episodes_per_stage']
        
        results = []
        
        for num_partitions in range(start_partitions, end_partitions + 1):
            print(f"\n📖 课程阶段: {num_partitions}个分区")
            
            # 更新配置
            stage_config = config.copy()
            stage_config['environment']['num_partitions'] = num_partitions
            stage_config['training']['num_episodes'] = episodes_per_stage
            
            # 运行该阶段的训练
            stage_result = self._run_standard_training(stage_config)
            results.append(stage_result)
            
            if not stage_result['success']:
                break
        
        return {
            'success': all(r['success'] for r in results),
            'mode': 'curriculum',
            'config': config,
            'stage_results': results
        }
    
    def run_demo(self) -> Dict[str, Any]:
        """运行完整系统演示"""
        print("\n🎪 运行完整系统演示")
        print("=" * 60)
        
        # 导入主演示函数
        from main import main as run_main_demo
        
        try:
            run_main_demo('quick')
            return {'success': True, 'mode': 'demo'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_training_report(self, results: Dict[str, Any]) -> str:
        """生成训练报告"""
        report_lines = [
            f"# 电力网络分区训练报告",
            f"",
            f"## 系统信息",
            f"- 训练模式: {results.get('mode', 'unknown')}",
            f"- 设备: {self.device}",
            f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## 配置信息"
        ]
        
        if 'config' in results:
            config = results['config']
            report_lines.extend([
                f"- 案例: {config['data']['case_name']}",
                f"- 分区数: {config['environment']['num_partitions']}",
                f"- 训练回合: {config['training']['num_episodes']}",
                f"- 最大步数: {config['training']['max_steps_per_episode']}"
            ])
        
        if 'eval_stats' in results:
            eval_stats = results['eval_stats']
            report_lines.extend([
                f"",
                f"## 评估结果",
                f"- 平均奖励: {eval_stats.get('mean_reward', 0):.4f}",
                f"- 成功率: {eval_stats.get('success_rate', 0):.4f}",
                f"- 平均负载CV: {eval_stats.get('mean_load_cv', 0):.4f}"
            ])
        
        if 'baseline_results' in results and results['baseline_results'] is not None:
            report_lines.extend([
                f"",
                f"## 基线方法对比",
                f"基线方法对比结果已保存"
            ])
        
        return "\n".join(report_lines)
    
    def save_results(self, results: Dict[str, Any], output_dir: str = 'experiments'):
        """保存训练结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        mode = results.get('mode', 'unknown')
        
        # 保存结果JSON
        results_file = output_path / f"{mode}_results_{timestamp}.json"
        
        # 清理不能序列化的对象
        clean_results = {}
        for key, value in results.items():
            if key in ['history', 'eval_stats', 'config']:
                clean_results[key] = value
            elif key == 'baseline_results' and value is not None:
                clean_results[key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)
            elif isinstance(value, (str, int, float, bool, type(None))):
                clean_results[key] = value
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        # 保存训练报告
        report = self.create_training_report(results)
        report_file = output_path / f"{mode}_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 结果已保存到: {output_path}")
        print(f"   - 结果文件: {results_file.name}")
        print(f"   - 报告文件: {report_file.name}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='电力网络分区统一训练系统')
    
    # 基本参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['quick', 'standard', 'full', 'ieee118', 'parallel', 'curriculum', 'demo'],
                       help='训练模式')
    
    # 训练参数
    parser.add_argument('--case', type=str, help='电网案例名称')
    parser.add_argument('--episodes', type=int, help='训练回合数')
    parser.add_argument('--partitions', type=int, help='分区数量')
    parser.add_argument('--device', type=str, help='计算设备')
    
    # 功能开关
    parser.add_argument('--no-baselines', action='store_true', help='跳过基线方法对比')
    parser.add_argument('--no-viz', action='store_true', help='跳过可视化')
    parser.add_argument('--save-results', action='store_true', help='保存训练结果')
    
    # 特殊模式
    parser.add_argument('--list-configs', action='store_true', help='列出所有可用配置')
    parser.add_argument('--check-deps', action='store_true', help='检查依赖')
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check_deps:
        deps = check_dependencies()
        print("📋 依赖检查结果:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")
        return
    
    # 初始化系统
    system = UnifiedTrainingSystem(args.config)
    
    # 列出配置
    if args.list_configs:
        configs = system.get_training_configs()
        print("📋 可用训练配置:")
        for name in configs.keys():
            print(f"  - {name}")
        return
    
    # 构建参数覆盖
    overrides = {}
    if args.case:
        overrides['data.case_name'] = args.case
    if args.episodes:
        overrides['training.num_episodes'] = args.episodes
    if args.partitions:
        overrides['environment.num_partitions'] = args.partitions
    if args.device:
        overrides['system.device'] = args.device
    if args.no_baselines:
        overrides['evaluation.include_baselines'] = False
    if args.no_viz:
        overrides['visualization.enabled'] = False
    
    # 运行训练
    print("\n🎯 电力网络分区统一训练系统启动")
    print("=" * 60)
    
    start_time = time.time()
    
    if args.mode == 'demo':
        results = system.run_demo()
    else:
        results = system.run_training(args.mode, **overrides)
    
    elapsed_time = time.time() - start_time
    
    # 结果汇总
    print("\n📊 训练完成总结")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"耗时: {elapsed_time/3600:.2f} 小时")
    print(f"状态: {'✅ 成功' if results.get('success', False) else '❌ 失败'}")
    
    if results.get('success', False):
        if 'best_reward' in results:
            print(f"最佳奖励: {results['best_reward']:.4f}")
        if 'eval_stats' in results:
            print(f"最终成功率: {results['eval_stats'].get('success_rate', 0):.4f}")
    else:
        print(f"错误: {results.get('error', 'Unknown error')}")
    
    # 保存结果
    if args.save_results and results.get('success', False):
        system.save_results(results)
    
    print("\n🎉 统一训练系统运行完成！")


if __name__ == "__main__":
    main() 