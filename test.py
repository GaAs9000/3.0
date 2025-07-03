#!/usr/bin/env python3
"""
电力网络分区强化学习测试评估模块

专门用于模型测试，与训练分离：
- 跨网络泛化评估
- 可视化分析  
- 基线方法对比
- 性能指标分析
"""

import torch
import numpy as np
import argparse
import yaml
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 添加路径
sys.path.append(str(Path(__file__).parent / 'code' / 'src'))
sys.path.append(str(Path(__file__).parent / 'code'))

from train import UnifiedTrainingSystem, load_power_grid_data


class QuickTester:
    """快速测试器 - 用于验证跨网络泛化评估"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.system = UnifiedTrainingSystem(config_path)
        self.config = self.system.config
        self.device = self.system.device
        
        print(f"🔬 快速测试器初始化")
        print(f"   训练网络: {self.config['data']['case_name']}")
        print(f"   设备: {self.device}")
    
    def run_generalization_test(self, num_episodes: int = 5):
        """运行快速跨网络泛化测试"""
        print(f"\n🌐 开始跨网络泛化测试")
        print("=" * 50)
        
        # 1. 快速训练模型
        print("🎓 快速训练模型...")
        training_results = self.system.run_training(mode='fast')
        
        if not training_results.get('success', False):
            print("❌ 训练失败，无法进行测试")
            return
        
        print(f"✅ 训练完成，最佳奖励: {training_results.get('best_reward', 0):.3f}")
        
        # 2. 创建测试环境
        print("\n🔧 创建测试环境...")
        agent, env = self._create_test_env()
        
        # 3. 跨网络泛化测试
        print(f"\n🔍 跨网络泛化测试 (每个网络{num_episodes}轮)...")
        
        train_case = self.config['data']['case_name']
        test_cases = ['ieee14', 'ieee30', 'ieee57']
        test_cases = [case for case in test_cases if case != train_case][:2]  # 只测试2个网络节省时间
        
        print(f"   训练网络: {train_case}")
        print(f"   测试网络: {test_cases}")
        
        results = []
        
        for test_case in test_cases:
            print(f"\n   🔍 测试 {test_case.upper()}...")
            
            try:
                # 简化的跨网络测试
                test_result = self._simple_cross_network_test(agent, test_case, num_episodes)
                results.append({
                    'network': test_case,
                    'success_rate': test_result['success_rate'],
                    'avg_reward': test_result['avg_reward']
                })
                
                # 即时反馈
                if test_result['success_rate'] > 0.4:
                    status = "🎉 优秀泛化"
                elif test_result['success_rate'] > 0.2:
                    status = "✅ 良好泛化"
                else:
                    status = "⚠️ 需要改进"
                
                print(f"     结果: 成功率 {test_result['success_rate']:.1%}, 平均奖励 {test_result['avg_reward']:.3f} - {status}")
                
            except Exception as e:
                print(f"     ❌ 测试失败: {e}")
                results.append({
                    'network': test_case,
                    'success_rate': 0.0,
                    'avg_reward': -10.0,
                    'error': str(e)
                })
        
        # 4. 总结结果
        self._print_quick_summary(train_case, results)
        
        return results
    
    def _create_test_env(self):
        """创建测试环境和智能体"""
        from code.src.data_processing import PowerGridDataProcessor
        from code.src.gat import create_hetero_graph_encoder
        from code.src.rl.environment import PowerGridPartitioningEnv
        from code.src.rl.agent import PPOAgent
        
        # 数据处理
        processor = PowerGridDataProcessor(
            normalize=self.config['data']['normalize'],
            cache_dir=self.config['data']['cache_dir']
        )
        mpc = load_power_grid_data(self.config['data']['case_name'])
        hetero_data = processor.graph_from_mpc(mpc, self.config).to(self.device)
        
        # GAT编码器
        gat_config = self.config['gat']
        encoder = create_hetero_graph_encoder(
            hetero_data,
            hidden_channels=gat_config['hidden_channels'],
            gnn_layers=gat_config['gnn_layers'],
            heads=gat_config['heads'],
            output_dim=gat_config['output_dim']
        ).to(self.device)
        
        with torch.no_grad():
            node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data, self.config)
        
        # 环境
        env_config = self.config['environment']
        env = PowerGridPartitioningEnv(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=env_config['num_partitions'],
            reward_weights=env_config['reward_weights'],
            max_steps=env_config['max_steps'],
            device=self.device,
            attention_weights=attention_weights,
            config=self.config
        )
        
        # 智能体
        agent_config = self.config['agent']
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
        
        return agent, env
    
    def _simple_cross_network_test(self, agent, test_case: str, num_episodes: int):
        """简化的跨网络测试"""
        # 加载测试网络
        test_mpc = load_power_grid_data(test_case)
        
        # 创建测试配置
        test_config = self.config.copy()
        test_config['data']['case_name'] = test_case
        
        # 动态调整分区数
        test_bus_count = test_mpc['bus'].shape[0]
        if test_bus_count <= 14:
            test_partitions = 3
        elif test_bus_count <= 30:
            test_partitions = 4
        else:
            test_partitions = 5
        
        test_config['environment']['num_partitions'] = test_partitions
        
        # 快速创建测试环境
        from code.src.data_processing import PowerGridDataProcessor
        from code.src.gat import create_hetero_graph_encoder
        from code.src.rl.environment import PowerGridPartitioningEnv
        
        processor = PowerGridDataProcessor(
            normalize=test_config['data']['normalize'],
            cache_dir=test_config['data']['cache_dir']
        )
        
        test_hetero_data = processor.graph_from_mpc(test_mpc, test_config).to(self.device)
        
        gat_config = test_config['gat']
        test_encoder = create_hetero_graph_encoder(
            test_hetero_data,
            hidden_channels=gat_config['hidden_channels'],
            gnn_layers=gat_config['gnn_layers'],
            heads=gat_config['heads'],
            output_dim=gat_config['output_dim']
        ).to(self.device)
        
        with torch.no_grad():
            test_node_embeddings, test_attention_weights = test_encoder.encode_nodes_with_attention(
                test_hetero_data, test_config
            )
        
        test_env = PowerGridPartitioningEnv(
            hetero_data=test_hetero_data,
            node_embeddings=test_node_embeddings,
            num_partitions=test_partitions,
            reward_weights=test_config['environment']['reward_weights'],
            max_steps=test_config['environment']['max_steps'],
            device=self.device,
            attention_weights=test_attention_weights,
            config=test_config
        )
        
        print(f"       环境: {test_env.total_nodes}节点 → {test_partitions}分区")
        
        # 评估
        eval_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = test_env.reset()
            episode_reward = 0
            episode_info = {}
            
            for step in range(200):
                action, _, _ = agent.select_action(state, training=False)
                if action is None:
                    break
                
                state, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if info:
                    episode_info.update(info)
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            
            # 评估成功
            is_success = self._evaluate_success(episode_reward, episode_info)
            if is_success:
                success_count += 1
        
        return {
            'success_rate': success_count / num_episodes,
            'avg_reward': np.mean(eval_rewards)
        }
    
    def _evaluate_success(self, episode_reward: float, episode_info: Dict[str, Any]) -> bool:
        """评估成功标准"""
        if episode_reward > -1.0:
            return True
        
        if episode_info and 'metrics' in episode_info:
            metrics = episode_info['metrics']
            cv = metrics.get('cv', metrics.get('load_cv', 1.0))
            if cv < 0.5:
                return True
        
        if episode_reward > -2.5:
            return True
        
        return False
    
    def _print_quick_summary(self, train_case: str, results: List[Dict]):
        """打印快速测试总结"""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            
            console = Console()
            
            # 创建结果表格
            table = Table(title="🌐 跨网络泛化测试结果", show_header=True, header_style="bold cyan")
            table.add_column("测试网络", style="bold")
            table.add_column("成功率", justify="center")
            table.add_column("平均奖励", justify="center")
            table.add_column("泛化评价", justify="center")
            
            total_success_rate = 0
            valid_count = 0
            
            for result in results:
                network = result['network']
                success_rate = result['success_rate']
                avg_reward = result['avg_reward']
                
                if 'error' not in result:
                    total_success_rate += success_rate
                    valid_count += 1
                
                if success_rate > 0.4:
                    evaluation = "🎉 优秀"
                    style = "bold green"
                elif success_rate > 0.2:
                    evaluation = "✅ 良好"
                    style = "green"
                elif success_rate > 0.1:
                    evaluation = "⚠️ 有限"
                    style = "yellow"
                else:
                    evaluation = "❌ 不足"
                    style = "red"
                
                table.add_row(
                    network.upper(),
                    f"[{style}]{success_rate:.1%}[/{style}]",
                    f"[{style}]{avg_reward:.3f}[/{style}]",
                    evaluation
                )
            
            # 计算总体泛化分数
            overall_success_rate = total_success_rate / valid_count if valid_count > 0 else 0
            
            if overall_success_rate > 0.4:
                overall_status = "[bold green]🌟 优秀泛化能力[/bold green]"
            elif overall_success_rate > 0.2:
                overall_status = "[green]✅ 良好泛化能力[/green]"
            elif overall_success_rate > 0.1:
                overall_status = "[yellow]⚠️ 有限泛化能力[/yellow]"
            else:
                overall_status = "[red]❌ 泛化能力不足[/red]"
            
            console.print(table)
            
            # 总结面板
            summary_text = f"训练网络: {train_case.upper()}\n"
            summary_text += f"测试网络: {len(results)}个\n"
            summary_text += f"平均成功率: {overall_success_rate:.1%}\n"
            summary_text += f"总体评价: {overall_status}"
            
            console.print(Panel(summary_text, title="📊 泛化能力总结", border_style="blue"))
            
        except ImportError:
            # 备用输出
            print(f"\n📊 跨网络泛化测试总结:")
            print(f"   训练网络: {train_case}")
            
            total_success_rate = 0
            valid_count = 0
            
            for result in results:
                network = result['network']
                success_rate = result['success_rate']
                
                if 'error' not in result:
                    total_success_rate += success_rate
                    valid_count += 1
                
                print(f"   {network}: 成功率 {success_rate:.1%}")
            
            overall_success_rate = total_success_rate / valid_count if valid_count > 0 else 0
            print(f"   总体泛化能力: {overall_success_rate:.1%}")

    def generate_performance_dashboard(self, test_results: List[Dict], output_filename: Optional[str] = None) -> Optional[Path]:
        """
        生成性能分析HTML仪表板

        Args:
            test_results: 测试结果数据
            output_filename: 输出文件名，如果为None则自动生成

        Returns:
            生成的HTML文件路径，失败则返回None
        """
        try:
            from code.src.html_dashboard_generator import HTMLDashboardGenerator

            # 创建HTML仪表板生成器
            dashboard_config = self.config.get('html_dashboard', {})
            generator = HTMLDashboardGenerator(dashboard_config)

            # 准备测试数据
            networks = [result['network'] for result in test_results]
            success_rates = [result['success_rate'] for result in test_results if 'error' not in result]
            avg_rewards = [result['avg_reward'] for result in test_results if 'error' not in result]

            # 计算总体指标
            overall_success_rate = np.mean(success_rates) if success_rates else 0
            overall_avg_reward = np.mean(avg_rewards) if avg_rewards else 0

            performance_data = {
                'test_type': 'cross_network_generalization',
                'train_network': self.config['data']['case_name'],
                'test_networks': networks,
                'success_rates': success_rates,
                'avg_rewards': avg_rewards,
                'overall_success_rate': overall_success_rate,
                'overall_avg_reward': overall_avg_reward,
                'test_results': test_results,
                'config': self.config,
                'session_name': f"Performance_Test_{time.strftime('%Y%m%d_%H%M%S')}"
            }

            # 生成HTML仪表板
            html_path = generator.generate_performance_dashboard(
                performance_data, output_filename
            )

            return html_path

        except Exception as e:
            print(f"⚠️ 性能分析仪表板生成失败: {e}")
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='电力网络分区RL模型测试系统')
    
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=5, help='每个测试的评估回合数')
    parser.add_argument('--quick', action='store_true', help='快速跨网络泛化测试')
    
    args = parser.parse_args()
    
    try:
        print("🔬 启动模型测试系统")
        print("=" * 50)
        
        if args.quick:
            # 快速泛化测试
            tester = QuickTester(config_path=args.config)
            results = tester.run_generalization_test(num_episodes=args.episodes)

            # 生成性能分析HTML仪表板
            print(f"\n📊 生成性能分析仪表板...")
            html_path = tester.generate_performance_dashboard(results)
            if html_path:
                print(f"✅ 性能分析仪表板已生成: {html_path}")

            print(f"\n✅ 快速测试完成！")
            print(f"💡 使用 'python test.py --help' 查看更多测试选项")
            
        else:
            print("📋 完整测试功能开发中...")
            print("🚀 当前可用:")
            print("   python test.py --quick           # 快速跨网络泛化测试")
            print("   python test.py --quick --episodes 10  # 更多测试回合")
        
        return 0
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
