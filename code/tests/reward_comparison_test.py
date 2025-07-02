#!/usr/bin/env python3
"""
奖励系统对比测试脚本

模拟对比实验，验证简单相对改进奖励是否真的解决了跨场景偏向问题

测试内容：
1. 模拟不同场景下的训练过程
2. 对比传统绝对奖励 vs 简单相对奖励
3. 分析场景访问分布和性能差异
4. 生成可视化报告
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class ScenarioSimulator:
    """场景模拟器 - 模拟不同难度的电力系统场景"""

    def __init__(self):
        self.scenarios = {
            'normal': {'base_quality': 0.75, 'difficulty': 1.0, 'weight': 0.4},
            'light_fault': {'base_quality': 0.60, 'difficulty': 1.5, 'weight': 0.25},
            'severe_fault': {'base_quality': 0.35, 'difficulty': 2.5, 'weight': 0.15},
            'high_load': {'base_quality': 0.50, 'difficulty': 2.0, 'weight': 0.15},
            'generation_fluctuation': {'base_quality': 0.45, 'difficulty': 2.2, 'weight': 0.05}
        }

    def sample_scenario(self) -> str:
        """根据权重随机采样场景"""
        scenarios = list(self.scenarios.keys())
        weights = [self.scenarios[s]['weight'] for s in scenarios]
        return np.random.choice(scenarios, p=weights)

    def get_scenario_info(self, scenario: str) -> Dict:
        """获取场景信息"""
        return self.scenarios[scenario]

    def simulate_quality_improvement(self, scenario: str, action_quality: float) -> float:
        """
        模拟在特定场景下的质量改进

        Args:
            scenario: 场景类型
            action_quality: 动作质量 [0, 1]，表示算法选择的动作好坏

        Returns:
            改进后的质量分数
        """
        info = self.scenarios[scenario]
        base = info['base_quality']
        difficulty = info['difficulty']

        # 在困难场景下，相同的动作质量产生更小的改进
        max_improvement = 0.3 / difficulty  # 困难场景改进空间更小
        actual_improvement = action_quality * max_improvement

        # 添加噪声
        noise = np.random.normal(0, 0.02)
        new_quality = base + actual_improvement + noise

        return np.clip(new_quality, 0.0, 1.0)

class RewardComparator:
    """奖励系统对比器"""

    def __init__(self):
        self.simulator = ScenarioSimulator()

    def absolute_reward(self, prev_quality: float, curr_quality: float) -> float:
        """传统绝对奖励"""
        gamma = 0.99
        return gamma * curr_quality - prev_quality

    def relative_reward(self, prev_quality: float, curr_quality: float) -> float:
        """简单相对改进奖励"""
        if prev_quality > 0.01:
            relative_improvement = (curr_quality - prev_quality) / prev_quality
        else:
            relative_improvement = curr_quality - prev_quality
        return np.clip(relative_improvement, -1.0, 1.0)

    def simulate_training_episode(self, reward_type: str, steps: int = 50) -> Dict:
        """
        模拟一个训练episode

        Args:
            reward_type: 'absolute' 或 'relative'
            steps: episode步数

        Returns:
            episode结果字典
        """
        scenario = self.simulator.sample_scenario()
        scenario_info = self.simulator.get_scenario_info(scenario)

        # 初始质量
        current_quality = scenario_info['base_quality'] + np.random.normal(0, 0.05)
        current_quality = np.clip(current_quality, 0.1, 0.9)

        rewards = []
        qualities = [current_quality]
        actions = []

        for step in range(steps):
            # 模拟智能体选择动作（简化为随机动作质量）
            action_quality = np.random.beta(2, 2)  # 偏向中等质量的动作
            actions.append(action_quality)

            # 计算新的质量分数
            prev_quality = current_quality
            current_quality = self.simulator.simulate_quality_improvement(scenario, action_quality)
            qualities.append(current_quality)

            # 计算奖励
            if reward_type == 'absolute':
                reward = self.absolute_reward(prev_quality, current_quality)
            else:  # relative
                reward = self.relative_reward(prev_quality, current_quality)

            rewards.append(reward)

        return {
            'scenario': scenario,
            'scenario_difficulty': scenario_info['difficulty'],
            'initial_quality': qualities[0],
            'final_quality': qualities[-1],
            'quality_improvement': qualities[-1] - qualities[0],
            'relative_improvement': (qualities[-1] - qualities[0]) / qualities[0] if qualities[0] > 0 else 0,
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards),
            'rewards': rewards,
            'qualities': qualities,
            'actions': actions
        }

    def run_comparison_experiment(self, num_episodes: int = 1000) -> Dict:
        """运行对比实验"""
        print(f"🧪 运行对比实验 ({num_episodes} episodes)")

        results = {
            'absolute': [],
            'relative': []
        }

        # 运行实验
        for reward_type in ['absolute', 'relative']:
            print(f"   测试 {reward_type} 奖励...")
            for i in range(num_episodes):
                if (i + 1) % 200 == 0:
                    print(f"     完成 {i + 1}/{num_episodes} episodes")

                episode_result = self.simulate_training_episode(reward_type)
                results[reward_type].append(episode_result)

        return results

def analyze_scenario_bias(results: Dict) -> Dict:
    """分析场景偏向问题"""
    print("\n📊 分析场景偏向问题")

    analysis = {}

    for reward_type in ['absolute', 'relative']:
        episodes = results[reward_type]

        # 统计各场景的访问和性能
        scenario_stats = defaultdict(list)
        scenario_counts = Counter()

        for episode in episodes:
            scenario = episode['scenario']
            scenario_counts[scenario] += 1
            scenario_stats[scenario].append({
                'final_quality': episode['final_quality'],
                'quality_improvement': episode['quality_improvement'],
                'relative_improvement': episode['relative_improvement'],
                'total_reward': episode['total_reward']
            })

        # 计算统计指标
        scenario_analysis = {}
        for scenario, stats in scenario_stats.items():
            scenario_analysis[scenario] = {
                'count': scenario_counts[scenario],
                'avg_final_quality': np.mean([s['final_quality'] for s in stats]),
                'avg_quality_improvement': np.mean([s['quality_improvement'] for s in stats]),
                'avg_relative_improvement': np.mean([s['relative_improvement'] for s in stats]),
                'avg_total_reward': np.mean([s['total_reward'] for s in stats]),
                'std_final_quality': np.std([s['final_quality'] for s in stats])
            }

        analysis[reward_type] = scenario_analysis

    return analysis

def generate_visualizations(results: Dict, analysis: Dict):
    """生成可视化图表"""
    print("\n📈 生成可视化图表")

    # 创建输出目录
    os.makedirs('data/experiments/reward_comparison', exist_ok=True)

    # 1. 场景性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('奖励系统对比分析', fontsize=16, fontweight='bold')

    # 1.1 各场景最终质量对比
    scenarios = list(analysis['absolute'].keys())
    absolute_qualities = [analysis['absolute'][s]['avg_final_quality'] for s in scenarios]
    relative_qualities = [analysis['relative'][s]['avg_final_quality'] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    axes[0,0].bar(x - width/2, absolute_qualities, width, label='绝对奖励', alpha=0.8)
    axes[0,0].bar(x + width/2, relative_qualities, width, label='相对奖励', alpha=0.8)
    axes[0,0].set_xlabel('场景类型')
    axes[0,0].set_ylabel('平均最终质量')
    axes[0,0].set_title('各场景最终质量对比')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(scenarios, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 1.2 相对改进对比
    absolute_improvements = [analysis['absolute'][s]['avg_relative_improvement'] for s in scenarios]
    relative_improvements = [analysis['relative'][s]['avg_relative_improvement'] for s in scenarios]

    axes[0,1].bar(x - width/2, absolute_improvements, width, label='绝对奖励', alpha=0.8)
    axes[0,1].bar(x + width/2, relative_improvements, width, label='相对奖励', alpha=0.8)
    axes[0,1].set_xlabel('场景类型')
    axes[0,1].set_ylabel('平均相对改进率')
    axes[0,1].set_title('各场景相对改进对比')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(scenarios, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 1.3 场景访问分布
    absolute_counts = [analysis['absolute'][s]['count'] for s in scenarios]
    relative_counts = [analysis['relative'][s]['count'] for s in scenarios]

    axes[1,0].bar(x - width/2, absolute_counts, width, label='绝对奖励', alpha=0.8)
    axes[1,0].bar(x + width/2, relative_counts, width, label='相对奖励', alpha=0.8)
    axes[1,0].set_xlabel('场景类型')
    axes[1,0].set_ylabel('访问次数')
    axes[1,0].set_title('场景访问分布')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(scenarios, rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 1.4 奖励分布对比
    absolute_rewards = [episode['total_reward'] for episode in results['absolute']]
    relative_rewards = [episode['total_reward'] for episode in results['relative']]

    axes[1,1].hist(absolute_rewards, bins=30, alpha=0.7, label='绝对奖励', density=True)
    axes[1,1].hist(relative_rewards, bins=30, alpha=0.7, label='相对奖励', density=True)
    axes[1,1].set_xlabel('总奖励')
    axes[1,1].set_ylabel('密度')
    axes[1,1].set_title('奖励分布对比')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/experiments/reward_comparison/comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("   ✅ 保存对比概览图: data/experiments/reward_comparison/comparison_overview.png")

def generate_detailed_report(results: Dict, analysis: Dict):
    """生成详细报告"""
    print("\n📄 生成详细报告")

    report = {
        'experiment_summary': {
            'total_episodes_per_method': len(results['absolute']),
            'scenarios_tested': list(analysis['absolute'].keys()),
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'key_findings': [],
        'scenario_analysis': analysis,
        'recommendations': []
    }

    # 分析关键发现
    for scenario in analysis['absolute'].keys():
        abs_quality = analysis['absolute'][scenario]['avg_final_quality']
        rel_quality = analysis['relative'][scenario]['avg_final_quality']
        improvement = (rel_quality - abs_quality) / abs_quality * 100

        if improvement > 1:  # 超过1%改进
            report['key_findings'].append(
                f"{scenario}场景: 相对奖励比绝对奖励性能提升{improvement:.1f}%"
            )
        elif improvement < -1:  # 超过1%退化
            report['key_findings'].append(
                f"{scenario}场景: 相对奖励比绝对奖励性能下降{abs(improvement):.1f}%"
            )

    # 计算困难场景的整体表现
    difficult_scenarios = ['severe_fault', 'high_load', 'generation_fluctuation']
    abs_difficult_avg = np.mean([analysis['absolute'][s]['avg_final_quality']
                                for s in difficult_scenarios if s in analysis['absolute']])
    rel_difficult_avg = np.mean([analysis['relative'][s]['avg_final_quality']
                                for s in difficult_scenarios if s in analysis['relative']])

    difficult_improvement = (rel_difficult_avg - abs_difficult_avg) / abs_difficult_avg * 100

    if difficult_improvement > 0:
        report['key_findings'].append(
            f"困难场景整体: 相对奖励提升{difficult_improvement:.1f}%"
        )
        report['recommendations'].append("建议采用简单相对奖励系统")
    else:
        report['key_findings'].append(
            f"困难场景整体: 相对奖励下降{abs(difficult_improvement):.1f}%"
        )
        report['recommendations'].append("需要进一步优化相对奖励算法")

    # 保存报告
    with open('data/experiments/reward_comparison/detailed_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("   ✅ 保存详细报告: data/experiments/reward_comparison/detailed_report.json")

    return report

def print_summary(report: Dict):
    """打印总结"""
    print("\n" + "="*60)
    print("🎯 实验总结")
    print("="*60)

    print(f"📊 实验规模: 每种方法测试 {report['experiment_summary']['total_episodes_per_method']} episodes")
    print(f"🎭 测试场景: {', '.join(report['experiment_summary']['scenarios_tested'])}")

    print("\n🔍 关键发现:")
    for finding in report['key_findings']:
        print(f"   • {finding}")

    print("\n💡 建议:")
    for rec in report['recommendations']:
        print(f"   • {rec}")

    print("\n📁 输出文件:")
    print("   • data/experiments/reward_comparison/comparison_overview.png")
    print("   • data/experiments/reward_comparison/detailed_report.json")

def main():
    """主函数"""
    print("🚀 奖励系统对比测试")
    print("解决跨场景训练偏向问题")
    print("="*60)

    # 1. 运行对比实验
    comparator = RewardComparator()
    results = comparator.run_comparison_experiment(num_episodes=1000)

    # 2. 分析结果
    analysis = analyze_scenario_bias(results)

    # 3. 生成可视化
    generate_visualizations(results, analysis)

    # 4. 生成详细报告
    report = generate_detailed_report(results, analysis)

    # 5. 打印总结
    print_summary(report)

    print("\n✨ 测试完成！")

if __name__ == "__main__":
    main()