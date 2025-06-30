#!/usr/bin/env python3
"""
智能自适应课程学习系统测试脚本

测试智能导演系统的各个组件功能
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_adaptive_curriculum_components():
    """测试智能自适应课程学习的各个组件"""
    print("🧪 开始测试智能自适应课程学习系统组件...")
    
    try:
        # 1. 测试导入
        print("\n1️⃣ 测试组件导入...")
        from code.src.rl.adaptive_curriculum import (
            AdaptiveCurriculumDirector,
            PerformanceAnalyzer,
            ParameterScheduler,
            SafetyMonitor,
            StageTransitionCriteria,
            ParameterEvolutionConfig,
            SafetyConfig
        )
        print("✅ 所有组件导入成功")
        
        # 2. 测试配置加载
        print("\n2️⃣ 测试配置加载...")
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查智能自适应配置
        adaptive_config = config.get('adaptive_curriculum', {})
        print(f"✅ 智能自适应配置加载成功: {len(adaptive_config)} 个配置项")
        
        # 3. 测试性能分析器
        print("\n3️⃣ 测试性能分析器...")
        analyzer = PerformanceAnalyzer(config)
        
        # 模拟episode数据
        test_episodes = [
            {'episode_length': 10, 'reward': -5.0, 'load_cv': 0.8, 'coupling_ratio': 0.6, 'connectivity': 0.7, 'success': False},
            {'episode_length': 15, 'reward': -3.0, 'load_cv': 0.6, 'coupling_ratio': 0.5, 'connectivity': 0.8, 'success': True},
            {'episode_length': 18, 'reward': -1.0, 'load_cv': 0.4, 'coupling_ratio': 0.4, 'connectivity': 0.9, 'success': True},
        ]
        
        for i, episode_info in enumerate(test_episodes):
            performance = analyzer.analyze_episode(episode_info)
            print(f"   Episode {i+1}: 复合评分 = {performance['composite_score']:.3f}")
        
        trend_analysis = analyzer.get_trend_analysis(window_size=3)
        print(f"✅ 趋势分析: 趋势={trend_analysis['trend']:.3f}, 稳定性={trend_analysis['stability']:.3f}")
        
        # 4. 测试参数调度器
        print("\n4️⃣ 测试参数调度器...")
        param_config = ParameterEvolutionConfig()
        scheduler = ParameterScheduler(param_config)
        
        # 测试不同阶段的参数
        for stage in range(1, 5):
            for progress in [0.0, 0.5, 1.0]:
                params = scheduler.get_stage_parameters(stage, progress)
                print(f"   阶段{stage} 进度{progress:.1f}: {params['stage_name']}, "
                      f"连通性惩罚={params['connectivity_penalty']:.2f}")
        
        print("✅ 参数调度器测试成功")
        
        # 5. 测试安全监控器
        print("\n5️⃣ 测试安全监控器...")
        safety_config = SafetyConfig()
        monitor = SafetyMonitor(safety_config)
        
        # 测试正常性能
        normal_performance = {'episode_length': 20, 'reward': 5.0, 'composite_score': 0.8}
        safety_status = monitor.check_training_safety(normal_performance)
        print(f"   正常性能: 安全={safety_status['is_safe']}, 紧急模式={safety_status['emergency_mode']}")
        
        # 测试异常性能
        abnormal_performance = {'episode_length': 2, 'reward': -200.0, 'composite_score': 0.1}
        safety_status = monitor.check_training_safety(abnormal_performance)
        print(f"   异常性能: 安全={safety_status['is_safe']}, 紧急模式={safety_status['emergency_mode']}")
        
        if safety_status['emergency_mode']:
            emergency_params = monitor.get_emergency_params()
            print(f"   紧急参数: {emergency_params['stage_name']}")
        
        print("✅ 安全监控器测试成功")
        
        # 6. 测试智能导演
        print("\n6️⃣ 测试智能导演...")
        director = AdaptiveCurriculumDirector(config)
        
        # 模拟训练过程
        for episode in range(10):
            episode_info = {
                'episode': episode,
                'reward': -5.0 + episode * 0.5,  # 逐渐改善
                'episode_length': 10 + episode,
                'success': episode > 5,
                'load_cv': 0.8 - episode * 0.05,
                'coupling_ratio': 0.6 - episode * 0.03,
                'connectivity': 0.7 + episode * 0.02
            }
            
            decision = director.step(episode, episode_info)
            
            if episode % 3 == 0:  # 每3个episode打印一次
                stage_info = decision['stage_info']
                print(f"   Episode {episode}: 阶段{stage_info['current_stage']} "
                      f"({stage_info['stage_name']}), 进度={stage_info['stage_progress']:.2f}")
        
        # 获取最终状态
        final_status = director.get_status_summary()
        print(f"✅ 智能导演测试成功: 最终阶段={final_status['current_stage']}, "
              f"转换次数={len(final_status['transition_history'])}")
        
        print("\n🎉 所有组件测试通过！智能自适应课程学习系统准备就绪。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_training():
    """测试与训练系统的集成"""
    print("\n🔗 测试与训练系统的集成...")
    
    try:
        # 测试配置加载
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查adaptive模式配置
        if 'adaptive' in config:
            adaptive_config = config['adaptive']
            print("✅ adaptive训练模式配置存在")
            
            # 检查关键配置项
            required_keys = ['training', 'adaptive_curriculum', 'scenario_generation']
            for key in required_keys:
                if key in adaptive_config:
                    print(f"   ✓ {key} 配置存在")
                else:
                    print(f"   ✗ {key} 配置缺失")
            
            # 检查智能自适应是否启用
            if adaptive_config.get('adaptive_curriculum', {}).get('enabled', False):
                print("✅ 智能自适应课程学习已启用")
            else:
                print("⚠️ 智能自适应课程学习未启用")
        else:
            print("❌ adaptive训练模式配置不存在")
            return False
        
        # 测试训练脚本导入
        try:
            from train import UnifiedTrainingSystem
            print("✅ 训练系统导入成功")

            # 创建训练系统实例
            system = UnifiedTrainingSystem()
            print("✅ 训练系统实例化成功")
            
        except Exception as e:
            print(f"❌ 训练系统导入失败: {e}")
            return False
        
        print("✅ 集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 智能自适应课程学习系统测试")
    print("=" * 60)
    
    # 测试组件
    component_test_passed = test_adaptive_curriculum_components()
    
    # 测试集成
    integration_test_passed = test_integration_with_training()
    
    # 总结
    print("\n" + "=" * 60)
    if component_test_passed and integration_test_passed:
        print("🎉 所有测试通过！系统准备就绪。")
        print("\n📋 使用方法:")
        print("   python train.py --mode adaptive")
        print("   或者在配置文件中启用 adaptive_curriculum.enabled: true")
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit(main())
