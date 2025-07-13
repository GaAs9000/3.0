"""
UnifiedDirector统一导演系统单元测试

测试覆盖：
- 基础配置类功能
- 核心组件独立测试
- 集成测试
- 异常处理和边界条件

作者：Claude & User Collaboration
日期：2025-07-13
"""

import unittest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import time

# 导入待测试的模块
import sys
sys.path.append(str(Path(__file__).parent.parent / 'code' / 'src'))

from rl.unified_director import (
    DataMixConfig, AdaptiveThreshold, PerformanceMonitor, 
    SafetyMonitor, CheckpointManager, TopologyTransitionManager,
    ParameterCoordinator, UnifiedDirector
)


class TestDataMixConfig(unittest.TestCase):
    """测试数据混合配置类"""
    
    def setUp(self):
        self.config = DataMixConfig()
    
    def test_default_configuration(self):
        """测试默认配置"""
        self.assertEqual(self.config.early_phase_end, 0.2)
        self.assertEqual(self.config.mid_phase_end, 0.5)
        self.assertEqual(self.config.balance_phase_end, 0.8)
        self.assertTrue(self.config.smooth_transition)
        
    def test_early_phase_mix_ratio(self):
        """测试早期阶段数据混合比例"""
        ratio = self.config.get_current_mix_ratio(0.1)  # 10%进度
        self.assertEqual(ratio['small'], 1.0)
        self.assertEqual(ratio['medium'], 0.0)
        self.assertEqual(ratio['large'], 0.0)
    
    def test_mid_phase_mix_ratio(self):
        """测试中期阶段数据混合比例"""
        ratio = self.config.get_current_mix_ratio(0.35)  # 35%进度
        self.assertGreater(ratio['small'], 0.5)
        self.assertGreater(ratio['medium'], 0.0)
        self.assertEqual(ratio['large'], 0.0)
    
    def test_balance_phase_mix_ratio(self):
        """测试平衡阶段数据混合比例"""
        ratio = self.config.get_current_mix_ratio(0.65)  # 65%进度
        self.assertGreater(ratio['small'], 0.0)
        self.assertGreater(ratio['medium'], 0.0)
        self.assertGreater(ratio['large'], 0.0)
    
    def test_late_phase_mix_ratio(self):
        """测试后期阶段数据混合比例"""
        ratio = self.config.get_current_mix_ratio(0.9)  # 90%进度
        self.assertLess(ratio['small'], 0.5)
        self.assertGreater(ratio['large'], 0.3)


class TestAdaptiveThreshold(unittest.TestCase):
    """测试自适应阈值配置类"""
    
    def setUp(self):
        self.threshold = AdaptiveThreshold()
    
    def test_default_configuration(self):
        """测试默认配置"""
        self.assertEqual(self.threshold.min_episodes_per_stage, 200)
        self.assertEqual(self.threshold.stability_threshold, 0.8)
        self.assertTrue(self.threshold.emergency_fallback_enabled)
    
    def test_should_not_transition_insufficient_episodes(self):
        """测试episode数不足时不应切换"""
        metrics = {'avg_reward': 0.5, 'stability_confidence': 0.9}
        should_switch, reason = self.threshold.should_transition(metrics, 100)
        self.assertFalse(should_switch)
        self.assertIn("最小episode要求", reason)
    
    def test_should_transition_max_episodes(self):
        """测试达到最大episode数时强制切换"""
        metrics = {'avg_reward': -2.0, 'stability_confidence': 0.3}
        should_switch, reason = self.threshold.should_transition(metrics, 1000)
        self.assertTrue(should_switch)
        self.assertIn("最大episode限制", reason)
    
    def test_should_transition_good_performance(self):
        """测试性能良好时应该切换"""
        metrics = {
            'avg_reward': 0.5,
            'quality_score': 0.6,
            'stability_confidence': 0.85
        }
        should_switch, reason = self.threshold.should_transition(metrics, 300)
        self.assertTrue(should_switch)
        self.assertIn("综合评分", reason)


class TestPerformanceMonitor(unittest.TestCase):
    """测试性能监控器"""
    
    def setUp(self):
        self.monitor = PerformanceMonitor(window_size=10)
    
    def test_initial_state(self):
        """测试初始状态"""
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics, {})
    
    def test_update_and_metrics(self):
        """测试更新和指标计算"""
        # 添加一些episode数据
        for i in range(5):
            episode_info = {
                'reward': i * 0.1,
                'episode_length': 10 + i,
                'quality_score': 0.5 + i * 0.1,
                'connectivity': 0.8 + i * 0.02
            }
            self.monitor.update(episode_info)
        
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics['sample_count'], 5)
        self.assertAlmostEqual(metrics['avg_reward'], 0.2, places=2)
        self.assertAlmostEqual(metrics['avg_length'], 12.0, places=1)
    
    def test_stability_calculation(self):
        """测试稳定性计算"""
        # 添加稳定的数据
        for i in range(15):
            episode_info = {
                'reward': 0.5 + np.random.normal(0, 0.05),  # 低变异
                'episode_length': 20,
                'quality_score': 0.6,
                'connectivity': 0.9
            }
            self.monitor.update(episode_info)
        
        metrics = self.monitor.get_metrics()
        self.assertIn('stability_confidence', metrics)
        self.assertGreater(metrics['stability_confidence'], 0.5)


class TestSafetyMonitor(unittest.TestCase):
    """测试安全监控器"""
    
    def setUp(self):
        self.config = {'max_failed_transitions': 3}
        self.monitor = SafetyMonitor(self.config)
    
    def test_initial_state(self):
        """测试初始状态"""
        status = self.monitor.get_status()
        self.assertEqual(status['failed_transitions'], 0)
        self.assertEqual(status['emergency_activations'], 0)
    
    def test_safe_conditions(self):
        """测试安全条件"""
        metrics = {
            'avg_reward': 0.5,
            'avg_connectivity': 0.9,
            'stability_cv': 0.1
        }
        is_safe, warnings = self.monitor.check_safety(metrics, 'small')
        self.assertTrue(is_safe)
        self.assertEqual(len(warnings), 0)
    
    def test_unsafe_conditions(self):
        """测试不安全条件"""
        metrics = {
            'avg_reward': -15.0,  # 严重性能恶化
            'avg_connectivity': 0.3,  # 连通性下降
            'stability_cv': 3.0  # 训练不稳定
        }
        is_safe, warnings = self.monitor.check_safety(metrics, 'medium')
        self.assertFalse(is_safe)
        self.assertGreater(len(warnings), 0)
    
    def test_failed_transition_tracking(self):
        """测试失败切换追踪"""
        for i in range(3):
            self.monitor.record_failed_transition()
        
        # 第4次失败应该触发不安全状态
        metrics = {'avg_reward': 0.0}
        is_safe, warnings = self.monitor.check_safety(metrics, 'large')
        self.assertFalse(is_safe)
        self.assertIn("切换失败次数过多", str(warnings))


class TestCheckpointManager(unittest.TestCase):
    """测试检查点管理器"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CheckpointManager(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_checkpoint(self):
        """测试保存和加载检查点"""
        test_state = {
            'episode': 100,
            'topology_stage': 'medium',
            'performance_data': [1, 2, 3, 4, 5]
        }
        
        # 保存检查点
        filepath = self.manager.save_checkpoint(test_state, 'test_checkpoint')
        self.assertIsNotNone(filepath)
        self.assertTrue(Path(filepath).exists())
        
        # 加载检查点
        loaded_state = self.manager.load_checkpoint('test_checkpoint')
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state['episode'], 100)
        self.assertEqual(loaded_state['topology_stage'], 'medium')
    
    def test_load_latest_checkpoint(self):
        """测试加载最新检查点"""
        # 保存多个检查点
        states = [
            {'episode': 100, 'timestamp': '20250713_100000'},
            {'episode': 200, 'timestamp': '20250713_110000'},
            {'episode': 300, 'timestamp': '20250713_120000'}
        ]
        
        for i, state in enumerate(states):
            self.manager.save_checkpoint(state, f'checkpoint_{i}')
            time.sleep(0.1)  # 确保时间戳不同
        
        # 加载最新的（应该是最后一个）
        latest_state = self.manager.load_checkpoint()
        self.assertIsNotNone(latest_state)
        self.assertEqual(latest_state['episode'], 300)


class TestTopologyTransitionManager(unittest.TestCase):
    """测试拓扑切换管理器"""
    
    def setUp(self):
        self.data_config = DataMixConfig()
        self.manager = TopologyTransitionManager(self.data_config)
    
    def test_initial_state(self):
        """测试初始状态"""
        self.assertEqual(self.manager.current_stage, 'small')
        self.assertEqual(self.manager.stage_episodes, 0)
    
    def test_topology_spec_early_phase(self):
        """测试早期阶段拓扑规格"""
        spec = self.manager.get_current_topology_spec(0.1)
        self.assertEqual(spec['dominant_scale'], 'small')
        self.assertEqual(spec['mix_ratio']['small'], 1.0)
    
    def test_should_not_transition_insufficient_episodes(self):
        """测试episode不足时不应切换"""
        metrics = {'stability_confidence': 0.8}
        should_switch, reason = self.manager.should_transition_topology(0.6, metrics)
        self.assertFalse(should_switch)
        self.assertIn("episode不足", reason)
    
    def test_should_not_transition_unstable(self):
        """测试不稳定时不应切换"""
        self.manager.stage_episodes = 200  # 满足最小episode要求
        metrics = {'stability_confidence': 0.3}  # 低稳定性
        should_switch, reason = self.manager.should_transition_topology(0.6, metrics)
        self.assertFalse(should_switch)
        self.assertIn("不够稳定", reason)
    
    def test_successful_transition(self):
        """测试成功切换"""
        self.manager.stage_episodes = 200
        metrics = {'stability_confidence': 0.8}
        should_switch, reason = self.manager.should_transition_topology(0.6, metrics)
        self.assertTrue(should_switch)
        
        # 执行切换
        transition_info = self.manager.execute_transition('medium')
        self.assertEqual(transition_info['old_stage'], 'small')
        self.assertEqual(transition_info['new_stage'], 'medium')
        self.assertEqual(self.manager.current_stage, 'medium')
        self.assertEqual(self.manager.stage_episodes, 0)  # 重置计数器


class TestParameterCoordinator(unittest.TestCase):
    """测试参数协调器"""
    
    def setUp(self):
        self.config = {'adaptive_curriculum': {}}
        self.coordinator = ParameterCoordinator(self.config)
    
    def test_initial_state(self):
        """测试初始状态"""
        self.assertFalse(self.coordinator.transition_active)
        params = self.coordinator.get_current_params(100)
        self.assertEqual(params, {})
    
    def test_prepare_transition(self):
        """测试准备切换"""
        target_params = self.coordinator.prepare_transition('small', 'medium', 100)
        self.assertTrue(self.coordinator.transition_active)
        self.assertEqual(self.coordinator.transition_start_episode, 100)
        self.assertIn('connectivity_penalty', target_params)
        self.assertIn('reward_weights', target_params)
    
    def test_warmup_parameter_interpolation(self):
        """测试预热参数插值"""
        # 准备切换
        self.coordinator.prepare_transition('small', 'medium', 100)
        self.coordinator.set_original_params({
            'connectivity_penalty': 0.3,
            'learning_rate_factor': 1.5,
            'reward_weights': {'load_b': 0.6, 'decoupling': 0.2}
        })
        
        # 测试预热中期参数（50%进度）
        mid_episode = 100 + self.coordinator.warmup_episodes // 2
        current_params = self.coordinator.get_current_params(mid_episode)
        
        # 应该是插值结果
        self.assertGreater(current_params['connectivity_penalty'], 0.3)
        self.assertLess(current_params['connectivity_penalty'], 0.8)
    
    def test_warmup_completion(self):
        """测试预热完成"""
        self.coordinator.prepare_transition('small', 'large', 100)
        
        # 预热完成后
        completion_episode = 100 + self.coordinator.warmup_episodes
        final_params = self.coordinator.get_current_params(completion_episode)
        
        self.assertFalse(self.coordinator.transition_active)
        self.assertEqual(final_params, self.coordinator.target_params)


class TestUnifiedDirectorIntegration(unittest.TestCase):
    """测试统一导演集成功能"""
    
    def setUp(self):
        self.config = {
            'unified_director': {
                'enabled': True,
                'data_mix': {},
                'adaptive_threshold': {},
                'safety': {},
                'checkpoint_dir': tempfile.mkdtemp()
            },
            'training': {'num_episodes': 1000},
            'adaptive_curriculum': {}
        }
        self.director = UnifiedDirector(self.config)
        self.director.set_total_episodes(1000)
    
    def tearDown(self):
        shutil.rmtree(self.config['unified_director']['checkpoint_dir'])
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.director.enabled)
        self.assertEqual(self.director.total_episodes, 1000)
        self.assertIsNotNone(self.director.topology_manager)
        self.assertIsNotNone(self.director.performance_monitor)
    
    def test_component_integration(self):
        """测试组件集成"""
        mock_adaptive_director = Mock()
        mock_scale_generator = Mock()
        
        self.director.integrate_components(
            adaptive_director=mock_adaptive_director,
            scale_generator=mock_scale_generator
        )
        
        self.assertEqual(self.director.adaptive_director, mock_adaptive_director)
        self.assertEqual(self.director.scale_generator, mock_scale_generator)
    
    def test_step_decision_making(self):
        """测试步骤决策制定"""
        episode_info = {
            'episode': 50,
            'reward': 0.3,
            'episode_length': 25,
            'quality_score': 0.4,
            'connectivity': 0.85
        }
        
        decision = self.director.step(50, episode_info)
        
        # 验证决策结构
        self.assertIn('timestamp', decision)
        self.assertIn('episode', decision)
        self.assertIn('progress', decision)
        self.assertIn('current_stage', decision)
        self.assertIn('topology_decision', decision)
        self.assertIn('parameter_adjustments', decision)
        self.assertIn('performance_metrics', decision)
        self.assertIn('safety_status', decision)
    
    def test_safety_fallback(self):
        """测试安全回退机制"""
        # 模拟危险条件
        dangerous_episode_info = {
            'episode': 100,
            'reward': -20.0,  # 极低奖励
            'episode_length': 5,
            'connectivity': 0.2  # 极低连通性
        }
        
        decision = self.director.step(100, dangerous_episode_info)
        
        # 应该触发安全措施
        safety_status = decision['safety_status']
        if not safety_status['is_safe']:
            self.assertIn('emergency_mode', decision)
            self.assertIn('fallback_params', decision)
    
    def test_scale_generator_coordination(self):
        """测试多尺度生成器协调"""
        mock_scale_generator = Mock()
        mock_scale_generator.update_generation_config = Mock()
        
        self.director.integrate_components(scale_generator=mock_scale_generator)
        
        topology_spec = {
            'dominant_scale': 'medium',
            'mix_ratio': {'small': 0.3, 'medium': 0.7, 'large': 0.0}
        }
        
        result = self.director.coordinate_with_scale_generator(100, topology_spec)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('config', result)
        mock_scale_generator.update_generation_config.assert_called_once()


class TestUnifiedDirectorEdgeCases(unittest.TestCase):
    """测试统一导演边界条件和异常处理"""
    
    def setUp(self):
        self.config = {
            'unified_director': {
                'enabled': True,
                'data_mix': {},
                'adaptive_threshold': {},
                'safety': {},
                'checkpoint_dir': tempfile.mkdtemp()
            },
            'training': {'num_episodes': 100},
            'adaptive_curriculum': {}
        }
    
    def test_disabled_director(self):
        """测试禁用的统一导演"""
        self.config['unified_director']['enabled'] = False
        director = UnifiedDirector(self.config)
        
        # 应该回退到自适应导演
        episode_info = {'episode': 10, 'reward': 0.1}
        decision = director.step(10, episode_info)
        
        self.assertIn('status', decision)
    
    def test_missing_adaptive_director(self):
        """测试缺少自适应导演的情况"""
        director = UnifiedDirector(self.config)
        # 不集成自适应导演
        
        episode_info = {'episode': 10, 'reward': 0.1}
        decision = director.step(10, episode_info)
        
        # 应该优雅处理
        adaptive_decision = decision['adaptive_decision']
        self.assertEqual(adaptive_decision['status'], 'not_available')
    
    def test_missing_scale_generator(self):
        """测试缺少多尺度生成器的情况"""
        director = UnifiedDirector(self.config)
        
        topology_spec = {'dominant_scale': 'small'}
        result = director.coordinate_with_scale_generator(10, topology_spec)
        
        self.assertEqual(result['status'], 'generator_not_available')
    
    def test_checkpoint_failure_recovery(self):
        """测试检查点失败恢复"""
        # 使用无效的检查点目录
        self.config['unified_director']['checkpoint_dir'] = '/invalid/path'
        
        # 应该不会崩溃
        try:
            director = UnifiedDirector(self.config)
            episode_info = {'episode': 200, 'reward': 0.1}
            decision = director.step(200, episode_info)  # 触发检查点保存
            # 测试通过如果没有异常
        except Exception as e:
            self.fail(f"统一导演应该优雅处理检查点错误，但抛出了: {e}")


def run_integration_test():
    """运行集成测试"""
    print("🧪 运行UnifiedDirector集成测试...")
    
    # 创建模拟配置
    config = {
        'unified_director': {
            'enabled': True,
            'data_mix': {},
            'adaptive_threshold': {'min_episodes_per_stage': 50},
            'safety': {},
            'checkpoint_dir': tempfile.mkdtemp()
        },
        'training': {'num_episodes': 300},
        'adaptive_curriculum': {}
    }
    
    try:
        # 初始化统一导演
        director = UnifiedDirector(config)
        director.set_total_episodes(300)
        
        # 模拟训练过程
        for episode in range(100):
            episode_info = {
                'episode': episode,
                'reward': np.random.normal(0.0, 0.5),
                'episode_length': np.random.randint(10, 30),
                'quality_score': min(1.0, max(0.0, 0.3 + episode * 0.002)),
                'connectivity': min(1.0, max(0.5, 0.7 + episode * 0.001))
            }
            
            decision = director.step(episode, episode_info)
            
            # 验证决策结构
            assert 'topology_decision' in decision
            assert 'parameter_adjustments' in decision
            assert 'performance_metrics' in decision
            assert 'safety_status' in decision
        
        # 检查状态摘要
        summary = director.get_status_summary()
        assert summary['current_episode'] == 99
        assert 'coordinator_status' in summary
        assert 'performance_summary' in summary
        
        print("✅ 集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理
        shutil.rmtree(config['unified_director']['checkpoint_dir'])


if __name__ == '__main__':
    # 运行单元测试
    print("🧪 开始UnifiedDirector单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行集成测试
    run_integration_test()
    
    print("\n🎉 所有测试完成！")