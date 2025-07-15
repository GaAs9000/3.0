"""
UnifiedDirector统一导演系统

解决双重课程学习冲突的核心实现。
统一协调拓扑切换（ScaleAwareSyntheticGenerator）和训练策略调整（AdaptiveDirector）。

核心设计理念：
- 智能导演成为唯一的训练策略控制者
- 多尺度生成器作为导演的"工具"
- 基于性能稳定性而非固定时间进行切换
- 拓扑切换时的参数预热退火机制

作者：Claude & User Collaboration
日期：2025-07-13
版本：2.0 - 统一导演架构
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# =====================================================
# 基础配置类
# =====================================================

@dataclass
class DataMixConfig:
    """数据混合配置 - 控制不同拓扑规模的训练数据分布"""
    
    # 三阶段拓扑规模定义
    small_scale: Dict[str, Any] = field(default_factory=lambda: {
        'node_range': [10, 40],
        'base_cases': ['ieee14', 'ieee30', 'ieee39'],
        'k_range': [3, 4],
        'weight': 1.0
    })
    
    medium_scale: Dict[str, Any] = field(default_factory=lambda: {
        'node_range': [40, 100], 
        'base_cases': ['ieee57', 'ieee118'],
        'k_range': [4, 6, 8],
        'weight': 0.8
    })
    
    large_scale: Dict[str, Any] = field(default_factory=lambda: {
        'node_range': [100, 300],
        'base_cases': ['ieee118'],  # 会合成更大规模
        'k_range': [8, 12, 16],
        'weight': 0.6
    })
    
    # 渐进式训练计划 (进度百分比)
    early_phase_end: float = 0.2      # 前20%专注小规模
    mid_phase_end: float = 0.5        # 20%-50%引入中等规模  
    balance_phase_end: float = 0.8    # 50%-80%平衡发展
    # 80%-100%全规模训练
    
    # 数据混合策略
    smooth_transition: bool = True     # 平滑过渡而非硬切换
    overlap_ratio: float = 0.3        # 阶段重叠比例
    
    def get_current_mix_ratio(self, progress: float) -> Dict[str, float]:
        """根据训练进度获取当前数据混合比例
        
        Args:
            progress: 训练进度 (0.0 - 1.0)
            
        Returns:
            各规模的权重比例字典
        """
        if progress <= self.early_phase_end:
            # 早期阶段：仅小规模
            return {'small': 1.0, 'medium': 0.0, 'large': 0.0}
        
        elif progress <= self.mid_phase_end:
            # 中期阶段：小规模主导，逐步引入中等规模
            transition = (progress - self.early_phase_end) / (self.mid_phase_end - self.early_phase_end)
            if self.smooth_transition:
                small_weight = 1.0 - 0.5 * transition
                medium_weight = 0.5 * transition
            else:
                small_weight = 0.7
                medium_weight = 0.3
            return {'small': small_weight, 'medium': medium_weight, 'large': 0.0}
        
        elif progress <= self.balance_phase_end:
            # 平衡阶段：三种规模平衡发展
            transition = (progress - self.mid_phase_end) / (self.balance_phase_end - self.mid_phase_end)
            if self.smooth_transition:
                small_weight = 0.5 - 0.2 * transition
                medium_weight = 0.5
                large_weight = 0.2 * transition
            else:
                small_weight = 0.4
                medium_weight = 0.4  
                large_weight = 0.2
            return {'small': small_weight, 'medium': medium_weight, 'large': large_weight}
        
        else:
            # 后期阶段：全规模训练，重点大规模
            if self.smooth_transition:
                return {'small': 0.2, 'medium': 0.3, 'large': 0.5}
            else:
                return {'small': 0.1, 'medium': 0.3, 'large': 0.6}


@dataclass  
class AdaptiveThreshold:
    """自适应阈值配置 - 控制切换时机的判断标准"""
    
    # 性能稳定性判断
    stability_window: int = 40         # 稳定性检测窗口大小
    stability_threshold: float = 0.8   # 稳定性阈值
    min_improvement_rate: float = 0.005  # 最小改进率阈值
    
    # 进度控制
    min_episodes_per_stage: int = 200   # 每阶段最少episode数
    max_episodes_per_stage: int = 1000  # 每阶段最多episode数
    
    # 质量门槛
    quality_threshold: float = 0.4      # 质量分数门槛
    connectivity_threshold: float = 0.9  # 连通性阈值
    
    # 切换条件权重
    performance_weight: float = 0.4     # 性能指标权重
    stability_weight: float = 0.3       # 稳定性权重
    progress_weight: float = 0.3        # 进度权重
    
    # 安全保护
    emergency_fallback_enabled: bool = True  # 启用紧急回退
    performance_deterioration_threshold: float = 0.2  # 性能恶化阈值
    max_failed_transitions: int = 3     # 最大失败切换次数
    
    def should_transition(self, performance_metrics: Dict[str, Any], 
                         current_stage_episodes: int) -> Tuple[bool, str]:
        """判断是否应该进行阶段切换
        
        Args:
            performance_metrics: 性能指标字典
            current_stage_episodes: 当前阶段已训练episode数
            
        Returns:
            (是否切换, 切换原因)
        """
        reasons = []
        scores = []
        
        # 1. 检查最小episode要求
        if current_stage_episodes < self.min_episodes_per_stage:
            return False, f"未达到最小episode要求 ({current_stage_episodes}/{self.min_episodes_per_stage})"
        
        # 2. 检查最大episode限制
        if current_stage_episodes >= self.max_episodes_per_stage:
            return True, f"达到最大episode限制 ({current_stage_episodes}/{self.max_episodes_per_stage})"
        
        # 3. 性能指标评估
        performance_score = 0.0
        if 'avg_reward' in performance_metrics:
            avg_reward = performance_metrics['avg_reward']
            if avg_reward > -1.0:  # 合理的性能水平
                performance_score = min(1.0, (avg_reward + 5.0) / 5.0)  # 归一化到[0,1]
        
        if 'quality_score' in performance_metrics:
            quality_score = performance_metrics['quality_score']
            if quality_score > self.quality_threshold:
                performance_score = max(performance_score, quality_score)
        
        scores.append(('performance', performance_score, self.performance_weight))
        if performance_score > 0.5:
            reasons.append(f"性能良好 ({performance_score:.3f})")
        
        # 4. 稳定性评估
        stability_score = 0.0
        if 'stability_confidence' in performance_metrics:
            stability_confidence = performance_metrics['stability_confidence']
            if stability_confidence > self.stability_threshold:
                stability_score = stability_confidence
                reasons.append(f"性能稳定 ({stability_confidence:.3f})")
        
        scores.append(('stability', stability_score, self.stability_weight))
        
        # 5. 进度评估
        progress_score = min(1.0, current_stage_episodes / (self.min_episodes_per_stage * 2))
        scores.append(('progress', progress_score, self.progress_weight))
        if progress_score > 0.7:
            reasons.append(f"训练充分 ({current_stage_episodes} episodes)")
        
        # 6. 综合评分
        weighted_score = sum(score * weight for _, score, weight in scores)
        
        # 7. 切换决策
        should_switch = weighted_score > 0.6 and len(reasons) >= 2
        
        if should_switch:
            reason = f"综合评分 {weighted_score:.3f} > 0.6, 满足条件: {', '.join(reasons)}"
        else:
            reason = f"综合评分 {weighted_score:.3f} < 0.6, 需要更多训练"
        
        return should_switch, reason


# =====================================================
# 核心监控组件
# =====================================================

class PerformanceMonitor:
    """性能监控器 - 跟踪训练性能和稳定性指标"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """重置监控状态"""
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size) 
        self.quality_scores = deque(maxlen=self.window_size)
        self.connectivity_rates = deque(maxlen=self.window_size)
        self.last_update_time = time.time()
        
    def update(self, episode_info: Dict[str, Any]):
        """更新性能指标

        Args:
            episode_info: episode信息字典
        """
        self.episode_rewards.append(episode_info.get('reward', 0.0))
        self.episode_lengths.append(episode_info.get('episode_length', 0))

        # 质量指标
        quality_score = episode_info.get('quality_score', 0.0)
        if 'reward_components' in episode_info:
            components = episode_info['reward_components']
            quality_score = components.get('quality_score', quality_score)
        self.quality_scores.append(quality_score)

        # 连通性 - 添加调试信息
        connectivity = episode_info.get('connectivity', 0.0)

        # 调试：检查连通性数据来源
        if 'metrics' in episode_info:
            metrics_connectivity = episode_info['metrics'].get('connectivity', None)
            if metrics_connectivity is not None:
                connectivity = metrics_connectivity
                logger.debug(f"从metrics获取连通性: {connectivity:.3f}")

        logger.debug(f"Episode连通性数据: {connectivity:.3f}")
        self.connectivity_rates.append(connectivity)

        self.last_update_time = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前性能指标
        
        Returns:
            性能指标字典
        """
        if not self.episode_rewards:
            return {}
        
        rewards = list(self.episode_rewards)
        lengths = list(self.episode_lengths)
        qualities = list(self.quality_scores)
        connectivities = list(self.connectivity_rates)
        
        # 基础统计
        metrics = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_length': np.mean(lengths),
            'avg_quality': np.mean(qualities),
            'avg_connectivity': np.mean(connectivities),
            'sample_count': len(rewards)
        }
        
        # 稳定性指标
        if len(rewards) >= 10:
            recent_rewards = rewards[-10:]
            earlier_rewards = rewards[:-10] if len(rewards) > 10 else rewards[:5]
            
            # 变异系数（越小越稳定）
            cv = np.std(recent_rewards) / (np.abs(np.mean(recent_rewards)) + 1e-8)
            metrics['stability_cv'] = cv
            
            # 改进趋势
            recent_avg = np.mean(recent_rewards)
            earlier_avg = np.mean(earlier_rewards) 
            improvement = (recent_avg - earlier_avg) / (np.abs(earlier_avg) + 1e-8)
            metrics['improvement_rate'] = improvement
            
            # 稳定性置信度
            stability_confidence = max(0.0, 1.0 - cv) * (1.0 + max(0.0, improvement))
            metrics['stability_confidence'] = min(1.0, stability_confidence)
        
        return metrics
    
    def is_stable(self, threshold: float = 0.8) -> bool:
        """判断性能是否稳定
        
        Args:
            threshold: 稳定性阈值
            
        Returns:
            是否稳定
        """
        metrics = self.get_metrics()
        return metrics.get('stability_confidence', 0.0) >= threshold


class SafetyMonitor:
    """安全监控器 - 监控训练安全性和异常情况"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reset()
    
    def reset(self):
        """重置安全监控状态"""
        self.failed_transitions = 0
        self.emergency_activations = 0
        self.last_safe_state = None
        self.alert_history = deque(maxlen=100)
        self.start_time = time.time()
        
    def check_safety(self, metrics: Dict[str, Any], current_stage: str) -> Tuple[bool, List[str]]:
        """检查训练安全性
        
        Args:
            metrics: 性能指标
            current_stage: 当前训练阶段
            
        Returns:
            (是否安全, 警告信息列表)
        """
        warnings = []
        is_safe = True
        
        # 1. 检查性能恶化
        avg_reward = metrics.get('avg_reward', 0.0)
        if avg_reward < -10.0:  # 性能严重恶化
            warnings.append(f"性能严重恶化: 平均奖励 {avg_reward:.3f}")
            is_safe = False
        
        # 2. 检查训练崩溃指标
        if 'stability_cv' in metrics:
            cv = metrics['stability_cv']
            if cv > 2.0:  # 变异系数过大
                warnings.append(f"训练不稳定: CV={cv:.3f}")
        
        # 3. 检查连通性失效 - 修复检测逻辑
        connectivity = metrics.get('avg_connectivity', 1.0)

        # 调试信息：记录连通性值
        logger.debug(f"当前平均连通性: {connectivity:.3f}")

        # 只有在连通性确实很低且有足够样本时才报警
        sample_count = metrics.get('sample_count', 0)
        if sample_count >= 5 and connectivity < 0.3:  # 需要至少5个样本，阈值降低到0.3
            warnings.append(f"连通性下降: {connectivity:.3f}")
            # 不设置is_safe=False，只是警告，不触发紧急恢复
        
        # 4. 检查失败切换次数
        if self.failed_transitions >= self.config.get('max_failed_transitions', 3):
            warnings.append(f"切换失败次数过多: {self.failed_transitions}")
            is_safe = False
        
        # 记录警告
        if warnings:
            alert = {
                'timestamp': time.time(),
                'stage': current_stage,
                'warnings': warnings,
                'metrics': metrics
            }
            self.alert_history.append(alert)
        
        return is_safe, warnings
    
    def record_failed_transition(self):
        """记录失败的阶段切换"""
        self.failed_transitions += 1
        logger.warning(f"记录失败切换，总计: {self.failed_transitions}")
    
    def record_emergency_activation(self):
        """记录紧急恢复激活"""
        self.emergency_activations += 1
        logger.error(f"紧急恢复激活，总计: {self.emergency_activations}")
    
    def save_safe_state(self, state: Dict[str, Any]):
        """保存安全状态用于回滚"""
        self.last_safe_state = {
            'timestamp': time.time(),
            'state': state.copy()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取安全监控状态"""
        return {
            'failed_transitions': self.failed_transitions,
            'emergency_activations': self.emergency_activations,
            'total_alerts': len(self.alert_history),
            'recent_alerts': list(self.alert_history)[-5:],  # 最近5个警告
            'has_safe_state': self.last_safe_state is not None,
            'uptime': time.time() - self.start_time
        }


class CheckpointManager:
    """检查点管理器 - 管理训练状态的保存和恢复"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_history = []
    
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_id: str) -> str:
        """保存检查点
        
        Args:
            state: 要保存的状态字典
            checkpoint_id: 检查点标识符
            
        Returns:
            检查点文件路径
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"unified_director_checkpoint_{checkpoint_id}_{timestamp}.json"
        filepath = self.checkpoint_dir / filename
        
        checkpoint_data = {
            'timestamp': timestamp,
            'checkpoint_id': checkpoint_id,
            'state': state,
            'metadata': {
                'save_time': time.time(),
                'version': '2.0'
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            self.checkpoint_history.append({
                'checkpoint_id': checkpoint_id,
                'filepath': str(filepath),
                'timestamp': timestamp
            })
            
            logger.info(f"检查点已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_id: str = None) -> Optional[Dict[str, Any]]:
        """加载检查点
        
        Args:
            checkpoint_id: 检查点标识符，None表示加载最新的
            
        Returns:
            检查点状态字典
        """
        try:
            if checkpoint_id is None:
                # 加载最新检查点
                if not self.checkpoint_history:
                    return None
                latest = max(self.checkpoint_history, key=lambda x: x['timestamp'])
                filepath = latest['filepath']
            else:
                # 加载指定检查点
                matching = [cp for cp in self.checkpoint_history if cp['checkpoint_id'] == checkpoint_id]
                if not matching:
                    return None
                filepath = matching[-1]['filepath']  # 最新的匹配项
            
            with open(filepath, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"检查点已加载: {filepath}")
            return checkpoint_data['state']
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, str]]:
        """列出所有可用的检查点"""
        return self.checkpoint_history.copy()
    
    def cleanup_old_checkpoints(self, keep_count: int = 10):
        """清理旧的检查点文件，保留最新的几个
        
        Args:
            keep_count: 保留的检查点数量
        """
        if len(self.checkpoint_history) <= keep_count:
            return
        
        # 按时间戳排序，删除旧的
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['timestamp'])
        to_delete = sorted_checkpoints[:-keep_count]
        
        deleted_count = 0
        for checkpoint in to_delete:
            try:
                filepath = Path(checkpoint['filepath'])
                if filepath.exists():
                    filepath.unlink()
                    deleted_count += 1
                self.checkpoint_history.remove(checkpoint)
            except Exception as e:
                logger.warning(f"删除检查点失败 {checkpoint['filepath']}: {e}")
        
        if deleted_count > 0:
            logger.info(f"已清理 {deleted_count} 个旧检查点")


# =====================================================
# 拓扑切换和参数协调组件
# =====================================================

class TopologyTransitionManager:
    """拓扑切换管理器 - 协调拓扑规模的智能切换"""
    
    def __init__(self, config: DataMixConfig):
        self.config = config
        self.current_stage = 'small'  # 初始阶段
        self.stage_episodes = 0
        self.transition_history = []
        
    def get_current_topology_spec(self, progress: float) -> Dict[str, Any]:
        """根据进度获取当前拓扑规格
        
        Args:
            progress: 训练进度 (0.0 - 1.0)
            
        Returns:
            拓扑规格字典
        """
        mix_ratio = self.config.get_current_mix_ratio(progress)
        
        # 根据权重确定主导规模
        dominant_scale = max(mix_ratio.items(), key=lambda x: x[1])[0]
        
        if dominant_scale == 'small':
            spec = self.config.small_scale.copy()
            self.current_stage = 'small'
        elif dominant_scale == 'medium':
            spec = self.config.medium_scale.copy() 
            self.current_stage = 'medium'
        else:
            spec = self.config.large_scale.copy()
            self.current_stage = 'large'
        
        # 添加混合权重信息
        spec['mix_ratio'] = mix_ratio
        spec['dominant_scale'] = dominant_scale
        
        return spec
    
    def should_transition_topology(self, current_progress: float, 
                                 performance_metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """判断是否应该切换拓扑
        
        Args:
            current_progress: 当前训练进度
            performance_metrics: 性能指标
            
        Returns:
            (是否切换, 切换原因)
        """
        new_spec = self.get_current_topology_spec(current_progress)
        new_stage = new_spec['dominant_scale']
        
        # 如果阶段没有变化，不需要切换
        if new_stage == self.current_stage:
            return False, f"保持当前阶段: {self.current_stage}"
        
        # 检查是否满足切换条件
        min_episodes = 150  # 最小阶段episode数
        if self.stage_episodes < min_episodes:
            return False, f"当前阶段episode不足: {self.stage_episodes}/{min_episodes}"
        
        # 检查性能稳定性
        stability = performance_metrics.get('stability_confidence', 0.0)
        if stability < 0.6:
            return False, f"性能不够稳定: {stability:.3f} < 0.6"
        
        return True, f"切换 {self.current_stage} -> {new_stage}, 稳定性: {stability:.3f}"
    
    def execute_transition(self, new_stage: str) -> Dict[str, Any]:
        """执行拓扑切换
        
        Args:
            new_stage: 新的拓扑阶段
            
        Returns:
            切换信息字典
        """
        old_stage = self.current_stage
        transition_info = {
            'timestamp': time.time(),
            'old_stage': old_stage,
            'new_stage': new_stage,
            'old_stage_episodes': self.stage_episodes,
            'transition_type': 'planned'
        }
        
        self.transition_history.append(transition_info)
        self.current_stage = new_stage
        self.stage_episodes = 0  # 重置阶段计数器
        
        logger.info(f"拓扑切换完成: {old_stage} -> {new_stage}")
        return transition_info


class ParameterCoordinator:
    """参数协调器 - 实现拓扑切换时的参数预热退火机制"""
    
    def __init__(self, adaptive_config: Dict[str, Any]):
        self.adaptive_config = adaptive_config
        self.transition_active = False
        self.transition_start_episode = 0
        self.original_params = {}
        self.target_params = {}
        self.warmup_episodes = 0
        
    def prepare_transition(self, old_stage: str, new_stage: str, 
                         current_episode: int) -> Dict[str, Any]:
        """准备参数切换
        
        Args:
            old_stage: 原阶段
            new_stage: 新阶段  
            current_episode: 当前episode数
            
        Returns:
            预热参数配置
        """
        self.transition_active = True
        self.transition_start_episode = current_episode
        
        # 根据阶段规模确定预热参数
        if new_stage == 'small':
            self.warmup_episodes = 20
            self.target_params = {
                'connectivity_penalty': 0.5,
                'learning_rate_factor': 1.2,
                'entropy_coef': 0.02,
                'reward_weights': {
                    'load_b': 0.5,
                    'decoupling': 0.3,
                    'power_b': 0.2
                }
            }
        elif new_stage == 'medium':
            self.warmup_episodes = 40  
            self.target_params = {
                'connectivity_penalty': 0.8,
                'learning_rate_factor': 1.0,
                'entropy_coef': 0.015,
                'reward_weights': {
                    'load_b': 0.4,
                    'decoupling': 0.4, 
                    'power_b': 0.2
                }
            }
        else:  # large
            self.warmup_episodes = 60
            self.target_params = {
                'connectivity_penalty': 1.2,
                'learning_rate_factor': 0.8,
                'entropy_coef': 0.01,
                'reward_weights': {
                    'load_b': 0.3,
                    'decoupling': 0.5,
                    'power_b': 0.2
                }
            }
        
        logger.info(f"参数预热准备完成: {old_stage} -> {new_stage}, 预热{self.warmup_episodes}轮")
        return self.target_params.copy()
    
    def get_current_params(self, current_episode: int) -> Dict[str, Any]:
        """获取当前应使用的参数
        
        Args:
            current_episode: 当前episode数
            
        Returns:
            当前参数配置
        """
        if not self.transition_active:
            return {}
        
        episodes_since_transition = current_episode - self.transition_start_episode
        
        if episodes_since_transition >= self.warmup_episodes:
            # 预热完成，使用目标参数
            self.transition_active = False
            return self.target_params.copy()
        
        # 预热期间，线性插值
        progress = episodes_since_transition / self.warmup_episodes
        current_params = {}
        
        for key, target_value in self.target_params.items():
            if key == 'reward_weights':
                # 嵌套字典的处理
                current_params[key] = {}
                for sub_key, sub_target in target_value.items():
                    original_value = self.original_params.get(key, {}).get(sub_key, sub_target)
                    current_value = original_value + (sub_target - original_value) * progress
                    current_params[key][sub_key] = current_value
            else:
                original_value = self.original_params.get(key, target_value)
                current_value = original_value + (target_value - original_value) * progress
                current_params[key] = current_value
        
        return current_params
    
    def set_original_params(self, params: Dict[str, Any]):
        """设置原始参数作为插值起点"""
        self.original_params = params.copy()


# =====================================================
# 统一导演核心类
# =====================================================

class UnifiedDirector:
    """统一导演系统 - 协调拓扑切换和训练策略调整的核心控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化统一导演系统
        
        Args:
            config: 完整的训练配置字典
        """
        self.config = config
        self.unified_config = config.get('unified_director', {})
        
        # 初始化核心组件
        self.data_mix_config = DataMixConfig(**self.unified_config.get('data_mix', {}))
        self.adaptive_threshold = AdaptiveThreshold(**self.unified_config.get('adaptive_threshold', {}))
        
        # 初始化监控和管理组件
        self.performance_monitor = PerformanceMonitor(
            window_size=self.unified_config.get('performance_window', 50)
        )
        self.safety_monitor = SafetyMonitor(self.unified_config.get('safety', {}))
        
        checkpoint_dir = self.unified_config.get('checkpoint_dir', 'data/latest/unified_checkpoints')
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 初始化协调组件
        self.topology_manager = TopologyTransitionManager(self.data_mix_config)
        self.parameter_coordinator = ParameterCoordinator(
            self.config.get('adaptive_curriculum', {})
        )
        
        # 导演状态
        self.total_episodes = 0
        self.current_episode = 0
        self.last_transition_episode = 0
        self.director_decisions = []
        self.enabled = self.unified_config.get('enabled', True)
        
        # 集成现有组件
        self.adaptive_director = None  # 将在运行时注入
        self.scale_generator = None    # 将在运行时注入
        
        logger.info("UnifiedDirector 初始化完成")
    
    def set_total_episodes(self, total_episodes: int):
        """设置总episode数用于进度计算"""
        self.total_episodes = total_episodes
    
    def coordinate_with_scale_generator(self, episode: int, topology_spec: Dict[str, Any]) -> Dict[str, Any]:
        """与多尺度生成器协调，控制其生成策略
        
        Args:
            episode: 当前episode编号
            topology_spec: 拓扑规格字典
            
        Returns:
            生成器配置字典
        """
        if not self.scale_generator:
            return {'status': 'generator_not_available'}
        
        current_stage = topology_spec.get('dominant_scale', 'small')
        mix_ratio = topology_spec.get('mix_ratio', {'small': 1.0, 'medium': 0.0, 'large': 0.0})
        
        # 根据当前阶段配置生成器
        generator_config = {
            'dominant_scale': current_stage,
            'mix_ratio': mix_ratio,
            'episode': episode,
            'transition_active': self.parameter_coordinator.transition_active
        }
        
        # 阶段特定的生成配置
        if current_stage == 'small':
            generator_config.update({
                'preferred_node_range': [10, 40],
                'preferred_cases': ['ieee14', 'ieee30'],
                'topology_variation_prob': 0.1,  # 小规模时减少拓扑变化
                'load_variation_intensity': 0.8
            })
        elif current_stage == 'medium':
            generator_config.update({
                'preferred_node_range': [40, 100],
                'preferred_cases': ['ieee57'],
                'topology_variation_prob': 0.2,  # 中等规模适中变化
                'load_variation_intensity': 1.0
            })
        else:  # large
            generator_config.update({
                'preferred_node_range': [100, 300],
                'preferred_cases': ['ieee118'],
                'topology_variation_prob': 0.3,  # 大规模时更多拓扑变化
                'load_variation_intensity': 1.2
            })
        
        # 如果正在进行拓扑切换，暂时降低变化强度
        if self.parameter_coordinator.transition_active:
            generator_config['topology_variation_prob'] *= 0.5
            generator_config['load_variation_intensity'] *= 0.8
            generator_config['transition_mode'] = True
        
        try:
            # 尝试更新生成器配置（如果生成器支持）
            if hasattr(self.scale_generator, 'update_generation_config'):
                self.scale_generator.update_generation_config(generator_config)
                logger.info(f"已更新多尺度生成器配置: {current_stage}阶段")
            
            return {'status': 'success', 'config': generator_config}
            
        except Exception as e:
            logger.warning(f"协调多尺度生成器失败: {e}")
            return {'status': 'error', 'error': str(e), 'config': generator_config}
    
    def integrate_components(self, adaptive_director=None, scale_generator=None):
        """集成现有的自适应导演和多尺度生成器
        
        Args:
            adaptive_director: AdaptiveDirector实例
            scale_generator: ScaleAwareSyntheticGenerator实例
        """
        self.adaptive_director = adaptive_director
        self.scale_generator = scale_generator
        
        if adaptive_director:
            logger.info("已集成 AdaptiveDirector")
        if scale_generator:
            logger.info("已集成 ScaleAwareSyntheticGenerator")
            
            # 如果生成器支持，设置初始配置
            if hasattr(scale_generator, 'set_unified_director'):
                scale_generator.set_unified_director(self)
                logger.info("已在多尺度生成器中设置统一导演引用")
    
    def step(self, episode: int, episode_info: Dict[str, Any]) -> Dict[str, Any]:
        """统一导演的主要决策步骤
        
        Args:
            episode: 当前episode编号
            episode_info: episode信息字典
            
        Returns:
            统一导演决策字典
        """
        if not self.enabled:
            return self._fallback_to_adaptive_director(episode, episode_info)
        
        self.current_episode = episode
        current_progress = episode / self.total_episodes if self.total_episodes > 0 else 0.0
        
        # 1. 更新性能监控
        self.performance_monitor.update(episode_info)
        metrics = self.performance_monitor.get_metrics()
        
        # 2. 安全检查
        is_safe, warnings = self.safety_monitor.check_safety(metrics, self.topology_manager.current_stage)
        if not is_safe:
            return self._handle_safety_issue(warnings, episode_info)
        
        # 3. 拓扑切换决策
        topology_decision = self._evaluate_topology_transition(current_progress, metrics)
        
        # 4. 协调多尺度生成器（如果有拓扑变化）
        generator_coordination = {}
        if topology_decision.get('should_transition', False) or episode % 50 == 0:  # 定期协调
            current_spec = self.topology_manager.get_current_topology_spec(current_progress)
            generator_coordination = self.coordinate_with_scale_generator(episode, current_spec)
        
        # 5. 参数协调
        parameter_adjustments = self._coordinate_parameters(episode, topology_decision)
        
        # 6. 集成自适应导演决策
        adaptive_decision = self._integrate_adaptive_director(episode, episode_info, metrics)
        
        # 7. 生成统一决策
        unified_decision = {
            'timestamp': time.time(),
            'episode': episode,
            'progress': current_progress,
            'current_stage': self.topology_manager.current_stage,
            'topology_decision': topology_decision,
            'generator_coordination': generator_coordination,  # 新增生成器协调信息
            'parameter_adjustments': parameter_adjustments,
            'adaptive_decision': adaptive_decision,
            'performance_metrics': metrics,
            'safety_status': {'is_safe': is_safe, 'warnings': warnings},
            'coordinator_status': self._get_coordinator_status()
        }
        
        # 8. 记录决策历史
        self.director_decisions.append(unified_decision)
        self.topology_manager.stage_episodes += 1
        
        # 9. 定期保存检查点
        if episode % 200 == 0:
            self._save_checkpoint(episode)
        
        return unified_decision
    
    def _evaluate_topology_transition(self, progress: float, 
                                    metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估拓扑切换决策"""
        # 检查是否应该切换拓扑
        should_switch, reason = self.topology_manager.should_transition_topology(progress, metrics)
        
        topology_decision = {
            'should_transition': should_switch,
            'reason': reason,
            'current_stage': self.topology_manager.current_stage,
            'stage_episodes': self.topology_manager.stage_episodes
        }
        
        if should_switch:
            # 执行拓扑切换
            new_spec = self.topology_manager.get_current_topology_spec(progress)
            new_stage = new_spec['dominant_scale']
            
            transition_info = self.topology_manager.execute_transition(new_stage)
            topology_decision['transition_info'] = transition_info
            topology_decision['new_spec'] = new_spec
            
            # 准备参数预热
            self.parameter_coordinator.prepare_transition(
                transition_info['old_stage'], 
                new_stage, 
                self.current_episode
            )
            
            logger.info(f"拓扑切换决策: {transition_info['old_stage']} -> {new_stage}")
        
        return topology_decision
    
    def _coordinate_parameters(self, episode: int, 
                             topology_decision: Dict[str, Any]) -> Dict[str, Any]:
        """协调参数调整"""
        # 获取当前参数（包括预热退火）
        warmup_params = self.parameter_coordinator.get_current_params(episode)
        
        parameter_adjustments = {
            'warmup_active': self.parameter_coordinator.transition_active,
            'warmup_params': warmup_params
        }
        
        # 如果有拓扑切换，记录参数变化
        if topology_decision.get('should_transition', False):
            parameter_adjustments['transition_triggered'] = True
            parameter_adjustments['target_params'] = self.parameter_coordinator.target_params.copy()
        
        return parameter_adjustments
    
    def _integrate_adaptive_director(self, episode: int, episode_info: Dict[str, Any], 
                                   metrics: Dict[str, Any]) -> Dict[str, Any]:
        """集成自适应导演的决策"""
        if not self.adaptive_director:
            return {'status': 'not_available'}
        
        try:
            # 调用原有的自适应导演
            adaptive_decision = self.adaptive_director.step(episode, episode_info)
            
            # 如果统一导演正在进行参数协调，需要协调冲突
            if self.parameter_coordinator.transition_active:
                # 统一导演的参数优先级更高
                warmup_params = self.parameter_coordinator.get_current_params(episode)
                if warmup_params:
                    # 合并参数，统一导演的预热参数覆盖自适应导演的参数
                    for key, value in warmup_params.items():
                        adaptive_decision[key] = value
                    adaptive_decision['unified_override'] = True
            
            return adaptive_decision
            
        except Exception as e:
            logger.warning(f"自适应导演集成失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _handle_safety_issue(self, warnings: List[str], 
                           episode_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理安全问题"""
        logger.error(f"检测到安全问题: {warnings}")
        self.safety_monitor.record_emergency_activation()
        
        # 紧急回滚到安全状态
        safe_state = self.safety_monitor.last_safe_state
        if safe_state:
            logger.info("执行紧急回滚到上次安全状态")
            # 这里可以实现具体的回滚逻辑
            
        return {
            'emergency_mode': True,
            'warnings': warnings,
            'action': 'emergency_fallback',
            'fallback_params': {
                'connectivity_penalty': 0.1,  # 降低约束
                'learning_rate_factor': 0.5,  # 降低学习率
                'entropy_coef': 0.02  # 增加探索
            }
        }
    
    def _fallback_to_adaptive_director(self, episode: int, 
                                     episode_info: Dict[str, Any]) -> Dict[str, Any]:
        """回退到原有的自适应导演"""
        if self.adaptive_director:
            decision = self.adaptive_director.step(episode, episode_info)
            decision['unified_director_status'] = 'disabled_fallback'
            return decision
        else:
            return {'status': 'no_director_available'}
    
    def _get_coordinator_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        return {
            'topology_stage': self.topology_manager.current_stage,
            'stage_episodes': self.topology_manager.stage_episodes,
            'transition_active': self.parameter_coordinator.transition_active,
            'warmup_episodes_remaining': max(0, 
                self.parameter_coordinator.warmup_episodes - 
                (self.current_episode - self.parameter_coordinator.transition_start_episode)
            ) if self.parameter_coordinator.transition_active else 0,
            'total_transitions': len(self.topology_manager.transition_history),
            'safety_alerts': len(self.safety_monitor.alert_history)
        }
    
    def _save_checkpoint(self, episode: int):
        """保存统一导演状态检查点"""
        state = {
            'episode': episode,
            'topology_manager_state': {
                'current_stage': self.topology_manager.current_stage,
                'stage_episodes': self.topology_manager.stage_episodes,
                'transition_history': self.topology_manager.transition_history
            },
            'parameter_coordinator_state': {
                'transition_active': self.parameter_coordinator.transition_active,
                'transition_start_episode': self.parameter_coordinator.transition_start_episode,
                'warmup_episodes': self.parameter_coordinator.warmup_episodes
            },
            'safety_monitor_state': self.safety_monitor.get_status(),
            'performance_metrics': self.performance_monitor.get_metrics(),
            'director_decisions_count': len(self.director_decisions)
        }
        
        self.checkpoint_manager.save_checkpoint(state, f"episode_{episode}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """获取统一导演状态摘要"""
        return {
            'enabled': self.enabled,
            'current_episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'progress': self.current_episode / self.total_episodes if self.total_episodes > 0 else 0.0,
            'coordinator_status': self._get_coordinator_status(),
            'performance_summary': self.performance_monitor.get_metrics(),
            'safety_summary': self.safety_monitor.get_status(),
            'recent_decisions_count': len(self.director_decisions),
            'integration_status': {
                'adaptive_director': self.adaptive_director is not None,
                'scale_generator': self.scale_generator is not None
            }
        }


# =====================================================
# 导出接口
# =====================================================

__all__ = [
    'DataMixConfig',
    'AdaptiveThreshold', 
    'PerformanceMonitor',
    'SafetyMonitor',
    'CheckpointManager',
    'TopologyTransitionManager',
    'ParameterCoordinator', 
    'UnifiedDirector'
]