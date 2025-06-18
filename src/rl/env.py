import torch
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

# Import types that will be defined in other modules
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env import PowerGridPartitionEnv, PartitionMetrics

class CurriculumLearningEnv:
    """
    课程学习环境
    
    渐进式增加任务难度，加速训练收敛
    """
    
    def __init__(self, base_env: 'PowerGridPartitionEnv'):
        self.base_env = base_env
        self.difficulty = 0.0  # 难度等级 [0, 1]
        self.success_history = deque(maxlen=100)
        self.episode_count = 0
        
        # 难度参数
        self.min_preset_ratio = 0.5  # 最简单时预设50%节点
        self.max_constraint_tightness = 2.0  # 最难时约束加倍
    
    def reset(self) -> Dict:
        """重置环境（根据难度调整）"""
        state = self.base_env.reset()
        
        # 根据难度调整初始状态
        if self.difficulty < 0.3:
            # 简单：预分配部分节点
            self._preset_easy_nodes()
        elif self.difficulty > 0.7:
            # 困难：收紧约束
            self._tighten_constraints()
        
        return self.base_env.get_state()
    
    def _preset_easy_nodes(self):
        """预分配容易的节点"""
        preset_ratio = self.min_preset_ratio * (1 - self.difficulty / 0.3)
        num_preset = int(self.base_env.N * preset_ratio)
        
        # 为每个区域预分配一些明显的节点
        for k in range(1, self.base_env.K + 1):
            # 找到种子节点的直接邻居
            seed_mask = (self.base_env.z == k)
            if seed_mask.any():
                seed_idx = torch.where(seed_mask)[0][0].item()
                
                if seed_idx in self.base_env.adj_list:
                    neighbors = self.base_env.adj_list[seed_idx]
                    
                    # 分配部分邻居
                    n_assign = min(len(neighbors), num_preset // self.base_env.K)
                    for neighbor in neighbors[:n_assign]:
                        if self.base_env.z[neighbor] == 0:
                            self.base_env.z[neighbor] = k
        
        # 更新环境状态
        self.base_env._update_state()
    
    def _tighten_constraints(self):
        """收紧约束条件"""
        tightness = 1 + (self.difficulty - 0.7) / 0.3 * (self.max_constraint_tightness - 1)
        
        # 调整奖励权重，增加物理约束的重要性
        self.base_env.reward_weights['power_balance'] *= tightness
        self.base_env.reward_weights['coupling'] *= tightness
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool, Dict]:
        """执行动作"""
        next_state, reward, done, info = self.base_env.step(action)
        
        # 记录成功信息
        if done:
            success = self._evaluate_success(info['metrics'])
            self.success_history.append(success)
            self.episode_count += 1
            
            # 定期更新难度
            if self.episode_count % 10 == 0:
                self._update_difficulty()
        
        return next_state, reward, done, info
    
    def _evaluate_success(self, metrics: 'PartitionMetrics') -> bool:
        """评估是否成功完成任务"""
        # 成功标准（随难度调整）
        cv_threshold = 0.3 + 0.2 * self.difficulty
        coupling_threshold = 2.0 - 0.5 * self.difficulty
        
        return (metrics.load_cv < cv_threshold and 
                metrics.total_coupling < coupling_threshold and
                metrics.connectivity == 1.0)
    
    def _update_difficulty(self):
        """根据成功率更新难度"""
        if len(self.success_history) < 50:
            return
        
        success_rate = np.mean(self.success_history)
        
        # 动态调整难度
        if success_rate > 0.8:
            self.difficulty = min(1.0, self.difficulty + 0.1)
            print(f"📈 难度提升到 {self.difficulty:.2f} (成功率: {success_rate:.2%})")
        elif success_rate < 0.3:
            self.difficulty = max(0.0, self.difficulty - 0.1)
            print(f"📉 难度降低到 {self.difficulty:.2f} (成功率: {success_rate:.2%})")
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """获取有效动作"""
        return self.base_env.get_valid_actions()
    
    def __getattr__(self, name):
        """代理到基础环境的属性"""
        return getattr(self.base_env, name)


# 创建课程学习环境
def initialize_curriculum_env(env):
    """Test function for curriculum learning environment"""
    print("\n📚 创建课程学习环境...")
    curriculum_env = CurriculumLearningEnv(env)
    print("✅ 课程学习环境创建成功！")
    return curriculum_env

