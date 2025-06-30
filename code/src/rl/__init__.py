"""
强化学习模块
"""

from .environment import PowerGridPartitioningEnv
from .agent import PPOAgent
from .state import StateManager
from .action_space import ActionSpace, ActionMask
from .reward import RewardFunction, DualLayerRewardFunction
from .utils import MetisInitializer, PartitionEvaluator
from .scenario_generator import ScenarioGenerator

# 可选导入gym_wrapper（如果gym可用）
try:
    from .gym_wrapper import PowerGridPartitionGymEnv, make_parallel_env
    _gym_available = True
except ImportError:
    _gym_available = False
    PowerGridPartitionGymEnv = None
    make_parallel_env = None

__all__ = [
    'PowerGridPartitioningEnv',
    'PPOAgent',
    'StateManager',
    'ActionSpace',
    'ActionMask',
    'RewardFunction',
    'MetisInitializer',
    'PartitionEvaluator',
    'ScenarioGenerator'
]

if _gym_available:
    __all__.extend(['PowerGridPartitionGymEnv', 'make_parallel_env'])
