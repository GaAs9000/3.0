"""
强化学习模块
"""

from .environment import PowerGridPartitioningEnv
from .agent import PPOAgent
from .state import StateManager
from .action_space import ActionSpace, ActionMask
from .reward import RewardFunction
from .utils import MetisInitializer, PartitionEvaluator
from .scenario_generator import ScenarioGenerator
from .decision_context import AbstractDecisionContext, DecisionContextBatch, create_decision_context

# 智能自适应课程学习
try:
    from .adaptive import AdaptiveDirector
    _adaptive_available = True
except ImportError:
    _adaptive_available = False
    AdaptiveDirector = None

# 统一导演系统
try:
    from .unified_director import UnifiedDirector, DataMixConfig, AdaptiveThreshold
    _unified_director_available = True
except ImportError:
    _unified_director_available = False
    UnifiedDirector = None
    DataMixConfig = None 
    AdaptiveThreshold = None

# 多尺度生成器
try:
    from .scale_aware_generator import ScaleAwareSyntheticGenerator
    _scale_generator_available = True
except ImportError:
    _scale_generator_available = False
    ScaleAwareSyntheticGenerator = None

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

if _adaptive_available:
    __all__.extend(['AdaptiveDirector'])

if _unified_director_available:
    __all__.extend(['UnifiedDirector', 'DataMixConfig', 'AdaptiveThreshold'])

if _scale_generator_available:
    __all__.extend(['ScaleAwareSyntheticGenerator'])

if _gym_available:
    __all__.extend(['PowerGridPartitionGymEnv', 'make_parallel_env'])
