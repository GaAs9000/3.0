#!/usr/bin/env python3
"""
电力网络分区强化学习训练系统 v2.0

重构版本 - 四个解耦的训练模式：
1. BasicTrainer - 纯基础训练（无课程学习、无拓扑变化）
2. TopologyTrainer - 只拓扑变化训练（多尺度生成器）
3. AdaptiveTrainer - 只参数自适应训练（AdaptiveDirector）
4. UnifiedTrainer - 统一导演训练（拓扑+参数协调）

作者：Claude & User Collaboration
日期：2025-07-13
版本：2.0 - 清晰架构重构版
"""

import sys
import os
from pathlib import Path

# 将 code/src 目录添加到 Python 搜索路径，以解决模块导入问题
# 这使得脚本无论从哪里运行，都能正确找到 rl 等模块
project_root = Path(__file__).resolve().parent
src_path = project_root / 'code' / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np
import argparse
import yaml
import os
import sys
import time
import json
import warnings
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
from abc import ABC, abstractmethod
import logging
from functools import wraps

# 添加代码路径
# sys.path.append(str(Path(__file__).parent / 'code' / 'src'))  <- 冗余，已由顶部的代码取代
# sys.path.append(str(Path(__file__).parent / 'code')) <- 冗余
# sys.path.append(str(Path(__file__).parent / 'code' / 'utils')) <- 冗余


def handle_exceptions(func):
    """统一异常处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    return wrapper


def ensure_device_consistency(obj, target_device):
    """确保对象在目标设备上，避免重复转移"""
    if hasattr(obj, 'device') and obj.device == target_device:
        return obj  # 已经在目标设备上，避免重复转移
    elif hasattr(obj, 'to'):
        return obj.to(target_device)
    else:
        return obj

# 核心模块导入
try:
    import gymnasium as gym
    from torch_geometric.data import HeteroData
    from code.src.data_processing import PowerGridDataProcessor
    from code.src.gat import create_production_encoder
    from code.src.rl.environment import PowerGridPartitioningEnv
    from code.src.rl.enhanced_environment import EnhancedPowerGridPartitioningEnv, create_enhanced_environment
    from code.src.rl.enhanced_agent import create_enhanced_agent
    from code.src.pretrain.gnn_pretrainer import GNNPretrainer, PretrainConfig
    from code.src.rl.scale_aware_generator import ScaleAwareSyntheticGenerator, GridGenerationConfig
    from code.src.rl.adaptive import AdaptiveDirector
    from code.src.rl.unified_director import UnifiedDirector
    from code.src.rl.gym_wrapper import PowerGridPartitionGymEnv
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = e

# 全局配置
torch.autograd.set_detect_anomaly(False)
warnings.filterwarnings("error", message=".*NaN.*", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设置日志
def setup_logging(verbose: bool = False, debug: bool = False):
    """设置日志级别和格式"""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # 设置根日志级别
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 设置特定模块的日志级别
    if not verbose:
        # 静默第三方库的日志
        logging.getLogger('pandapower').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        # 保留GNN预训练的关键信息，但减少详细输出
        logging.getLogger('pretrain.gnn_pretrainer').setLevel(logging.INFO)
        logging.getLogger('rl.enhanced_environment').setLevel(logging.WARNING)
        logging.getLogger('models.actor_network').setLevel(logging.WARNING)
        logging.getLogger('models.partition_encoder').setLevel(logging.WARNING)
        logging.getLogger('rl.enhanced_agent').setLevel(logging.WARNING)
        logging.getLogger('rl.unified_director').setLevel(logging.WARNING)
        logging.getLogger('rl.adaptive').setLevel(logging.WARNING)

    # 确保主模块日志始终可见
    logging.getLogger(__name__).setLevel(logging.INFO if verbose else logging.WARNING)

# 默认设置
setup_logging()
logger = logging.getLogger(__name__)

# 简洁输出函数
def print_header(title: str):
    """打印标题"""
    print(f"\n🚀 {title}")

def print_status(message: str, status: str = "info"):
    """打印状态信息"""
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    icon = icons.get(status, "ℹ️")
    print(f"{icon} {message}")

def print_progress(current: int, total: int, message: str = ""):
    """打印进度信息"""
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"📊 进度: {current}/{total} ({percentage:.1f}%) {message}")

def print_result(key: str, value: Any):
    """打印结果信息"""
    print(f"📈 {key}: {value}")


# =====================================================
# 基础设施层
# =====================================================

class DataManager:
    """数据加载和处理管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config.get('data', {})
        self.processor = None
        
    def load_power_grid_data(self, case_name: str) -> Dict:
        """加载电力网络数据
        
        Args:
            case_name: 案例名称 (ieee14, ieee30, ieee57, ieee118)
            
        Returns:
            MATPOWER格式的电网数据字典
        """
        try:
            import pandapower as pp
            import pandapower.networks as pn
            from pandapower.converter import to_mpc
            
            # 映射案例名称到PandaPower函数
            case_mapping = {
                'ieee14': pn.case14,
                'ieee30': pn.case30,
                'ieee39': pn.case39,
                'ieee57': pn.case57,
                'ieee118': pn.case118
            }
            
            if case_name not in case_mapping:
                raise ValueError(f"不支持的案例: {case_name}")
            
            # 加载PandaPower网络
            net = case_mapping[case_name]()
            logger.info(f"成功加载 {case_name.upper()}: {len(net.bus)} 节点, {len(net.line)} 线路")
            
            # 修复数据类型问题
            self._fix_pandapower_data_types(net)

            # 运行潮流分析
            try:
                if getattr(net, "res_bus", None) is None or net.res_bus.empty:
                    pp.runpp(net, init="flat", calculate_voltage_angles=False, numba=False)
            except Exception as e:
                logger.warning(f"潮流分析失败，使用默认值: {e}")
                pass
            
            # 转换为MATPOWER格式
            mpc_wrap = to_mpc(net)
            mpc = mpc_wrap["mpc"] if isinstance(mpc_wrap, dict) and "mpc" in mpc_wrap else mpc_wrap
            
            return mpc
            
        except ImportError:
            raise ImportError("Pandapower库未安装或不可用，无法加载电网数据")
        except Exception as e:
            logger.error(f"加载案例 '{case_name}' 失败: {e}")
            raise

    def _fix_pandapower_data_types(self, net):
        """修复PandaPower数据类型问题"""
        try:
            # 修复变压器数据类型问题
            if hasattr(net, 'trafo') and not net.trafo.empty:
                # 确保变压器参数为数值类型
                numeric_cols = ['tap_neutral', 'tap_min', 'tap_max', 'tap_pos']
                for col in numeric_cols:
                    if col in net.trafo.columns:
                        net.trafo[col] = pd.to_numeric(net.trafo[col], errors='coerce')
                        # 修复pandas FutureWarning - 使用正确的fillna方式
                        if col == 'tap_neutral':
                            net.trafo[col] = net.trafo[col].fillna(0.0)
                        elif col in ['tap_min', 'tap_max']:
                            net.trafo[col] = net.trafo[col].fillna(1.0)
                        elif col == 'tap_pos':
                            net.trafo[col] = net.trafo[col].fillna(0)

            # 修复发电机电压限制问题
            if hasattr(net, 'gen') and not net.gen.empty and hasattr(net, 'bus') and not net.bus.empty:
                for idx, gen in net.gen.iterrows():
                    bus_idx = gen['bus']
                    if bus_idx in net.bus.index:
                        # 确保发电机电压设定值在母线电压限制范围内
                        vm_pu = gen.get('vm_pu', 1.0)
                        max_vm_pu = net.bus.loc[bus_idx, 'max_vm_pu']
                        min_vm_pu = net.bus.loc[bus_idx, 'min_vm_pu']

                        if vm_pu > max_vm_pu:
                            net.gen.loc[idx, 'vm_pu'] = max_vm_pu
                        elif vm_pu < min_vm_pu:
                            net.gen.loc[idx, 'vm_pu'] = min_vm_pu

        except Exception as e:
            logger.warning(f"修复PandaPower数据类型时出错: {e}")
            pass
    
    def setup_processor(self, device: torch.device, timestamp_dir: Optional[str] = None) -> PowerGridDataProcessor:
        """设置数据处理器"""
        if self.processor is None:
            # 如果提供了时间戳目录，使用它；否则使用配置中的默认值
            if timestamp_dir:
                cache_dir = str(Path(timestamp_dir) / "cache")
            else:
                cache_dir = self.data_config.get('cache_dir', 'data/latest/cache')

            self.processor = PowerGridDataProcessor(
                normalize=self.data_config.get('normalize', True),
                cache_dir=cache_dir
            )
        return self.processor
    
    def process_data(self, mpc: Dict, config: Dict[str, Any], device: torch.device, timestamp_dir: Optional[str] = None) -> HeteroData:
        """处理电网数据为图数据"""
        processor = self.setup_processor(device, timestamp_dir)
        hetero_data = processor.graph_from_mpc(mpc, config).to(str(device))
        logger.info(f"数据处理完成: {hetero_data}")
        return hetero_data


class EnvironmentFactory:
    """环境创建工厂"""
    
    @staticmethod
    @handle_exceptions
    def create_basic_environment(hetero_data: HeteroData, config: Dict[str, Any]) -> EnhancedPowerGridPartitioningEnv:
        """创建基础环境（无场景生成）"""
        env_config = config['environment']

        # 确保设备配置正确
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_copy = config.copy()
        config_copy['device'] = str(device) # 确保传递的是字符串
        config_copy['reward_weights'] = env_config.get('reward_weights', {})
        config_copy['max_steps'] = env_config.get('max_steps', 200)
        config_copy['enable_features_cache'] = env_config.get('enable_features_cache', True)

        # 使用预训练的GAT编码器生成节点嵌入
        encoder = AgentFactory.setup_gnn_pretraining(hetero_data, config)
        encoder = encoder.to(device)

        with torch.no_grad():
            node_embeddings = encoder.encode_nodes(hetero_data)

        # 调试：打印节点嵌入的维度信息
        for node_type, embeddings in node_embeddings.items():
            logger.info(f"节点类型 {node_type}: 嵌入形状 {embeddings.shape}")

        logger.info(f"GAT编码器生成节点嵌入完成: {list(node_embeddings.keys())}")

        # 使用原有的create_enhanced_environment函数，这个函数会处理所有必需参数
        # �� 重要修复：传递编码器实例进行统一管理
        env = create_enhanced_environment(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=env_config['num_partitions'],
            config=config_copy,
            use_enhanced=True,
            gat_encoder=encoder
        )
        
        assert env is not None, "环境创建失败，返回了None"
        logger.info(f"基础环境创建完成: {env.total_nodes}节点, {env.num_partitions}分区")
        return env
    
    @staticmethod
    def create_scenario_environment(mpc: Dict, config: Dict[str, Any], 
                                  scale_generator: Optional[ScaleAwareSyntheticGenerator] = None,
                                  gat_encoder: Optional[Any] = None) -> Tuple[EnhancedPowerGridPartitioningEnv, PowerGridPartitionGymEnv]:
        """创建场景生成环境"""
        try:
            # 确保设备配置正确传递
            config_copy = config.copy()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            config_copy['system']['device'] = str(device)
            
            # 🔧 重要修复：传递预训练编码器实例避免重复创建
            gym_env = PowerGridPartitionGymEnv(
                base_case_data=mpc,
                config=config_copy,
                gat_encoder=gat_encoder,  # 传递编码器实例
                use_scenario_generator=scale_generator is None,  # 如果有多尺度生成器就不用默认生成器
                scenario_seed=config['system']['seed'],
                scale_generator=scale_generator
            )
            
            # 重置环境获取初始状态
            obs_array, info = gym_env.reset()
            env = gym_env.internal_env
            
            assert env is not None, "内部环境初始化失败，返回了None"
            logger.info(f"场景生成环境创建完成: {env.total_nodes}节点, {env.num_partitions}分区")
            return env, gym_env
            
        except Exception as e:
            logger.error(f"创建场景生成环境失败: {e}")
            raise


class AgentFactory:
    """智能体创建工厂"""
    
    @staticmethod
    def create_agent(env: EnhancedPowerGridPartitioningEnv, config: Dict[str, Any], 
                    device: torch.device) -> Any:
        """创建增强智能体（双塔架构）"""
        agent_config = config['agent']

        assert env is not None and env.state_manager is not None
        # 调试：打印状态维度信息
        logger.info(f"环境状态维度: {env.state_manager.embedding_dim}")
        logger.info(f"分区数: {env.num_partitions}")

        agent = create_enhanced_agent(
            state_dim=env.state_manager.embedding_dim,
            num_partitions=env.num_partitions,
            config=agent_config,
            device=str(device) # 确保传递的是字符串
        )
        
        logger.info("增强智能体创建完成（双塔架构）")
        return agent
    
    @staticmethod
    def setup_gnn_pretraining(hetero_data: Any, config: Dict[str, Any],
                            training_source: Union[List, Any] = None) -> Any:
        """设置GNN预训练并选择性地加载权重（v3.0增强）

        Args:
            hetero_data: 异构图数据，用于创建GAT编码器
            config: 配置字典
            training_source: 训练数据源（可选）

        Returns:
            GAT编码器，可能已加载预训练权重
        """
        # 🔧 重要修复：添加配置验证逻辑
        if not isinstance(config, dict):
            raise ValueError("config参数必须是字典类型")
        
        if 'gat' not in config:
            raise ValueError("配置中缺少'gat'部分")
        
        if not hasattr(hetero_data, 'x_dict') or not hetero_data.x_dict:
            raise ValueError("hetero_data必须包含有效的x_dict")
        
        pretrain_config = config.get('gnn_pretrain', {})
        
        # 统一创建编码器
        from code.src.gat import create_production_encoder
        gat_config = config.get('gat', {})
        
        try:
            gat_encoder = create_production_encoder(hetero_data, gat_config)
            logger.info(f"创建GAT编码器: {type(gat_encoder).__name__}")
        except Exception as e:
            logger.error(f"GAT编码器创建失败: {e}")
            raise RuntimeError(f"无法创建GAT编码器: {e}")
        
        # 检查是否需要执行预训练
        if pretrain_config.get('enabled', False) and training_source:
            try:
                # ... (此处省略了预训练的详细实现，因为它在 run_pretrain.py 中)
                logger.info("在此处将调用 run_pretrain.py，此处只负责加载")
            except Exception as e:
                logger.error(f"GNN预训练执行失败: {e}")
                
        # 检查是否需要加载预训练权重
        if pretrain_config.get('load_pretrained_weights', True): # 默认为True
            checkpoint_dir = Path(pretrain_config.get('checkpoint_dir', 'data/latest/pretrain_checkpoints'))
            best_model_path = checkpoint_dir / 'best_model.pt'
            
            if best_model_path.exists():
                try:
                    logger.info(f"正在从 {best_model_path} 加载预训练权重...")
                    # 🔧 统一设备处理：确保权重加载到正确设备
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    checkpoint = torch.load(best_model_path, map_location=device)
                    
                    # 兼容旧格式和新格式的检查点
                    if 'encoder_state_dict' in checkpoint:
                        weights = checkpoint['encoder_state_dict']
                    else:
                        weights = checkpoint

                    gat_encoder.load_state_dict(weights)
                    logger.info("✅ 成功加载预训练GNN权重。")
                except Exception as e:
                    logger.warning(f"⚠️ 加载预训练GNN权重失败: {e}。将使用随机初始化的GNN。")
            else:
                logger.warning(f"⚠️ 未找到预训练权重文件: {best_model_path}。将使用随机初始化的GNN。")
        else:
            logger.info("未启用加载预训练权重，使用随机初始化的GNN。")
            
        return gat_encoder


class LoggerFactory:
    """日志系统工厂"""
    
    @staticmethod
    def create_training_logger(config: Dict[str, Any], total_episodes: int) -> 'TrainingLogger':
        """创建训练日志记录器"""
        return TrainingLogger(config, total_episodes)


# =====================================================
# 核心组件层
# =====================================================

class ScaleGeneratorManager:
    """多尺度生成器管理器"""
    
    @staticmethod
    def create_scale_generator(config: Dict[str, Any]) -> Optional[ScaleAwareSyntheticGenerator]:
        """创建多尺度生成器"""
        multi_scale_cfg = config.get('multi_scale_generation', {})
        if not multi_scale_cfg.get('enabled', False):
            return None
            
        try:
            topo_cfg = multi_scale_cfg.get('topology_variation', {})
            loadgen_cfg = multi_scale_cfg.get('load_gen_variation', {})
            
            gen_cfg_kwargs = {
                'topology_variation_prob': topo_cfg.get('variation_prob', 0.2),
                'node_addition_prob': topo_cfg.get('node_addition_prob', 0.1),
                'node_removal_prob': topo_cfg.get('node_removal_prob', 0.05),
                'edge_addition_prob': topo_cfg.get('edge_addition_prob', 0.1),
                'edge_removal_prob': topo_cfg.get('edge_removal_prob', 0.05),
                'load_variation_range': tuple(loadgen_cfg.get('load_range', [0.7, 1.3])),
                'gen_variation_range': tuple(loadgen_cfg.get('gen_range', [0.8, 1.2])),
                'ensure_connectivity': topo_cfg.get('ensure_connectivity', True),
                'min_degree': topo_cfg.get('min_degree', 2),
                'max_degree': topo_cfg.get('max_degree', 6)
            }
            
            grid_gen_config = GridGenerationConfig(**gen_cfg_kwargs)
            data_manager = DataManager(config)
            
            scale_generator = ScaleAwareSyntheticGenerator(
                base_case_loader=data_manager.load_power_grid_data,
                config=grid_gen_config,
                seed=config['system']['seed']
            )
            
            logger.info("多尺度生成器创建完成")
            return scale_generator
            
        except Exception as e:
            logger.warning(f"多尺度生成器创建失败: {e}")
            return None


class AdaptiveDirectorManager:
    """自适应导演管理器"""
    
    @staticmethod
    def create_adaptive_director(config: Dict[str, Any], base_mode: str = 'fast') -> Optional[AdaptiveDirector]:
        """创建自适应导演"""
        # 兼容新旧配置结构：首先检查嵌套配置，然后回退到顶层
        adaptive_mode_cfg = config.get('adaptive_mode', {})
        is_enabled_in_mode = adaptive_mode_cfg.get('adaptive_curriculum', {}).get('enabled', False)
        is_enabled_at_top = config.get('adaptive_curriculum', {}).get('enabled', False)

        if not (is_enabled_in_mode or is_enabled_at_top):
            return None
            
        try:
            director = AdaptiveDirector(config, base_mode)
            logger.info(f"自适应导演创建完成 (基础模式: {base_mode})")
            return director
        except Exception as e:
            logger.warning(f"自适应导演创建失败: {e}")
            return None


class UnifiedDirectorManager:
    """统一导演管理器"""
    
    @staticmethod
    def create_unified_director(config: Dict[str, Any]) -> Optional[UnifiedDirector]:
        """创建统一导演"""
        # 兼容新旧配置结构：首先检查嵌套配置，然后回退到顶层
        unified_mode_cfg = config.get('unified_mode', {})
        is_enabled_in_mode = unified_mode_cfg.get('unified_director', {}).get('enabled', False)
        is_enabled_at_top = config.get('unified_director', {}).get('enabled', False)

        if not (is_enabled_in_mode or is_enabled_at_top):
            return None
            
        try:
            director = UnifiedDirector(config)
            director.set_total_episodes(config['training']['num_episodes'])
            logger.info("统一导演创建完成")
            return director
        except Exception as e:
            logger.warning(f"统一导演创建失败: {e}")
            return None


# =====================================================
# 训练日志记录器
# =====================================================

class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, config: Dict[str, Any], total_episodes: int):
        self.config = config
        self.start_time = time.time()
        self.total_episodes = total_episodes

        # 创建时间戳目录
        self.timestamp_dir = self._create_timestamp_directory()

        # 训练指标
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.success_rates = []
        self.best_reward = -float('inf')

        # 设置TensorBoard
        self.use_tensorboard = config.get('logging', {}).get('use_tensorboard', False)
        self.tensorboard_writer = self._setup_tensorboard() if self.use_tensorboard else None

        # 设置进度条
        self.progress_bar = self._setup_progress_bar()

        # 只在verbose模式下显示详细信息
        verbose = config.get('system', {}).get('verbose', False)
        if verbose:
            logger.info(f"训练日志记录器初始化完成: {total_episodes} episodes")
            logger.info(f"训练目录: {self.timestamp_dir}")
        else:
            print_status(f"训练目录: {Path(self.timestamp_dir).name}")

    def _create_timestamp_directory(self) -> str:
        """创建时间戳目录结构"""
        from datetime import datetime

        # 创建时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        timestamp_dir = Path("data") / timestamp

        # 创建完整的目录结构
        subdirs = ['logs', 'checkpoints', 'models', 'figures', 'cache', 'output']
        for subdir in subdirs:
            (timestamp_dir / subdir).mkdir(parents=True, exist_ok=True)

        return str(timestamp_dir)

    def _setup_tensorboard(self):
        """设置TensorBoard"""
        try:
            from torch.utils.tensorboard.writer import SummaryWriter
            log_dir = Path(self.timestamp_dir) / "logs"
            writer = SummaryWriter(str(log_dir))
            verbose = self.config.get('system', {}).get('verbose', False)
            if verbose:
                logger.info(f"TensorBoard已启用: {log_dir}")
            else:
                print_status("TensorBoard已启用")
            return writer
        except ImportError:
            if self.config.get('system', {}).get('verbose', False):
                logger.warning("TensorBoard不可用")
            return None
        except Exception as e:
            if self.config.get('system', {}).get('verbose', False):
                logger.warning(f"TensorBoard初始化失败: {e}")
            return None
    
    def _setup_progress_bar(self):
        """设置进度条"""
        try:
            from tqdm import tqdm
            return tqdm(total=self.total_episodes, desc="🚀 训练进度",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        except ImportError:
            return None
    
    def log_episode(self, episode: int, reward: float, length: int, info: Optional[Dict] = None):
        """记录单个回合"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if reward > self.best_reward:
            self.best_reward = reward
        
        # 更新进度条
        if self.progress_bar:
            avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10)
            desc = f"🚀 训练进度 | 当前: {reward:.3f} | 最佳: {self.best_reward:.3f} | 平均: {avg_reward:.3f}"
            self.progress_bar.set_description(desc)
            self.progress_bar.update(1)
        
        # TensorBoard记录
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Core/Episode_Reward', reward, episode)
            self.tensorboard_writer.add_scalar('Core/Episode_Length', length, episode)
            
            if info:
                # 记录成功率
                if 'success' in info:
                    self.tensorboard_writer.add_scalar('Core/Success_Rate', float(info['success']), episode)
                
                # 记录奖励组件
                reward_components = ['balance_reward', 'decoupling_reward', 'power_reward', 'quality_reward', 'final_reward']
                for component in reward_components:
                    if component in info and isinstance(info[component], (int, float)):
                        self.tensorboard_writer.add_scalar(f'Reward/{component.capitalize()}', info[component], episode)
                
                # 记录质量指标
                quality_metrics = ['quality_score', 'plateau_confidence']
                for metric in quality_metrics:
                    if metric in info and isinstance(info[metric], (int, float)):
                        self.tensorboard_writer.add_scalar(f'Quality/{metric.capitalize()}', info[metric], episode)

                # 记录连通性和分区指标
                if 'metrics' in info:
                    metrics = info['metrics']
                    connectivity_metrics = ['connectivity', 'avg_connectivity']
                    for metric in connectivity_metrics:
                        if metric in metrics and isinstance(metrics[metric], (int, float)):
                            self.tensorboard_writer.add_scalar(f'Connectivity/{metric.capitalize()}', metrics[metric], episode)

                    partition_metrics = ['load_balance', 'decoupling', 'power_balance']
                    for metric in partition_metrics:
                        if metric in metrics and isinstance(metrics[metric], (int, float)):
                            self.tensorboard_writer.add_scalar(f'Partition/{metric.capitalize()}', metrics[metric], episode)

                # 记录训练状态
                if 'training_state' in info:
                    state = info['training_state']
                    if isinstance(state, dict):
                        for key, value in state.items():
                            if isinstance(value, (int, float)):
                                self.tensorboard_writer.add_scalar(f'Training_State/{key.capitalize()}', value, episode)
            
            # 定期刷新
            if episode % 10 == 0:
                self.tensorboard_writer.flush()
    
    def log_training_step(self, episode: int, actor_loss: Optional[float] = None,
                         critic_loss: Optional[float] = None, entropy: Optional[float] = None):
        """记录训练步骤"""
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/Actor_Loss', actor_loss, episode)
        
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/Critic_Loss', critic_loss, episode)
        
        if entropy is not None:
            self.entropies.append(entropy)
            if self.use_tensorboard and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Training/Policy_Entropy', entropy, episode)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'best_reward': self.best_reward,
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'training_time': time.time() - self.start_time,
            'success_rate': np.mean(self.success_rates) if self.success_rates else 0.0
        }
    
    def close(self):
        """关闭日志记录器"""
        if self.progress_bar:
            self.progress_bar.close()
        if self.tensorboard_writer:
            self.tensorboard_writer.close()


# =====================================================
# 抽象训练器基类
# =====================================================

class BaseTrainer(ABC):
    """抽象训练器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.data_manager = DataManager(config)
        self.logger = None
        
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_config = self.config['system'].get('device', 'auto')
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        logger.info(f"使用设备: {device}")
        return device
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """训练接口 - 子类必须实现"""
        pass
    
    def evaluate(self, agent: Any, env: Any, num_episodes: int = 10) -> Dict[str, Any]:
        """评估智能体"""
        logger.info(f"开始评估 {num_episodes} 轮...")
        
        eval_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_info = {}
            
            for step in range(200):  # 最大步数
                # EnhancedPPOAgent返回4个值，普通PPOAgent返回3个值
                result = agent.select_action(state, training=False)
                if len(result) == 4:
                    action, _, _, _ = result  # 忽略log_prob, value, embeddings
                else:
                    action, _, _ = result  # 忽略log_prob, value
                
                if action is None:
                    break
                
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if info:
                    episode_info.update(info)
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            
            # 评估成功标准
            is_success = self._evaluate_episode_success(episode_reward, episode_info)
            if is_success:
                success_count += 1
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'success_rate': success_count / num_episodes,
            'rewards': eval_rewards
        }
    
    def _evaluate_episode_success(self, episode_reward: float, episode_info: Dict[str, Any]) -> bool:
        """评估单个episode是否成功"""
        # 成功标准1: 奖励超过合理阈值
        if episode_reward > -1.0:
            return True
        
        # 成功标准2: 基于质量指标
        if episode_info and 'metrics' in episode_info:
            metrics = episode_info['metrics']
            
            # 检查负载平衡
            cv = metrics.get('cv', metrics.get('load_cv', 1.0))
            if cv < 0.5:
                return True
            
            # 检查连通性
            connectivity = metrics.get('connectivity', 0.0)
            if connectivity > 0.9:
                return True
            
            # 检查耦合比率
            coupling_ratio = metrics.get('coupling_ratio', 1.0)
            if coupling_ratio < 0.3:
                return True
        
        # 成功标准3: 基于质量分数
        if episode_info and 'reward_components' in episode_info:
            components = episode_info['reward_components']
            quality_score = components.get('quality_score', 0.0)
            if quality_score > 0.4:
                return True
        
        # 相对改进标准
        if episode_reward > -2.5:
            return True
        
        return False
    
    def _save_model(self, agent: Any, final_stats: Dict[str, Any],
                   model_suffix: str = "") -> Optional[str]:
        """保存训练模型"""
        try:
            assert self.logger is not None
            models_dir = Path(self.logger.timestamp_dir) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            model_filename = f"agent_{model_suffix}.pth"
            model_path = models_dir / model_filename
            
            # 保存模型
            checkpoint = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'training_stats': final_stats,
                'timestamp': Path(self.logger.timestamp_dir).name,
                'training_directory': self.logger.timestamp_dir,
                'config': self.config,
                'model_info': {
                    'node_embedding_dim': agent.node_embedding_dim,
                    'region_embedding_dim': agent.region_embedding_dim,
                    'num_partitions': agent.num_partitions,
                    'device': str(agent.device)
                }
            }
            
            torch.save(checkpoint, model_path)
            
            logger.info(f"模型已保存: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return None


# =====================================================
# 四个解耦的训练器实现
# =====================================================

class BasicTrainer(BaseTrainer):
    """纯基础训练器 - 无课程学习、无拓扑变化"""
    
    def train(self) -> Dict[str, Any]:
        """执行基础训练"""
        logger.info("🔥 开始基础训练模式 - 无课程学习、无拓扑变化")
        
        try:
            # 1. 训练配置
            training_config = self.config['training']
            num_episodes = training_config['num_episodes']
            max_steps_per_episode = training_config['max_steps_per_episode']
            update_interval = training_config['update_interval']

            # 2. 初始化日志记录器（需要先创建时间戳目录）
            self.logger = LoggerFactory.create_training_logger(self.config, num_episodes)

            # 3. 数据加载和处理
            ref_case = self.config['data'].get('reference_case', 'ieee14')
            mpc = self.data_manager.load_power_grid_data(ref_case)
            hetero_data = self.data_manager.process_data(mpc, self.config, self.device, self.logger.timestamp_dir)

            # 4. 创建基础环境（无场景生成）
            try:
                env = EnvironmentFactory.create_basic_environment(hetero_data, self.config)
            except Exception as e:
                logger.error(f"创建基础环境失败: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                raise

            # 5. 创建智能体
            try:
                agent = AgentFactory.create_agent(env, self.config, self.device)
            except Exception as e:
                logger.error(f"创建智能体失败: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                raise

            # 6. GNN预训练已在环境创建时完成
            
            # 7. 基础训练循环
            logger.info(f"开始基础训练: {num_episodes} episodes")

            for episode in range(num_episodes):
                try:
                    state, _ = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    episode_info = {}

                    # Episode执行
                    for step in range(max_steps_per_episode):
                        try:
                            # EnhancedPPOAgent返回4个值，普通PPOAgent返回3个值
                            result = agent.select_action(state, training=True)
                            if len(result) == 4:
                                action, log_prob, value, _ = result  # 忽略embeddings
                            else:
                                action, log_prob, value = result

                            if action is None:
                                break

                            next_state, reward, terminated, truncated, info = env.step(action)
                            done = terminated or truncated

                            agent.store_experience(state, action, reward, log_prob, value, done)
                            episode_reward += reward
                            episode_length += 1
                            state = next_state

                            if info:
                                episode_info.update(info)

                            if done:
                                break
                        except Exception as e:
                            logger.error(f"Episode {episode}, Step {step} 执行失败: {e}")
                            import traceback
                            logger.error(f"详细错误信息: {traceback.format_exc()}")
                            raise
                except Exception as e:
                    logger.error(f"Episode {episode} 失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    raise
                
                # 记录训练日志
                self.logger.log_episode(episode, episode_reward, episode_length, episode_info)
                
                # 智能体更新
                if episode % update_interval == 0 and episode > 0:
                    try:
                        training_stats = agent.update()
                        if training_stats:
                            self.logger.log_training_step(
                                episode,
                                training_stats.get('actor_loss'),
                                training_stats.get('critic_loss'),
                                training_stats.get('entropy')
                            )
                    except Exception as e:
                        logger.warning(f"Episode {episode} 智能体更新失败: {e}")
            
            # 8. 训练完成
            final_stats = self.logger.get_statistics()
            logger.info("基础训练完成")
            
            # 9. 模型保存
            model_path = self._save_model(agent, final_stats, f"basic_{ref_case}")
            
            # 10. 评估
            eval_stats = self.evaluate(agent, env)
            
            # 11. 清理
            self.logger.close()
            
            return {
                'success': True,
                'mode': 'basic',
                'config': self.config,
                'training_stats': final_stats,
                'eval_stats': eval_stats,
                'model_path': model_path,
                'best_reward': self.logger.best_reward
            }
            
        except Exception as e:
            logger.error(f"基础训练失败: {e}")
            if self.logger:
                self.logger.close()
            return {'success': False, 'error': str(e), 'mode': 'basic'}


class TopologyTrainer(BaseTrainer):
    """拓扑变化训练器 - 只拓扑变化、无参数自适应"""
    
    def train(self) -> Dict[str, Any]:
        """执行拓扑变化训练"""
        logger.info("🌐 开始拓扑变化训练模式 - 只多尺度生成器")
        
        try:
            # 1. 训练配置
            training_config = self.config['training']
            num_episodes = training_config['num_episodes']
            max_steps_per_episode = training_config['max_steps_per_episode']
            update_interval = training_config['update_interval']

            # 2. 初始化日志记录器 (提前)
            self.logger = LoggerFactory.create_training_logger(self.config, num_episodes)
            
            # 3. 数据加载
            ref_case = self.config['data'].get('reference_case', 'ieee14')
            mpc = self.data_manager.load_power_grid_data(ref_case)
            
            # 4. 创建多尺度生成器
            scale_generator = ScaleGeneratorManager.create_scale_generator(self.config)

            # 5. GNN预训练/加载 (在创建环境前完成)
            sample_hetero_data = self.data_manager.process_data(mpc, self.config, self.device, self.logger.timestamp_dir)
            gat_encoder = AgentFactory.setup_gnn_pretraining(sample_hetero_data, self.config, scale_generator)
            
            # 6. 创建场景生成环境
            env, gym_env = EnvironmentFactory.create_scenario_environment(
                mpc, self.config, scale_generator, gat_encoder
            )
            
            # 7. 创建智能体
            agent = AgentFactory.create_agent(env, self.config, self.device)
            
            # 8. 拓扑变化训练循环
            logger.info(f"开始拓扑变化训练: {num_episodes} episodes")
            
            for episode in range(num_episodes):
                # 重置环境（会生成新的拓扑变化）
                assert gym_env is not None, "Gym环境未初始化"
                obs_array, info = gym_env.reset()
                env = gym_env.internal_env  # 获取新的内部环境
                assert env is not None, "内部环境在重置后为None"
                state, _ = env.reset()
                
                episode_reward = 0
                episode_length = 0
                episode_info = {}
                
                # Episode执行
                for step in range(max_steps_per_episode):
                    action, log_prob, value, _ = agent.select_action(state, training=True)
                    
                    if action is None:
                        break
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    agent.store_experience(state, action, reward, log_prob, value, done)
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    
                    if info:
                        episode_info.update(info)
                    
                    if done:
                        break
                
                # 记录训练日志
                self.logger.log_episode(episode, episode_reward, episode_length, episode_info)
                
                # 智能体更新
                if episode % update_interval == 0 and episode > 0:
                    try:
                        training_stats = agent.update()
                        if training_stats:
                            self.logger.log_training_step(
                                episode,
                                training_stats.get('actor_loss'),
                                training_stats.get('critic_loss'),
                                training_stats.get('entropy')
                            )
                    except Exception as e:
                        logger.warning(f"Episode {episode} 智能体更新失败: {e}")
            
            # 9. 训练完成
            final_stats = self.logger.get_statistics()
            logger.info("拓扑变化训练完成")
            
            # 10. 模型保存
            model_path = self._save_model(agent, final_stats, f"topology_{ref_case}")
            
            # 11. 评估
            eval_stats = self.evaluate(agent, env)
            
            # 12. 清理
            self.logger.close()
            
            return {
                'success': True,
                'mode': 'topology',
                'config': self.config,
                'training_stats': final_stats,
                'eval_stats': eval_stats,
                'model_path': model_path,
                'best_reward': self.logger.best_reward,
                'scale_generator_enabled': True
            }
            
        except Exception as e:
            logger.error(f"拓扑变化训练失败: {e}")
            if self.logger:
                self.logger.close()
            return {'success': False, 'error': str(e), 'mode': 'topology'}


class AdaptiveTrainer(BaseTrainer):
    """自适应训练器 - 只参数自适应、无拓扑变化"""
    
    def train(self) -> Dict[str, Any]:
        """执行自适应训练"""
        logger.info("🧠 开始自适应训练模式 - 只参数自适应")
        
        try:
            # 1. 训练配置
            training_config = self.config['training']
            num_episodes = training_config['num_episodes']
            max_steps_per_episode = training_config['max_steps_per_episode']
            update_interval = training_config['update_interval']
            
            # 2. 初始化日志记录器 (提前)
            self.logger = LoggerFactory.create_training_logger(self.config, num_episodes)

            # 3. 数据加载
            ref_case = self.config['data'].get('reference_case', 'ieee14')
            mpc = self.data_manager.load_power_grid_data(ref_case)
            
            # 4. 创建自适应导演
            base_mode = self._detect_base_mode()
            adaptive_director = AdaptiveDirectorManager.create_adaptive_director(self.config, base_mode)
            
            if adaptive_director is None:
                logger.warning("自适应导演未启用，回退到基础训练")
                basic_trainer = BasicTrainer(self.config)
                return basic_trainer.train()
            
            # 5. GNN预训练/加载
            sample_hetero_data = self.data_manager.process_data(mpc, self.config, self.device, self.logger.timestamp_dir)
            training_source = [sample_hetero_data]
            gat_encoder = AgentFactory.setup_gnn_pretraining(sample_hetero_data, self.config, training_source)
            
            # 6. 创建场景环境 (无拓扑变化)
            env, gym_env = EnvironmentFactory.create_scenario_environment(mpc, self.config, None, gat_encoder)
            
            # 7. 创建智能体
            agent = AgentFactory.create_agent(env, self.config, self.device)
            
            # 8. 自适应训练循环
            logger.info(f"开始自适应训练: {num_episodes} episodes")
            director_decisions = []
            
            for episode in range(num_episodes):
                # 重置环境
                assert gym_env is not None, "Gym环境未初始化"
                if gym_env is not None:
                    obs_array, info = gym_env.reset()
                    env = gym_env.internal_env
                    assert env is not None, "内部环境在重置后为None"
                    state, _ = env.reset()
                else:
                    # 这个分支理论上不应该执行，因为adaptive_director检查会提前返回
                    state, _ = env.reset()
                
                episode_reward = 0
                episode_length = 0
                episode_info = {}
                
                # Episode执行
                for step in range(max_steps_per_episode):
                    action, log_prob, value, _ = agent.select_action(state, training=True)
                    
                    if action is None:
                        break
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    agent.store_experience(state, action, reward, log_prob, value, done)
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    
                    if info:
                        episode_info.update(info)
                    
                    if done:
                        break
                
                # 收集episode信息供自适应导演决策
                episode_summary = {
                    'episode': episode,
                    'reward': episode_reward,
                    'episode_length': episode_length,
                    'success': episode_info.get('success', False),
                    'cv': episode_info.get('cv', episode_info.get('load_cv', 1.0)),
                    'coupling_ratio': episode_info.get('coupling_ratio', 1.0),
                    'connectivity': episode_info.get('connectivity', 0.0),
                    **episode_info
                }
                
                # 自适应导演决策
                director_decision = adaptive_director.step(episode, episode_summary)
                director_decisions.append(director_decision)
                
                # 应用自适应导演的参数调整
                self._apply_adaptive_decision(env, agent, director_decision)
                
                # 记录训练日志
                self.logger.log_episode(episode, episode_reward, episode_length, episode_info)
                
                # 智能体更新
                if episode % update_interval == 0 and episode > 0:
                    try:
                        training_stats = agent.update()
                        if training_stats:
                            self.logger.log_training_step(
                                episode,
                                training_stats.get('actor_loss'),
                                training_stats.get('critic_loss'),
                                training_stats.get('entropy')
                            )
                    except Exception as e:
                        logger.warning(f"Episode {episode} 智能体更新失败: {e}")
            
            # 9. 训练完成
            final_stats = self.logger.get_statistics()
            director_summary = adaptive_director.get_status_summary()
            logger.info("自适应训练完成")
            
            # 10. 模型保存
            model_path = self._save_model(agent, final_stats, f"adaptive_{ref_case}")
            
            # 11. 评估
            eval_stats = self.evaluate(agent, env)
            
            # 12. 清理
            self.logger.close()
            
            return {
                'success': True,
                'mode': 'adaptive',
                'config': self.config,
                'training_stats': final_stats,
                'eval_stats': eval_stats,
                'model_path': model_path,
                'best_reward': self.logger.best_reward,
                'director_decisions': director_decisions,
                'director_summary': director_summary,
                'adaptive_director_enabled': True
            }
            
        except Exception as e:
            logger.error(f"自适应训练失败: {e}")
            if self.logger:
                self.logger.close()
            return {'success': False, 'error': str(e), 'mode': 'adaptive'}
    
    def _detect_base_mode(self) -> str:
        """检测基础训练模式"""
        num_episodes = self.config.get('training', {}).get('num_episodes', 1000)
        if num_episodes >= 3000:
            return 'ieee57'
        elif num_episodes >= 2000:
            return 'full'
        else:
            return 'fast'
    
    def _apply_adaptive_decision(self, env: Any, agent: Any, decision: Dict[str, Any]):
        """应用自适应导演的决策"""
        try:
            # 更新环境奖励参数
            if hasattr(env, 'reward_function') and 'reward_weights' in decision:
                reward_weights = decision['reward_weights']
                if hasattr(env.reward_function, 'update_weights'):
                    env.reward_function.update_weights(reward_weights)
            
            # 更新连通性惩罚
            if 'connectivity_penalty' in decision:
                if hasattr(env, 'connectivity_penalty'):
                    env.connectivity_penalty = decision['connectivity_penalty']
            
            # 更新智能体学习率
            if 'learning_rate_factor' in decision and decision['learning_rate_factor'] != 1.0:
                if hasattr(agent, 'update_learning_rate'):
                    agent.update_learning_rate(decision['learning_rate_factor'])
                    
        except Exception as e:
            logger.warning(f"应用自适应决策失败: {e}")


class UnifiedTrainer(BaseTrainer):
    """统一导演训练器 - 拓扑变化 + 参数协调"""
    
    def train(self) -> Dict[str, Any]:
        """执行统一导演训练"""
        logger.info("🎭 开始统一导演训练模式 - 拓扑变化 + 参数协调")
        
        try:
            # 1. 训练配置
            training_config = self.config['training']
            num_episodes = training_config['num_episodes']
            max_steps_per_episode = training_config['max_steps_per_episode']
            update_interval = training_config['update_interval']

            # 2. 初始化日志记录器 (提前)
            self.logger = LoggerFactory.create_training_logger(self.config, num_episodes)

            # 3. 数据加载
            ref_case = self.config['data'].get('reference_case', 'ieee14')
            mpc = self.data_manager.load_power_grid_data(ref_case)
            
            # 4. 创建统一导演
            unified_director = UnifiedDirectorManager.create_unified_director(self.config)
            
            if unified_director is None:
                logger.warning("统一导演未启用，回退到自适应训练")
                adaptive_trainer = AdaptiveTrainer(self.config)
                return adaptive_trainer.train()
            
            # 5. 创建多尺度生成器
            scale_generator = ScaleGeneratorManager.create_scale_generator(self.config)
            
            # 6. 创建自适应导演作为组件
            base_mode = self._detect_base_mode()
            adaptive_director = AdaptiveDirectorManager.create_adaptive_director(self.config, base_mode)
            
            # 7. 集成组件到统一导演
            unified_director.integrate_components(
                adaptive_director=adaptive_director,
                scale_generator=scale_generator
            )
            
            # 8. GNN预训练/加载
            sample_hetero_data = self.data_manager.process_data(mpc, self.config, self.device, self.logger.timestamp_dir)
            training_source = scale_generator if scale_generator else None
            gat_encoder = AgentFactory.setup_gnn_pretraining(sample_hetero_data, self.config, training_source)
            
            # 9. 创建场景生成环境
            env, gym_env = EnvironmentFactory.create_scenario_environment(
                mpc, self.config, scale_generator, gat_encoder
            )
            
            # 10. 创建智能体
            agent = AgentFactory.create_agent(env, self.config, self.device)

            # 确保智能体使用预训练后的编码器（如果智能体需要的话）
            if hasattr(agent, 'gat_encoder'):
                agent.gat_encoder = gat_encoder
            
            # 11. 训练循环
            logger.info(f"开始统一导演训练: {num_episodes} episodes")
            unified_decisions = []
            
            for episode in range(num_episodes):
                # 重置环境
                assert gym_env is not None, "Gym环境未初始化"
                if gym_env is not None:
                    obs_array, info = gym_env.reset()
                    env = gym_env.internal_env
                    assert env is not None, "内部环境在重置后为None"
                    state, _ = env.reset()
                else:
                    state, _ = env.reset()
                
                episode_reward = 0
                episode_length = 0
                episode_info = {}
                
                # Episode执行
                for step in range(max_steps_per_episode):
                    action, log_prob, value, _ = agent.select_action(state, training=True)
                    
                    if action is None:
                        break
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    agent.store_experience(state, action, reward, log_prob, value, done)
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    
                    if info:
                        episode_info.update(info)
                    
                    if done:
                        break
                
                # 收集episode信息供统一导演决策
                episode_summary = {
                    'episode': episode,
                    'reward': episode_reward,
                    'episode_length': episode_length,
                    'success': episode_info.get('success', False),
                    'cv': episode_info.get('cv', episode_info.get('load_cv', 1.0)),
                    'coupling_ratio': episode_info.get('coupling_ratio', 1.0),
                    'connectivity': episode_info.get('connectivity', 0.0),
                    'quality_score': episode_info.get('quality_score', 0.0),
                    **episode_info
                }
                
                # 统一导演决策
                unified_decision = unified_director.step(episode, episode_summary)
                unified_decisions.append(unified_decision)
                
                # 应用统一导演的决策
                self._apply_unified_decision(env, agent, unified_decision)
                
                # 记录训练日志
                self.logger.log_episode(episode, episode_reward, episode_length, episode_info)
                
                # 智能体更新
                if episode % update_interval == 0 and episode > 0:
                    try:
                        training_stats = agent.update(env=env)
                        if training_stats:
                            self.logger.log_training_step(
                                episode,
                                training_stats.get('actor_loss'),
                                training_stats.get('critic_loss'),
                                training_stats.get('entropy')
                            )
                    except Exception as e:
                        logger.warning(f"Episode {episode} 智能体更新失败: {e}")
            
            # 12. 训练完成
            final_stats = self.logger.get_statistics()
            director_summary = unified_director.get_status_summary()
            logger.info("统一导演训练完成")
            
            # 13. 模型保存
            model_path = self._save_model(agent, final_stats, f"unified_{ref_case}")
            
            # 14. 评估
            eval_stats = self.evaluate(agent, env)
            
            # 15. 清理
            self.logger.close()
            
            return {
                'success': True,
                'mode': 'unified',
                'config': self.config,
                'training_stats': final_stats,
                'eval_stats': eval_stats,
                'model_path': model_path,
                'best_reward': self.logger.best_reward,
                'unified_decisions': unified_decisions,
                'director_summary': director_summary,
                'unified_director_enabled': True,
                'scale_generator_enabled': scale_generator is not None,
                'adaptive_director_enabled': adaptive_director is not None
            }
            
        except Exception as e:
            import traceback
            logger.error(f"统一导演训练失败: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            if self.logger:
                self.logger.close()
            return {'success': False, 'error': str(e), 'mode': 'unified'}
    
    def _detect_base_mode(self) -> str:
        """检测基础训练模式"""
        num_episodes = self.config.get('training', {}).get('num_episodes', 1000)
        if num_episodes >= 3000:
            return 'ieee57'
        elif num_episodes >= 2000:
            return 'full'
        else:
            return 'fast'
    
    def _apply_unified_decision(self, env: Any, agent: Any, unified_decision: Dict[str, Any]):
        """应用统一导演的决策"""
        try:
            # 获取参数调整
            parameter_adjustments = unified_decision.get('parameter_adjustments', {})
            warmup_params = parameter_adjustments.get('warmup_params', {})
            
            # 应用预热参数（如果有）
            if warmup_params:
                # 更新环境参数
                if hasattr(env, 'reward_function') and 'reward_weights' in warmup_params:
                    reward_weights = warmup_params['reward_weights']
                    if hasattr(env.reward_function, 'update_weights'):
                        env.reward_function.update_weights(reward_weights)
                
                if 'connectivity_penalty' in warmup_params:
                    if hasattr(env, 'connectivity_penalty'):
                        env.connectivity_penalty = warmup_params['connectivity_penalty']
                
                # 更新智能体参数
                if 'learning_rate_factor' in warmup_params:
                    if hasattr(agent, 'update_learning_rate'):
                        agent.update_learning_rate(warmup_params['learning_rate_factor'])
            
            # 应用自适应导演的决策（如果有且不冲突）
            adaptive_decision = unified_decision.get('adaptive_decision', {})
            if adaptive_decision and not adaptive_decision.get('unified_override', False):
                self._apply_adaptive_decision(env, agent, adaptive_decision)
                
        except Exception as e:
            logger.warning(f"应用统一导演决策失败: {e}")
    
    def _apply_adaptive_decision(self, env: Any, agent: Any, decision: Dict[str, Any]):
        """应用自适应导演决策（辅助方法）"""
        try:
            # 更新环境奖励参数
            if hasattr(env, 'reward_function') and 'reward_weights' in decision:
                reward_weights = decision['reward_weights']
                if hasattr(env.reward_function, 'update_weights'):
                    env.reward_function.update_weights(reward_weights)
            
            # 更新连通性惩罚
            if 'connectivity_penalty' in decision:
                if hasattr(env, 'connectivity_penalty'):
                    env.connectivity_penalty = decision['connectivity_penalty']
            
            # 更新智能体学习率
            if 'learning_rate_factor' in decision and decision['learning_rate_factor'] != 1.0:
                if hasattr(agent, 'update_learning_rate'):
                    agent.update_learning_rate(decision['learning_rate_factor'])
                    
        except Exception as e:
            logger.warning(f"应用自适应决策失败: {e}")


# =====================================================
# 训练管理器
# =====================================================

class TrainingManager:
    """训练管理器 - 统一管理四种训练模式"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode_mapping = {
            'basic': BasicTrainer,
            'topology': TopologyTrainer, 
            'adaptive': AdaptiveTrainer,
            'unified': UnifiedTrainer
        }
        
    def select_training_mode(self, mode: Optional[str] = None) -> str:
        """选择训练模式
        
        Args:
            mode: 指定的训练模式，None表示自动选择
            
        Returns:
            选择的训练模式名称
        """
        if mode and mode in self.mode_mapping:
            return mode
            
        # 自动模式选择逻辑 - 检查新的配置结构
        config = self.config
        
        # 检查各模式是否启用
        if config.get('unified_mode', {}).get('enabled', False):
            logger.info("检测到unified_mode启用，选择unified模式")
            return 'unified'
        
        if config.get('adaptive_mode', {}).get('enabled', False):
            logger.info("检测到adaptive_mode启用，选择adaptive模式")
            return 'adaptive'
        
        if config.get('topology_mode', {}).get('enabled', False):
            logger.info("检测到topology_mode启用，选择topology模式")
            return 'topology'
        
        if config.get('basic_mode', {}).get('enabled', False):
            logger.info("检测到basic_mode启用，选择basic模式")
            return 'basic'
        
        # 回退逻辑 - 检查旧的配置键（向后兼容）
        if config.get('unified_director', {}).get('enabled', False):
            logger.info("检测到unified_director启用，选择unified模式")
            return 'unified'
        
        if config.get('adaptive_curriculum', {}).get('enabled', False):
            logger.info("检测到adaptive_curriculum启用，选择adaptive模式")
            return 'adaptive'
        
        if config.get('multi_scale_generation', {}).get('enabled', False):
            logger.info("检测到multi_scale_generation启用，选择topology模式")
            return 'topology'
        
        # 默认基础模式
        logger.info("未检测到课程学习组件，选择basic模式")
        return 'basic'
    
    def create_trainer(self, mode: str) -> BaseTrainer:
        """创建训练器实例
        
        Args:
            mode: 训练模式
            
        Returns:
            训练器实例
        """
        if mode not in self.mode_mapping:
            raise ValueError(f"不支持的训练模式: {mode}. 支持的模式: {list(self.mode_mapping.keys())}")
        
        trainer_class = self.mode_mapping[mode]
        return trainer_class(self.config)
    
    def validate_config(self, mode: str) -> Tuple[bool, List[str]]:
        """验证配置的完整性
        
        Args:
            mode: 训练模式
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查基础配置
        required_sections = ['training', 'environment', 'agent', 'gat']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"缺少必需的配置节: {section}")
        
        # 模式特定验证 - 检查新的配置结构
        if mode == 'topology':
            topology_config = self.config.get('topology_mode', {})
            if topology_config.get('multi_scale_generation', {}).get('enabled', False):
                # topology_mode下有配置
                pass
            elif self.config.get('multi_scale_generation', {}).get('enabled', False):
                # 全局配置中有
                pass
            else:
                errors.append("topology模式需要启用multi_scale_generation（在topology_mode或全局配置中）")
        
        elif mode == 'adaptive':
            adaptive_config = self.config.get('adaptive_mode', {})
            if adaptive_config.get('adaptive_curriculum', {}).get('enabled', False):
                # adaptive_mode下有配置
                pass
            elif self.config.get('adaptive_curriculum', {}).get('enabled', False):
                # 全局配置中有
                pass
            else:
                errors.append("adaptive模式需要启用adaptive_curriculum（在adaptive_mode或全局配置中）")
        
        elif mode == 'unified':
            unified_config = self.config.get('unified_mode', {})
            if unified_config.get('unified_director', {}).get('enabled', False):
                # unified_mode下有配置
                pass
            elif self.config.get('unified_director', {}).get('enabled', False):
                # 全局配置中有
                pass
            else:
                errors.append("unified模式需要启用unified_director（在unified_mode或全局配置中）")
        
        # 检查依赖库
        if not DEPENDENCIES_OK:
            errors.append(f"依赖库导入失败: {IMPORT_ERROR}")
        
        return len(errors) == 0, errors
    
    def run_training(self, mode: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """执行训练
        
        Args:
            mode: 训练模式
            **kwargs: 额外的训练参数
            
        Returns:
            训练结果字典
        """
        # 选择训练模式
        selected_mode = self.select_training_mode(mode)
        logger.info(f"选择训练模式: {selected_mode}")
        
        # 验证配置
        is_valid, errors = self.validate_config(selected_mode)
        if not is_valid:
            error_msg = f"配置验证失败: {'; '.join(errors)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'mode': selected_mode,
                'config_errors': errors
            }
        
        # 创建训练器
        try:
            trainer = self.create_trainer(selected_mode)
            logger.info(f"创建{selected_mode}训练器成功")
        except Exception as e:
            error_msg = f"创建训练器失败: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'mode': selected_mode
            }
        
        # 执行训练
        try:
            start_time = time.time()
            logger.info(f"开始{selected_mode}模式训练")
            
            result = trainer.train()
            
            # 添加运行时信息
            result.update({
                'selected_mode': selected_mode,
                'training_time': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            if result.get('success', False):
                logger.info(f"{selected_mode}模式训练完成")
            else:
                logger.error(f"{selected_mode}模式训练失败: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"{selected_mode}模式训练异常: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'mode': selected_mode,
                'exception': str(e)
            }


# =====================================================
# 命令行接口和主入口
# =====================================================

def setup_argument_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='电力网络分区强化学习训练系统 v2.0 - 四模式解耦架构',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 核心参数
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['basic', 'topology', 'adaptive', 'unified', 'auto'],
                       default='auto', help='训练模式选择')
    parser.add_argument('--run', action='store_true',
                       help='立即开始训练')
    
    # 调试参数
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    parser.add_argument('--verbose', action='store_true',
                       help='详细日志输出')
    parser.add_argument('--dry-run', action='store_true',
                       help='模拟运行，验证配置')
    
    # 输出控制
    parser.add_argument('--output-dir', type=str,
                       help='输出目录路径')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='禁用TensorBoard日志')
    
    return parser


def load_and_merge_config(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """加载并合并配置
    
    Args:
        config_path: 配置文件路径
        args: 命令行参数
        
    Returns:
        合并后的配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析失败: {e}")
        raise
    
    # 应用模式配置
    if args.mode and args.mode != 'auto':
        mode_key = f"{args.mode}_mode"
        if mode_key in config:
            config[mode_key]['enabled'] = True
            logger.info(f"启用 {args.mode} 模式")
            
            # 禁用其他模式
            other_modes = ['basic', 'topology', 'adaptive', 'unified']
            for mode in other_modes:
                if mode != args.mode:
                    other_key = f"{mode}_mode"
                    if other_key in config:
                        config[other_key]['enabled'] = False
    
    # 应用命令行参数覆盖
    if args.debug:
        config['debug']['enabled'] = True
        config['debug']['verbose_logging'] = True
        if args.verbose:
            logger.info("已启用调试模式")

    # 传递verbose参数到配置
    if 'system' not in config:
        config['system'] = {}
    config['system']['verbose'] = args.verbose or args.debug
    
    if args.verbose:
        config['debug']['verbose_logging'] = True
        config['logging']['console_log_interval'] = 1
    
    if args.no_tensorboard:
        config['logging']['use_tensorboard'] = False
        logger.info("已禁用TensorBoard")
    
    if args.output_dir:
        config['logging']['log_dir'] = args.output_dir + '/logs'
        config['logging']['checkpoint_dir'] = args.output_dir + '/checkpoints'
        config['visualization']['figures_dir'] = args.output_dir + '/figures'
        logger.info(f"输出目录设置为: {args.output_dir}")
    
    return config


def merge_dict_recursive(base_dict: Dict, override_dict: Dict):
    """递归合并字典"""
    for key, value in override_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            merge_dict_recursive(base_dict[key], value)
        else:
            base_dict[key] = value


def print_system_info(verbose: bool = False):
    """打印系统信息"""
    if verbose:
        logger.info("=" * 60)
        logger.info("电力网络分区强化学习训练系统 v2.0")
        logger.info("=" * 60)
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            logger.info(f"当前GPU: {torch.cuda.get_device_name()}")
        logger.info(f"依赖库状态: {'✓ 正常' if DEPENDENCIES_OK else '✗ 错误'}")
        logger.info("=" * 60)
    else:
        print_header("电力网络分区强化学习训练系统 v2.0")
        device_info = f"CUDA ({torch.cuda.get_device_name()})" if torch.cuda.is_available() else "CPU"
        print_status(f"设备: {device_info}")
        if not DEPENDENCIES_OK:
            print_status("依赖库检查失败", "error")


def print_training_summary(result: Dict[str, Any], verbose: bool = False):
    """打印训练结果摘要"""
    if verbose:
        logger.info("=" * 60)
        logger.info("训练完成摘要")
        logger.info("=" * 60)

        if result.get('success', False):
            logger.info(f"✓ 训练成功")
            logger.info(f"模式: {result.get('selected_mode', 'unknown')}")
            logger.info(f"训练时间: {result.get('training_time', 0):.2f}秒")

            if 'best_reward' in result:
                logger.info(f"最佳奖励: {result['best_reward']:.4f}")

            if 'model_path' in result:
                logger.info(f"模型保存: {result['model_path']}")

            # 评估结果
            eval_stats = result.get('eval_stats', {})
            if eval_stats:
                logger.info(f"评估平均奖励: {eval_stats.get('avg_reward', 0):.4f}")
                logger.info(f"评估成功率: {eval_stats.get('success_rate', 0):.2%}")

        else:
            logger.error(f"✗ 训练失败")
            logger.error(f"错误: {result.get('error', 'Unknown error')}")

        logger.info("=" * 60)
    else:
        # 简洁输出
        print_header("训练完成")

        if result.get('success', False):
            print_status(f"训练成功 ({result.get('selected_mode', 'unknown')}模式)", "success")
            print_result("训练时间", f"{result.get('training_time', 0):.2f}秒")

            if 'best_reward' in result:
                print_result("最佳奖励", f"{result['best_reward']:.4f}")

            if 'model_path' in result:
                print_status(f"模型已保存: {Path(result['model_path']).name}")

            # 评估结果
            eval_stats = result.get('eval_stats', {})
            if eval_stats:
                print_result("评估平均奖励", f"{eval_stats.get('avg_reward', 0):.4f}")
                print_result("评估成功率", f"{eval_stats.get('success_rate', 0):.2%}")
        else:
            print_status(f"训练失败: {result.get('error', 'Unknown error')}", "error")


def main():
    """主入口函数"""
    # 解析命令行参数
    parser = setup_argument_parser()
    args = parser.parse_args()

    # 设置日志级别
    setup_logging(verbose=args.verbose, debug=args.debug)

    try:
        # 打印系统信息
        print_system_info(verbose=args.verbose)

        # 检查依赖
        if not DEPENDENCIES_OK:
            print_status(f"依赖库检查失败: {IMPORT_ERROR}", "error")
            print_status("请检查环境配置和依赖库安装", "error")
            return 1

        # 加载配置
        if args.verbose:
            logger.info(f"加载配置文件: {args.config}")
        config = load_and_merge_config(args.config, args)
        
        # 模拟运行模式
        if args.dry_run:
            print_header("模拟运行模式")
            print_status("验证配置和依赖")
            training_manager = TrainingManager(config)

            # 验证所有模式
            for mode in ['basic', 'topology', 'adaptive', 'unified']:
                is_valid, errors = training_manager.validate_config(mode)
                status = "success" if is_valid else "error"
                print_status(f"{mode}模式配置验证: {'通过' if is_valid else '失败'}", status)
                if errors and args.verbose:
                    for error in errors:
                        print_status(f"  - {error}", "warning")

            print_status("模拟运行完成", "success")
            return 0
        
        # 创建训练管理器
        training_manager = TrainingManager(config)
        
        # 确定训练模式
        training_mode = args.mode if args.mode != 'auto' else None
        
        # 非运行模式 - 仅显示信息
        if not args.run:
            selected_mode = training_manager.select_training_mode(training_mode)

            # 验证配置
            is_valid, errors = training_manager.validate_config(selected_mode)
            if is_valid:
                print_status(f"配置验证通过 ({selected_mode}模式)", "success")
                print_status("使用 --run 参数开始训练")
            else:
                print_status("配置验证失败", "error")
                if args.verbose:
                    for error in errors:
                        print_status(f"  - {error}", "error")
                return 1

            return 0

        # 执行训练
        print_header(f"开始{training_mode or 'auto'}模式训练")
        result = training_manager.run_training(training_mode)

        # 打印结果摘要
        print_training_summary(result, verbose=args.verbose)

        # 返回状态码
        return 0 if result.get('success', False) else 1
        
    except KeyboardInterrupt:
        print_status("用户中断训练", "warning")
        return 1
    except Exception as e:
        print_status(f"程序异常: {e}", "error")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())