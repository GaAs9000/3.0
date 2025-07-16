import gymnasium as gym
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional

# === 可选：多尺度拓扑生成 ===
try:
    from rl.scale_aware_generator import ScaleAwareSyntheticGenerator
except ImportError:
    ScaleAwareSyntheticGenerator = None  # 运行时按需检查
from torch_geometric.data import HeteroData

try:
    from .environment import PowerGridPartitioningEnv
    from .scenario_generator import ScenarioGenerator
    from .scenario_context import ScenarioContext
    from data_processing import PowerGridDataProcessor
    from gat import create_hetero_graph_encoder, HeteroGraphEncoder
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from rl.environment import PowerGridPartitioningEnv
    from rl.scenario_generator import ScenarioGenerator
    from rl.scenario_context import ScenarioContext
    from data_processing import PowerGridDataProcessor
    from gat import create_hetero_graph_encoder, HeteroGraphEncoder


class PowerGridPartitionGymEnv(gym.Env):
    """
    兼容Gymnasium的电力网络分区环境
    扩展特性：
    1. 场景生成 (负荷/故障扰动)
    2. 多尺度拓扑生成 (ScaleAwareSyntheticGenerator)
    3. 并行训练友好
    """
    
    def __init__(self, 
                 base_case_data: Dict,
                 config: Dict[str, Any],
                 gat_encoder: Optional["HeteroGraphEncoder"] = None,
                 use_scenario_generator: bool = True,
                 scenario_seed: Optional[int] = None,
                 scale_generator: Optional["ScaleAwareSyntheticGenerator"] = None):
        """
        初始化Gymnasium环境
        
        Args:
            base_case_data: 基础电网案例数据（MATPOWER格式）
            config: 完整配置字典
            gat_encoder: 预训练的GAT编码器实例（避免重复创建）
            use_scenario_generator: 是否使用场景生成器
            scenario_seed: 场景生成器的随机种子
            scale_generator: 多尺度拓扑生成器
        """
        super().__init__()
        
        self.base_case = base_case_data
        self.config = config
        self.device = torch.device(config['system']['device'])
        
        # 🔧 重要修复：接收并验证GAT编码器
        self.gat_encoder = gat_encoder
        if self.gat_encoder is not None:
            # 确保编码器在正确的设备上
            self.gat_encoder = self.gat_encoder.to(self.device)
            # 设置为评估模式以保持权重一致性
            self.gat_encoder.eval()
        
        # 环境基础参数（在调用 _setup_spaces 之前定义）
        self.num_partitions = config['environment']['num_partitions']
        self.max_steps = config['environment']['max_steps']
        self.reward_weights = config['environment']['reward_weights']
        
        # 初始化数据处理器
        self.processor = PowerGridDataProcessor(
            normalize=config['data']['normalize'],
            cache_dir=config['data']['cache_dir']
        )
        
        # === 1) 场景/拓扑生成器选项 ===
        self.use_scenario_generator = use_scenario_generator
        self.scale_generator = scale_generator
        self.use_multi_scale_generator = self.scale_generator is not None

        # 如果启用了传统场景扰动且未启用多尺度生成器，则实例化ScenarioGenerator
        if self.use_scenario_generator and not self.use_multi_scale_generator:
            self.scenario_generator = ScenarioGenerator(base_case_data, scenario_seed, config)
        else:
            self.scenario_generator = None
        
        # 预处理基础案例以获取维度信息
        self._setup_spaces()
        
        # 内部环境实例（在reset时创建）
        self.internal_env = None
        self.current_hetero_data = None
        self.current_node_embeddings = None
        self.current_attention_weights = None
        
    def _setup_spaces(self):
        """设置动作和观察空间"""
        # 使用基础案例获取图的维度
        base_hetero = self.processor.graph_from_mpc(self.base_case)
        
        # 计算总节点数
        total_nodes = sum(x.shape[0] for x in base_hetero.x_dict.values())
        
        # 动作空间：(node_idx, target_partition)
        # 注意：对于多尺度生成器，节点数可能变化；
        # 这里只用于与Gym兼容，在本项目训练流程中不会直接使用。
        self.action_space = gym.spaces.Discrete(total_nodes * self.num_partitions)
        
        # 观察空间：使用GAT输出维度
        gat_output_dim = self.config['gat']['output_dim']
        
        # 观察包括：
        # - 节点嵌入 (total_nodes, gat_output_dim)
        # - 当前分区 (total_nodes,)
        # - 分区统计信息 (num_partitions, 4) - 每个分区的大小、负载、发电、边界节点数
        obs_dim = total_nodes * (gat_output_dim + 1) + self.num_partitions * 4
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 存储维度信息
        self.total_nodes = total_nodes
        self.gat_output_dim = gat_output_dim
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            
        Returns:
            observation: 初始观察（numpy数组）
            info: 附加信息
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # === 生成新的电网案例 ===
        if self.use_multi_scale_generator:
            # 使用多尺度生成器生成完全新的拓扑
            # 使用一个简单均匀权重或根据配置自定义
            try:
                stage_cfg = {"small": 0.5, "medium": 0.3, "large": 0.2}
                sample = self.scale_generator.generate_batch(stage_cfg, batch_size=1)[0]
                current_case, _k = sample  # 忽略推荐的K，保持配置一致
                scenario_context = ScenarioContext()
            except Exception as e:
                print(f"⚠️ ScaleAwareSyntheticGenerator 生成失败: {e}, 回退到基础案例")
                current_case = self.base_case
                scenario_context = ScenarioContext()
        elif self.use_scenario_generator:
            current_case, scenario_context = self.scenario_generator.generate_random_scene()
        else:
            current_case = self.base_case
            scenario_context = ScenarioContext()  # 默认场景上下文
        
        # 将案例转换为异构图
        self.current_hetero_data = self.processor.graph_from_mpc(current_case, self.config).to(self.device)
        
        # 🔧 重要修复：使用传入的预训练GAT编码器，避免重复创建
        if self.gat_encoder is not None:
            # 使用预训练的编码器
            encoder = self.gat_encoder
        else:
            # 回退：如果没有传入编码器，则创建新的（保持向后兼容）
            from gat import create_hetero_graph_encoder
            gat_config = self.config['gat'].copy()
            supported_params = ['hidden_channels', 'gnn_layers', 'heads', 'output_dim', 'dropout']
            filtered_gat_config = {k: v for k, v in gat_config.items() if k in supported_params}
            
            encoder = create_hetero_graph_encoder(
                self.current_hetero_data,
                **filtered_gat_config
            ).to(self.device)
            print("⚠️ 警告：未传入预训练编码器，创建新编码器实例（可能导致性能下降）")
        
        # 生成节点嵌入
        with torch.no_grad():
            self.current_node_embeddings, self.current_attention_weights = \
                encoder.encode_nodes_with_attention(self.current_hetero_data, self.config)
        
        # 创建内部环境（使用增强环境以支持分区特征）
        from .enhanced_environment import EnhancedPowerGridPartitioningEnv
        self.internal_env = EnhancedPowerGridPartitioningEnv(
            hetero_data=self.current_hetero_data,
            node_embeddings=self.current_node_embeddings,
            num_partitions=self.num_partitions,
            reward_weights=self.reward_weights,
            max_steps=self.max_steps,
            device=self.device,
            attention_weights=self.current_attention_weights,
            config=self.config,
            enable_features_cache=True
        )
        
        # 🔧 重要修复：确保内部环境也有gat_encoder属性
        if encoder is not None:
            self.internal_env.gat_encoder = encoder
        
        # 重置内部环境
        obs_dict, info = self.internal_env.reset(scenario_context=scenario_context)
        
        # 将观察转换为numpy数组
        obs_array = self._dict_obs_to_array(obs_dict)
        
        return obs_array, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作ID（离散）
            
        Returns:
            observation: 下一观察
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 附加信息
        """
        # 解码动作
        node_idx = action // self.num_partitions
        target_partition = (action % self.num_partitions) + 1  # 分区从1开始
        
        # 执行动作
        obs_dict, reward, terminated, truncated, info = self.internal_env.step(
            (node_idx, target_partition)
        )
        
        # 转换观察
        obs_array = self._dict_obs_to_array(obs_dict)
        
        return obs_array, reward, terminated, truncated, info
    
    def _dict_obs_to_array(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        将字典形式的观察转换为numpy数组
        
        Args:
            obs_dict: 字典形式的观察
            
        Returns:
            扁平化的numpy数组
        """
        obs_parts = []
        
        # 添加节点嵌入
        if 'node_embeddings' in obs_dict:
            emb = obs_dict['node_embeddings']
            obs_parts.append(emb.cpu().numpy().flatten())
        
        # 添加当前分区
        if 'current_partition' in obs_dict:
            partition = obs_dict['current_partition']
            obs_parts.append(partition.cpu().numpy().flatten())
        
        # 添加分区统计
        if 'partition_stats' in obs_dict:
            stats = obs_dict['partition_stats']
            obs_parts.append(stats.cpu().numpy().flatten())
        
        # 连接所有部分
        obs_array = np.concatenate(obs_parts).astype(np.float32)
        
        return obs_array
    
    def get_action_mask(self) -> np.ndarray:
        """
        获取有效动作掩码
        
        Returns:
            布尔掩码，指示哪些动作有效
        """
        if self.internal_env is None:
            return np.ones(self.action_space.n, dtype=bool)
        
        # 获取有效动作
        valid_actions = self.internal_env.action_space.get_valid_actions(
            self.internal_env.state_manager.current_partition,
            self.internal_env.state_manager.get_boundary_nodes()
        )
        
        # 创建掩码
        mask = np.zeros(self.action_space.n, dtype=bool)
        for node_idx, target_partition in valid_actions:
            action_id = node_idx * self.num_partitions + (target_partition - 1)
            mask[action_id] = True
        
        return mask
    
    def render(self, mode='human'):
        """渲染环境"""
        if self.internal_env is not None:
            return self.internal_env.render(mode)
    
    def close(self):
        """关闭环境"""
        if self.internal_env is not None:
            self.internal_env.close()
    
    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.scenario_generator is not None:
                self.scenario_generator = ScenarioGenerator(self.base_case, seed, self.config)
        return [seed]


def make_parallel_env(base_case_data: Dict,
                     config: Dict[str, Any],
                     num_envs: int = 12,
                     use_scenario_generator: bool = True) -> gym.vector.VectorEnv:
    """
    创建并行向量化环境
    
    Args:
        base_case_data: 基础电网案例数据
        config: 配置字典
        num_envs: 并行环境数量
        use_scenario_generator: 是否使用场景生成器
        
    Returns:
        向量化环境
    """
    def make_env(rank: int):
        def _init():
            env = PowerGridPartitionGymEnv(
                base_case_data=base_case_data,
                config=config,
                use_scenario_generator=use_scenario_generator,
                scenario_seed=rank  # 每个环境使用不同的种子
            )
            env.seed(rank)
            return env
        return _init
    
    # 创建环境列表
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # 创建向量化环境
    return gym.vector.AsyncVectorEnv(env_fns) 