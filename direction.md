# 电力网络分区强化学习系统 - 技术指导文档

本文档详细介绍电力网络分区强化学习系统的技术实现、参数配置、模块设计和扩展方法。

## 📋 目录

- [1. 系统架构](#1-系统架构)
- [2. 核心模块详解](#2-核心模块详解)
- [3. 参数配置指南](#3-参数配置指南)
- [4. 训练流程详解](#4-训练流程详解)
- [5. 性能优化](#5-性能优化)
- [6. 扩展开发](#6-扩展开发)
- [7. 调试和故障排除](#7-调试和故障排除)

## 1. 系统架构

### 1.1 整体设计

系统采用模块化设计，主要分为以下几个层次：

```
┌─────────────────────────────────────────────┐
│                主训练系统                    │
│            (main.py)                       │
├─────────────────────────────────────────────┤
│             配置管理层                       │
│    (config.yaml + 动态配置)                 │
├─────────────────────────────────────────────┤
│              算法实现层                      │
│  ┌─────────────┬─────────────┬─────────────┐ │
│  │  数据处理   │   图编码器   │   RL智能体   │ │
│  │ data_proc.. │    gat.py   │   agent.py  │ │
│  └─────────────┴─────────────┴─────────────┘ │
├─────────────────────────────────────────────┤  
│              环境仿真层                      │
│  ┌─────────┬─────────┬─────────┬─────────┐   │
│  │  环境   │  状态   │  动作   │  奖励   │   │
│  │ env.py  │state.py │action.. │reward.. │   │
│  └─────────┴─────────┴─────────┴─────────┘   │
├─────────────────────────────────────────────┤
│              基线对比层                      │
│  (baseline/ - 多种传统算法)                  │
├─────────────────────────────────────────────┤
│              监控可视化层                    │
│  (visualization.py + TensorBoard)           │
└─────────────────────────────────────────────┘
```

### 1.2 数据流

```
MATPOWER数据 → 异构图构建 → 图嵌入生成 → 状态表示 
     ↓
环境初始化 → 状态观察 → 智能体决策 → 动作执行 → 奖励计算
     ↓                     ↑
   环境更新 ← 分区更新 ← 动作应用   ↓
     ↓                          ↓
   终止检查 → 经验存储 → PPO更新 → 策略改进
```

## 2. 核心模块详解

### 2.1 数据处理模块 (`code/src/data_processing.py`)

#### 功能概述
- 处理MATPOWER格式电力系统数据
- 构建异构图数据结构
- 特征提取和预处理
- 数据缓存机制

#### 关键类：`PowerGridDataProcessor`

```python
class PowerGridDataProcessor:
    def __init__(self, normalize: bool = True, cache_dir: str = 'cache')
    
    # 主要方法
    def graph_from_mpc(self, mpc: Dict) -> HeteroData
    def _process_matpower_data(self, mpc: Dict)
    def _create_simplified_hetero_data(self, df_nodes, df_edges, df_edge_features)
```

#### 技术细节

**异构图设计**：
- 统一节点类型为'bus'，将物理类型(PQ/PV/Slack)转为独热编码特征
- 统一边关系为('bus', 'connects', 'bus')，保留物理类型作为边特征
- 支持GNN权重共享和高效学习

**特征工程**：
```python
# 节点特征 (9维基础 + 5维发电机 + 3维类型独热编码)
节点特征 = [Pd, Qd, Gs, Bs, Vm, Va, Vmax, Vmin, degree, 
          Pg, Qg, Pg_max, Pg_min, is_gen,
          type_1, type_2, type_3]

# 边特征 (9维)
边特征 = [r, x, b, |z|, y, rateA, angle_diff, is_transformer, status]
```

#### 参数配置
```python
# 在main.py中配置
data_config = {
    'normalize': True,           # 是否标准化特征
    'cache_dir': 'cache',       # 缓存目录
    'use_cache': True,          # 是否使用缓存
}
```

### 2.2 图神经网络模块 (`code/src/gat.py`)

#### 功能概述
- 物理增强的GATv2编码器
- 异构图处理
- 注意力机制可视化
- 节点和图级别嵌入生成

#### 关键类

**PhysicsGATv2Conv**：
```python
class PhysicsGATv2Conv(GATv2Conv):
    def __init__(self, in_channels, out_channels, heads=8, 
                 temperature=1.0, z_index=3, physics_weight=1.0)
    
    # 物理先验融入
    def physics_enhanced_edge_attr(self, edge_attr: torch.Tensor)
```

**HeteroGraphEncoder**：
```python
class HeteroGraphEncoder(nn.Module):
    def __init__(self, node_feature_dims, edge_feature_dims, metadata,
                 hidden_channels=64, gnn_layers=3, heads=4, dropout=0.3)
    
    # 主要方法
    def forward(self, data, return_attention_weights=False, return_graph_embedding=False)
    def encode_nodes(self, data)
    def encode_graph(self, data)
```

#### 技术特点

**物理先验融入**：
```python
# 标准GATv2: α_ij = softmax(W_a^T LeakyReLU(W_l[W_r h_i || W_r h_j] + b))
# 物理GATv2: α_ij = softmax(...... + τ/|Z_ij|)
# 其中|Z_ij|是线路阻抗模长，τ是可学习的温度参数
```

**注意力权重提取**：
- 自动收集所有GATv2层的注意力权重
- 按边类型组织注意力信息
- 支持可视化分析

#### 参数配置
```python
gnn_config = {
    'hidden_channels': 64,      # 隐藏层维度
    'gnn_layers': 3,           # GNN层数
    'heads': 4,                # 注意力头数
    'dropout': 0.3,            # Dropout概率
    'output_dim': 64,          # 输出嵌入维度
    'temperature': 1.0,        # 物理先验温度参数
    'physics_weight': 1.0,     # 物理权重系数
}
```

### 2.3 强化学习智能体 (`code/src/rl/agent.py`)

#### 功能概述
- PPO算法实现
- 两阶段动作选择
- 动作掩码处理
- 数值稳定性保证

#### 关键类

**PPOAgent**：
```python
class PPOAgent:
    def __init__(self, node_embedding_dim, region_embedding_dim, num_partitions,
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, eps_clip=0.2,
                 k_epochs=4, entropy_coef=0.01, value_coef=0.5)
    
    # 主要方法
    def select_action(self, state, training=True)
    def store_experience(self, state, action, reward, log_prob, value, done)
    def update(self)
```

**ActorNetwork & CriticNetwork**：
```python
class ActorNetwork(nn.Module):
    # 两阶段输出：节点选择 + 分区选择
    def forward(self, node_embeddings, region_embeddings, boundary_nodes, action_mask)
    
class CriticNetwork(nn.Module):
    # 状态价值估计
    def forward(self, node_embeddings, region_embeddings, boundary_nodes)
```

#### 技术特点

**两阶段动作空间**：
1. 阶段1：从边界节点中选择要移动的节点
2. 阶段2：为选中节点选择目标分区

**数值稳定性**：
```python
# 安全的掩码softmax
def masked_softmax(logits, mask, dim=-1, epsilon=1e-12):
    masked_logits = logits.masked_fill(~mask, -1e9)  # 避免-inf
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.float()
    probs_sum = probs.sum(dim=dim, keepdim=True).clamp(min=epsilon)
    return probs / probs_sum

# 安全的对数概率
def safe_log_prob(probs, epsilon=1e-12):
    return torch.log(probs.clamp(min=epsilon))
```

#### 参数配置
```python
ppo_config = {
    'lr_actor': 3e-4,          # Actor学习率
    'lr_critic': 1e-3,         # Critic学习率  
    'gamma': 0.99,             # 折扣因子
    'eps_clip': 0.2,           # PPO裁剪参数
    'k_epochs': 4,             # PPO训练轮数
    'entropy_coef': 0.01,      # 熵系数
    'value_coef': 0.5,         # 价值损失系数
    'max_grad_norm': 0.5,      # 梯度裁剪
}
```

### 2.4 环境模块 (`code/src/rl/environment.py`)

#### 功能概述
- MDP环境定义
- 状态转移逻辑
- 终止条件检查
- 观察空间生成

#### 关键类：`PowerGridPartitioningEnv`

```python
class PowerGridPartitioningEnv:
    def __init__(self, hetero_data, node_embeddings, num_partitions,
                 reward_weights=None, max_steps=200, device=None)
    
    # 主要方法
    def reset(self, seed=None)
    def step(self, action)
    def _check_termination(self)
    def get_action_mask(self)
```

#### 状态空间设计

```python
observation = {
    'node_embeddings': torch.Tensor,      # 节点嵌入 [N, D_node]
    'region_embeddings': torch.Tensor,    # 区域嵌入 [K, D_region]  
    'boundary_nodes': torch.Tensor,       # 边界节点索引 [N_boundary]
    'current_partition': torch.Tensor,    # 当前分区分配 [N]
    'partition_sizes': torch.Tensor,      # 各分区大小 [K]
    'step_count': int,                    # 当前步数
}
```

#### 动作空间设计

```python
# 动作表示：(node_idx, target_partition)
action = (int, int)  # (要移动的节点, 目标分区)

# 动作掩码：[N, K] 布尔张量
action_mask[i, j] = True   # 节点i可以移动到分区j+1
```

#### 参数配置
```python
env_config = {
    'num_partitions': 4,       # 目标分区数
    'max_steps': 200,          # 最大步数
    'reward_weights': {        # 奖励权重
        'load_balance': 0.4,
        'electrical_decoupling': 0.4,
        'power_balance': 0.2
    },
    'early_termination': True, # 早停机制
    'convergence_window': 10,  # 收敛窗口
    'convergence_threshold': 0.01, # 收敛阈值
}
```

### 2.5 奖励函数 (`code/src/rl/reward.py`)

#### 功能概述
- 多目标奖励设计
- 负载平衡评估
- 电气解耦评估
- 功率平衡评估

#### 关键类：`RewardFunction`

```python
class RewardFunction:
    def __init__(self, hetero_data, reward_weights=None, device=None)
    
    def compute_reward(self, current_partition, boundary_nodes, action,
                      return_components=False)
    
    # 子奖励计算
    def _compute_load_balance_reward(self, current_partition)
    def _compute_electrical_decoupling_reward(self, current_partition)  
    def _compute_power_balance_reward(self, current_partition)
```

#### 奖励设计

**负载平衡奖励**：
```python
# 基于变异系数
load_cv = std(partition_loads) / mean(partition_loads)
load_balance_reward = -load_cv  # 越小奖励越高
```

**电气解耦奖励**：
```python
# 基于跨分区线路数量
inter_partition_lines = count_inter_partition_connections()
coupling_reward = -inter_partition_lines / total_lines
```

**功率平衡奖励**：
```python
# 各分区功率不平衡度
power_imbalance = |generation - load| for each partition
balance_reward = -mean(power_imbalance)
```

#### 参数配置
```python
reward_config = {
    'weights': {
        'load_balance': 0.4,           # 负载平衡权重
        'electrical_decoupling': 0.4,  # 电气解耦权重
        'power_balance': 0.2,          # 功率平衡权重
    },
    'normalization': True,             # 是否归一化奖励
    'debug_mode': False,               # 调试模式
}
```

## 3. 参数配置指南

### 3.1 配置文件结构

主配置文件 `config.yaml` 的结构：

```yaml
# 环境配置
environment:
  case_name: "ieee30"
  num_partitions: 4
  max_steps: 200
  reward_weights:
    load_balance: 0.4
    electrical_decoupling: 0.4
    power_balance: 0.2

# 模型配置  
model:
  hidden_channels: 64
  gnn_layers: 3
  heads: 4
  dropout: 0.3
  output_dim: 64

# 训练配置
training:
  episodes: 1000
  learning_rate: 0.0003
  batch_size: 32
  update_interval: 10
  gamma: 0.99
  eps_clip: 0.2
  entropy_coef: 0.01

# 系统配置
system:
  device: "auto"
  num_workers: 4
  seed: 42
  use_tensorboard: true
  save_interval: 100
```

### 3.2 关键参数调优

#### 学习率调整

```python
# 不同网络组件的学习率
lr_config = {
    'lr_actor': 3e-4,      # Actor网络学习率
    'lr_critic': 1e-3,     # Critic网络学习率（通常更大）
    'lr_scheduler': {      # 学习率调度
        'type': 'cosine',
        'T_max': 1000,
        'eta_min': 1e-5
    }
}
```

#### 网络架构调整

```python
# GNN架构参数
gnn_params = {
    'hidden_channels': [32, 64, 128],  # 可选维度
    'gnn_layers': [2, 3, 4, 5],       # 推荐3-4层
    'heads': [2, 4, 8],               # 注意力头数
    'dropout': [0.1, 0.3, 0.5],      # Dropout率
}

# 选择原则：
# - 小网络(IEEE14/30): hidden_channels=32, layers=2-3
# - 中等网络(IEEE57): hidden_channels=64, layers=3-4  
# - 大网络(IEEE118+): hidden_channels=128, layers=4-5
```

#### PPO超参数

```python
# PPO关键参数调优
ppo_params = {
    'gamma': [0.95, 0.99, 0.995],     # 折扣因子
    'eps_clip': [0.1, 0.2, 0.3],      # 裁剪参数
    'k_epochs': [3, 4, 5, 8],         # 更新轮数
    'entropy_coef': [0.001, 0.01, 0.1], # 熵系数
}

# 调优建议：
# - gamma: 长期任务用0.99，短期任务用0.95
# - eps_clip: 复杂环境用0.1-0.2，简单环境用0.2-0.3
# - entropy_coef: 初期用0.01-0.1，后期可降低
```

### 3.3 多目标权重调整

```python
# 奖励权重的调整策略
reward_weight_strategies = {
    # 平衡策略（默认）
    'balanced': {
        'load_balance': 0.4,
        'electrical_decoupling': 0.4, 
        'power_balance': 0.2
    },
    
    # 负载优先策略
    'load_focused': {
        'load_balance': 0.6,
        'electrical_decoupling': 0.3,
        'power_balance': 0.1
    },
    
    # 拓扑优先策略
    'topology_focused': {
        'load_balance': 0.2,
        'electrical_decoupling': 0.6,
        'power_balance': 0.2
    },
    
    # 功率平衡优先策略
    'power_focused': {
        'load_balance': 0.3,
        'electrical_decoupling': 0.3,
        'power_balance': 0.4
    }
}
```

## 4. 训练流程详解

### 4.1 训练管道

```python
# 完整训练流程
def training_pipeline():
    # 1. 环境初始化
    env = setup_environment(config)
    
    # 2. 模型创建
    encoder = create_hetero_graph_encoder(data, **model_config)
    agent = PPOAgent(**agent_config)
    
    # 3. 训练循环
    for episode in range(num_episodes):
        # 环境重置
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        # 单回合交互
        while not done:
            # 智能体决策
            action, log_prob, value = agent.select_action(obs)
            
            # 环境步进
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 经验存储
            agent.store_experience(obs, action, reward, log_prob, value, done)
            
            obs = next_obs
            episode_reward += reward
        
        # PPO更新
        if episode % update_interval == 0:
            agent.update()
        
        # 日志记录
        logger.log_episode(episode, episode_reward, info)
    
    # 4. 结果评估
    evaluate_and_compare(env, agent)
```

### 4.2 训练模式实现

#### Quick模式（快速验证）
```python
quick_config = {
    'episodes': 100,
    'max_steps': 50,
    'hidden_channels': 32,
    'gnn_layers': 2,
    'update_interval': 5,
    'save_interval': 50,
}
```

#### Standard模式（标准训练）  
```python
standard_config = {
    'episodes': 1000,
    'max_steps': 200,
    'hidden_channels': 64,
    'gnn_layers': 3,
    'update_interval': 10,
    'save_interval': 100,
}
```

#### Full模式（完整训练）
```python
full_config = {
    'episodes': 5000,
    'max_steps': 300,
    'hidden_channels': 128,
    'gnn_layers': 4,
    'update_interval': 20,
    'save_interval': 200,
    'curriculum_learning': True,
    'early_stopping': True,
}
```

#### Parallel模式（并行训练）
```python
parallel_config = {
    'num_workers': 8,
    'episodes_per_worker': 125,  # 总episodes = 8 * 125 = 1000
    'sync_interval': 50,
    'shared_memory': True,
}
```

### 4.3 训练监控和调试

#### TensorBoard集成
```python
# 日志记录类别
tensorboard_logs = {
    'scalars': {
        'reward/episode_reward',
        'reward/mean_reward',
        'loss/actor_loss',
        'loss/critic_loss',
        'metrics/entropy',
        'metrics/load_cv',
        'metrics/coupling_degree',
    },
    'histograms': {
        'gradients/actor_gradients',
        'gradients/critic_gradients', 
        'weights/actor_weights',
        'weights/critic_weights',
    },
    'images': {
        'partition/current_partition',
        'attention/attention_weights',
    }
}
```

#### 检查点保存
```python
checkpoint_config = {
    'save_best': True,          # 保存最佳模型
    'save_latest': True,        # 保存最新模型
    'save_interval': 100,       # 定期保存间隔
    'max_checkpoints': 5,       # 最大保存数量
    'resume_training': True,    # 支持恢复训练
}
```

## 5. 性能优化

### 5.1 计算优化

#### GPU内存优化
```python
# 内存优化策略
memory_optimization = {
    'gradient_checkpointing': True,    # 梯度检查点
    'mixed_precision': True,           # 混合精度训练
    'batch_size_scaling': 'auto',      # 自动批量大小
    'cache_management': 'smart',       # 智能缓存管理
}

# 实现示例
def optimize_gpu_memory():
    # 启用混合精度
    scaler = torch.cuda.amp.GradScaler()
    
    # 动态批量大小调整
    try:
        batch_size = initial_batch_size
        while True:
            try:
                # 尝试当前批量大小
                train_batch(batch_size)
                break
            except torch.cuda.OutOfMemoryError:
                batch_size //= 2
                torch.cuda.empty_cache()
    except:
        raise RuntimeError("GPU内存不足")
```

#### 并行计算优化
```python
# 多进程训练配置
multiprocessing_config = {
    'method': 'spawn',              # 进程启动方法
    'shared_model': True,           # 共享模型权重
    'async_update': True,           # 异步参数更新
    'gradient_averaging': True,     # 梯度平均
}

# 数据并行
def setup_data_parallel():
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行")
```

### 5.2 算法优化

#### 经验回放优化
```python
# 优先经验回放
class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        
    def add(self, state, action, reward, next_state, done, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        # 存储经验和优先级
        
    def sample(self, batch_size):
        # 基于优先级采样
        pass
```

#### 课程学习
```python
# 课程学习策略
curriculum_config = {
    'stages': [
        {'episodes': 200, 'case': 'ieee14', 'partitions': 3},
        {'episodes': 300, 'case': 'ieee30', 'partitions': 4}, 
        {'episodes': 500, 'case': 'ieee57', 'partitions': 5},
    ],
    'transition_criterion': 'reward_threshold',
    'reward_thresholds': [-0.5, -0.3, -0.1],
}
```

### 5.3 系统优化

#### 缓存策略
```python
# 多级缓存系统
cache_config = {
    'levels': {
        'memory': {
            'size': '2GB',
            'eviction': 'LRU',
        },
        'disk': {
            'size': '10GB', 
            'format': 'compressed',
        },
        'network': {
            'enabled': False,
            'backend': 'redis',
        }
    }
}
```

#### I/O优化
```python
# 异步I/O配置
async_io_config = {
    'data_loading': {
        'num_workers': 4,
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    'logging': {
        'buffer_size': 1024,
        'flush_interval': 10,
        'async_write': True,
    }
}
```

## 6. 扩展开发

### 6.1 添加新的基线算法

```python
# 1. 创建新的分区器类
class MyCustomPartitioner(BasePartitioner):
    def __init__(self, seed=42, **kwargs):
        super().__init__(seed)
        self.custom_params = kwargs
    
    def partition(self, env):
        """实现自定义分区算法"""
        # 获取网络信息
        adjacency_matrix = self._build_adjacency(env)
        node_features = self._extract_features(env)
        
        # 实现分区逻辑
        labels = self._my_partition_algorithm(
            adjacency_matrix, node_features, env.num_partitions
        )
        
        return labels + 1  # 转换为1-based标签

# 2. 注册到基线系统
def register_custom_partitioner():
    from code.baseline import baseline_registry
    baseline_registry['my_method'] = MyCustomPartitioner

# 3. 在比较中使用
comparison_methods = ['spectral', 'kmeans', 'random', 'my_method']
```

### 6.2 自定义奖励函数

```python
# 扩展奖励函数
class ExtendedRewardFunction(RewardFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_weights = kwargs.get('custom_weights', {})
    
    def compute_reward(self, current_partition, boundary_nodes, action,
                      return_components=False):
        # 调用基础奖励
        base_reward = super().compute_reward(
            current_partition, boundary_nodes, action, return_components=True
        )
        
        # 添加自定义组件
        custom_components = self._compute_custom_rewards(current_partition)
        
        # 合并奖励
        total_reward = self._combine_rewards(base_reward, custom_components)
        
        return total_reward if not return_components else (total_reward, base_reward, custom_components)
    
    def _compute_custom_rewards(self, partition):
        """实现自定义奖励组件"""
        rewards = {}
        
        # 示例：连通性奖励
        rewards['connectivity'] = self._compute_connectivity_reward(partition)
        
        # 示例：紧密性奖励
        rewards['compactness'] = self._compute_compactness_reward(partition)
        
        return rewards
```

### 6.3 网络架构扩展

```python
# 自定义GNN层
class CustomGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # 实现自定义图卷积层
        
    def forward(self, x, edge_index, edge_attr=None):
        # 自定义前向传播逻辑
        pass

# 集成到编码器
class ExtendedHeteroGraphEncoder(HeteroGraphEncoder):
    def __init__(self, *args, custom_layer_type='default', **kwargs):
        super().__init__(*args, **kwargs)
        
        if custom_layer_type == 'custom':
            self._replace_gnn_layers()
    
    def _replace_gnn_layers(self):
        """替换默认GNN层为自定义层"""
        # 实现层替换逻辑
        pass
```

### 6.4 环境扩展

```python
# 自定义环境变体
class MultiObjectivePartitionEnv(PowerGridPartitioningEnv):
    """多目标分区环境"""
    
    def __init__(self, *args, objectives=['load', 'topology', 'stability'], **kwargs):
        super().__init__(*args, **kwargs)
        self.objectives = objectives
        self.pareto_front = []
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 计算多目标指标
        objective_values = self._compute_objectives()
        info['objectives'] = objective_values
        
        # 更新帕累托前沿
        self._update_pareto_front(objective_values)
        
        return obs, reward, terminated, truncated, info
    
    def _compute_objectives(self):
        """计算多个目标函数值"""
        objectives = {}
        for obj in self.objectives:
            objectives[obj] = getattr(self, f'_compute_{obj}_objective')()
        return objectives
```

## 7. 调试和故障排除

### 7.1 常见错误诊断

#### 数值不稳定问题
```python
# 检查点列表
numerical_stability_checklist = {
    'gradients': {
        'nan_check': 'torch.isnan(gradients).any()',
        'inf_check': 'torch.isinf(gradients).any()',
        'magnitude_check': 'gradients.norm() > 1000',
    },
    'losses': {
        'nan_loss': 'torch.isnan(loss)',
        'negative_loss': 'loss < 0 (for non-adversarial)',
        'exploding_loss': 'loss > 1e6',
    },
    'probabilities': {
        'sum_check': 'probs.sum() != 1.0',
        'negative_prob': '(probs < 0).any()',
        'zero_prob': '(probs == 0).all()',
    }
}

# 自动诊断函数
def diagnose_numerical_issues(model, loss, gradients, probs):
    issues = []
    
    # 检查梯度
    if torch.isnan(gradients).any():
        issues.append("NaN梯度检测")
    
    # 检查损失
    if torch.isnan(loss):
        issues.append("NaN损失检测")
        
    # 检查概率
    if (probs < 0).any() or not torch.allclose(probs.sum(), torch.tensor(1.0)):
        issues.append("无效概率分布")
    
    return issues
```

#### 内存泄漏诊断
```python
# 内存监控
import psutil
import gc

class MemoryMonitor:
    def __init__(self):
        self.initial_memory = psutil.virtual_memory().used
        
    def check_memory_usage(self, step):
        current_memory = psutil.virtual_memory().used
        memory_increase = current_memory - self.initial_memory
        
        if memory_increase > 1e9:  # 1GB
            print(f"步骤 {step}: 内存使用增加 {memory_increase/1e6:.1f}MB")
            self._suggest_cleanup()
    
    def _suggest_cleanup(self):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

### 7.2 性能分析工具

#### 训练性能分析
```python
# 性能分析器
class TrainingProfiler:
    def __init__(self):
        self.timers = {}
        self.counters = {}
    
    def start_timer(self, name):
        self.timers[name] = time.time()
    
    def end_timer(self, name):
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            self.counters[name] = self.counters.get(name, []) + [elapsed]
    
    def report(self):
        for name, times in self.counters.items():
            avg_time = sum(times) / len(times)
            print(f"{name}: 平均 {avg_time:.4f}s, 总计 {sum(times):.2f}s")

# 使用示例
profiler = TrainingProfiler()

# 在训练循环中
profiler.start_timer('forward_pass')
output = model(input)
profiler.end_timer('forward_pass')

profiler.start_timer('backward_pass') 
loss.backward()
profiler.end_timer('backward_pass')
```

### 7.3 调试配置

```python
# 调试模式配置
debug_config = {
    'verbose': True,               # 详细输出
    'check_gradients': True,       # 梯度检查
    'profile_memory': True,        # 内存分析
    'save_intermediate': True,     # 保存中间结果
    'visualize_attention': True,   # 可视化注意力
    'log_rewards': True,          # 记录奖励组件
}

# 调试输出示例
def debug_training_step(episode, step, reward_components, attention_weights):
    if debug_config['verbose']:
        print(f"回合 {episode}, 步骤 {step}:")
        print(f"  奖励组件: {reward_components}")
        print(f"  注意力权重形状: {[w.shape for w in attention_weights]}")
        
    if debug_config['save_intermediate']:
        torch.save({
            'episode': episode,
            'step': step,
            'reward_components': reward_components,
            'attention_weights': attention_weights,
        }, f'debug/step_{episode}_{step}.pt')
```

---

本文档涵盖了电力网络分区强化学习系统的主要技术细节。对于特定问题或高级定制需求，建议查阅源代码中的详细注释或联系开发团队。 