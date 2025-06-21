# 电力网络分区强化学习系统技术文档

本文档详细介绍系统各个模块的技术实现和使用方法。

## 📁 系统架构

```
src/
├── data_processing.py     # 异构图数据处理
├── gat.py                # 物理增强GAT编码器
├── visualization.py      # 可视化管理器
└── rl/                   # 强化学习模块
    ├── environment.py    # 分区环境
    ├── agent.py          # PPO智能体
    ├── state.py          # 状态管理
    ├── action_space.py   # 动作空间
    ├── reward.py         # 奖励函数
    ├── utils.py          # 工具函数
    ├── scenario_generator.py  # 场景生成器
    └── gym_wrapper.py    # OpenAI Gym包装器
```

## 🔧 核心模块详解

### 1. 数据处理模块 (`src/data_processing.py`)

**功能**: 将MATPOWER格式的电力系统数据转换为PyTorch Geometric异构图

**核心类**: `PowerGridDataProcessor`

**主要特性**:
- **异构图构建** - 区分不同类型的节点（PQ/PV/Slack）和边（线路/变压器）
- **物理特征提取** - 14维节点特征，9维边特征
- **智能缓存** - MD5哈希缓存机制，避免重复计算
- **标准化处理** - RobustScaler统一特征尺度

**使用示例**:
```python
from src.data_processing import PowerGridDataProcessor

# 创建处理器
processor = PowerGridDataProcessor(normalize=True, cache_dir='cache')

# 处理MATPOWER数据
hetero_data = processor.graph_from_mpc(mpc_data)

# 获取节点和边特征
node_features = hetero_data.x_dict
edge_features = hetero_data.edge_attr_dict
```

**技术细节**:
- 支持IEEE标准测试系统（14/30/57/118节点）
- 自动处理缺失值和异常数据
- 生成无向异构图结构
- 保持全局索引到局部索引的映射关系

### 2. GAT编码器模块 (`src/gat.py`)

**功能**: 基于物理约束的异构图注意力网络编码器

**核心类**: 
- `PhysicsGATv2Conv` - 物理增强的GAT卷积层
- `HeteroGraphEncoder` - 异构图编码器

**主要特性**:
- **物理增强注意力** - 电气阻抗指导的注意力机制
- **异构图支持** - 自动处理不同类型的节点和边
- **多层架构** - 3-4层GNN捕获长程依赖
- **注意力权重提取** - 支持可解释性分析

**使用示例**:
```python
from src.gat import create_hetero_graph_encoder

# 创建编码器
encoder = create_hetero_graph_encoder(
    data=hetero_data,
    hidden_channels=64,
    gnn_layers=3,
    heads=4,
    output_dim=128
)

# 编码节点
node_embeddings = encoder.encode_nodes(hetero_data)

# 获取注意力权重
node_emb, attention_weights = encoder.encode_nodes_with_attention(hetero_data)
```

**技术细节**:
- 基于GATv2架构，具有更强表达能力
- 融入电气阻抗先验：α_ij = softmax(attention + τ/|Z_ij|)
- 支持残差连接和层归一化
- 可学习的温度参数控制物理先验影响

### 3. 强化学习环境 (`src/rl/environment.py`)

**功能**: 电力网络分区的MDP环境实现

**核心类**: `PowerGridPartitioningEnv`

**主要特性**:
- **完整MDP建模** - 状态、动作、奖励、转移
- **约束处理** - 连通性、拓扑约束自动满足
- **状态管理** - 高效的增量状态更新
- **终止条件** - 收敛检测和最大步数限制

**状态空间**:
- 节点特征嵌入 (H): 从GAT编码器获得
- 节点分配标签 (z_t): 动态分区分配
- 边界节点 (Bdry_t): 与不同分区相邻的节点
- 区域嵌入: 每个分区的聚合嵌入

**动作空间**:
- 两阶段决策: 节点选择 → 分区选择
- 动作掩码: 只有边界节点可移动
- 约束满足: 连通性和邻接约束

**使用示例**:
```python
from src.rl.environment import PowerGridPartitioningEnv

# 创建环境
env = PowerGridPartitioningEnv(
    hetero_data=hetero_data,
    node_embeddings=node_embeddings,
    num_partitions=3,
    max_steps=200
)

# 重置环境
state, info = env.reset()

# 执行动作
action = (node_idx, target_partition)
next_state, reward, done, truncated, info = env.step(action)
```

### 4. PPO智能体 (`src/rl/agent.py`)

**功能**: 基于PPO算法的强化学习智能体

**核心类**: 
- `ActorNetwork` - 策略网络
- `CriticNetwork` - 价值网络
- `PPOAgent` - PPO智能体

**主要特性**:
- **Actor-Critic架构** - 分离的策略和价值网络
- **动作掩码支持** - 约束满足的有效动作生成
- **经验回放** - 高效的批量更新机制
- **自适应学习** - 动态学习率调整

**使用示例**:
```python
from src.rl.agent import PPOAgent

# 创建智能体
agent = PPOAgent(
    node_embedding_dim=128,
    region_embedding_dim=256,
    num_partitions=3,
    lr_actor=3e-4,
    lr_critic=1e-3
)

# 选择动作
action, log_prob, value = agent.select_action(state, training=True)

# 存储经验
agent.store_experience(state, action, reward, log_prob, value, done)

# 更新网络
losses = agent.update()
```

### 5. 状态管理 (`src/rl/state.py`)

**功能**: 管理MDP状态的表示和更新

**核心类**: `StateManager`

**主要特性**:
- **异构图状态** - 支持多种节点和边类型
- **增量更新** - 高效的状态转移
- **边界节点跟踪** - 动态计算可移动节点
- **区域嵌入** - 分区级别的特征聚合

**技术细节**:
- 维护全局到局部索引映射
- 使用邻接列表加速边界节点计算
- 支持mean和max pooling的区域聚合
- 提供完整的观测空间接口

### 6. 动作空间管理 (`src/rl/action_space.py`)

**功能**: 管理两阶段动作空间和约束

**核心类**: 
- `ActionSpace` - 动作空间管理
- `ActionMask` - 约束掩码处理

**主要特性**:
- **两阶段动作** - 节点选择 + 分区选择
- **约束处理** - 拓扑、连通性、大小平衡
- **掩码生成** - 高效的有效动作枚举
- **约束验证** - 动作有效性检查

**约束类型**:
- 边界节点约束: 只有边界节点可移动
- 邻接约束: 只能移动到相邻分区
- 连通性约束: 保持分区内部连通
- 大小平衡约束: 避免分区过大或过小

### 7. 奖励函数 (`src/rl/reward.py`)

**功能**: 多目标复合奖励函数

**核心类**: `RewardFunction`

**奖励组件**:
1. **负载平衡** (R_balance): -Var(L₁, ..., Lₖ)
2. **电气解耦** (R_decoupling): -Σ|Y_uv| (跨分区边)
3. **功率平衡** (R_internal): -Σ(P_gen - P_load)²

**使用示例**:
```python
from src.rl.reward import RewardFunction

# 创建奖励函数
reward_fn = RewardFunction(
    hetero_data=hetero_data,
    reward_weights={
        'load_balance': 0.4,
        'electrical_decoupling': 0.4,
        'power_balance': 0.2
    }
)

# 计算奖励
reward = reward_fn.compute_reward(current_partition, boundary_nodes, action)
```

### 8. 场景生成器 (`src/rl/scenario_generator.py`)

**功能**: 生成多样化的训练场景

**核心类**: `ScenarioGenerator`

**场景类型**:
- **N-1故障**: 随机支路故障
- **负荷波动**: 负荷和发电变化
- **组合场景**: 多种扰动同时应用

**使用示例**:
```python
from src.rl.scenario_generator import ScenarioGenerator

# 创建生成器
generator = ScenarioGenerator(base_case_data)

# 生成随机场景
perturbed_case = generator.generate_random_scene()

# 批量生成
scenarios = generator.generate_batch_scenarios(num_scenarios=100)

# 特定故障
case_n1 = generator.apply_specific_contingency(base_case_data, branch_idx=10)
```

### 9. Gym包装器 (`src/rl/gym_wrapper.py`)

**功能**: OpenAI Gym兼容的环境包装器

**核心类**: `PowerGridPartitionGymEnv`

**主要特性**:
- **Gym接口兼容** - 标准的reset/step接口
- **并行训练支持** - 多进程环境
- **Stable-Baselines3集成** - 支持SB3算法
- **场景生成集成** - 自动场景多样化

**使用示例**:
```python
from src.rl.gym_wrapper import make_parallel_env

# 创建并行环境
parallel_env = make_parallel_env(
    base_case_data=case_data,
    config=config,
    num_envs=12,
    use_scenario_generator=True
)

# 使用Stable-Baselines3训练
from stable_baselines3 import PPO
model = PPO("MlpPolicy", parallel_env)
model.learn(total_timesteps=1000000)
```

### 10. 工具函数 (`src/rl/utils.py`)

**功能**: 分区初始化和评估工具

**核心类**:
- `MetisInitializer` - METIS分区初始化
- `PartitionEvaluator` - 分区质量评估

**主要功能**:
- METIS图分区（带谱聚类备选）
- 分区连通性检查
- 负载平衡评估
- 电气解耦计算
- 模块度计算

### 11. 可视化模块 (`src/visualization.py`)

**功能**: 训练过程和结果可视化

**核心类**: `VisualizationManager`

**可视化类型**:
- **分区可视化** - 网络拓扑和分区边界
- **训练曲线** - 奖励、损失、指标趋势
- **交互式仪表板** - Plotly动态可视化
- **对比分析** - 多方法性能对比

**使用示例**:
```python
from src.visualization import VisualizationManager

# 创建可视化管理器
viz = VisualizationManager(config)

# 可视化分区结果
viz.visualize_partition(env, title="Final Partition", save_path="partition.png")

# 绘制训练曲线
viz.plot_training_curves(history, save_path="training_curves.png")

# 创建交互式可视化
fig = viz.create_interactive_visualization(env, comparison_df)
```

## 🔄 模块间交互

```
数据处理 → GAT编码器 → RL环境 → PPO智能体
    ↓           ↓          ↓         ↓
  异构图    → 节点嵌入  → MDP状态  → 策略网络
    ↓           ↓          ↓         ↓
  缓存数据   → 注意力权重 → 奖励信号 → 动作选择
```

## 🛠️ 扩展指南

### 添加新的奖励组件
```python
# 在 reward.py 中添加新方法
def _compute_custom_reward(self, current_partition):
    # 实现自定义奖励逻辑
    return custom_reward

# 在 compute_reward 中集成
total_reward += self.weights['custom'] * custom_reward
```

### 添加新的约束
```python
# 在 action_space.py 中添加约束方法
def apply_custom_constraint(self, action_mask, current_partition):
    # 实现约束逻辑
    return updated_mask
```

### 自定义GAT层
```python
# 继承 PhysicsGATv2Conv
class CustomGATConv(PhysicsGATv2Conv):
    def physics_enhanced_edge_attr(self, edge_attr):
        # 实现自定义物理增强
        return enhanced_edge_attr
```

## 📊 性能优化建议

1. **GPU加速**: 在配置中设置 `device: cuda`
2. **批量处理**: 增加 `batch_size` 和 `update_interval`
3. **内存优化**: 使用较小的嵌入维度
4. **并行训练**: 启用 `parallel_training.enabled: true`
5. **缓存优化**: 确保缓存目录可写且有足够空间

## 🔍 调试指南

### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 检查数据流
```python
# 检查异构图数据
print(f"节点类型: {hetero_data.node_types}")
print(f"边类型: {hetero_data.edge_types}")

# 检查嵌入维度
for node_type, embeddings in node_embeddings.items():
    print(f"{node_type}: {embeddings.shape}")
```

### 验证环境
```python
# 测试环境重置
state, info = env.reset()
print(f"初始状态: {state.keys()}")

# 测试动作空间
valid_actions = env.get_valid_actions()
print(f"有效动作数: {len(valid_actions)}")
```

## 📋 常见问题

**Q: 训练不收敛怎么办？**
A: 降低学习率，增加训练回合数，检查奖励函数权重

**Q: 内存不足怎么办？**
A: 减少批大小，降低嵌入维度，使用CPU训练

**Q: 如何添加新的电网案例？**
A: 准备MATPOWER格式文件，放入data目录，在配置中指定路径

**Q: 如何自定义训练模式？**
A: 在config_unified.yaml中添加新的预设配置，在train_unified.py中注册新模式 