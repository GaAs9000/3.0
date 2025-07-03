# API参考文档

## 🎯 核心接口概览

本文档提供系统各模块的关键API接口说明，包含使用示例和参数详解。

## 🔧 数据特征提取API

### HeteroGATEncoder

**主要接口:**

```python
# 初始化编码器
encoder = HeteroGATEncoder(
    node_types=['bus_pv', 'bus_slack', 'gen'],
    edge_types=[('bus_pv', 'line', 'bus_slack'), ('gen', 'connects', 'bus_pv')],
    hidden_channels=128,
    output_dim=64,
    num_layers=3,
    heads=8
)

# 编码节点特征
node_embeddings = encoder.encode_nodes(hetero_data)
# 返回: {'bus_pv': tensor, 'bus_slack': tensor, 'gen': tensor}

# 获取注意力权重
attention_weights = encoder.get_attention_weights()
# 返回: {'edge_type_1': tensor, 'edge_type_2': tensor, ...}

# 完整前向传播
node_emb, attn_weights, graph_emb = encoder.forward(
    hetero_data, 
    return_attention_weights=True,
    return_graph_embedding=True
)
```

**参数说明:**
- `node_types`: 节点类型列表
- `edge_types`: 边类型三元组列表 (源节点, 关系, 目标节点)
- `hidden_channels`: 隐藏层维度
- `output_dim`: 输出嵌入维度
- `heads`: 注意力头数

## 🏗️ 强化学习环境API

### PowerGridPartitioningEnv

**标准RL接口:**

```python
# 环境初始化
env = PowerGridPartitioningEnv(
    hetero_data=hetero_data,
    node_embeddings=node_embeddings,
    num_partitions=4,
    max_steps=50,
    reward_weights={'balance': 0.4, 'decoupling': 0.4, 'power': 0.2}
)

# 重置环境
observation = env.reset()
# 返回: Dict[str, torch.Tensor] - 状态观察

# 执行动作
obs, reward, terminated, truncated, info = env.step(action)
# action: Tuple[int, int] - (节点索引, 目标分区)
# 返回: (观察, 奖励, 是否终止, 是否截断, 附加信息)

# 获取有效动作
valid_actions = env.get_valid_actions()
# 返回: List[Tuple[int, int]] - 有效动作列表

# 渲染当前状态
env.render(mode='human')  # 可视化显示
env.render(mode='rgb_array')  # 返回图像数组
```

**配置参数:**
```python
config = {
    'max_steps': 50,
    'reward_weights': {
        'balance': 0.4,
        'decoupling': 0.4, 
        'power': 0.2
    },
    'dynamic_constraints': {
        'constraint_mode': 'soft',  # 'hard' or 'soft'
        'connectivity_penalty': 1.0,
        'enable_adaptive': True
    }
}
```

## 🎯 奖励系统API

### RewardFunction

**核心方法:**

```python
# 初始化奖励函数
reward_func = RewardFunction(
    hetero_data=hetero_data,
    config=reward_config,
    device=device
)

# 计算即时奖励
reward, plateau_result = reward_func.compute_incremental_reward(
    current_partition=partition_tensor,
    action=(node_idx, target_partition),
    scenario_context=scenario_context  # 可选
)

# 计算终局奖励
final_reward, components = reward_func.compute_final_reward(
    final_partition=final_partition,
    termination_type='natural'  # 'natural', 'timeout', 'stuck'
)

# 质量分数计算
quality_score = reward_func.compute_quality_score(partition_tensor)

# 平台期检测
is_plateau = reward_func.should_early_stop(partition_tensor)
```

**奖励组件详情:**
```python
# components字典包含:
{
    'balance_reward': float,      # 平衡性奖励
    'decoupling_reward': float,   # 解耦奖励
    'power_reward': float,        # 功率平衡奖励
    'quality_reward': float,      # 综合质量奖励
    'threshold_bonus': float,     # 阈值奖励
    'termination_discount': float, # 终止折扣
    'final_reward': float,        # 最终奖励
    'metrics': dict              # 详细指标
}
```

## 🧠 智能体API

### PPOAgent

**训练接口:**

```python
# 初始化智能体
agent = PPOAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    config=ppo_config
)

# 选择动作
action, log_prob, value = agent.select_action(observation)

# 批量训练
train_stats = agent.train_batch(
    observations=obs_batch,
    actions=action_batch,
    rewards=reward_batch,
    dones=done_batch,
    log_probs=log_prob_batch,
    values=value_batch
)

# 保存/加载模型
agent.save_model('path/to/model.pth')
agent.load_model('path/to/model.pth')
```

**配置参数:**
```python
ppo_config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5,
    'update_epochs': 4,
    'batch_size': 64
}
```

## 📚 自适应课程学习API

### AdaptiveCurriculumDirector

**核心接口:**

```python
# 初始化导演系统
director = AdaptiveCurriculumDirector(
    total_episodes=5000,
    config=curriculum_config
)

# 更新训练参数
updated_params = director.update_training_parameters(
    episode_num=current_episode,
    performance_metrics={
        'average_reward': avg_reward,
        'episode_length': avg_length,
        'quality_score': avg_quality,
        'connectivity_rate': conn_rate
    }
)

# 获取当前阶段信息
stage_info = director.get_current_stage_info()
# 返回: {'stage': str, 'progress': float, 'parameters': dict}

# 检查是否需要阶段转换
should_advance = director.should_advance_stage(recent_performance)
```

**课程配置:**
```python
curriculum_config = {
    'stages': {
        'exploration': {'episodes_ratio': 0.25, 'connectivity_penalty': 0.05},
        'transition': {'episodes_ratio': 0.25, 'connectivity_penalty': 0.5},
        'refinement': {'episodes_ratio': 0.3, 'connectivity_penalty': 1.0},
        'fine_tuning': {'episodes_ratio': 0.2, 'connectivity_penalty': 1.5}
    },
    'transition_criteria': {
        'min_stability': 0.8,
        'min_improvement': 0.01,
        'confidence_threshold': 0.8
    }
}
```

## 🔍 数据处理API

### DataProcessor

**数据加载与预处理:**

```python
# 初始化数据处理器
processor = DataProcessor(config=data_config)

# 加载MATPOWER数据
hetero_data = processor.load_matpower_case('ieee30.mat')

# 数据预处理
processed_data = processor.preprocess_network_data(
    hetero_data,
    normalize_features=True,
    add_virtual_nodes=False
)

# 场景生成
scenario_data = processor.generate_scenario(
    base_data=hetero_data,
    scenario_type='n_minus_1',  # 'normal', 'n_minus_1', 'load_variation'
    severity=0.5
)
```

## 🎨 可视化API

### 训练监控

```python
# TensorBoard日志记录
from code.src.utils.logger import TensorBoardLogger

logger = TensorBoardLogger(log_dir='data/logs')

# 记录标量指标
logger.log_scalar('train/reward', reward, step)
logger.log_scalar('train/episode_length', length, step)

# 记录分区可视化
logger.log_partition_visualization(
    partition=final_partition,
    hetero_data=hetero_data,
    step=episode
)

# 记录注意力权重
logger.log_attention_weights(
    attention_weights=attn_weights,
    step=episode
)
```

### 结果分析

```python
# 性能分析
from code.src.utils.analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# 分析训练结果
analysis_report = analyzer.analyze_training_results(
    log_dir='data/logs',
    metrics=['reward', 'quality', 'episode_length']
)

# 生成对比报告
comparison = analyzer.compare_experiments(
    experiment_dirs=['exp1', 'exp2', 'exp3'],
    metrics=['final_quality', 'convergence_speed']
)
```

## 🛠️ 实用工具API

### 配置管理

```python
# 配置加载
from code.src.utils.config import load_config, save_config

config = load_config('config.yaml')
save_config(config, 'modified_config.yaml')

# 配置验证
from code.src.utils.config import validate_config

is_valid, errors = validate_config(config)
```

### 模型评估

```python
# 模型评估器
from code.src.utils.evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# 跨网络评估
results = evaluator.evaluate_cross_network(
    model_path='models/best_model.pth',
    test_cases=['ieee14', 'ieee30', 'ieee57'],
    num_episodes=10
)

# 生成评估报告
report = evaluator.generate_evaluation_report(results)
```

## 🚀 快速开始示例

### 完整训练流程

```python
# 1. 数据准备
processor = DataProcessor()
hetero_data = processor.load_matpower_case('ieee30.mat')

# 2. 特征提取
encoder = HeteroGATEncoder(...)
node_embeddings = encoder.encode_nodes(hetero_data)

# 3. 环境初始化
env = PowerGridPartitioningEnv(hetero_data, node_embeddings, num_partitions=4)

# 4. 智能体初始化
agent = PPOAgent(env.observation_space, env.action_space)

# 5. 训练循环
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 存储经验并训练
        agent.store_transition(obs, action, reward, next_obs, done, log_prob, value)
        obs = next_obs
        done = terminated or truncated
    
    # 批量更新
    if episode % update_interval == 0:
        agent.train_batch()
```

这些API接口提供了完整的系统功能访问，支持灵活的定制和扩展。
