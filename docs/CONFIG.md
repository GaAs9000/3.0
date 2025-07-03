# 配置参数指南

## 🎯 配置文件概览

系统使用YAML格式的配置文件，支持模块化配置和参数继承。主配置文件为`config.yaml`，包含所有核心参数设置。

## 📊 训练模式配置

### 基础训练模式

```yaml
# 快速训练模式（推荐用于开发和测试）
fast:
  episodes: 1500
  parallel_training: false
  scenario_generation: true
  save_interval: 100
  
# 完整训练模式（推荐用于正式训练）
full:
  episodes: 5000
  parallel_training: true
  scenario_generation: true
  save_interval: 200
  
# IEEE118大规模训练
ieee118:
  episodes: 3000
  parallel_training: true
  scenario_generation: true
  network_case: 'ieee118'
  save_interval: 150
```

### 自适应训练配置

```yaml
# 启用自适应课程学习
adaptive_curriculum:
  enabled: true
  
  # 4阶段课程设置
  stages:
    exploration:
      episodes_ratio: 0.25
      connectivity_penalty: 0.05
      action_mask_relaxation: 0.7
      reward_weights:
        balance: 0.8
        decoupling: 0.2
        power: 0.0
        
    transition:
      episodes_ratio: 0.25
      connectivity_penalty: 0.5
      action_mask_relaxation: 0.5
      reward_weights:
        balance: 0.6
        decoupling: 0.3
        power: 0.1
        
    refinement:
      episodes_ratio: 0.3
      connectivity_penalty: 1.0
      action_mask_relaxation: 0.3
      reward_weights:
        balance: 0.4
        decoupling: 0.4
        power: 0.2
        
    fine_tuning:
      episodes_ratio: 0.2
      connectivity_penalty: 1.5
      action_mask_relaxation: 0.2
      reward_weights:
        balance: 0.4
        decoupling: 0.4
        power: 0.2
  
  # 阶段转换条件
  transition_criteria:
    min_stability: 0.8
    min_improvement_rate: 0.01
    confidence_threshold: 0.8
    window_size: 50
```

## 🔧 GAT编码器配置

### 网络架构参数

```yaml
gat_encoder:
  # 基础架构
  hidden_channels: 128
  output_dim: 64
  num_layers: 3
  heads: 8
  dropout: 0.6
  
  # 物理增强参数
  physics_enhanced: true
  temperature: 1.0
  physics_weight: 1.0
  z_index: 3  # 阻抗模长在边特征中的索引
  
  # 特征预处理
  feature_normalization: true
  feature_projection: true
  
  # 异构图设置
  node_types: ['bus_pv', 'bus_slack', 'gen']
  edge_types:
    - ['bus_pv', 'line', 'bus_slack']
    - ['bus_pv', 'line', 'bus_pv']
    - ['bus_slack', 'line', 'bus_pv']
    - ['gen', 'connects', 'bus_pv']
```

### 注意力机制调优

```yaml
attention_config:
  # 注意力头配置
  heads: 8
  concat: true
  
  # 激活函数
  activation: 'leaky_relu'
  negative_slope: 0.2
  
  # 边特征处理
  edge_dim: 4
  edge_processor: 'linear'
  
  # 数值稳定性
  attention_dropout: 0.1
  temperature_clipping: [0.1, 10.0]
  gradient_clipping: 1.0
```

## 🎯 奖励系统配置

### 奖励权重设置

```yaml
reward_config:
  # 主要奖励组件权重
  weights:
    balance: 0.4      # 负载平衡权重
    decoupling: 0.4   # 电气解耦权重
    power: 0.2        # 功率平衡权重
  
  # 势函数参数
  potential_function:
    balance_steepness: 2.0    # exp(-2.0 * CV)
    decoupling_steepness: 5.0 # sigmoid(5 * (x - 0.5))
    power_steepness: 3.0      # exp(-3.0 * normalized_imbalance)
  
  # 相对奖励设置
  relative_reward:
    enabled: true
    min_denominator: 0.01  # 防除零阈值
    clip_range: [-2.0, 2.0]  # 奖励裁剪范围
  
  # 平台期检测
  plateau_detection:
    enabled: true
    window_size: 50
    min_improvement_threshold: 0.01
    stability_threshold: 0.8
    confidence_threshold: 0.8
  
  # 效率奖励
  efficiency_reward:
    enabled: true
    weight: 0.1
    activation_condition: 'plateau_detected'
```

### 数值稳定性配置

```yaml
numerical_stability:
  # NaN/Inf检测
  nan_detection: true
  inf_detection: true
  
  # 数值裁剪
  reward_clipping: [-10.0, 10.0]
  quality_clipping: [0.0, 1.0]
  
  # 安全除法
  safe_division_epsilon: 1e-8
  safe_exp_clipping: [-500, 500]
```

## 🏗️ 环境配置

### MDP环境设置

```yaml
environment:
  # 基础设置
  max_steps: 50
  num_partitions: 4
  
  # 约束系统
  constraint_mode: 'soft'  # 'hard' or 'soft'
  connectivity_penalty: 1.0
  
  # 动态约束参数
  dynamic_constraints:
    enabled: true
    connectivity_penalty_range: [0.05, 1.5]
    action_mask_relaxation_range: [0.2, 0.7]
  
  # 终止条件
  termination:
    natural_completion: true
    max_steps_termination: true
    quality_collapse_detection: true
    min_acceptable_quality: 0.3
  
  # 状态表示
  state_representation:
    include_node_embeddings: true
    include_boundary_mask: true
    include_region_embeddings: true
    compact_representation: false
```

### 动作空间配置

```yaml
action_space:
  # 动作选择策略
  selection_strategy: 'two_stage'  # 'two_stage' or 'direct'
  
  # 有效性检查
  validity_checks:
    boundary_node_check: true
    adjacency_check: true
    connectivity_check: true
  
  # 动作掩码
  action_masking:
    enabled: true
    mask_invalid_actions: true
    mask_connectivity_violations: false  # 软约束模式下设为false
```

## 🧠 PPO智能体配置

### 算法参数

```yaml
ppo_agent:
  # 学习参数
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  
  # PPO特定参数
  clip_epsilon: 0.2
  entropy_coef: 0.01
  value_loss_coef: 0.5
  max_grad_norm: 0.5
  
  # 训练设置
  update_epochs: 4
  batch_size: 64
  mini_batch_size: 16
  
  # 网络架构
  actor_network:
    hidden_sizes: [256, 256]
    activation: 'tanh'
    output_activation: 'softmax'
  
  critic_network:
    hidden_sizes: [256, 256]
    activation: 'tanh'
    output_activation: 'linear'
```

### 学习率调度

```yaml
learning_rate_schedule:
  type: 'cosine_annealing'  # 'constant', 'linear_decay', 'cosine_annealing'
  
  # 余弦退火参数
  cosine_annealing:
    T_max: 5000  # 最大episode数
    eta_min: 1e-6  # 最小学习率
  
  # 线性衰减参数
  linear_decay:
    decay_rate: 0.99
    min_lr: 1e-6
```

## 📊 数据处理配置

### 网络数据设置

```yaml
data_config:
  # 数据路径
  data_dir: 'data/matpower'
  case_files:
    ieee14: 'case14.mat'
    ieee30: 'case30.mat'
    ieee57: 'case57.mat'
    ieee118: 'case118.mat'
  
  # 特征预处理
  feature_preprocessing:
    normalization: 'z_score'  # 'z_score', 'min_max', 'none'
    missing_value_handling: 'interpolation'
    outlier_detection: true
    outlier_threshold: 3.0
  
  # 异构图构建
  hetero_graph:
    add_self_loops: true
    add_virtual_nodes: false
    edge_feature_dim: 4
    node_feature_dim: 8
```

### 场景生成配置

```yaml
scenario_generation:
  enabled: true
  
  # 场景类型权重
  scenario_weights:
    normal: 0.4
    n_minus_1: 0.3
    load_variation: 0.2
    generation_variation: 0.1
  
  # N-1故障场景
  n_minus_1:
    failure_probability: 0.1
    critical_lines_only: false
    
  # 负荷变化场景
  load_variation:
    variation_range: [0.8, 1.2]
    correlation_factor: 0.7
    
  # 发电变化场景
  generation_variation:
    variation_range: [0.9, 1.1]
    renewable_volatility: 0.15
```

## 🔍 日志与监控配置

### TensorBoard设置

```yaml
logging:
  # 基础设置
  log_dir: 'data/logs'
  log_level: 'INFO'
  
  # TensorBoard
  tensorboard:
    enabled: true
    log_interval: 10
    
    # 记录内容
    log_scalars: true
    log_histograms: true
    log_images: true
    log_attention_weights: true
  
  # 文件日志
  file_logging:
    enabled: true
    log_file: 'training.log'
    rotation: 'daily'
    max_size: '100MB'
```

### 性能监控

```yaml
monitoring:
  # 性能指标
  metrics:
    - 'episode_reward'
    - 'episode_length'
    - 'quality_score'
    - 'connectivity_rate'
    - 'convergence_speed'
  
  # 监控频率
  monitoring_interval: 50
  
  # 异常检测
  anomaly_detection:
    enabled: true
    reward_collapse_threshold: -5.0
    length_drop_threshold: 0.5
```

## 🚀 快速配置模板

### 开发环境配置

```yaml
# 快速开发配置
development:
  mode: 'fast'
  episodes: 500
  parallel_training: false
  tensorboard: true
  debug_mode: true
  save_interval: 50
```

### 生产环境配置

```yaml
# 生产训练配置
production:
  mode: 'full'
  episodes: 5000
  parallel_training: true
  adaptive_curriculum: true
  scenario_generation: true
  save_interval: 200
  checkpoint_interval: 1000
```

这些配置参数提供了系统的完整控制能力，支持从快速原型到生产部署的各种需求。
