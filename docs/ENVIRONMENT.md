# MDP建模与环境设计

## 🎯 核心问题

电力网络分区是一个复杂的组合优化问题，需要将其建模为马尔可夫决策过程(MDP)，同时处理连通性约束、负载平衡和电气解耦等多重目标。

## 🔬 MDP建模框架

### 状态空间设计

**完整状态表示:**
```
s_t = (H', z_t, Bdry_t, R_t)

其中:
H' ∈ R^{n×d}: 节点嵌入矩阵（GAT编码后）
z_t ∈ {0,1,2,...,K}^n: 当前分区分配向量
Bdry_t ⊆ {1,2,...,n}: 边界节点集合
R_t ∈ R^{K×2d}: 区域聚合嵌入矩阵
```

**边界节点定义:**
```
Bdry_t = {i | ∃j ∈ N(i), z_t[i] ≠ z_t[j] ∧ z_t[i] > 0 ∧ z_t[j] > 0}

即：与不同分区邻居相连的已分配节点
```

**状态特征构建:**
```python
def construct_state_features(self):
    """构建状态特征向量"""
    features = torch.cat([
        self.node_embeddings.flatten(),      # 节点嵌入
        self.partition_one_hot.flatten(),    # 分区独热编码
        self.boundary_mask.float(),          # 边界节点掩码
        self.region_embeddings.flatten()     # 区域聚合特征
    ], dim=0)
    return features
```

### 动作空间设计

**两阶段动作选择:**
```
阶段1: 选择边界节点 i_t ∈ Bdry_t
阶段2: 选择目标分区 k_new ≠ z_t[i_t]

动作表示: a_t = (i_t, k_new)
动作空间大小: |A_t| = |Bdry_t| × (K-1)
```

**动作有效性约束:**
```python
def is_valid_action(self, node_idx, target_partition, current_state):
    """检查动作有效性"""
    # 1. 基础有效性检查
    if node_idx not in current_state.boundary_nodes:
        return False, "节点不是边界节点"
    
    if current_state.partition[node_idx] == target_partition:
        return False, "目标分区与当前分区相同"
    
    # 2. 邻接性检查
    if not self._has_neighbor_in_partition(node_idx, target_partition):
        return False, "目标分区中无邻接节点"
    
    return True, "动作有效"
```

### 奖励函数设计

**分解结构:**
```
R(s_t, a_t, s_{t+1}) = R_potential + R_constraint + R_efficiency

其中:
R_potential = γΦ(s_{t+1}) - Φ(s_t)  # 势函数主奖励
R_constraint = -λ_penalty · P(a_t)   # 约束违规惩罚
R_efficiency = η(t) · I_plateau      # 效率奖励（平台期激活）
```

## 🔧 环境实现

### PowerGridPartitioningEnv核心架构

<augment_code_snippet path="code/src/rl/environment.py" mode="EXCERPT">
````python
class PowerGridPartitioningEnv:
    def __init__(self, hetero_data, node_embeddings, num_partitions, config):
        # 核心组件初始化
        self.state_manager = StateManager(hetero_data, node_embeddings, device, config)
        self.action_space = ActionSpace(hetero_data, num_partitions, device)
        self.reward_function = RewardFunction(hetero_data, config, device)
        
        # 软约束系统
        self.dynamic_constraint_params = config.get('dynamic_constraints', {})
        
    def step(self, action):
        """执行一步环境交互"""
        # 1. 动作有效性检查
        if not self._validate_action(action):
            return self._invalid_action_response(action)
        
        # 2. 软约束检查
        constraint_penalty = self._check_soft_constraints(action)
        
        # 3. 执行状态转换
        next_state = self.state_manager.apply_action(action)
        
        # 4. 计算奖励
        reward = self.reward_function.compute_incremental_reward(
            next_state.partition, action
        )
        
        # 5. 应用约束惩罚
        total_reward = reward + constraint_penalty
        
        return next_state.observation, total_reward, self._check_termination(), False, {}
````
</augment_code_snippet>

### 智能软约束系统

**约束模式切换:**
```python
def check_connectivity_violation(self, current_partition, action):
    """检查连通性违规"""
    node_idx, target_partition = action
    
    # 模拟执行动作
    temp_partition = current_partition.clone()
    temp_partition[node_idx] = target_partition
    
    # 检查连通性
    connectivity_info = self._analyze_connectivity(temp_partition)
    
    if not connectivity_info['is_connected']:
        return {
            'violates_connectivity': True,
            'violation_severity': connectivity_info['disconnected_components'],
            'suggested_penalty': self._calculate_penalty(connectivity_info)
        }
    
    return {'violates_connectivity': False}
```

**动态惩罚计算:**
```python
def calculate_dynamic_penalty(self, violation_info, training_stage):
    """根据训练阶段动态计算惩罚"""
    base_penalty = violation_info['violation_severity'] * 0.5
    
    # 阶段调整因子
    stage_factors = {
        'exploration': 0.1,    # 探索期：轻微惩罚
        'transition': 0.5,     # 过渡期：中等惩罚
        'refinement': 1.0,     # 精炼期：标准惩罚
        'fine_tuning': 1.5     # 微调期：严格惩罚
    }
    
    stage_factor = stage_factors.get(training_stage, 1.0)
    return -base_penalty * stage_factor
```

## 📊 状态管理

### StateManager核心功能

<augment_code_snippet path="code/src/rl/state.py" mode="EXCERPT">
````python
class StateManager:
    def apply_action(self, action):
        """应用动作并更新状态"""
        node_idx, target_partition = action
        
        # 1. 更新分区分配
        old_partition = self.current_partition[node_idx].item()
        self.current_partition[node_idx] = target_partition
        
        # 2. 更新边界节点集合
        self._update_boundary_nodes(node_idx, old_partition, target_partition)
        
        # 3. 更新区域嵌入
        self._update_region_embeddings(old_partition, target_partition)
        
        # 4. 记录历史
        self.action_history.append(action)
        
        return self.get_current_state()
    
    def _update_boundary_nodes(self, changed_node, old_partition, new_partition):
        """高效更新边界节点集合"""
        # 只检查受影响的节点及其邻居
        affected_nodes = [changed_node] + self.get_neighbors(changed_node)
        
        for node in affected_nodes:
            if self._is_boundary_node(node):
                self.boundary_nodes.add(node)
            else:
                self.boundary_nodes.discard(node)
````
</augment_code_snippet>

### 高效状态表示

**稀疏表示优化:**
```python
def get_compact_observation(self):
    """获取紧凑的状态观察"""
    # 只包含边界节点的详细信息
    boundary_features = self.node_embeddings[list(self.boundary_nodes)]
    
    # 分区统计信息
    partition_stats = self._compute_partition_statistics()
    
    # 全局上下文
    global_context = self._compute_global_context()
    
    return {
        'boundary_features': boundary_features,
        'partition_stats': partition_stats,
        'global_context': global_context,
        'action_mask': self._generate_action_mask()
    }
```

## 🎯 终止条件

### 自然终止

**完成条件:**
```python
def check_natural_termination(self):
    """检查自然终止条件"""
    # 1. 所有节点已分配
    if (self.current_partition > 0).all():
        # 2. 连通性满足
        if self._check_all_partitions_connected():
            # 3. 质量可接受
            quality_score = self.reward_function.compute_quality_score(
                self.current_partition
            )
            if quality_score > self.min_acceptable_quality:
                return True, "自然完成"
    
    return False, "未完成"
```

### 强制终止

**异常终止条件:**
```python
def check_forced_termination(self):
    """检查强制终止条件"""
    # 1. 步数超限
    if self.current_step >= self.max_steps:
        return True, "步数超限"
    
    # 2. 无可行动作
    if len(self.get_valid_actions()) == 0:
        return True, "无可行动作"
    
    # 3. 质量严重恶化
    if self._detect_quality_collapse():
        return True, "质量崩溃"
    
    return False, "继续"
```

## 🔍 调试与监控

### 环境状态可视化

```python
def visualize_environment_state(self):
    """可视化当前环境状态"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 分区分配图
    self._plot_partition_assignment(axes[0, 0])
    
    # 边界节点分布
    self._plot_boundary_nodes(axes[0, 1])
    
    # 奖励历史
    self._plot_reward_history(axes[1, 0])
    
    # 动作空间大小变化
    self._plot_action_space_size(axes[1, 1])
    
    plt.tight_layout()
    return fig
```

### 性能监控

```python
def collect_environment_metrics(self):
    """收集环境性能指标"""
    return {
        'episode_length': self.current_step,
        'boundary_nodes_count': len(self.boundary_nodes),
        'action_space_size': len(self.get_valid_actions()),
        'partition_balance': self._compute_partition_balance(),
        'connectivity_violations': self.connectivity_violation_count,
        'average_reward': np.mean(self.reward_history[-50:])
    }
```

## 🚀 扩展性设计

### 多网络支持

```python
def adapt_to_network(self, new_hetero_data):
    """适应新的电力网络"""
    # 自动调整状态空间维度
    self._resize_state_space(new_hetero_data.num_nodes)
    
    # 重新初始化组件
    self.state_manager.reset_for_new_network(new_hetero_data)
    self.action_space.update_network_topology(new_hetero_data)
```

### 约束系统扩展

```python
def add_custom_constraint(self, constraint_func, penalty_func):
    """添加自定义约束"""
    self.custom_constraints.append({
        'check': constraint_func,
        'penalty': penalty_func,
        'weight': 1.0
    })
```

这种MDP环境设计提供了灵活、高效且可扩展的强化学习框架，为电力网络分区问题提供了强大的建模能力。
