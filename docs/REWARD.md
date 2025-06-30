# 奖励函数系统

## 概述

本系统实现了基于势函数理论的激励结构，用于电力网络分区的强化学习训练。该系统由**即时奖励（reward）**和**终局奖励（Reward）**组成，旨在提供密集反馈的同时确保智能体追求高质量的最终分区方案。

## 核心设计原则

### 1. 理论基础
- **势函数奖励塑造**：基于Ng等人的势函数理论，确保奖励塑造不改变最优策略
- **增量奖励机制**：即时奖励等价于势函数差值，提供密集的学习信号
- **质量导向终局**：终局奖励基于最终分区质量，定义成功标准

### 2. 数值稳定性
- **NaN/inf保护**：所有数学计算都有epsilon保护和异常处理
- **值域限制**：奖励值被限制在合理范围内，防止梯度爆炸
- **鲁棒性设计**：能够处理极端输入和边界情况

### 3. 向后兼容
- **多模式支持**：支持legacy、enhanced、dual_layer三种奖励模式
- **配置驱动**：所有参数通过配置文件控制，便于调优
- **接口一致**：保持与原有系统相同的调用接口

## 系统架构

### DualLayerRewardFunction类

```python
class DualLayerRewardFunction:
    """奖励函数系统核心类"""
    
    def __init__(self, hetero_data, config, device):
        """初始化奖励函数"""
        
    def compute_incremental_reward(self, partition, action):
        """计算即时奖励"""
        
    def compute_final_reward(self, partition, termination_type):
        """计算终局奖励"""
```

### 核心组件

1. **_compute_core_metrics()**: 中心化指标计算，避免重复计算
2. **previous_metrics**: 前一步指标缓存，支持增量计算
3. **数值稳定性保护**: 全面的NaN/inf检查和处理

## 即时奖励系统

### 计算公式
```
reward_t = tanh(5.0 * (w1*Δbalance + w2*Δdecoupling + w3*Δpower))
```

其中：
- **Δbalance** = CV(t-1) - CV(t)：负载平衡改善量
- **Δdecoupling** = coupling_ratio(t-1) - coupling_ratio(t)：电气解耦改善量  
- **Δpower** = power_imbalance(t-1) - power_imbalance(t)：功率平衡改善量

### 核心指标

#### 1. 负载平衡指标 (CV)
```
CV = σ(L1, L2, ..., LK) / μ(L1, L2, ..., LK)
```
- **Lk**: 第k个分区的总有功功率负载
- **σ**: 标准差函数
- **μ**: 平均值函数

#### 2. 电气解耦指标
```
coupling_ratio = Σ(跨区导纳) / Σ(总导纳)
edge_decoupling_ratio = 1 - (跨区边数 / 总边数)
```

#### 3. 功率平衡指标
```
power_imbalance_normalized = Σ|Pgen,k - Pload,k| / Σ|Pd,i|
```

### 特性
- **值域稳定**：使用tanh归一化到[-1, 1]范围
- **敏感度控制**：5倍放大因子增加对改善的敏感性
- **增量特性**：符合势函数奖励塑造理论

## 终局奖励系统

### 计算公式
```
Reward = (w1*R_balance + w2*R_decoupling + w3*R_power + threshold_bonus) * termination_discount
```

### 核心奖励组件

#### 1. 负载平衡奖励
```
R_balance = exp(-2.0 * CV)
```
- CV越小，奖励越接近1.0
- 指数函数提供非线性激励

#### 2. 电气解耦奖励
```
R_decoupling = 0.5 * σ(5*(r_edge - 0.5)) + 0.5 * σ(5*(r_admittance - 0.5))
```
- **r_edge**: 拓扑解耦率 = 1 - (跨区边数/总边数)
- **r_admittance**: 导纳解耦率 = 1 - (跨区导纳/总导纳)
- **σ**: sigmoid函数

#### 3. 功率平衡奖励
```
R_power = exp(-3.0 * I_normalized)
```
- **I_normalized**: 归一化功率不平衡度
- 指数函数强调功率平衡的重要性

### 非线性阈值奖励

根据质量阈值提供分段奖励：

| 指标 | 卓越阈值 | 卓越奖励 | 良好阈值 | 良好奖励 |
|------|----------|----------|----------|----------|
| CV | < 0.1 | +10.0 | < 0.2 | +5.0 |
| 耦合比率 | < 0.3 | +8.0 | < 0.5 | +4.0 |
| 功率不平衡 | < 10% | +6.0 | < 50% | +3.0 |

### 终止条件折扣

| 终止类型 | 折扣系数 | 说明 |
|----------|----------|------|
| natural | 1.0 | 自然完成，无折扣 |
| timeout | 0.7 | 超时完成，70%折扣 |
| stuck | 0.3 | 提前卡住，30%折扣 |

## 配置参数

### 基本配置
```yaml
environment:
  reward_weights:
    reward_mode: dual_layer  # 启用新奖励系统
    
    dual_layer_config:
      # 即时奖励权重
      balance_weight: 1.0
      decoupling_weight: 1.0
      power_weight: 1.0
      
      # 终局奖励权重
      final_balance_weight: 0.4
      final_decoupling_weight: 0.4
      final_power_weight: 0.2
```

### 质量阈值配置
```yaml
    thresholds:
      excellent_cv: 0.1
      good_cv: 0.2
      excellent_coupling: 0.3
      good_coupling: 0.5
      excellent_power: 10.0
      good_power: 50.0
```

### 预设配置
系统默认启用新的奖励系统，所有训练模式都使用该奖励函数。

## 使用方法

### 1. 配置启用
在`config.yaml`中设置：
```yaml
environment:
  reward_weights:
    reward_mode: dual_layer
```

### 2. 训练启动
```bash
python train.py --mode fast
```

### 3. 环境集成
系统自动集成到`PowerGridPartitioningEnv`中，无需修改训练代码。

## 技术特性

### 数值稳定性保护
- **输入验证**：检查NaN/inf输入
- **计算保护**：epsilon保护除零操作
- **结果限制**：将结果限制在合理范围
- **异常处理**：捕获并处理计算异常

### 性能优化
- **缓存机制**：前一步指标缓存避免重复计算
- **中心化计算**：_compute_core_metrics()统一计算所有指标
- **向量化操作**：使用PyTorch张量操作提高效率

### 调试支持
- **组件分解**：终局奖励返回详细的组件分解
- **指标暴露**：get_current_metrics()方法用于调试分析
- **状态管理**：reset_episode()方法重置内部状态

## 理论验证

### 势函数理论符合性
即时奖励严格遵循势函数奖励塑造理论：
```
F(s,a,s') = γ * Φ(s') - Φ(s)
```
其中势函数Φ(s) = -(w1*CV + w2*coupling + w3*power_imbalance)

### 公式正确性
所有奖励公式都经过数学验证，确保：
- 负载平衡奖励：R_balance = exp(-2.0 * CV)
- 功率平衡奖励：R_power = exp(-3.0 * I_normalized)
- 电气解耦奖励：基于sigmoid函数的组合

## 与原系统对比

| 特性 | 原系统 | 新奖励系统 |
|------|--------|-------------|
| 理论基础 | 启发式设计 | 势函数理论 |
| 奖励密度 | 稀疏 | 密集+终局 |
| 数值稳定性 | 基本保护 | 全面保护 |
| 配置灵活性 | 有限 | 高度可配置 |
| 调试支持 | 基本 | 详细分解 |

## 最佳实践

### 1. 参数调优
- 从默认配置开始
- 根据网络特性调整权重
- 监控奖励组件分解进行优化

### 2. 训练策略
- 使用fast模式进行日常训练
- 监控即时奖励和终局奖励的平衡
- 根据收敛情况调整阈值

### 3. 性能监控
- 关注奖励组件的分布
- 监控数值稳定性警告
- 分析终止类型分布

## 故障排除

### 常见问题
1. **奖励值异常**：检查输入数据的有效性
2. **收敛缓慢**：调整权重平衡或阈值设置
3. **数值不稳定**：启用调试模式查看详细信息

### 调试方法
1. 使用`get_current_metrics()`检查指标计算
2. 分析终局奖励的组件分解
3. 监控即时奖励的变化趋势
