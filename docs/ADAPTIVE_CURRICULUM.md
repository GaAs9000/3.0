# 智能自适应课程学习系统

## 🎯 概述

智能自适应课程学习系统将传统的"固定剧本"课程学习升级为"智能导演"模式，实现了基于性能驱动的自适应训练参数调整。

**🚀 重大更新 (2025-06-30)**: 完成与智能软约束系统的深度集成，实现真正的端到端自适应训练！

### 核心特性

- 🧠 **智能软约束集成**：动态约束参数，支持软惩罚机制
- 🎭 **智能导演模式**：4阶段自适应课程学习
- 📈 **平滑参数演化**：连续参数调整，避免训练震荡
- 🔍 **平台化检测**：智能阶段转换和自动回退机制
- 🛡️ **安全保障机制**：训练崩溃检测与自动恢复
- 🔄 **向后兼容**：保持现有配置系统，支持传统模式

## 🏗️ 系统架构

### 核心组件

```
AdaptiveCurriculumDirector (智能导演)
├── PerformanceAnalyzer (性能分析器)
├── ParameterScheduler (参数调度器)
└── SafetyMonitor (安全监控器)
```

### 4阶段智能课程

1. **🔍 探索期 (Exploration)**：高度放松约束，鼓励探索
   - 软约束惩罚: 0.05
   - 动作掩码放松: 0.7
   - 重点: 大胆探索，建立基础策略

2. **🔄 过渡期 (Transition)**：逐渐收紧约束，平滑参数演化
   - 软约束惩罚: 0.1 → 1.5 (平滑过渡)
   - 动作掩码放松: 0.7 → 0.0
   - 重点: 平滑收紧，稳定提升

3. **⚡ 精炼期 (Refinement)**：严格约束，追求高质量解
   - 软约束惩罚: 1.5
   - 动作掩码放松: 0.0
   - 重点: 严格约束，追求卓越

4. **🎯 微调期 (Fine-tuning)**：学习率退火，精细优化
   - 软约束惩罚: 1.5
   - 学习率衰减: 动态调整
   - 重点: 精细优化，完美收敛

## 🚀 快速开始

### 1. 启用智能自适应课程学习

```bash
# 方法1：使用预设的adaptive模式
python train.py --mode adaptive

# 方法2：在配置文件中启用
# config.yaml
adaptive_curriculum:
  enabled: true
```

### 2. 配置参数

```yaml
adaptive_curriculum:
  enabled: true
  
  # 阶段转换触发条件
  stage_transition:
    episode_length_target: 15              # 目标Episode长度
    episode_length_window: 50              # 评估窗口大小
    episode_length_stability: 0.8          # 80%的episodes达到目标
    composite_score_target: 0.7            # 复合评分目标
    composite_score_window: 100            # 复合评分评估窗口
    connectivity_rate_threshold: 0.9       # 连通性达标率
  
  # 参数演化策略
  parameter_evolution:
    connectivity_penalty_range: [0.1, 1.5]
    action_mask_relaxation_range: [0.0, 0.7]
    balance_weight_range: [0.6, 0.8]
    decoupling_weight_range: [0.2, 0.6]
    power_weight_range: [0.0, 0.3]
    learning_rate_decay_factor: 0.1
  
  # 安全监控配置
  safety_monitoring:
    min_episode_length: 3
    max_reward_threshold: -100
    performance_deterioration_patience: 50
    performance_deterioration_threshold: 0.8
```

## 📊 智能决策机制

### 阶段转换条件

#### 阶段1 → 阶段2
- **触发条件**：Episode长度稳定性达标
- **判定标准**：80%的episodes达到目标长度（默认15步）
- **评估窗口**：最近50个episodes

#### 阶段2 → 阶段3
- **触发条件**：复合性能评分达标
- **判定标准**：平均复合评分 ≥ 0.7
- **评估窗口**：最近100个episodes

#### 阶段3 → 阶段4
- **触发条件**：性能稳定且停留时间充足
- **判定标准**：在阶段3停留≥300个episodes且性能稳定

### 参数演化策略

```python
# 阶段2的平滑参数演化示例
connectivity_penalty = 0.1 + 1.4 * progress  # 从0.1平滑过渡到1.5
action_mask_relaxation = 0.7 * (1 - progress)  # 从70%放松到0%
balance_weight = 0.8 - 0.2 * progress  # 从0.8降到0.6
decoupling_weight = 0.2 + 0.4 * progress  # 从0.2升到0.6
power_weight = 0.3 * progress  # 从0逐渐升到0.3
```

## 🛡️ 安全保障机制

### 训练崩溃检测

系统自动检测以下异常信号：
- Episode长度过短（< 3步）
- 奖励异常低（< -100）
- 损失爆炸（> 10）

### 自动恢复机制

检测到训练崩溃时，系统自动：
1. 切换到紧急恢复模式
2. 极度放松约束参数
3. 降低学习率
4. 记录恢复过程

### 性能回退保护

- 监控性能恶化趋势
- 超过容忍度时建议回退到上一阶段
- 保护训练稳定性

## 📈 监控与日志

### 新增监控指标

- 当前课程阶段和进度
- 参数演化轨迹
- 阶段转换触发原因
- 安全监控状态
- 智能导演决策历史

### TensorBoard可视化

```bash
tensorboard --logdir logs
```

新增可视化内容：
- 课程学习阶段时间线
- 参数演化曲线
- 性能趋势分析
- 阶段转换事件

## 🔧 高级配置

### 自定义阶段转换条件

```yaml
stage_transition:
  # 自定义阶段1→2转换
  episode_length_target: 20      # 提高目标长度
  episode_length_stability: 0.9  # 提高稳定性要求
  
  # 自定义阶段2→3转换
  composite_score_target: 0.8    # 提高质量要求
  composite_score_window: 150    # 扩大评估窗口
```

### 自定义参数演化

```yaml
parameter_evolution:
  # 调整连通性惩罚范围
  connectivity_penalty_range: [0.05, 2.0]
  
  # 调整学习率衰减
  learning_rate_decay_factor: 0.05  # 更激进的衰减
```

### 自定义安全阈值

```yaml
safety_monitoring:
  min_episode_length: 5          # 更宽松的最小长度
  max_reward_threshold: -50      # 更严格的奖励阈值
  performance_deterioration_patience: 100  # 更大的容忍度
```

## 🧪 测试与验证

### 运行组件测试

```bash
python test_adaptive_curriculum.py
```

测试内容：
- 所有组件功能验证
- 配置加载测试
- 集成测试
- 性能分析验证

### 对比实验

```bash
# 传统课程学习
python train.py --mode curriculum

# 智能自适应课程学习
python train.py --mode adaptive

# 标准训练（对照组）
python train.py --mode fast
```

## 📋 最佳实践

### 1. 初次使用建议

- 从默认配置开始
- 在小规模系统（IEEE14）上验证
- 观察阶段转换是否合理
- 根据需要调整转换条件

### 2. 参数调优指南

- **保守调优**：小幅调整转换阈值
- **激进调优**：缩短评估窗口，提高转换频率
- **稳定优先**：增大安全监控容忍度

### 3. 故障排除

#### 阶段转换过快
- 增大评估窗口
- 提高转换阈值
- 检查性能指标计算

#### 阶段转换过慢
- 减小评估窗口
- 降低转换阈值
- 检查数据质量

#### 训练不稳定
- 启用更严格的安全监控
- 减小参数演化速度
- 检查学习率设置

## 🔮 未来扩展

### 计划中的功能

- **元学习调度**：使用神经网络预测最优参数
- **多智能体协调**：训练智能体与调度智能体协同
- **预测性调度**：基于趋势预测提前调整参数
- **自适应评估窗口**：根据训练稳定性动态调整窗口大小

### 扩展接口

系统设计了灵活的扩展接口，支持：
- 自定义性能分析器
- 自定义参数调度策略
- 自定义安全监控规则
- 自定义阶段转换逻辑

---

## 📞 支持与反馈

如有问题或建议，请：
1. 查看日志文件中的详细信息
2. 运行测试脚本验证系统状态
3. 检查配置文件的参数设置
4. 参考故障排除指南

智能自适应课程学习系统让您的电力系统强化学习训练更加智能、高效、稳定！
