# 智能电网分区系统

基于强化学习的电网图分区项目，使用PPO算法和物理引导的GAT编码器实现连通性约束下的智能分区。

## 📁 项目结构

```
power_grid_partition_project/
├── 📄 README.md                    # 项目说明文档（英文）
├── 🎯 main.py                      # 主入口：运行整个项目流程，串联所有模块
├── ⚙️ config.py                    # 配置中心：统一管理项目的所有参数
├── 🧠 models.py                    # 核心模型：物理引导的图注意力网络(GAT)
├── 🤖 agent.py                     # PPO智能体：Actor-Critic网络和训练逻辑
├── 🌍 env.py                       # 课程学习环境：动态调整训练难度
├── 🏃 training.py                  # 训练脚本：强化学习训练循环
│
├── 📚 docs/                        # 📖 文档和报告
│   ├── README.md                   #   详细项目文档（英文）
│   ├── README_CN.md                #   详细项目文档（中文）
│   ├── ConnectivityViolationSuppression_Report.md  # 连通性分析报告
│   └── RewardSignalRefinement_Report.md            # 奖励函数优化报告
│
├── 🛠️ utils/                       # 🔧 工具模块
│   ├── data_processing.py          #   数据处理：加载并清洗电网数据，构建图结构
│   ├── metrics.py                  #   环境和评估：RL环境定义和分区质量指标
│   ├── utils.py                    #   通用工具：随机种子、设备配置等辅助函数
│   └── visualization.py            #   可视化：分区结果和训练过程图表展示
│
├── 📜 scripts/                     # 🚀 训练和测试脚本
│   ├── run_training.py             #   主要训练脚本（快速/完整训练模式）
│   ├── run_reward_refinement_training.py  # 奖励函数优化训练
│   ├── run_connectivity_test.py    #   连通性测试脚本
│   ├── baseline.py                 #   基线方法：传统分区算法对比
│   ├── mini_training_test.py       #   快速训练测试
│   ├── quick_integration_test.py   #   集成测试
│   ├── test_action_mask.py         #   动作掩码测试
│   ├── verify_action_mask.py       #   动作掩码验证
│   ├── simple_mask_test.py         #   简单掩码测试
│   ├── final_verification_summary.py  # 最终验证总结
│   └── verify_analytics_report.py #   分析报告验证
│
├── 🧪 tests/                       # ✅ 单元测试
│   ├── test_actor_mask.py          #   Actor模型动作掩码测试
│   ├── test_config_integration.py  #   配置集成测试
│   ├── test_environment_integration.py  # 环境集成测试
│   ├── test_environment_reward.py  #   环境奖励函数测试
│   └── test_metrics.py             #   指标计算测试
│
├── 💾 cache/                       # 🗄️ 缓存文件
├── 📊 data/                        # � 数据文件
├── 🏛️ models/                      # 💾 保存的模型文件
├── 📋 logs/                        # 📝 训练日志和分析数据
├── 📈 results/                     # 🎯 训练结果输出
└── 🖼️ figures/                     # 📊 生成的图表和可视化
```

## 文件功能详解：

*   **`main.py`**:
    *   **核心作用**: 项目的启动器和总指挥。
    *   **具体功能**:
        *   初始化全局设置（如选择CPU或GPU）。
        *   按顺序调用其他模块的核心功能，完成从数据加载、模型构建、环境设置、智能体训练、基准比较到结果可视化的完整流程。
        *   统一处理程序运行中的顶层错误和日志输出。

*   **`config.py`**:
    *   **核心作用**: 统一管理项目的所有参数，如分区数量、学习率等。
    *   **具体功能**:
        *   集中管理所有与项目相关的配置参数，便于统一管理和维护。

*   **`data_processing.py`**:
    *   **核心作用**: 电网数据的预处理和图结构构建。
    *   **具体功能**:
        *   `PowerGridDataProcessor` 类：
            *   读取MATPOWER等标准格式的原始电网数据。
            *   清洗数据，例如处理缺失值。
            *   提取节点（母线）特征：有功/无功负荷、电压、节点度数、节点类型（PQ, PV, Slack）等。
            *   提取边（支路）特征：电阻、电抗、导纳、线路容量、是否为变压器等。
            *   提取发电机特征并关联到相应节点。
            *   对提取的特征进行标准化（如使用`RobustScaler`）。
            *   将处理后的数据转换为 `torch_geometric.data.Data` 对象，供图神经网络使用。
        *   `generate_random_power_grid`: 一个辅助函数，用于生成随机的电网数据，方便测试。

*   **`models.py`**:
    *   **核心作用**: 定义图神经网络模型，特别是用于节点嵌入学习的Physics-Guided GAT。
    *   **具体功能**:
        *   `PhysicsGuidedGATConv` 类：
            *   自定义的GAT卷积层，继承自 `torch_geometric.nn.GATConv`。
            *   **创新点**: 在计算注意力权重时，融入了电网的物理特性（如线路阻抗/导纳），使得物理上连接更紧密的节点间能有更高的注意力。包含一个可学习的`temperature`参数来调整物理先验的影响。
        *   `EnhancedGATEncoder` 类：
            *   构建一个多层的GAT编码器，使用 `PhysicsGuidedGATConv` 作为核心层。
            *   集成了残差连接、层归一化、自适应Dropout等技术来提升模型性能和训练稳定性。
            *   可选地加入位置编码（基于图拉普拉斯特征向量），帮助模型理解节点的全局结构角色。

*   **`env.py`**:
    *   **核心作用**: 实现课程学习机制，包装基础RL环境。
    *   **具体功能**:
        *   `CurriculumLearningEnv` 类：
            *   包装一个基础的 `PowerGridPartitionEnv` 实例（定义在 `metrics.py`）。
            *   动态调整任务难度：开始时任务较简单（如预设部分节点分区），随着智能体表现提升，逐渐增加难度（如收紧约束条件、减少预设）。
            *   通过监控智能体的成功率来调整难度级别，旨在加速训练收敛并提高最终性能。

*   **`metrics.py`**:
    *   **核心作用**: 定义强化学习的基础环境 (`PowerGridPartitionEnv`) 和评估分区质量的各项指标。
    *   **具体功能**:
        *   `PartitionMetrics` dataclass: 用于存储一套完整的分区评估指标。
        *   `PowerGridPartitionEnv` 类：
            *   **状态 (State)**: 表示当前电网的分区情况，可能包括节点分配向量 `z`、边界节点信息、各区域的嵌入表示、全局上下文信息等。
            *   **动作 (Action)**: 定义智能体可以执行的操作，通常是将一个未分配的边界节点分配给一个已有的分区。
            *   **奖励 (Reward)**: 设计多目标奖励函数，综合考虑负荷均衡度（如CV系数、Gini系数）、解耦度（区域间连接线的电气耦合）、区域内部连通性、功率平衡、分区进度等。
            *   **环境动态**: 实现 `reset` 方法（初始化环境，选择种子节点开始分区）和 `step` 方法（执行动作、更新状态、计算奖励、判断是否结束）。
            *   **指标计算**: 包含计算各种详细分区指标的方法，如：
                *   `load_cv`, `load_gini`: 衡量各分区负荷的均衡程度。
                *   `total_coupling`: 区域间线路的总耦合强度。
                *   `inter_region_lines`: 跨区联络线的数量。
                *   `connectivity`: 确保每个分区内部是连通的。
                *   `power_balance`: 区域内发电与负荷的匹配程度。
                *   `modularity`: 图论中衡量社区划分质量的指标。

*   **`agent.py`**:
    *   **核心作用**: 实现PPO强化学习智能体。
    *   **具体功能**:
        *   `PPOMemory` 类: 用于存储智能体与环境交互过程中产生的经验（状态、动作、奖励、概率等），并支持GAE（广义优势估计）的计算。
        *   `HierarchicalActor` 类 (策略网络):
            *   根据当前状态决定采取哪个动作（即如何分配下一个节点）。
            *   可能采用分层决策来处理复杂的动作空间。
        *   `Critic` 类 (价值网络):
            *   评估当前状态的价值（即从当前状态开始能获得的预期累积奖励）。
        *   `PPOAgent` 类:
            *   整合Actor和Critic网络。
            *   实现 `select_action` 方法，根据策略网络选择动作并与环境交互。
            *   实现 `update` 方法，使用收集到的经验数据，按照PPO算法的优化目标（包含裁剪的概率比率、价值函数损失和熵奖励）来更新Actor和Critic网络的参数。
            *   包含模型保存和加载的功能。

*   **`training.py`**:
    *   **核心作用**: 管理和执行强化学习的训练过程。
    *   **具体功能**:
        *   `train_ppo` 函数：
            *   组织主要的训练循环：在多个回合（episodes）中进行。
            *   在每个回合中，智能体与环境交互（`env.step()`），收集经验。
            *   周期性地调用 `agent.update()` 来训练智能体。
            *   记录训练过程中的各种数据，如每回合奖励、分区指标、损失函数值等。
            *   支持使用 TensorBoard 进行训练过程的可视化监控。
            *   周期性地保存训练好的模型。

*   **`baseline.py`**:
    *   **核心作用**: 实现传统的、非AI的电网分区方法，作为AI方法性能的对比基准。
    *   **具体功能**:
        *   `BaselinePartitioner` 类：
            *   实现多种经典分区算法，如：
                *   谱聚类 (`_spectral_partition`): 基于图拉普拉斯矩阵的特征向量进行聚类。
                *   K-means聚类 (`_kmeans_partition`): 基于节点嵌入向量进行聚类。
                *   随机分区 (`_random_partition`): 作为最简单的对比。
        *   `evaluate_partition_method`: 使用 `metrics.py` 中定义的指标来评估这些基线方法产生的分区结果。
        *   `compare_methods`: 统一调用AI方法和各种基线方法进行分区，并收集它们的性能指标，生成对比表格。

*   **`visualization.py`**:
    *   **核心作用**: 将抽象的分区结果和训练数据以直观的图形方式展示。
    *   **具体功能**:
        *   `visualize_partition`:
            *   将电网分区结果绘制成网络图：节点按所属分区着色，跨区联络线高亮显示。
            *   可以同时展示相关的分区指标、各区域负荷分布条形图、区域间耦合矩阵热力图等。
        *   `plot_training_curves`:
            *   绘制训练过程中的指标变化曲线，如奖励曲线、损失函数曲线、负荷均衡度指标曲线等，帮助分析训练效果和模型收敛情况。
        *   `create_interactive_visualization` (可选): 可能使用Plotly等库创建交互式的可视化图表。

*   **`utils.py`**:
    *   **核心作用**: 提供项目通用的辅助函数和配置。
    *   **具体功能**:
        *   集中导入常用的标准库和第三方库。
        *   `set_seed`: 设置全局随机种子（Python内置random, NumPy, PyTorch），以保证实验结果的可复现性。
        *   配置计算设备（自动选择GPU或CPU）。
        *   创建项目运行时所需的输出目录（如 `models/`, `figures/`, `logs/`）。
        *   可能包含其他通用的工具函数。

## 模型性能评估指标

connectity 是第一道门槛 = 1 意味着子图内部联通，否则方案是失败的

1.  **`load_cv` (Load Coefficient of Variation - 负荷变异系数)**:

    这个值越小，表示各个分区的负荷量越接近，负荷分配越均衡。理想情况下趋近于0。

2.  **`load_gini` (Load Gini Coefficient - 负荷基尼系数)**:

    [0,1] 越接近0，表示负荷分配越平均；越接近1，表示负荷分配越不平均，即少数分区承担了大部分负荷。

3.  **`total_coupling` (Total Coupling - 总耦合度)**:

    这个值越小，表示分区之间的电气耦合越弱，分区独立性越好。

4.  **`inter_region_lines` (Inter-Region Lines - 跨区联络线数量)**:

    数量越少，表示分区之间的物理连接越少，结构上更独立。

5.  **`power_balance` (Power Balance - 功率平衡)**:

    理想情况下，每个分区应尽可能实现功率自给自足，即发电量约等于消耗量。这个指标可能衡量的是各分区功率不平衡量的总和或平均值。越接近0（或不平衡度越小）越好。

6.  **`modularity` (Modularity - 模块度)**:

    模块度的值通常在-0.5到1之间。越高的模块度表示社区结构越明显，即社区内部的连接远比社区之间的连接要紧密。



