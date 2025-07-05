# 电力网络分区智能体评估说明（test.md）

> 本文档汇总了 **evaluation_results** 中各项指标的含义、计算公式与测试流程，方便项目成员或第三方快速理解评估体系。

---

## 1. 测试框架概览

1. **运行脚本**：`python test.py`，支持三种模式
   - `--baseline`    只在指定网络上与 Baseline 方法对比。
   - `--generalization`   训练–测试跨网络泛化能力评估。
   - `--comprehensive`   先基线对比再泛化测试，生成完整报告。
2. **本次示例**执行了 `--baseline`，载入预训练模型 `agent_ieee57_adaptive_best.pth`，在同一张 **IEEE-57** 网络的 4 个场景下测试：
   - *normal*（正常负荷）
   - *high_load*（高负荷）
   - *unbalanced*（节点负荷失衡）
   - *fault*（线路故障）
3. **对比对象**
   - `my_agent`          强化学习智能体（PPO）
   - `random`            随机分区
   - `degree_based`     基于度数启发式
   - `spectral`         谱聚类（Laplacian）
4. **重复实验**：默认每个场景运行 10 次，取平均值后再比较。
5. **输出**
   - 图表：`evaluation_results/comparison_visualization.png`
   - 报告：`evaluation_results/evaluation_report.txt`

---

## 2. 核心指标

| 指标简称 | 原始符号 (Raw) | 物理 / 数学意义 | 期望方向 | 计算公式 |
|---------|---------------|----------------|---------|----------|
| 负载平衡度 | `load_b` (=CV) | 各分区总负荷的变异系数 (Coefficient of Variation) | 越小越好 | \[ \operatorname{CV}=\frac{\sigma_{P}}{\mu_{P}} \]
| 电气耦合率 | `decoupling` (coupling ratio) | 跨分区线路数占比 | 越小越好 | \[ r_{\text{coup}} = \frac{\#\text{cross-edges}}{\#\text{total edges}} \]
| 功率不平衡 | `power_b` | 分区内发–负荷偏差的归一化绝对和 | 越小越好 | \[ r_{\text{imb}} = \frac{\sum_{k}|P^{\text{gen}}_k-P^{\text{load}}_k|}{\sum_{i}|P^{\text{load}}_i|} \]

> 以上原始指标“越小越好”，不利于直观比较，故统一映射为 **Score**（越大越好）。

### 2.1 指标映射 (Raw → Score)

| Score 名称 | 映射公式 | 取值范围 |
|------------|---------|-----------|
| `load_b_score` | \[ S_{\text{load}} = \frac{1}{1+\text{CV}} \] | (0, 1] |
| `decoupling_score` | \[ S_{\text{dec}} = 1- r_{\text{coup}} \] | [0, 1] |
| `power_b_score` | \[ S_{\text{pow}} = \frac{1}{1+r_{\text{imb}}} \] | (0, 1] |

### 2.2 综合质量分数

采用加权平均：

\[
Q = w_{\text{load}}\,S_{\text{load}} + w_{\text{dec}}\,S_{\text{dec}} + w_{\text{pow}}\,S_{\text{pow}}, \quad \sum w_i =1
\]

> 默认权重 `w = {0.4, 0.4, 0.2}`，可在 `evaluation_config['metrics_weights_v2']` 中调整。

---

## 3. 场景生成与实验流程

1. **场景生成**：`ScenarioGenerator.generate_test_scenarios()` 基于原始 MPC 数据动态构造 *high_load / unbalanced / fault* 版本：
   - **高负荷**：将所有负荷放大 50 %。
   - **失衡**：随机挑选 30 % 节点，将其负荷放大 80 %，其余缩小 20 %。
   - **故障**：删除一条关键线路造成拓扑缺口。
2. **环境初始化**：统一使用 3 分区（57 及以下节点）或 5 分区（>57 节点），保证与预训练模型结构匹配。
3. **智能体评估**：在 `PowerGridPartitioningEnv` 中执行最多 200 步操作；终局分区经 `RewardFunction.get_current_metrics()` 产出 Raw 指标。
4. **Baseline 评估**：直接调用分区算法得到静态结果，无 RL 迭代。
5. **结果聚合**：对每个场景-方法取 10 次平均，保存 Raw + Score + Q。
6. **可视化**：
   - **上排 4 图**：三个 Score + Q，Y 轴“↑ Better”。
   - **下排 3 图**：对应 Raw 指标，Y 轴“↓ Better”。

---

## 4. 文件结构与输出解析

```
└─ evaluation_results/
   ├─ comparison_visualization.png   # 本文档嵌入的 2×4 柱状图
   └─ evaluation_report.txt          # 文本报告（Raw、Score、Q 及增益统计）
```

报告字段说明：

| 列名 | 对应变量 | 单位 | 说明 |
|------|----------|------|------|
| 负载CV | `load_b` | — | Raw 负载平衡指标，越小越好 |
| 耦合率 | `decoupling` | — | Raw 电气耦合率 |
| 功率不平衡 | `power_b` | — | Raw 功率不平衡 |
| 负载S / 解耦S / 功率S | `*_score` | — | Raw 指标映射后的 Score |
| 综合分 | `comprehensive_score` | — | 三个 Score 的加权和 |

---

## 5. 过时 / 待删除文件

| 文件 | 理由 | 建议操作 |
|------|------|-----------|
| `code/test/evaluation_config.py` | 仍包含旧版 `inter_region_cv` / `intra_region_cv` 阈值配置，当前评估流程已迁移至新版 Score 体系。 | 若无其他模块引用，可删除或重构为新版权重格式。 |
| （暂无） | 此次重构未发现其他明显冗余文件；如需进一步精简，可搜索旧指标 `inter_region_cv` / `intra_region_cv` 或未调用模块后手动移除。 |

> ⚠️ 删除文件前请确保未被训练脚本、可视化或文档引用，以免破坏向后兼容。

---

## 6. 后续改进方向

1. **可视化交互化**：将静态图迁移到 Plotly/Dash 以支持鼠标悬浮与筛选。
2. **指标可扩展**：若将来引入网络可靠性、稳定裕度等新维度，可在 `convert_to_scores()` 内统一映射并更新综合权重。
3. **自动废弃检测**：CI 步骤中添加 `ruff --unused` 或 pytest 覆盖率扫描，防止遗留死代码。

---

如对指标或流程仍有疑问，欢迎在 Issues 中讨论。
