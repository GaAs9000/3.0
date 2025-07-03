# 🔋 电力网络分区强化学习系统 2.0

基于图注意力网络(GAT)和强化学习(PPO)的电力网络智能分区系统，采用智能自适应训练。

## 🚀 快速开始

### 安装依赖
```bash
# 基础依赖
pip install torch torch-geometric numpy matplotlib pyyaml tqdm

# TUI前端界面和可视化依赖
pip install textual plotly jinja2

# 可选依赖（电力系统分析）
pip install pandapower networkx seaborn
```

### 立即开始训练
```bash
# 默认快速训练（推荐）
python train.py

# 指定训练模式
python train.py --mode fast      # 快速训练 - 1000回合
python train.py --mode full      # 完整训练 - 5000回合
python train.py --mode ieee118   # 大规模训练 - IEEE118节点

# 调整参数
python train.py --mode fast --episodes 1500 --case ieee30

# 启用TUI前端界面（推荐！）
python train.py --mode fast --tui
```

### 模型测试和评估
```bash
# 🔬 快速跨网络泛化测试（推荐）
python test.py --quick

# 详细跨网络测试
python test.py --quick --episodes 10

# 完整评估系统（开发中）
python test.py --help
```
# 📈 TensorBoard监控
tensorboard --logdir=data/logs --port=6006

## 📊 训练模式
11
| 模式 | 回合数 | 时间 | 特点 |
|------|--------|------|------|
| `fast` | 1000 | ~1小时 | 快速训练，智能自适应 |
| `full` | 5000 | ~4小时 | 完整训练，并行优化 |
| `ieee118` | 3000 | ~6小时 | 大规模系统训练 |

> 💡 所有模式都默认启用智能自适应功能

## 🎯 核心特性

- **🧠 智能自适应**: 4阶段课程学习，自动调优，平台化检测
- **🎭 场景生成**: N-1故障，负荷扰动，组合场景
- **📊 数值稳定**: 全面NaN/inf保护，梯度裁剪
- **🖥️ TUI前端界面**: 基于Textual的现代化交互式监控界面
- **🔬 模型测试**: 跨网络泛化测试，性能评估，基线对比
- **📈 可视化分析**: TensorBoard监控，Plotly交互式图表，HTML仪表板
- **⚙️ 并行训练**: Stable-Baselines3并行优化
- **🎨 美化输出**: Rich库彩色终端显示

## 📁 项目结构

```
├── train.py              # 主训练脚本
├── test.py               # 模型测试和评估脚本
├── config.yaml           # 配置文件
├── code/src/             # 核心代码
│   ├── rl/               # 强化学习模块
│   ├── tui_monitor.py    # TUI前端界面
│   ├── html_dashboard_generator.py  # HTML仪表板
│   ├── plotly_chart_factory.py      # 交互式图表
│   └── visualization.py             # 可视化系统
├── data/                 # 数据和日志
└── docs/                 # 文档
```

## 🔧 配置说明

智能自适应默认启用，可在`config.yaml`中调整：

```yaml
# 智能自适应配置（默认启用）
adaptive_curriculum:
  enabled: true
  stage_transition:
    episode_length_target: 10
    plateau_detection_enabled: true

# 可视化功能
visualization:
  enabled: true
  interactive: true
  save_figures: true

html_dashboard:
  output_dir: output/dashboards
  enable_compression: true

# 监控和日志
logging:
  use_tensorboard: true
  generate_html_dashboard: true

# TUI前端界面
tui:
  enabled: false  # 通过--tui参数启用
```

## 📞 技术支持

遇到问题请检查：
1. 依赖安装：`python train.py --help`
2. 配置文件：`config.yaml`
3. 日志目录：`data/logs/`

---

**一键启动训练：**
```bash
python train.py
```

**查看训练监控：**
```bash
# 🔥 推荐：TUI前端界面（实时交互式监控）
python train.py --mode fast --tui

# TensorBoard监控
tensorboard --logdir=data/logs --port=6006

# 查看生成的图表和仪表板
# 结果位置：data/figures/ 和 output/dashboards/
```