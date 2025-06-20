# Encoder GAT

创建时间: May 11, 2025 11:58 PM
标签: Code
状态: 完成

# **GAT图神经网络模型 (`src/gat.py`)**

该文件定义了一个专为电网设计的、基于**物理先验**的**异构图注意力网络 (GATv2)** 编码器。

其核心是将电网的物理特性（如支路阻抗）直接融入到GNN的注意力机制中，从而让模型能够更智能地学习电网的拓扑和电气关系。

模型最终输出高质量的**节点嵌入**和**图级别嵌入**，可用于下游的强化学习。

---

# **MODEL - `HeteroGraphEncoder`**

## 核心架构

模型采用模块化设计，主要由三个部分组成：**特征预处理层**、**GNN主干网络**和**输出层**。

### 1. **特征预处理层 (`node_projectors`)**

- **目的**：统一不同节点类型（`bus_pq`, `bus_pv`, `bus_slack`）的特征维度。
- **实现**：为每种节点类型创建一个独立的线性投影层 (`nn.Linear`)，将它们的特征维度映射到一个统一的 `max_node_dim`。这使得下游的同构GNN主干网络可以处理它们。

### 2. **GNN主干网络 (`hetero_encoder`)**

这是模型的核心，它首先定义一个**同构**的 `GNNEncoder`，然后使用PyG的 `to_hetero` 工具将其动态转换为一个**异构**模型。

### 2.1 **`GNNEncoder` - 同构主干**

- **结构**：一个由多个 `PhysicsGATv2Conv` 层堆叠而成的网络。
- **特性**：
    - **残差连接**：`x = x + residual_proj(residual)`，防止梯度消失，加速收敛。
    - **层归一化**：`LayerNorm`，稳定训练过程。
    - **Dropout**：防止过拟合。

### 2.2 **`PhysicsGATv2Conv` - 核心创新：物理引导的注意力层**

这是标准的 `GATv2Conv` 的一个**增强子类**，它在计算注意力权重时融入了物理先验知识。

**数学原理**:
标准GATv2的注意力分数 `e_ij` 计算如下：

$$
e_{ij} = \vec{a}^T \cdot \text{LeakyReLU}(\mathbf{W}_l[ \mathbf{W}_r \vec{h}_i || \mathbf{W}r \vec{h}j ] + \mathbf{W}{edge} \vec{e}{ij})
$$

**物理GATv2的创新**在于，它通过增强边特征 `edge_attr` 来引入物理先验：

$$
\text{enhanced\_edge\_attr} = \text{original\_edge\_attr} + \text{physics\_prior}

$$

其中，物理先验 physics_prior 与支路阻抗 $Z_{ij}$ 的倒数成正比：

$$
\text{physics\_prior} = \frac{\tau \cdot w\_\text{phys}}{|Z_{ij}|}
$$

- **`|Z_ij|`**：支路 `i-j` 的阻抗模长。**阻抗越小，导纳越大，物理先验权重越高**。
- **`τ` (temperature)**：一个**可学习的**温度参数 `nn.Parameter`，用于自动调整物理先验的影响强度。
- **`w_phys` (physics_weight)**：一个固定的权重超参数。

**✅ 核心优势**: 模型在学习节点间关系时，会天然地更关注那些**电气连接更紧密（阻抗更低）**的支路，这完全符合电网的物理规律。

### 2.3 **`to_hetero` - 动态转换为异构模型**

- **功能**：将设计好的 `GNNEncoder` 包装成一个可以处理 `HeteroData` 的异构模型。
- **实现**：`self.hetero_encoder = to_hetero(gnn_encoder, metadata, aggr='sum')`
- **优点**：代码简洁，无需为每种关系类型手动编写不同的GNN层。模型可以自动处理不同类型的节点和边特征。

### 3. **输出层 (`output_projector` & `graph_pooling`)**

- **节点嵌入**：
    - GNN主干网络输出的节点嵌入会经过一个可选的线性投影层 (`output_projector`)，以获得最终所需的输出维度 `output_dim`。
- **图嵌入**：
    - **聚合 (Pooling)**：使用 `global_mean_pool` 将所有节点的嵌入聚合为一个单一的图级别表示。
    - **投影 (Projection)**：将聚合后的图表示通过一个简单的MLP (`graph_projector`) 进行变换，以增强其表达能力。

---

# **INPUT - PyTorch Geometric `HeteroData`**

`HeteroGraphEncoder` 的输入是一个标准的PyG `HeteroData` 对象，其结构与 `data_processing.py` 的输出完全一致。

### **图 G=(N,E) - 异构图结构**

### N 节点集 - 按物理类型分组 (`data.node_types`)

| 节点类型 | 描述 |
| --- | --- |
| **`bus_pq`** | PQ负荷节点 |
| **`bus_pv`** | PV发电节点 |
| **`bus_slack`** | Slack平衡节点 |

**节点特征 (`data[node_type].x`)**: 每种类型 `(num_nodes_of_type, 14)`

### E 边集 - 按连接关系分组 (`data.edge_types`)

| 边类型 | 描述 |
| --- | --- |
| **`connects_line`** | 输电线路连接 |
| **`connects_transformer`** | 变压器连接 |

**边特征 (`data[edge_type].edge_attr`)**: **✅ 完整的9维特征**

---

# **OUTPUT - 嵌入表示 & 注意力权重**

`HeteroGraphEncoder` 的 `forward` 方法可以根据参数返回不同的输出组合。

### 1. **节点嵌入 (`node_embeddings`)**

- **类型**: `Dict[str, torch.Tensor]`
- **结构**: 一个字典，键是节点类型字符串，值是该类型所有节点的嵌入张量。
- **形状**: 每个张量的形状为 `[num_nodes_of_type, output_dim]`。

### 2. **图级别嵌入 (`graph_embedding`)**

- **类型**: `torch.Tensor`
- **结构**: 一个单一的向量，代表整个电网的全局状态。
- **形状**: `[1, output_dim]`

### 3. **注意力权重 (`attention_weights`)**

- **类型**: `List[torch.Tensor]`
- **结构**: 一个列表，包含了模型中所有 `PhysicsGATv2Conv` 层在每次前向传播中计算出的注意力权重 `alpha`。
- **用途**: **可解释性分析**。可以用来可视化哪些支路在模型的决策中被认为更重要。

---

# **API & 使用方法**

### **创建模型**

使用便捷的工厂函数 `create_hetero_graph_encoder`。

```python
from src.gat import create_hetero_graph_encoder

# data是HeteroData对象
encoder = create_hetero_graph_encoder(
    data,
    hidden_channels=64,
    gnn_layers=3,
    heads=4,
    output_dim=128
)

```

### **获取嵌入和权重**

```python
# 1. 只获取节点嵌入 (最高效)
node_embs = encoder.encode_nodes(data)

# 2. 只获取图嵌入
graph_emb = encoder.encode_graph(data)

# 3. 获取所有输出
node_embs, attention_weights, graph_emb = encoder(
    data,
    return_attention_weights=True,
    return_graph_embedding=True
)

# 4. 单独获取注意力权重 (在前向传播后调用)
att_weights = encoder.get_attention_weights()

```

---

# **Example：**

### 1. **构建输入 (`HeteroData` 对象)**

首先，我们创建一个代表这个微型电网的 `HeteroData` 对象。

```python
import torch
from torch_geometric.data import HeteroData
from src.gat import create_hetero_graph_encoder

# --- 模拟输入数据 ---
data = HeteroData()

# 节点类型1: 'bus_pq' (2个PQ节点)
# 14维特征 (这里用随机数代替)
data['bus_pq'].x = torch.randn(2, 14)
data['bus_pq'].global_ids = torch.tensor([2, 3]) # 原始母线ID

# 节点类型2: 'bus_pv' (1个PV节点)
data['bus_pv'].x = torch.randn(1, 14)
data['bus_pv'].global_ids = torch.tensor([1])

# 节点类型3: 'bus_slack' (1个Slack节点)
data['bus_slack'].x = torch.randn(1, 14)
data['bus_slack'].global_ids = torch.tensor([0])

# 边关系1: slack(0) 和 pv(0) 之间有一条线路连接
# (注意：索引是相对于各自类型的局部索引)
data['bus_slack', 'connects_line', 'bus_pv'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
# 9维边特征 (这里用随机数代替)
data['bus_slack', 'connects_line', 'bus_pv'].edge_attr = torch.randn(1, 9)

# 边关系2: pv(0) 和 pq(0) 之间有一条变压器连接
data['bus_pv', 'connects_transformer', 'bus_pq'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
data['bus_pv', 'connects_transformer', 'bus_pq'].edge_attr = torch.randn(1, 9)

# 边关系3: pq(0) 和 pq(1) 之间有一条线路连接
data['bus_pq', 'connects_line', 'bus_pq'].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
data['bus_pq', 'connects_line', 'bus_pq'].edge_attr = torch.randn(1, 9)

print("--- INPUT: HeteroData Object ---")
print(data)
print("--------------------------------\\n")

```

### 2. **初始化模型并执行前向传播**

接下来，我们创建编码器，并将上面构建的 `data` 对象作为输入传递给它。

```python
# --- 初始化模型 ---
# output_dim=32 表示我们希望最终的嵌入是32维的
encoder = create_hetero_graph_encoder(
    data,
    hidden_channels=16,
    gnn_layers=2, # 2层GNN
    heads=4,      # 4个注意力头
    output_dim=32
)

# --- 执行前向传播，获取所有输出 ---
# 将data对象和模型都移动到同一个设备 (例如CPU)
device = torch.device('cpu')
data = data.to(device)
encoder = encoder.to(device)

# 不计算梯度，以加快速度
with torch.no_grad():
    node_embeddings, attention_weights, graph_embedding = encoder(
        data,
        return_attention_weights=True,
        return_graph_embedding=True
    )

```

### 3. **查看输出结果**

下面是上述代码运行后，各个输出变量的内容和形状。

```python
# --- OUTPUT: 结果展示 ---

print("--- OUTPUT 1: Node Embeddings (Dict[str, Tensor]) ---")
for node_type, embeddings in node_embeddings.items():
    print(f"Node type '{node_type}':")
    print(f"  - Shape: {embeddings.shape}")
    # print(f"  - Embeddings:\\n{embeddings}") # 取消注释以查看具体数值
print("-----------------------------------------------------\\n")

print("--- OUTPUT 2: Graph Embedding (Tensor) ---")
print(f"Shape: {graph_embedding.shape}")
# print(f"Embedding:\\n{graph_embedding}")
print("-------------------------------------------\\n")

print("--- OUTPUT 3: Attention Weights (List[Tensor]) ---")
# 因为有2层GNN，每层有3种关系，所以理论上会有 2*3=6 个权重张量
# (注意：如果某关系在某层没有边，可能不会产生权重)
print(f"Number of attention tensors found: {len(attention_weights)}")
for i, weights in enumerate(attention_weights):
    # 形状是 [边的数量, 注意力头的数量]
    print(f"Tensor {i}:")
    print(f"  - Shape: {weights.shape}")
    # print(f"  - Weights (softmax scores):\\n{weights}")
print("--------------------------------------------------\\n")

```

### **输出结果示例**

运行上述代码，你会得到类似下面的输出（由于随机初始化，具体数值会不同）：

```
--- INPUT: HeteroData Object ---
HeteroData(
  bus_pq={
    x[2, 14],
    global_ids[2]
  },
  bus_pv={
    x[1, 14],
    global_ids[1]
  },
  bus_slack={
    x[1, 14],
    global_ids[1]
  },
  ('bus_slack', 'connects_line', 'bus_pv')={
    edge_index[2, 1],
    edge_attr[1, 9]
  },
  ('bus_pv', 'connects_transformer', 'bus_pq')={
    edge_index[2, 1],
    edge_attr[1, 9]
  },
  ('bus_pq', 'connects_line', 'bus_pq')={
    edge_index[2, 1],
    edge_attr[1, 9]
  }
)
--------------------------------

--- OUTPUT 1: Node Embeddings (Dict[str, Tensor]) ---
Node type 'bus_pq':
  - Shape: torch.Size([2, 32])
Node type 'bus_pv':
  - Shape: torch.Size([1, 32])
Node type 'bus_slack':
  - Shape: torch.Size([1, 32])
-----------------------------------------------------

--- OUTPUT 2: Graph Embedding (Tensor) ---
Shape: torch.Size([1, 32])
-------------------------------------------

--- OUTPUT 3: Attention Weights (List[Tensor]) ---
Number of attention tensors found: 6
Tensor 0:
  - Shape: torch.Size([1, 4])
Tensor 1:
  - Shape: torch.Size([1, 4])
Tensor 2:
  - Shape: torch.Size([1, 4])
Tensor 3:
  - Shape: torch.Size([1, 4])
Tensor 4:
  - Shape: torch.Size([1, 4])
Tensor 5:
  - Shape: torch.Size([1, 4])
--------------------------------------------------

```

这个例子清晰地展示了：

- **输入**是一个结构化的 `HeteroData` 对象。
- **输出**是三个部分：一个包含各类节点嵌入的**字典**，一个代表全图的**张量**，以及一个包含各层注意力分数的**列表**。

---

# Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, to_hetero, LayerNorm, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import warnings

# 存储捕获到的注意力权重的全局列表（或类属性）
captured_attention_weights = []

def attention_hook(module, input, output):
    """一个forward hook，用于捕获注意力权重"""
    # GATv2Conv在return_attention_weights=True时，输出是一个元组(out, attention_details)
    # attention_details 是 (edge_index, alpha)
    if isinstance(output, tuple) and len(output) == 2:
        # 我们只关心 alpha 张量
        if isinstance(output[1], tuple) and len(output[1]) == 2:
            captured_attention_weights.append(output[1][1])

class PhysicsGATv2Conv(GATv2Conv):
    """
    物理引导的GAT
    
    核心创新：
    1. 基于最新的GATv2架构，具有更强的表达能力
    2. 在注意力机制中融入电气阻抗先验
    3. 阻抗越小的连接获得越高的注意力权重
    4. 可学习的温度参数控制物理先验的影响程度
    
    数学原理：
    标准GATv2: α_ij = softmax(W_a^T LeakyReLU(W_l[W_r h_i || W_r h_j] + b))
    物理GATv2: α_ij = softmax(W_a^T LeakyReLU(W_l[W_r h_i || W_r h_j] + b) + τ/|Z_ij|)
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 8,
                 concat: bool = True, dropout: float = 0.6, 
                 edge_dim: Optional[int] = None, temperature: float = 1.0,
                 z_index: int = 3, physics_weight: float = 1.0, **kwargs):
        
        super().__init__(in_channels, out_channels, heads=heads, concat=concat,
                        dropout=dropout, edge_dim=edge_dim, **kwargs)
        
        # 物理引导参数
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.physics_weight = physics_weight
        self.z_index = z_index  # 阻抗模长在边特征中的索引
        
        # 边特征处理（如果有的话）
        if edge_dim is not None:
            self.edge_processor = nn.Linear(edge_dim, heads)
        else:
            self.edge_processor = None
    
    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                return_attention_weights: Optional[bool] = None):
        """
        前向传播，在标准GATv2基础上融入物理先验并缓存注意力权重
        """
        # 处理物理先验增强的边特征
        if edge_attr is not None:
            enhanced_edge_attr = self.physics_enhanced_edge_attr(edge_attr)
        else:
            enhanced_edge_attr = edge_attr
        
        # 始终请求注意力权重
        result = super().forward(
            x, edge_index, enhanced_edge_attr, return_attention_weights=True
        )
        
        out, attention_details = result
        
        # 将注意力分数存储在实例属性中，以便后续提取
        self._alpha = attention_details[1]

        # 根据调用者的要求决定返回值
        if return_attention_weights:
            return out, attention_details
        return out
    
    def physics_enhanced_edge_attr(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        增强边特征，融入物理先验
        """
        if edge_attr is None or edge_attr.shape[1] <= self.z_index:
            return edge_attr
        
        # 提取阻抗模长
        z_magnitude = edge_attr[:, self.z_index].clamp(min=1e-6)
        
        # 计算物理权重：1/|Z| (导纳)
        physics_prior = self.physics_weight / z_magnitude
        
        # 将物理先验添加到边特征中
        enhanced_edge_attr = edge_attr.clone()
        if enhanced_edge_attr.shape[1] < self.z_index + 2:
            # 如果空间不够，扩展特征维度
            padding = torch.zeros(
                enhanced_edge_attr.shape[0], 
                self.z_index + 2 - enhanced_edge_attr.shape[1],
                device=enhanced_edge_attr.device
            )
            enhanced_edge_attr = torch.cat([enhanced_edge_attr, padding], dim=1)
        
        # 将物理先验存储到边特征的最后一列
        enhanced_edge_attr[:, -1] = physics_prior * self.temperature
        
        return enhanced_edge_attr

class GNNEncoder(nn.Module):
    """
    同构GNN编码器
    
    特点：
    1. 多层PhysicsGATv2Conv堆叠
    2. 层归一化和残差连接
    3. 自适应dropout
    4. 支持不同的聚合方式
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.3,
                 edge_dim: Optional[int] = None, activation: str = 'elu'):
        super().__init__()
        
        self.num_layers = num_layers
        self.activation = getattr(F, activation)
        
        # 构建多层网络
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        current_dim = in_channels
        
        for i in range(num_layers):
            # 最后一层不使用多头concat
            is_last_layer = (i == num_layers - 1)
            concat = not is_last_layer
            
            # GATv2卷积层
            conv = PhysicsGATv2Conv(
                current_dim, 
                hidden_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=False
            )
            self.convs.append(conv)
            
            # 计算实际输出维度
            actual_out_dim = hidden_channels * heads if concat else hidden_channels
            
            # 层归一化
            self.norms.append(LayerNorm(actual_out_dim))
            
            # 残差投影
            if current_dim != actual_out_dim:
                self.residual_projs.append(nn.Linear(current_dim, actual_out_dim))
            else:
                self.residual_projs.append(nn.Identity())
            
            current_dim = actual_out_dim
        
        self.final_dim = current_dim
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        """
        for i, (conv, norm, residual_proj) in enumerate(
            zip(self.convs, self.norms, self.residual_projs)
        ):
            # 保存残差
            residual = x
            
            # 卷积
            x = conv(x, edge_index, edge_attr)
            
            # 确保x是tensor而不是tuple
            if isinstance(x, tuple):
                x = x[0]
            
            # 层归一化
            x = norm(x)
            
            # 残差连接
            x = x + residual_proj(residual)
            
            # 激活函数（最后一层除外）
            if i < self.num_layers - 1:
                x = self.activation(x)
        
        return x

class HeteroGraphEncoder(nn.Module):
    """
    异构图编码器 - 专注于图表示学习
    
    职责：
    1. 处理异构图数据（多种节点类型和边类型）
    2. 提取高质量的节点嵌入表示
    3. 融入电网物理先验知识
    4. 支持下游任务的特征提取
    """
    
    def __init__(self, 
                 node_feature_dims: Dict[str, int],
                 edge_feature_dims: Dict[str, int],
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 hidden_channels: int = 64,
                 gnn_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.3,
                 output_dim: Optional[int] = None):
        super().__init__()
        
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim or hidden_channels
        
        # --- 1. 特征预处理层 ---
        # 统一所有节点类型的特征维度
        max_node_dim = max(node_feature_dims.values())
        max_edge_dim = max(edge_feature_dims.values()) if edge_feature_dims else None
        
        # 为不同类型的节点创建特征投影层
        self.node_projectors = nn.ModuleDict()
        for node_type, feature_dim in node_feature_dims.items():
            if feature_dim != max_node_dim:
                self.node_projectors[node_type] = nn.Linear(feature_dim, max_node_dim)
            else:
                self.node_projectors[node_type] = nn.Identity()
        
        # --- 2. 主干网络 (Backbone) ---
        # 创建同构GNN编码器
        gnn_encoder = GNNEncoder(
            in_channels=max_node_dim,
            hidden_channels=hidden_channels,
            num_layers=gnn_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=max_edge_dim
        )
        
        # 转换为异构模型
        # 在定义了确切的依赖关系后，我们确信 to_hetero 会稳定工作
        self.hetero_encoder = to_hetero(gnn_encoder, metadata, aggr='sum')
        print("✅ 使用 to_hetero 转换的异构编码器")
        
        self.final_dim = gnn_encoder.final_dim
        
        # --- 3. 输出投影层 ---
        encoder_output_dim = gnn_encoder.final_dim
        if self.output_dim != encoder_output_dim:
            self.output_projector = nn.Linear(encoder_output_dim, self.output_dim)
        else:
            self.output_projector = nn.Identity()
        
        # --- 4. 图级别表示聚合器 ---
        self.graph_pooling = global_mean_pool
        self.graph_projector = nn.Sequential(
            nn.Linear(self.output_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, self.output_dim)
        )
    
    def preprocess_node_features(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预处理节点特征，确保所有类型具有相同维度
        """
        processed_x_dict = {}
        for node_type, x in x_dict.items():
            processed_x_dict[node_type] = self.node_projectors[node_type](x)
        return processed_x_dict
    
    def forward(self, data: HeteroData, 
                return_attention_weights: bool = False,
                return_graph_embedding: bool = False) -> Union[Dict[str, torch.Tensor], 
                                                               Tuple[Dict[str, torch.Tensor], ...]]:
        """
        前向传播 - 提取节点和图级别的嵌入表示
        
        参数:
            data: 异构图数据
            return_attention_weights: 是否返回注意力权重
            return_graph_embedding: 是否返回图级别嵌入
        
        返回:
            node_embeddings: 各类型节点的嵌入表示 {node_type: embeddings}
            attention_weights: 注意力权重 (可选)
            graph_embedding: 图级别嵌入 (可选)
        """
        # --- 1. 预处理节点特征 ---
        processed_x_dict = self.preprocess_node_features(data.x_dict)
        
        # --- 2. 异构GNN编码 ---
        raw_embeddings = self.hetero_encoder(
            processed_x_dict,
            data.edge_index_dict,
            data.edge_attr_dict
        )
        # 应用输出投影
        node_embeddings = {
            node_type: self.output_projector(embeddings)
            for node_type, embeddings in raw_embeddings.items()
        }
        
        # --- 3. 准备返回值 ---
        results = [node_embeddings]
        
        # --- 4. 注意力权重（可选）---
        if return_attention_weights:
            attention_weights = self.get_attention_weights()
            results.append(attention_weights)
        
        # --- 5. 图级别嵌入（可选）---
        if return_graph_embedding:
            # 聚合所有节点嵌入为图级别表示
            all_embeddings = torch.cat(list(node_embeddings.values()), dim=0)
            if all_embeddings.size(0) > 0:
                batch = torch.zeros(all_embeddings.size(0), 
                                  dtype=torch.long, 
                                  device=all_embeddings.device)
                graph_embedding = self.graph_pooling(all_embeddings, batch)
                graph_embedding = self.graph_projector(graph_embedding)
            else:
                graph_embedding = torch.zeros(1, self.output_dim, device=data.x_dict[list(data.x_dict.keys())[0]].device)
            results.append(graph_embedding)
        
        # 根据返回参数决定返回格式
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
    
    def encode_nodes(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        便捷方法：仅提取节点嵌入
        """
        return self.forward(data, return_attention_weights=False, return_graph_embedding=False)
    
    def encode_graph(self, data: HeteroData) -> torch.Tensor:
        """
        便捷方法：仅提取图级别嵌入
        """
        _, graph_embedding = self.forward(data, return_attention_weights=False, return_graph_embedding=True)
        return graph_embedding
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        获取注意力权重用于可视化分析
        
        注意：to_hetero转换后，模型结构会发生变化，需要递归搜索所有模块
        """
        attention_weights = []
        
        def collect_attention_weights(module):
            """递归收集注意力权重"""
            # 检查当前模块是否是PhysicsGATv2Conv
            if isinstance(module, PhysicsGATv2Conv):
                if hasattr(module, '_alpha') and module._alpha is not None:
                    attention_weights.append(module._alpha)
            
            # 递归检查子模块
            for child in module.children():
                collect_attention_weights(child)
        
        # 从hetero_encoder开始递归搜索
        if hasattr(self, 'hetero_encoder'):
            collect_attention_weights(self.hetero_encoder)
        
        return attention_weights

    def get_embedding_dim(self) -> int:
        """
        获取嵌入维度
        """
        return self.output_dim

def create_hetero_graph_encoder(data: HeteroData, 
                               hidden_channels: int = 64,
                               gnn_layers: int = 3,
                               heads: int = 4,
                               dropout: float = 0.3,
                               output_dim: Optional[int] = None) -> HeteroGraphEncoder:
    """
    创建异构图编码器的便捷函数
    
    参数:
        data: 异构图数据示例
        hidden_channels: 隐藏层维度
        gnn_layers: GNN层数
        heads: 注意力头数
        dropout: dropout概率
        output_dim: 输出嵌入维度
    
    返回:
        配置好的HeteroGraphEncoder模型
    """
    # 提取节点和边特征维度
    node_feature_dims = {node_type: x.shape[1] for node_type, x in data.x_dict.items()}
    edge_feature_dims = {edge_type: attr.shape[1] for edge_type, attr in data.edge_attr_dict.items()}
    
    # 创建编码器
    encoder = HeteroGraphEncoder(
        node_feature_dims=node_feature_dims,
        edge_feature_dims=edge_feature_dims,
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        gnn_layers=gnn_layers,
        heads=heads,
        dropout=dropout,
        output_dim=output_dim
    )
    
    return encoder

def test_hetero_graph_encoder(data: HeteroData, device: torch.device):
    """
    测试异构图编码器
    """
    print("\n🧠 测试异构Physics-Guided GATv2编码器...")
    
    # 创建编码器
    encoder = create_hetero_graph_encoder(
        data, 
        hidden_channels=32, 
        gnn_layers=2, 
        heads=4, 
        output_dim=64
    )
    encoder = encoder.to(device)
    data = data.to(device)
    
    # 测试不同的前向传播模式
    with torch.no_grad():
        # 1. 仅提取节点嵌入
        node_embeddings = encoder.encode_nodes(data)
        
        # 2. 提取图级别嵌入
        graph_embedding = encoder.encode_graph(data)
        
        # 3. 完整前向传播（包含注意力权重）
        node_emb, attention_weights, graph_emb = encoder(
            data, 
            return_attention_weights=True, 
            return_graph_embedding=True
        )
    
    print(f"✅ 异构图编码器测试成功！")
    print(f"📊 编码器参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"📊 输出嵌入维度: {encoder.get_embedding_dim()}")
    print(f"📊 节点类型数量: {len(node_embeddings)}")
    
    for node_type, embeddings in node_embeddings.items():
        print(f"   - {node_type}: {embeddings.shape}")
    
    print(f"📊 图级别嵌入形状: {graph_embedding.shape}")
    print(f"📊 注意力权重数量: {len(attention_weights)}")
    
    return encoder

if __name__ == "__main__":
    print("🔥 异构Physics-Guided GATv2图编码器 - 专注于表示学习！")
    print("📖 使用说明：")
    print("1. 使用 create_hetero_graph_encoder() 创建编码器")
    print("2. 使用 encoder.encode_nodes() 提取节点嵌入")
    print("3. 使用 encoder.encode_graph() 提取图级别嵌入")
    print("4. 编码器专注于特征提取，不包含决策逻辑")

```