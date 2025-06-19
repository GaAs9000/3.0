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

