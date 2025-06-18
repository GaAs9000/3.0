import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax
from typing import Optional, List, Dict, Tuple
import numpy as np

class PhysicsGuidedGATConv(GATConv):
    """
    物理约束的图注意力卷积层
    
    核心创新：
    1. 在注意力机制中融入电气阻抗先验
    2. 阻抗越小的连接获得越高的注意力权重
    3. 可学习的温度参数控制物理先验的影响程度
    
    数学原理：
    标准GAT: α_ij = softmax(LeakyReLU(a^T[Wh_i || Wh_j]))
    物理GAT: α_ij = softmax(LeakyReLU(a^T[Wh_i || Wh_j]) + τ/|Z_ij|)
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 8,
                 concat: bool = True, dropout: float = 0.6, 
                 edge_dim: int = 8, physics_weight: float = 1.0, **kwargs):
        # 调用父类构造函数，启用边特征
        super().__init__(in_channels, out_channels, heads=heads, concat=concat,
                        dropout=dropout, edge_dim=edge_dim, **kwargs)
        
        # 物理引导参数
        self.physics_weight = physics_weight
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 可学习的温度参数
        
        # 边特征投影层（将边特征投影到注意力空间）
        self.edge_proj = nn.Linear(edge_dim, heads)
        
    def edge_update(self, alpha_j: torch.Tensor, alpha_i: Optional[torch.Tensor],
                   edge_attr: Optional[torch.Tensor], index: torch.Tensor,
                   ptr: Optional[torch.Tensor] = None,
                   size_i: Optional[int] = None) -> torch.Tensor:
        """
        边更新函数：融入物理先验
        
        在计算注意力分数时加入阻抗信息：
        - 使用边特征中的阻抗模长 |Z| (第3列)
        - 阻抗越小，导电性越好，注意力权重应该越大
        """
        # 计算基础注意力分数
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        
        # 融入物理先验
        if edge_attr is not None and edge_attr.shape[1] > 3:
            # 提取阻抗模长（第3列）
            z_magnitude = edge_attr[:, 3].clamp(min=1e-6)  # 避免除零
            
            # 计算物理权重：1/|Z| (导纳)
            physics_prior = 1.0 / z_magnitude
            
            # 投影边特征到多头空间
            edge_weights = self.edge_proj(edge_attr)  # [E, heads]
            
            # 将物理先验加入注意力分数
            # 使用温度参数控制物理先验的影响
            physics_contribution = self.temperature * self.physics_weight * physics_prior.unsqueeze(-1)
            
            # 融合原始注意力和物理先验
            alpha = alpha + physics_contribution + edge_weights
        
        # 应用softmax归一化
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return alpha


class EnhancedGATEncoder(nn.Module):
    """
    增强的GAT编码器
    
    特点：
    1. 多层Physics-Guided GAT
    2. 残差连接和层归一化
    3. 自适应dropout
    4. 位置编码
    5. 多尺度特征融合
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, heads: int = 8, dropout: float = 0.3,
                 edge_dim: int = 8, add_positional_encoding: bool = True,
                 physics_weight: float = 1.0):
        super().__init__()
        
        self.num_layers = num_layers
        self.add_positional_encoding = add_positional_encoding
        
        # 位置编码（基于拉普拉斯特征向量）
        if add_positional_encoding:
            self.pos_encoding_dim = 16
            self.pos_proj = nn.Linear(self.pos_encoding_dim, in_channels)
        
        # GAT层列表
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # 构建多层网络
        current_dim = in_channels
        
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            
            # 输出维度
            if is_last_layer:
                out_dim = out_channels
                concat = False
            else:
                out_dim = hidden_channels
                concat = True
            
            # Physics-Guided GAT层
            gat = PhysicsGuidedGATConv(
                current_dim, out_dim, heads=heads,
                concat=concat, dropout=dropout,
                edge_dim=edge_dim, physics_weight=physics_weight,
                add_self_loops=False  # 避免重复添加自环
            )
            self.gat_layers.append(gat)
            
            # 计算实际输出维度
            actual_out_dim = out_dim * heads if concat else out_dim
            
            # 层归一化
            self.layer_norms.append(nn.LayerNorm(actual_out_dim))
            
            # 残差投影
            if current_dim != actual_out_dim:
                self.residual_projs.append(nn.Linear(current_dim, actual_out_dim))
            else:
                self.residual_projs.append(nn.Identity())
            
            # 自适应dropout
            self.dropout_layers.append(nn.Dropout(dropout * (1 - i / num_layers)))
            
            current_dim = actual_out_dim
        
        # 输出投影层
        self.output_proj = nn.Linear(current_dim, out_channels)
        
    def compute_positional_encoding(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        计算基于图拉普拉斯的位置编码
        
        使用归一化拉普拉斯矩阵的特征向量作为位置编码
        这能捕捉图的全局结构信息
        """
        device = edge_index.device
        
        # 构建邻接矩阵
        row, col = edge_index
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        adj[row, col] = 1.0
        adj = adj + adj.T  # 对称化
        
        # 度矩阵
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 归一化拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
        norm_adj = deg_inv_sqrt.unsqueeze(0) * adj * deg_inv_sqrt.unsqueeze(1)
        laplacian = torch.eye(num_nodes, device=device) - norm_adj
        
        # 特征分解（只取前k个最小特征值对应的特征向量）
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
            # 取前pos_encoding_dim个特征向量（跳过常数特征向量）
            pos_encoding = eigenvectors[:, 1:self.pos_encoding_dim+1]
            
            # 如果维度不够，填充零
            if pos_encoding.shape[1] < self.pos_encoding_dim:
                padding = torch.zeros(
                    num_nodes, 
                    self.pos_encoding_dim - pos_encoding.shape[1],
                    device=device
                )
                pos_encoding = torch.cat([pos_encoding, padding], dim=1)
        except:
            # 如果特征分解失败，使用随机初始化
            pos_encoding = torch.randn(num_nodes, self.pos_encoding_dim, device=device) * 0.1
        
        return pos_encoding
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
               edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, edge_dim]
            
        返回:
            节点嵌入 [N, out_channels]
        """
        # 添加位置编码
        if self.add_positional_encoding:
            pos_encoding = self.compute_positional_encoding(edge_index, x.size(0))
            pos_features = self.pos_proj(pos_encoding)
            x = x + pos_features
        
        # 多层GAT处理
        for i in range(self.num_layers):
            # 保存输入用于残差连接
            residual = x
            
            # GAT卷积
            x = self.gat_layers[i](x, edge_index, edge_attr)
            
            # 层归一化
            x = self.layer_norms[i](x)
            
            # 残差连接
            x = x + self.residual_projs[i](residual)
            
            # 激活和dropout（最后一层除外）
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = self.dropout_layers[i](x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        获取各层的注意力权重（用于可视化）
        """
        attention_weights = []
        for gat_layer in self.gat_layers:
            if hasattr(gat_layer, '_alpha'):
                attention_weights.append(gat_layer._alpha)
        return attention_weights


# Test function can be called from main if needed
def initialize_gat_encoder(data, device):
    """Test function for GAT encoder"""
    print("\n🧠 测试Physics-Guided GAT编码器...")
    encoder = EnhancedGATEncoder(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        out_channels=64,
        num_layers=3,
        heads=8,
        edge_dim=data.edge_attr.shape[1]
    ).to(device)

    # 将数据移到设备
    data = data.to(device)

    # 前向传播测试
    with torch.no_grad():
        embeddings = encoder(data.x, data.edge_index, data.edge_attr)
        
    print(f"✅ GAT编码器测试成功！")
    print(f"📊 输入维度: {data.x.shape}")
    print(f"📊 输出嵌入维度: {embeddings.shape}")
    print(f"📊 模型参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    
    return encoder, embeddings

