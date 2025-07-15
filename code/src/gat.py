import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, to_hetero, LayerNorm, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
from typing import Optional, List, Dict, Tuple, Union, Any
import numpy as np
import warnings

# 存储捕获到的注意力权重的全局列表
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
        # 为所有节点类型创建投影层，统一投影到hidden_channels维度
        max_edge_dim = max(edge_feature_dims.values()) if edge_feature_dims else None

        # 为所有节点类型创建特征投影层（投影到hidden_channels）
        self.node_projectors = nn.ModuleDict()
        for node_type, feature_dim in node_feature_dims.items():
            # 所有节点类型都需要投影层，统一投影到hidden_channels维度
            self.node_projectors[node_type] = nn.Linear(feature_dim, hidden_channels)

        # --- 2. 主干网络 (Backbone) ---
        # 创建同构GNN编码器，输入维度为hidden_channels
        gnn_encoder = GNNEncoder(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=gnn_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=max_edge_dim
        )
        
        # 转换为异构模型
        # 在定义了确切的依赖关系后，我们确信 to_hetero 会稳定工作
        self.hetero_encoder = to_hetero(gnn_encoder, metadata, aggr='sum')
        from utils_common import safe_rich_debug
        safe_rich_debug("使用 to_hetero 转换的异构编码器", "attention")
        
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

        将所有节点类型的特征统一投影到hidden_channels维度

        Args:
            x_dict: 节点特征字典 {node_type: features}

        Returns:
            processed_x_dict: 投影后的特征字典，所有特征维度为hidden_channels

        Raises:
            KeyError: 如果节点类型没有对应的投影层
            RuntimeError: 如果特征维度不匹配
        """
        processed_x_dict = {}

        for node_type, x in x_dict.items():
            # 验证节点类型是否有对应的投影层
            if node_type not in self.node_projectors:
                raise KeyError(f"节点类型 '{node_type}' 没有对应的投影层。"
                             f"可用的节点类型: {list(self.node_projectors.keys())}")

            # 验证输入特征维度
            projector = self.node_projectors[node_type]
            expected_input_dim = projector.in_features if hasattr(projector, 'in_features') else None

            if expected_input_dim is not None and x.shape[-1] != expected_input_dim:
                raise RuntimeError(f"节点类型 '{node_type}' 的特征维度不匹配。"
                                 f"期望: {expected_input_dim}, 实际: {x.shape[-1]}")

            # 应用投影层
            try:
                projected_x = projector(x)
                processed_x_dict[node_type] = projected_x

                # 验证输出维度
                if projected_x.shape[-1] != self.hidden_channels:
                    raise RuntimeError(f"投影后的特征维度不正确。"
                                     f"期望: {self.hidden_channels}, 实际: {projected_x.shape[-1]}")

            except Exception as e:
                raise RuntimeError(f"节点类型 '{node_type}' 的特征投影失败: {str(e)}")

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

    def encode_nodes_with_attention(self, data: HeteroData, config: Dict[str, Any] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        便捷方法：提取节点嵌入和注意力权重

        返回:
            node_embeddings: 节点嵌入字典
            attention_weights: 注意力权重字典
        """
        node_embeddings = self.forward(data, return_attention_weights=False, return_graph_embedding=False)
        attention_weights = self.get_attention_weights(config)
        return node_embeddings, attention_weights
    
    def encode_graph(self, data: HeteroData) -> torch.Tensor:
        """
        便捷方法：仅提取图级别嵌入
        """
        _, graph_embedding = self.forward(data, return_attention_weights=False, return_graph_embedding=True)
        return graph_embedding
    
    def get_attention_weights(self, config: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        """
        获取注意力权重用于增强嵌入生成

        返回:
            attention_weights: 按边类型组织的注意力权重字典
                格式: {edge_type_key: attention_tensor}
                其中 edge_type_key = f"{src_type}__{relation}__{dst_type}"
        """
        # 获取调试配置
        debug_config = config.get('debug', {}) if config else {}
        training_output = debug_config.get('training_output', {})
        show_attention_collection = training_output.get('show_attention_collection', True)
        only_show_errors = training_output.get('only_show_errors', False)

        attention_weights = {}

        if show_attention_collection and not only_show_errors:
            from utils_common import safe_rich_debug
            safe_rich_debug("开始收集注意力权重...", "attention")

        def collect_attention_weights_with_names(module, prefix=""):
            """递归收集注意力权重并记录对应的边类型"""
            # 检查当前模块是否是PhysicsGATv2Conv
            if isinstance(module, PhysicsGATv2Conv):
                if hasattr(module, '_alpha') and module._alpha is not None:
                    if show_attention_collection and not only_show_errors:
                        print(f"  发现PhysicsGATv2Conv模块: {prefix}")
                        print(f"     注意力权重形状: {module._alpha.shape}")

                    # 从模块名称推断边类型
                    edge_type_key = self._extract_edge_type_from_module_name(prefix)
                    if edge_type_key:
                        attention_weights[edge_type_key] = module._alpha
                        if show_attention_collection and not only_show_errors:
                            print(f"     映射到边类型: {edge_type_key}")
                    else:
                        if show_attention_collection and not only_show_errors:
                            print(f"     WARNING: 无法映射边类型，模块名: {prefix}")
                        # 使用模块名作为备用键
                        fallback_key = f"module_{prefix.replace('.', '_')}"
                        attention_weights[fallback_key] = module._alpha
                        if show_attention_collection and not only_show_errors:
                            print(f"     使用备用键: {fallback_key}")
                elif show_attention_collection and not only_show_errors:
                    print(f"  发现PhysicsGATv2Conv模块但无注意力权重: {prefix}")

            # 递归检查子模块
            for name, child in module.named_children():
                new_prefix = f"{prefix}.{name}" if prefix else name
                collect_attention_weights_with_names(child, new_prefix)

        # 从hetero_encoder开始递归搜索
        if hasattr(self, 'hetero_encoder'):
            collect_attention_weights_with_names(self.hetero_encoder)

        if show_attention_collection and not only_show_errors:
            print(f"收集到 {len(attention_weights)} 个注意力权重:")
            for key, tensor in attention_weights.items():
                print(f"   - {key}: {tensor.shape}")

        return attention_weights

    def _extract_edge_type_from_module_name(self, module_name: str) -> Optional[str]:
        """
        从模块名称中提取边类型信息

        to_hetero转换后的模块名称格式通常为:
        convs.{layer_idx}.{edge_type_encoded}

        参数:
            module_name: 模块的完整名称路径

        返回:
            edge_type_key: 格式化的边类型键，如 "bus_pv__connects_line__bus_slack"
        """
        from utils_common import safe_rich_debug
        safe_rich_debug(f"尝试从模块名提取边类型: {module_name}", "attention")

        # 解析模块名称以提取边类型信息
        # to_hetero会将边类型编码到模块名中

        # 方法1: 尝试从已知的边类型中匹配
        for edge_type in self.edge_types:
            src_type, relation, dst_type = edge_type
            edge_type_key = f"{src_type}__{relation}__{dst_type}"

            # 检查模块名称是否包含边类型信息
            if (src_type in module_name and dst_type in module_name and
                relation in module_name):
                from utils_common import safe_rich_debug
                safe_rich_debug(f"匹配到边类型: {edge_type_key}", "attention")
                return edge_type_key

        # 方法2: 尝试解析to_hetero的编码格式
        # to_hetero通常使用数字索引来编码边类型
        import re

        # 查找类似 "convs.0.1" 的模式，其中最后的数字可能是边类型索引
        pattern = r'convs\.(\d+)\.(\d+)'
        match = re.search(pattern, module_name)

        if match:
            layer_idx = int(match.group(1))
            edge_type_idx = int(match.group(2))

            # 尝试根据索引映射到边类型
            if edge_type_idx < len(self.edge_types):
                edge_type = self.edge_types[edge_type_idx]
                src_type, relation, dst_type = edge_type
                edge_type_key = f"{src_type}__{relation}__{dst_type}"
                from utils_common import safe_rich_debug
                safe_rich_debug(f"通过索引匹配到边类型: {edge_type_key} (索引: {edge_type_idx})", "attention")
                return edge_type_key

        # 方法3: 如果无法匹配，返回通用键
        if "conv" in module_name:
            fallback_key = f"unknown_edge_type_{module_name.replace('.', '_')}"
            from utils_common import safe_rich_debug
            safe_rich_debug(f"使用备用键: {fallback_key}", "attention")
            return fallback_key

        from utils_common import safe_rich_debug
        safe_rich_debug("无法提取边类型", "attention")
        return None



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


def validate_encoder_architecture(data: HeteroData,
                                 hidden_channels: int = 64,
                                 gnn_layers: int = 3,
                                 heads: int = 8,
                                 output_dim: int = 128,
                                 device: torch.device = None) -> 'HeteroGraphEncoder':
    """
    验证并创建生产级异构图编码器

    Args:
        data: 异构图数据
        hidden_channels: 隐藏层维度
        gnn_layers: GNN层数
        heads: 注意力头数
        output_dim: 输出嵌入维度
        device: 计算设备

    Returns:
        验证通过的编码器实例

    Raises:
        ValueError: 如果架构验证失败
        RuntimeError: 如果前向传播失败
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 验证输入数据格式
    if not isinstance(data, HeteroData):
        raise ValueError("输入数据必须是HeteroData格式")

    # 验证必要的节点类型和边类型
    required_node_types = ['bus']
    required_edge_types = [('bus', 'connects', 'bus')]

    for node_type in required_node_types:
        if node_type not in data.node_types:
            raise ValueError(f"缺少必要的节点类型: {node_type}")

    for edge_type in required_edge_types:
        if edge_type not in data.edge_types:
            raise ValueError(f"缺少必要的边类型: {edge_type}")

    # 创建编码器
    try:
        encoder = create_hetero_graph_encoder(
            data,
            hidden_channels=hidden_channels,
            gnn_layers=gnn_layers,
            heads=heads,
            output_dim=output_dim
        )
        encoder = encoder.to(device)
        data = data.to(device)
    except Exception as e:
        raise RuntimeError(f"编码器创建失败: {str(e)}")

    # 验证前向传播
    try:
        with torch.no_grad():
            # 验证节点嵌入提取
            node_embeddings = encoder.encode_nodes(data)

            # 验证图级别嵌入
            graph_embedding = encoder.encode_graph(data)

            # 验证完整前向传播
            node_emb, attention_weights, graph_emb = encoder(
                data,
                return_attention_weights=True,
                return_graph_embedding=True
            )

            # 验证输出维度
            expected_dim = encoder.get_embedding_dim()
            for node_type, embeddings in node_embeddings.items():
                if embeddings.shape[-1] != expected_dim:
                    raise ValueError(f"节点类型 {node_type} 的嵌入维度不匹配: "
                                   f"期望 {expected_dim}, 实际 {embeddings.shape[-1]}")

            if graph_embedding.shape[-1] != expected_dim:
                raise ValueError(f"图级别嵌入维度不匹配: "
                               f"期望 {expected_dim}, 实际 {graph_embedding.shape[-1]}")

    except Exception as e:
        raise RuntimeError(f"前向传播验证失败: {str(e)}")

    return encoder


def create_production_encoder(data: HeteroData, config: Dict[str, Any] = None) -> 'HeteroGraphEncoder':
    """
    创建生产级异构图编码器

    Args:
        data: 异构图数据
        config: 编码器配置参数

    Returns:
        配置好的编码器实例
    """
    if config is None:
        config = {
            'hidden_channels': 64,
            'gnn_layers': 3,
            'heads': 8,
            'output_dim': 128,
            'dropout': 0.1,
            'physics_enhanced': True,
            'temperature': 1.0,
            'physics_weight': 1.0
        }

    return validate_encoder_architecture(
        data,
        hidden_channels=config.get('hidden_channels', 64),
        gnn_layers=config.get('gnn_layers', 3),
        heads=config.get('heads', 8),
        output_dim=config.get('output_dim', 128)
    )

