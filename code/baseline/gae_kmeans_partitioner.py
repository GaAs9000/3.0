import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from ..src.rl.environment import PowerGridPartitioningEnv

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import negative_sampling
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class GraphAutoEncoder(nn.Module):
    """
    图自编码器模型
    
    使用图卷积网络学习节点的低维表示，然后重构图的邻接关系。
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32,
                 encoder_type: str = 'gcn', num_layers: int = 2, dropout: float = 0.1):
        super(GraphAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 构建编码器
        if encoder_type == 'gcn':
            self.encoder = self._build_gcn_encoder()
        elif encoder_type == 'gat':
            self.encoder = self._build_gat_encoder()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # 解码器（内积解码器）
        self.decoder = InnerProductDecoder()
    
    def _build_gcn_encoder(self) -> nn.ModuleList:
        """构建GCN编码器"""
        layers = nn.ModuleList()
        
        # 输入层
        layers.append(GCNConv(self.input_dim, self.hidden_dim))
        
        # 隐藏层
        for _ in range(self.num_layers - 2):
            layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        
        # 输出层
        layers.append(GCNConv(self.hidden_dim, self.embedding_dim))
        
        return layers
    
    def _build_gat_encoder(self) -> nn.ModuleList:
        """构建GAT编码器"""
        layers = nn.ModuleList()
        
        # 输入层
        layers.append(GATConv(self.input_dim, self.hidden_dim, heads=4, concat=False))
        
        # 隐藏层
        for _ in range(self.num_layers - 2):
            layers.append(GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False))
        
        # 输出层
        layers.append(GATConv(self.hidden_dim, self.embedding_dim, heads=1, concat=False))
        
        return layers
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """编码节点特征"""
        h = x
        
        for i, layer in enumerate(self.encoder):
            h = layer(h, edge_index)
            
            # 除了最后一层，都使用ReLU和Dropout
            if i < len(self.encoder) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """解码边概率"""
        return self.decoder(z, edge_index)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 编码
        z = self.encode(x, edge_index)
        
        # 解码
        edge_prob = self.decode(z, edge_index)
        
        return z, edge_prob


class InnerProductDecoder(nn.Module):
    """内积解码器"""
    
    def __init__(self):
        super(InnerProductDecoder, self).__init__()
    
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """计算边的重构概率"""
        # 获取边的端点
        row, col = edge_index
        
        # 计算内积
        inner_product = (z[row] * z[col]).sum(dim=1)
        
        # 应用sigmoid激活
        return torch.sigmoid(inner_product)


class GAEKMeansPartitioner(BasePartitioner):
    """
    图自编码器 + K-Means 分区方法
    
    两步骤方法：
    1. 使用图自编码器学习节点的低维表示
    2. 对学到的嵌入向量进行K-Means聚类
    """
    
    def __init__(self, seed: int = 42, hidden_dim: int = 64, embedding_dim: int = 32,
                 encoder_type: str = 'gcn', num_layers: int = 2, dropout: float = 0.1,
                 learning_rate: float = 0.01, epochs: int = 200, early_stopping: int = 20,
                 kmeans_init: str = 'k-means++', kmeans_n_init: int = 10):
        """
        初始化GAE+K-Means分区器
        
        Args:
            seed: 随机种子
            hidden_dim: 隐藏层维度
            embedding_dim: 嵌入维度
            encoder_type: 编码器类型 ('gcn', 'gat')
            num_layers: 编码器层数
            dropout: Dropout概率
            learning_rate: 学习率
            epochs: 训练轮数
            early_stopping: 早停轮数
            kmeans_init: K-Means初始化方法
            kmeans_n_init: K-Means重复次数
        """
        super().__init__(seed)
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric库未安装。请运行: pip install torch-geometric")
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        
        # 模型相关
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_trained = False
    
    def partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """执行GAE+K-Means分区"""
        # 设置随机种子
        self._set_seed()
        
        try:
            # 准备数据
            graph_data = self._prepare_graph_data(env)
            
            # 训练GAE（如果尚未训练）
            if not self.is_trained:
                self._train_gae(graph_data)
            
            # 生成节点嵌入
            embeddings = self._generate_embeddings(graph_data)
            
            # K-Means聚类
            labels = self._perform_kmeans(embeddings, env.num_partitions)
            
            return labels + 1  # 转换为1-based标签
            
        except Exception as e:
            print(f"❌ GAE+K-Means分区失败: {str(e)}")
            return self._fallback_partition(env)
    
    def _prepare_graph_data(self, env: 'PowerGridPartitioningEnv') -> Data:
        """准备图数据"""
        # 获取边信息
        edge_index = env.edge_info['edge_index'].cpu()
        
        # 获取节点特征
        if hasattr(env, 'node_features'):
            node_features = env.node_features.cpu()
        else:
            # 如果没有节点特征，使用简单的特征
            total_nodes = env.total_nodes
            node_features = torch.randn(total_nodes, 8)  # 8维随机特征
        
        # 创建PyG数据对象
        data = Data(x=node_features, edge_index=edge_index)
        
        return data
    
    def _train_gae(self, data: Data):
        """训练图自编码器"""
        print("🔄 开始训练图自编码器...")
        
        # 设置设备
        data = data.to(self.device)
        
        # 创建模型
        self.model = GraphAutoEncoder(
            input_dim=data.x.shape[1],
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            encoder_type=self.encoder_type,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练循环
        best_loss = float('inf')
        patience = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # 前向传播
            z, edge_prob = self.model(data.x, data.edge_index)
            
            # 计算损失
            loss = self._compute_gae_loss(edge_prob, data.edge_index, data.x.shape[0], data)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 早停检查
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping:
                    print(f"   早停于第{epoch+1}轮")
                    break
            
            # 打印进度
            if (epoch + 1) % 50 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print("✅ 图自编码器训练完成")
    
    def _compute_gae_loss(self, edge_prob: torch.Tensor, edge_index: torch.Tensor, 
                         num_nodes: int, data: Any) -> torch.Tensor:
        """计算GAE损失"""
        # 正样本损失
        pos_loss = -torch.log(edge_prob + 1e-15).mean()
        
        # 负样本
        neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, 
                                          num_neg_samples=edge_index.shape[1])
        
        # 计算负样本概率
        neg_edge_prob = self.model.decode(self.model.encode(data.x, edge_index), 
                                         neg_edge_index)
        
        # 负样本损失
        neg_loss = -torch.log(1 - neg_edge_prob + 1e-15).mean()
        
        return pos_loss + neg_loss
    
    def _generate_embeddings(self, data: Data) -> np.ndarray:
        """生成节点嵌入"""
        data = data.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(data.x, data.edge_index)
        
        return embeddings.cpu().numpy()
    
    def _perform_kmeans(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """执行K-Means聚类"""
        # 处理异常值
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # K-Means聚类
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=self.kmeans_init,
            n_init=self.kmeans_n_init,
            random_state=self.seed
        )
        
        labels = kmeans.fit_predict(embeddings)
        
        return labels
    
    def _fallback_partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """降级分区方法"""
        print("⚠️ 使用降级分区方法...")
        
        # 简单的轮询分配
        labels = np.zeros(env.total_nodes, dtype=int)
        for i in range(env.total_nodes):
            labels[i] = (i % env.num_partitions) + 1
        
        return labels
    
    def pretrain_on_multiple_networks(self, envs: List['PowerGridPartitioningEnv']):
        """在多个网络上预训练GAE"""
        print("🔄 开始在多个网络上预训练GAE...")
        
        # 准备批量数据
        batch_data = []
        for env in envs:
            data = self._prepare_graph_data(env)
            batch_data.append(data)
        
        # 创建批量数据
        batch = Batch.from_data_list(batch_data)
        
        # 训练
        self._train_gae(batch)
        
        print("✅ 多网络预训练完成")
    
    def get_embedding_quality(self, env: 'PowerGridPartitioningEnv') -> Dict[str, float]:
        """评估嵌入质量"""
        try:
            if not self.is_trained:
                return {'error': 'Model not trained'}
            
            # 生成嵌入
            data = self._prepare_graph_data(env)
            embeddings = self._generate_embeddings(data)
            
            # 计算嵌入质量指标
            # 1. 嵌入方差
            embedding_variance = np.var(embeddings, axis=0).mean()
            
            # 2. 嵌入范数
            embedding_norm = np.linalg.norm(embeddings, axis=1).mean()
            
            # 3. 嵌入分离度
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(embeddings)
            avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
            
            quality = {
                'embedding_variance': float(embedding_variance),
                'embedding_norm': float(embedding_norm),
                'avg_pairwise_distance': float(avg_distance),
                'embedding_dim': self.embedding_dim
            }
            
            return quality
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_model(self, filepath: str):
        """保存训练好的模型"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_dim': self.model.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'embedding_dim': self.embedding_dim,
                    'encoder_type': self.encoder_type,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout
                },
                'is_trained': self.is_trained
            }, filepath)
            print(f"✅ 模型已保存到: {filepath}")
        else:
            print("❌ 没有训练好的模型可以保存")
    
    def load_model(self, filepath: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 重建模型
            config = checkpoint['model_config']
            self.model = GraphAutoEncoder(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                embedding_dim=config['embedding_dim'],
                encoder_type=config['encoder_type'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(self.device)
            
            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = checkpoint['is_trained']
            
            print(f"✅ 模型已从 {filepath} 加载")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise