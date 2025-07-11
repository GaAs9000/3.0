import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any, Callable
from .baseline import BasePartitioner, set_baseline_seed

if TYPE_CHECKING:
    from ..src.rl.environment import PowerGridPartitioningEnv

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class SupervisedGATEncoder(nn.Module):
    """
    有监督图注意力网络编码器
    
    通过预测与实时负荷相关的代理任务（如总负荷、节点电压等）来学习有意义的节点表示。
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 prediction_task: str = 'total_load', output_dim: int = 1):
        super(SupervisedGATEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_task = prediction_task
        self.output_dim = output_dim
        
        # 构建GAT层
        self.gat_layers = nn.ModuleList()
        
        # 第一层
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=num_heads, 
                                      concat=True, dropout=dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                          heads=num_heads, concat=True, dropout=dropout))
        
        # 最后一层 - 产生嵌入
        self.gat_layers.append(GATConv(hidden_dim * num_heads, embedding_dim, 
                                      heads=1, concat=False, dropout=dropout))
        
        # 预测头
        self.prediction_head = self._build_prediction_head()
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers - 1)
        ])
    
    def _build_prediction_head(self) -> nn.Module:
        """构建预测头"""
        if self.prediction_task == 'total_load':
            # 预测总负荷
            return nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        elif self.prediction_task == 'node_voltage':
            # 预测节点电压
            return nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        elif self.prediction_task == 'load_distribution':
            # 预测负荷分布
            return nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        else:
            raise ValueError(f"Unknown prediction task: {self.prediction_task}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 通过GAT层
        h = x
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(h, edge_index)
            
            # 除了最后一层，都使用批归一化和激活
            if i < len(self.gat_layers) - 1:
                h = self.batch_norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 节点嵌入
        node_embeddings = h
        
        # 图级别的预测
        if batch is not None:
            # 批处理模式
            graph_embedding = global_mean_pool(node_embeddings, batch)
        else:
            # 单图模式
            graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
        
        # 预测
        predictions = self.prediction_head(graph_embedding)
        
        return node_embeddings, predictions
    
    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """只获取节点嵌入（用于聚类）"""
        node_embeddings, _ = self.forward(x, edge_index)
        return node_embeddings


class SupervisedEmbeddingKMeansPartitioner(BasePartitioner):
    """
    有监督嵌入 + K-Means 分区方法
    
    两步骤方法：
    1. 通过有监督学习训练GAT来预测与负荷相关的代理任务
    2. 使用训练好的GAT为当前状态生成动态嵌入，然后进行K-Means聚类
    """
    
    def __init__(self, seed: int = 42, hidden_dim: int = 64, embedding_dim: int = 32,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 prediction_task: str = 'total_load', learning_rate: float = 0.01,
                 epochs: int = 200, early_stopping: int = 20, batch_size: int = 32,
                 kmeans_init: str = 'k-means++', kmeans_n_init: int = 10):
        """
        初始化有监督嵌入+K-Means分区器
        
        Args:
            seed: 随机种子
            hidden_dim: 隐藏层维度
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: GAT层数
            dropout: Dropout概率
            prediction_task: 代理任务类型
            learning_rate: 学习率
            epochs: 训练轮数
            early_stopping: 早停轮数
            batch_size: 批大小
            kmeans_init: K-Means初始化方法
            kmeans_n_init: K-Means重复次数
        """
        super().__init__(seed)
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric库未安装。请运行: pip install torch-geometric")
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_task = prediction_task
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        
        # 模型相关
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 训练数据
        self.training_data = []
        self.training_labels = []
    
    def partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """执行有监督嵌入+K-Means分区"""
        # 设置随机种子
        self._set_seed()
        
        try:
            # 如果模型未训练，尝试自训练
            if not self.is_trained:
                print("🔄 模型未训练，尝试自训练...")
                success = self._self_train(env)
                if not success:
                    print("⚠️ 自训练失败，使用随机初始化进行分区")
                    return self._fallback_partition(env)
            
            # 准备当前状态数据
            current_data = self._prepare_graph_data(env)
            
            # 生成动态嵌入
            embeddings = self._generate_dynamic_embeddings(current_data)
            
            # K-Means聚类
            labels = self._perform_kmeans(embeddings, env.num_partitions)
            
            return labels + 1  # 转换为1-based标签
            
        except Exception as e:
            print(f"❌ 有监督嵌入+K-Means分区失败: {str(e)}")
            return self._fallback_partition(env)
    
    def _prepare_graph_data(self, env: 'PowerGridPartitioningEnv') -> Data:
        """准备图数据"""
        # 获取边信息
        edge_index = env.edge_info['edge_index'].cpu()
        
        # 获取动态节点特征（包含当前负荷状态）
        node_features = self._extract_dynamic_node_features(env)
        
        # 创建PyG数据对象
        data = Data(x=node_features, edge_index=edge_index)
        
        return data
    
    def _extract_dynamic_node_features(self, env: 'PowerGridPartitioningEnv') -> torch.Tensor:
        """提取动态节点特征"""
        try:
            # 基础特征
            if hasattr(env, 'node_features'):
                base_features = env.node_features.cpu()
            else:
                total_nodes = env.total_nodes
                base_features = torch.randn(total_nodes, 4)
            
            # 动态特征（负荷状态）
            if hasattr(env, 'current_load_state'):
                load_features = env.current_load_state.cpu()
            else:
                # 模拟动态负荷特征
                total_nodes = env.total_nodes
                load_features = torch.randn(total_nodes, 4)
            
            # 时间特征
            if hasattr(env, 'time_features'):
                time_features = env.time_features.cpu()
            else:
                # 模拟时间特征（小时、星期几等）
                total_nodes = env.total_nodes
                time_features = torch.randn(total_nodes, 2)
            
            # 合并特征
            dynamic_features = torch.cat([base_features, load_features, time_features], dim=1)
            
            return dynamic_features
            
        except Exception as e:
            print(f"⚠️ 动态特征提取失败: {e}")
            # 返回随机特征
            total_nodes = env.total_nodes
            return torch.randn(total_nodes, 10)
    
    def train_on_scenarios(self, training_scenarios: List[Dict[str, Any]]):
        """在多个场景上训练有监督模型"""
        print("🔄 开始有监督训练...")
        
        # 准备训练数据
        self._prepare_training_data(training_scenarios)
        
        if len(self.training_data) == 0:
            print("❌ 没有可用的训练数据")
            return
        
        # 创建数据加载器
        train_loader = self._create_data_loader()
        
        # 确定输入维度和输出维度
        sample_data = self.training_data[0]
        input_dim = sample_data.x.shape[1]
        output_dim = self._get_output_dimension()
        
        # 创建模型
        self.model = SupervisedGATEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            prediction_task=self.prediction_task,
            output_dim=output_dim
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = self._get_loss_function()
        
        # 训练循环
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.model.train()
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                node_embeddings, predictions = self.model(batch.x, batch.edge_index, batch.batch)
                
                # 计算损失
                loss = criterion(predictions, batch.y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping:
                    print(f"   早停于第{epoch+1}轮")
                    break
            
            # 打印进度
            if (epoch + 1) % 50 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        print("✅ 有监督训练完成")
    
    def _prepare_training_data(self, training_scenarios: List[Dict[str, Any]]):
        """准备训练数据"""
        self.training_data = []
        self.training_labels = []
        
        for scenario in training_scenarios:
            try:
                # 提取场景数据
                env = scenario['env']
                target_value = scenario['target']
                
                # 创建图数据
                data = self._prepare_graph_data(env)
                
                # 添加目标值
                data.y = torch.tensor([target_value], dtype=torch.float32)
                
                self.training_data.append(data)
                self.training_labels.append(target_value)
                
            except Exception as e:
                print(f"⚠️ 场景数据准备失败: {e}")
                continue
        
        # 标准化标签
        if len(self.training_labels) > 0:
            labels_array = np.array(self.training_labels).reshape(-1, 1)
            self.scaler.fit(labels_array)
            
            # 更新数据中的标签
            for i, data in enumerate(self.training_data):
                normalized_label = self.scaler.transform([[self.training_labels[i]]])
                data.y = torch.tensor(normalized_label[0], dtype=torch.float32)
    
    def _create_data_loader(self):
        """创建数据加载器"""
        from torch_geometric.loader import DataLoader
        
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)
    
    def _get_output_dimension(self) -> int:
        """获取输出维度"""
        if self.prediction_task == 'total_load':
            return 1
        elif self.prediction_task == 'node_voltage':
            return 1
        elif self.prediction_task == 'load_distribution':
            return 3  # 例如：峰值、均值、标准差
        else:
            return 1
    
    def _get_loss_function(self) -> Callable:
        """获取损失函数"""
        if self.prediction_task in ['total_load', 'node_voltage']:
            return nn.MSELoss()
        elif self.prediction_task == 'load_distribution':
            return nn.MSELoss()
        else:
            return nn.MSELoss()
    
    def _generate_dynamic_embeddings(self, data: Data) -> np.ndarray:
        """生成动态节点嵌入"""
        data = data.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_node_embeddings(data.x, data.edge_index)
        
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
    
    def _self_train(self, env: 'PowerGridPartitioningEnv') -> bool:
        """自训练方法，使用合成数据训练模型"""
        try:
            print("🔄 开始自训练...")
            
            # 生成合成训练数据
            train_data, train_targets = self._generate_synthetic_training_data(env)
            
            if len(train_data) == 0:
                print("❌ 无法生成合成训练数据")
                return False
            
            # 初始化模型
            input_dim = train_data[0].x.shape[1]
            self.model = SupervisedGATEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout,
                prediction_task=self.prediction_task,
                output_dim=1
            ).to(self.device)
            
            # 训练模型
            success = self._train_model(train_data, train_targets)
            
            if success:
                self.is_trained = True
                print("✅ 自训练完成")
                return True
            else:
                print("❌ 自训练失败")
                return False
                
        except Exception as e:
            print(f"❌ 自训练过程失败: {str(e)}")
            return False
    
    def _generate_synthetic_training_data(self, env: 'PowerGridPartitioningEnv') -> Tuple[List[Data], List[float]]:
        """生成合成训练数据"""
        train_data = []
        train_targets = []
        
        try:
            # 生成多个不同的负荷状态
            num_samples = 100
            
            for i in range(num_samples):
                # 随机扰动负荷
                data = self._prepare_graph_data(env)
                
                # 添加随机扰动到节点特征
                noise = torch.randn_like(data.x) * 0.1
                data.x = data.x + noise
                
                # 根据预测任务生成目标
                if self.prediction_task == 'total_load':
                    # 总负荷 = 节点特征的和
                    target = float(data.x.sum())
                elif self.prediction_task == 'node_voltage':
                    # 平均电压（模拟）
                    target = float(data.x.mean())
                else:
                    # 默认目标
                    target = float(data.x.norm())
                
                train_data.append(data)
                train_targets.append(target)
            
            print(f"📊 生成了 {len(train_data)} 个训练样本")
            return train_data, train_targets
            
        except Exception as e:
            print(f"❌ 生成合成数据失败: {str(e)}")
            return [], []
    
    def _train_model(self, train_data: List[Data], train_targets: List[float]) -> bool:
        """训练模型"""
        try:
            # 准备训练数据
            train_targets = torch.tensor(train_targets, dtype=torch.float32).to(self.device)
            
            # 标准化目标
            train_targets = (train_targets - train_targets.mean()) / (train_targets.std() + 1e-8)
            
            # 优化器和损失函数
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # 训练循环
            self.model.train()
            best_loss = float('inf')
            patience = 0
            
            # 减少训练轮数以加快速度
            max_epochs = min(self.epochs, 50)
            
            for epoch in range(max_epochs):
                total_loss = 0.0
                
                # 随机打乱数据
                indices = torch.randperm(len(train_data))
                
                for i in indices:
                    data = train_data[i].to(self.device)
                    target = train_targets[i:i+1]
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    _, pred = self.model(data.x, data.edge_index)
                    
                    # 计算损失
                    loss = criterion(pred, target)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_data)
                
                # 早停检查
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stopping:
                        print(f"   早停于第{epoch+1}轮")
                        break
                
                if epoch % 10 == 0:
                    print(f"   轮次 {epoch+1}/{max_epochs}, 损失: {avg_loss:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型训练失败: {str(e)}")
            return False
    
    def _fallback_partition(self, env: 'PowerGridPartitioningEnv') -> np.ndarray:
        """降级分区方法"""
        print("⚠️ 使用降级分区方法...")
        
        # 简单的轮询分配
        labels = np.zeros(env.total_nodes, dtype=int)
        for i in range(env.total_nodes):
            labels[i] = (i % env.num_partitions) + 1
        
        return labels
    
    def predict_target(self, env: 'PowerGridPartitioningEnv') -> float:
        """预测目标值（用于验证模型性能）"""
        try:
            if not self.is_trained:
                return 0.0
            
            data = self._prepare_graph_data(env)
            data = data.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                _, predictions = self.model(data.x, data.edge_index)
            
            # 反标准化
            pred_value = predictions.cpu().numpy()[0]
            if hasattr(self.scaler, 'inverse_transform'):
                pred_value = self.scaler.inverse_transform([pred_value])[0]
            
            return float(pred_value)
            
        except Exception as e:
            print(f"⚠️ 预测失败: {e}")
            return 0.0
    
    def evaluate_prediction_accuracy(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估预测准确性"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            predictions = []
            targets = []
            
            for scenario in test_scenarios:
                env = scenario['env']
                target = scenario['target']
                
                pred = self.predict_target(env)
                predictions.append(pred)
                targets.append(target)
            
            # 计算指标
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            # 相关系数
            correlation = np.corrcoef(predictions, targets)[0, 1]
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'correlation': float(correlation),
                'num_samples': len(predictions)
            }
            
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
                    'num_heads': self.num_heads,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout,
                    'prediction_task': self.prediction_task,
                    'output_dim': self.model.output_dim
                },
                'scaler': self.scaler,
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
            self.model = SupervisedGATEncoder(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                embedding_dim=config['embedding_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                prediction_task=config['prediction_task'],
                output_dim=config['output_dim']
            ).to(self.device)
            
            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.is_trained = checkpoint['is_trained']
            
            print(f"✅ 模型已从 {filepath} 加载")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise