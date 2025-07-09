#!/usr/bin/env python3
"""
电力网络分区智能体综合评估系统

集成所有评估功能：
- 区域内平衡指标计算
- 多种baseline方法对比
- 跨场景测试和泛化评估
- 自动化报告生成
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from sklearn.cluster import SpectralClustering
import sys

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))

from train import UnifiedTrainingSystem, load_power_grid_data
from data_processing import PowerGridDataProcessor
from gat import create_hetero_graph_encoder
from rl.environment import PowerGridPartitioningEnv
from rl.agent import PPOAgent


class BaselinePartitioner:
    """Baseline分区方法集合"""
    
    @staticmethod
    def random_partition(adjacency_matrix: np.ndarray, num_partitions: int, seed: int = 42) -> np.ndarray:
        """随机分区方法"""
        np.random.seed(seed)
        num_nodes = adjacency_matrix.shape[0]
        partition = np.random.randint(1, num_partitions + 1, size=num_nodes)
        return partition
    
    @staticmethod
    def degree_based_partition(adjacency_matrix: np.ndarray, num_partitions: int) -> np.ndarray:
        """基于度中心性的分区方法"""
        # 计算节点度数
        degrees = np.sum(adjacency_matrix, axis=1)
        
        # 按度数排序
        sorted_indices = np.argsort(degrees)[::-1]  # 降序
        
        # 轮流分配到各分区
        partition = np.zeros(len(degrees), dtype=int)
        for i, node_idx in enumerate(sorted_indices):
            partition[node_idx] = (i % num_partitions) + 1
            
        return partition
    
    @staticmethod
    def spectral_partition(adjacency_matrix: np.ndarray, num_partitions: int, random_state: int = 42) -> np.ndarray:
        """谱聚类分区方法"""
        try:
            # 确保邻接矩阵是对称的
            adj_symmetric = (adjacency_matrix + adjacency_matrix.T) / 2
            
            # 谱聚类
            spectral = SpectralClustering(
                n_clusters=num_partitions,
                affinity='precomputed',
                random_state=random_state,
                assign_labels='discretize'
            )
            
            labels = spectral.fit_predict(adj_symmetric)
            # 转换为1-based索引
            partition = labels + 1
            
            return partition
            
        except Exception as e:
            print(f"⚠️ 谱聚类失败，使用随机分区: {e}")
            return BaselinePartitioner.random_partition(adjacency_matrix, num_partitions, random_state)


class ScenarioGenerator:
    """测试场景生成器"""
    
    @staticmethod
    def generate_test_scenarios(original_mpc: Dict, network_name: str) -> Dict[str, Dict]:
        """生成4种测试场景"""
        scenarios = {}
        
        # 1. 正常场景
        scenarios['normal'] = original_mpc.copy()
        
        # 2. 高负载场景 (所有负载×1.5)
        high_load_mpc = original_mpc.copy()
        high_load_mpc['bus'] = high_load_mpc['bus'].copy()
        high_load_mpc['bus'][:, [2, 3]] *= 1.5  # PD, QD列
        scenarios['high_load'] = high_load_mpc
        
        # 3. 不平衡场景 (随机30%节点负载×2，其他×0.8)
        unbalanced_mpc = original_mpc.copy()
        unbalanced_mpc['bus'] = unbalanced_mpc['bus'].copy()
        num_nodes = unbalanced_mpc['bus'].shape[0]
        np.random.seed(42)  # 确保可重复
        high_load_nodes = np.random.choice(num_nodes, size=int(0.3 * num_nodes), replace=False)
        
        # 高负载节点
        for node in high_load_nodes:
            unbalanced_mpc['bus'][node, [2, 3]] *= 2.0
        # 其他节点
        other_nodes = np.setdiff1d(np.arange(num_nodes), high_load_nodes)
        for node in other_nodes:
            unbalanced_mpc['bus'][node, [2, 3]] *= 0.8
        scenarios['unbalanced'] = unbalanced_mpc
        
        # 4. 故障场景 (移除度数最高的1-2条边)
        fault_mpc = original_mpc.copy()
        fault_mpc['branch'] = fault_mpc['branch'].copy()
        
        # 计算边的重要性（连接度数高的节点）
        branch_importance = []
        for i, branch in enumerate(fault_mpc['branch']):
            from_bus = int(branch[0]) - 1  # 转换为0-based索引
            to_bus = int(branch[1]) - 1
            # 简单的重要性评估：连接节点的度数之和
            importance = from_bus + to_bus  # 简化版本
            branch_importance.append((importance, i))
        
        # 移除最重要的1条边
        branch_importance.sort(reverse=True)
        if len(branch_importance) > 1:
            fault_branch_idx = branch_importance[0][1]
            fault_mpc['branch'] = np.delete(fault_mpc['branch'], fault_branch_idx, axis=0)
        
        scenarios['fault'] = fault_mpc
        
        return scenarios


class MetricsCalculator:
    """指标计算器 - 包含区域内平衡指标"""
    
    @staticmethod
    def calculate_intra_region_balance(partition: np.ndarray, node_loads: np.ndarray) -> float:
        """计算区域内部平衡度 - 新增指标"""
        unique_partitions = np.unique(partition[partition > 0])
        intra_cvs = []
        
        for pid in unique_partitions:
            mask = (partition == pid)
            region_loads = np.abs(node_loads[mask])
            
            if len(region_loads) <= 1:
                intra_cvs.append(0.0)  # 单节点分区完全均衡
            else:
                mean_load = np.mean(region_loads)
                std_load = np.std(region_loads)
                cv = std_load / (mean_load + 1e-8)
                intra_cvs.append(cv)
        
        return np.mean(intra_cvs) if intra_cvs else 1.0
    
    @staticmethod
    def calculate_inter_region_balance(partition: np.ndarray, node_loads: np.ndarray) -> float:
        """计算分区间平衡度 - 传统CV指标"""
        unique_partitions = np.unique(partition[partition > 0])
        partition_loads = []
        
        for pid in unique_partitions:
            mask = (partition == pid)
            total_load = np.sum(np.abs(node_loads[mask]))
            partition_loads.append(total_load)
        
        if len(partition_loads) <= 1:
            return 0.0
        
        mean_load = np.mean(partition_loads)
        std_load = np.std(partition_loads)
        cv = std_load / (mean_load + 1e-8)
        
        return cv
    
    @staticmethod
    def calculate_decoupling(partition: np.ndarray, edge_index: np.ndarray) -> float:
        """计算电气解耦度"""
        if edge_index.shape[1] == 0:
            return 1.0
        
        cross_partition_edges = 0
        total_edges = edge_index.shape[1]
        
        for i in range(total_edges):
            from_node = edge_index[0, i]
            to_node = edge_index[1, i]
            
            if partition[from_node] != partition[to_node]:
                cross_partition_edges += 1
        
        coupling_ratio = cross_partition_edges / total_edges
        decoupling_ratio = 1.0 - coupling_ratio
        
        return decoupling_ratio
    
    @staticmethod
    def calculate_comprehensive_score(inter_cv: float, intra_cv: float, decoupling: float,
                                    weights: Dict[str, float] = None) -> float:
        """计算综合质量分数"""
        if weights is None:
            weights = {'inter_balance': 0.3, 'intra_balance': 0.3, 'decoupling': 0.4}
        
        # 转换为分数（越大越好）
        inter_score = 1.0 / (1.0 + inter_cv)
        intra_score = 1.0 / (1.0 + intra_cv)
        decoupling_score = decoupling
        
        comprehensive_score = (
            weights['inter_balance'] * inter_score +
            weights['intra_balance'] * intra_score +
            weights['decoupling'] * decoupling_score
        )
        
        return comprehensive_score

    @staticmethod
    def convert_to_scores(load_b: float, decoupling: float, power_b: float):
        """将原始(越小越好)指标转换为分数(越大越好)"""
        load_b_score = 1.0 / (1.0 + load_b)
        decoupling_score = 1.0 - decoupling
        power_b_score = 1.0 / (1.0 + power_b)
        return load_b_score, decoupling_score, power_b_score

    @staticmethod
    def calculate_comprehensive_score_v2(load_b_score: float, decoupling_score: float, power_b_score: float,
                                         weights: Dict[str, float] = None) -> float:
        """根据三个分数计算综合质量分数 (越大越好)"""
        if weights is None:
            weights = {'load_b': 0.4, 'decoupling': 0.4, 'power_b': 0.2}
        # 归一化权重
        total_w = sum(weights.values()) + 1e-8
        w_load = weights['load_b'] / total_w
        w_dec = weights['decoupling'] / total_w
        w_pow = weights['power_b'] / total_w
        return w_load * load_b_score + w_dec * decoupling_score + w_pow * power_b_score


class ComprehensiveAgentEvaluator:
    """综合智能体评估器"""

    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None):
        """初始化评估器"""
        self.system = UnifiedTrainingSystem(config_path)
        self.config = self.system.config
        self.device = self.system.device
        self.model_path = model_path

        # 评估配置
        self.evaluation_config = {
            'baseline_comparison': {
                'enabled': True,
                'test_networks': ['ieee14', 'ieee30'],
                'scenarios': ['normal', 'high_load', 'unbalanced', 'fault'],
                'runs_per_test': 10
            },
            'generalization_test': {
                'enabled': True,
                'train_network': 'ieee14',
                'test_networks': ['ieee30', 'ieee57', 'ieee118'],
                'runs_per_network': 20
            },
            'metrics_weights': {
                'inter_balance': 0.3,
                'intra_balance': 0.3,
                'decoupling': 0.4
            },
            # 新指标权重 (score 越大越好)
            'metrics_weights_v2': {
                'load_b': 0.4,
                'decoupling': 0.4,
                'power_b': 0.2
            },
            'thresholds': {
                'excellent_generalization': 10,  # <10%性能下降
                'good_generalization': 20       # 10-20%性能下降
            }
        }
        
        print(f"🔬 综合评估器初始化完成")
        print(f"   训练网络: {self.config['data']['case_name']}")
        print(f"   设备: {self.device}")
    
    def get_evaluation_config(self, mode: str = 'standard') -> Dict[str, Any]:
        """获取指定模式的评估配置"""
        modes = {
            'quick': {
                'baseline_comparison': {
                    'enabled': True,
                    'test_networks': ['ieee14', 'ieee30'],
                    'scenarios': ['normal', 'high_load'],
                    'runs_per_test': 5
                },
                'generalization_test': {
                    'enabled': True,
                    'train_network': 'ieee14',
                    'test_networks': ['ieee30'],
                    'runs_per_network': 10
                }
            },
            'standard': {
                'baseline_comparison': {
                    'enabled': True,
                    'test_networks': ['ieee14', 'ieee30', 'ieee57'],
                    'scenarios': ['normal', 'high_load', 'unbalanced'],
                    'runs_per_test': 10
                },
                'generalization_test': {
                    'enabled': True,
                    'train_network': 'ieee14',
                    'test_networks': ['ieee30', 'ieee57'],
                    'runs_per_network': 20
                }
            },
            'comprehensive': {
                'baseline_comparison': {
                    'enabled': True,
                    'test_networks': ['ieee14', 'ieee30', 'ieee57'],
                    'scenarios': ['normal', 'high_load', 'unbalanced', 'fault'],
                    'runs_per_test': 15
                },
                'generalization_test': {
                    'enabled': True,
                    'train_network': 'ieee14',
                    'test_networks': ['ieee30', 'ieee57', 'ieee118'],
                    'runs_per_network': 30
                }
            }
        }
        
        return modes.get(mode, modes['standard'])
    
    def run_baseline_comparison(self, network_name: str = 'ieee30') -> Dict[str, Any]:
        """运行baseline对比测试"""
        print(f"\n🔍 开始baseline对比测试 - {network_name.upper()}")
        print("=" * 50)

        # 1. 必须使用预训练模型
        if not self.model_path:
            print("❌ 错误：必须提供预训练模型路径")
            print("💡 使用方法: python test.py --baseline --model <模型路径>")
            return {'success': False, 'error': 'No pretrained model provided'}

        print(f"📂 加载预训练模型: {self.model_path}")
        agent, env = self._create_test_environment(network_name)
        try:
            self._load_pretrained_model(agent, self.model_path)
            print(f"✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return {'success': False, 'error': f'Model loading failed: {e}'}

        # 2. 准备测试数据
        test_mpc = load_power_grid_data(network_name)
        scenarios = ScenarioGenerator.generate_test_scenarios(test_mpc, network_name)
        
        # 4. 运行对比测试
        results = {}
        
        for scenario_name, mpc_data in scenarios.items():
            print(f"\n   🔍 测试场景: {scenario_name}")
            
            scenario_results = {}
            
            # 测试我的智能体
            agent_metrics = self._test_agent_on_scenario(agent, mpc_data, scenario_name)
            scenario_results['my_agent'] = agent_metrics
            
            # 测试baseline方法
            baseline_methods = {
                'random': BaselinePartitioner.random_partition,
                'degree_based': BaselinePartitioner.degree_based_partition,
                'spectral': BaselinePartitioner.spectral_partition
            }
            
            for baseline_name, baseline_method in baseline_methods.items():
                baseline_metrics = self._test_baseline_on_scenario(
                    baseline_method, mpc_data, scenario_name
                )
                scenario_results[baseline_name] = baseline_metrics
            
            results[scenario_name] = scenario_results
        
        # 5. 格式化结果
        comparison_results = self._format_comparison_results(results)
        
        return {
            'success': True,
            'network': network_name,
            'results': comparison_results,
            'summary': self._generate_comparison_summary(comparison_results)
        }

    def run_generalization_test(self, train_network: str = 'ieee14') -> Dict[str, Any]:
        """运行泛化能力测试"""
        print(f"\n🌐 开始泛化能力测试")
        print("=" * 50)

        # 1. 必须使用预训练模型
        if not self.model_path:
            print("❌ 错误：必须提供预训练模型路径")
            print("💡 使用方法: python test.py --generalization --model <模型路径>")
            return {'success': False, 'error': 'No pretrained model provided'}

        print(f"📂 加载预训练模型: {self.model_path}")
        train_agent, train_env = self._create_test_environment(train_network)
        try:
            self._load_pretrained_model(train_agent, self.model_path)
            print(f"✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return {'success': False, 'error': f'Model loading failed: {e}'}

        # 2. 获取训练网络baseline性能
        train_mpc = load_power_grid_data(train_network)
        train_score = self._test_agent_on_scenario(train_agent, train_mpc, 'normal')

        print(f"✅ 训练网络性能: {train_score['comprehensive_score']:.3f}")

        # 3. 测试泛化能力
        test_networks = self.evaluation_config['generalization_test']['test_networks']
        test_networks = [net for net in test_networks if net != train_network]

        results = {train_network: train_score}

        for test_network in test_networks:
            print(f"\n   🔍 测试泛化到 {test_network.upper()}...")

            try:
                test_mpc = load_power_grid_data(test_network)
                test_score = self._test_agent_cross_network(train_agent, test_mpc, test_network)

                # 计算性能下降
                degradation = (train_score['comprehensive_score'] - test_score['comprehensive_score']) / train_score['comprehensive_score'] * 100

                results[test_network] = {
                    **test_score,
                    'degradation_pct': degradation,
                    'status': self._classify_generalization(degradation)
                }

                print(f"     结果: 综合分数 {test_score['comprehensive_score']:.3f}, 性能下降 {degradation:.1f}% - {results[test_network]['status']}")

            except Exception as e:
                print(f"     ❌ 测试失败: {e}")
                results[test_network] = {
                    'comprehensive_score': 0.0,
                    'degradation_pct': 100.0,
                    'status': '测试失败',
                    'error': str(e)
                }

        return {
            'success': True,
            'train_network': train_network,
            'results': results,
            'summary': self._generate_generalization_summary(results, train_network)
        }

    def _create_test_environment(self, network_name: str):
        """创建测试环境"""
        # 加载数据
        mpc = load_power_grid_data(network_name)

        # 创建测试配置
        test_config = self.config.copy()
        test_config['data']['case_name'] = network_name

        # 动态调整分区数 - 统一使用3分区以匹配预训练模型
        test_bus_count = mpc['bus'].shape[0]
        if test_bus_count <= 57:
            test_partitions = 3  # IEEE14, IEEE30, IEEE57都使用3分区以匹配预训练模型
        else:
            test_partitions = 5  # IEEE118等大型网络使用5分区

        test_config['environment']['num_partitions'] = test_partitions

        # 数据处理
        processor = PowerGridDataProcessor(
            normalize=test_config['data']['normalize'],
            cache_dir=test_config['data']['cache_dir']
        )
        hetero_data = processor.graph_from_mpc(mpc, test_config).to(self.device)

        # GAT编码器
        gat_config = test_config['gat']
        encoder = create_hetero_graph_encoder(
            hetero_data,
            hidden_channels=gat_config['hidden_channels'],
            gnn_layers=gat_config['gnn_layers'],
            heads=gat_config['heads'],
            output_dim=gat_config['output_dim']
        ).to(self.device)

        with torch.no_grad():
            node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data, test_config)

        # 环境
        env_config = test_config['environment']
        env = PowerGridPartitioningEnv(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=test_partitions,
            reward_weights=env_config['reward_weights'],
            max_steps=env_config['max_steps'],
            device=self.device,
            attention_weights=attention_weights,
            config=test_config
        )

        # 智能体
        agent_config = test_config['agent']
        node_embedding_dim = env.state_manager.embedding_dim
        region_embedding_dim = node_embedding_dim * 2

        agent = PPOAgent(
            node_embedding_dim=node_embedding_dim,
            region_embedding_dim=region_embedding_dim,
            num_partitions=env.num_partitions,
            lr_actor=agent_config['lr_actor'],
            lr_critic=agent_config['lr_critic'],
            gamma=agent_config['gamma'],
            eps_clip=agent_config['eps_clip'],
            k_epochs=agent_config['k_epochs'],
            entropy_coef=agent_config['entropy_coef'],
            value_coef=agent_config['value_coef'],
            device=self.device
        )

        return agent, env

    def _load_pretrained_model(self, agent, model_path: str):
        """加载预训练模型"""
        import torch
        from pathlib import Path

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型状态 (使用安全的weights_only模式)
        try:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        except Exception as e:
            # 如果weights_only=True失败，回退到传统模式但发出警告
            print(f"Warning: Failed to load with weights_only=True, falling back to legacy mode: {e}")
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)

        # 检查checkpoint格式
        if 'actor_state_dict' in checkpoint and 'critic_state_dict' in checkpoint:
            # 标准格式
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])

            # 如果有优化器状态也加载
            if 'actor_optimizer_state_dict' in checkpoint:
                agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            if 'critic_optimizer_state_dict' in checkpoint:
                agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        else:
            raise ValueError(f"不支持的模型格式: {model_path}")

        # 设置为评估模式
        agent.actor.eval()
        agent.critic.eval()

        print(f"✅ 成功加载模型: {model_path}")

    def _test_agent_on_scenario(self, agent, mpc_data: Dict, scenario_name: str, num_runs: int = 10) -> Dict[str, float]:
        """测试智能体在特定场景上的性能"""
        # 创建场景环境
        env = self._create_scenario_environment(mpc_data, scenario_name)

        metrics_list = []

        for run in range(num_runs):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(200):  # 最大步数
                action, _, _ = agent.select_action(state, training=False)
                if action is None:
                    break

                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            # 获取最终分区
            final_partition = env.state_manager.current_partition.cpu().numpy()

            # 计算指标
            metrics = self._calculate_all_metrics(final_partition, env)
            metrics_list.append(metrics)

        # 平均指标
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])

        return avg_metrics

    def _test_baseline_on_scenario(self, baseline_method, mpc_data: Dict, scenario_name: str, num_runs: int = 10) -> Dict[str, float]:
        """测试baseline方法在特定场景上的性能"""
        # 创建场景环境
        env = self._create_scenario_environment(mpc_data, scenario_name)

        # 获取邻接矩阵
        edge_index = env.edge_info['edge_index'].cpu().numpy()
        num_nodes = env.total_nodes
        adjacency_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            adjacency_matrix[u, v] = 1
            adjacency_matrix[v, u] = 1

        metrics_list = []

        for run in range(num_runs):
            # 生成分区
            partition = baseline_method(adjacency_matrix, env.num_partitions)

            # 计算指标
            metrics = self._calculate_all_metrics(partition, env)
            metrics_list.append(metrics)

        # 平均指标
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])

        return avg_metrics

    def _test_agent_cross_network(self, agent, mpc_data: Dict, network_name: str) -> Dict[str, float]:
        """测试智能体跨网络性能"""
        return self._test_agent_on_scenario(agent, mpc_data, 'cross_network', num_runs=20)

    def _create_scenario_environment(self, mpc_data: Dict, scenario_name: str):
        """为特定场景创建环境"""
        # 创建临时配置
        temp_config = self.config.copy()

        # 动态调整分区数 - 统一使用3分区以匹配预训练模型
        test_bus_count = mpc_data['bus'].shape[0]
        if test_bus_count <= 57:
            test_partitions = 3  # IEEE14, IEEE30, IEEE57都使用3分区以匹配预训练模型
        else:
            test_partitions = 5  # IEEE118等大型网络使用5分区

        temp_config['environment']['num_partitions'] = test_partitions

        # 数据处理
        processor = PowerGridDataProcessor(
            normalize=temp_config['data']['normalize'],
            cache_dir=temp_config['data']['cache_dir']
        )
        hetero_data = processor.graph_from_mpc(mpc_data, temp_config).to(self.device)

        # GAT编码器
        gat_config = temp_config['gat']
        encoder = create_hetero_graph_encoder(
            hetero_data,
            hidden_channels=gat_config['hidden_channels'],
            gnn_layers=gat_config['gnn_layers'],
            heads=gat_config['heads'],
            output_dim=gat_config['output_dim']
        ).to(self.device)

        with torch.no_grad():
            node_embeddings, attention_weights = encoder.encode_nodes_with_attention(hetero_data, temp_config)

        # 环境
        env_config = temp_config['environment']
        env = PowerGridPartitioningEnv(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=test_partitions,
            reward_weights=env_config['reward_weights'],
            max_steps=env_config['max_steps'],
            device=self.device,
            attention_weights=attention_weights,
            config=temp_config
        )

        return env

    def _calculate_all_metrics(self, partition: np.ndarray, env) -> Dict[str, float]:
        """计算所有评估指标 (新版本)"""
        # 将分区转换为torch张量以使用reward_function
        import torch
        partition_tensor = torch.tensor(partition, dtype=torch.long, device=env.device)

        # 使用统一奖励系统的核心指标接口
        core_metrics = env.reward_function.get_current_metrics(partition_tensor)

        load_b = core_metrics.get('cv', 1.0)
        decoupling_raw = core_metrics.get('coupling_ratio', 1.0)  # 越小越好
        power_b = core_metrics.get('power_imbalance_normalized', 1.0)

        # 分数转换
        load_b_score, decoupling_score, power_b_score = MetricsCalculator.convert_to_scores(
            load_b, decoupling_raw, power_b)

        # 综合分数
        comprehensive_score = MetricsCalculator.calculate_comprehensive_score_v2(
            load_b_score, decoupling_score, power_b_score, self.evaluation_config.get('metrics_weights_v2', None))

        return {
            'load_b': load_b,
            'decoupling': decoupling_raw,
            'power_b': power_b,
            'load_b_score': load_b_score,
            'decoupling_score': decoupling_score,
            'power_b_score': power_b_score,
            'comprehensive_score': comprehensive_score
        }

    def _classify_generalization(self, degradation_pct: float) -> str:
        """分类泛化能力"""
        if degradation_pct < self.evaluation_config['thresholds']['excellent_generalization']:
            return "🎉 优秀泛化"
        elif degradation_pct < self.evaluation_config['thresholds']['good_generalization']:
            return "✅ 良好泛化"
        else:
            return "⚠️ 泛化不足"

    def _format_comparison_results(self, results: Dict) -> Dict[str, Any]:
        """格式化对比结果 (updated)"""
        formatted_results = {}

        for scenario_name, scenario_data in results.items():
            formatted_scenario = {}

            for method_name, metrics in scenario_data.items():
                formatted_scenario[method_name] = {
                    'load_b': metrics['load_b'],
                    'decoupling': metrics['decoupling'],
                    'power_b': metrics['power_b'],
                    'load_b_score': metrics['load_b_score'],
                    'decoupling_score': metrics['decoupling_score'],
                    'power_b_score': metrics['power_b_score'],
                    'comprehensive_score': metrics['comprehensive_score']
                }

            formatted_results[scenario_name] = formatted_scenario

        return formatted_results

    def _generate_comparison_summary(self, results: Dict) -> Dict[str, Any]:
        """生成对比总结"""
        summary = {
            'overall_improvement': {},
            'best_scenarios': {},
            'method_rankings': {}
        }

        # 计算总体改善幅度
        all_improvements = []

        for scenario_name, scenario_data in results.items():
            my_agent_score = scenario_data['my_agent']['comprehensive_score']

            # 找到最佳baseline
            baseline_scores = {k: v['comprehensive_score'] for k, v in scenario_data.items() if k != 'my_agent'}
            best_baseline_score = max(baseline_scores.values()) if baseline_scores else 0
            best_baseline_name = max(baseline_scores, key=baseline_scores.get) if baseline_scores else 'none'

            # 计算改善幅度
            if best_baseline_score > 0:
                improvement = (my_agent_score - best_baseline_score) / best_baseline_score * 100
            else:
                improvement = 100.0

            all_improvements.append(improvement)

            summary['best_scenarios'][scenario_name] = {
                'my_agent_score': my_agent_score,
                'best_baseline': best_baseline_name,
                'best_baseline_score': best_baseline_score,
                'improvement_pct': improvement
            }

        summary['overall_improvement']['average_improvement'] = np.mean(all_improvements)
        summary['overall_improvement']['min_improvement'] = np.min(all_improvements)
        summary['overall_improvement']['max_improvement'] = np.max(all_improvements)

        return summary

    def _generate_generalization_summary(self, results: Dict, train_network: str) -> Dict[str, Any]:
        """生成泛化能力总结"""
        summary = {
            'train_network': train_network,
            'train_score': results[train_network]['comprehensive_score'],
            'test_results': {},
            'overall_generalization': {}
        }

        degradations = []

        for network, result in results.items():
            if network != train_network and 'error' not in result:
                summary['test_results'][network] = {
                    'score': result['comprehensive_score'],
                    'degradation_pct': result['degradation_pct'],
                    'status': result['status']
                }
                degradations.append(result['degradation_pct'])

        if degradations:
            avg_degradation = np.mean(degradations)
            summary['overall_generalization'] = {
                'average_degradation': avg_degradation,
                'status': self._classify_generalization(avg_degradation),
                'tested_networks': len(degradations)
            }

        return summary

    def create_comparison_visualization(self, results: Dict, save_path: Optional[str] = None) -> None:
        """Create comparison visualization chart (updated version)"""
        print("📊 Generating comparison visualization chart...")

        # Prepare data
        scenarios = list(results.keys())
        methods = list(results[scenarios[0]].keys())

        # 【新】指标列表
        score_metrics = ['load_b_score', 'decoupling_score', 'power_b_score', 'comprehensive_score']
        raw_metrics = ['load_b', 'decoupling', 'power_b']

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle('Agent vs Baseline Performance Comparison', fontsize=16, fontweight='bold')

        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        # --- 绘制score指标 (第一行) ---
        for idx, metric in enumerate(score_metrics):
            ax = axes[0, idx]
            width = 0.8 / len(methods)
            x = np.arange(len(scenarios))

            for j, method in enumerate(methods):
                values = [results[sc][method][metric] for sc in scenarios]
                ax.bar(x + j * width, values, width, label=method, color=colors[j], alpha=0.85)

            title_map = {
                'load_b_score': 'Load Balance Score',
                'decoupling_score': 'Decoupling Score',
                'power_b_score': 'Power Balance Score',
                'comprehensive_score': 'Comprehensive Quality Score'
            }
            ax.set_title(title_map.get(metric, metric), fontsize=11, fontweight='bold')
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(scenarios, rotation=45)
            ax.set_ylabel('Score (↑ Better)')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()

        # --- 绘制原始指标 (第二行) ---
        for idx, metric in enumerate(raw_metrics):
            ax = axes[1, idx]
            width = 0.8 / len(methods)
            x = np.arange(len(scenarios))

            for j, method in enumerate(methods):
                values = [results[sc][method][metric] for sc in scenarios]
                ax.bar(x + j * width, values, width, label=method, color=colors[j], alpha=0.85)

            title_map_raw = {
                'load_b': 'Load CV (raw)',
                'decoupling': 'Coupling Ratio (raw)',
                'power_b': 'Power Imbalance (raw)'
            }
            ax.set_title(title_map_raw.get(metric, metric), fontsize=11, fontweight='bold')
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(scenarios, rotation=45)
            ax.set_ylabel('Raw Value (↓ Better)')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()

        # 移除多余子图 (axes[1,3])
        fig.delaxes(axes[1, 3])

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        if save_path is None:
            from path_helper import create_timestamp_directory
            eval_dir = create_timestamp_directory('evaluation', ['reports', 'metrics', 'comparisons'])
            save_path = f'{eval_dir}/comparisons/comparison_visualization.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison chart saved to: {save_path}")
        plt.close()

    def create_generalization_visualization(self, results: Dict, train_network: str, save_path: Optional[str] = None) -> None:
        """Create generalization capability visualization chart"""
        print("🌐 Generating generalization capability chart...")

        # Prepare data
        networks = []
        scores = []
        degradations = []

        for network, result in results.items():
            if 'error' not in result:
                networks.append(network)
                scores.append(result['comprehensive_score'])
                if network != train_network:
                    degradations.append(result['degradation_pct'])
                else:
                    degradations.append(0.0)

        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Cross-Network Generalization Analysis', fontsize=16, fontweight='bold')

        # Subplot 1: Performance score comparison
        colors = ['red' if net == train_network else 'blue' for net in networks]
        bars1 = ax1.bar(networks, scores, color=colors, alpha=0.7)
        ax1.set_title('Performance Score Comparison Across Networks')
        ax1.set_ylabel('Comprehensive Quality Score')
        ax1.set_xlabel('Networks')
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # Subplot 2: Performance degradation trend
        test_networks = [net for net in networks if net != train_network]
        test_degradations = [deg for net, deg in zip(networks, degradations) if net != train_network]

        if test_degradations:
            colors2 = ['green' if deg < 10 else 'orange' if deg < 20 else 'red' for deg in test_degradations]
            bars2 = ax2.bar(test_networks, test_degradations, color=colors2, alpha=0.7)
            ax2.set_title('Performance Degradation Percentage')
            ax2.set_ylabel('Performance Degradation (%)')
            ax2.set_xlabel('Test Networks')
            ax2.grid(True, alpha=0.3)

            # Add threshold lines
            ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Good Threshold (10%)')
            ax2.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Acceptable Threshold (20%)')
            ax2.legend()

            # Add value labels
            for bar, deg in zip(bars2, test_degradations):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{deg:.1f}%', ha='center', va='bottom')

        plt.tight_layout()

        # Save figure
        if save_path is None:
            from path_helper import create_timestamp_directory
            eval_dir = create_timestamp_directory('evaluation', ['reports', 'metrics', 'comparisons'])
            save_path = f'{eval_dir}/comparisons/generalization_visualization.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Generalization chart saved to: {save_path}")
        plt.close()

    def generate_evaluation_report(self, comparison_results: Dict = None, generalization_results: Dict = None,
                                 save_path: Optional[str] = None) -> str:
        """生成评估报告"""
        print("📝 生成评估报告...")

        report_lines = []
        report_lines.append("🎯 智能体性能评估报告")
        report_lines.append("=" * 50)
        report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Baseline对比结果
        if comparison_results:
            report_lines.append("📊 Baseline对比结果")
            report_lines.append("-" * 30)

            network = comparison_results['network']
            results = comparison_results['results']
            summary = comparison_results['summary']

            report_lines.append(f"测试网络: {network.upper()}")
            report_lines.append("")

            # 详细对比表格
            report_lines.append("指标对比详情:")
            report_lines.append("")

            # 表头
            header = f"{'场景':<10} {'方法':<12} {'负载CV':<10} {'耦合率':<10} {'功率不平衡':<12} {'负载S':<8} {'解耦S':<8} {'功率S':<8} {'综合分':<8}"
            report_lines.append(header)
            report_lines.append("-" * len(header))

            for scenario_name, scenario_data in results.items():
                for i, (method_name, metrics) in enumerate(scenario_data.items()):
                    scenario_display = scenario_name if i == 0 else ""
                    line = (
                        f"{scenario_display:<10} {method_name:<12} "
                        f"{metrics['load_b']:<10.3f} {metrics['decoupling']:<10.3f} {metrics['power_b']:<12.3f} "
                        f"{metrics['load_b_score']:<8.3f} {metrics['decoupling_score']:<8.3f} {metrics['power_b_score']:<8.3f} "
                        f"{metrics['comprehensive_score']:<8.3f}"
                    )
                    report_lines.append(line)
                report_lines.append("")

            # 改善幅度总结
            report_lines.append("🏆 性能改善总结:")
            avg_improvement = summary['overall_improvement']['average_improvement']
            min_improvement = summary['overall_improvement']['min_improvement']
            max_improvement = summary['overall_improvement']['max_improvement']

            report_lines.append(f"平均性能提升: {avg_improvement:.1f}%")
            report_lines.append(f"最小性能提升: {min_improvement:.1f}%")
            report_lines.append(f"最大性能提升: {max_improvement:.1f}%")
            report_lines.append("")

            # 各场景最佳表现
            for scenario_name, scenario_summary in summary['best_scenarios'].items():
                improvement = scenario_summary['improvement_pct']
                status = "🎉 优秀" if improvement > 15 else "✅ 良好" if improvement > 5 else "⚠️ 有限"
                report_lines.append(f"{scenario_name}: 提升 {improvement:.1f}% - {status}")

            report_lines.append("")

        # 泛化能力结果
        if generalization_results:
            report_lines.append("🌐 泛化能力测试结果")
            report_lines.append("-" * 30)

            train_network = generalization_results['train_network']
            results = generalization_results['results']
            summary = generalization_results['summary']

            report_lines.append(f"训练网络: {train_network.upper()}")
            report_lines.append(f"训练网络性能: {summary['train_score']:.3f}")
            report_lines.append("")

            # 泛化测试详情
            report_lines.append("泛化测试详情:")
            report_lines.append("")

            header = f"{'测试网络':<10} {'质量分数':<10} {'性能下降':<10} {'泛化评级':<12}"
            report_lines.append(header)
            report_lines.append("-" * len(header))

            for network, result in summary['test_results'].items():
                line = f"{network.upper():<10} {result['score']:<10.3f} {result['degradation_pct']:<10.1f}% {result['status']:<12}"
                report_lines.append(line)

            report_lines.append("")

            # 总体泛化评价
            overall = summary['overall_generalization']
            report_lines.append("🏆 总体泛化能力:")
            report_lines.append(f"平均性能下降: {overall['average_degradation']:.1f}%")
            report_lines.append(f"测试网络数量: {overall['tested_networks']}")
            report_lines.append(f"总体评级: {overall['status']}")
            report_lines.append("")

        # 总体结论
        report_lines.append("🎯 总体结论")
        report_lines.append("-" * 30)

        if comparison_results and generalization_results:
            avg_improvement = comparison_results['summary']['overall_improvement']['average_improvement']
            avg_degradation = generalization_results['summary']['overall_generalization']['average_degradation']

            if avg_improvement > 10 and avg_degradation < 15:
                conclusion = "🌟 优秀 - 在所有测试中表现卓越"
            elif avg_improvement > 5 and avg_degradation < 25:
                conclusion = "✅ 良好 - 整体性能令人满意"
            else:
                conclusion = "⚠️ 需要改进 - 存在性能提升空间"

            report_lines.append(f"✅ 在baseline对比中平均提升: {avg_improvement:.1f}%")
            report_lines.append(f"🌐 跨网络泛化平均下降: {avg_degradation:.1f}%")
            report_lines.append(f"⭐ 综合评级: {conclusion}")

        elif comparison_results:
            avg_improvement = comparison_results['summary']['overall_improvement']['average_improvement']
            if avg_improvement > 10:
                conclusion = "🌟 优秀性能提升"
            elif avg_improvement > 5:
                conclusion = "✅ 良好性能提升"
            else:
                conclusion = "⚠️ 有限性能提升"

            report_lines.append(f"✅ 平均性能提升: {avg_improvement:.1f}%")
            report_lines.append(f"⭐ 评级: {conclusion}")

        elif generalization_results:
            avg_degradation = generalization_results['summary']['overall_generalization']['average_degradation']
            if avg_degradation < 10:
                conclusion = "🌟 优秀泛化能力"
            elif avg_degradation < 20:
                conclusion = "✅ 良好泛化能力"
            else:
                conclusion = "⚠️ 有限泛化能力"

            report_lines.append(f"🌐 平均性能下降: {avg_degradation:.1f}%")
            report_lines.append(f"⭐ 评级: {conclusion}")

        # 生成报告文本
        report_text = "\n".join(report_lines)

        # 保存报告
        if save_path is None:
            from path_helper import create_timestamp_directory
            eval_dir = create_timestamp_directory('evaluation', ['reports', 'metrics', 'comparisons'])
            save_path = f'{eval_dir}/reports/evaluation_report.txt'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"✅ 评估报告已保存到: {save_path}")

        return report_text
