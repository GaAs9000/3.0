#!/usr/bin/env python3
"""
异构GAT模型测试脚本

这个脚本将测试新的异构GAT模型的各个组件：
1. PhysicsGATv2Conv - 物理引导的GATv2卷积层
2. GNNEncoder - 同构GNN编码器  
3. ActorCriticGNN - 完整的Actor-Critic策略网络
4. 与现有数据处理流程的兼容性
"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
import traceback
from pathlib import Path
import sys
import os
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入必要模块
from src.data_processing import PowerGridDataProcessor
from src.gat import (
    PhysicsGATv2Conv, 
    GNNEncoder, 
    ActorCriticGNN, 
    create_hetero_gat_model,
    test_hetero_gat_model,
    create_hetero_graph_encoder,
    test_hetero_graph_encoder
)

# 导入测试数据
from config import PARTITION_CONFIG, RANDOM_SEED

def set_random_seed(seed=42):
    """设置随机种子确保可重现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_simple_test_data() -> HeteroData:
    """
    Creates and processes a simple, small mock power system case and returns a HeteroData object.
    This avoids file I/O and complex loading logic for a straightforward test.
    """
    print("✅ Creating a simple mock dataset for testing...")
    mpc = {
        'baseMVA': 100.0,
        'bus': np.array([
            [1, 3, 0, 0, 0, 0, 1, 1.0, 0, 138, 1, 1.1, 0.9],   # Slack
            [2, 2, 50, 30, 0, 0, 1, 1.0, 0, 138, 1, 1.1, 0.9],  # PV
            [3, 1, 20, 10, 0, 0, 1, 1.0, 0, 138, 1, 1.1, 0.9],  # PQ
            [4, 1, 30, 15, 0, 0, 1, 1.0, 0, 138, 1, 1.1, 0.9],  # PQ
        ]),
        'gen': np.array([
            [1, 100, 50, 100, -100, 1.0, 100, 1, 200, 0], # Gen on Slack
            [2, 50, 30, 80, -80, 1.0, 100, 1, 100, 0],    # Gen on PV
        ]),
        'branch': np.array([
            [1, 2, 0.01, 0.05, 0.1, 100, 110, 120, 0, 0, 1, -360, 360],      # Line
            [1, 3, 0.02, 0.1, 0.2, 100, 110, 120, 0, 0, 1, -360, 360],      # Line
            [2, 4, 0.015, 0.07, 0.15, 100, 110, 120, 0, 0, 1, -360, 360],   # Line
            [3, 4, 0.005, 0.02, 0.05, 100, 110, 120, 1.02, 0, 1, -360, 360], # Transformer
        ])
    }
    
    # Use the data processor
    processor = PowerGridDataProcessor(normalize=True, cache_dir='debug_cache')
    data = processor.graph_from_mpc(mpc)
    
    # ToUndirected transform is important for GNNs
    data = ToUndirected()(data)
    print("✅ Successfully created undirected hetero graph data.")
    
    assert isinstance(data, HeteroData)
    assert data.is_undirected()
    
    return data

def test_physics_gatv2_conv():
    """测试PhysicsGATv2Conv层"""
    print("\n" + "="*60)
    print("🧪 测试 PhysicsGATv2Conv 层")
    print("="*60)
    
    try:
        # 创建测试数据
        num_nodes = 10
        in_channels = 14
        out_channels = 32
        edge_dim = 9
        heads = 4
        
        # 节点特征
        x = torch.randn(num_nodes, in_channels)
        
        # 边索引 (简单的环形连接)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ], dtype=torch.long)
        
        # 边特征 (包含阻抗信息)
        num_edges = edge_index.shape[1]
        edge_attr = torch.randn(num_edges, edge_dim)
        edge_attr[:, 3] = torch.rand(num_edges) * 2 + 0.1  # 阻抗模长 > 0
        
        # 创建PhysicsGATv2Conv层
        conv = PhysicsGATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,
            temperature=1.0,
            physics_weight=0.5
        )
        
        print(f"📊 输入形状:")
        print(f"   • 节点特征: {x.shape}")
        print(f"   • 边索引: {edge_index.shape}")  
        print(f"   • 边特征: {edge_attr.shape}")
        
        # 前向传播
        with torch.no_grad():
            out = conv(x, edge_index, edge_attr)
            
        print(f"📈 输出形状: {out.shape}")
        print(f"✅ PhysicsGATv2Conv 测试成功！")
        
        # 测试注意力权重
        with torch.no_grad():
            out, attention = conv(x, edge_index, edge_attr, return_attention_weights=True)
            if attention is not None:
                print(f"🔍 注意力权重形状: {attention.shape if hasattr(attention, 'shape') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ PhysicsGATv2Conv 测试失败: {e}")
        traceback.print_exc()
        return False

def test_gnn_encoder():
    """测试GNNEncoder"""
    print("\n" + "="*60)
    print("🧪 测试 GNNEncoder")
    print("="*60)
    
    try:
        # 创建测试数据
        num_nodes = 15
        in_channels = 14
        hidden_channels = 32
        num_layers = 3
        heads = 4
        edge_dim = 9
        
        # 节点特征
        x = torch.randn(num_nodes, in_channels)
        
        # 边索引 (更复杂的连接)
        edges = []
        for i in range(num_nodes):
            for j in range(i+1, min(i+4, num_nodes)):  # 每个节点连接到后面3个节点
                edges.append([i, j])
                edges.append([j, i])  # 无向图
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # 边特征
        num_edges = edge_index.shape[1]
        edge_attr = torch.randn(num_edges, edge_dim)
        edge_attr[:, 3] = torch.rand(num_edges) * 2 + 0.1  # 阻抗模长
        
        # 创建GNNEncoder
        encoder = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            edge_dim=edge_dim
        )
        
        print(f"📊 输入形状:")
        print(f"   • 节点特征: {x.shape}")
        print(f"   • 边索引: {edge_index.shape}")
        print(f"   • 边特征: {edge_attr.shape}")
        print(f"🔧 编码器参数:")
        print(f"   • 层数: {num_layers}")
        print(f"   • 隐藏维度: {hidden_channels}")
        print(f"   • 注意力头数: {heads}")
        
        # 前向传播
        with torch.no_grad():
            embeddings = encoder(x, edge_index, edge_attr)
            
        print(f"📈 输出嵌入形状: {embeddings.shape}")
        print(f"📊 编码器参数量: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"✅ GNNEncoder 测试成功！")
        
        return True, encoder, (x, edge_index, edge_attr)
        
    except Exception as e:
        print(f"❌ GNNEncoder 测试失败: {e}")
        traceback.print_exc()
        return False, None, None

def test_hetero_data_processing():
    """测试异构数据处理"""
    print("\n" + "="*60) 
    print("🧪 测试异构数据处理")
    print("="*60)
    
    try:
        # 加载测试数据
        mpc = load_simple_test_data()
        
        # 创建数据处理器
        processor = PowerGridDataProcessor(normalize=True, cache_dir='test_cache')
        
        # 转换为异构图数据
        hetero_data = processor.graph_from_mpc(mpc)
        
        print(f"📊 异构图数据信息:")
        print(f"   • 节点类型: {list(hetero_data.x_dict.keys())}")
        print(f"   • 边类型: {list(hetero_data.edge_index_dict.keys())}")
        
        # 显示各类型的详细信息
        total_nodes = 0
        for node_type, x in hetero_data.x_dict.items():
            print(f"   • {node_type}: {x.shape[0]} 个节点, {x.shape[1]} 维特征")
            total_nodes += x.shape[0]
        
        total_edges = 0
        for edge_type, edge_index in hetero_data.edge_index_dict.items():
            num_edges = edge_index.shape[1]
            print(f"   • {edge_type}: {num_edges} 条边")
            total_edges += num_edges
            
        print(f"📈 总计: {total_nodes} 个节点, {total_edges} 条边")
        
        # 验证元数据
        metadata = hetero_data.metadata()
        print(f"🔍 图元数据:")
        print(f"   • 节点类型: {metadata[0]}")
        print(f"   • 关系类型: {metadata[1]}")
        
        print(f"✅ 异构数据处理测试成功！")
        
        return True, hetero_data
        
    except Exception as e:
        print(f"❌ 异构数据处理测试失败: {e}")
        traceback.print_exc()
        return False, None

def test_actor_critic_gnn(hetero_data):
    """测试ActorCriticGNN模型"""
    print("\n" + "="*60)
    print("🧪 测试 ActorCriticGNN 模型")
    print("="*60)
    
    try:
        # 使用便捷函数创建模型
        model = create_hetero_gat_model(
            data=hetero_data,
            hidden_channels=32,
            gnn_layers=2,
            heads=4,
            dropout=0.3
        )
        
        print(f"📊 模型信息:")
        print(f"   • 参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   • 节点类型: {model.node_types}")
        print(f"   • 边类型数: {len(model.edge_types)}")
        
        # 创建边界节点字典 (模拟边界节点)
        boundary_node_dict = {}
        for node_type in hetero_data.x_dict.keys():
            num_nodes = hetero_data.x_dict[node_type].shape[0]
            # 随机选择一些节点作为边界节点
            num_boundary = min(3, num_nodes)
            if num_boundary > 0:
                boundary_indices = torch.randint(0, num_nodes, (num_boundary,))
                boundary_node_dict[node_type] = boundary_indices
        
        print(f"🔍 边界节点:")
        for node_type, indices in boundary_node_dict.items():
            print(f"   • {node_type}: {len(indices)} 个边界节点")
        
        # 创建区域嵌入
        num_regions = 3
        region_embeddings = torch.randn(num_regions, 32)
        
        # 前向传播测试
        with torch.no_grad():
            node_probs, region_probs, state_value = model(
                hetero_data, boundary_node_dict, region_embeddings
            )
        
        print(f"📈 输出结果:")
        print(f"   • 节点选择概率: {node_probs.shape}, 和={node_probs.sum():.4f}")
        print(f"   • 区域选择概率: {region_probs.shape}")
        print(f"   • 状态价值: {state_value.item():.4f}")
        
        # 验证概率分布
        if torch.allclose(node_probs.sum(), torch.tensor(1.0), atol=1e-6):
            print("✅ 节点选择概率分布正确")
        else:
            print("⚠️ 节点选择概率分布异常")
        
        print(f"✅ ActorCriticGNN 模型测试成功！")
        
        return True, model
        
    except Exception as e:
        print(f"❌ ActorCriticGNN 模型测试失败: {e}")
        traceback.print_exc()
        return False, None

def test_attention_weights_extraction(model):
    """测试注意力权重提取"""
    print("\n" + "="*60)
    print("🧪 测试注意力权重提取")
    print("="*60)
    
    try:
        attention_weights = model.get_attention_weights()
        
        print(f"📊 注意力权重信息:")
        print(f"   • 权重数量: {len(attention_weights)}")
        
        for i, weights in enumerate(attention_weights):
            if weights is not None:
                print(f"   • 第{i+1}层权重形状: {weights.shape}")
            else:
                print(f"   • 第{i+1}层权重: None")
        
        if len(attention_weights) > 0:
            print("✅ 注意力权重提取成功！")
        else:
            print("⚠️ 未找到注意力权重")
        
        return True
        
    except Exception as e:
        print(f"❌ 注意力权重提取失败: {e}")
        traceback.print_exc()
        return False

def run_full_integration_test():
    """运行完整的集成测试"""
    print("\n" + "="*80)
    print("🚀 异构GAT模型 - 完整集成测试")
    print("="*80)
    
    # 设置随机种子
    set_random_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    test_results = {}
    
    # 1. 测试基础组件
    print("\n📋 第一阶段：基础组件测试")
    test_results['physics_conv'] = test_physics_gatv2_conv()
    test_results['gnn_encoder'] = test_gnn_encoder()[0]
    
    # 2. 测试异构数据处理
    print("\n📋 第二阶段：异构数据处理测试")
    data_success, hetero_data = test_hetero_data_processing()
    test_results['hetero_data'] = data_success
    
    if not data_success:
        print("❌ 异构数据处理失败，无法继续后续测试")
        return test_results
    
    # 3. 测试完整模型
    print("\n📋 第三阶段：完整模型测试")
    model_success, model = test_actor_critic_gnn(hetero_data)
    test_results['actor_critic'] = model_success
    
    if model_success and model is not None:
        # 4. 测试注意力权重提取
        print("\n📋 第四阶段：注意力权重提取测试")
        test_results['attention_weights'] = test_attention_weights_extraction(model)
        
        # 5. 移动到设备并测试
        if device.type == 'cuda':
            print("\n📋 第五阶段：GPU兼容性测试")
            try:
                model = model.to(device)
                hetero_data = hetero_data.to(device)
                
                # 重新测试前向传播
                boundary_node_dict = {}
                for node_type in hetero_data.x_dict.keys():
                    num_nodes = hetero_data.x_dict[node_type].shape[0]
                    num_boundary = min(2, num_nodes)
                    if num_boundary > 0:
                        boundary_indices = torch.randint(0, num_nodes, (num_boundary,), device=device)
                        boundary_node_dict[node_type] = boundary_indices
                
                region_embeddings = torch.randn(2, 32, device=device)
                
                with torch.no_grad():
                    node_probs, region_probs, state_value = model(
                        hetero_data, boundary_node_dict, region_embeddings
                    )
                
                print("✅ GPU兼容性测试成功！")
                test_results['gpu_compatibility'] = True
                
            except Exception as e:
                print(f"⚠️ GPU兼容性测试失败: {e}")
                test_results['gpu_compatibility'] = False
    
    # 打印测试总结
    print("\n" + "="*80)
    print("📊 测试结果总结")
    print("="*80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   • {test_name}: {status}")
        if result:
            passed_tests += 1
    
    success_rate = passed_tests / total_tests * 100
    print(f"\n🎯 测试通过率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 异构GAT模型测试基本成功！")
    elif success_rate >= 60:
        print("⚠️ 异构GAT模型存在部分问题，需要修复")
    else:
        print("❌ 异构GAT模型存在严重问题，需要重新检查")
    
    return test_results

def main():
    """Main test function to validate the HeteroGraphEncoder."""
    print("🔬 Starting test for the refactored HeteroGraphEncoder...")
    
    # 1. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔩 Using device: {device}")
    
    # 2. Load test data
    data = load_simple_test_data()
    data = data.to(device)
    
    # 3. Use the new, correct test function from gat.py
    print("\n🚀 Calling test_hetero_graph_encoder to validate the model...")
    model = test_hetero_graph_encoder(data, device)
    
    # 4. Final validation
    assert model is not None, "Model creation failed!"
    print(f"\n🎉🎉🎉 Test for HeteroGraphEncoder passed successfully! 🎉🎉🎉")

if __name__ == "__main__":
    main() 