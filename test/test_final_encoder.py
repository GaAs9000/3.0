#!/usr/bin/env python3
"""
测试最终重构后的异构图编码器 (HeteroGraphEncoder)
"""

import sys
sys.path.append('src')
sys.path.append('.')

import torch
from torch_geometric.data import HeteroData
from src.gat import create_hetero_graph_encoder

def create_test_hetero_data():
    """创建测试用的异构图数据"""
    data = HeteroData()
    
    # 节点类型1: bus (母线)
    data['bus'].x = torch.randn(10, 14)  # 10个母线节点，14维特征
    
    # 节点类型2: gen (发电机)  
    data['gen'].x = torch.randn(3, 8)   # 3个发电机节点，8维特征
    
    # 边类型1: bus-bus (母线间连接)
    data['bus', 'connects', 'bus'].edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,5]], dtype=torch.long)
    data['bus', 'connects', 'bus'].edge_attr = torch.randn(5, 9)  # 5条边，9维特征
    
    # 边类型2: gen-bus (发电机-母线连接)
    data['gen', 'connects', 'bus'].edge_index = torch.tensor([[0,1,2], [0,3,7]], dtype=torch.long)
    data['gen', 'connects', 'bus'].edge_attr = torch.randn(3, 9)  # 3条边，9维特征
    
    # 边类型3: gen-gen (为'gen'节点添加自循环以确保它们能被更新)
    num_gens = data['gen'].num_nodes
    data['gen', 'loops', 'gen'].edge_index = torch.arange(num_gens).unsqueeze(0).repeat(2, 1)
    data['gen', 'loops', 'gen'].edge_attr = torch.ones(num_gens, 9) # 提供虚拟的边特征
    
    return data

def main():
    """主测试函数"""
    print("🔬 开始测试最终重构后的 HeteroGraphEncoder...")
    
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔩 使用设备: {device}")
    
    # 1. 创建测试数据
    try:
        data = create_test_hetero_data()
        data = data.to(device)
        print("✅ 成功创建测试异构图数据")
    except Exception as e:
        print(f"❌ 创建测试数据失败: {e}")
        return

    # 2. 创建编码器
    try:
        encoder = create_hetero_graph_encoder(
            data, 
            hidden_channels=32, 
            gnn_layers=2, 
            heads=4, 
            output_dim=64
        )
        encoder = encoder.to(device)
        print("✅ 成功创建异构图编码器")
        print(f"📊 编码器参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    except Exception as e:
        print(f"❌ 创建编码器失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 执行前向传播并验证输出
    try:
        print("\n🚀 执行完整前向传播并验证...")
        with torch.no_grad():
            node_emb, attention_weights, graph_emb = encoder(
                data, 
                return_attention_weights=True, 
                return_graph_embedding=True
            )

        print("✅ 完整前向传播测试通过")
        
        # 验证节点嵌入
        print("\n--- 节点嵌入验证 ---")
        for node_type, embeddings in node_emb.items():
            print(f"   - 节点类型 '{node_type}' -> 嵌入形状: {embeddings.shape}")
            assert embeddings.shape == (data[node_type].num_nodes, 64)
            assert embeddings.device.type == device.type
        print("   ✅ 形状和设备正确")

        # 验证图嵌入
        print("\n--- 图嵌入验证 ---")
        print(f"   - 图级别嵌入形状: {graph_emb.shape}")
        assert graph_emb.shape == (1, 64)
        assert graph_emb.device.type == device.type
        print("   ✅ 形状和设备正确")

        # 验证注意力权重
        print("\n--- 注意力权重验证 ---")
        print(f"   - 提取的注意力权重数量: {len(attention_weights)}")
        assert len(attention_weights) > 0, "关键失败：未能提取到注意力权重！"
        print("   ✅ 成功提取注意力权重")

    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉🎉🎉 所有测试均已通过！您的编码器已准备就绪。🎉🎉🎉")

if __name__ == "__main__":
    main() 