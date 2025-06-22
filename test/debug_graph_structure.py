import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_unified import load_power_grid_data
from src.data_processing import PowerGridDataProcessor

def debug_ieee14_structure():
    """
    详细调试IEEE14数据处理过程
    """
    print("🔍 IEEE 14节点系统数据诊断")
    print("=" * 60)
    
    # 1. 检查原始数据
    print("\n📊 原始MATPOWER数据检查:")
    mpc = load_power_grid_data('ieee14')
    
    print(f"节点数: {mpc['bus'].shape[0]}")
    print(f"支路数: {mpc['branch'].shape[0]}")
    
    # 2. 分析支路数据
    print("\n🔗 支路数据分析:")
    branch = mpc['branch']
    
    # 检查阻抗值
    r_values = branch[:, 2]  # 电阻
    x_values = branch[:, 3]  # 电抗
    impedance_magnitude = np.sqrt(r_values**2 + x_values**2)
    
    print(f"电阻范围: {r_values.min():.6f} - {r_values.max():.6f}")
    print(f"电抗范围: {x_values.min():.6f} - {x_values.max():.6f}")
    print(f"阻抗幅值范围: {impedance_magnitude.min():.6f} - {impedance_magnitude.max():.6f}")
    
    # 找出异常的支路
    print(f"\n⚠️ 异常支路（阻抗 > 1000）:")
    for i, (r, x, z) in enumerate(zip(r_values, x_values, impedance_magnitude)):
        if z > 1000:
            from_bus = int(branch[i, 0])
            to_bus = int(branch[i, 1])
            print(f"  支路 {i}: {from_bus} -> {to_bus}, R={r:.2f}, X={x:.2f}, |Z|={z:.2f}")
    
    # 3. 数据处理过程分析
    print("\n🔨 数据处理过程分析:")
    processor = PowerGridDataProcessor(normalize=False, cache_dir='debug_cache')
    
    try:
        # 原始数据处理
        baseMVA, df_nodes, df_edges, df_edge_features = processor.process_matpower_data(mpc)
        
        print(f"处理后节点数: {len(df_nodes)}")
        print(f"处理后边数: {len(df_edges)}")
        
        # 检查边的有效性
        print(f"\n📋 边的有效性检查:")
        valid_edges = 0
        invalid_edges = 0
        
        for i, row in df_edge_features.iterrows():
            z_magnitude = row['|z|']
            if z_magnitude < 1000:  # 合理的阻抗值
                valid_edges += 1
            else:
                invalid_edges += 1
                from_bus = df_edges.iloc[i]['from_bus']
                to_bus = df_edges.iloc[i]['to_bus']
                print(f"  无效边 {i}: {from_bus} -> {to_bus}, |Z|={z_magnitude:.2f}")
        
        print(f"有效边数: {valid_edges}")
        print(f"无效边数: {invalid_edges}")
        
        # 4. 创建异构图并分析
        print(f"\n🌐 异构图创建与分析:")
        data = processor.create_pyg_hetero_data(df_nodes, df_edges, df_edge_features)
        
        print(f"节点类型: {list(data.node_types)}")
        print(f"边类型: {list(data.edge_types)}")
        
        # 统计每种边类型的数量
        total_edges = 0
        for edge_type in data.edge_types:
            edge_count = data[edge_type].edge_index.shape[1]
            total_edges += edge_count
            print(f"  {edge_type}: {edge_count} 条边")
        
        print(f"总边数: {total_edges}")
        
        # 5. 与MetisInitializer的连接检查
        print(f"\n🔧 MetisInitializer邻接列表检查:")
        from src.rl.utils import MetisInitializer
        
        device = torch.device('cpu')
        initializer = MetisInitializer(data, device)
        
        # 检查邻接列表
        node_degrees = [len(initializer.adjacency_list[i]) for i in range(initializer.total_nodes)]
        print(f"节点度数: {node_degrees}")
        print(f"非零度数节点: {sum(1 for d in node_degrees if d > 0)}")
        print(f"孤立节点: {sum(1 for d in node_degrees if d == 0)}")
        
        # 打印实际的连接
        print(f"\n🔗 实际连接关系:")
        edge_count = 0
        for i in range(initializer.total_nodes):
            for neighbor in initializer.adjacency_list[i]:
                if i < neighbor:  # 避免重复
                    edge_count += 1
                    print(f"  {i} - {neighbor}")
        
        print(f"实际边数: {edge_count}")
        
        return data, initializer
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_with_standard_ieee14():
    """
    与标准IEEE14数据进行对比
    """
    print("\n📊 与标准IEEE14拓扑对比:")
    print("=" * 60)
    
    # 标准IEEE14连接关系（根据您提供的单线图）
    standard_connections = [
        (1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5),
        (4, 7), (4, 9), (5, 6), (6, 11), (6, 12), (6, 13),
        (7, 8), (7, 9), (9, 10), (9, 14), (10, 11), (12, 13), (13, 14)
    ]
    
    print(f"标准IEEE14应有连接数: {len(standard_connections)}")
    print("标准连接列表:")
    for i, (a, b) in enumerate(standard_connections):
        print(f"  {i+1:2d}: {a:2d} - {b:2d}")
    
    # 检查我们的数据
    mpc = load_power_grid_data('ieee14')
    actual_connections = []
    for i, branch in enumerate(mpc['branch']):
        from_bus = int(branch[0])
        to_bus = int(branch[1])
        z_mag = np.sqrt(branch[2]**2 + branch[3]**2)
        
        # 只包含合理阻抗的连接
        if z_mag < 1000:
            actual_connections.append((from_bus, to_bus))
    
    print(f"\n实际有效连接数: {len(actual_connections)}")
    print("实际连接列表:")
    for i, (a, b) in enumerate(actual_connections):
        print(f"  {i+1:2d}: {a:2d} - {b:2d}")
    
    # 找出缺失和多余的连接
    standard_set = set(standard_connections)
    actual_set = set(actual_connections)
    
    missing = standard_set - actual_set
    extra = actual_set - standard_set
    
    if missing:
        print(f"\n❌ 缺失的连接 ({len(missing)} 个):")
        for a, b in sorted(missing):
            print(f"  {a} - {b}")
    
    if extra:
        print(f"\n➕ 多余的连接 ({len(extra)} 个):")
        for a, b in sorted(extra):
            print(f"  {a} - {b}")
    
    if not missing and not extra:
        print(f"\n✅ 连接关系完全正确！")

if __name__ == "__main__":
    print("🚀 开始IEEE14数据结构诊断...")
    
    # 主要诊断
    data, initializer = debug_ieee14_structure()
    
    # 对比分析
    compare_with_standard_ieee14()
    
    print("\n" + "=" * 60)
    print("🏁 诊断完成！") 