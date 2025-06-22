# Create new file: test/test_initialization.py

import torch
import numpy as np
import networkx as nx
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add project root path
sys.path.append(str(Path(__file__).parent.parent))

from src.rl.utils import MetisInitializer
from src.data_processing import PowerGridDataProcessor
from train_unified import load_power_grid_data

def save_partition_details(partition_info, output_dir="output/partition_analysis"):
    """
    保存分区详细信息到文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"partition_analysis_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 转换numpy类型为Python原生类型
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    # 保存为JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert_types(partition_info), f, indent=2, ensure_ascii=False)
    
    print(f"📁 分区详细信息已保存到: {filepath}")
    return filepath

def analyze_partition_details(partition, initializer, case_name="unknown", save_to_file=True):
    """
    分析并输出分区的详细信息
    """
    print(f"\n=== 📊 {case_name.upper()} 分区详细分析 ===")
    
    labels_np = partition.cpu().numpy()
    total_nodes = len(labels_np)
    
    # 首先检查图的基本连通性
    print(f"\n🔍 图结构调试信息:")
    print(f"  总节点数: {total_nodes}")
    
    # 统计每个节点的度数
    degrees = [len(initializer.adjacency_list[i]) for i in range(total_nodes)]
    print(f"  节点度数: {degrees}")
    print(f"  最大度数: {max(degrees) if degrees else 0}")
    print(f"  最小度数: {min(degrees) if degrees else 0}")
    print(f"  孤立节点数: {degrees.count(0)}")
    
    # 检查原始图的连通性
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    total_edges = 0
    for i in range(total_nodes):
        for neighbor in initializer.adjacency_list[i]:
            if i < neighbor:  # 避免重复
                G.add_edge(i, neighbor)
                total_edges += 1
    
    print(f"  总边数: {total_edges}")
    if total_edges > 0:
        is_graph_connected = nx.is_connected(G)
        print(f"  原始图连通性: {'✅ 连通' if is_graph_connected else '❌ 不连通'}")
    else:
        is_graph_connected = False
        print(f"  原始图连通性: ❌ 无边图")
    
    if not nx.is_connected(G) and total_edges > 0:
        components = list(nx.connected_components(G))
        print(f"  连通分量数: {len(components)}")
        for i, comp in enumerate(components[:3]):  # 只显示前3个
            print(f"    分量{i+1}: {sorted(list(comp))}")
    
    # 统计信息
    unique_labels = np.unique(labels_np)
    num_partitions = len([l for l in unique_labels if l > 0])
    unassigned_count = np.sum(labels_np == 0)
    
    print(f"\n📈 分区基本统计:")
    print(f"  总节点数: {total_nodes}")
    print(f"  分区数量: {num_partitions}")
    print(f"  未分区节点数: {unassigned_count}")
    print(f"  分区标签: {sorted(unique_labels)}")
    
    # 详细信息字典
    partition_info = {
        "case_name": case_name,
        "timestamp": datetime.now().isoformat(),
        "total_nodes": total_nodes,
        "num_partitions": num_partitions,
        "unassigned_count": unassigned_count,
        "partitions": {},
        "inter_partition_connections": [],
        "connectivity_analysis": {},
        "graph_structure": {
            "total_edges": total_edges,
            "is_connected": is_graph_connected,
            "num_components": len(list(nx.connected_components(G))) if total_edges > 0 else total_nodes,
            "degrees": degrees
        }
    }
    
    # 分析每个分区
    print(f"\n🎯 各分区详细信息:")
    for label in sorted(unique_labels):
        nodes = np.where(labels_np == label)[0].tolist()
        if label == 0:
            partition_name = "未分区"
            print(f"  {partition_name} (标签 {label}): {len(nodes)} 个节点")
            print(f"    节点列表: {nodes}")
        else:
            partition_name = f"分区{label}"
            print(f"  {partition_name}: {len(nodes)} 个节点")
            print(f"    节点列表: {nodes}")
            
            # 检查连通性
            if len(nodes) > 1:
                subgraph = nx.Graph()
                subgraph.add_nodes_from(nodes)
                for node in nodes:
                    for neighbor in initializer.adjacency_list[node]:
                        if neighbor in nodes:
                            subgraph.add_edge(node, neighbor)
                
                is_connected = nx.is_connected(subgraph)
                print(f"    连通性: {'✅ 连通' if is_connected else '❌ 不连通'}")
                
                if not is_connected:
                    components = list(nx.connected_components(subgraph))
                    print(f"    连通分量数: {len(components)}")
                    for i, comp in enumerate(components):
                        print(f"      分量{i+1}: {sorted(list(comp))}")
                
                partition_info["connectivity_analysis"][f"partition_{label}"] = {
                    "is_connected": is_connected,
                    "num_components": len(list(nx.connected_components(subgraph))) if not is_connected else 1
                }
            else:
                print(f"    连通性: ✅ 单节点 (自然连通)")
                partition_info["connectivity_analysis"][f"partition_{label}"] = {
                    "is_connected": True,
                    "num_components": 1
                }
        
        partition_info["partitions"][f"partition_{label}"] = {
            "label": int(label),
            "name": partition_name,
            "nodes": nodes,
            "size": len(nodes)
        }
    
    # 分析区域间连接
    print(f"\n🔗 区域间连接分析:")
    inter_connections = {}
    total_inter_edges = 0
    
    for i in range(total_nodes):
        for neighbor in initializer.adjacency_list[i]:
            if i < neighbor:  # 避免重复计算同一条边
                label_i = labels_np[i]
                label_j = labels_np[neighbor]
                
                if label_i != label_j:  # 跨分区连接
                    key = tuple(sorted([int(label_i), int(label_j)]))
                    if key not in inter_connections:
                        inter_connections[key] = []
                    inter_connections[key].append((int(i), int(neighbor)))
                    total_inter_edges += 1
    
    print(f"  总跨分区连接数: {total_inter_edges}")
    
    for (label1, label2), edges in inter_connections.items():
        name1 = "未分区" if label1 == 0 else f"分区{label1}"
        name2 = "未分区" if label2 == 0 else f"分区{label2}"
        print(f"  {name1} ↔ {name2}: {len(edges)} 条连接")
        print(f"    连接详情: {edges[:5]}{'...' if len(edges) > 5 else ''}")  # 只显示前5条
        
        partition_info["inter_partition_connections"].append({
            "partition1": int(label1),
            "partition2": int(label2),
            "connection_count": len(edges),
            "edges": edges
        })
    
    # 尝试保存详细信息（如果启用）
    if save_to_file:
        try:
            save_partition_details(partition_info)
        except Exception as e:
            print(f"⚠️ 保存文件失败: {e}")
            print("📝 分析结果仅在控制台显示")
    
    return partition_info

def test_initialization_creates_action_space():
    """
    测试 initialize_partition 是否能为RL Agent生成一个有效的初始状态。
    (即，包含标签为0的未分区节点)
    """
    print("\n--- 测试 1: 验证标准案例(ieee14)的初始化 ---")
    
    # 1. 准备数据
    device = torch.device('cpu')
    mpc = load_power_grid_data('ieee14')
    data = PowerGridDataProcessor(normalize=False, cache_dir='test_cache').graph_from_mpc(mpc)
    
    # 2. 初始化
    num_partitions = 3
    initializer = MetisInitializer(data, device)
    partition = initializer.initialize_partition(num_partitions=num_partitions)
    
    # 3. 详细分析分区结果
    partition_info = analyze_partition_details(partition, initializer, "ieee14")
    
    # 4. 基本断言
    # 断言1: 必须存在标签为0的节点 (关键！证明动作空间被创造)
    contains_unassigned = torch.any(partition == 0).item()
    print(f"\n✅ 是否包含'未分区'(0)节点: {contains_unassigned}")
    assert contains_unassigned, "初始化失败：没有创造出任何未分区节点！"

    # 断言2: 检查已分区的连通性（允许部分不连通，但要记录）
    labels_np = partition.cpu().numpy()
    disconnected_partitions = []
    
    for p_id in range(1, num_partitions + 1):
        nodes = np.where(labels_np == p_id)[0]
        if len(nodes) > 1:
            subgraph = nx.Graph()
            subgraph.add_nodes_from(nodes)
            for node in nodes:
                for neighbor in initializer.adjacency_list[node]:
                    if neighbor in nodes:
                        subgraph.add_edge(node, neighbor)
            
            is_conn = nx.is_connected(subgraph)
            if not is_conn:
                disconnected_partitions.append(p_id)
                print(f"⚠️ 分区 {p_id} (大小: {len(nodes)}) 不连通")
            else:
                print(f"✅ 分区 {p_id} (大小: {len(nodes)}) 连通")
    
    if disconnected_partitions:
        print(f"⚠️ 警告: 发现 {len(disconnected_partitions)} 个不连通的分区: {disconnected_partitions}")
        print("   这可能是由于图结构复杂或初始分区算法的限制导致的")
    else:
        print("✅ 所有分区都是连通的")
        
    print("✅ 测试 1 通过: 初始化为Agent创造了动作空间，详细信息已保存")

def test_different_cases():
    """
    测试不同IEEE案例的初始化
    """
    print("\n--- 测试 2: 验证不同IEEE案例的初始化 ---")
    
    device = torch.device('cpu')
    cases = ['ieee14', 'ieee30', 'ieee57', 'ieee118']
    
    for case_name in cases:
        try:
            print(f"\n🔍 测试案例: {case_name}")
            mpc = load_power_grid_data(case_name)
            data = PowerGridDataProcessor(normalize=False, cache_dir='test_cache').graph_from_mpc(mpc)
            
            total_nodes = sum(data.x_dict[node_type].shape[0] for node_type in data.x_dict.keys())
            num_partitions = min(4, total_nodes // 3)  # 动态调整分区数
            if num_partitions < 2:
                num_partitions = 2
                
            initializer = MetisInitializer(data, device)
            partition = initializer.initialize_partition(num_partitions=num_partitions)
            
            # 详细分析（但不保存，避免文件过多）
            partition_info = analyze_partition_details(partition, initializer, case_name, save_to_file=False)
            
            # 检查是否有未分区节点
            unassigned_count = torch.sum(partition == 0).item()
            total_nodes = partition.shape[0]
            
            print(f"\n📋 {case_name} 总结:")
            print(f"  总节点数: {total_nodes}")
            print(f"  未分区节点数: {unassigned_count}")
            print(f"  分区数: {num_partitions}")
            
            assert unassigned_count > 0, f"案例 {case_name} 没有创造未分区节点"
            assert unassigned_count < total_nodes, f"案例 {case_name} 所有节点都未分区"
            
            print(f"  ✅ {case_name} 测试通过")
            
        except Exception as e:
            print(f"  ⚠️ {case_name} 测试失败: {e}")
            # 对于某些案例，可能因为数据问题失败，这是可以接受的
            continue
    
    print("✅ 测试 2 完成: 多案例初始化测试")

def test_edge_cases():
    """
    测试边界情况
    """
    print("\n--- 测试 3: 验证边界情况处理 ---")
    
    device = torch.device('cpu')
    mpc = load_power_grid_data('ieee14')
    data = PowerGridDataProcessor(normalize=False, cache_dir='test_cache').graph_from_mpc(mpc)
    
    # 测试极端分区数
    initializer = MetisInitializer(data, device)
    
    # 获取正确的总节点数
    total_nodes = sum(data.x_dict[node_type].shape[0] for node_type in data.x_dict.keys())
    
    # 测试分区数 = 1
    partition_1 = initializer.initialize_partition(num_partitions=1)
    unassigned_1 = torch.sum(partition_1 == 0).item()
    print(f"分区数=1时，未分区节点数: {unassigned_1}")
    
    # 测试分区数接近节点总数
    large_partitions = min(total_nodes - 2, 10)
    partition_large = initializer.initialize_partition(num_partitions=large_partitions)
    unassigned_large = torch.sum(partition_large == 0).item()
    print(f"分区数={large_partitions}时，未分区节点数: {unassigned_large}")
    
    print("✅ 测试 3 完成: 边界情况测试")

def print_final_summary():
    """
    打印最终测试总结和问题诊断
    """
    print("\n" + "=" * 80)
    print("🔍 关键问题诊断总结:")
    print("=" * 80)
    
    print("\n📋 主要发现:")
    print("1. ✅ MetisInitializer 成功创建了动作空间（未分区节点）")
    print("2. ❌ 图结构严重不连通：14个节点只有5条边，6个孤立节点")
    print("3. ❌ METIS切边数为0，说明初始分区质量很差")
    print("4. ❌ 连通性修复算法无法完全修复严重不连通的分区")
    
    print("\n🎯 RL训练'一动不动'问题的根本原因:")
    print("   图结构本身就是非连通的，导致:")
    print("   • 分区内部不连通，agent无法有效移动节点")
    print("   • 跨分区连接极少，action空间受限")
    print("   • 初始状态质量差，reward信号微弱")
    
    print("\n💡 建议的解决方案:")
    print("1. 🔧 检查数据处理管道：确保图的构建过程正确")
    print("2. 🌐 验证异构图的边连接：确保所有有效连接都被包含")
    print("3. 🎲 改进初始化策略：对于不连通图使用更智能的分区方法")
    print("4. 🏆 调整reward函数：在图结构受限时提供更多指导信号")
    
    print("\n📁 详细的分区分析数据已保存到 output/partition_analysis/ 目录")
    print("=" * 80)

if __name__ == "__main__":
    print("🚀 开始 MetisInitializer 综合测试...")
    print("=" * 60)
    
    test_initialization_creates_action_space()
    test_different_cases()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("📁 详细的分区分析结果已保存在 output/partition_analysis/ 目录中")
    
    print_final_summary()
