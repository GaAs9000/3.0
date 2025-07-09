"""
边界节点计算性能对比测试

比较向量化实现与循环实现的性能差异
"""

import torch
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rl.state import StateManager
from torch_geometric.data import HeteroData


def compute_boundary_nodes_loop(current_partition, adjacency_list):
    """原始的循环实现"""
    boundary_set = set()
    
    for node_idx in range(len(adjacency_list)):
        node_partition = current_partition[node_idx].item()
        
        # 检查是否有邻居在不同分区中
        for neighbor_idx in adjacency_list[node_idx]:
            neighbor_partition = current_partition[neighbor_idx].item()
            if neighbor_partition != node_partition:
                boundary_set.add(node_idx)
                break
                
    return torch.tensor(list(boundary_set), dtype=torch.long)


def compute_boundary_nodes_vectorized(current_partition, edge_index):
    """向量化实现"""
    if edge_index.shape[1] > 0:
        src, dst = edge_index[0], edge_index[1]
        
        # 检查哪些边连接不同分区的节点
        diff_mask = current_partition[src] != current_partition[dst]
        
        # 找到所有涉及跨区连接的节点
        boundary_nodes = torch.unique(
            torch.cat([src[diff_mask], dst[diff_mask]])
        )
        
        return boundary_nodes
    else:
        return torch.empty(0, dtype=torch.long)


def create_test_graph(num_nodes, edge_density=0.05):
    """创建测试图"""
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简单的同构图用于测试
    num_edges = int(num_nodes * num_nodes * edge_density)
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    
    # 去除自环
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    # 创建无向图
    edge_index = torch.unique(torch.cat([
        torch.stack([src, dst]),
        torch.stack([dst, src])
    ], dim=1), dim=1)
    
    # 创建邻接列表（用于循环实现）
    adjacency_list = [[] for _ in range(num_nodes)]
    for s, d in zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()):
        adjacency_list[s].append(d)
    
    return edge_index.to(device), adjacency_list


def performance_comparison():
    """性能对比测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("=" * 60)
    
    node_counts = [100, 500, 1000, 2000, 5000]
    num_partitions = 10
    num_iterations = 20
    
    print(f"{'节点数':>8} | {'循环实现(ms)':>12} | {'向量化实现(ms)':>14} | {'加速比':>8}")
    print("-" * 60)
    
    for num_nodes in node_counts:
        # 创建测试数据
        edge_index, adjacency_list = create_test_graph(num_nodes)
        current_partition = torch.randint(1, num_partitions + 1, (num_nodes,), device=device)
        
        # 测试循环实现
        if device.type == 'cpu' or num_nodes <= 1000:  # GPU上循环太慢，只在小规模或CPU上测试
            loop_times = []
            for _ in range(num_iterations):
                start = time.time()
                _ = compute_boundary_nodes_loop(current_partition, adjacency_list)
                end = time.time()
                loop_times.append(end - start)
            loop_avg = sum(loop_times) / len(loop_times) * 1000
        else:
            loop_avg = float('inf')  # 跳过大规模的循环测试
        
        # 测试向量化实现
        vec_times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = compute_boundary_nodes_vectorized(current_partition, edge_index)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            vec_times.append(end - start)
        vec_avg = sum(vec_times) / len(vec_times) * 1000
        
        # 计算加速比
        if loop_avg != float('inf'):
            speedup = loop_avg / vec_avg
            print(f"{num_nodes:>8} | {loop_avg:>12.2f} | {vec_avg:>14.2f} | {speedup:>8.1f}x")
        else:
            print(f"{num_nodes:>8} | {'N/A':>12} | {vec_avg:>14.2f} | {'N/A':>8}")
    
    print("=" * 60)
    print("注：大规模图的循环实现太慢，已跳过")


def verify_correctness():
    """验证向量化实现的正确性"""
    print("\n验证正确性...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = 100
    num_partitions = 5
    
    # 创建测试数据
    edge_index, adjacency_list = create_test_graph(num_nodes)
    current_partition = torch.randint(1, num_partitions + 1, (num_nodes,), device=device)
    
    # 计算结果
    loop_result = compute_boundary_nodes_loop(current_partition, adjacency_list)
    vec_result = compute_boundary_nodes_vectorized(current_partition, edge_index)
    
    # 比较结果
    loop_set = set(loop_result.cpu().numpy())
    vec_set = set(vec_result.cpu().numpy())
    
    if loop_set == vec_set:
        print("✅ 向量化实现结果正确")
        print(f"   边界节点数量: {len(loop_set)}")
    else:
        print("❌ 向量化实现结果不正确")
        print(f"   循环实现: {sorted(loop_set)}")
        print(f"   向量化实现: {sorted(vec_set)}")
        print(f"   差异: {loop_set.symmetric_difference(vec_set)}")


if __name__ == "__main__":
    print("边界节点计算性能对比")
    print("=" * 60)
    
    # 验证正确性
    verify_correctness()
    
    print("\n性能对比测试")
    print("=" * 60)
    
    # 性能对比
    performance_comparison() 