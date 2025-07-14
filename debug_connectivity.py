#!/usr/bin/env python3
"""
调试连通性安全分区问题
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code', 'src'))

import torch
import numpy as np
from rl.enhanced_environment import EnhancedPowerGridPartitioningEnv
from rl.enhanced_agent import EnhancedPPOAgent
from gat import create_production_encoder
from data_processing import PowerGridDataProcessor
import pandapower.networks as pn
import pandapower as pp
from pandapower.converter import to_mpc

def debug_connectivity_issue():
    """调试连通性问题"""
    print("🔍 调试连通性安全分区问题...")
    
    # 加载IEEE30案例（经常出现问题的案例）
    net = pn.case30()
    pp.runpp(net, init="flat", calculate_voltage_angles=False, numba=False)
    mpc_wrap = to_mpc(net)
    mpc = mpc_wrap["mpc"] if isinstance(mpc_wrap, dict) and "mpc" in mpc_wrap else mpc_wrap
    
    processor = PowerGridDataProcessor()
    hetero_data = processor.graph_from_mpc(mpc)
    device = torch.device('cpu')
    
    # 创建编码器
    encoder = create_production_encoder(hetero_data, {})
    with torch.no_grad():
        node_embeddings = encoder.encode_nodes(hetero_data)
    
    # 创建环境
    env = EnhancedPowerGridPartitioningEnv(
        hetero_data=hetero_data,
        node_embeddings=node_embeddings,
        num_partitions=3,  # 使用K=3测试
        device=device
    )
    
    # 重置环境
    obs, info = env.reset()
    print(f"初始分区: {env.state_manager.current_partition}")
    print(f"边界节点: {env.state_manager.get_boundary_nodes()}")
    
    # 检查每个边界节点的连通性安全分区
    boundary_nodes = env.state_manager.get_boundary_nodes()
    for node in boundary_nodes:
        node_int = int(node.item())
        print(f"\n--- 节点 {node_int} ---")
        
        # 获取当前分区
        current_partition = env.state_manager.current_partition[node_int].item()
        print(f"当前分区: {current_partition}")
        
        # 获取邻居
        neighbors = list(env.nx_graph.neighbors(node_int))
        print(f"邻居节点: {neighbors}")
        
        # 获取邻居分区
        neighbor_partitions = set()
        for neighbor in neighbors:
            neighbor_partition = env.state_manager.current_partition[neighbor].item()
            if neighbor_partition != current_partition and neighbor_partition > 0:
                neighbor_partitions.add(neighbor_partition)
        print(f"邻居分区: {neighbor_partitions}")
        
        # 测试每个邻居分区的连通性
        safe_partitions = []
        for target_partition in neighbor_partitions:
            print(f"  测试移动到分区 {target_partition}...")
            
            # 临时移动节点
            test_partition = env.state_manager.current_partition.clone()
            test_partition[node_int] = target_partition
            
            # 检查连通性
            is_safe = env._check_partition_connectivity(test_partition)
            print(f"    连通性检查: {'✅ 安全' if is_safe else '❌ 不安全'}")
            
            if is_safe:
                safe_partitions.append(target_partition)
        
        print(f"安全分区: {safe_partitions}")
        
        # 使用环境方法获取
        env_safe_partitions = env.get_connectivity_safe_partitions(node_int)
        print(f"环境返回的安全分区: {env_safe_partitions}")
        
        if not env_safe_partitions:
            print(f"❌ 节点 {node_int} 没有安全分区！")
            
            # 进一步调试：检查为什么没有安全分区
            print("  详细分析:")
            for target_partition in neighbor_partitions:
                test_partition = env.state_manager.current_partition.clone()
                test_partition[node_int] = target_partition
                
                # 检查每个分区的连通性
                partition_nodes = defaultdict(set)
                for node_idx in range(len(test_partition)):
                    pid = test_partition[node_idx].item()
                    if pid > 0:
                        partition_nodes[pid].add(node_idx)
                
                print(f"    分区 {target_partition} 移动后:")
                for pid, nodes in partition_nodes.items():
                    is_connected = env._is_connected_subgraph(nodes)
                    print(f"      分区 {pid}: {len(nodes)} 个节点, 连通性: {'✅' if is_connected else '❌'}")
                    if not is_connected:
                        print(f"        节点: {sorted(nodes)}")

if __name__ == "__main__":
    from collections import defaultdict
    debug_connectivity_issue()
