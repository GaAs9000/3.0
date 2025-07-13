#!/usr/bin/env python3
"""
简化的调试脚本 - 测试基本功能
"""

import sys
import os
import torch

# 添加代码路径
sys.path.append('code/src')

def test_basic_imports():
    """测试基本导入"""
    print("=== 测试基本导入 ===")
    
    try:
        from data_processing import PowerGridDataProcessor
        print("✓ PowerGridDataProcessor 导入成功")
    except ImportError as e:
        print(f"✗ PowerGridDataProcessor 导入失败: {e}")
        return False
    
    try:
        import pandapower.networks as pn
        print("✓ pandapower 导入成功")
    except ImportError as e:
        print(f"✗ pandapower 导入失败: {e}")
        return False
    
    try:
        from torch_geometric.data import HeteroData
        print("✓ PyTorch Geometric 导入成功")
    except ImportError as e:
        print(f"✗ PyTorch Geometric 导入失败: {e}")
        return False
    
    return True

def test_data_processing():
    """测试数据处理"""
    print("\n=== 测试数据处理 ===")
    
    try:
        # 导入必要模块
        from data_processing import PowerGridDataProcessor
        import pandapower.networks as pn
        from pandapower.converter import to_mpc
        
        # 加载IEEE14数据
        net = pn.case14()
        mpc = to_mpc(net)
        print("✓ IEEE14数据加载成功")
        print(f"  - 节点数: {mpc['bus'].shape[0]}")
        print(f"  - 线路数: {mpc['branch'].shape[0]}")
        
        # 创建处理器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = PowerGridDataProcessor(device=device)
        print(f"✓ 数据处理器创建成功 (设备: {device})")
        
        # 处理数据
        config = {'debug': {'training_output': {'only_show_errors': True}}}
        hetero_data = processor.graph_from_mpc(mpc, config)
        print("✓ 数据处理成功")
        print(f"  - HeteroData: {hetero_data}")
        print(f"  - 节点类型: {hetero_data.node_types}")
        print(f"  - 边类型: {hetero_data.edge_types}")
        
        # 测试访问bus数据
        bus_data = hetero_data['bus']
        print(f"✓ Bus数据访问成功: {bus_data.x.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_creation():
    """测试环境创建"""
    print("\n=== 测试环境创建 ===")
    
    try:
        # 导入必要模块
        from data_processing import PowerGridDataProcessor
        from gat import create_production_encoder
        from rl.enhanced_environment import create_enhanced_environment
        import pandapower.networks as pn
        from pandapower.converter import to_mpc
        
        # 准备数据
        net = pn.case14()
        mpc = to_mpc(net)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = PowerGridDataProcessor(device=device)
        config = {'debug': {'training_output': {'only_show_errors': True}}}
        hetero_data = processor.graph_from_mpc(mpc, config).to(device)
        
        # 创建编码器和节点嵌入
        encoder = create_production_encoder(hetero_data)
        encoder = encoder.to(device)
        
        with torch.no_grad():
            node_embeddings = encoder.encode_nodes(hetero_data)
        
        print("✓ 节点嵌入生成成功")
        for node_type, embeddings in node_embeddings.items():
            print(f"  - {node_type}: {embeddings.shape}")
        
        # 创建环境
        env_config = {
            'reward_weights': {},
            'max_steps': 200,
            'device': device,
            'enable_features_cache': True
        }
        
        env = create_enhanced_environment(
            hetero_data=hetero_data,
            node_embeddings=node_embeddings,
            num_partitions=3,
            config=env_config,
            use_enhanced=True
        )
        
        print(f"✓ 环境创建成功: {env.total_nodes}节点, {env.num_partitions}分区")
        return True
        
    except Exception as e:
        print(f"✗ 环境创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔍 开始系统诊断...")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
    
    # 检查工作目录
    print(f"当前工作目录: {os.getcwd()}")
    print(f"代码路径存在: {os.path.exists('code/src')}")
    
    # 运行测试
    tests = [
        ("基本导入", test_basic_imports),
        ("数据处理", test_data_processing),
        ("环境创建", test_environment_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("🎯 测试结果总结")
    print("="*50)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 所有测试通过！系统运行正常。")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return all_passed

if __name__ == "__main__":
    main()
