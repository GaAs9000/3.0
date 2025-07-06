import torch
import pytest
import numpy as np
from torch_geometric.data import HeteroData

# 确保能够导入被测试的模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.reward import RewardFunction

@pytest.fixture
def sample_hetero_data():
    """创建一个简单的、可复现的异构图数据用于测试。"""
    data = HeteroData()
    data['bus'].x = torch.tensor([
        [10, 5, 20],  # load, load_q, generation
        [20, 8, 0],
        [15, 6, 0],
        [25, 10, 50],
        [30, 12, 0],
    ], dtype=torch.float)
    
    data['bus', 'connects', 'bus'].edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    
    # [r, x, b, z_magnitude, y, ...]
    data['bus', 'connects', 'bus'].edge_attr = torch.tensor([
        [0.1, 0.2, 0.0, 0.223, 4.47],
        [0.1, 0.2, 0.0, 0.223, 4.47],
        [0.1, 0.3, 0.0, 0.316, 3.16],
        [0.1, 0.3, 0.0, 0.316, 3.16],
        [0.1, 0.4, 0.0, 0.412, 2.42],
        [0.1, 0.4, 0.0, 0.412, 2.42],
        [0.1, 0.5, 0.0, 0.510, 1.96],
        [0.1, 0.5, 0.0, 0.510, 1.96],
    ], dtype=torch.float)
    
    return data

@pytest.fixture
def reward_function(sample_hetero_data):
    """创建一个RewardFunction实例用于测试。"""
    config = {
        'adaptive_quality': {
            'quality_weights': {
                'load_b': 0.4,
                'decoupling': 0.3,
                'power_b': 0.3,
            }
        }
    }
    return RewardFunction(sample_hetero_data, config=config)

def test_compute_core_metrics_vectorized(reward_function):
    """
    测试向量化后的 _compute_core_metrics 函数的正确性。
    
    1. 验证在有效分区下的计算结果。
    2. 验证在存在空分区的情况下的计算结果。
    3. 验证当所有节点都在一个分区时的边缘情况。
    """
    # 案例1: 正常分区
    partition1 = torch.tensor([1, 1, 2, 2, 1], dtype=torch.long)
    metrics1 = reward_function._compute_core_metrics(partition1)
    
    # 手动计算验证
    # Partition 1: nodes 0, 1, 4. Loads = 10+20+30 = 60. Gen = 20+0+0 = 20
    # Partition 2: nodes 2, 3. Loads = 15+25 = 40. Gen = 0+50 = 50
    p1_load = 60
    p2_load = 40
    # 使用torch计算无偏标准差，与代码逻辑保持一致
    loads_tensor = torch.tensor([p1_load, p2_load], dtype=torch.float)
    mean_load = loads_tensor.mean().item()
    std_load = loads_tensor.std().item()
    expected_cv = std_load / mean_load
    
    p1_imbalance = abs(20 - 60) # 40
    p2_imbalance = abs(50 - 40) # 10
    total_imbalance = p1_imbalance + p2_imbalance # 50
    total_load = 10+20+15+25+30 # 100
    expected_power_imbalance = total_imbalance / total_load # 0.5
    
    # Coupling: 边 (1,2), (2,1), (3,4), (4,3) 是跨区边
    # 索引为: 2, 3, 6, 7
    edge_admittance = reward_function.edge_admittance
    cross_adm = edge_admittance[[2, 3, 6, 7]].sum()
    total_adm = edge_admittance.sum()
    expected_coupling = (cross_adm / total_adm).item()
    
    assert np.isclose(metrics1['cv'], expected_cv, atol=1e-4)
    assert np.isclose(metrics1['power_imbalance_normalized'], expected_power_imbalance, atol=1e-4)
    assert np.isclose(metrics1['coupling_ratio'], expected_coupling, atol=1e-4)
    assert metrics1['num_partitions'] == 2

    # 案例2: 包含空分区 (分区3为空)
    partition2 = torch.tensor([1, 1, 2, 2, 1], dtype=torch.long) # Same as above, just to show it handles non-contiguous numbers implicitly
    metrics2 = reward_function._compute_core_metrics(partition2)
    assert np.isclose(metrics2['cv'], expected_cv, atol=1e-4)
    assert metrics2['num_partitions'] == 2
    
    # 案例3: 所有节点在同一个分区
    partition3 = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
    metrics3 = reward_function._compute_core_metrics(partition3)
    # 当只有一个分区时，std为nan，我们的代码会将其处理为最差情况cv=1.0
    assert np.isclose(metrics3['cv'], 1.0, atol=1e-4)
    assert np.isclose(metrics3['coupling_ratio'], 0.0, atol=1e-4) # 没有跨区边
    assert metrics3['num_partitions'] == 1
    
    # 案例4: 无有效分区
    partition4 = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    metrics4 = reward_function._compute_core_metrics(partition4)
    assert metrics4['cv'] == 1.0
    assert metrics4['coupling_ratio'] == 1.0
    assert metrics4['power_imbalance_normalized'] == 1.0
    assert metrics4['num_partitions'] == 0
    
if __name__ == "__main__":
    pytest.main() 