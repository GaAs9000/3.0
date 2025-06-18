# 系统配置文件
"""
Power Grid Partition System Configuration
电网分区系统配置文件
"""

# 分区配置
PARTITION_CONFIG = {
    # 分区数量（核心参数）
    'num_regions': 5,  # 修改这里可以改变分区数量
    
    # 环境配置
    'enable_physics_constraints': True,
    'device': 'auto',  # 'cpu', 'cuda', 'auto'
    
    # 训练配置
    'quick_mode_episodes': 100,
    'full_mode_episodes': 2000,
    'random_seed': 42,
    
    # 网络配置
    'gat_hidden_dim': 64,
    'gat_num_layers': 3,
    'gat_heads': 8,
    
    # PPO配置
    'actor_hidden_dim': 256,
    'critic_hidden_dim': 256,
    'learning_rate_actor': 3e-4,
    'learning_rate_critic': 1e-3,
    
    # 奖励权重（结合稠密奖励并强化连通性）
    'reward_weights': {
        'balance': 0.20,              # 负荷均衡
        'coupling': 0.15,             # 解耦度
        'connectivity': 0.40,         # 连通性（最重要）
        'neighbor_consistency': 0.15, # 邻居一致性（稠密奖励）
        'size_balance': 0.05,         # 区域大小平衡（稠密奖励）
        'node_importance': 0.05,      # 关键节点（稠密奖励）
    }
}

# 方便访问的常量
NUM_REGIONS = PARTITION_CONFIG['num_regions']
RANDOM_SEED = PARTITION_CONFIG['random_seed']

def get_config():
    """获取配置字典"""
    return PARTITION_CONFIG.copy()

def update_config(**kwargs):
    """更新配置参数"""
    PARTITION_CONFIG.update(kwargs)
    global NUM_REGIONS
    NUM_REGIONS = PARTITION_CONFIG['num_regions']

def print_config():
    """打印当前配置"""
    print("📋 当前系统配置:")
    print("=" * 50)
    print(f"🎯 分区数量: {PARTITION_CONFIG['num_regions']}")
    print(f"🔧 物理约束: {PARTITION_CONFIG['enable_physics_constraints']}")
    print(f"🎲 随机种子: {PARTITION_CONFIG['random_seed']}")
    print(f"💻 设备: {PARTITION_CONFIG['device']}")
    print(f"🚀 快速模式回合: {PARTITION_CONFIG['quick_mode_episodes']}")
    print(f"🎯 完整模式回合: {PARTITION_CONFIG['full_mode_episodes']}")
    print("=" * 50)

# 使用示例
if __name__ == "__main__":
    print_config()
    
    # 修改分区数量示例
    print("\n修改分区数量为5:")
    update_config(num_regions=5)
    print_config() 