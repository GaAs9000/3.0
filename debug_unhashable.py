#!/usr/bin/env python3
"""
调试unhashable type错误的脚本
在agent.py的_ppo_epoch方法第867行前添加调试打印
"""

# 读取原始文件
with open('code/src/rl/agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到第867行附近的代码，添加调试打印
debug_code = """                # e) 计算LogProb和熵
                try:
                    # 🔧 DEBUG: 打印关键数据类型
                    print(f"\\n=== DEBUG Episode {i} ===")
                    print(f"actions[i] = {actions[i]}")
                    print(f"type(actions[i]) = {type(actions[i])}")
                    if isinstance(actions[i], (list, tuple)) and len(actions[i]) >= 2:
                        print(f"partition_id = {actions[i][1]}")
                        print(f"type(partition_id) = {type(actions[i][1])}")
                    print(f"available_partitions = {available_partitions}")
                    print(f"type(available_partitions) = {type(available_partitions)}")
                    if available_partitions:
                        print(f"available_partitions[0] type = {type(available_partitions[0])}")
                    print("=== END DEBUG ===\\n")
                    
                    part_action_idx_in_list = available_partitions.index(partition_id)"""

# 替换原有的代码
original_pattern = """                # e) 计算LogProb和熵
                try:
                    part_action_idx_in_list = available_partitions.index(partition_id)"""

if original_pattern in content:
    new_content = content.replace(original_pattern, debug_code)
    
    # 写入修改后的文件
    with open('code/src/rl/agent.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 调试代码已添加到agent.py")
    print("📍 位置: _ppo_epoch方法第867行前")
    print("🔍 将在Episode 10时打印详细的数据类型信息")
else:
    print("❌ 未找到目标代码位置")
    print("请检查agent.py的_ppo_epoch方法") 