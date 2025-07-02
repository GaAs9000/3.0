#!/usr/bin/env python3
"""
真实训练测试脚本

快速验证简单相对奖励在实际代码中的效果
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path

def backup_original_reward_file():
    """备份原始奖励文件"""
    reward_file = "code/src/rl/reward.py"
    backup_file = "code/src/rl/reward_backup.py"
    
    if os.path.exists(reward_file) and not os.path.exists(backup_file):
        subprocess.run(["cp", reward_file, backup_file], shell=True)
        print("✅ 已备份原始奖励文件")
    else:
        print("ℹ️ 备份文件已存在或原文件不存在")

def run_quick_training_test(output_dir: str, test_name: str) -> bool:
    """运行快速训练测试"""
    print(f"🚀 运行 {test_name} 测试...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行训练命令
    cmd = [
        "python", "train.py",
        "--mode", "fast",
        "--case", "ieee14",
        "--episodes", "100",  # 快速测试，只用100个episodes
        "--output-dir", output_dir,
        "--save-results"  # 使用正确的参数名
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {test_name} 测试完成 (耗时: {end_time - start_time:.1f}秒)")
            return True
        else:
            print(f"❌ {test_name} 测试失败:")
            print(f"   错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} 测试超时")
        return False
    except Exception as e:
        print(f"❌ {test_name} 测试异常: {e}")
        return False

def analyze_training_results(baseline_dir: str, relative_dir: str):
    """分析训练结果"""
    print("\n📊 分析训练结果...")
    
    # 查找日志文件
    baseline_logs = list(Path(baseline_dir).glob("**/*.json"))
    relative_logs = list(Path(relative_dir).glob("**/*.json"))
    
    print(f"   基线测试日志: {len(baseline_logs)} 个文件")
    print(f"   相对奖励日志: {len(relative_logs)} 个文件")
    
    # 简单分析（如果有日志文件的话）
    if baseline_logs and relative_logs:
        try:
            # 读取第一个日志文件作为示例
            with open(baseline_logs[0], 'r') as f:
                baseline_data = json.load(f)
            with open(relative_logs[0], 'r') as f:
                relative_data = json.load(f)
            
            print("   📈 简单对比:")
            print(f"     基线方法: {baseline_data.get('summary', 'N/A')}")
            print(f"     相对奖励: {relative_data.get('summary', 'N/A')}")
            
        except Exception as e:
            print(f"   ⚠️ 日志分析失败: {e}")
    
    # 检查输出文件
    baseline_files = list(Path(baseline_dir).glob("**/*"))
    relative_files = list(Path(relative_dir).glob("**/*"))
    
    print(f"   📁 输出文件:")
    print(f"     基线测试: {len(baseline_files)} 个文件")
    print(f"     相对奖励: {len(relative_files)} 个文件")

def check_current_implementation():
    """检查当前实现状态"""
    print("🔍 检查当前实现状态...")
    
    reward_file = "code/src/rl/reward.py"
    
    if not os.path.exists(reward_file):
        print("❌ 奖励文件不存在")
        return False
    
    # 检查是否包含相对奖励方法
    with open(reward_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '_compute_simple_relative_reward' in content:
        print("✅ 发现简单相对奖励方法")
        
        # 检查是否在使用
        if 'self._compute_simple_relative_reward' in content:
            print("✅ 相对奖励方法正在被使用")
            return True
        else:
            print("⚠️ 相对奖励方法存在但未被使用")
            return False
    else:
        print("❌ 未发现简单相对奖励方法")
        return False

def main():
    """主函数"""
    print("🧪 真实训练测试")
    print("验证简单相对奖励在实际代码中的效果")
    print("=" * 60)
    
    # 1. 检查当前实现
    if not check_current_implementation():
        print("\n❌ 当前代码中未正确实现简单相对奖励")
        print("💡 请确保已经应用了代码修改:")
        print("   1. 添加了 _compute_simple_relative_reward 方法")
        print("   2. 在 compute_incremental_reward 中使用了该方法")
        return
    
    # 2. 备份原始文件
    backup_original_reward_file()
    
    # 3. 运行快速测试
    print("\n🚀 开始快速训练测试...")
    
    # 当前实现应该已经是相对奖励了
    relative_success = run_quick_training_test(
        "data/experiments/relative_reward_test", 
        "相对奖励"
    )
    
    if not relative_success:
        print("\n❌ 相对奖励测试失败")
        return
    
    # 4. 分析结果
    print("\n📊 测试结果分析:")
    print("   ✅ 相对奖励系统运行正常")
    print("   📁 输出保存在: data/experiments/relative_reward_test/")
    
    # 5. 给出建议
    print("\n💡 下一步建议:")
    print("   1. 运行更长时间的训练 (--episodes 1000+)")
    print("   2. 在不同网络案例上测试 (ieee30, ieee118)")
    print("   3. 对比训练前后的TensorBoard日志")
    print("   4. 分析不同场景下的性能表现")
    
    print("\n✨ 快速测试完成！")
    print("🎯 简单相对奖励系统已成功集成并运行")

if __name__ == "__main__":
    main()
