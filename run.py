#!/usr/bin/env python3
"""
IEEE57 电力系统分区训练脚本
IEEE57 Power System Partitioning Training Script

默认运行: IEEE57 系统，4个分区，3000次训练
Default run: IEEE57 system, 4 partitions, 3000 episodes

使用方法:
- python run.py                    # 运行 IEEE57 训练 (默认)
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """执行 IEEE57 训练命令"""
    # 确保在正确的目录下运行
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # 固定使用 ieee57 模式
    mode = "ieee57"

    # 构建命令
    command = ["python", "train.py", "--mode", mode, "--run"]

    print("🚀 正在启动 IEEE57 电力系统分区训练...")
    print("📊 配置: IEEE57 系统 | 4个分区 | 3000次训练")
    print(f"📝 命令: {' '.join(command)}")
    print("=" * 50)

    try:
        # 运行命令并实时显示输出
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出结果
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print("\n✅ 训练完成!")
        else:
            print(f"\n❌ 训练失败，退出码: {return_code}")
            sys.exit(return_code)
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断训练")
        process.terminate()
        sys.exit(1)
    except FileNotFoundError:
        print("❌ 错误: 找不到 train.py 文件")
        print("请确保在正确的项目目录下运行此脚本")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 