#!/usr/bin/env python3
"""
快速缓存清理脚本
专门用于清理PyTorch和系统缓存
"""

import torch
import gc
import os
from pathlib import Path

def quick_clear():
    """快速清理缓存"""
    print("🧹 快速清理缓存...")
    
    # 1. 清理PyTorch GPU缓存
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name()}")
        before = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 1024**2
        print(f"   GPU内存: {before:.1f}MB → {after:.1f}MB")
    else:
        print("   GPU不可用，跳过GPU缓存清理")
    
    # 2. Python垃圾回收
    collected = gc.collect()
    print(f"   垃圾回收: 清理了 {collected} 个对象")
    
    # 3. 检查缓存目录状态
    cache_dirs = ['cache', 'logs', 'checkpoints']
    for dir_name in cache_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file()) / 1024**2
            print(f"   {dir_name}: {size:.1f}MB")
        else:
            print(f"   {dir_name}: 不存在")
    
    print("✅ 快速清理完成！")

if __name__ == "__main__":
    quick_clear() 