#!/usr/bin/env python3
"""
路径辅助工具 - 处理时间戳目录和latest链接
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import glob


def get_latest_timestamp_dir(base_dir: str) -> Optional[str]:
    """
    获取指定目录下最新的时间戳目录
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        最新时间戳目录的完整路径，如果没有找到则返回None
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # 查找所有时间戳格式的目录 (YYYY-MM-DD-HH-mm-ss)
    timestamp_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and len(item.name) == 19 and item.name.count('-') == 5:
            try:
                # 验证是否为有效的时间戳格式
                parts = item.name.split('-')
                if (len(parts) == 6 and 
                    len(parts[0]) == 4 and  # year
                    len(parts[1]) == 2 and  # month
                    len(parts[2]) == 2 and  # day
                    len(parts[3]) == 2 and  # hour
                    len(parts[4]) == 2 and  # minute
                    len(parts[5]) == 2):    # second
                    timestamp_dirs.append(item)
            except:
                continue
    
    if not timestamp_dirs:
        return None
    
    # 按时间戳排序，返回最新的
    latest_dir = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
    return str(latest_dir)


def resolve_latest_path(config_path: str) -> str:
    """
    解析包含'latest'的配置路径为实际的时间戳路径
    
    Args:
        config_path: 配置中的路径，可能包含'latest'
        
    Returns:
        解析后的实际路径
    """
    if 'latest' not in config_path:
        return config_path
    
    # 分解路径
    parts = config_path.split('/')
    if 'latest' in parts:
        latest_index = parts.index('latest')
        base_dir = '/'.join(parts[:latest_index])
        
        # 获取最新的时间戳目录
        latest_timestamp_dir = get_latest_timestamp_dir(base_dir)
        if latest_timestamp_dir:
            # 替换latest为实际的时间戳目录名
            timestamp_name = Path(latest_timestamp_dir).name
            parts[latest_index] = timestamp_name
            return '/'.join(parts)
    
    return config_path


def create_timestamp_directory(base_dir: str, subdirs: list = None) -> str:
    """
    创建新的时间戳目录
    
    Args:
        base_dir: 基础目录
        subdirs: 要创建的子目录列表
        
    Returns:
        创建的时间戳目录路径
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    timestamp_dir = Path(base_dir) / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    
    if subdirs:
        for subdir in subdirs:
            (timestamp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return str(timestamp_dir)


def ensure_directory_exists(path: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_all_timestamp_dirs(base_dir: str) -> list:
    """
    获取指定目录下所有的时间戳目录
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        时间戳目录列表，按时间排序
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    timestamp_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and len(item.name) == 19 and item.name.count('-') == 5:
            timestamp_dirs.append(str(item))
    
    return sorted(timestamp_dirs, key=lambda x: Path(x).name)


def cleanup_old_timestamps(base_dir: str, keep_count: int = 5) -> None:
    """
    清理旧的时间戳目录，只保留最新的几个
    
    Args:
        base_dir: 基础目录路径
        keep_count: 保留的目录数量
    """
    timestamp_dirs = get_all_timestamp_dirs(base_dir)
    
    if len(timestamp_dirs) > keep_count:
        dirs_to_remove = timestamp_dirs[:-keep_count]
        for dir_path in dirs_to_remove:
            try:
                shutil.rmtree(dir_path)
                print(f"已删除旧的时间戳目录: {dir_path}")
            except Exception as e:
                print(f"删除目录失败 {dir_path}: {e}")
