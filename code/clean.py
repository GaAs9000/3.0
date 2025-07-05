#!/usr/bin/env python3
"""
训练缓存清理工具

这个脚本用于清理电力网络分区强化学习项目中的各种缓存和临时文件，
包括训练日志、检查点、TensorBoard缓存、模型文件等。

使用方法:
    python clean.py --all                    # 清理所有缓存
    python clean.py --logs                   # 只清理日志
    python clean.py --checkpoints            # 只清理检查点
    python clean.py --tensorboard            # 只清理TensorBoard缓存
    python clean.py --interactive            # 交互模式
"""

import os
import shutil
import argparse
import time
from pathlib import Path
from typing import List, Dict
import subprocess
import psutil


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.total_size_cleaned = 0
        
        # 定义需要清理的目录和文件模式
        self.cache_config = {
            'logs': {
                'paths': ['logs', 'runs'],
                'description': 'TensorBoard日志和训练记录',
                'exclude': []  # 可以排除某些子目录
            },
            'checkpoints': {
                'paths': ['checkpoints', 'models'],
                'description': '模型检查点和保存的模型',
                'exclude': []
            },
            'cache': {
                'paths': ['cache'],
                'description': '数据缓存文件',
                'exclude': []
            },
            'pycache': {
                'paths': ['__pycache__'],
                'description': 'Python字节码缓存',
                'exclude': [],
                'recursive': True  # 递归查找所有__pycache__目录
            },
            'output': {
                'paths': ['output'],
                'description': '输出文件和结果',
                'exclude': []
            },
            'figures': {
                'paths': ['figures'],
                'description': '生成的图表和可视化文件（旧版本）',
                'exclude': []
            },
            'temp': {
                'paths': ['.tmp', 'tmp', 'temp'],
                'description': '临时文件',
                'exclude': []
            },
            'experiments': {
                'paths': ['experiments'],
                'description': '实验数据和结果',
                'exclude': []
            }
        }
        
        # 文件扩展名模式
        self.file_patterns = {
            'model_files': ['*.pt', '*.pth', '*.pkl', '*.pickle'],
            'log_files': ['*.log'],
            'temp_files': ['*.tmp', '*.temp', '*~'],
            'backup_files': ['*.bak', '*.backup'],
            'tensorboard_files': ['events.out.tfevents.*']
        }

    def get_directory_size(self, path: Path) -> int:
        """计算目录大小（字节）"""
        if not path.exists():
            return 0
        
        total_size = 0
        try:
            if path.is_file():
                return path.stat().st_size
            
            for item in path.rglob('*'):
                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                    except (OSError, FileNotFoundError):
                        continue
        except (OSError, PermissionError):
            pass
        
        return total_size

    def format_size(self, size_bytes: int) -> str:
        """格式化文件大小显示"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def kill_tensorboard_processes(self):
        """终止所有TensorBoard进程"""
        killed_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'tensorboard' in proc.info['name'].lower():
                    proc.kill()
                    killed_processes.append(proc.info['pid'])
                elif proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline']).lower()
                    if 'tensorboard' in cmdline:
                        proc.kill()
                        killed_processes.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        if killed_processes:
            print(f"🔄 已终止 {len(killed_processes)} 个TensorBoard进程")
            time.sleep(2)  # 等待进程完全终止
        
        return killed_processes

    def scan_cache_usage(self) -> Dict[str, Dict]:
        """扫描所有缓存使用情况"""
        print("🔍 正在扫描缓存使用情况...")
        
        cache_info = {}
        
        for cache_type, config in self.cache_config.items():
            cache_info[cache_type] = {
                'paths': [],
                'total_size': 0,
                'file_count': 0,
                'description': config['description']
            }
            
            # 检查是否需要递归查找
            if config.get('recursive', False):
                # 递归查找所有匹配的目录
                for path_pattern in config['paths']:
                    for found_path in self.project_root.rglob(path_pattern):
                        if found_path.is_dir():
                            size = self.get_directory_size(found_path)
                            file_count = len(list(found_path.rglob('*'))) if found_path.is_dir() else 1
                            
                            cache_info[cache_type]['paths'].append({
                                'path': found_path,
                                'size': size,
                                'file_count': file_count
                            })
                            cache_info[cache_type]['total_size'] += size
                            cache_info[cache_type]['file_count'] += file_count
            else:
                # 只查找直接路径
                for path_pattern in config['paths']:
                    full_path = self.project_root / path_pattern
                    if full_path.exists():
                        size = self.get_directory_size(full_path)
                        file_count = len(list(full_path.rglob('*'))) if full_path.is_dir() else 1
                        
                        cache_info[cache_type]['paths'].append({
                            'path': full_path,
                            'size': size,
                            'file_count': file_count
                        })
                        cache_info[cache_type]['total_size'] += size
                        cache_info[cache_type]['file_count'] += file_count
        
        # 扫描特定文件模式
        for pattern_type, patterns in self.file_patterns.items():
            if pattern_type not in cache_info:
                cache_info[pattern_type] = {
                    'paths': [],
                    'total_size': 0,
                    'file_count': 0,
                    'description': f'{pattern_type.replace("_", " ").title()}'
                }
            
            for pattern in patterns:
                for file_path in self.project_root.rglob(pattern):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        cache_info[pattern_type]['paths'].append({
                            'path': file_path,
                            'size': size,
                            'file_count': 1
                        })
                        cache_info[pattern_type]['total_size'] += size
                        cache_info[pattern_type]['file_count'] += 1
        
        return cache_info

    def display_cache_summary(self, cache_info: Dict[str, Dict]):
        """显示缓存使用摘要"""
        print("\n📊 缓存使用情况摘要:")
        print("=" * 80)
        
        total_size = 0
        total_files = 0
        
        for cache_type, info in cache_info.items():
            if info['total_size'] > 0:
                print(f"{cache_type.upper():15} | "
                      f"{self.format_size(info['total_size']):>10} | "
                      f"{info['file_count']:>8} 文件 | "
                      f"{info['description']}")
                total_size += info['total_size']
                total_files += info['file_count']
        
        print("=" * 80)
        print(f"{'总计':15} | {self.format_size(total_size):>10} | {total_files:>8} 文件")
        print()

    def clean_cache_type(self, cache_type: str, cache_info: Dict, confirm: bool = True) -> bool:
        """清理指定类型的缓存"""
        if cache_type not in cache_info or cache_info[cache_type]['total_size'] == 0:
            print(f"⚠️  没有找到 {cache_type} 类型的缓存文件")
            return False
        
        info = cache_info[cache_type]
        size_to_clean = info['total_size']
        files_to_clean = info['file_count']
        
        if confirm:
            response = input(f"🗑️  确定要清理 {cache_type} ({self.format_size(size_to_clean)}, {files_to_clean} 文件)? [y/N]: ")
            if response.lower() not in ['y', 'yes', '是']:
                print("❌ 取消清理")
                return False
        
        print(f"🧹 正在清理 {cache_type}...")
        
        cleaned_size = 0
        cleaned_files = 0
        
        for path_info in info['paths']:
            path = path_info['path']
            try:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        print(f"   删除目录: {path.relative_to(self.project_root)}")
                    else:
                        path.unlink()
                        print(f"   删除文件: {path.relative_to(self.project_root)}")
                    
                    cleaned_size += path_info['size']
                    cleaned_files += path_info['file_count']
            except Exception as e:
                print(f"   ⚠️  无法删除 {path}: {e}")
        
        self.total_size_cleaned += cleaned_size
        print(f"✅ {cache_type} 清理完成: {self.format_size(cleaned_size)}, {cleaned_files} 文件")
        return True

    def clean_all_cache(self, confirm: bool = True):
        """清理所有缓存"""
        print("开始全面清理缓存...")

        # 首先终止TensorBoard进程
        self.kill_tensorboard_processes()

        # 扫描缓存
        cache_info = self.scan_cache_usage()

        if confirm:
            self.display_cache_summary(cache_info)
            response = input("确定要清理所有缓存吗? [y/N]: ")
            if response.lower() not in ['y', 'yes', '是']:
                print("取消清理")
                return

        # 清理每种类型的缓存
        for cache_type in cache_info.keys():
            if cache_info[cache_type]['total_size'] > 0:
                self.clean_cache_type(cache_type, cache_info, confirm=False)

        print(f"\n清理完成! 总共释放了 {self.format_size(self.total_size_cleaned)} 的磁盘空间")

    def interactive_cleanup(self):
        """交互式清理模式"""
        print("进入交互式清理模式")
        print("您可以选择要清理的缓存类型")
        
        # 首先终止TensorBoard进程
        self.kill_tensorboard_processes()
        
        while True:
            # 扫描最新的缓存信息
            cache_info = self.scan_cache_usage()
            self.display_cache_summary(cache_info)
            
            print("\n可用选项:")
            print("1. 清理训练日志 (logs)")
            print("2. 清理模型检查点 (checkpoints)")
            print("3. 清理数据缓存 (cache)")
            print("4. 清理Python字节码 (pycache)")
            print("5. 清理输出文件 (output)")
            print("6. 清理图表文件 (figures)")
            print("7. 清理临时文件 (temp)")
            print("8. 清理实验数据 (experiments)")
            print("9. 清理所有缓存")
            print("r. 重新扫描")
            print("0. 退出")
            
            choice = input("\n请选择操作 [0-9,r]: ").strip().lower()
            
            if choice == '0':
                break
            elif choice == '1':
                self.clean_cache_type('logs', cache_info)
            elif choice == '2':
                self.clean_cache_type('checkpoints', cache_info)
            elif choice == '3':
                self.clean_cache_type('cache', cache_info)
            elif choice == '4':
                self.clean_cache_type('pycache', cache_info)
            elif choice == '5':
                self.clean_cache_type('output', cache_info)
            elif choice == '6':
                self.clean_cache_type('figures', cache_info)
            elif choice == '7':
                self.clean_cache_type('temp', cache_info)
            elif choice == '8':
                self.clean_cache_type('experiments', cache_info)
            elif choice == '9':
                self.clean_all_cache()
                break
            elif choice == 'r':
                print("🔄 重新扫描中...")
                continue
            else:
                print("❌ 无效选择，请重试")
            
            print()

    def create_gitignore_backup(self):
        """创建.gitignore备份以防止重要文件被误删"""
        gitignore_path = self.project_root / '.gitignore'
        backup_path = self.project_root / '.gitignore.backup'
        
        if gitignore_path.exists() and not backup_path.exists():
            shutil.copy2(gitignore_path, backup_path)
            print("📋 已创建 .gitignore 备份")

    def quick_clean_pycache(self):
        """快速清理所有__pycache__目录"""
        print("🚀 快速清理所有 __pycache__ 目录...")
        
        pycache_dirs = list(self.project_root.rglob('__pycache__'))
        if not pycache_dirs:
            print("✅ 没有找到 __pycache__ 目录")
            return
        
        total_size = 0
        for pycache_dir in pycache_dirs:
            if pycache_dir.is_dir():
                size = self.get_directory_size(pycache_dir)
                total_size += size
                try:
                    shutil.rmtree(pycache_dir)
                    print(f"   删除: {pycache_dir.relative_to(self.project_root)}")
                except Exception as e:
                    print(f"   ⚠️  无法删除 {pycache_dir}: {e}")
        
        print(f"✅ 清理完成！释放了 {self.format_size(total_size)} 的磁盘空间")
        self.total_size_cleaned += total_size


def main():
    parser = argparse.ArgumentParser(
        description="电力网络分区强化学习项目缓存清理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python cleanup_cache.py --all                    # 清理所有缓存
  python cleanup_cache.py --logs                   # 只清理训练日志
  python cleanup_cache.py --checkpoints            # 只清理模型检查点
  python cleanup_cache.py --interactive            # 交互模式选择清理内容
  python cleanup_cache.py --scan                   # 只扫描不清理
        """
    )
    
    parser.add_argument('--all', action='store_true', help='清理所有缓存')
    parser.add_argument('--logs', action='store_true', help='清理训练日志')
    parser.add_argument('--checkpoints', action='store_true', help='清理模型检查点')
    parser.add_argument('--cache', action='store_true', help='清理数据缓存')
    parser.add_argument('--pycache', action='store_true', help='清理Python字节码缓存')
    parser.add_argument('--output', action='store_true', help='清理输出文件')
    parser.add_argument('--figures', action='store_true', help='清理图表文件')
    parser.add_argument('--temp', action='store_true', help='清理临时文件')
    parser.add_argument('--experiments', action='store_true', help='清理实验数据')
    parser.add_argument('--tensorboard', action='store_true', help='终止TensorBoard进程并清理相关缓存')
    parser.add_argument('--interactive', action='store_true', help='交互式清理模式')
    parser.add_argument('--scan', action='store_true', help='只扫描缓存使用情况，不进行清理')
    parser.add_argument('--force', action='store_true', help='强制清理，不询问确认')
    parser.add_argument('--quick-pycache', action='store_true', help='快速清理所有__pycache__目录')
    parser.add_argument('--project-root', default='.', help='项目根目录路径')
    
    args = parser.parse_args()
    
    # 创建缓存管理器
    cache_manager = CacheManager(args.project_root)
    
    print("🧹 电力网络分区强化学习缓存清理工具")
    print("=" * 50)
    
    # 创建备份
    cache_manager.create_gitignore_backup()
    
    try:
        if args.interactive:
            cache_manager.interactive_cleanup()
        elif args.scan:
            cache_info = cache_manager.scan_cache_usage()
            cache_manager.display_cache_summary(cache_info)
        elif args.all:
            cache_manager.clean_all_cache(confirm=not args.force)
        elif args.tensorboard:
            cache_manager.kill_tensorboard_processes()
            cache_info = cache_manager.scan_cache_usage()
            cache_manager.clean_cache_type('logs', cache_info, confirm=not args.force)
        elif args.quick_pycache:
            cache_manager.quick_clean_pycache()
        else:
            # 扫描缓存信息
            cache_info = cache_manager.scan_cache_usage()
            
            # 根据参数清理指定类型
            cleaned_any = False
            for cache_type in ['logs', 'checkpoints', 'cache', 'pycache', 'output', 'figures', 'temp', 'experiments']:
                if getattr(args, cache_type):
                    cache_manager.clean_cache_type(cache_type, cache_info, confirm=not args.force)
                    cleaned_any = True
            
            if not cleaned_any:
                # 如果没有指定任何清理选项，显示帮助
                cache_manager.display_cache_summary(cache_info)
                print("\n使用 --help 查看可用选项，或使用 --interactive 进入交互模式")

    except KeyboardInterrupt:
        print("\n\n清理操作被用户中断")
    except Exception as e:
        print(f"\nERROR: 清理过程中发生错误: {e}")

    print("\n清理工具退出")


if __name__ == "__main__":
    main() 