#!/usr/bin/env python3
"""
存储工具模块

功能：
- 预训练数据路径管理
- 数据存储和加载工具
- 目录结构管理
"""

import torch
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class PretrainDataPaths:
    """预训练数据路径管理"""
    
    def __init__(self, data_root: str = "data/pretrain_data"):
        self.data_root = Path(data_root)
        self.datasets_dir = self.data_root / "datasets"
        self.metadata_dir = self.data_root / "metadata"
        self.logs_dir = self.data_root / "logs"
        
        # 三阶段数据路径
        self.early_phase_dir = self.datasets_dir / "early_phase"
        self.balance_phase_dir = self.datasets_dir / "balance_phase"
        self.late_phase_dir = self.datasets_dir / "late_phase"
        self.validation_dir = self.datasets_dir / "validation"
        
        # 元数据文件路径
        self.generation_config_file = self.metadata_dir / "generation_config.json"
        self.curriculum_info_file = self.metadata_dir / "curriculum_info.json"
        self.dataset_stats_file = self.metadata_dir / "dataset_stats.json"
        self.phase_distributions_file = self.metadata_dir / "phase_distributions.json"
        
        # 日志文件路径
        self.generation_log_file = self.logs_dir / "generation.log"
        self.validation_log_file = self.logs_dir / "validation.log"
        self.performance_log_file = self.logs_dir / "performance.log"
        
    def ensure_directories(self):
        """确保所有必要目录存在"""
        directories = [
            self.datasets_dir, self.metadata_dir, self.logs_dir,
            self.early_phase_dir, self.balance_phase_dir, 
            self.late_phase_dir, self.validation_dir
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"数据目录结构已创建: {self.data_root}")
        
    def get_phase_dir(self, phase: str) -> Path:
        """获取指定阶段的目录路径"""
        phase_mapping = {
            'early_phase': self.early_phase_dir,
            'balance_phase': self.balance_phase_dir,
            'late_phase': self.late_phase_dir,
            'validation': self.validation_dir
        }
        
        if phase not in phase_mapping:
            raise ValueError(f"未知的阶段: {phase}，支持的阶段: {list(phase_mapping.keys())}")
            
        return phase_mapping[phase]
    
    def get_batch_file_path(self, phase: str, batch_idx: int) -> Path:
        """获取批次文件路径"""
        phase_dir = self.get_phase_dir(phase)
        if phase == 'validation':
            return phase_dir / f"val_batch_{batch_idx:03d}.pt"
        else:
            return phase_dir / f"train_batch_{batch_idx:03d}.pt"
    
    def list_batch_files(self, phase: str) -> List[Path]:
        """列出指定阶段的所有批次文件"""
        phase_dir = self.get_phase_dir(phase)
        if not phase_dir.exists():
            return []
        
        pattern = "val_batch_*.pt" if phase == 'validation' else "train_batch_*.pt"
        return sorted(phase_dir.glob(pattern))
    
    def get_dataset_size(self, phase: str) -> int:
        """获取指定阶段的数据集大小"""
        batch_files = self.list_batch_files(phase)
        return len(batch_files)
    
    def exists(self) -> bool:
        """检查数据集是否存在"""
        return (self.data_root.exists() and 
                self.datasets_dir.exists() and
                self.metadata_dir.exists())
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        info = {
            'data_root': str(self.data_root),
            'created_at': datetime.now().isoformat(),
            'phases': {}
        }
        
        for phase in ['early_phase', 'balance_phase', 'late_phase', 'validation']:
            phase_dir = self.get_phase_dir(phase)
            batch_files = self.list_batch_files(phase)
            
            info['phases'][phase] = {
                'directory': str(phase_dir),
                'exists': phase_dir.exists(),
                'batch_count': len(batch_files),
                'total_size_mb': sum(f.stat().st_size for f in batch_files) / (1024 * 1024) if batch_files else 0
            }
            
        return info


class StorageManager:
    """存储管理器"""
    
    def __init__(self, paths: PretrainDataPaths):
        self.paths = paths
        
    def save_batch(self, batch_data: List[Any], phase: str, batch_idx: int, 
                   compression: bool = True) -> Path:
        """保存批次数据"""
        file_path = self.paths.get_batch_file_path(phase, batch_idx)
        
        try:
            if compression:
                # 使用高效的序列化和压缩
                torch.save(batch_data, file_path,
                          _use_new_zipfile_serialization=True,
                          pickle_protocol=4)
            else:
                torch.save(batch_data, file_path)
                
            logger.debug(f"批次数据已保存: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存批次数据失败: {file_path}, 错误: {e}")
            raise
    
    def load_batch(self, phase: str, batch_idx: int) -> List[Any]:
        """加载批次数据"""
        file_path = self.paths.get_batch_file_path(phase, batch_idx)
        
        if not file_path.exists():
            raise FileNotFoundError(f"批次文件不存在: {file_path}")
        
        try:
            batch_data = torch.load(file_path, map_location='cpu')
            logger.debug(f"批次数据已加载: {file_path}")
            return batch_data
            
        except Exception as e:
            logger.error(f"加载批次数据失败: {file_path}, 错误: {e}")
            raise
    
    def save_metadata(self, metadata: Dict[str, Any], metadata_type: str):
        """保存元数据"""
        metadata_files = {
            'generation_config': self.paths.generation_config_file,
            'curriculum_info': self.paths.curriculum_info_file,
            'dataset_stats': self.paths.dataset_stats_file,
            'phase_distributions': self.paths.phase_distributions_file
        }
        
        if metadata_type not in metadata_files:
            raise ValueError(f"未知的元数据类型: {metadata_type}")
        
        file_path = metadata_files[metadata_type]
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"元数据已保存: {file_path}")
            
        except Exception as e:
            logger.error(f"保存元数据失败: {file_path}, 错误: {e}")
            raise
    
    def load_metadata(self, metadata_type: str) -> Dict[str, Any]:
        """加载元数据"""
        metadata_files = {
            'generation_config': self.paths.generation_config_file,
            'curriculum_info': self.paths.curriculum_info_file,
            'dataset_stats': self.paths.dataset_stats_file,
            'phase_distributions': self.paths.phase_distributions_file
        }
        
        if metadata_type not in metadata_files:
            raise ValueError(f"未知的元数据类型: {metadata_type}")
        
        file_path = metadata_files[metadata_type]
        
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.debug(f"元数据已加载: {file_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"加载元数据失败: {file_path}, 错误: {e}")
            raise
    
    def clear_phase_data(self, phase: str):
        """清除指定阶段的数据"""
        phase_dir = self.paths.get_phase_dir(phase)
        
        if not phase_dir.exists():
            return
        
        batch_files = self.paths.list_batch_files(phase)
        
        for file_path in batch_files:
            try:
                file_path.unlink()
                logger.debug(f"已删除文件: {file_path}")
            except Exception as e:
                logger.warning(f"删除文件失败: {file_path}, 错误: {e}")
        
        logger.info(f"已清除阶段数据: {phase}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0.0,
            'phases': {}
        }
        
        for phase in ['early_phase', 'balance_phase', 'late_phase', 'validation']:
            batch_files = self.paths.list_batch_files(phase)
            
            phase_stats = {
                'file_count': len(batch_files),
                'size_mb': 0.0,
                'files': []
            }
            
            for file_path in batch_files:
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    phase_stats['size_mb'] += size_mb
                    phase_stats['files'].append({
                        'name': file_path.name,
                        'size_mb': round(size_mb, 2)
                    })
            
            stats['phases'][phase] = phase_stats
            stats['total_files'] += phase_stats['file_count']
            stats['total_size_mb'] += phase_stats['size_mb']
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats