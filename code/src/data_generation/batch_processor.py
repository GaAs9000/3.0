#!/usr/bin/env python3
"""
批次处理工具

功能：
- 高效批次序列化
- 内存优化加载
- 并行处理支持
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BatchProcessor:
    """批次处理器"""
    
    def __init__(self, 
                 storage_manager,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化批次处理器
        
        Args:
            storage_manager: 存储管理器
            config: 配置字典
        """
        self.storage_manager = storage_manager
        self.config = config or {}
        self.paths = storage_manager.paths
        
        # 配置参数
        self.use_compression = self.config.get('use_compression', True)
        self.max_workers = self.config.get('max_workers', min(8, mp.cpu_count()))
        self.chunk_size = self.config.get('chunk_size', 10)
        
        logger.info(f"批次处理器初始化完成，压缩: {self.use_compression}, 最大工作线程: {self.max_workers}")
    
    def save_phase_batches(self, 
                          phase: str,
                          batch_data: List[List[Any]],
                          parallel: bool = True) -> int:
        """
        保存阶段批次数据
        
        Args:
            phase: 阶段名称
            batch_data: 批次数据列表
            parallel: 是否使用并行处理
            
        Returns:
            成功保存的批次数量
        """
        if not batch_data:
            logger.warning(f"阶段 {phase} 的批次数据为空")
            return 0
        
        logger.info(f"开始保存 {phase} 阶段的 {len(batch_data)} 个批次...")
        
        if parallel and len(batch_data) > self.chunk_size:
            return self._save_batches_parallel(phase, batch_data)
        else:
            return self._save_batches_sequential(phase, batch_data)
    
    def _save_batches_sequential(self, phase: str, batch_data: List[List[Any]]) -> int:
        """顺序保存批次数据"""
        saved_count = 0
        
        for batch_idx, batch in enumerate(batch_data):
            try:
                self.storage_manager.save_batch(
                    batch_data=batch,
                    phase=phase,
                    batch_idx=batch_idx,
                    compression=self.use_compression
                )
                saved_count += 1
                
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"已保存 {batch_idx + 1}/{len(batch_data)} 个批次")
                    
            except Exception as e:
                logger.error(f"保存批次失败: phase={phase}, batch_idx={batch_idx}, 错误: {e}")
        
        logger.info(f"{phase} 阶段保存完成，成功保存 {saved_count}/{len(batch_data)} 个批次")
        return saved_count
    
    def _save_batches_parallel(self, phase: str, batch_data: List[List[Any]]) -> int:
        """并行保存批次数据"""
        saved_count = 0
        
        def save_single_batch(args):
            batch_idx, batch = args
            try:
                self.storage_manager.save_batch(
                    batch_data=batch,
                    phase=phase,
                    batch_idx=batch_idx,
                    compression=self.use_compression
                )
                return True
            except Exception as e:
                logger.error(f"保存批次失败: phase={phase}, batch_idx={batch_idx}, 错误: {e}")
                return False
        
        # 准备任务参数
        tasks = [(batch_idx, batch) for batch_idx, batch in enumerate(batch_data)]
        
        # 使用线程池执行保存任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_batch = {executor.submit(save_single_batch, task): task[0] for task in tasks}
            
            # 收集结果
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    success = future.result()
                    if success:
                        saved_count += 1
                    
                    if (batch_idx + 1) % 50 == 0:
                        logger.info(f"已保存 {saved_count}/{len(batch_data)} 个批次")
                        
                except Exception as e:
                    logger.error(f"并行保存任务异常: batch_idx={batch_idx}, 错误: {e}")
        
        logger.info(f"{phase} 阶段并行保存完成，成功保存 {saved_count}/{len(batch_data)} 个批次")
        return saved_count
    
    def load_phase_data(self, phase: str, epoch: int = 0) -> List[Any]:
        """
        加载指定阶段的数据
        
        Args:
            phase: 阶段名称
            epoch: 阶段内的epoch编号
            
        Returns:
            epoch数据列表
        """
        batch_files = self.paths.list_batch_files(phase)
        
        if not batch_files:
            raise FileNotFoundError(f"阶段 {phase} 没有找到批次文件")
        
        # 计算每个epoch的批次数量
        curriculum_info = self.storage_manager.load_metadata('curriculum_info')
        batches_per_epoch = 32  # 默认值
        
        if curriculum_info and 'phase_details' in curriculum_info:
            phase_detail = curriculum_info['phase_details'].get(phase, {})
            samples_per_epoch = phase_detail.get('expected_samples_per_epoch', 1024)
            batches_per_epoch = samples_per_epoch // 32  # 假设batch_size=32
        
        # 计算需要加载的批次范围
        start_batch = epoch * batches_per_epoch
        end_batch = min(start_batch + batches_per_epoch, len(batch_files))
        
        if start_batch >= len(batch_files):
            raise IndexError(f"epoch {epoch} 超出了 {phase} 阶段的可用数据范围")
        
        # 加载批次数据
        epoch_data = []
        for batch_idx in range(start_batch, end_batch):
            try:
                batch_data = self.storage_manager.load_batch(phase, batch_idx)
                epoch_data.extend(batch_data)  # 展开批次数据
            except Exception as e:
                logger.error(f"加载批次失败: phase={phase}, batch_idx={batch_idx}, 错误: {e}")
        
        logger.debug(f"加载 {phase} 阶段 epoch {epoch} 数据完成，共 {len(epoch_data)} 个样本")
        return epoch_data
    
    def load_batch(self, phase: str, batch_idx: int) -> List[Any]:
        """
        加载单个批次数据
        
        Args:
            phase: 阶段名称
            batch_idx: 批次索引
            
        Returns:
            批次数据
        """
        return self.storage_manager.load_batch(phase, batch_idx)
    
    def create_dataloader(self, phase: str, batch_size: int = 32, shuffle: bool = True):
        """
        创建数据加载器
        
        Args:
            phase: 阶段名称
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            数据加载器
        """
        batch_files = self.paths.list_batch_files(phase)
        
        if not batch_files:
            raise FileNotFoundError(f"阶段 {phase} 没有找到批次文件")
        
        # 创建自定义数据集
        dataset = PretrainDataset(
            storage_manager=self.storage_manager,
            phase=phase,
            batch_files=batch_files
        )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(4, mp.cpu_count()),
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn
        )
        
        return dataloader
    
    def _collate_fn(self, batch):
        """自定义collate函数"""
        # 这里可以根据需要自定义批次数据的整理方式
        return batch
    
    def get_batch_stats(self, phase: str) -> Dict[str, Any]:
        """
        获取批次统计信息
        
        Args:
            phase: 阶段名称
            
        Returns:
            批次统计信息
        """
        batch_files = self.paths.list_batch_files(phase)
        
        if not batch_files:
            return {
                'batch_count': 0,
                'total_size_mb': 0.0,
                'avg_size_mb': 0.0
            }
        
        total_size = 0
        sample_counts = []
        
        # 统计前10个批次的样本数量
        sample_batch_files = batch_files[:10]
        
        for file_path in sample_batch_files:
            try:
                # 文件大小
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # 加载批次检查样本数量
                batch_idx = int(file_path.stem.split('_')[-1])
                batch_data = self.storage_manager.load_batch(phase, batch_idx)
                sample_counts.append(len(batch_data))
                
            except Exception as e:
                logger.warning(f"统计批次文件时出错: {file_path}, 错误: {e}")
        
        # 计算统计信息
        avg_samples_per_batch = sum(sample_counts) / len(sample_counts) if sample_counts else 0
        total_size_mb = sum(f.stat().st_size for f in batch_files) / (1024 * 1024)
        avg_size_mb = total_size_mb / len(batch_files)
        
        return {
            'batch_count': len(batch_files),
            'total_size_mb': round(total_size_mb, 2),
            'avg_size_mb': round(avg_size_mb, 3),
            'avg_samples_per_batch': round(avg_samples_per_batch, 1),
            'sample_batch_count': len(sample_batch_files)
        }
    
    def optimize_batch_storage(self, phase: str) -> Dict[str, Any]:
        """
        优化批次存储
        
        Args:
            phase: 阶段名称
            
        Returns:
            优化结果
        """
        batch_files = self.paths.list_batch_files(phase)
        
        if not batch_files:
            return {'optimized': False, 'message': '没有找到批次文件'}
        
        logger.info(f"开始优化 {phase} 阶段的批次存储...")
        
        optimized_count = 0
        original_size = 0
        optimized_size = 0
        
        for file_path in batch_files:
            try:
                # 记录原始大小
                original_size += file_path.stat().st_size
                
                # 加载数据
                batch_idx = int(file_path.stem.split('_')[-1])
                batch_data = self.storage_manager.load_batch(phase, batch_idx)
                
                # 重新保存（使用压缩）
                self.storage_manager.save_batch(
                    batch_data=batch_data,
                    phase=phase,
                    batch_idx=batch_idx,
                    compression=True
                )
                
                # 记录优化后大小
                optimized_size += file_path.stat().st_size
                optimized_count += 1
                
            except Exception as e:
                logger.error(f"优化批次文件失败: {file_path}, 错误: {e}")
        
        compression_ratio = (original_size - optimized_size) / original_size if original_size > 0 else 0
        
        result = {
            'optimized': True,
            'optimized_count': optimized_count,
            'original_size_mb': round(original_size / (1024 * 1024), 2),
            'optimized_size_mb': round(optimized_size / (1024 * 1024), 2),
            'compression_ratio': round(compression_ratio, 3),
            'space_saved_mb': round((original_size - optimized_size) / (1024 * 1024), 2)
        }
        
        logger.info(f"{phase} 阶段存储优化完成，压缩率: {compression_ratio:.1%}")
        return result


class PretrainDataset(torch.utils.data.Dataset):
    """预训练数据集"""
    
    def __init__(self, storage_manager, phase: str, batch_files: List[Path]):
        self.storage_manager = storage_manager
        self.phase = phase
        self.batch_files = batch_files
        
        # 计算总样本数（估算）
        self.total_samples = len(batch_files) * 32  # 假设每个批次32个样本
        
    def __len__(self):
        return len(self.batch_files)
    
    def __getitem__(self, idx):
        batch_idx = idx
        try:
            batch_data = self.storage_manager.load_batch(self.phase, batch_idx)
            return batch_data
        except Exception as e:
            logger.error(f"加载数据集项失败: phase={self.phase}, idx={idx}, 错误: {e}")
            # 返回一个空的批次
            return []