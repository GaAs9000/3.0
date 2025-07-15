#!/usr/bin/env python3
"""
预训练数据集管理器

功能：
- 统一数据集生成流程管理
- 与现有GNNPretrainer集成
- 数据集生命周期管理
- 三阶段课程学习数据集生成和管理
"""

import sys
import os
import torch
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import time

# 添加项目路径
sys.path.append('code/src')

from data_processing import PowerGridDataProcessor
from data_generation.utils.storage_utils import PretrainDataPaths, StorageManager
from data_generation.utils.metadata_utils import MetadataManager
from data_generation.curriculum_generator import CurriculumGenerator
from data_generation.batch_processor import BatchProcessor
from data_generation.dataset_validator import DatasetValidator

logger = logging.getLogger(__name__)


class PretrainDatasetManager:
    """预训练数据集管理器"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 data_root: str = "data/pretrain_data"):
        """
        初始化预训练数据集管理器
        
        Args:
            config: 配置字典
            data_root: 数据根目录
        """
        self.config = config or {}
        self.data_root = data_root
        
        # 初始化路径管理器
        self.paths = PretrainDataPaths(data_root)
        self.paths.ensure_directories()
        
        # 初始化存储管理器
        self.storage_manager = StorageManager(self.paths)
        
        # 初始化元数据管理器
        self.metadata_manager = MetadataManager(self.storage_manager)
        
        # 初始化数据处理器
        self.data_processor = PowerGridDataProcessor()
        
        # 初始化其他组件（延迟初始化）
        self.curriculum_generator = None
        self.batch_processor = None
        self.dataset_validator = None
        
        # 状态跟踪
        self.is_initialized = False
        self.generation_in_progress = False
        
        logger.info(f"预训练数据集管理器初始化完成，数据根目录: {data_root}")
    
    def initialize_components(self):
        """初始化所有组件"""
        if self.is_initialized:
            return
        
        try:
            # 初始化课程学习生成器
            self.curriculum_generator = CurriculumGenerator(
                base_case_loader=self.data_processor.graph_from_mpc,
                config=self.config,
                seed=self.config.get('seed', 42)
            )
            
            # 初始化批次处理器
            self.batch_processor = BatchProcessor(
                storage_manager=self.storage_manager,
                config=self.config
            )
            
            # 初始化数据集验证器
            self.dataset_validator = DatasetValidator(
                storage_manager=self.storage_manager,
                paths=self.paths
            )
            
            self.is_initialized = True
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def has_pregenerated_datasets(self) -> bool:
        """检查是否有预生成的数据集"""
        if not self.paths.exists():
            return False
        
        # 检查三个阶段的数据是否都存在
        required_phases = ['early_phase', 'balance_phase', 'late_phase', 'validation']
        
        for phase in required_phases:
            if self.paths.get_dataset_size(phase) == 0:
                return False
        
        # 检查元数据是否完整
        try:
            metadata_types = ['generation_config', 'curriculum_info', 'dataset_stats']
            for metadata_type in metadata_types:
                metadata = self.storage_manager.load_metadata(metadata_type)
                if not metadata:
                    return False
        except:
            return False
        
        return True
    
    def generate_all_datasets(self, 
                             samples_per_epoch: int = 1024,
                             batches_per_epoch: int = 32,
                             validation_samples: int = 1024,
                             validation_batches: int = 32,
                             force_regenerate: bool = False) -> Dict[str, Any]:
        """
        生成所有数据集
        
        Args:
            samples_per_epoch: 每个epoch的样本数
            batches_per_epoch: 每个epoch的批次数
            validation_samples: 验证样本总数
            validation_batches: 验证批次数
            force_regenerate: 是否强制重新生成
            
        Returns:
            生成结果统计
        """
        if self.generation_in_progress:
            raise RuntimeError("数据生成正在进行中，请等待完成")
        
        # 检查是否需要重新生成
        if not force_regenerate and self.has_pregenerated_datasets():
            logger.info("预生成数据集已存在，跳过生成")
            return self.get_generation_summary()
        
        # 初始化组件
        self.initialize_components()
        
        self.generation_in_progress = True
        generation_start_time = time.time()
        
        try:
            logger.info("开始生成完整的预训练数据集...")
            
            # 创建生成配置元数据
            generation_config = {
                'samples_per_epoch': samples_per_epoch,
                'batches_per_epoch': batches_per_epoch,
                'validation_samples': validation_samples,
                'validation_batches': validation_batches,
                'force_regenerate': force_regenerate,
                'generation_start_time': datetime.now().isoformat()
            }
            
            self.metadata_manager.create_generation_config(generation_config)
            self.metadata_manager.create_curriculum_info(total_epochs=50)
            self.metadata_manager.create_phase_distributions()
            
            # 生成各阶段数据
            phases = ['early_phase', 'balance_phase', 'late_phase']
            generation_results = {}
            
            for phase in phases:
                logger.info(f"开始生成 {phase} 阶段数据...")
                phase_start_time = time.time()
                
                # 生成阶段数据
                phase_batches = self.curriculum_generator.generate_phase_data(
                    phase=phase,
                    samples_per_epoch=samples_per_epoch,
                    batches_per_epoch=batches_per_epoch
                )
                
                # 保存阶段数据
                saved_count = self.batch_processor.save_phase_batches(
                    phase=phase,
                    batch_data=phase_batches
                )
                
                phase_duration = time.time() - phase_start_time
                
                # 更新阶段统计信息
                phase_stats = {
                    'batches_generated': len(phase_batches),
                    'batches_saved': saved_count,
                    'generation_time_seconds': phase_duration,
                    'samples_per_epoch': samples_per_epoch,
                    'batches_per_epoch': batches_per_epoch
                }
                
                self.metadata_manager.update_dataset_stats(phase, phase_stats)
                generation_results[phase] = phase_stats
                
                logger.info(f"{phase} 阶段完成，用时: {phase_duration:.2f}秒")
            
            # 生成验证数据
            logger.info("开始生成验证数据...")
            validation_start_time = time.time()
            
            validation_batches = self.curriculum_generator.generate_validation_data(
                validation_samples=validation_samples,
                validation_batches=validation_batches
            )
            
            # 保存验证数据
            saved_validation_count = self.batch_processor.save_phase_batches(
                phase='validation',
                batch_data=validation_batches
            )
            
            validation_duration = time.time() - validation_start_time
            
            # 更新验证数据统计信息
            validation_stats = {
                'batches_generated': len(validation_batches),
                'batches_saved': saved_validation_count,
                'generation_time_seconds': validation_duration,
                'total_samples': validation_samples,
                'total_batches': validation_batches
            }
            
            self.metadata_manager.update_dataset_stats('validation', validation_stats)
            generation_results['validation'] = validation_stats
            
            logger.info(f"验证数据完成，用时: {validation_duration:.2f}秒")
            
            # 完成统计信息
            final_stats = self.metadata_manager.finalize_dataset_stats()
            
            total_duration = time.time() - generation_start_time
            
            # 验证生成的数据集
            logger.info("开始验证数据集...")
            validation_results = self.dataset_validator.validate_all_datasets()
            
            # 汇总结果
            summary = {
                'success': True,
                'total_generation_time_seconds': total_duration,
                'phases_results': generation_results,
                'validation_results': validation_results,
                'final_stats': final_stats,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"数据集生成完成！总用时: {total_duration:.2f}秒")
            
            return summary
            
        except Exception as e:
            logger.error(f"数据集生成失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
            
        finally:
            self.generation_in_progress = False
    
    def load_epoch_data(self, epoch: int) -> List[Any]:
        """
        加载指定epoch的数据
        
        Args:
            epoch: epoch编号
            
        Returns:
            epoch数据列表
        """
        if not self.has_pregenerated_datasets():
            raise RuntimeError("预生成数据集不存在，请先生成数据集")
        
        # 根据epoch确定阶段
        if epoch < 15:
            phase = 'early_phase'
            phase_epoch = epoch
        elif epoch < 35:
            phase = 'balance_phase'
            phase_epoch = epoch - 15
        else:
            phase = 'late_phase'
            phase_epoch = epoch - 35
        
        # 加载阶段数据
        return self.load_phase_data(phase, phase_epoch)
    
    def load_phase_data(self, phase: str, epoch: int = 0) -> List[Any]:
        """
        加载指定阶段的数据
        
        Args:
            phase: 阶段名称
            epoch: 阶段内的epoch编号
            
        Returns:
            阶段数据列表
        """
        if not self.is_initialized:
            self.initialize_components()
        
        return self.batch_processor.load_phase_data(phase, epoch)
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """获取生成摘要"""
        if not self.has_pregenerated_datasets():
            return {
                'exists': False,
                'message': '预生成数据集不存在'
            }
        
        try:
            # 获取元数据摘要
            metadata_summary = self.metadata_manager.get_metadata_summary()
            
            # 获取存储统计信息
            storage_stats = self.storage_manager.get_storage_stats()
            
            # 获取存储信息
            storage_info = self.paths.get_storage_info()
            
            return {
                'exists': True,
                'metadata_summary': metadata_summary,
                'storage_stats': storage_stats,
                'storage_info': storage_info,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'exists': True,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def validate_datasets(self) -> Dict[str, Any]:
        """验证数据集完整性"""
        if not self.is_initialized:
            self.initialize_components()
        
        return self.dataset_validator.validate_all_datasets()
    
    def clear_all_datasets(self):
        """清除所有数据集"""
        logger.warning("清除所有预生成数据集...")
        
        phases = ['early_phase', 'balance_phase', 'late_phase', 'validation']
        
        for phase in phases:
            self.storage_manager.clear_phase_data(phase)
        
        logger.info("所有数据集已清除")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            'has_pregenerated': self.has_pregenerated_datasets(),
            'data_root': str(self.paths.data_root),
            'paths_info': self.paths.get_storage_info(),
            'components_initialized': self.is_initialized,
            'generation_in_progress': self.generation_in_progress
        }
        
        if self.has_pregenerated_datasets():
            info['generation_summary'] = self.get_generation_summary()
        
        return info