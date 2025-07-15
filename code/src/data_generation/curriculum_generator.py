#!/usr/bin/env python3
"""
课程学习数据生成器

功能：
- 三阶段课程学习数据生成
- 与scale_aware_generator集成
- 规模分布配置管理
"""

import sys
import torch
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import copy

# 添加项目路径
sys.path.append('code/src')
sys.path.append('code/src/rl')

from rl.scale_aware_generator import ScaleAwareSyntheticGenerator, GridGenerationConfig
from data_processing import PowerGridDataProcessor

logger = logging.getLogger(__name__)


class CurriculumGenerator:
    """课程学习数据生成器"""
    
    def __init__(self, 
                 base_case_loader,
                 config: Optional[Dict[str, Any]] = None,
                 seed: int = 42):
        """
        初始化课程学习数据生成器
        
        Args:
            base_case_loader: 基础案例加载函数
            config: 配置字典
            seed: 随机种子
        """
        self.base_case_loader = base_case_loader
        self.config = config or {}
        self.seed = seed
        
        # 设置随机种子
        self._set_random_seeds(seed)
        
        # 创建GridGenerationConfig
        self.grid_config = GridGenerationConfig(
            topology_variation_prob=self.config.get('topology_variation_prob', 0.2),
            node_addition_prob=self.config.get('node_addition_prob', 0.1),
            node_removal_prob=self.config.get('node_removal_prob', 0.05),
            edge_addition_prob=self.config.get('edge_addition_prob', 0.1),
            edge_removal_prob=self.config.get('edge_removal_prob', 0.05),
            ensure_connectivity=self.config.get('ensure_connectivity', True),
            min_degree=self.config.get('min_degree', 2),
            max_degree=self.config.get('max_degree', 6)
        )
        
        # 创建数据生成器
        self.generator = ScaleAwareSyntheticGenerator(
            base_case_loader=base_case_loader,
            config=self.grid_config,
            seed=seed
        )
        
        # 阶段配置
        self.phase_configs = {
            'early_phase': {
                'epochs': (0, 15),
                'distribution': {'small': 0.7, 'medium': 0.3, 'large': 0.0},
                'description': '早期阶段：专注小规模数据'
            },
            'balance_phase': {
                'epochs': (15, 35),
                'distribution': {'small': 0.4, 'medium': 0.4, 'large': 0.2},
                'description': '平衡阶段：三种规模平衡发展'
            },
            'late_phase': {
                'epochs': (35, 50),
                'distribution': {'small': 0.2, 'medium': 0.3, 'large': 0.5},
                'description': '后期阶段：重点大规模数据'
            },
            'validation': {
                'epochs': 'all',
                'distribution': {'small': 0.4, 'medium': 0.4, 'large': 0.2},
                'description': '验证集：平衡分布'
            }
        }
        
        logger.info(f"课程学习数据生成器初始化完成，种子: {seed}")
    
    def _set_random_seeds(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def generate_phase_data(self, 
                           phase: str,
                           samples_per_epoch: int = 1024,
                           batches_per_epoch: int = 32) -> List[List[Dict]]:
        """
        生成指定阶段的数据
        
        Args:
            phase: 阶段名称 ('early_phase', 'balance_phase', 'late_phase', 'validation')
            samples_per_epoch: 每个epoch的样本数
            batches_per_epoch: 每个epoch的批次数
            
        Returns:
            List of batches, each batch is a list of grid data
        """
        if phase not in self.phase_configs:
            raise ValueError(f"未知的阶段: {phase}，支持的阶段: {list(self.phase_configs.keys())}")
        
        phase_config = self.phase_configs[phase]
        distribution = phase_config['distribution']
        
        logger.info(f"开始生成 {phase} 阶段数据...")
        logger.info(f"分布配置: {distribution}")
        logger.info(f"每个epoch样本数: {samples_per_epoch}, 批次数: {batches_per_epoch}")
        
        # 计算每个epoch的批次数据
        if phase == 'validation':
            # 验证集只生成一套数据
            epochs_count = 1
            total_samples = samples_per_epoch
        else:
            # 训练集根据阶段epoch范围生成
            start_epoch, end_epoch = phase_config['epochs']
            epochs_count = end_epoch - start_epoch
            total_samples = samples_per_epoch * epochs_count
        
        logger.info(f"生成 {epochs_count} 个epoch的数据，总样本数: {total_samples}")
        
        # 生成所有数据
        all_batches = []
        samples_per_batch = samples_per_epoch // batches_per_epoch
        
        for epoch in range(epochs_count):
            logger.info(f"生成第 {epoch + 1}/{epochs_count} 个epoch的数据...")
            
            # 为当前epoch生成批次数据
            epoch_batches = []
            
            for batch_idx in range(batches_per_epoch):
                try:
                    # 生成当前批次的数据
                    batch_data = self.generator.generate_batch(
                        stage_config=distribution,
                        batch_size=samples_per_batch,
                        return_k_values=False  # 预训练不需要K值
                    )
                    
                    # 处理数据格式：如果是元组，提取字典部分
                    processed_batch = []
                    for item in batch_data:
                        if isinstance(item, tuple):
                            # (grid_data, k_value) 格式
                            processed_batch.append(item[0])
                        else:
                            # 直接是grid_data字典
                            processed_batch.append(item)
                    
                    epoch_batches.append(processed_batch)
                    
                    if (batch_idx + 1) % 10 == 0:
                        logger.debug(f"已生成 {batch_idx + 1}/{batches_per_epoch} 个批次")
                        
                except Exception as e:
                    logger.error(f"生成批次数据失败: epoch={epoch}, batch={batch_idx}, 错误: {e}")
                    # 创建一个最小的替代批次
                    fallback_batch = self._create_fallback_batch(samples_per_batch)
                    epoch_batches.append(fallback_batch)
            
            all_batches.extend(epoch_batches)
        
        logger.info(f"{phase} 阶段数据生成完成，共 {len(all_batches)} 个批次")
        return all_batches
    
    def _create_fallback_batch(self, batch_size: int) -> List[Dict]:
        """创建一个最小的替代批次"""
        logger.warning("创建最小替代批次")
        
        fallback_batch = []
        for _ in range(batch_size):
            # 创建一个最小的IEEE14电网数据
            try:
                grid_data = self.base_case_loader('ieee14')
                fallback_batch.append(grid_data)
            except Exception as e:
                logger.error(f"创建替代批次失败: {e}")
                # 创建一个最小的数据结构
                minimal_grid = {
                    'bus': np.array([[1, 1, 0.1, 0.05, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9]]),
                    'branch': np.array([[1, 1, 0.01, 0.01, 0, 100, 100, 100, 1, 0, 1]])
                }
                fallback_batch.append(minimal_grid)
        
        return fallback_batch
    
    def generate_validation_data(self, 
                                validation_samples: int = 1024,
                                validation_batches: int = 32) -> List[List[Dict]]:
        """
        生成验证数据集
        
        Args:
            validation_samples: 验证样本总数
            validation_batches: 验证批次数
            
        Returns:
            List of validation batches
        """
        logger.info("开始生成验证数据集...")
        
        # 设置固定种子确保验证集可重现
        self._set_random_seeds(42)
        
        # 生成验证数据
        validation_data = self.generate_phase_data(
            phase='validation',
            samples_per_epoch=validation_samples,
            batches_per_epoch=validation_batches
        )
        
        logger.info(f"验证数据集生成完成，共 {len(validation_data)} 个批次")
        return validation_data
    
    def get_phase_info(self, phase: str) -> Dict[str, Any]:
        """获取阶段信息"""
        if phase not in self.phase_configs:
            raise ValueError(f"未知的阶段: {phase}")
        
        phase_config = self.phase_configs[phase]
        
        return {
            'phase': phase,
            'epochs': phase_config['epochs'],
            'distribution': phase_config['distribution'],
            'description': phase_config['description'],
            'generator_config': {
                'topology_variation_prob': self.grid_config.topology_variation_prob,
                'node_addition_prob': self.grid_config.node_addition_prob,
                'node_removal_prob': self.grid_config.node_removal_prob,
                'edge_addition_prob': self.grid_config.edge_addition_prob,
                'edge_removal_prob': self.grid_config.edge_removal_prob,
                'ensure_connectivity': self.grid_config.ensure_connectivity,
                'min_degree': self.grid_config.min_degree,
                'max_degree': self.grid_config.max_degree
            }
        }
    
    def get_all_phases_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有阶段信息"""
        return {
            phase: self.get_phase_info(phase)
            for phase in self.phase_configs.keys()
        }
    
    def estimate_generation_time(self, 
                                samples_per_epoch: int = 1024,
                                batches_per_epoch: int = 32) -> Dict[str, float]:
        """
        估计生成时间
        
        Args:
            samples_per_epoch: 每个epoch的样本数
            batches_per_epoch: 每个epoch的批次数
            
        Returns:
            各阶段的预估生成时间（秒）
        """
        # 基于经验的单样本生成时间（秒）
        time_per_sample = 0.01
        
        estimates = {}
        
        for phase, config in self.phase_configs.items():
            if phase == 'validation':
                total_samples = samples_per_epoch
            else:
                start_epoch, end_epoch = config['epochs']
                epochs_count = end_epoch - start_epoch
                total_samples = samples_per_epoch * epochs_count
            
            estimated_time = total_samples * time_per_sample
            estimates[phase] = estimated_time
        
        estimates['total'] = sum(estimates.values())
        
        return estimates
    
    def validate_generation_config(self) -> Dict[str, Any]:
        """验证生成配置"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # 检查base_case_loader
            if not callable(self.base_case_loader):
                validation_result['issues'].append("base_case_loader不是可调用对象")
                validation_result['is_valid'] = False
            
            # 检查阶段配置
            for phase, config in self.phase_configs.items():
                distribution = config['distribution']
                
                # 检查分布总和
                total_prob = sum(distribution.values())
                if abs(total_prob - 1.0) > 1e-6:
                    validation_result['issues'].append(f"{phase} 阶段分布概率总和不等于1: {total_prob}")
                    validation_result['is_valid'] = False
                
                # 检查分布值
                for scale, prob in distribution.items():
                    if not (0 <= prob <= 1):
                        validation_result['issues'].append(f"{phase} 阶段 {scale} 规模概率超出范围: {prob}")
                        validation_result['is_valid'] = False
            
            # 测试数据生成器
            try:
                test_batch = self.generator.generate_batch(
                    stage_config={'small': 1.0, 'medium': 0.0, 'large': 0.0},
                    batch_size=2,
                    return_k_values=False
                )
                
                if not test_batch or len(test_batch) == 0:
                    validation_result['issues'].append("测试数据生成失败")
                    validation_result['is_valid'] = False
                    
            except Exception as e:
                validation_result['issues'].append(f"数据生成器测试失败: {e}")
                validation_result['is_valid'] = False
        
        except Exception as e:
            validation_result['issues'].append(f"配置验证过程中发生错误: {e}")
            validation_result['is_valid'] = False
        
        return validation_result