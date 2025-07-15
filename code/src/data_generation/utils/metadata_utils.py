#!/usr/bin/env python3
"""
元数据管理工具

功能：
- 数据集元数据管理
- 生成配置记录
- 统计信息管理
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class MetadataManager:
    """元数据管理器"""
    
    def __init__(self, storage_manager):
        self.storage_manager = storage_manager
        self.paths = storage_manager.paths
        
    def create_generation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建生成配置元数据"""
        generation_config = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'config': config,
            'phases': {
                'early_phase': {
                    'epochs': (0, 15),
                    'distribution': {'small': 0.7, 'medium': 0.3, 'large': 0.0},
                    'description': '早期阶段：专注小规模数据，建立基础表示'
                },
                'balance_phase': {
                    'epochs': (15, 35),
                    'distribution': {'small': 0.4, 'medium': 0.4, 'large': 0.2},
                    'description': '平衡阶段：三种规模平衡发展'
                },
                'late_phase': {
                    'epochs': (35, 50),
                    'distribution': {'small': 0.2, 'medium': 0.3, 'large': 0.5},
                    'description': '后期阶段：重点大规模数据，提升复杂性处理能力'
                }
            }
        }
        
        self.storage_manager.save_metadata(generation_config, 'generation_config')
        return generation_config
    
    def create_curriculum_info(self, total_epochs: int = 50) -> Dict[str, Any]:
        """创建课程学习信息"""
        curriculum_info = {
            'total_epochs': total_epochs,
            'early_phase_end': 0.3,
            'balance_phase_end': 0.7,
            'phase_details': {
                'early_phase': {
                    'epoch_range': [0, int(total_epochs * 0.3)],
                    'focus': 'small_scale_learning',
                    'expected_samples_per_epoch': 1024,
                    'total_epochs': int(total_epochs * 0.3)
                },
                'balance_phase': {
                    'epoch_range': [int(total_epochs * 0.3), int(total_epochs * 0.7)],
                    'focus': 'balanced_multi_scale',
                    'expected_samples_per_epoch': 1024,
                    'total_epochs': int(total_epochs * 0.4)
                },
                'late_phase': {
                    'epoch_range': [int(total_epochs * 0.7), total_epochs],
                    'focus': 'large_scale_mastery',
                    'expected_samples_per_epoch': 1024,
                    'total_epochs': int(total_epochs * 0.3)
                }
            },
            'created_at': datetime.now().isoformat()
        }
        
        self.storage_manager.save_metadata(curriculum_info, 'curriculum_info')
        return curriculum_info
    
    def create_phase_distributions(self) -> Dict[str, Any]:
        """创建阶段分布信息"""
        distributions = {
            'early_phase': {
                'small': 0.7,
                'medium': 0.3,
                'large': 0.0,
                'rationale': '早期阶段专注小规模数据，建立基础图结构理解'
            },
            'balance_phase': {
                'small': 0.4,
                'medium': 0.4,
                'large': 0.2,
                'rationale': '平衡阶段引入多种规模，提升泛化能力'
            },
            'late_phase': {
                'small': 0.2,
                'medium': 0.3,
                'large': 0.5,
                'rationale': '后期阶段重点大规模数据，掌握复杂图结构'
            },
            'validation': {
                'small': 0.4,
                'medium': 0.4,
                'large': 0.2,
                'rationale': '验证集保持平衡分布，客观评估模型性能'
            },
            'created_at': datetime.now().isoformat()
        }
        
        self.storage_manager.save_metadata(distributions, 'phase_distributions')
        return distributions
    
    def update_dataset_stats(self, phase: str, stats: Dict[str, Any]):
        """更新数据集统计信息"""
        # 加载现有统计信息
        existing_stats = self.storage_manager.load_metadata('dataset_stats')
        
        if not existing_stats:
            existing_stats = {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'phases': {}
            }
        
        # 更新指定阶段的统计信息
        existing_stats['phases'][phase] = {
            **stats,
            'updated_at': datetime.now().isoformat()
        }
        existing_stats['last_updated'] = datetime.now().isoformat()
        
        self.storage_manager.save_metadata(existing_stats, 'dataset_stats')
        
    def finalize_dataset_stats(self):
        """完成数据集统计信息"""
        # 获取存储统计信息
        storage_stats = self.storage_manager.get_storage_stats()
        
        # 加载现有统计信息
        existing_stats = self.storage_manager.load_metadata('dataset_stats')
        
        if not existing_stats:
            existing_stats = {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'phases': {}
            }
        
        # 更新总体统计信息
        final_stats = {
            **existing_stats,
            'finalized_at': datetime.now().isoformat(),
            'storage_stats': storage_stats,
            'summary': {
                'total_files': storage_stats['total_files'],
                'total_size_mb': storage_stats['total_size_mb'],
                'phases_count': len(storage_stats['phases']),
                'validation_set_size': storage_stats['phases'].get('validation', {}).get('file_count', 0)
            }
        }
        
        self.storage_manager.save_metadata(final_stats, 'dataset_stats')
        return final_stats
    
    def validate_metadata_consistency(self) -> Dict[str, Any]:
        """验证元数据一致性"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # 检查所有元数据文件是否存在
            metadata_types = ['generation_config', 'curriculum_info', 'dataset_stats', 'phase_distributions']
            
            for metadata_type in metadata_types:
                try:
                    metadata = self.storage_manager.load_metadata(metadata_type)
                    if not metadata:
                        validation_results['issues'].append(f"元数据文件为空: {metadata_type}")
                        validation_results['is_valid'] = False
                except Exception as e:
                    validation_results['issues'].append(f"无法加载元数据: {metadata_type}, 错误: {e}")
                    validation_results['is_valid'] = False
            
            # 检查数据集统计信息与实际文件的一致性
            try:
                dataset_stats = self.storage_manager.load_metadata('dataset_stats')
                storage_stats = self.storage_manager.get_storage_stats()
                
                if dataset_stats and 'storage_stats' in dataset_stats:
                    recorded_files = dataset_stats['storage_stats']['total_files']
                    actual_files = storage_stats['total_files']
                    
                    if recorded_files != actual_files:
                        validation_results['warnings'].append(
                            f"记录的文件数量({recorded_files})与实际文件数量({actual_files})不一致"
                        )
                        
            except Exception as e:
                validation_results['issues'].append(f"统计信息一致性检查失败: {e}")
                validation_results['is_valid'] = False
                
        except Exception as e:
            validation_results['issues'].append(f"元数据验证过程中发生错误: {e}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """获取元数据摘要"""
        summary = {
            'available_metadata': {},
            'dataset_overview': {},
            'generation_info': {}
        }
        
        # 检查可用的元数据
        metadata_types = ['generation_config', 'curriculum_info', 'dataset_stats', 'phase_distributions']
        
        for metadata_type in metadata_types:
            try:
                metadata = self.storage_manager.load_metadata(metadata_type)
                summary['available_metadata'][metadata_type] = {
                    'exists': bool(metadata),
                    'size': len(str(metadata)) if metadata else 0
                }
            except Exception as e:
                summary['available_metadata'][metadata_type] = {
                    'exists': False,
                    'error': str(e)
                }
        
        # 数据集概览
        try:
            dataset_stats = self.storage_manager.load_metadata('dataset_stats')
            if dataset_stats and 'summary' in dataset_stats:
                summary['dataset_overview'] = dataset_stats['summary']
        except Exception as e:
            summary['dataset_overview'] = {'error': str(e)}
        
        # 生成信息
        try:
            generation_config = self.storage_manager.load_metadata('generation_config')
            if generation_config:
                summary['generation_info'] = {
                    'version': generation_config.get('version'),
                    'created_at': generation_config.get('created_at'),
                    'phases_count': len(generation_config.get('phases', {}))
                }
        except Exception as e:
            summary['generation_info'] = {'error': str(e)}
        
        return summary