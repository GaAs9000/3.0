#!/usr/bin/env python3
"""
数据集验证工具

功能：
- 数据完整性检查
- 分布统计验证
- 错误数据检测和修复
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, storage_manager, paths):
        """
        初始化数据集验证器
        
        Args:
            storage_manager: 存储管理器
            paths: 路径管理器
        """
        self.storage_manager = storage_manager
        self.paths = paths
        
        # 验证配置
        self.expected_phases = ['early_phase', 'balance_phase', 'late_phase', 'validation']
        self.expected_distributions = {
            'early_phase': {'small': 0.7, 'medium': 0.3, 'large': 0.0},
            'balance_phase': {'small': 0.4, 'medium': 0.4, 'large': 0.2},
            'late_phase': {'small': 0.2, 'medium': 0.3, 'large': 0.5},
            'validation': {'small': 0.4, 'medium': 0.4, 'large': 0.2}
        }
        
        logger.info("数据集验证器初始化完成")
    
    def validate_all_datasets(self) -> Dict[str, Any]:
        """
        验证所有数据集
        
        Returns:
            验证结果
        """
        logger.info("开始验证所有数据集...")
        
        validation_results = {
            'overall_valid': True,
            'validation_time': datetime.now().isoformat(),
            'phases': {},
            'metadata_validation': {},
            'issues': [],
            'warnings': []
        }
        
        try:
            # 验证各阶段数据
            for phase in self.expected_phases:
                logger.info(f"验证 {phase} 阶段...")
                phase_result = self.validate_phase(phase)
                validation_results['phases'][phase] = phase_result
                
                if not phase_result['valid']:
                    validation_results['overall_valid'] = False
                    validation_results['issues'].extend(phase_result.get('issues', []))
                
                validation_results['warnings'].extend(phase_result.get('warnings', []))
            
            # 验证元数据
            logger.info("验证元数据...")
            metadata_result = self.validate_metadata()
            validation_results['metadata_validation'] = metadata_result
            
            if not metadata_result['valid']:
                validation_results['overall_valid'] = False
                validation_results['issues'].extend(metadata_result.get('issues', []))
            
            # 验证阶段间一致性
            logger.info("验证阶段间一致性...")
            consistency_result = self.validate_cross_phase_consistency()
            validation_results['consistency_validation'] = consistency_result
            
            if not consistency_result['valid']:
                validation_results['overall_valid'] = False
                validation_results['issues'].extend(consistency_result.get('issues', []))
            
            logger.info(f"数据集验证完成，总体状态: {'有效' if validation_results['overall_valid'] else '无效'}")
            
        except Exception as e:
            logger.error(f"数据集验证过程中发生错误: {e}")
            validation_results['overall_valid'] = False
            validation_results['issues'].append(f"验证过程异常: {e}")
        
        return validation_results
    
    def validate_phase(self, phase: str) -> Dict[str, Any]:
        """
        验证单个阶段
        
        Args:
            phase: 阶段名称
            
        Returns:
            阶段验证结果
        """
        result = {
            'valid': True,
            'phase': phase,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # 检查批次文件是否存在
            batch_files = self.paths.list_batch_files(phase)
            
            if not batch_files:
                result['valid'] = False
                result['issues'].append(f"阶段 {phase} 没有找到批次文件")
                return result
            
            result['statistics']['batch_count'] = len(batch_files)
            logger.info(f"阶段 {phase} 找到 {len(batch_files)} 个批次文件")
            
            # 抽样验证批次数据
            sample_size = min(10, len(batch_files))
            sample_indices = np.random.choice(len(batch_files), sample_size, replace=False)
            
            sample_stats = {
                'valid_batches': 0,
                'total_samples': 0,
                'avg_samples_per_batch': 0,
                'data_types': Counter(),
                'scale_distribution': Counter()
            }
            
            for idx in sample_indices:
                try:
                    batch_data = self.storage_manager.load_batch(phase, idx)
                    
                    # 验证批次数据
                    batch_validation = self.validate_batch_data(batch_data)
                    
                    if batch_validation['valid']:
                        sample_stats['valid_batches'] += 1
                        sample_stats['total_samples'] += len(batch_data)
                        
                        # 收集数据类型统计
                        for item in batch_data:
                            if isinstance(item, dict):
                                sample_stats['data_types']['dict'] += 1
                                # 估算数据规模
                                scale = self._estimate_data_scale(item)
                                sample_stats['scale_distribution'][scale] += 1
                            else:
                                sample_stats['data_types']['other'] += 1
                    else:
                        result['issues'].extend(batch_validation['issues'])
                        result['warnings'].extend(batch_validation['warnings'])
                        
                except Exception as e:
                    result['warnings'].append(f"批次 {idx} 验证失败: {e}")
            
            # 计算平均值
            if sample_stats['valid_batches'] > 0:
                sample_stats['avg_samples_per_batch'] = sample_stats['total_samples'] / sample_stats['valid_batches']
            
            result['statistics']['sample_validation'] = sample_stats
            
            # 验证规模分布
            if sample_stats['scale_distribution']:
                distribution_validation = self.validate_scale_distribution(
                    phase, sample_stats['scale_distribution']
                )
                result['statistics']['distribution_validation'] = distribution_validation
                
                if not distribution_validation['valid']:
                    result['warnings'].extend(distribution_validation['warnings'])
            
            # 检查文件大小一致性
            size_validation = self.validate_file_sizes(batch_files)
            result['statistics']['size_validation'] = size_validation
            
            if not size_validation['valid']:
                result['warnings'].extend(size_validation['warnings'])
            
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"阶段 {phase} 验证过程异常: {e}")
        
        return result
    
    def validate_batch_data(self, batch_data: List[Any]) -> Dict[str, Any]:
        """
        验证批次数据
        
        Args:
            batch_data: 批次数据
            
        Returns:
            批次验证结果
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # 检查批次数据格式
            if not isinstance(batch_data, list):
                result['valid'] = False
                result['issues'].append(f"批次数据不是列表格式: {type(batch_data)}")
                return result
            
            if len(batch_data) == 0:
                result['valid'] = False
                result['issues'].append("批次数据为空")
                return result
            
            # 检查批次中的每个样本
            for i, item in enumerate(batch_data):
                if not isinstance(item, dict):
                    result['warnings'].append(f"样本 {i} 不是字典格式: {type(item)}")
                    continue
                
                # 检查必需的键
                required_keys = ['bus', 'branch']
                for key in required_keys:
                    if key not in item:
                        result['issues'].append(f"样本 {i} 缺少必需键: {key}")
                        result['valid'] = False
                        continue
                
                # 验证bus数据
                if not self._validate_bus_data(item['bus']):
                    result['warnings'].append(f"样本 {i} 的bus数据可能有问题")
                
                # 验证branch数据
                if not self._validate_branch_data(item['branch']):
                    result['warnings'].append(f"样本 {i} 的branch数据可能有问题")
                
                # 只检查前10个样本的详细信息
                if i >= 10:
                    break
        
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"批次数据验证异常: {e}")
        
        return result
    
    def _validate_bus_data(self, bus_data) -> bool:
        """验证bus数据"""
        try:
            if not isinstance(bus_data, np.ndarray):
                return False
            
            if len(bus_data.shape) != 2:
                return False
            
            if bus_data.shape[0] < 2:  # 至少2个节点
                return False
            
            return True
        except:
            return False
    
    def _validate_branch_data(self, branch_data) -> bool:
        """验证branch数据"""
        try:
            if not isinstance(branch_data, np.ndarray):
                return False
            
            if len(branch_data.shape) != 2:
                return False
            
            if branch_data.shape[0] < 1:  # 至少1条支路
                return False
            
            return True
        except:
            return False
    
    def _estimate_data_scale(self, grid_data: Dict) -> str:
        """估算数据规模"""
        try:
            if 'bus' in grid_data:
                node_count = len(grid_data['bus'])
                
                if node_count <= 40:
                    return 'small'
                elif node_count <= 100:
                    return 'medium'
                else:
                    return 'large'
            return 'unknown'
        except:
            return 'unknown'
    
    def validate_scale_distribution(self, phase: str, scale_counter: Counter) -> Dict[str, Any]:
        """
        验证规模分布
        
        Args:
            phase: 阶段名称
            scale_counter: 规模计数器
            
        Returns:
            分布验证结果
        """
        result = {
            'valid': True,
            'warnings': [],
            'actual_distribution': {},
            'expected_distribution': {},
            'deviations': {}
        }
        
        try:
            expected_dist = self.expected_distributions.get(phase, {})
            result['expected_distribution'] = expected_dist
            
            # 计算实际分布
            total_samples = sum(scale_counter.values())
            if total_samples > 0:
                actual_dist = {
                    scale: count / total_samples 
                    for scale, count in scale_counter.items()
                }
                result['actual_distribution'] = actual_dist
                
                # 计算偏差
                for scale in ['small', 'medium', 'large']:
                    expected = expected_dist.get(scale, 0)
                    actual = actual_dist.get(scale, 0)
                    deviation = abs(actual - expected)
                    result['deviations'][scale] = deviation
                    
                    # 检查偏差是否过大（阈值0.2）
                    if deviation > 0.2:
                        result['warnings'].append(
                            f"阶段 {phase} 的 {scale} 规模分布偏差过大: "
                            f"期望 {expected:.2f}, 实际 {actual:.2f}, 偏差 {deviation:.2f}"
                        )
            
        except Exception as e:
            result['valid'] = False
            result['warnings'].append(f"分布验证异常: {e}")
        
        return result
    
    def validate_file_sizes(self, batch_files: List[Path]) -> Dict[str, Any]:
        """
        验证文件大小一致性
        
        Args:
            batch_files: 批次文件列表
            
        Returns:
            文件大小验证结果
        """
        result = {
            'valid': True,
            'warnings': [],
            'size_stats': {}
        }
        
        try:
            file_sizes = []
            for file_path in batch_files:
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    file_sizes.append(size_mb)
            
            if file_sizes:
                mean_size = np.mean(file_sizes)
                std_size = np.std(file_sizes)
                
                result['size_stats'] = {
                    'mean_size_mb': round(mean_size, 3),
                    'std_size_mb': round(std_size, 3),
                    'min_size_mb': round(min(file_sizes), 3),
                    'max_size_mb': round(max(file_sizes), 3),
                    'file_count': len(file_sizes)
                }
                
                # 检查是否有异常大小的文件
                for i, size in enumerate(file_sizes):
                    if abs(size - mean_size) > 3 * std_size:  # 3倍标准差
                        result['warnings'].append(
                            f"文件 {batch_files[i].name} 大小异常: {size:.3f}MB "
                            f"(平均: {mean_size:.3f}MB, 标准差: {std_size:.3f}MB)"
                        )
        
        except Exception as e:
            result['valid'] = False
            result['warnings'].append(f"文件大小验证异常: {e}")
        
        return result
    
    def validate_metadata(self) -> Dict[str, Any]:
        """
        验证元数据
        
        Returns:
            元数据验证结果
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # 验证元数据一致性
            metadata_validation = self.storage_manager.metadata_manager.validate_metadata_consistency()
            
            if not metadata_validation['is_valid']:
                result['valid'] = False
                result['issues'].extend(metadata_validation['issues'])
            
            result['warnings'].extend(metadata_validation['warnings'])
            
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"元数据验证异常: {e}")
        
        return result
    
    def validate_cross_phase_consistency(self) -> Dict[str, Any]:
        """
        验证阶段间一致性
        
        Returns:
            一致性验证结果
        """
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # 检查各阶段的批次数量是否符合预期
            curriculum_info = self.storage_manager.load_metadata('curriculum_info')
            
            if curriculum_info and 'phase_details' in curriculum_info:
                for phase in ['early_phase', 'balance_phase', 'late_phase']:
                    expected_epochs = curriculum_info['phase_details'][phase]['total_epochs']
                    actual_batches = self.paths.get_dataset_size(phase)
                    
                    # 假设每个epoch 32个批次
                    expected_batches = expected_epochs * 32
                    
                    if actual_batches != expected_batches:
                        result['warnings'].append(
                            f"阶段 {phase} 批次数量不符合预期: "
                            f"期望 {expected_batches}, 实际 {actual_batches}"
                        )
            
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"一致性验证异常: {e}")
        
        return result
    
    def generate_validation_report(self) -> str:
        """
        生成验证报告
        
        Returns:
            验证报告文本
        """
        validation_results = self.validate_all_datasets()
        
        report_lines = [
            "=" * 60,
            "预训练数据集验证报告",
            "=" * 60,
            f"验证时间: {validation_results['validation_time']}",
            f"总体状态: {'✓ 有效' if validation_results['overall_valid'] else '✗ 无效'}",
            ""
        ]
        
        # 各阶段验证结果
        report_lines.append("阶段验证结果:")
        report_lines.append("-" * 40)
        
        for phase, result in validation_results['phases'].items():
            status = "✓ 有效" if result['valid'] else "✗ 无效"
            batch_count = result['statistics'].get('batch_count', 0)
            
            report_lines.append(f"{phase:15} {status:10} ({batch_count} 个批次)")
            
            if result['issues']:
                for issue in result['issues']:
                    report_lines.append(f"  ✗ {issue}")
            
            if result['warnings']:
                for warning in result['warnings'][:3]:  # 只显示前3个警告
                    report_lines.append(f"  ⚠ {warning}")
        
        report_lines.append("")
        
        # 元数据验证结果
        metadata_result = validation_results['metadata_validation']
        status = "✓ 有效" if metadata_result['valid'] else "✗ 无效"
        report_lines.append(f"元数据验证: {status}")
        
        if metadata_result['issues']:
            for issue in metadata_result['issues']:
                report_lines.append(f"  ✗ {issue}")
        
        # 一致性验证结果
        consistency_result = validation_results['consistency_validation']
        status = "✓ 有效" if consistency_result['valid'] else "✗ 无效"
        report_lines.append(f"一致性验证: {status}")
        
        if consistency_result['issues']:
            for issue in consistency_result['issues']:
                report_lines.append(f"  ✗ {issue}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)