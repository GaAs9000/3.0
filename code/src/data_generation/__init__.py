#!/usr/bin/env python3
"""
数据生成模块

功能：
- 预训练数据集生成和管理
- 三阶段课程学习数据生成
- 数据集验证和统计
"""

__version__ = "1.0.0"
__author__ = "Claude & User Collaboration"

from data_generation.pretrain_dataset_manager import PretrainDatasetManager
from data_generation.curriculum_generator import CurriculumGenerator
from data_generation.dataset_validator import DatasetValidator
from data_generation.batch_processor import BatchProcessor

__all__ = [
    "PretrainDatasetManager",
    "CurriculumGenerator", 
    "DatasetValidator",
    "BatchProcessor"
]