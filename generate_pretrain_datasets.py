#!/usr/bin/env python3
"""
预训练数据集生成脚本

执行三阶段课程学习数据集的生成
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append('/home/owari/2.0/code/src')
sys.path.append('/home/owari/2.0/code/src/data_generation')

# 切换到正确的工作目录
os.chdir('/home/owari/2.0')

from data_generation.pretrain_dataset_manager import PretrainDatasetManager

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/home/owari/2.0/data/pretrain_data/logs/generation.log')
        ]
    )

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 开始生成预训练数据集...")
    logger.info("=" * 60)
    logger.info("开始生成预训练数据集")
    logger.info("=" * 60)
    
    try:
        # 创建数据集管理器
        config = {
            'seed': 42,
            'topology_variation_prob': 0.2,
            'node_addition_prob': 0.1,
            'node_removal_prob': 0.05,
            'edge_addition_prob': 0.1,
            'edge_removal_prob': 0.05,
            'ensure_connectivity': True,
            'min_degree': 2,
            'max_degree': 6,
            'use_compression': True,
            'max_workers': 4
        }
        
        manager = PretrainDatasetManager(
            config=config,
            data_root="/home/owari/2.0/data/pretrain_data"
        )
        
        print("✅ 数据集管理器初始化完成")
        logger.info("数据集管理器初始化完成")
        
        # 检查是否已有数据集
        if manager.has_pregenerated_datasets():
            print("⚠️  发现已存在的数据集，将重新生成...")
            logger.info("发现已存在的数据集，将重新生成")
        
        # 生成数据集
        print("🔄 开始生成数据集...")
        logger.info("开始生成数据集")
        
        generation_results = manager.generate_all_datasets(
            samples_per_epoch=1024,     # 每个epoch 1024个样本
            batches_per_epoch=32,       # 每个epoch 32个批次
            validation_samples=1024,    # 验证集 1024个样本
            validation_batches=32,      # 验证集 32个批次
            force_regenerate=True       # 强制重新生成
        )
        
        if generation_results['success']:
            print("✅ 数据集生成成功！")
            logger.info("数据集生成成功")
            
            # 显示生成统计
            total_time = generation_results['total_generation_time_seconds']
            print(f"⏱️  总耗时: {total_time:.2f} 秒")
            
            for phase, stats in generation_results['phases_results'].items():
                batches = stats['batches_generated']
                time_sec = stats['generation_time_seconds']
                print(f"   {phase}: {batches} 个批次, {time_sec:.2f} 秒")
            
            # 显示验证结果
            validation_results = generation_results['validation_results']
            if validation_results['overall_valid']:
                print("✅ 数据集验证通过")
            else:
                print("⚠️  数据集验证发现问题")
                for issue in validation_results['issues'][:3]:  # 只显示前3个问题
                    print(f"   ❌ {issue}")
            
            # 显示存储统计
            storage_stats = generation_results['final_stats']['storage_stats']
            total_size = storage_stats['total_size_mb']
            total_files = storage_stats['total_files']
            print(f"💾 存储统计: {total_files} 个文件, {total_size:.2f} MB")
            
            logger.info(f"数据集生成完成，总耗时: {total_time:.2f} 秒")
            
        else:
            print("❌ 数据集生成失败")
            logger.error(f"数据集生成失败: {generation_results.get('error', '未知错误')}")
            return 1
            
    except Exception as e:
        print(f"❌ 执行过程中发生错误: {e}")
        logger.error(f"执行过程中发生错误: {e}")
        return 1
    
    print("🎉 预训练数据集生成完成！")
    logger.info("预训练数据集生成完成")
    return 0

if __name__ == "__main__":
    exit(main())