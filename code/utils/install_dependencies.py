#!/usr/bin/env python3
"""
电力网络分区强化学习系统依赖安装脚本

自动检测环境并安装所需依赖包
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(cmd, description=""):
    """执行命令并处理错误"""
    print(f"\n🔄 {description}")
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ 成功: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {description}")
        print(f"错误信息: {e.stderr}")
        return False

def check_package(package_name):
    """检查包是否已安装"""
    try:
        if package_name == 'scikit-learn':
            __import__('sklearn')
        elif package_name == 'torch-scatter':
            __import__('torch_scatter')
        elif package_name == 'torch-sparse':
            __import__('torch_sparse')
        else:
            __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def get_torch_version():
    """获取PyTorch版本信息"""
    try:
        import torch
        version = torch.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        return version, cuda_version
    except ImportError:
        return None, None

def install_pytorch_geometric_deps():
    """安装PyTorch Geometric依赖"""
    torch_version, cuda_version = get_torch_version()
    
    if not torch_version:
        print("❌ 未检测到PyTorch，请先安装PyTorch")
        return False
    
    print(f"检测到PyTorch版本: {torch_version}")
    if cuda_version:
        print(f"检测到CUDA版本: {cuda_version}")
        # 构建CUDA版本的wheel URL
        cuda_ver = cuda_version.replace('.', '')[:3]  # 例如: 12.1 -> 121
        wheel_url = f"https://data.pyg.org/whl/torch-{torch_version.split('+')[0]}+cu{cuda_ver}.html"
    else:
        print("未检测到CUDA，使用CPU版本")
        wheel_url = f"https://data.pyg.org/whl/torch-{torch_version.split('+')[0]}+cpu.html"
    
    # 安装torch-scatter和torch-sparse
    packages = ['torch-scatter', 'torch-sparse']
    for pkg in packages:
        if not check_package(pkg):
            cmd = f"pip install {pkg} -f {wheel_url}"
            if not run_command(cmd, f"安装{pkg}"):
                print(f"WARNING: {pkg}安装失败，尝试使用conda安装")
                conda_cmd = f"conda install pyg::{pkg} -c pyg -y"
                run_command(conda_cmd, f"通过conda安装{pkg}")

def install_critical_packages():
    """安装关键包"""
    critical_packages = [
        ('h5py', 'HDF5数据格式支持'),
        ('kaleido', 'Plotly静态图像导出'),
        ('scikit-learn', '机器学习算法')
    ]
    
    for pkg, desc in critical_packages:
        if not check_package(pkg):
            cmd = f"pip install {pkg}"
            run_command(cmd, f"安装{pkg} - {desc}")

def install_optional_packages():
    """安装可选包"""
    optional_packages = [
        ('memory-profiler', '内存分析'),
        ('line-profiler', '行级性能分析'),
        ('gpustat', 'GPU状态监控'),
        ('statsmodels', '统计建模')
    ]
    
    print("\n📦 可选包安装 (可跳过)")
    for pkg, desc in optional_packages:
        if not check_package(pkg):
            response = input(f"是否安装 {pkg} ({desc})? [y/N]: ").lower()
            if response in ['y', 'yes']:
                cmd = f"pip install {pkg}"
                run_command(cmd, f"安装{pkg}")

def main():
    """主函数"""
    print("🚀 电力网络分区强化学习系统依赖安装")
    print("=" * 50)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 9):
        print("❌ 需要Python 3.9或更高版本")
        sys.exit(1)
    
    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"当前conda环境: {conda_env}")
    
    # 1. 安装基础依赖
    print("\n1️⃣ 安装基础依赖...")
    if Path('requirements.txt').exists():
        run_command("pip install -r requirements.txt", "安装requirements.txt中的包")
    
    # 2. 安装PyTorch Geometric依赖
    print("\n2. 安装PyTorch Geometric依赖...")
    install_pytorch_geometric_deps()

    # 3. 安装关键包
    print("\n3. 安装关键包...")
    install_critical_packages()

    # 4. 可选包安装
    print("\n4. 可选包安装...")
    install_optional_packages()

    # 5. 验证安装
    print("\n5. 验证安装...")
    verification_packages = [
        'torch', 'torch_geometric', 'stable_baselines3', 
        'gymnasium', 'numpy', 'matplotlib', 'networkx',
        'pandapower', 'textual', 'rich'
    ]
    
    failed_packages = []
    for pkg in verification_packages:
        if check_package(pkg):
            print(f"OK: {pkg}")
        else:
            print(f"FAILED: {pkg}")
            failed_packages.append(pkg)
    
    if failed_packages:
        print(f"\n⚠️  以下包安装失败: {', '.join(failed_packages)}")
        print("请手动安装这些包或检查错误信息")
    else:
        print("\n🎉 所有依赖安装完成！")
        print("\n下一步:")
        print("  python train.py --mode fast --tui  # 开始训练")
        print("  python test.py --quick              # 快速测试")

if __name__ == "__main__":
    main()
