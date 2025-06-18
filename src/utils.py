# 标准库导入
import os
import sys
import math
import random
import time
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import copy

# 数据处理和科学计算
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import SpectralClustering

# 深度学习框架
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# 图神经网络
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree, to_networkx, softmax

# 可视化
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 进度条
from tqdm.notebook import tqdm, trange

# 设置随机种子确保可重复性
def set_seed(seed: int = 42):
    """设置所有随机种子以确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
print(f"📦 PyTorch version: {torch.__version__}")

# 忽略警告
warnings.filterwarnings('ignore')

# 创建必要的目录
for dir_name in ['data', 'models', 'results', 'logs', 'figures']:
    Path(dir_name).mkdir(exist_ok=True)

