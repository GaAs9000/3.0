import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import hashlib
import pickle
import os
from pathlib import Path

class PowerGridDataProcessor:
    """
    电网数据处理器 - 专为RL动态分区训练设计
    
    主要功能：
    1. MATPOWER格式数据清洗和特征提取
    2. 构建PyTorch Geometric图数据
    3. 支持哈希缓存机制
    """
    
    def __init__(self, normalize: bool = True, cache_dir: str = 'cache'):
        self.normalize = normalize
        self.node_scaler = None
        self.edge_scaler = None
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(exist_ok=True, parents=True)

    def graph_from_mpc(self, mpc: Dict) -> Data:
        """
        将MATPOWER格式数据转换为PyTorch Geometric图数据
        
        参数:
            mpc: MATPOWER格式的电网数据字典
            
        返回:
            data: PyTorch Geometric Data对象
        """
        # 1. 计算数据哈希用于缓存
        raw_bytes = pickle.dumps((mpc["bus"].tolist(), mpc["branch"].tolist()))
        case_hash = hashlib.md5(raw_bytes).hexdigest()[:8]
        cache_file = f"{self.cache_dir}/{case_hash}.pt"
        
        # 2. 尝试从缓存加载
        if os.path.exists(cache_file):
            print(f"📂 从缓存加载: {cache_file}")
            return torch.load(cache_file, map_location="cpu", weights_only=False)
        
        # 3. 首次构建图数据
        print(f"🔨 首次构建图数据...")
        baseMVA, df_nodes, df_edges, df_edge_features = self.process_matpower_data(mpc)
        data = self.create_pyg_data(df_nodes, df_edges, df_edge_features)
        
        # 4. 保存到缓存
        torch.save(data, cache_file, pickle_protocol=pickle.DEFAULT_PROTOCOL)
        print(f"💾 已缓存到: {cache_file}")
        
        return data   
    
    def process_matpower_data(self, mpc: Dict) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        处理MATPOWER格式数据
        
        参数:
            mpc: MATPOWER格式的电网数据字典
            
        返回:
            baseMVA: 基准功率(MVA)
            df_nodes: 节点特征DataFrame
            df_edges: 边索引DataFrame
            df_edge_features: 边特征DataFrame
        """
        # 1. 提取基准功率和基础数据
        baseMVA = float(mpc['baseMVA'])
        bus = np.asarray(mpc['bus']).copy()
        branch = np.asarray(mpc['branch']).copy()
        
        # 2. 修复支路载流限制的缺失值
        for col in [5, 6, 7]:  # rateA, rateB, rateC
            mask = branch[:, col] == 0
            branch[mask, col] = baseMVA
        
        # 3. 提取节点和边特征
        node_features = self._extract_node_features(bus, branch, baseMVA)
        edge_features, edge_index = self._extract_edge_features(branch, baseMVA)
        
        # 4. 添加发电机特征
        if 'gen' in mpc:
            gen_features = self._extract_generator_features(mpc['gen'], bus, baseMVA)
            node_features = np.hstack([node_features, gen_features])
        
        # 5. 标准化特征
        if self.normalize:
            node_features = self._normalize_features(node_features, 'node')
            edge_features = self._normalize_features(edge_features, 'edge')
        
        # 6. 创建DataFrame
        node_columns = ['Pd', 'Qd', 'Gs', 'Bs', 'Vm', 'Va', 'Vmax', 'Vmin', 'degree', 'type_PQ', 'type_PV', 'type_slack']
        if 'gen' in mpc:
            node_columns.extend(['Pg', 'Qg', 'Pg_max', 'Pg_min', 'is_gen'])
            
        df_nodes = pd.DataFrame(node_features, columns=node_columns)
        df_edges = pd.DataFrame(edge_index, columns=['from_bus', 'to_bus'])
        df_edge_features = pd.DataFrame(
            edge_features,
            columns=['r', 'x', 'b', '|z|', 'y', 'rateA', 'angle_diff', 'is_transformer', 'status']
        )
        
        return baseMVA, df_nodes, df_edges, df_edge_features
    
    def _extract_node_features(self, bus: np.ndarray, branch: np.ndarray, baseMVA: float) -> np.ndarray:
        """
        提取节点特征
        
        参数:
            bus: 母线数据矩阵
            branch: 支路数据矩阵  
            baseMVA: 基准功率
            
        返回:
            features: 节点特征矩阵 [n_bus, n_features]
            
        特征包括:
            - 有功/无功负荷 (当前值，用于RL状态表征)
            - 并联电导/电纳
            - 电压运行状态和约束
            - 节点度数
            - 节点类型 (one-hot编码)
        """
        n_bus = bus.shape[0]
        
        # 1. 运行状态特征
        Pd_pu = bus[:, 2] / baseMVA
        Qd_pu = bus[:, 3] / baseMVA
        Gs = bus[:, 4]
        Bs = bus[:, 5]
        Vm = bus[:, 7]
        Va = np.deg2rad(bus[:, 8])
        
        # 2. 电压约束
        Vmax = bus[:, 11] if bus.shape[1] > 11 else np.ones(n_bus) * 1.1
        Vmin = bus[:, 12] if bus.shape[1] > 12 else np.ones(n_bus) * 0.9
        
        # 3. 拓扑特征：计算节点度数
        from_idx = branch[:, 0].astype(int) - 1  # 转换为0-based索引
        to_idx = branch[:, 1].astype(int) - 1
        degree = np.bincount(np.hstack([from_idx, to_idx]), minlength=n_bus)
        
        # 4. 节点类型 one-hot编码 (1=PQ, 2=PV, 3=Slack)
        bus_types = bus[:, 1].astype(int)
        type_PQ = (bus_types == 1).astype(float)
        type_PV = (bus_types == 2).astype(float)
        type_slack = (bus_types == 3).astype(float)
        
        # 5. 组合所有特征
        features = np.column_stack([
            Pd_pu, Qd_pu, Gs, Bs, Vm, Va, Vmax, Vmin, degree,
            type_PQ, type_PV, type_slack
        ])
        
        return features
    
    def _extract_edge_features(self, branch: np.ndarray, baseMVA: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取边特征
        
        参数:
            branch: 支路数据矩阵
            baseMVA: 基准功率
            
        返回:
            features: 边特征矩阵 [n_branch, n_features]
            edge_index: 边索引矩阵 [n_branch, 2]
            
        特征包括:
            - 电阻、电抗、电纳
            - 阻抗模长、导纳
            - 载流限制
            - 相角差限制  
            - 是否为变压器
            - 支路运行状态
        """
        # 1. 电气参数
        r_raw = branch[:, 2]
        x_raw = branch[:, 3]
        b = branch[:, 4]

        # 2. 处理NaN值
        valid_mask = ~np.isnan(r_raw) & ~np.isnan(x_raw)
        r_filled = np.nan_to_num(r_raw, nan=0.0)
        x_filled = np.nan_to_num(x_raw, nan=0.0)
        
        # 3. 计算阻抗模长和导纳
        z_mag_calc = np.sqrt(r_filled**2 + x_filled**2)
        z_magnitude = np.where(valid_mask, z_mag_calc, np.inf)
        y = np.where(np.isinf(z_magnitude), 0.0, 1.0 / (z_magnitude + 1e-10))
        
        # 4. 载流限制和相角约束
        rateA = branch[:, 5] / baseMVA
        angle_min = branch[:, 11] if branch.shape[1] > 11 else -np.pi * np.ones(len(branch))
        angle_max = branch[:, 12] if branch.shape[1] > 12 else np.pi * np.ones(len(branch))
        angle_diff = angle_max - angle_min
        
        # 5. 变压器标识和运行状态
        tap_ratio = branch[:, 8]
        is_transformer = ((tap_ratio != 0) & (tap_ratio != 1)).astype(float)
        status = branch[:, 10].astype(float) if branch.shape[1] > 10 else np.ones(len(branch))
        
        # 6. 边索引 (转换为0-based)
        edge_index = np.column_stack([
            branch[:, 0].astype(int) - 1,
            branch[:, 1].astype(int) - 1
        ])
        
        # 7. 组合所有特征
        features = np.column_stack([
            r_filled, x_filled, b, z_magnitude, y, rateA, angle_diff, is_transformer, status
        ])
        
        # 8. 最终NaN检查和处理
        if np.any(np.isnan(features)):
            print("⚠️ 警告：清理特征矩阵中的NaN值...")
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return features, edge_index
    
    def _extract_generator_features(self, gen: np.ndarray, bus: np.ndarray, baseMVA: float) -> np.ndarray:
        """
        提取发电机特征并映射到节点
        
        参数:
            gen: 发电机数据矩阵
            bus: 母线数据矩阵
            baseMVA: 基准功率
            
        返回:
            features: 发电机特征矩阵 [n_bus, n_gen_features]
            
        特征包括:
            - 当前有功/无功出力 (用于状态表征)
            - 有功发电容量上下限 (用于约束)
            - 发电机标识
        """
        n_bus = bus.shape[0]
        
        # 1. 初始化发电机特征
        Pg = np.zeros(n_bus)
        Qg = np.zeros(n_bus)
        Pg_max = np.zeros(n_bus)
        Pg_min = np.zeros(n_bus)
        is_gen = np.zeros(n_bus)
        
        # 2. 聚合每个节点的发电机特征
        if len(gen) > 0:
            idx = gen[:, 0].astype(int) - 1  # 转换为0-based索引
            # 当前出力 (用于状态表征)
            np.add.at(Pg, idx, gen[:, 1] / baseMVA)      # 当前有功出力
            np.add.at(Qg, idx, gen[:, 2] / baseMVA)      # 当前无功出力
            # 容量约束 (用于可行域约束)
            np.add.at(Pg_max, idx, gen[:, 8] / baseMVA)  # 有功容量上限
            np.add.at(Pg_min, idx, gen[:, 9] / baseMVA)  # 有功容量下限
            is_gen[idx] = 1.0
        
        return np.column_stack([Pg, Qg, Pg_max, Pg_min, is_gen])
    
    def _normalize_features(self, features: np.ndarray, feature_type: str) -> np.ndarray:
        """使用RobustScaler标准化特征"""
        # 处理inf值
        if np.any(np.isinf(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        if feature_type == 'node' and self.node_scaler is None:
            self.node_scaler = RobustScaler()
            return self.node_scaler.fit_transform(features)
        elif feature_type == 'edge' and self.edge_scaler is None:
            self.edge_scaler = RobustScaler()
            return self.edge_scaler.fit_transform(features)
        elif feature_type == 'node':
            return self.node_scaler.transform(features)
        else:
            return self.edge_scaler.transform(features)
    
    def create_pyg_data(self, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, 
                       df_edge_features: pd.DataFrame) -> Data:
        """
        创建PyTorch Geometric数据对象
        
        参数:
            df_nodes: 节点特征DataFrame
            df_edges: 边索引DataFrame
            df_edge_features: 边特征DataFrame
            
        返回:
            data: PyTorch Geometric Data对象
            
        注意:
            - 自动创建无向图 (添加反向边)
            - 边特征与边索引保持对应关系
        """
        # 1. 节点特征
        x = torch.tensor(df_nodes.values, dtype=torch.float32)
        
        # 2. 创建无向图的边索引
        from_nodes = torch.tensor(df_edges['from_bus'].values, dtype=torch.long)
        to_nodes = torch.tensor(df_edges['to_bus'].values, dtype=torch.long)
        
        edge_index = torch.stack([
            torch.cat([from_nodes, to_nodes]),
            torch.cat([to_nodes, from_nodes])
        ])
        
        # 3. 边特征 (双向复制以匹配边索引)
        edge_attr = torch.tensor(
            np.vstack([df_edge_features.values, df_edge_features.values]),
            dtype=torch.float32
        )
        
        # 4. 创建Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(df_nodes)
        )
        
        # 5. 添加辅助信息
        data.node_names = df_nodes.index.tolist()
        data.edge_names = df_edges.index.tolist()
        
        return data

