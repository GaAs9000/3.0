import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import hashlib
import pickle
import os
from pathlib import Path

class PowerGridDataProcessor:
    """
    电网数据处理
    
    1. 全面升级以支持 PyTorch Geometric 的 HeteroData 格式
    2. 为节点（母线）和边（支路）定义了物理类型
    3. 重构了数据创建流程，以反映电网的异构性
    4. 支持哈希缓存机制
    
    主要功能：
    1. MATPOWER格式数据清洗和特征提取
    2. 构建PyTorch Geometric异构图数据
    3. 区分不同类型的电力设备和连接
    """
    
    def __init__(self, normalize: bool = True, cache_dir: str = 'cache'):
        self.normalize = normalize
        self.node_scaler = None
        self.edge_scaler = None
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(exist_ok=True, parents=True)
        
        # 定义类型映射关系，用于将MATPOWER中的数字代码转换为有意义的字符串
        self.BUS_TYPE_MAP = {1: 'pq', 2: 'pv', 3: 'slack'}
        self.BRANCH_TYPE_MAP = {0: 'line', 1: 'transformer'}

    def graph_from_mpc(self, mpc: Dict) -> HeteroData:
        """
        将MATPOWER格式数据转换为PyTorch Geometric的异构图数据 (HeteroData)
        
        参数:
            mpc: MATPOWER格式的电网数据字典
            
        返回:
            data: PyTorch Geometric HeteroData对象
        """
        # 1. 计算数据哈希用于缓存 (文件名中加入后缀以区分新旧版本)
        raw_bytes = pickle.dumps((mpc["bus"].tolist(), mpc["branch"].tolist()))
        case_hash = hashlib.md5(raw_bytes).hexdigest()[:8]
        cache_file = f"{self.cache_dir}/{case_hash}_hetero.pt"
        
        # 2. 尝试从缓存加载
        if os.path.exists(cache_file):
            print(f"📂 从缓存加载异构图: {cache_file}")
            return torch.load(cache_file, map_location="cpu", weights_only=False)
        
        # 3. 首次构建异构图数据
        print(f"🔨 首次构建异构图数据...")
        baseMVA, df_nodes, df_edges, df_edge_features = self.process_matpower_data(mpc)
        data = self.create_pyg_hetero_data(df_nodes, df_edges, df_edge_features)
        
        # 4. 保存到缓存
        torch.save(data, cache_file, pickle_protocol=pickle.DEFAULT_PROTOCOL)
        print(f"💾 已缓存异构图到: {cache_file}")
        
        return data   
    
    def process_matpower_data(self, mpc: Dict) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        处理MATPOWER格式数据，为构建异构图准备包含类型信息的DataFrame
        
        参数:
            mpc: MATPOWER格式的电网数据字典
            
        返回:
            baseMVA: 基准功率(MVA)
            df_nodes: 包含类型信息的节点特征DataFrame
            df_edges: 边索引DataFrame
            df_edge_features: 包含类型信息的边特征DataFrame
        """
        # 1. 提取基准功率和基础数据
        baseMVA = float(mpc['baseMVA'])
        bus = np.asarray(mpc['bus']).copy()
        branch = np.asarray(mpc['branch']).copy()
        
        # 2. 修复支路载流限制的缺失值
        for col in [5, 6, 7]:  # rateA, rateB, rateC
            mask = branch[:, col] == 0
            branch[mask, col] = baseMVA
        
        # 3. 提取基础特征
        node_features = self._extract_node_features(bus, branch, baseMVA)
        edge_features, edge_index = self._extract_edge_features(branch, baseMVA)
        
        # 4. 添加发电机特征
        if 'gen' in mpc:
            gen_features = self._extract_generator_features(mpc['gen'], bus, baseMVA)
            node_features = np.hstack([node_features, gen_features])
        
        # 5. 标准化特征 (在分割数据类型前进行，确保尺度一致)
        if self.normalize:
            node_features = self._normalize_features(node_features, 'node')
            edge_features = self._normalize_features(edge_features, 'edge')
            
        # 6. 创建包含类型信息的DataFrame
        # 定义节点特征列名 (移除了旧的one-hot类型列)
        node_columns = ['Pd', 'Qd', 'Gs', 'Bs', 'Vm', 'Va', 'Vmax', 'Vmin', 'degree']
        if 'gen' in mpc:
            node_columns.extend(['Pg', 'Qg', 'Pg_max', 'Pg_min', 'is_gen'])
            
        # 使用0-based的母线索引作为DataFrame的索引
        df_nodes = pd.DataFrame(node_features, columns=node_columns, index=bus[:,0].astype(int)-1)
        
        # 添加节点类型列
        bus_types_raw = bus[:, 1].astype(int)
        df_nodes['type'] = [self.BUS_TYPE_MAP.get(bt, 'pq') for bt in bus_types_raw] # 默认为pq类型

        # 定义边特征列名
        df_edge_features = pd.DataFrame(
            edge_features,
            columns=['r', 'x', 'b', '|z|', 'y', 'rateA', 'angle_diff', 'is_transformer', 'status']
        )
        # 添加边类型列
        is_transformer_int = df_edge_features['is_transformer'].astype(int)
        df_edge_features['type'] = [self.BRANCH_TYPE_MAP.get(bt, 'line') for bt in is_transformer_int]
        
        # 创建边索引DataFrame
        df_edges = pd.DataFrame(edge_index, columns=['from_bus', 'to_bus'])
        
        return baseMVA, df_nodes, df_edges, df_edge_features

    def _extract_node_features(self, bus: np.ndarray, branch: np.ndarray, baseMVA: float) -> np.ndarray:
        """
        提取节点特征 (V2.0: 不再进行one-hot编码，只提取数值特征)
        
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
            注：节点类型将在后续阶段通过DataFrame的'type'列单独处理
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
        
        # 4. 组合所有数值特征 (不再包含one-hot编码的类型信息)
        features = np.column_stack([
            Pd_pu, Qd_pu, Gs, Bs, Vm, Va, Vmax, Vmin, degree
        ])
        
        return features
    
    def _extract_edge_features(self, branch: np.ndarray, baseMVA: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取边特征 (逻辑基本不变)
        
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
        提取发电机特征并映射到节点 (逻辑基本不变)
        
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
        """
        标准化特征 (逻辑基本不变)
        
        参数:
            features: 特征矩阵
            feature_type: 特征类型 ('node' 或 'edge')
            
        返回:
            normalized_features: 标准化后的特征矩阵
        """
        # 处理inf值
        if np.any(np.isinf(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        scaler_attr = f"{feature_type}_scaler"
        if getattr(self, scaler_attr) is None:
            scaler = RobustScaler()
            setattr(self, scaler_attr, scaler)
            return scaler.fit_transform(features)
        else:
            return getattr(self, scaler_attr).transform(features)
    
    def create_pyg_hetero_data(self, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, 
                               df_edge_features: pd.DataFrame) -> HeteroData:
        """
        根据包含类型信息的DataFrame创建PyTorch Geometric的HeteroData对象
        这是本次升级的核心功能
        
        参数:
            df_nodes: 包含类型信息的节点特征DataFrame
            df_edges: 边索引DataFrame
            df_edge_features: 包含类型信息的边特征DataFrame
            
        返回:
            data: PyTorch Geometric HeteroData对象
            
        关键步骤:
            1. 为每种节点类型创建独立的特征张量和索引映射
            2. 为每种边类型创建连接关系
            3. 处理全局索引到局部索引的转换
            4. 创建无向图结构
        """
        data = HeteroData()
        
        # --- 1. 处理节点：为每种节点类型创建独立的数据结构 ---
        # 这是最关键的步骤：维护全局ID到各类型局部ID的映射关系
        global_to_local_maps = {}
        
        print(f"🔍 发现节点类型: {df_nodes['type'].unique()}")
        
        for node_type, group in df_nodes.groupby('type'):
            # 节点类型键，例如 'bus_pq', 'bus_pv', 'bus_slack'
            node_type_key = f'bus_{node_type}'
            
            # 获取该类型节点的全局ID列表 (0-based索引)
            type_global_indices = group.index.tolist()
            
            # 创建全局ID到该类型局部ID的映射
            global_to_local_maps[node_type_key] = {
                global_id: local_id for local_id, global_id in enumerate(type_global_indices)
            }
            
            # 提取特征 (去除类型列)
            feature_cols = [col for col in group.columns if col != 'type']
            data[node_type_key].x = torch.tensor(group[feature_cols].values, dtype=torch.float32)
            
            # 存储全局ID，方便未来追溯和调试
            data[node_type_key].global_ids = torch.tensor(type_global_indices, dtype=torch.long)
            
            print(f"  📍 {node_type_key}: {len(type_global_indices)} 个节点")

        # --- 2. 处理边：创建异构图的连接关系 ---
        # 合并边信息，方便处理
        full_edge_df = pd.concat([df_edges, df_edge_features], axis=1)
        
        print(f"🔗 发现边类型: {df_edge_features['type'].unique()}")
        
        # 用于统计关系类型
        relation_stats = {}
        
        for _, row in full_edge_df.iterrows():
            src_global, dst_global = int(row['from_bus']), int(row['to_bus'])
            
            # 安全地获取节点类型，如果节点ID不存在则跳过 (处理不一致的数据)
            try:
                src_type_name = df_nodes.loc[src_global, 'type']
                dst_type_name = df_nodes.loc[dst_global, 'type']
            except KeyError:
                print(f"⚠️ 警告：跳过不存在的节点连接 {src_global}-{dst_global}")
                continue

            src_type_key = f'bus_{src_type_name}'
            dst_type_key = f'bus_{dst_type_name}'
            edge_type = row['type']  # 'line' or 'transformer'

            # 构建关系元组，例如 ('bus_pq', 'connects_line', 'bus_pv')
            # 为了规范化，我们将节点类型按字母顺序排序，以创建唯一的关系键
            sorted_node_keys = sorted([src_type_key, dst_type_key])
            relation_tuple = (sorted_node_keys[0], f'connects_{edge_type}', sorted_node_keys[1])
            
            # 统计关系类型
            if relation_tuple not in relation_stats:
                relation_stats[relation_tuple] = 0
            relation_stats[relation_tuple] += 1
            
            # 从全局ID转换为相应类型的局部ID
            try:
                src_local = global_to_local_maps[src_type_key][src_global]
                dst_local = global_to_local_maps[dst_type_key][dst_global]
            except KeyError:
                print(f"⚠️ 警告：无法找到节点的局部索引映射")
                continue
            
            # 提取边特征 (去除类型列)
            edge_feature_cols = [col for col in df_edge_features.columns if col != 'type']
            edge_attr_values = row[edge_feature_cols].values.astype(np.float32)
            edge_attr = torch.tensor(edge_attr_values, dtype=torch.float32).unsqueeze(0)

            # 在data对象中初始化该关系类型的存储
            if relation_tuple not in data:
                data[relation_tuple].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[relation_tuple].edge_attr = torch.empty((0, len(edge_feature_cols)), dtype=torch.float32)
            
            # 添加边 (注意：根据排序后的元组，决定src和dst哪个在前)
            if src_type_key == sorted_node_keys[0]:
                edge_pair = torch.tensor([[src_local], [dst_local]], dtype=torch.long)
            else:
                edge_pair = torch.tensor([[dst_local], [src_local]], dtype=torch.long)

            data[relation_tuple].edge_index = torch.cat([data[relation_tuple].edge_index, edge_pair], dim=1)
            data[relation_tuple].edge_attr = torch.cat([data[relation_tuple].edge_attr, edge_attr], dim=0)

        # 打印关系统计信息
        print("🌐 异构图关系统计:")
        for relation, count in relation_stats.items():
            print(f"  🔗 {relation}: {count} 条边")

        # --- 3. 创建无向图 ---
        # PyG的ToUndirected可以方便地为异构图创建反向边
        try:
            from torch_geometric.transforms import ToUndirected
            data = ToUndirected()(data)
            print("✅ 成功创建无向异构图")
        except Exception as e:
            print(f"⚠️ 警告：无法使用ToUndirected转换，使用当前的有向图: {e}")

        return data

    # 为了向后兼容，保留原来的create_pyg_data方法
    def create_pyg_data(self, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, 
                       df_edge_features: pd.DataFrame):
        """
        向后兼容的方法，现在重定向到异构图创建
        """
        print("⚠️ 警告：create_pyg_data已弃用，请使用create_pyg_hetero_data")
        return self.create_pyg_hetero_data(df_nodes, df_edges, df_edge_features)

