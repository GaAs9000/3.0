import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
import hashlib
import pickle
import os
from pathlib import Path

class PowerGridDataProcessor:
    """
    电网数据处理 - 简化异构图版本
    
    核心改进：
    1. 统一节点类型为'bus'，将物理类型(PQ/PV/Slack)转换为独热编码特征
    2. 统一边关系为('bus', 'connects', 'bus')，将物理类型(线路/变压器)转换为边特征
    3. 支持GNN权重共享和高效学习
    4. 保持哈希缓存机制
    
    主要功能：
    1. MATPOWER格式数据清洗和特征提取
    2. 构建简化的PyTorch Geometric异构图数据
    3. 将物理类型信息特征化而非拓扑化
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

    def graph_from_mpc(self, mpc: Dict, config: Dict[str, Any] = None) -> HeteroData:
        """
        将MATPOWER格式数据转换为简化的PyTorch Geometric异构图数据 (HeteroData)

        参数:
            mpc: MATPOWER格式的电网数据字典
            config: 配置字典，用于控制输出详细程度

        返回:
            data: 简化的PyTorch Geometric HeteroData对象
        """
        # 获取调试配置
        debug_config = config.get('debug', {}) if config else {}
        training_output = debug_config.get('training_output', {})
        show_cache_loading = training_output.get('show_cache_loading', True)
        only_show_errors = training_output.get('only_show_errors', False)
        # 1. 计算数据哈希用于缓存 (文件名中加入v3后缀以区分新版本)
        raw_bytes = pickle.dumps((mpc["bus"].tolist(), mpc["branch"].tolist()))
        case_hash = hashlib.md5(raw_bytes).hexdigest()[:8]
        cache_file = f"{self.cache_dir}/{case_hash}_hetero_v3.pt"
        
        # 2. 尝试从缓存加载（带进程锁保护）
        lock_file = cache_file + ".lock"
        if os.path.exists(cache_file):
            try:
                # 等待锁文件释放（最多等待30秒）
                wait_count = 0
                while os.path.exists(lock_file) and wait_count < 30:
                    import time
                    time.sleep(1)
                    wait_count += 1
                
                if os.path.exists(cache_file):
                    if show_cache_loading and not only_show_errors:
                        try:
                            from rich_output import rich_debug
                            rich_debug(f"从缓存加载简化异构图: {cache_file}", "cache")
                        except ImportError:
                            pass
                    return torch.load(cache_file, map_location="cpu", weights_only=False)
            except Exception as e:
                try:
                    from rich_output import rich_warning
                    rich_warning(f"缓存加载失败: {e}，重新构建...")
                except ImportError:
                    print(f"⚠️ 缓存加载失败: {e}，重新构建...")
        
        # 3. 首次构建简化异构图数据
        if not only_show_errors:
            try:
                from rich_output import rich_debug
                rich_debug("首次构建简化异构图数据...", "cache")
            except ImportError:
                pass
        baseMVA, df_nodes, df_edges, df_edge_features = self._process_matpower_data(mpc)
        data = self._create_simplified_hetero_data(df_nodes, df_edges, df_edge_features)

        # 4. 保存到缓存（带进程锁保护）
        try:
            # 创建锁文件
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))

            torch.save(data, cache_file, pickle_protocol=pickle.DEFAULT_PROTOCOL)
            if not only_show_errors:
                try:
                    from rich_output import rich_debug
                    rich_debug(f"已缓存简化异构图到: {cache_file}", "cache")
                except ImportError:
                    pass
            
            # 删除锁文件
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            try:
                from rich_output import rich_warning
                rich_warning(f"缓存保存失败: {e}")
            except ImportError:
                print(f"⚠️ 缓存保存失败: {e}")
            if os.path.exists(lock_file):
                os.remove(lock_file)
        
        return data   
    
    def _process_matpower_data(self, mpc: Dict) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        处理MATPOWER格式数据，为构建简化异构图准备包含类型信息的DataFrame
        
        参数:
            mpc: MATPOWER格式的电网数据字典
            
        返回:
            baseMVA: 基准功率(MVA)
            df_nodes: 包含bus_type信息的节点特征DataFrame
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
            
        # 6. 创建包含bus_type信息的DataFrame，保留原始数值类型用于独热编码
        # 定义节点特征列名
        node_columns = ['Pd', 'Qd', 'Gs', 'Bs', 'Vm', 'Va', 'Vmax', 'Vmin', 'degree']
        if 'gen' in mpc:
            node_columns.extend(['Pg', 'Qg', 'Pg_max', 'Pg_min', 'is_gen'])
            
        # 使用0-based的母线索引作为DataFrame的索引
        df_nodes = pd.DataFrame(node_features, columns=node_columns, index=bus[:,0].astype(int)-1)
        
        # 添加bus_type列（保留数值1,2,3用于独热编码）
        bus_types_raw = bus[:, 1].astype(int)
        df_nodes['bus_type'] = bus_types_raw

        # 定义边特征列名
        df_edge_features = pd.DataFrame(
            edge_features,
            columns=['r', 'x', 'b', '|z|', 'y', 'rateA', 'angle_diff', 'is_transformer', 'status']
        )
        
        # 创建边索引DataFrame
        df_edges = pd.DataFrame(edge_index, columns=['from_bus', 'to_bus'])
        
        return baseMVA, df_nodes, df_edges, df_edge_features

    def _create_simplified_hetero_data(self, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, 
                                     df_edge_features: pd.DataFrame) -> HeteroData:
        """
        创建简化的异构图数据结构
        
        关键特性：
        1. 统一节点类型为'bus'
        2. 将bus_type转换为独热编码特征
        3. 统一边关系为('bus', 'connects', 'bus')
        4. 将is_transformer作为边特征保留
        
        参数:
            df_nodes: 包含bus_type信息的节点特征DataFrame
            df_edges: 边索引DataFrame
            df_edge_features: 边特征DataFrame
            
        返回:
            data: 简化的PyTorch Geometric HeteroData对象
        """
        data = HeteroData()
        
        # --- 1. 处理节点：统一类型 + 独热编码 ---
        try:
            from rich_output import rich_debug
            rich_debug(f"节点类型分布: {df_nodes['bus_type'].value_counts().to_dict()}", "cache")
        except ImportError:
            pass
        
        # 创建bus_type的独热编码
        bus_type_dummies = pd.get_dummies(df_nodes['bus_type'], prefix='type')
        
        # 确保包含所有可能的类型（type_1, type_2, type_3）
        for bus_type in [1, 2, 3]:
            col_name = f'type_{bus_type}'
            if col_name not in bus_type_dummies.columns:
                bus_type_dummies[col_name] = 0
        
        # 按照列名排序确保一致性
        bus_type_dummies = bus_type_dummies.reindex(sorted(bus_type_dummies.columns), axis=1)
        
        # 获取原始数值特征（去除bus_type列）
        original_features = df_nodes.drop('bus_type', axis=1)
        
        # 合并原始特征和独热编码类型特征
        combined_features = pd.concat([original_features, bus_type_dummies], axis=1)
        
        # 创建特征名称到索引的映射，供下游代码使用
        feature_names = combined_features.columns.tolist()
        feature_index_map = {name: idx for idx, name in enumerate(feature_names)}
        
        # 创建统一的'bus'节点类型
        data['bus'].x = torch.tensor(combined_features.values.astype(np.float32), dtype=torch.float32)
        data['bus'].global_ids = torch.tensor(df_nodes.index.tolist(), dtype=torch.long)
        
        # 存储特征映射信息供下游使用
        data['bus'].feature_names = feature_names
        data['bus'].feature_index_map = feature_index_map
        
        try:
            from rich_output import rich_debug
            rich_debug(f"bus: {len(df_nodes)} 个节点，特征维度: {combined_features.shape[1]}", "cache")
            rich_debug(f"原始数值特征: {original_features.shape[1]}", "cache")
            rich_debug(f"独热编码类型特征: {bus_type_dummies.shape[1]}", "cache")
            rich_debug(f"特征顺序: {feature_names}", "cache")
        except ImportError:
            pass
        
        # --- 2. 处理边：统一关系类型 ---
        try:
            from rich_output import rich_debug
            rich_debug(f"边类型分布: is_transformer = {df_edge_features['is_transformer'].value_counts().to_dict()}", "cache")
        except ImportError:
            pass
        
        # 构建边索引张量
        edge_index = torch.tensor(df_edges[['from_bus', 'to_bus']].values.T, dtype=torch.long)
        
        # 构建边特征张量（已包含is_transformer特征）
        edge_attr = torch.tensor(df_edge_features.values.astype(np.float32), dtype=torch.float32)
        
        # 创建边特征映射
        edge_feature_names = df_edge_features.columns.tolist()
        edge_feature_index_map = {name: idx for idx, name in enumerate(edge_feature_names)}
        
        # 创建统一的边关系
        data['bus', 'connects', 'bus'].edge_index = edge_index
        data['bus', 'connects', 'bus'].edge_attr = edge_attr
        data['bus', 'connects', 'bus'].edge_feature_names = edge_feature_names
        data['bus', 'connects', 'bus'].edge_feature_index_map = edge_feature_index_map
        
        try:
            from rich_output import rich_debug
            rich_debug(f"('bus', 'connects', 'bus'): {edge_index.shape[1]} 条边，特征维度: {edge_attr.shape[1]}", "cache")
            rich_debug(f"边特征顺序: {edge_feature_names}", "cache")
        except ImportError:
            pass
        
        # --- 3. 创建无向图 ---
        try:
            from torch_geometric.transforms import ToUndirected
            data = ToUndirected()(data)
            try:
                from rich_output import rich_success, rich_debug
                rich_success("成功创建无向简化异构图")
                rich_debug(f"无向化后边数: {data['bus', 'connects', 'bus'].edge_index.shape[1]}", "cache")
            except ImportError:
                pass
        except Exception as e:
            try:
                from rich_output import rich_warning
                rich_warning(f"无法使用ToUndirected转换: {e}")
            except ImportError:
                print(f"⚠️ 警告：无法使用ToUndirected转换: {e}")

        return data

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
        
        # 2. 电压约束 - 智能设置基于网络特性
        Vmax = bus[:, 11] if bus.shape[1] > 11 else self._estimate_voltage_constraints(bus, baseMVA, 'max')
        Vmin = bus[:, 12] if bus.shape[1] > 12 else self._estimate_voltage_constraints(bus, baseMVA, 'min')
        
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
        # 检查多个可能的变压器指标
        tap_ratio = branch[:, 8]
        phase_shift = branch[:, 9] if branch.shape[1] > 9 else np.zeros(len(branch))
        
        # 变压器识别：tap_ratio != 1 或 phase_shift != 0 或 显式变压器标记
        is_transformer_tap = (tap_ratio != 1.0).astype(float)
        is_transformer_phase = (phase_shift != 0.0).astype(float)
        
        # 如果有显式的变压器标记列（column[9]），使用它
        if branch.shape[1] > 9:
            is_transformer_explicit = branch[:, 9].astype(float)
            is_transformer = np.maximum(np.maximum(is_transformer_tap, is_transformer_phase), is_transformer_explicit)
        else:
            is_transformer = np.maximum(is_transformer_tap, is_transformer_phase)
            
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

    # 向后兼容方法
    def process_matpower_data(self, mpc: Dict) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        向后兼容的方法，现在调用私有的_process_matpower_data方法
        """
        print("⚠️ 提示：process_matpower_data现在使用简化异构图实现")
        return self._process_matpower_data(mpc)
    
    def create_pyg_hetero_data(self, df_nodes: pd.DataFrame, df_edges: pd.DataFrame, 
                               df_edge_features: pd.DataFrame) -> HeteroData:
        """
        向后兼容的方法，现在调用简化的异构图创建方法
        """
        print("⚠️ 提示：create_pyg_hetero_data现在使用简化异构图实现")
        return self._create_simplified_hetero_data(df_nodes, df_edges, df_edge_features)

    def create_pyg_data(self, df_nodes: pd.DataFrame, df_edges: pd.DataFrame,
                       df_edge_features: pd.DataFrame):
        """
        向后兼容的方法，现在重定向到简化异构图创建
        """
        print("⚠️ 提示：create_pyg_data现在使用简化异构图实现")
        return self._create_simplified_hetero_data(df_nodes, df_edges, df_edge_features)

    def _estimate_voltage_constraints(self, bus: np.ndarray, baseMVA: float, constraint_type: str) -> np.ndarray:
        """
        基于网络特性智能估算电压约束

        参数:
            bus: 母线数据矩阵
            baseMVA: 基准功率
            constraint_type: 'max' 或 'min'

        返回:
            电压约束数组
        """
        n_bus = bus.shape[0]

        # 1. 分析负载水平
        total_load = np.sum(np.abs(bus[:, 2:4]))  # 有功+无功负载
        avg_load_per_bus = total_load / (n_bus * baseMVA)

        # 2. 分析电压等级（如果有baseKV信息）
        if bus.shape[1] > 9:  # 有baseKV列
            base_kv = bus[:, 9]
            # 根据电压等级分类
            high_voltage_mask = base_kv >= 220  # 高压网络
            medium_voltage_mask = (base_kv >= 35) & (base_kv < 220)  # 中压网络
            low_voltage_mask = base_kv < 35  # 低压网络
        else:
            # 如果没有电压等级信息，基于负载水平推断
            high_voltage_mask = avg_load_per_bus > 0.5  # 高负载假设为高压
            medium_voltage_mask = (avg_load_per_bus >= 0.1) & (avg_load_per_bus <= 0.5)
            low_voltage_mask = avg_load_per_bus < 0.1

        # 3. 分析节点类型（如果有）
        bus_types = bus[:, 1].astype(int)
        slack_mask = (bus_types == 3)  # 平衡节点
        pv_mask = (bus_types == 2)     # PV节点
        pq_mask = (bus_types == 1)     # PQ节点

        # 4. 设置约束值
        if constraint_type == 'max':
            constraints = np.ones(n_bus) * 1.1  # 默认值

            # 高压网络：更严格的约束
            constraints[high_voltage_mask] = 1.05

            # 中压网络：标准约束
            constraints[medium_voltage_mask] = 1.1

            # 低压配电网：更宽松的约束
            constraints[low_voltage_mask] = 1.15

            # 平衡节点：更严格的约束
            constraints[slack_mask] = 1.05

            # PV节点：中等约束
            constraints[pv_mask] = 1.08

        else:  # constraint_type == 'min'
            constraints = np.ones(n_bus) * 0.9  # 默认值

            # 高压网络：更严格的约束
            constraints[high_voltage_mask] = 0.95

            # 中压网络：标准约束
            constraints[medium_voltage_mask] = 0.9

            # 低压配电网：更宽松的约束
            constraints[low_voltage_mask] = 0.85

            # 平衡节点：更严格的约束
            constraints[slack_mask] = 0.95

            # PV节点：中等约束
            constraints[pv_mask] = 0.92

        # 5. 基于负载密度进行微调
        load_density = np.abs(bus[:, 2]) / baseMVA  # 有功负载密度

        if constraint_type == 'max':
            # 高负载区域：稍微放宽上限
            high_load_mask = load_density > np.percentile(load_density, 80)
            constraints[high_load_mask] += 0.02
        else:
            # 高负载区域：稍微提高下限
            high_load_mask = load_density > np.percentile(load_density, 80)
            constraints[high_load_mask] += 0.02

        # 6. 确保约束在合理范围内
        if constraint_type == 'max':
            constraints = np.clip(constraints, 1.02, 1.2)
        else:
            constraints = np.clip(constraints, 0.8, 0.98)

        return constraints

