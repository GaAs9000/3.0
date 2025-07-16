"""
高性能PPO内存缓冲区 - 简化版本

专注于核心性能优化，去除复杂的缓存机制，
直接针对PPO训练的实际使用模式进行优化。

主要优化：
1. 预分配张量存储标量数据
2. 最小化数据拷贝和设备转换
3. 优化批次获取操作
4. 保持完全的接口兼容性
"""

import torch
from typing import Dict, List, Tuple, Any, Optional
from .decision_context import AbstractDecisionContext


class FastPPOMemory:
    """
    高性能PPO内存缓冲区
    
    专注于实际训练场景的性能优化，去除不必要的复杂性。
    完全兼容原始PPOMemory接口。
    """
    
    def __init__(self, 
                 capacity: int = 2048,
                 device: torch.device = None):
        """
        初始化高性能内存缓冲区
        
        Args:
            capacity: 缓冲区容量
            device: 计算设备
        """
        self.capacity = capacity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 简单的线性存储，不使用环形缓冲区
        self.size = 0
        
        # 状态存储（保持原始格式）
        self.states = []
        
        # 抽象决策上下文存储（v3.0架构）
        self.decision_contexts = []
        
        # 保持向后兼容的动作存储
        self.actions = []  # 将被逐步淘汰
        
        # 预分配张量用于标量数据
        self.rewards_buffer = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.log_probs_buffer = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.values_buffer = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.dones_buffer = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        
    def store(self, state: Dict[str, torch.Tensor], 
              action: Optional[Tuple[int, int]] = None,
              decision_context: Optional[AbstractDecisionContext] = None,
              reward: float = 0.0, log_prob: float = 0.0, 
              value: float = 0.0, done: bool = False):
        """
        存储经验数据 (支持v3.0抽象决策上下文)
        
        Args:
            state: 状态字典
            action: 动作元组 (node_idx, partition_idx) - 向后兼容，将被淘汰
            decision_context: 抽象决策上下文 - v3.0架构的核心
            reward: 奖励值
            log_prob: 对数概率 (在v3.0中被decision_context替代)
            value: 状态价值
            done: 是否结束
        """
        if self.size >= self.capacity:
            raise RuntimeError(f"内存缓冲区已满 (容量: {self.capacity})")
            
        # 确保传入了必要的参数
        if decision_context is None and action is None:
            raise ValueError("必须提供 decision_context 或 action 参数之一")
            
        # 存储状态
        self.states.append(state)
        
        # 优先存储决策上下文（v3.0架构）
        if decision_context is not None:
            if hasattr(decision_context, 'to_cpu'):
                self.decision_contexts.append(decision_context.to_cpu().detach())
            else:
                # 如果decision_context是字典或其他类型，转换为适当格式
                self.decision_contexts.append(decision_context)
            # 保持向后兼容：从决策上下文提取动作用于调试
            self.actions.append(None)  # 占位符，表示使用抽象表示
        else:
            # 向后兼容：存储具体动作
            self.decision_contexts.append(None)
            self.actions.append(action)
        
        # 存储标量数据到预分配张量
        self.rewards_buffer[self.size] = reward
        self.log_probs_buffer[self.size] = log_prob  # v3.0中将从context重新计算
        self.values_buffer[self.size] = value
        self.dones_buffer[self.size] = done
        
        self.size += 1
        
    def clear(self):
        """清除内存缓冲区"""
        self.size = 0
        self.states.clear()
        self.actions.clear()
        self.decision_contexts.clear()
        
        # 不需要清零张量，因为size控制有效范围
        
    def get_batch(self) -> Tuple[List, List, List, List, List, List]:
        """
        获取经验批次（兼容原始接口，但优先使用decision_contexts）
        
        Returns:
            (states, actions_or_contexts, rewards, log_probs, values, dones)
            注意：actions_or_contexts可能包含AbstractDecisionContext对象或传统action tuple
        """
        if self.size == 0:
            return [], [], [], [], [], []
            
        # 直接返回列表（已经是正确格式）
        states = self.states
        
        # 优先返回决策上下文，回退到传统动作
        actions_or_contexts = []
        for i in range(self.size):
            if self.decision_contexts[i] is not None:
                actions_or_contexts.append(self.decision_contexts[i])
            else:
                actions_or_contexts.append(self.actions[i])
        
        # 保持向后兼容
        actions = actions_or_contexts
        
        # 高效的标量数据转换
        if self.size > 0:
            # 使用切片获取有效数据，然后批量转换
            rewards_slice = self.rewards_buffer[:self.size]
            log_probs_slice = self.log_probs_buffer[:self.size]
            values_slice = self.values_buffer[:self.size]
            dones_slice = self.dones_buffer[:self.size]
            
            # 批量转换为列表（如果在GPU上，先转到CPU）
            if self.device.type == 'cuda':
                rewards = rewards_slice.cpu().tolist()
                log_probs = log_probs_slice.cpu().tolist()
                values = values_slice.cpu().tolist()
                dones = dones_slice.cpu().tolist()
            else:
                rewards = rewards_slice.tolist()
                log_probs = log_probs_slice.tolist()
                values = values_slice.tolist()
                dones = dones_slice.tolist()
        else:
            rewards = []
            log_probs = []
            values = []
            dones = []
        
        return states, actions, rewards, log_probs, values, dones
    
    def get_decision_contexts_batch(self) -> Tuple[List, List, List, List, List]:
        """
        获取v3.0架构的决策上下文批次
        
        Returns:
            (states, decision_contexts, rewards, values, dones)
            注意：log_probs从decision_contexts重新计算，不再使用存储的值
        """
        if self.size == 0:
            return [], [], [], [], []
            
        states = self.states
        decision_contexts = []
        
        # 过滤出包含决策上下文的经验
        valid_indices = []
        for i in range(self.size):
            if self.decision_contexts[i] is not None:
                decision_contexts.append(self.decision_contexts[i])
                valid_indices.append(i)
        
        if not valid_indices:
            return [], [], [], [], []
        
        # 获取对应的其他数据
        valid_states = [states[i] for i in valid_indices]
        
        # 高效获取标量数据
        valid_rewards = [self.rewards_buffer[i].item() for i in valid_indices]
        valid_values = [self.values_buffer[i].item() for i in valid_indices]
        valid_dones = [self.dones_buffer[i].item() for i in valid_indices]
        
        return valid_states, decision_contexts, valid_rewards, valid_values, valid_dones
    
    def has_decision_contexts(self) -> bool:
        """检查是否包含决策上下文数据"""
        return any(ctx is not None for ctx in self.decision_contexts[:self.size])
    
    def save_to_disk(self, filepath: str):
        """
        将内存缓冲区保存到磁盘（支持DecisionContext序列化）
        
        Args:
            filepath: 保存路径
        """
        import torch
        import pickle
        
        # 准备保存数据
        save_data = {
            'capacity': self.capacity,
            'size': self.size,
            'device': self.device.type,
            'states': self.states.copy(),
            'actions': self.actions.copy(),
            'decision_contexts': [],
            'rewards': self.rewards_buffer[:self.size].cpu().numpy() if self.size > 0 else [],
            'log_probs': self.log_probs_buffer[:self.size].cpu().numpy() if self.size > 0 else [],
            'values': self.values_buffer[:self.size].cpu().numpy() if self.size > 0 else [],
            'dones': self.dones_buffer[:self.size].cpu().numpy() if self.size > 0 else [],
        }
        
        # 序列化决策上下文（需要特殊处理torch.Tensor）
        for ctx in self.decision_contexts[:self.size]:
            if ctx is not None:
                # 将DecisionContext转换为可序列化的字典
                ctx_dict = {
                    'node_embedding': ctx.node_embedding.cpu().numpy(),
                    'strategy_vector': ctx.strategy_vector.cpu().numpy(),
                    'target_partition_embedding': ctx.target_partition_embedding.cpu().numpy(),
                    'candidate_embeddings': ctx.candidate_embeddings.cpu().numpy(),
                    'selection_similarities': ctx.selection_similarities.cpu().numpy(),
                    'selected_rank': ctx.selected_rank,
                    'decision_timestamp': ctx.decision_timestamp,
                    'num_candidates': ctx.num_candidates,
                    'node_context': ctx.node_context,
                    'environment_state_hash': ctx.environment_state_hash
                }
                save_data['decision_contexts'].append(ctx_dict)
            else:
                save_data['decision_contexts'].append(None)
        
        # 保存到文件
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_from_disk(self, filepath: str):
        """
        从磁盘加载内存缓冲区（支持DecisionContext反序列化）
        
        Args:
            filepath: 文件路径
        """
        import torch
        import pickle
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # 恢复基本属性
        self.capacity = save_data['capacity']
        self.size = save_data['size']
        device_type = save_data.get('device', 'cpu')
        self.device = torch.device(device_type if torch.cuda.is_available() or device_type == 'cpu' else 'cpu')
        
        # 重新分配张量缓冲区
        self.rewards_buffer = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.log_probs_buffer = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.values_buffer = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.dones_buffer = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        
        # 恢复数据
        self.states = save_data['states']
        self.actions = save_data['actions']
        
        # 恢复张量数据
        if self.size > 0:
            rewards_np = save_data['rewards']
            log_probs_np = save_data['log_probs']
            values_np = save_data['values']
            dones_np = save_data['dones']
            
            self.rewards_buffer[:self.size] = torch.from_numpy(rewards_np).to(self.device)
            self.log_probs_buffer[:self.size] = torch.from_numpy(log_probs_np).to(self.device)
            self.values_buffer[:self.size] = torch.from_numpy(values_np).to(self.device)
            self.dones_buffer[:self.size] = torch.from_numpy(dones_np).to(self.device)
        
        # 恢复决策上下文
        self.decision_contexts = []
        for ctx_data in save_data['decision_contexts']:
            if ctx_data is not None:
                # 重建AbstractDecisionContext
                ctx = AbstractDecisionContext(
                    node_embedding=torch.from_numpy(ctx_data['node_embedding']).to(self.device),
                    strategy_vector=torch.from_numpy(ctx_data['strategy_vector']).to(self.device),
                    target_partition_embedding=torch.from_numpy(ctx_data['target_partition_embedding']).to(self.device),
                    candidate_embeddings=torch.from_numpy(ctx_data['candidate_embeddings']).to(self.device),
                    selection_similarities=torch.from_numpy(ctx_data['selection_similarities']).to(self.device),
                    selected_rank=ctx_data['selected_rank'],
                    decision_timestamp=ctx_data['decision_timestamp'],
                    num_candidates=ctx_data['num_candidates'],
                    node_context=ctx_data['node_context'],
                    environment_state_hash=ctx_data['environment_state_hash']
                )
                self.decision_contexts.append(ctx)
            else:
                self.decision_contexts.append(None)
    
    def get_serialization_stats(self) -> Dict[str, Any]:
        """
        获取序列化统计信息
        
        Returns:
            包含内存使用和决策上下文统计的字典
        """
        stats = {
            'total_size': self.size,
            'capacity': self.capacity,
            'decision_contexts_count': sum(1 for ctx in self.decision_contexts[:self.size] if ctx is not None),
            'legacy_actions_count': sum(1 for action in self.actions[:self.size] if action is not None),
            'device': self.device.type,
            'memory_usage_mb': self._estimate_memory_usage()
        }
        
        # 决策上下文详细统计
        if self.has_decision_contexts():
            ctx_stats = self._analyze_decision_contexts()
            stats.update(ctx_stats)
        
        return stats
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        total_bytes = 0
        
        # 张量缓冲区
        total_bytes += self.rewards_buffer.numel() * self.rewards_buffer.element_size()
        total_bytes += self.log_probs_buffer.numel() * self.log_probs_buffer.element_size()
        total_bytes += self.values_buffer.numel() * self.values_buffer.element_size()
        total_bytes += self.dones_buffer.numel() * self.dones_buffer.element_size()
        
        # 决策上下文（粗略估算）
        for ctx in self.decision_contexts[:self.size]:
            if ctx is not None:
                total_bytes += ctx.node_embedding.numel() * ctx.node_embedding.element_size()
                total_bytes += ctx.strategy_vector.numel() * ctx.strategy_vector.element_size()
                total_bytes += ctx.target_partition_embedding.numel() * ctx.target_partition_embedding.element_size()
                total_bytes += ctx.candidate_embeddings.numel() * ctx.candidate_embeddings.element_size()
                total_bytes += ctx.selection_similarities.numel() * ctx.selection_similarities.element_size()
        
        return total_bytes / (1024 * 1024)  # 转换为MB
    
    def _analyze_decision_contexts(self) -> Dict[str, Any]:
        """分析决策上下文的统计信息"""
        contexts = [ctx for ctx in self.decision_contexts[:self.size] if ctx is not None]
        
        if not contexts:
            return {}
        
        # 嵌入维度统计
        node_embed_dims = [ctx.node_embedding.size(0) for ctx in contexts]
        strategy_dims = [ctx.strategy_vector.size(0) for ctx in contexts]
        partition_embed_dims = [ctx.target_partition_embedding.size(0) for ctx in contexts]
        candidate_counts = [ctx.num_candidates for ctx in contexts]
        
        return {
            'avg_node_embedding_dim': sum(node_embed_dims) / len(node_embed_dims),
            'avg_strategy_dim': sum(strategy_dims) / len(strategy_dims),
            'avg_partition_embedding_dim': sum(partition_embed_dims) / len(partition_embed_dims),
            'avg_candidates_count': sum(candidate_counts) / len(candidate_counts),
            'max_candidates_count': max(candidate_counts),
            'min_candidates_count': min(candidate_counts)
        }
        
    def get_batch_tensors(self) -> Tuple[List, List, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取经验批次（张量优化版本）
        
        Returns:
            (states, actions, rewards_tensor, log_probs_tensor, values_tensor, dones_tensor)
        """
        if self.size == 0:
            empty_tensor = torch.empty(0, device=self.device)
            return [], [], empty_tensor, empty_tensor, empty_tensor, empty_tensor.bool()
            
        # 直接返回张量切片（避免CPU转换）
        states = self.states
        actions = self.actions
        rewards_tensor = self.rewards_buffer[:self.size]
        log_probs_tensor = self.log_probs_buffer[:self.size]
        values_tensor = self.values_buffer[:self.size]
        dones_tensor = self.dones_buffer[:self.size]
        
        return states, actions, rewards_tensor, log_probs_tensor, values_tensor, dones_tensor
        
    def __len__(self) -> int:
        """返回当前存储的经验数量"""
        return self.size
        
    @property
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.size >= self.capacity
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用统计"""
        tensor_memory = (
            self.rewards_buffer.numel() * self.rewards_buffer.element_size() +
            self.log_probs_buffer.numel() * self.log_probs_buffer.element_size() +
            self.values_buffer.numel() * self.values_buffer.element_size() +
            self.dones_buffer.numel() * self.dones_buffer.element_size()
        )
        
        return {
            'capacity': self.capacity,
            'current_size': self.size,
            'utilization': self.size / self.capacity if self.capacity > 0 else 0,
            'tensor_memory_bytes': tensor_memory,
            'device': str(self.device),
            'implementation': 'FastPPOMemory'
        }


class HybridPPOMemory:
    """
    混合PPO内存缓冲区
    
    根据数据量自动选择最优的存储策略：
    - 小数据量：使用原始列表存储
    - 大数据量：使用张量化存储
    """
    
    def __init__(self, 
                 capacity: int = 2048,
                 device: torch.device = None,
                 tensor_threshold: int = 100):
        """
        初始化混合内存缓冲区
        
        Args:
            capacity: 缓冲区容量
            device: 计算设备
            tensor_threshold: 切换到张量存储的阈值
        """
        self.capacity = capacity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_threshold = tensor_threshold
        
        # 当前使用的存储模式
        self.use_tensor_storage = False
        
        # 列表存储（小数据量）
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # 张量存储（大数据量）
        self.tensor_storage = None
        
    def _switch_to_tensor_storage(self):
        """切换到张量存储模式"""
        if not self.use_tensor_storage:
            self.tensor_storage = FastPPOMemory(self.capacity, self.device)
            
            # 迁移现有数据
            for i in range(len(self.states)):
                self.tensor_storage.store(
                    self.states[i], self.actions[i], self.rewards[i],
                    self.log_probs[i], self.values[i], self.dones[i]
                )
                
            # 清空列表存储
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.log_probs.clear()
            self.values.clear()
            self.dones.clear()
            
            self.use_tensor_storage = True
            
    def store(self, state: Dict[str, torch.Tensor], action: Tuple[int, int], 
              reward: float, log_prob: float, value: float, done: bool):
        """存储经验数据"""
        if self.use_tensor_storage:
            self.tensor_storage.store(state, action, reward, log_prob, value, done)
        else:
            # 使用列表存储
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.dones.append(done)
            
            # 检查是否需要切换到张量存储
            if len(self.states) >= self.tensor_threshold:
                self._switch_to_tensor_storage()
                
    def clear(self):
        """清除内存缓冲区"""
        if self.use_tensor_storage and self.tensor_storage:
            self.tensor_storage.clear()
        else:
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.log_probs.clear()
            self.values.clear()
            self.dones.clear()
            
        self.use_tensor_storage = False
        self.tensor_storage = None
        
    def get_batch(self) -> Tuple[List, List, List, List, List, List]:
        """获取经验批次"""
        if self.use_tensor_storage and self.tensor_storage:
            return self.tensor_storage.get_batch()
        else:
            return (self.states, self.actions, self.rewards, 
                   self.log_probs, self.values, self.dones)
                   
    def __len__(self) -> int:
        """返回当前存储的经验数量"""
        if self.use_tensor_storage and self.tensor_storage:
            return len(self.tensor_storage)
        else:
            return len(self.states)
