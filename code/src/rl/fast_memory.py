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
from typing import Dict, List, Tuple, Any


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
        
        # 状态和动作存储（保持原始格式）
        self.states = []
        self.actions = []
        
        # 预分配张量用于标量数据
        self.rewards_buffer = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.log_probs_buffer = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.values_buffer = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.dones_buffer = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        
    def store(self, state: Dict[str, torch.Tensor], action: Tuple[int, int], 
              reward: float, log_prob: float, value: float, done: bool):
        """
        存储经验数据
        
        Args:
            state: 状态字典
            action: 动作元组 (node_idx, partition_idx)
            reward: 奖励值
            log_prob: 对数概率
            value: 状态价值
            done: 是否结束
        """
        if self.size >= self.capacity:
            raise RuntimeError(f"内存缓冲区已满 (容量: {self.capacity})")
            
        # 直接存储状态和动作（避免不必要的拷贝）
        self.states.append(state)
        self.actions.append(action)
        
        # 存储标量数据到预分配张量
        self.rewards_buffer[self.size] = reward
        self.log_probs_buffer[self.size] = log_prob
        self.values_buffer[self.size] = value
        self.dones_buffer[self.size] = done
        
        self.size += 1
        
    def clear(self):
        """清除内存缓冲区"""
        self.size = 0
        self.states.clear()
        self.actions.clear()
        
        # 不需要清零张量，因为size控制有效范围
        
    def get_batch(self) -> Tuple[List, List, List, List, List, List]:
        """
        获取经验批次（兼容原始接口）
        
        Returns:
            (states, actions, rewards, log_probs, values, dones)
        """
        if self.size == 0:
            return [], [], [], [], [], []
            
        # 直接返回列表（已经是正确格式）
        states = self.states
        actions = self.actions
        
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
