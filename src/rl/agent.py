"""
PPO Agent for Power Grid Partitioning

This module implements a Proximal Policy Optimization (PPO) agent
specifically designed for the power grid partitioning MDP with:
- Heterogeneous graph state representation
- Two-stage action space (node selection + partition selection)
- Action masking for constraint enforcement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import copy

from ..gat import HeteroGraphEncoder


class ActorNetwork(nn.Module):
    """
    Actor network for two-stage action selection
    
    Takes heterogeneous graph state and outputs:
    1. Node selection probabilities (over boundary nodes)
    2. Partition selection probabilities (for each node-partition pair)
    """
    
    def __init__(self,
                 node_embedding_dim: int,
                 region_embedding_dim: int,
                 num_partitions: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        """
        Initialize Actor Network
        
        Args:
            node_embedding_dim: Dimension of node embeddings
            region_embedding_dim: Dimension of region embeddings  
            num_partitions: Number of partitions
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.region_embedding_dim = region_embedding_dim
        self.num_partitions = num_partitions
        self.hidden_dim = hidden_dim
        
        # Node selection network
        self.node_selector = nn.Sequential(
            nn.Linear(node_embedding_dim + region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single output for node scoring
        )
        
        # Partition selection network
        self.partition_selector = nn.Sequential(
            nn.Linear(node_embedding_dim + region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_partitions)
        )
        
        # Global state encoder for context
        # Output dimension should match region_embedding_dim for concatenation
        self.global_encoder = nn.Sequential(
            nn.Linear(region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, region_embedding_dim)
        )
        
    def forward(self, 
                node_embeddings: torch.Tensor,
                region_embeddings: torch.Tensor,
                boundary_nodes: torch.Tensor,
                action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for action selection
        
        Args:
            node_embeddings: Node embeddings [total_nodes, node_dim]
            region_embeddings: Region embeddings [num_partitions, region_dim]
            boundary_nodes: Boundary node indices [num_boundary]
            action_mask: Action mask [total_nodes, num_partitions]
            
        Returns:
            node_logits: Node selection logits [num_boundary]
            partition_logits: Partition selection logits [num_boundary, num_partitions]
        """
        # Global context from region embeddings
        global_context = self.global_encoder(region_embeddings.mean(dim=0, keepdim=True))
        global_context = global_context.expand(len(boundary_nodes), -1)
        
        # Get boundary node embeddings
        boundary_embeddings = node_embeddings[boundary_nodes]  # [num_boundary, node_dim]
        
        # Combine node embeddings with global context
        combined_features = torch.cat([boundary_embeddings, global_context], dim=1)
        
        # Node selection logits
        node_logits = self.node_selector(combined_features).squeeze(-1)  # [num_boundary]
        
        # Partition selection logits for each boundary node
        partition_logits = self.partition_selector(combined_features)  # [num_boundary, num_partitions]
        
        # Apply action mask to partition logits
        if len(boundary_nodes) > 0:
            boundary_mask = action_mask[boundary_nodes]  # [num_boundary, num_partitions]
            partition_logits = partition_logits.masked_fill(~boundary_mask, float('-inf'))
        
        return node_logits, partition_logits


class CriticNetwork(nn.Module):
    """
    Critic network for value estimation
    
    Takes heterogeneous graph state and estimates state value
    """
    
    def __init__(self,
                 node_embedding_dim: int,
                 region_embedding_dim: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        """
        Initialize Critic Network
        
        Args:
            node_embedding_dim: Dimension of node embeddings
            region_embedding_dim: Dimension of region embeddings
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(region_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Boundary information encoder
        self.boundary_encoder = nn.Sequential(
            nn.Linear(node_embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self,
                node_embeddings: torch.Tensor,
                region_embeddings: torch.Tensor,
                boundary_nodes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for value estimation
        
        Args:
            node_embeddings: Node embeddings [total_nodes, node_dim]
            region_embeddings: Region embeddings [num_partitions, region_dim]
            boundary_nodes: Boundary node indices [num_boundary]
            
        Returns:
            state_value: Estimated state value [1]
        """
        # Encode global state from region embeddings
        global_state = self.state_encoder(region_embeddings.mean(dim=0, keepdim=True))
        
        # Encode boundary information
        if len(boundary_nodes) > 0:
            boundary_embeddings = node_embeddings[boundary_nodes]
            boundary_info = self.boundary_encoder(boundary_embeddings.mean(dim=0, keepdim=True))
        else:
            boundary_info = torch.zeros(1, self.boundary_encoder[-1].out_features, 
                                       device=node_embeddings.device)
        
        # Combine features
        combined_features = torch.cat([global_state, boundary_info], dim=1)
        
        # Estimate value
        value = self.value_head(combined_features)
        
        return value.squeeze(-1)


class PPOMemory:
    """
    Memory buffer for PPO training
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def store(self, state, action, reward, log_prob, value, done):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        """Clear memory"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
    def get_batch(self):
        """Get batch of experiences"""
        return (self.states, self.actions, self.rewards, 
                self.log_probs, self.values, self.dones)


class PPOAgent:
    """
    PPO Agent for Power Grid Partitioning
    
    Implements Proximal Policy Optimization with:
    - Two-stage action selection
    - Action masking
    - Heterogeneous graph state processing
    """
    
    def __init__(self,
                 node_embedding_dim: int,
                 region_embedding_dim: int,
                 num_partitions: int,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 device: torch.device = None):
        """
        Initialize PPO Agent
        
        Args:
            node_embedding_dim: Dimension of node embeddings
            region_embedding_dim: Dimension of region embeddings
            num_partitions: Number of partitions
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of PPO epochs
            entropy_coef: Entropy coefficient
            value_coef: Value loss coefficient
            device: Torch device
        """
        self.device = device or torch.device('cpu')
        self.num_partitions = num_partitions
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Networks
        self.actor = ActorNetwork(
            node_embedding_dim, region_embedding_dim, num_partitions
        ).to(self.device)
        
        self.critic = CriticNetwork(
            node_embedding_dim, region_embedding_dim
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Memory
        self.memory = PPOMemory()
        
        # Training stats
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100)
        }
        
    def select_action(self, state: Dict[str, torch.Tensor], training: bool = True) -> Tuple[Tuple[int, int], float, float]:
        """
        Select action using current policy
        
        Args:
            state: State observation
            training: Whether in training mode
            
        Returns:
            action: Selected action (node_idx, partition_idx)
            log_prob: Log probability of action
            value: State value estimate
        """
        node_embeddings = state['node_embeddings']
        region_embeddings = state['region_embeddings']
        boundary_nodes = state['boundary_nodes']
        
        # Get action mask
        action_mask = torch.zeros(
            node_embeddings.shape[0], self.num_partitions,
            dtype=torch.bool, device=self.device
        )
        
        # Set valid actions based on boundary nodes and current partition
        current_partition = state['current_partition']
        for node_idx in boundary_nodes:
            current_node_partition = current_partition[node_idx].item()
            # Allow moves to all other partitions (simplified)
            for p in range(self.num_partitions):
                if p + 1 != current_node_partition:  # +1 for 1-based partitions
                    action_mask[node_idx, p] = True
        
        with torch.no_grad():
            # Get network outputs
            node_logits, partition_logits = self.actor(
                node_embeddings, region_embeddings, boundary_nodes, action_mask
            )
            
            value = self.critic(node_embeddings, region_embeddings, boundary_nodes)
            
            if len(boundary_nodes) == 0:
                # No valid actions
                return None, 0.0, value.item()
            
            # Sample actions
            if training:
                # Sample from distributions
                node_probs = F.softmax(node_logits, dim=0)
                node_dist = torch.distributions.Categorical(node_probs)
                node_action = node_dist.sample()
                
                partition_probs = F.softmax(partition_logits[node_action], dim=0)
                partition_dist = torch.distributions.Categorical(partition_probs)
                partition_action = partition_dist.sample()
                
                # Compute log probability
                log_prob = node_dist.log_prob(node_action) + partition_dist.log_prob(partition_action)
            else:
                # Greedy selection
                node_action = torch.argmax(node_logits)
                partition_action = torch.argmax(partition_logits[node_action])
                log_prob = 0.0
            
            # Convert to actual indices
            selected_node = boundary_nodes[node_action].item()
            selected_partition = partition_action.item() + 1  # Convert to 1-based
            
            action = (selected_node, selected_partition)
            
        return action, log_prob.item() if hasattr(log_prob, 'item') else log_prob, value.item()
        
    def store_experience(self, state, action, reward, log_prob, value, done):
        """Store experience in memory"""
        self.memory.store(state, action, reward, log_prob, value, done)
        
    def update(self) -> Dict[str, float]:
        """
        Update networks using PPO
        
        Returns:
            Training statistics
        """
        if len(self.memory.states) == 0:
            return {}
            
        # Get batch
        states, actions, rewards, old_log_probs, old_values, dones = self.memory.get_batch()
        
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(old_values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(rewards, old_values, dones)
        
        # PPO update
        stats = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
        for _ in range(self.k_epochs):
            epoch_stats = self._ppo_epoch(states, actions, old_log_probs, advantages, returns)
            for key in stats:
                stats[key] += epoch_stats[key]
                
        # Average over epochs
        for key in stats:
            stats[key] /= self.k_epochs
            self.training_stats[key].append(stats[key])
            
        # Clear memory
        self.memory.clear()
        
        return stats
        
    def _compute_advantages(self, rewards, values, dones):
        """Compute advantages using GAE"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (~dones[t]).float() - values[t]
            gae = delta + self.gamma * 0.95 * (~dones[t]).float() * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
        
    def _ppo_epoch(self, states, actions, old_log_probs, advantages, returns):
        """Single PPO epoch"""
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            old_log_prob = old_log_probs[i]
            advantage = advantages[i]
            return_val = returns[i]
            
            # Get current policy outputs
            node_embeddings = state['node_embeddings']
            region_embeddings = state['region_embeddings']
            boundary_nodes = state['boundary_nodes']
            
            if len(boundary_nodes) == 0:
                continue
                
            # Action mask
            action_mask = torch.zeros(
                node_embeddings.shape[0], self.num_partitions,
                dtype=torch.bool, device=self.device
            )
            current_partition = state['current_partition']
            for node_idx in boundary_nodes:
                current_node_partition = current_partition[node_idx].item()
                for p in range(self.num_partitions):
                    if p + 1 != current_node_partition:
                        action_mask[node_idx, p] = True
            
            node_logits, partition_logits = self.actor(
                node_embeddings, region_embeddings, boundary_nodes, action_mask
            )
            
            value = self.critic(node_embeddings, region_embeddings, boundary_nodes)
            
            # Compute new log probability
            node_idx, partition_idx = action
            node_pos = (boundary_nodes == node_idx).nonzero(as_tuple=True)[0]
            
            if len(node_pos) == 0:
                continue
                
            node_pos = node_pos[0]
            
            node_probs = F.softmax(node_logits, dim=0)
            partition_probs = F.softmax(partition_logits[node_pos], dim=0)
            
            new_log_prob = (torch.log(node_probs[node_pos] + 1e-8) + 
                           torch.log(partition_probs[partition_idx - 1] + 1e-8))
            
            # PPO loss
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            actor_loss = -torch.min(surr1, surr2)
            
            # Critic loss
            critic_loss = F.mse_loss(value, return_val)
            
            # Entropy
            entropy = -(node_probs * torch.log(node_probs + 1e-8)).sum()
            entropy += -(partition_probs * torch.log(partition_probs + 1e-8)).sum()
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Update
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            
        return {
            'actor_loss': total_actor_loss / len(states),
            'critic_loss': total_critic_loss / len(states),
            'entropy': total_entropy / len(states)
        }
        
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
