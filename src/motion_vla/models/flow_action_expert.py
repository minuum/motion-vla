"""
Flow-matching Action Expert
Conditional Flow Matching을 사용하여 continuous action을 생성
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError(
        "torchdiffeq is required for Flow-matching. "
        "Install with: pip install torchdiffeq"
    )

logger = logging.getLogger(__name__)


class FlowActionExpert(nn.Module):
    """
    Flow-matching 기반 Action Expert
    
    Conditional Flow Matching (CFM)을 사용하여 VL embedding으로부터
    smooth continuous action sequence를 생성
    
    Args:
        embed_dim: VL Encoder 출력 dimension
        action_dim: Action space dimension (예: 7 for 6-DoF + gripper)
        chunk_size: 예측할 action sequence 길이
        hidden_dim: Velocity field network의 hidden dimension
        num_layers: Velocity field network depth
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        action_dim: int = 7,
        chunk_size: int = 10,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        
        # Velocity field network v_θ(a_t, t, c)
        # Input: [action_dim + 1 (time) + embed_dim (conditioning)]
        input_dim = action_dim + 1 + embed_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.velocity_net = nn.Sequential(*layers)
        
        logger.info(
            f"Flow Action Expert initialized: "
            f"embed_dim={embed_dim}, action_dim={action_dim}, "
            f"chunk_size={chunk_size}, hidden_dim={hidden_dim}"
        )
    
    def velocity_field(
        self,
        t: torch.Tensor,
        a: torch.Tensor,
        vl_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Velocity field v_θ(a_t, t, c)
        
        Args:
            t: Time (scalar tensor)
            a: Action at time t, shape (B, action_dim)
            vl_embed: VL conditioning, shape (B, embed_dim)
            
        Returns:
            velocity: da/dt, shape (B, action_dim)
        """
        batch_size = a.size(0)
        
        # Expand time to match batch
        t_expand = t * torch.ones(batch_size, 1).to(a.device)
        
        # Concatenate: [a_t, t, c]
        inp = torch.cat([a, t_expand, vl_embed], dim=-1)
        
        # Compute velocity
        velocity = self.velocity_net(inp)
        
        return velocity
    
    def forward(
        self,
        vl_embed: torch.Tensor,
        num_integration_steps: int = 10,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate action via flow-matching
        
        Args:
            vl_embed: (B, embed_dim) VL conditioning
            num_integration_steps: ODE solver steps
            return_trajectory: Return full trajectory or just final action
            
        Returns:
            actions: (B, action_dim) if not return_trajectory
                    or (num_integration_steps, B, action_dim) if return_trajectory
        """
        batch_size = vl_embed.size(0)
        device = vl_embed.device
        
        # Initial noise a_0 ~ N(0, I)
        a_0 = torch.randn(batch_size, self.action_dim).to(device)
        
        # Time span from t=0 to t=1
        t_span = torch.linspace(0, 1, num_integration_steps).to(device)
        
        # Define ODE function with conditioning
        def ode_func(t, a):
            return self.velocity_field(t, a, vl_embed)
        
        # Integrate flow: a_1 = a_0 + ∫_0^1 v_θ(a_t, t, c) dt
        trajectory = odeint(ode_func, a_0, t_span, method='dopri5')
        
        if return_trajectory:
            return trajectory  # (T, B, action_dim)
        else:
            return trajectory[-1]  # (B, action_dim)
    
    def compute_flow_matching_loss(
        self,
        vl_embed: torch.Tensor,
        target_actions: torch.Tensor,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Compute Flow Matching Loss for training
        
        Loss: E_t,a0,a1 [ ||v_θ(a_t, t, c) - (a1 - a0)||^2 ]
        
        Args:
            vl_embed: (B, embed_dim) VL conditioning
            target_actions: (B, action_dim) ground truth actions
            num_samples: Number of time samples for loss computation
            
        Returns:
            loss: Scalar loss value
        """
        batch_size = vl_embed.size(0)
        device = vl_embed.device
        
        # Sample random time points t ~ Uniform(0, 1)
        t_samples = torch.rand(num_samples, 1).to(device)
        
        total_loss = 0.0
        
        for t in t_samples:
            # Sample a_0 ~ N(0, I)
            a_0 = torch.randn(batch_size, self.action_dim).to(device)
            
            # a_1 = target_actions
            a_1 = target_actions
            
            # Linear interpolation: a_t = (1-t) * a_0 + t * a_1
            a_t = (1 - t) * a_0 + t * a_1
            
            # True velocity: d/dt [(1-t)a_0 + t*a_1] = a_1 - a_0
            true_velocity = a_1 - a_0
            
            # Predicted velocity
            pred_velocity = self.velocity_field(t, a_t, vl_embed)
            
            # MSE loss
            loss = ((pred_velocity - true_velocity) ** 2).mean()
            total_loss += loss
        
        # Average over time samples
        total_loss /= num_samples
        
        return total_loss
    
    def generate_action_chunk(
        self,
        vl_embed: torch.Tensor,
        num_integration_steps: int = 10,
    ) -> torch.Tensor:
        """
        Generate a chunk of actions (for multi-step prediction)
        
        현재 구현은 single action을 생성하지만,
        향후 chunk_size 만큼의 sequence를 생성하도록 확장 가능
        
        Args:
            vl_embed: (B, embed_dim)
            num_integration_steps: ODE steps
            
        Returns:
            action_chunk: (B, chunk_size, action_dim)
        """
        # 현재는 single action만 생성
        single_action = self.forward(vl_embed, num_integration_steps)
        
        # Reshape to chunk format (repeat for now)
        # 향후: sequence model로 확장
        action_chunk = single_action.unsqueeze(1).repeat(1, self.chunk_size, 1)
        
        return action_chunk
