"""
Residual Correction Head
Language-Guided Trajectory Correction (LGTC) 태스크를 위한 모듈
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict
import logging

logger = logging.getLogger(__name__)


class ResidualCorrectionHead(nn.Module):
    """
    Residual Correction Head for Language-Guided Trajectory Correction
    
    실시간 언어 피드백("오른쪽으로", "위로")을 받아
    Base action에 대한 수정량(Delta)을 예측
    
    Args:
        lang_embed_dim: 언어 인코더 출력 dimension
        action_dim: Action space dimension
        hidden_dim: MLP hidden dimension
        num_layers: MLP depth
    """
    
    def __init__(
        self,
        lang_embed_dim: int = 768,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.lang_embed_dim = lang_embed_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Correction network
        # Input: [base_action + lang_feedback_embed]
        input_dim = action_dim + lang_embed_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output delta action
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.correction_net = nn.Sequential(*layers)
        
        logger.info(
            f"Residual Correction Head initialized: "
            f"lang_embed_dim={lang_embed_dim}, action_dim={action_dim}, "
            f"hidden_dim={hidden_dim}"
        )
    
    def forward(
        self,
        base_action: torch.Tensor,
        lang_feedback_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict delta action based on language feedback
        
        Args:
            base_action: (B, action_dim) - Base action from Flow Expert
            lang_feedback_embed: (B, lang_embed_dim) - Encoded language feedback
            
        Returns:
            delta_action: (B, action_dim) - Correction to apply
        """
        # Concatenate base action and language feedback
        inp = torch.cat([base_action, lang_feedback_embed], dim=-1)
        
        # Predict delta
        delta_action = self.correction_net(inp)
        
        return delta_action
    
    def apply_correction(
        self,
        base_action: torch.Tensor,
        lang_feedback_embed: torch.Tensor,
        correction_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply correction to base action
        
        Args:
            base_action: (B, action_dim)
            lang_feedback_embed: (B, lang_embed_dim)
            correction_scale: Scaling factor for delta (default: 1.0)
            
        Returns:
            corrected_action: (B, action_dim)
        """
        delta = self.forward(base_action, lang_feedback_embed)
        corrected_action = base_action + correction_scale * delta
        
        return corrected_action
    
    def compute_correction_loss(
        self,
        base_action: torch.Tensor,
        lang_feedback_embed: torch.Tensor,
        target_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for correction training
        
        Args:
            base_action: (B, action_dim)
            lang_feedback_embed: (B, lang_embed_dim)
            target_delta: (B, action_dim) - Ground truth delta
            
        Returns:
            loss: Scalar MSE loss
        """
        pred_delta = self.forward(base_action, lang_feedback_embed)
        loss = torch.nn.functional.mse_loss(pred_delta, target_delta)
        
        return loss
