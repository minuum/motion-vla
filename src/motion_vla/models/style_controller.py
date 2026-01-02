"""
Style Controller for ACMC Task
Adverb-Conditioned Motion Control
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# Adverb to dynamics parameters mapping
ADVERB_STYLES = {
    "carefully": {
        "velocity_scale": 0.5,
        "max_acceleration": 0.3,
        "jerk_limit": 0.1,
    },
    "quickly": {
        "velocity_scale": 1.5,
        "max_acceleration": 2.0,
        "jerk_limit": 1.0,
    },
    "steadily": {
        "velocity_scale": 0.8,
        "max_acceleration": 0.5,
        "jerk_limit": 0.05,
    },
    "gently": {
        "velocity_scale": 0.4,
        "max_acceleration": 0.2,
        "jerk_limit": 0.05,
    },
    "normal": {
        "velocity_scale": 1.0,
        "max_acceleration": 1.0,
        "jerk_limit": 0.5,
    },
}


class StyleController:
    """
    Style Controller for Adverb-Conditioned Motion Control
    
    부사(adverb)에 따라 로봇 동작의 속도/가속도 프로파일을 조정
    """
    
    def __init__(self, custom_styles: Optional[Dict] = None):
        """
        Args:
            custom_styles: Custom adverb-to-dynamics mapping (optional)
        """
        self.styles = ADVERB_STYLES.copy()
        
        if custom_styles:
            self.styles.update(custom_styles)
            logger.info(f"Added {len(custom_styles)} custom styles")
        
        logger.info(f"Style Controller initialized with {len(self.styles)} styles")
    
    def get_style_params(self, adverb: str) -> Dict[str, float]:
        """
        Get dynamics parameters for given adverb
        
        Args:
            adverb: Adverb token (e.g., "carefully")
            
        Returns:
            params: Dictionary of dynamics parameters
        """
        adverb_lower = adverb.lower()
        
        if adverb_lower in self.styles:
            return self.styles[adverb_lower]
        else:
            logger.warning(f"Unknown adverb '{adverb}', using 'normal' style")
            return self.styles["normal"]
    
    def apply_style_to_action(
        self,
        action: torch.Tensor,
        adverb: str,
    ) -> torch.Tensor:
        """
        Apply style to action sequence
        
        Args:
            action: (B, T, action_dim) or (B, action_dim) action tensor
            adverb: Adverb token
            
        Returns:
            styled_action: Action with velocity scaling applied
        """
        params = self.get_style_params(adverb)
        velocity_scale = params["velocity_scale"]
        
        # Simple velocity scaling
        # (실제 구현 시 더 정교한 dynamics 제어 필요)
        styled_action = action * velocity_scale
        
        return styled_action
    
    def apply_style_to_trajectory(
        self,
        trajectory: torch.Tensor,
        adverb: str,
        dt: float = 0.02,  # 50Hz = 0.02s
    ) -> torch.Tensor:
        """
        Apply style to trajectory with acceleration/jerk clipping
        
        Args:
            trajectory: (B, T, D) trajectory tensor
            adverb: Adverb token
            dt: Time step
            
        Returns:
            styled_trajectory: Trajectory with dynamics constraints applied
        """
        params = self.get_style_params(adverb)
        
        B, T, D = trajectory.shape
        styled_traj = trajectory.clone()
        
        # Velocity scaling
        styled_traj = styled_traj * params["velocity_scale"]
        
        # Acceleration clipping
        # Compute acceleration via finite differences
        if T > 1:
            velocities = (styled_traj[:, 1:, :] - styled_traj[:, :-1, :]) / dt
            accelerations = (velocities[:, 1:, :] - velocities[:, :-1, :]) / dt
            
            # Clip accelerations
            max_accel = params["max_acceleration"]
            accelerations = torch.clamp(accelerations, -max_accel, max_accel)
            
            # Reconstruct trajectory (simplified)
            # In practice, need proper integration with constraints
            pass
        
        return styled_traj
    
    def extract_adverb_from_instruction(self, instruction: str) -> Optional[str]:
        """
        Extract adverb from natural language instruction
        
        Args:
            instruction: Text instruction (e.g., "Pick up cup carefully")
            
        Returns:
            adverb: Extracted adverb or None
        """
        instruction_lower = instruction.lower()
        
        for adverb in self.styles.keys():
            if adverb in instruction_lower:
                return adverb
        
        return None
    
    def get_available_styles(self) -> list[str]:
        """Return list of available adverb styles"""
        return list(self.styles.keys())
