"""
Base environment class for Isaac Sim tasks.

This module provides the foundational environment structure for all
Isaac Sim-based tasks in the Motion VLA project.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import omni
from omni.isaac.kit import SimulationApp


class BaseIsaacEnv(ABC):
    """Abstract base class for Isaac Sim environments."""
    
    def __init__(
        self,
        headless: bool = False,
        render_fps: int = 60,
        physics_dt: float = 1.0/50.0,  # 50Hz for flow-matching
    ):
        """
        Initialize Isaac Sim application.
        
        Args:
            headless: Run without GUI
            render_fps: Rendering framerate
            physics_dt: Physics simulation timestep (20ms for 50Hz)
        """
        self.simulation_app = SimulationApp({
            "headless": headless,
            "anti_aliasing": 3 if not headless else 0,
        })
        
        self.physics_dt = physics_dt
        self.render_fps = render_fps
        
        # Will be initialized in subclasses
        self.world = None
        self.robot = None
        self.cameras = {}
        
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action and return (obs, reward, done, info).
        
        Args:
            action: Robot action (e.g., 7-DoF for Dobot E6)
            
        Returns:
            obs: Current observation
            reward: Reward signal
            done: Episode termination flag
            info: Additional information
        """
        pass
    
    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from environment."""
        pass
    
    @abstractmethod
    def compute_reward(self) -> float:
        """Compute reward based on current state."""
        pass
    
    def close(self):
        """Clean up and close simulation."""
        if self.simulation_app:
            self.simulation_app.close()
