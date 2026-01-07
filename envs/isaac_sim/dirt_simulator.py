"""
Particle-based dirt simulation for wiping task.

Uses PhysX particle system to simulate dirt on table surface.
Tracks "cleaned" particles via collision detection with wiper.
"""

import numpy as np
from typing import List, Tuple
import omni
from pxr import Gf, UsdGeom, UsdPhysics


class DirtSimulator:
    """Manages dirt particles on table surface."""
    
    def __init__(
        self,
        table_size: Tuple[float, float] = (0.8, 0.6),
        table_height: float = 0.4,
        particle_count: int = 100,
        particle_radius: float = 0.005,  # 5mm particles
    ):
        """
        Initialize dirt simulator.
        
        Args:
            table_size: (width, depth) of table in meters
            table_height: Height of table surface
            particle_count: Number of dirt particles
            particle_radius: Radius of each particle
        """
        self.table_size = table_size
        self.table_height = table_height
        self.particle_count = particle_count
        self.particle_radius = particle_radius
        
        self.particles = []
        self.cleaned_particles = set()
        
    def spawn_particles(
        self,
        stage,
        pattern: str = "random",
        density: float = 0.5,
    ) -> List:
        """
        Spawn dirt particles on table surface.
        
        Args:
            stage: USD stage to add particles to
            pattern: "random", "grid", "cluster"
            density: Particle density (0-1, for random pattern)
            
        Returns:
            List of particle prims
        """
        self.particles = []
        self.cleaned_particles = set()
        
        positions = self._generate_positions(pattern, density)
        
        for i, pos in enumerate(positions):
            # Create sphere particle
            particle_path = f"/World/Dirt/Particle_{i}"
            particle = UsdGeom.Sphere.Define(stage, particle_path)
            
            # Set size and position
            particle.GetRadiusAttr().Set(self.particle_radius)
            particle.AddTranslateOp().Set(Gf.Vec3f(*pos))
            
            # Add physics (rigid body for collision detection)
            UsdPhysics.RigidBodyAPI.Apply(particle.GetPrim())
            UsdPhysics.CollisionAPI.Apply(particle.GetPrim())
            
            # Visual properties (brown dirt color)
            particle.CreateDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.25, 0.1)])
            
            self.particles.append(particle)
            
        print(f"✅ Spawned {len(self.particles)} dirt particles ({pattern} pattern)")
        return self.particles
    
    def _generate_positions(
        self,
        pattern: str,
        density: float,
    ) -> np.ndarray:
        """Generate particle positions based on pattern."""
        w, d = self.table_size
        h = self.table_height + self.particle_radius + 0.001  # Just above table
        
        if pattern == "random":
            # Random distribution within table bounds
            count = int(self.particle_count * density)
            x = np.random.uniform(-w/2, w/2, count)
            y = np.random.uniform(-d/2, d/2, count)
            z = np.full(count, h)
            
        elif pattern == "grid":
            # Uniform grid
            grid_size = int(np.sqrt(self.particle_count))
            x = np.linspace(-w/2 + 0.05, w/2 - 0.05, grid_size)
            y = np.linspace(-d/2 + 0.05, d/2 - 0.05, grid_size)
            xx, yy = np.meshgrid(x, y)
            x = xx.flatten()
            y = yy.flatten()
            z = np.full(len(x), h)
            
        elif pattern == "cluster":
            # Clustered "spills"
            num_clusters = 5
            positions = []
            for _ in range(num_clusters):
                center = np.random.uniform([-w/3, -d/3], [w/3, d/3])
                cluster_size = self.particle_count // num_clusters
                cluster = np.random.normal(
                    loc=center,
                    scale=0.05,  # 5cm cluster radius
                    size=(cluster_size, 2)
                )
                positions.append(cluster)
            xy = np.vstack(positions)
            x = xy[:, 0]
            y = xy[:, 1]
            z = np.full(len(x), h)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Shift to table center (assume table at x=0.4, y=0)
        x += 0.4
        
        return np.column_stack([x, y, z])
    
    def check_collision(
        self,
        wiper_position: np.ndarray,
        wiper_size: Tuple[float, float] = (0.05, 0.03),
    ) -> int:
        """
        Check if wiper collided with any particles.
        
        Args:
            wiper_position: (x, y, z) position of wiper
            wiper_size: (width, depth) of wiper
            
        Returns:
            Number of newly cleaned particles
        """
        newly_cleaned = 0
        wx, wy, wz = wiper_position
        w_width, w_depth = wiper_size
        
        for i, particle in enumerate(self.particles):
            if i in self.cleaned_particles:
                continue
                
            # Get particle position
            pos = particle.GetAttribute("xformOp:translate").Get()
            px, py, pz = pos[0], pos[1], pos[2]
            
            # Simple AABB collision check
            if (abs(px - wx) < w_width/2 and
                abs(py - wy) < w_depth/2 and
                abs(pz - wz) < 0.02):  # 2cm vertical tolerance
                
                self.cleaned_particles.add(i)
                # Hide cleaned particle (or remove from scene)
                particle.GetPrim().SetActive(False)
                newly_cleaned += 1
        
        return newly_cleaned
    
    def get_cleaning_rate(self) -> float:
        """
        Calculate current cleaning percentage.
        
        Returns:
            Cleaning rate (0-1)
        """
        if len(self.particles) == 0:
            return 1.0
        return len(self.cleaned_particles) / len(self.particles)
    
    def reset(self):
        """Reset all particles to active state."""
        self.cleaned_particles = set()
        for particle in self.particles:
            particle.GetPrim().SetActive(True)
        print("🔄 Dirt particles reset")


if __name__ == "__main__":
    # Test position generation
    simulator = DirtSimulator()
    
    for pattern in ["random", "grid", "cluster"]:
        positions = simulator._generate_positions(pattern, density=0.7)
        print(f"\n{pattern.upper()} pattern:")
        print(f"  Generated {len(positions)} positions")
        print(f"  X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
        print(f"  Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
