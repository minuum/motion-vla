"""
Vision-based metrics for wiping task evaluation.

Implements dirt pixel counting and cleaning rate calculation
using color-based segmentation.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional


class VisionMetrics:
    """Calculate vision-based metrics for wiping task."""
    
    def __init__(
        self,
        dirt_color_lower: Tuple[int, int, int] = (10, 50, 20),  # HSV
        dirt_color_upper: Tuple[int, int, int] = (30, 255, 100),
        min_dirt_area: int = 5,  # Minimum pixels for valid dirt
    ):
        """
        Initialize vision metrics calculator.
        
        Args:
            dirt_color_lower: Lower HSV bound for dirt color (brown)
            dirt_color_upper: Upper HSV bound for dirt color
            min_dirt_area: Minimum contour area to count as dirt
        """
        self.dirt_color_lower = np.array(dirt_color_lower)
        self.dirt_color_upper = np.array(dirt_color_upper)
        self.min_dirt_area = min_dirt_area
        
        self.initial_dirt_pixels = None
        
    def count_dirt_pixels(
        self,
        image: np.ndarray,
        debug: bool = False,
    ) -> int:
        """
        Count number of dirt pixels in image.
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
            debug: If True, return debug visualization
            
        Returns:
            Number of dirt pixels detected
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask for dirt color
        mask = cv2.inRange(hsv, self.dirt_color_lower, self.dirt_color_upper)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Count pixels
        dirt_pixels = np.sum(mask > 0)
        
        if debug:
            # Create debug visualization
            debug_img = image.copy()
            debug_img[mask > 0] = [255, 0, 0]  # Highlight dirt in red
            return dirt_pixels, debug_img
        
        return dirt_pixels
    
    def calculate_cleaning_rate(
        self,
        current_image: np.ndarray,
        initial_image: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate cleaning rate as percentage of dirt removed.
        
        Args:
            current_image: Current RGB image
            initial_image: Initial image before cleaning (optional)
            
        Returns:
            Cleaning rate (0-1)
        """
        current_dirt = self.count_dirt_pixels(current_image)
        
        # Store initial dirt count on first call
        if initial_image is not None:
            self.initial_dirt_pixels = self.count_dirt_pixels(initial_image)
        
        if self.initial_dirt_pixels is None or self.initial_dirt_pixels == 0:
            return 0.0
        
        cleaning_rate = 1.0 - (current_dirt / self.initial_dirt_pixels)
        return max(0.0, min(1.0, cleaning_rate))  # Clamp to [0, 1]
    
    def calculate_coverage(
        self,
        wiper_trajectory: np.ndarray,
        table_size: Tuple[float, float] = (0.8, 0.6),
        cell_size: float = 0.02,  # 2cm grid cells
    ) -> float:
        """
        Calculate percentage of table surface covered by wiper trajectory.
        
        Args:
            wiper_trajectory: (N, 3) array of wiper positions
            table_size: (width, depth) of table
            cell_size: Size of grid cells for coverage calculation
            
        Returns:
            Coverage percentage (0-1)
        """
        w, d = table_size
        grid_w = int(w / cell_size)
        grid_d = int(d / cell_size)
        
        # Create occupancy grid
        grid = np.zeros((grid_w, grid_d), dtype=bool)
        
        # Mark visited cells
        for pos in wiper_trajectory:
            x, y = pos[0], pos[1]
            # Convert to grid coordinates
            gx = int((x + w/2) / cell_size)
            gy = int((y + d/2) / cell_size)
            
            if 0 <= gx < grid_w and 0 <= gy < grid_d:
                grid[gx, gy] = True
        
        coverage = np.sum(grid) / (grid_w * grid_d)
        return coverage
    
    def calculate_metrics(
        self,
        current_image: np.ndarray,
        wiper_trajectory: np.ndarray,
        initial_image: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate all metrics for wiping task.
        
        Returns:
            Dictionary with metrics:
            - cleaning_rate: Percentage of dirt removed
            - coverage: Percentage of table covered
            - dirt_pixels: Current dirt pixel count
        """
        cleaning_rate = self.calculate_cleaning_rate(current_image, initial_image)
        coverage = self.calculate_coverage(wiper_trajectory)
        dirt_pixels = self.count_dirt_pixels(current_image)
        
        return {
            "cleaning_rate": cleaning_rate,
            "coverage": coverage,
            "dirt_pixels": dirt_pixels,
            "initial_dirt_pixels": self.initial_dirt_pixels or 0,
        }
    
    def reset(self):
        """Reset metrics for new episode."""
        self.initial_dirt_pixels = None


if __name__ == "__main__":
    # Test with synthetic image
    metrics = VisionMetrics()
    
    # Create test image with brown "dirt"
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Gray background
    
    # Add some brown dirt patches
    test_img[100:150, 200:250] = [139, 90, 43]  # Brown patch 1
    test_img[300:320, 400:450] = [139, 90, 43]  # Brown patch 2
    
    dirt_count = metrics.count_dirt_pixels(test_img)
    print(f"✅ Dirt pixel detection test:")
    print(f"  Detected {dirt_count} dirt pixels")
    
    # Test cleaning rate
    cleaned_img = test_img.copy()
    cleaned_img[100:150, 200:250] = [200, 200, 200]  # Clean patch 1
    
    rate = metrics.calculate_cleaning_rate(cleaned_img, test_img)
    print(f"  Cleaning rate: {rate:.1%}")
