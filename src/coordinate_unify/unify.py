"""Coordinate unification module"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import cv2


class CoordinateUnifier:
    """Coordinate unification module"""
    
    def __init__(self, court_width: float = 6.1, 
                 court_length: float = 13.4,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize coordinate unifier
        
        Args:
            court_width: Court width (meters)
            court_length: Court length (meters)
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
        """
        self.court_width = court_width
        self.court_length = court_length
        
        # Use default values if camera parameters not provided
        if camera_matrix is None:
            # Default camera intrinsics (needs calibration based on actual camera)
            self.camera_matrix = np.array([
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
        
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1))
        else:
            self.dist_coeffs = dist_coeffs
        
        # Court 3D coordinates (world coordinate system)
        # Assume court center as origin, x-axis along width, y-axis along length, z-axis upward
        self.court_3d_points = np.array([
            [0, 0, 0],  # Center
            [court_width/2, 0, 0],  # Right boundary center
            [-court_width/2, 0, 0],  # Left boundary center
            [0, court_length/2, 0],  # Front boundary center
            [0, -court_length/2, 0],  # Back boundary center
        ], dtype=np.float32)
    
    def unify_ball_coordinate(self, pixel_coord: Tuple[float, float],
                             frame_size: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to world coordinates (simplified version)
        
        Args:
            pixel_coord: Pixel coordinates (x, y)
            frame_size: Frame size (width, height)
            
        Returns:
            World coordinates (x, y, z) or (x, y, 0) if depth cannot be calculated
        """
        x_pixel, y_pixel = pixel_coord
        width, height = frame_size
        
        # Simplified method: assume ball is on ground (z=0), use inverse perspective transform
        # Here uses simple linear mapping (should use camera calibration and perspective transform in practice)
        
        # Normalize to [-1, 1]
        x_norm = (x_pixel - width/2) / (width/2)
        y_norm = (y_pixel - height/2) / (height/2)
        
        # Simple linear mapping (needs adjustment based on actual scene)
        # Assume image center corresponds to court center
        x_world = x_norm * self.court_width / 2
        y_world = y_norm * self.court_length / 2
        
        return (x_world, y_world, 0.0)
    
    def unify_pose_coordinates(self, keypoints: np.ndarray,
                              frame_size: Tuple[int, int]) -> np.ndarray:
        """
        Unify pose keypoint coordinates
        
        Args:
            keypoints: Keypoint array (N, 3) - (x, y, confidence)
            frame_size: Frame size (width, height)
            
        Returns:
            Unified keypoint array (N, 3) - (x_world, y_world, confidence)
        """
        unified_keypoints = keypoints.copy()
        
        for i in range(len(keypoints)):
            if keypoints[i, 2] > 0:  # If keypoint is valid
                x_world, y_world, _ = self.unify_ball_coordinate(
                    (keypoints[i, 0], keypoints[i, 1]), frame_size
                )
                unified_keypoints[i, 0] = x_world
                unified_keypoints[i, 1] = y_world
                # confidence remains unchanged
        
        return unified_keypoints
    
    def pixel_to_world(self, pixel_coords: List[Tuple[float, float]],
                      frame_size: Tuple[int, int]) -> List[Tuple[float, float, float]]:
        """
        Batch convert pixel coordinates to world coordinates
        
        Args:
            pixel_coords: List of pixel coordinates
            frame_size: Frame size
            
        Returns:
            List of world coordinates
        """
        world_coords = []
        for coord in pixel_coords:
            world_coord = self.unify_ball_coordinate(coord, frame_size)
            world_coords.append(world_coord)
        return world_coords
