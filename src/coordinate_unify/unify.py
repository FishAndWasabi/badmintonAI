"""Coordinate unification module"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import cv2


class CoordinateUnifier:
    """坐标统一模块"""
    
    def __init__(self, court_width: float = 6.1, 
                 court_length: float = 13.4,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        初始化坐标统一器
        
        Args:
            court_width: 球场宽度（米）
            court_length: 球场长度（米）
            camera_matrix: 相机内参矩阵 (3x3)
            dist_coeffs: 畸变系数
        """
        self.court_width = court_width
        self.court_length = court_length
        
        # 如果没有提供相机参数，使用默认值
        if camera_matrix is None:
            # 默认相机内参（需要根据实际相机进行标定）
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
        
        # 球场3D坐标（世界坐标系）
        # 假设球场中心为原点，x轴沿宽度方向，y轴沿长度方向，z轴向上
        self.court_3d_points = np.array([
            [0, 0, 0],  # 中心
            [court_width/2, 0, 0],  # 右边界中心
            [-court_width/2, 0, 0],  # 左边界中心
            [0, court_length/2, 0],  # 前边界中心
            [0, -court_length/2, 0],  # 后边界中心
        ], dtype=np.float32)
    
    def unify_ball_coordinate(self, pixel_coord: Tuple[float, float],
                             frame_size: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        将像素坐标转换为世界坐标（简化版本）
        
        Args:
            pixel_coord: 像素坐标 (x, y)
            frame_size: 帧尺寸 (width, height)
            
        Returns:
            世界坐标 (x, y, z) 或 (x, y, 0) 如果无法计算深度
        """
        x_pixel, y_pixel = pixel_coord
        width, height = frame_size
        
        # 简化方法：假设球在地面上（z=0），使用逆透视变换
        # 这里使用简单的线性映射（实际应该使用相机标定和透视变换）
        
        # 归一化到[-1, 1]
        x_norm = (x_pixel - width/2) / (width/2)
        y_norm = (y_pixel - height/2) / (height/2)
        
        # 简单的线性映射（需要根据实际场景调整）
        # 假设图像中心对应球场中心
        x_world = x_norm * self.court_width / 2
        y_world = y_norm * self.court_length / 2
        
        return (x_world, y_world, 0.0)
    
    def unify_pose_coordinates(self, keypoints: np.ndarray,
                              frame_size: Tuple[int, int]) -> np.ndarray:
        """
        统一姿态关键点坐标
        
        Args:
            keypoints: 关键点数组 (N, 3) - (x, y, confidence)
            frame_size: 帧尺寸 (width, height)
            
        Returns:
            统一后的关键点数组 (N, 3) - (x_world, y_world, confidence)
        """
        unified_keypoints = keypoints.copy()
        
        for i in range(len(keypoints)):
            if keypoints[i, 2] > 0:  # 如果关键点有效
                x_world, y_world, _ = self.unify_ball_coordinate(
                    (keypoints[i, 0], keypoints[i, 1]), frame_size
                )
                unified_keypoints[i, 0] = x_world
                unified_keypoints[i, 1] = y_world
                # confidence保持不变
        
        return unified_keypoints
    
    def pixel_to_world(self, pixel_coords: List[Tuple[float, float]],
                      frame_size: Tuple[int, int]) -> List[Tuple[float, float, float]]:
        """
        批量转换像素坐标到世界坐标
        
        Args:
            pixel_coords: 像素坐标列表
            frame_size: 帧尺寸
            
        Returns:
            世界坐标列表
        """
        world_coords = []
        for coord in pixel_coords:
            world_coord = self.unify_ball_coordinate(coord, frame_size)
            world_coords.append(world_coord)
        return world_coords
