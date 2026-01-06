"""Information integration module"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BallPhysics:
    """羽毛球物理信息"""
    position: Tuple[float, float, float]  # 位置 (x, y, z)
    velocity: float  # 速度（米/秒）
    direction: Tuple[float, float, float]  # 方向向量 (dx, dy, dz)
    acceleration: float  # 加速度（米/秒²）


@dataclass
class HitInfo:
    """击球信息"""
    frame_idx: int
    hitter: Optional[int]  # 击球方：0或1，None表示未知
    ball_position: Tuple[float, float, float]
    ball_velocity: float
    ball_direction: Tuple[float, float, float]


class InfoIntegrator:
    """信息整合模块"""
    
    def __init__(self, ball_speed_threshold: float = 5.0,
                 hit_detection_radius: float = 0.5,
                 fps: float = 30.0):
        """
        初始化信息整合器
        
        Args:
            ball_speed_threshold: 球速阈值（米/秒），低于此值认为球静止
            hit_detection_radius: 击球检测半径（米）
            fps: 视频帧率
        """
        self.ball_speed_threshold = ball_speed_threshold
        self.hit_detection_radius = hit_detection_radius
        self.fps = fps
        self.dt = 1.0 / fps  # 帧间隔时间
        
        # 历史轨迹
        self.ball_trajectory: List[Tuple[int, Tuple[float, float, float]]] = []
        self.player1_positions: List[Tuple[int, np.ndarray]] = []
        self.player2_positions: List[Tuple[int, np.ndarray]] = []
    
    def calculate_ball_physics(self, current_pos: Tuple[float, float, float],
                              previous_pos: Optional[Tuple[float, float, float]] = None) -> BallPhysics:
        """
        计算球的物理信息
        
        Args:
            current_pos: 当前位置
            previous_pos: 前一位置
            
        Returns:
            球的物理信息
        """
        if previous_pos is None:
            return BallPhysics(
                position=current_pos,
                velocity=0.0,
                direction=(0.0, 0.0, 0.0),
                acceleration=0.0
            )
        
        # 计算位移
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        dz = current_pos[2] - previous_pos[2]
        
        # 计算速度（米/秒）
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        velocity = distance / self.dt
        
        # 计算方向向量（归一化）
        if distance > 0:
            direction = (dx / distance, dy / distance, dz / distance)
        else:
            direction = (0.0, 0.0, 0.0)
        
        # 计算加速度（简化：使用前两帧的速度差）
        acceleration = 0.0
        if len(self.ball_trajectory) >= 2:
            prev_prev_pos = self.ball_trajectory[-2][1]
            prev_distance = np.sqrt(
                (previous_pos[0] - prev_prev_pos[0])**2 +
                (previous_pos[1] - prev_prev_pos[1])**2 +
                (previous_pos[2] - prev_prev_pos[2])**2
            )
            prev_velocity = prev_distance / self.dt
            acceleration = (velocity - prev_velocity) / self.dt
        
        return BallPhysics(
            position=current_pos,
            velocity=velocity,
            direction=direction,
            acceleration=acceleration
        )
    
    def detect_hitter(self, frame_idx: int,
                     ball_pos: Tuple[float, float, float],
                     player1_keypoints: Optional[np.ndarray],
                     player2_keypoints: Optional[np.ndarray]) -> Optional[int]:
        """
        检测击球方
        
        Args:
            frame_idx: 帧索引
            ball_pos: 球的位置
            player1_keypoints: 玩家1的关键点
            player2_keypoints: 玩家2的关键点
            
        Returns:
            击球方：0（玩家1）或1（玩家2），None表示无法确定
        """
        if player1_keypoints is None and player2_keypoints is None:
            return None
        
        min_distance = float('inf')
        hitter = None
        
        # 检查玩家1
        if player1_keypoints is not None:
            # 使用手腕位置（关键点9和10）或手部区域
            hand_keypoints = [9, 10]  # 左右手腕
            for kpt_idx in hand_keypoints:
                if kpt_idx < len(player1_keypoints) and player1_keypoints[kpt_idx, 2] > 0.5:
                    hand_pos = (player1_keypoints[kpt_idx, 0], 
                               player1_keypoints[kpt_idx, 1],
                               0.0)  # 假设z=0
                    distance = np.sqrt(
                        (ball_pos[0] - hand_pos[0])**2 +
                        (ball_pos[1] - hand_pos[1])**2 +
                        (ball_pos[2] - hand_pos[2])**2
                    )
                    if distance < min_distance and distance < self.hit_detection_radius:
                        min_distance = distance
                        hitter = 0
        
        # 检查玩家2
        if player2_keypoints is not None:
            hand_keypoints = [9, 10]
            for kpt_idx in hand_keypoints:
                if kpt_idx < len(player2_keypoints) and player2_keypoints[kpt_idx, 2] > 0.5:
                    hand_pos = (player2_keypoints[kpt_idx, 0],
                               player2_keypoints[kpt_idx, 1],
                               0.0)
                    distance = np.sqrt(
                        (ball_pos[0] - hand_pos[0])**2 +
                        (ball_pos[1] - hand_pos[1])**2 +
                        (ball_pos[2] - hand_pos[2])**2
                    )
                    if distance < min_distance and distance < self.hit_detection_radius:
                        min_distance = distance
                        hitter = 1
        
        return hitter
    
    def integrate(self, frame_idx: int,
                 ball_pos: Optional[Tuple[float, float, float]],
                 player1_keypoints: Optional[np.ndarray],
                 player2_keypoints: Optional[np.ndarray]) -> Optional[HitInfo]:
        """
        整合信息
        
        Args:
            frame_idx: 帧索引
            ball_pos: 球的位置
            player1_keypoints: 玩家1的关键点
            player2_keypoints: 玩家2的关键点
            
        Returns:
            击球信息（如果检测到击球）或None
        """
        if ball_pos is None:
            return None
        
        # 更新轨迹
        previous_pos = None
        if len(self.ball_trajectory) > 0:
            previous_pos = self.ball_trajectory[-1][1]
        
        self.ball_trajectory.append((frame_idx, ball_pos))
        
        # 计算球的物理信息
        ball_physics = self.calculate_ball_physics(ball_pos, previous_pos)
        
        # 检测击球（如果速度突然增加或方向改变）
        is_hit = False
        if len(self.ball_trajectory) >= 2:
            prev_physics = self.calculate_ball_physics(
                self.ball_trajectory[-2][1],
                self.ball_trajectory[-3][1] if len(self.ball_trajectory) >= 3 else None
            )
            
            # 速度突然增加或方向改变
            if (ball_physics.velocity > self.ball_speed_threshold and
                ball_physics.velocity > prev_physics.velocity * 1.5):
                is_hit = True
        
        if is_hit:
            # 检测击球方
            hitter = self.detect_hitter(frame_idx, ball_pos, player1_keypoints, player2_keypoints)
            
            return HitInfo(
                frame_idx=frame_idx,
                hitter=hitter,
                ball_position=ball_pos,
                ball_velocity=ball_physics.velocity,
                ball_direction=ball_physics.direction
            )
        
        return None
    
    def reset(self):
        """重置历史数据"""
        self.ball_trajectory.clear()
        self.player1_positions.clear()
        self.player2_positions.clear()
