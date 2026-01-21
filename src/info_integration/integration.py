"""Information integration module"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BallPhysics:
    """Badminton ball physics information"""
    position: Tuple[float, float, float]  # Position (x, y, z)
    velocity: float  # Velocity (m/s)
    direction: Tuple[float, float, float]  # Direction vector (dx, dy, dz)
    acceleration: float  # Acceleration (m/s²)


@dataclass
class HitInfo:
    """Hit event information"""
    frame_idx: int
    hitter: Optional[int]  # Hitter: 0 or 1, None if unknown
    ball_position: Tuple[float, float, float]
    ball_velocity: float
    ball_direction: Tuple[float, float, float]


class InfoIntegrator:
    """Information integration module"""
    
    def __init__(self, ball_speed_threshold: float = 5.0,
                 hit_detection_radius: float = 0.5,
                 fps: float = 30.0):
        """
        Initialize information integrator
        
        Args:
            ball_speed_threshold: Ball speed threshold (m/s), below which ball is considered stationary
            hit_detection_radius: Hit detection radius (m)
            fps: Video frame rate
        """
        self.ball_speed_threshold = ball_speed_threshold
        self.hit_detection_radius = hit_detection_radius
        self.fps = fps
        self.dt = 1.0 / fps  # Frame interval time
        
        # Historical trajectory
        self.ball_trajectory: List[Tuple[int, Tuple[float, float, float]]] = []
        self.ball_velocities: List[float] = []  # Store ball speed for each frame
        self.player1_positions: List[Tuple[int, np.ndarray]] = []
        self.player2_positions: List[Tuple[int, np.ndarray]] = []
    
    def calculate_ball_physics(self, current_pos: Tuple[float, float, float],
                              previous_pos: Optional[Tuple[float, float, float]] = None,
                              use_smoothing: bool = True) -> BallPhysics:
        """
        Calculate ball physics information (improved version with smoothing support)
        
        Args:
            current_pos: Current position
            previous_pos: Previous position
            use_smoothing: Whether to use multi-frame smoothing
            
        Returns:
            Ball physics information
        """
        if previous_pos is None:
            return BallPhysics(
                position=current_pos,
                velocity=0.0,
                direction=(0.0, 0.0, 0.0),
                acceleration=0.0
            )
        
        # Calculate displacement
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        dz = current_pos[2] - previous_pos[2]
        
        # Calculate velocity (m/s)
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        velocity = distance / self.dt
        
        # Use multi-frame smoothing to reduce noise (if enabled)
        if use_smoothing and len(self.ball_trajectory) >= 3:
            # Calculate average velocity using last 3 frames
            velocities = []
            for i in range(len(self.ball_trajectory) - 1, max(0, len(self.ball_trajectory) - 3), -1):
                if i > 0:
                    pos1 = self.ball_trajectory[i][1]
                    pos2 = self.ball_trajectory[i-1][1]
                    d = np.sqrt(
                        (pos1[0] - pos2[0])**2 +
                        (pos1[1] - pos2[1])**2 +
                        (pos1[2] - pos2[2])**2
                    )
                    v = d / self.dt
                    velocities.append(v)
            
            if velocities:
                # Use weighted average (more recent frames have higher weight)
                weights = np.array([0.5, 0.3, 0.2][:len(velocities)])
                weights = weights / weights.sum()
                velocity = np.average(velocities, weights=weights)
        
        # Calculate direction vector (normalized)
        if distance > 0:
            direction = (dx / distance, dy / distance, dz / distance)
        else:
            direction = (0.0, 0.0, 0.0)
        
        # Calculate acceleration (using velocity difference from previous two frames)
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
        Detect which player hit the ball
        
        Args:
            frame_idx: Frame index
            ball_pos: Ball position
            player1_keypoints: Player 1 keypoints
            player2_keypoints: Player 2 keypoints
            
        Returns:
            Hitter: 0 (player 1) or 1 (player 2), None if cannot determine
        """
        if player1_keypoints is None and player2_keypoints is None:
            return None
        
        min_distance = float('inf')
        hitter = None
        hand_keypoints = [9, 10]  # Left and right wrists
        
        # Check player 1
        if player1_keypoints is not None:
            hitter, min_distance = self._check_player_hit(
                player1_keypoints, ball_pos, hand_keypoints, 0, min_distance, hitter
            )
        
        # Check player 2
        if player2_keypoints is not None:
            hitter, min_distance = self._check_player_hit(
                player2_keypoints, ball_pos, hand_keypoints, 1, min_distance, hitter
            )
        
        return hitter
    
    def _check_player_hit(self, keypoints: np.ndarray, ball_pos: Tuple[float, float, float],
                         hand_keypoints: List[int], player_id: int,
                         min_distance: float, current_hitter: Optional[int]) -> Tuple[Optional[int], float]:
        """Check if a player hit the ball based on hand proximity"""
        for kpt_idx in hand_keypoints:
            if kpt_idx < len(keypoints) and keypoints[kpt_idx, 2] > 0.5:
                # Use hand position with z coordinate from ball position for better distance calculation
                # In practice, hand z might be different, but for hit detection we use ball z
                hand_pos = (keypoints[kpt_idx, 0], keypoints[kpt_idx, 1], ball_pos[2])
                distance = np.sqrt(
                    (ball_pos[0] - hand_pos[0])**2 +
                    (ball_pos[1] - hand_pos[1])**2 +
                    (ball_pos[2] - hand_pos[2])**2
                )
                if distance < min_distance and distance < self.hit_detection_radius:
                    min_distance = distance
                    current_hitter = player_id
        return current_hitter, min_distance
    
    def integrate(self, frame_idx: int,
                 ball_pos: Optional[Tuple[float, float, float]],
                 player1_keypoints: Optional[np.ndarray],
                 player2_keypoints: Optional[np.ndarray]) -> Optional[HitInfo]:
        """
        Integrate information
        
        Args:
            frame_idx: Frame index
            ball_pos: Ball position
            player1_keypoints: Player 1 keypoints
            player2_keypoints: Player 2 keypoints
            
        Returns:
            Hit information (if hit detected) or None
        """
        if ball_pos is None:
            return None
        
        # Update trajectory
        previous_pos = self.ball_trajectory[-1][1] if len(self.ball_trajectory) > 0 else None
        self.ball_trajectory.append((frame_idx, ball_pos))
        
        # Calculate ball physics
        ball_physics = self.calculate_ball_physics(ball_pos, previous_pos, use_smoothing=True)
        self.ball_velocities.append(ball_physics.velocity)
        
        # Detect hit (if velocity suddenly increases or direction changes)
        is_hit = self._detect_hit(ball_physics)
        
        if is_hit:
            hitter = self.detect_hitter(frame_idx, ball_pos, player1_keypoints, player2_keypoints)
            return HitInfo(
                frame_idx=frame_idx,
                hitter=hitter,
                ball_position=ball_pos,
                ball_velocity=ball_physics.velocity,
                ball_direction=ball_physics.direction
            )
        
        return None
    
    def _detect_hit(self, ball_physics: BallPhysics) -> bool:
        """Detect if a hit occurred based on velocity change"""
        if len(self.ball_trajectory) < 2:
            return False
        
        prev_physics = self.calculate_ball_physics(
            self.ball_trajectory[-2][1],
            self.ball_trajectory[-3][1] if len(self.ball_trajectory) >= 3 else None
        )
        
        # Sudden velocity increase indicates a hit
        return (ball_physics.velocity > self.ball_speed_threshold and
                ball_physics.velocity > prev_physics.velocity * 1.5)
    
    def get_current_ball_velocity(self) -> float:
        """
        Get current ball velocity
        
        Returns:
            Current ball velocity (m/s), 0 if no data
        """
        return self.ball_velocities[-1] if len(self.ball_velocities) > 0 else 0.0
    
    def get_average_ball_velocity(self, window_size: int = 5) -> float:
        """
        Get average ball velocity (for smooth display)
        
        Args:
            window_size: Window size (number of frames)
            
        Returns:
            Average ball velocity (m/s)
        """
        if len(self.ball_velocities) == 0:
            return 0.0
        
        recent_velocities = self.ball_velocities[-window_size:]
        return np.mean(recent_velocities) if recent_velocities else 0.0
    
    def reset(self):
        """Reset historical data"""
        self.ball_trajectory.clear()
        self.ball_velocities.clear()
        self.player1_positions.clear()
        self.player2_positions.clear()
