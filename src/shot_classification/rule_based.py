"""Rule-based shot classification"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from src.utils.registry import SHOT_CLASSIFIERS


@dataclass
class ShotInfo:
    """Shot type information"""
    shot_type: str  # Shot type category
    confidence: float  # Confidence score


@SHOT_CLASSIFIERS.register_module(name='rule_based')
class RuleBasedClassifier:
    """Rule-based shot type classifier"""
    
    # Shot type definitions
    SHOT_TYPES = {
        'serve': 'serve',
        'clear': 'clear',
        'drop': 'drop',
        'smash': 'smash',
        'drive': 'drive',
        'net': 'net',
        'lob': 'lob',
        'unknown': 'unknown'
    }
    
    def __init__(self, speed_thresholds: Dict[str, float] = None,
                 angle_thresholds: Dict[str, float] = None):
        """
        Initialize rule-based classifier
        
        Args:
            speed_thresholds: Speed thresholds {'high': 20.0, 'medium': 10.0}
            angle_thresholds: Angle thresholds {'steep': 45.0, 'flat': 15.0}
        """
        self.speed_thresholds = speed_thresholds or {
            'high': 20.0,
            'medium': 10.0
        }
        self.angle_thresholds = angle_thresholds or {
            'steep': 45.0,
            'flat': 15.0
        }
    
    def calculate_angle(self, direction: Tuple[float, float, float]) -> float:
        """
        Calculate ball flight angle (angle with horizontal plane)
        
        Args:
            direction: Direction vector (dx, dy, dz)
            
        Returns:
            Angle (degrees)
        """
        dx, dy, dz = direction
        horizontal_speed = np.sqrt(dx**2 + dy**2)
        if horizontal_speed == 0:
            return 90.0 if dz > 0 else -90.0
        
        angle = np.arctan2(dz, horizontal_speed) * 180 / np.pi
        return angle
    
    def classify(self, ball_velocity: float,
                ball_direction: Tuple[float, float, float],
                ball_position: Tuple[float, float, float],
                hitter: Optional[int] = None,
                player_keypoints: Optional[np.ndarray] = None) -> ShotInfo:
        """
        Classify shot type
        
        Args:
            ball_velocity: Ball velocity (m/s)
            ball_direction: Direction vector (dx, dy, dz)
            ball_position: Ball position (x, y, z)
            hitter: Hitter (0 or 1)
            player_keypoints: Hitter's keypoints
            
        Returns:
            Shot type information
        """
        # Calculate angle
        angle = self.calculate_angle(ball_direction)
        
        # Calculate horizontal direction
        dx, dy, dz = ball_direction
        horizontal_speed = np.sqrt(dx**2 + dy**2)
        
        # Determine ball position (front court/back court)
        # Assume court center as origin, positive y-axis as front court
        is_front_court = ball_position[1] > 0
        is_back_court = ball_position[1] < -3.0  # Back court area
        
        # Rule-based classification
        shot_type = 'unknown'
        confidence = 0.5
        
        # 1. Smash
        if (ball_velocity > self.speed_thresholds['high'] and
            angle < -self.angle_thresholds['steep']):
            shot_type = 'smash'
            confidence = 0.9
        
        # 2. Clear
        elif (ball_velocity > self.speed_thresholds['medium'] and
              angle > self.angle_thresholds['steep'] and
              is_back_court):
            shot_type = 'clear'
            confidence = 0.85
        
        # 3. Drop
        elif (ball_velocity < self.speed_thresholds['medium'] and
              angle < -self.angle_thresholds['flat'] and
              is_front_court):
            shot_type = 'drop'
            confidence = 0.8
        
        # 4. Drive
        elif (ball_velocity > self.speed_thresholds['medium'] and
              abs(angle) < self.angle_thresholds['flat']):
            shot_type = 'drive'
            confidence = 0.75
        
        # 5. Net
        elif (ball_velocity < self.speed_thresholds['medium'] and
              is_front_court and
              abs(angle) < self.angle_thresholds['flat']):
            shot_type = 'net'
            confidence = 0.7
        
        # 6. Lob
        elif (ball_velocity < self.speed_thresholds['medium'] and
              angle > self.angle_thresholds['steep'] and
              is_front_court):
            shot_type = 'lob'
            confidence = 0.75
        
        # 7. Serve - usually at first frame or when velocity suddenly increases
        elif ball_velocity > self.speed_thresholds['high'] * 0.8:
            # Need context to determine, simplified here
            shot_type = 'serve'
            confidence = 0.6
        
        return ShotInfo(shot_type=shot_type, confidence=confidence)
    
    def get_shot_type_name(self, shot_type: str, lang: str = 'en') -> str:
        """
        Get shot type name
        
        Args:
            shot_type: Shot type code
            lang: Language ('zh' or 'en')
            
        Returns:
            Shot type name
        """
        if lang == 'zh':
            return self.SHOT_TYPES.get(shot_type, '未知')
        else:
            return shot_type
