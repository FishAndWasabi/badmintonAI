"""Rule-based shot classification"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ShotInfo:
    """球种信息"""
    shot_type: str  # 球种类别
    confidence: float  # 置信度


class RuleBasedClassifier:
    """基于规则的球种分类器"""
    
    # 球种定义
    SHOT_TYPES = {
        'serve': '发球',
        'clear': '高远球',
        'drop': '吊球',
        'smash': '扣杀',
        'drive': '平抽',
        'net': '网前球',
        'lob': '挑球',
        'unknown': '未知'
    }
    
    def __init__(self, speed_thresholds: Dict[str, float] = None,
                 angle_thresholds: Dict[str, float] = None):
        """
        初始化规则分类器
        
        Args:
            speed_thresholds: 速度阈值 {'high': 20.0, 'medium': 10.0}
            angle_thresholds: 角度阈值 {'steep': 45.0, 'flat': 15.0}
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
        计算球的飞行角度（与水平面的夹角）
        
        Args:
            direction: 方向向量 (dx, dy, dz)
            
        Returns:
            角度（度）
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
        分类球种
        
        Args:
            ball_velocity: 球速（米/秒）
            ball_direction: 方向向量 (dx, dy, dz)
            ball_position: 球的位置 (x, y, z)
            hitter: 击球方（0或1）
            player_keypoints: 击球方的关键点
            
        Returns:
            球种信息
        """
        # 计算角度
        angle = self.calculate_angle(ball_direction)
        
        # 计算水平方向
        dx, dy, dz = ball_direction
        horizontal_speed = np.sqrt(dx**2 + dy**2)
        
        # 判断球的位置（前场/后场）
        # 假设球场中心为原点，y轴正方向为前场
        is_front_court = ball_position[1] > 0
        is_back_court = ball_position[1] < -3.0  # 后场区域
        
        # 规则判断
        shot_type = 'unknown'
        confidence = 0.5
        
        # 1. 扣杀（Smash）
        if (ball_velocity > self.speed_thresholds['high'] and
            angle < -self.angle_thresholds['steep']):
            shot_type = 'smash'
            confidence = 0.9
        
        # 2. 高远球（Clear）
        elif (ball_velocity > self.speed_thresholds['medium'] and
              angle > self.angle_thresholds['steep'] and
              is_back_court):
            shot_type = 'clear'
            confidence = 0.85
        
        # 3. 吊球（Drop）
        elif (ball_velocity < self.speed_thresholds['medium'] and
              angle < -self.angle_thresholds['flat'] and
              is_front_court):
            shot_type = 'drop'
            confidence = 0.8
        
        # 4. 平抽（Drive）
        elif (ball_velocity > self.speed_thresholds['medium'] and
              abs(angle) < self.angle_thresholds['flat']):
            shot_type = 'drive'
            confidence = 0.75
        
        # 5. 网前球（Net）
        elif (ball_velocity < self.speed_thresholds['medium'] and
              is_front_court and
              abs(angle) < self.angle_thresholds['flat']):
            shot_type = 'net'
            confidence = 0.7
        
        # 6. 挑球（Lob）
        elif (ball_velocity < self.speed_thresholds['medium'] and
              angle > self.angle_thresholds['steep'] and
              is_front_court):
            shot_type = 'lob'
            confidence = 0.75
        
        # 7. 发球（Serve）- 通常在第一帧或速度突然增加时
        elif ball_velocity > self.speed_thresholds['high'] * 0.8:
            # 需要结合上下文判断，这里简化处理
            shot_type = 'serve'
            confidence = 0.6
        
        return ShotInfo(shot_type=shot_type, confidence=confidence)
    
    def get_shot_type_name(self, shot_type: str, lang: str = 'zh') -> str:
        """
        获取球种名称
        
        Args:
            shot_type: 球种代码
            lang: 语言 ('zh' 或 'en')
            
        Returns:
            球种名称
        """
        if lang == 'zh':
            return self.SHOT_TYPES.get(shot_type, '未知')
        else:
            return shot_type
