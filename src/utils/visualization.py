"""Visualization utilities"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def draw_ball(frame: np.ndarray, center: Tuple[float, float], 
              confidence: float, radius: int = 5) -> np.ndarray:
    """
    在帧上绘制羽毛球
    
    Args:
        frame: 视频帧
        center: 球心坐标 (x, y)
        confidence: 置信度
        radius: 绘制半径
        
    Returns:
        绘制后的帧
    """
    frame = frame.copy()
    x, y = int(center[0]), int(center[1])
    
    # 根据置信度设置颜色
    color = (0, int(255 * confidence), int(255 * (1 - confidence)))
    cv2.circle(frame, (x, y), radius, color, -1)
    cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 2)
    
    return frame


def draw_pose(frame: np.ndarray, keypoints: np.ndarray, 
              confidence_threshold: float = 0.5) -> np.ndarray:
    """
    在帧上绘制姿态关键点
    
    Args:
        frame: 视频帧
        keypoints: 关键点数组 (N, 3) - (x, y, confidence)
        confidence_threshold: 置信度阈值
        
    Returns:
        绘制后的帧
    """
    frame = frame.copy()
    
    # COCO姿态关键点连接关系（简化版）
    skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 手臂
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],  # 腿部
        [5, 11], [6, 12]  # 躯干
    ]
    
    # 绘制关键点
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # 绘制骨架
    for start_idx, end_idx in skeleton:
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            keypoints[start_idx][2] > confidence_threshold and
            keypoints[end_idx][2] > confidence_threshold):
            pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    return frame


def draw_trajectory(frame: np.ndarray, trajectory: List[Tuple[float, float]],
                   color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    在帧上绘制轨迹
    
    Args:
        frame: 视频帧
        trajectory: 轨迹点列表 [(x, y), ...]
        color: 轨迹颜色
        
    Returns:
        绘制后的帧
    """
    frame = frame.copy()
    
    if len(trajectory) < 2:
        return frame
    
    points = np.array([(int(x), int(y)) for x, y in trajectory], dtype=np.int32)
    cv2.polylines(frame, [points], False, color, 2)
    
    # 绘制起点和终点
    if len(trajectory) > 0:
        cv2.circle(frame, points[0], 5, (0, 255, 0), -1)  # 起点绿色
    if len(trajectory) > 1:
        cv2.circle(frame, points[-1], 5, (0, 0, 255), -1)  # 终点红色
    
    return frame


def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int],
             font_scale: float = 0.6, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    在帧上绘制文本
    
    Args:
        frame: 视频帧
        text: 文本内容
        position: 文本位置 (x, y)
        font_scale: 字体大小
        color: 文本颜色
        
    Returns:
        绘制后的帧
    """
    frame = frame.copy()
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, 2)
    return frame
