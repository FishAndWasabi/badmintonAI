"""Visualization utilities"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def draw_ball(frame: np.ndarray, center: Tuple[float, float], 
              confidence: float = 0.5, radius: int = 5) -> np.ndarray:
    """
    Draw badminton ball on frame
    
    Args:
        frame: Video frame
        center: Ball center coordinates (x, y)
        confidence: Confidence score
        radius: Drawing radius
        
    Returns:
        Frame with ball drawn
    """
    frame = frame.copy()
    x, y = int(center[0]), int(center[1])
    
    # Set color based on confidence (green to red gradient)
    color = (0, int(255 * confidence), int(255 * (1 - confidence)))
    cv2.circle(frame, (x, y), radius, color, -1)
    cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 2)
    
    return frame


def draw_pose(frame: np.ndarray, keypoints: np.ndarray, 
              confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Draw pose keypoints on frame
    
    Args:
        frame: Video frame
        keypoints: Keypoint array (N, 3) - (x, y, confidence)
        confidence_threshold: Confidence threshold
        
    Returns:
        Frame with pose drawn
    """
    frame = frame.copy()
    
    # COCO pose keypoint connections (simplified)
    skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4],  # Head
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],  # Legs
        [5, 11], [6, 12]  # Torso
    ]
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # Draw skeleton
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
    Draw trajectory on frame
    
    Args:
        frame: Video frame
        trajectory: List of trajectory points [(x, y), ...]
        color: Trajectory color
        
    Returns:
        Frame with trajectory drawn
    """
    frame = frame.copy()
    
    if len(trajectory) < 2:
        return frame
    
    points = np.array([(int(x), int(y)) for x, y in trajectory], dtype=np.int32)
    cv2.polylines(frame, [points], False, color, 2)
    
    # Draw start and end points
    if len(trajectory) > 0:
        cv2.circle(frame, points[0], 5, (0, 255, 0), -1)  # Start point green
    if len(trajectory) > 1:
        cv2.circle(frame, points[-1], 5, (0, 0, 255), -1)  # End point red
    
    return frame


def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int],
             font_scale: float = 0.6, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Draw text on frame
    
    Args:
        frame: Video frame
        text: Text content
        position: Text position (x, y)
        font_scale: Font scale
        color: Text color
        
    Returns:
        Frame with text drawn
    """
    frame = frame.copy()
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, 2)
    return frame
