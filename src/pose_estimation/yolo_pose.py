"""YOLOv11-n-Pose for player pose estimation"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
from src.utils.registry import POSE_ESTIMATORS


@POSE_ESTIMATORS.register_module(name='yolo_pose')
class YOLOPoseEstimator:
    """YOLOv11-n-Pose pose estimator"""
    
    def __init__(self, model: str = "yolo11n-pose.pt",
                 conf_threshold: float = 0.5,
                 keypoint_threshold: float = 0.5):
        """
        Initialize pose estimator
        
        Args:
            model: Model name or path
            conf_threshold: Detection confidence threshold
            keypoint_threshold: Keypoint confidence threshold
        """
        self.model = YOLO(model)
        self.conf_threshold = conf_threshold
        self.keypoint_threshold = keypoint_threshold
    
    def estimate(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Estimate poses for all players in video frame
        
        Args:
            frame: Video frame
            
        Returns:
            List of keypoints, each element is a player's keypoint array (17, 3) - (x, y, confidence)
            COCO format: 0-nose, 1-left eye, 2-right eye, 3-left ear, 4-right ear, 5-left shoulder, 6-right shoulder,
                     7-left elbow, 8-right elbow, 9-left wrist, 10-right wrist, 11-left hip, 12-right hip,
                     13-left knee, 14-right knee, 15-left ankle, 16-right ankle
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        poses = []
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                for kpt in keypoints:
                    # Filter low confidence keypoints
                    filtered_kpt = kpt.copy()
                    filtered_kpt[filtered_kpt[:, 2] < self.keypoint_threshold] = [0, 0, 0]
                    poses.append(filtered_kpt)
        
        return poses
    
    def estimate_top2(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate poses for top 2 players (badminton typically has 2 players)
        
        Args:
            frame: Video frame
            
        Returns:
            (player1_keypoints, player2_keypoints)
        """
        poses = self.estimate(frame)
        
        # Sort by y coordinate (assume lower player is player1, upper is player2)
        if len(poses) >= 2:
            # Calculate average y coordinate for each player
            player_positions = []
            for i, pose in enumerate(poses):
                valid_kpts = pose[pose[:, 2] > self.keypoint_threshold]
                if len(valid_kpts) > 0:
                    avg_y = np.mean(valid_kpts[:, 1])
                    player_positions.append((i, avg_y))
            
            # Sort by y coordinate (larger y means lower position)
            player_positions.sort(key=lambda x: x[1], reverse=True)
            
            player1 = poses[player_positions[0][0]] if len(player_positions) > 0 else None
            player2 = poses[player_positions[1][0]] if len(player_positions) > 1 else None
            
            return player1, player2
        elif len(poses) == 1:
            return poses[0], None
        else:
            return None, None
