"""YOLOv11-n-Pose for player pose estimation"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO


class YOLOPoseEstimator:
    """YOLOv11-n-Pose姿态估计器"""
    
    def __init__(self, model: str = "yolo11n-pose.pt",
                 conf_threshold: float = 0.5,
                 keypoint_threshold: float = 0.5):
        """
        初始化姿态估计器
        
        Args:
            model: 模型名称或路径
            conf_threshold: 检测置信度阈值
            keypoint_threshold: 关键点置信度阈值
        """
        self.model = YOLO(model)
        self.conf_threshold = conf_threshold
        self.keypoint_threshold = keypoint_threshold
    
    def estimate(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        估计视频帧中所有玩家的姿态
        
        Args:
            frame: 视频帧
            
        Returns:
            关键点列表，每个元素是一个玩家的关键点数组 (17, 3) - (x, y, confidence)
            COCO格式：0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳, 5-左肩, 6-右肩,
                     7-左肘, 8-右肘, 9-左手腕, 10-右手腕, 11-左髋, 12-右髋,
                     13-左膝, 14-右膝, 15-左脚踝, 16-右脚踝
        """
        results = self.model(frame, conf=self.conf_threshold)
        
        poses = []
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                for kpt in keypoints:
                    # 过滤低置信度的关键点
                    filtered_kpt = kpt.copy()
                    filtered_kpt[filtered_kpt[:, 2] < self.keypoint_threshold] = [0, 0, 0]
                    poses.append(filtered_kpt)
        
        return poses
    
    def estimate_top2(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        估计前两个玩家的姿态（通常羽毛球比赛有2名玩家）
        
        Args:
            frame: 视频帧
            
        Returns:
            (player1_keypoints, player2_keypoints)
        """
        poses = self.estimate(frame)
        
        # 根据检测框的y坐标排序（假设下方的玩家是player1，上方的是player2）
        if len(poses) >= 2:
            # 计算每个玩家的平均y坐标
            player_positions = []
            for i, pose in enumerate(poses):
                valid_kpts = pose[pose[:, 2] > self.keypoint_threshold]
                if len(valid_kpts) > 0:
                    avg_y = np.mean(valid_kpts[:, 1])
                    player_positions.append((i, avg_y))
            
            # 按y坐标排序（y越大越靠下）
            player_positions.sort(key=lambda x: x[1], reverse=True)
            
            player1 = poses[player_positions[0][0]] if len(player_positions) > 0 else None
            player2 = poses[player_positions[1][0]] if len(player_positions) > 1 else None
            
            return player1, player2
        elif len(poses) == 1:
            return poses[0], None
        else:
            return None, None
