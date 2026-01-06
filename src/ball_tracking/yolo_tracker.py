"""YOLO detection + lightweight single object tracker"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter


class YOLOBallTracker:
    """YOLO检测 + 卡尔曼滤波追踪器"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        初始化追踪器
        
        Args:
            model_path: YOLO模型路径，如果为None则使用预训练模型
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # 使用YOLOv8n作为默认模型（需要针对羽毛球进行微调）
            self.model = YOLO('yolo11n.pt')
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0., 0., 0., 0.])  # [x, y, vx, vy]
        self.kf.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        self.kf.P *= 1000.
        self.kf.R = np.eye(2) * 10
        self.kf.Q = np.eye(4) * 0.1
        
        self.tracked = False
        self.missed_frames = 0
        self.max_missed_frames = 5
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        检测羽毛球
        
        Args:
            frame: 视频帧
            
        Returns:
            (x, y, confidence) 或 None
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # 查找羽毛球检测结果（类别0通常是球类，需要根据实际模型调整）
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                # 选择置信度最高的检测结果
                best_box = boxes[0]
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                conf = best_box.conf[0].cpu().numpy()
                
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                return (center_x, center_y, float(conf))
        
        return None
    
    def track(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        追踪羽毛球
        
        Args:
            frame: 视频帧
            
        Returns:
            (x, y, confidence) 或 None
        """
        # 先进行检测
        detection = self.detect(frame)
        
        if detection is not None:
            x, y, conf = detection
            
            if self.tracked:
                # 更新卡尔曼滤波器
                self.kf.update(np.array([x, y]))
            else:
                # 初始化追踪
                self.kf.x = np.array([x, y, 0., 0.])
                self.tracked = True
                self.missed_frames = 0
            
            # 预测下一帧位置
            self.kf.predict()
            
            # 使用卡尔曼滤波后的位置
            state = self.kf.x
            return (float(state[0]), float(state[1]), conf)
        else:
            if self.tracked:
                self.missed_frames += 1
                if self.missed_frames <= self.max_missed_frames:
                    # 使用预测位置
                    self.kf.predict()
                    state = self.kf.x
                    return (float(state[0]), float(state[1]), 0.5)  # 降低置信度
                else:
                    # 丢失追踪
                    self.tracked = False
                    self.missed_frames = 0
            
            return None
    
    def reset(self):
        """重置追踪器"""
        self.tracked = False
        self.missed_frames = 0
        self.kf.x = np.array([0., 0., 0., 0.])
