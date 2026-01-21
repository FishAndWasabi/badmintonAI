"""YOLO detection + lightweight single object tracker"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from src.utils.registry import BALL_TRACKERS


@BALL_TRACKERS.register_module(name='yolo_tracker')
class YOLOBallTracker:
    """YOLO detection + Kalman filter tracker"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        Initialize tracker
        
        Args:
            model_path: YOLO model path, if None use pretrained model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Use YOLOv8n as default model (needs fine-tuning for badminton)
            self.model = YOLO('yolo11n.pt')
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize Kalman filter
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
        Detect badminton ball
        
        Args:
            frame: Video frame
            
        Returns:
            (x, y, confidence) or None
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Find badminton ball detection (class 0 is usually ball, adjust based on actual model)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                # Select detection with highest confidence
                best_box = boxes[0]
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                conf = best_box.conf[0].cpu().numpy()
                
                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                return (center_x, center_y, float(conf))
        
        return None
    
    def track(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Track badminton ball
        
        Args:
            frame: Video frame
            
        Returns:
            (x, y, confidence) or None
        """
        # Detect first
        detection = self.detect(frame)
        
        if detection is not None:
            x, y, conf = detection
            
            if self.tracked:
                # Update Kalman filter
                self.kf.update(np.array([x, y]))
            else:
                # Initialize tracking
                self.kf.x = np.array([x, y, 0., 0.])
                self.tracked = True
                self.missed_frames = 0
            
            # Predict next frame position
            self.kf.predict()
            
            # Use Kalman filtered position
            state = self.kf.x
            return (float(state[0]), float(state[1]), conf)
        else:
            if self.tracked:
                self.missed_frames += 1
                if self.missed_frames <= self.max_missed_frames:
                    # Use predicted position
                    self.kf.predict()
                    state = self.kf.x
                    return (float(state[0]), float(state[1]), 0.5)  # Lower confidence
                else:
                    # Lost tracking
                    self.tracked = False
                    self.missed_frames = 0
            
            return None
    
    def reset(self):
        """Reset tracker"""
        self.tracked = False
        self.missed_frames = 0
        self.kf.x = np.array([0., 0., 0., 0.])
