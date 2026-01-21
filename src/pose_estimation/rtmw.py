"""RTMW model for player pose estimation using mmpose with RTMDet detector"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
import sys
import os
from pathlib import Path

# Add mmpose path
mmpose_path = Path(__file__).parent.parent.parent / "models" / "pose_estimation" / "mmpose"
sys.path.insert(0, str(mmpose_path))

from mmpose.apis import init_model, inference_topdown
from mmpose.evaluation.functional import nms
from mmengine.config import Config
from src.utils.registry import POSE_ESTIMATORS

# Try to import mmdet for human detection
try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.utils import adapt_mmdet_pipeline
    HAS_MMDET = True
except (ImportError, ModuleNotFoundError):
    HAS_MMDET = False


@POSE_ESTIMATORS.register_module(name='rtmw')
class RTMWPoseEstimator:
    """RTMW pose estimator (using mmpose)"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None,
                 det_config: Optional[str] = None,
                 det_checkpoint: Optional[str] = None,
                 det_score_thr: float = 0.3,
                 det_nms_thr: float = 0.3):
        """
        Initialize RTMW pose estimator with RTMDet detector for multi-person detection
        
        Args:
            config_path: mmpose config file path (cocktail14 config)
            checkpoint_path: Model weight file path
            device: Device ('cuda:0', 'cpu', etc.), auto-select if None
            det_config: RTMDet detector config path (for multi-person detection)
            det_checkpoint: RTMDet detector checkpoint path
            det_score_thr: Detection score threshold (default: 0.3)
            det_nms_thr: NMS IOU threshold for detection (default: 0.3)
        """
        # Set device
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Default config path (cocktail14 RTMW-L 256x192)
        if config_path is None:
            config_path = str(mmpose_path / "configs" / "wholebody_2d_keypoint" / 
                            "rtmpose" / "cocktail14" / 
                            "rtmw-l_8xb1024-270e_cocktail14-256x192.py")
        
        # Check if config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Default checkpoint path (if not specified)
        if checkpoint_path is None:
            checkpoint_path = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth"
        
        print(f"Loading RTMW model...")
        print(f"  Config file: {config_path}")
        print(f"  Checkpoint file: {checkpoint_path}")
        print(f"  Device: {self.device}")
        
        # Load pose model
        self.model = init_model(
            config=config_path,
            checkpoint=checkpoint_path,
            device=self.device
        )
        print("✓ RTMW model loaded successfully")
        
        # Get input size (read from config)
        self.input_size = tuple(self.model.cfg.model.data_preprocessor.get('input_size', [256, 192]))
        if len(self.input_size) == 2:
            self.input_size = (self.input_size[1], self.input_size[0])  # (height, width)
        else:
            self.input_size = (192, 256)  # Default value
        
        print(f"  Input size: {self.input_size}")
        
        # Initialize detector for multi-person detection
        self.detector = None
        self.det_score_thr = det_score_thr
        self.det_nms_thr = det_nms_thr
        self.det_cat_id = 0  # Person class ID in COCO
        
        if HAS_MMDET:
            # Default detector config (RTMDet-M for person detection)
            if det_config is None:
                det_config = str(mmpose_path / "demo" / "mmdetection_cfg" / 
                               "rtmdet_m_640-8xb32_coco-person.py")
            
            # Default detector checkpoint
            if det_checkpoint is None:
                det_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
            
            # Check if detector config exists
            if os.path.exists(det_config):
                print(f"Loading RTMDet detector for multi-person detection...")
                print(f"  Detector config: {det_config}")
                print(f"  Detector checkpoint: {det_checkpoint}")
                
                self.detector = init_detector(
                    det_config,
                    det_checkpoint,
                    device=self.device
                )
                self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
                print("✓ RTMDet detector loaded successfully")
            else:
                print(f"Warning: Detector config not found: {det_config}")
                print("  Multi-person detection will be limited")
        else:
            print("Warning: mmdet not available, multi-person detection disabled")
    
    def _extract_body_keypoints(self, data_samples: List, num_body_kpts: int = 17) -> List[np.ndarray]:
        """
        Extract body keypoints from mmpose output (COCO format 17 keypoints)
        
        Args:
            data_samples: mmpose inference results
            num_body_kpts: Number of body keypoints (17 for COCO)
            
        Returns:
            List of keypoints, each element is (num_body_kpts, 3) - (x, y, confidence)
        """
        keypoints_list = []
        
        for data_sample in data_samples:
            if hasattr(data_sample, 'pred_instances'):
                instances = data_sample.pred_instances
                
                if hasattr(instances, 'keypoints') and len(instances.keypoints) > 0:
                    # Handle multiple instances
                    num_instances = len(instances.keypoints)
                    for i in range(num_instances):
                        # RTMW outputs 133 keypoints (wholebody), need to extract first 17 body keypoints
                        if instances.keypoints.ndim == 3:
                            wholebody_kpts = instances.keypoints[i]  # (133, 2)
                            wholebody_scores = instances.keypoint_scores[i]  # (133,)
                        else:
                            # Single instance case
                            wholebody_kpts = instances.keypoints[0]  # (133, 2)
                            wholebody_scores = instances.keypoint_scores[0]  # (133,)
                        
                        # Extract first 17 body keypoints (COCO format)
                        body_kpts = wholebody_kpts[:num_body_kpts]  # (17, 2)
                        body_scores = wholebody_scores[:num_body_kpts]  # (17,)
                        
                        # Combine as (17, 3) format: (x, y, confidence)
                        keypoints = np.concatenate([
                            body_kpts,
                            body_scores[:, None]
                        ], axis=1)
                        
                        keypoints_list.append(keypoints)
        
        return keypoints_list
    
    def estimate(self, frame: np.ndarray, bboxes: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Estimate poses for all people in video frame
        
        Args:
            frame: Video frame (BGR format)
            bboxes: Optional human detection boxes, shape (N, 4) or (N, 5), format xyxy or xyxy+score
                   If None, use RTMDet detector to detect all people in the image
            
        Returns:
            List of keypoints, each element is (17, 3) - (x, y, confidence), COCO format
        """
        if self.model is None:
            return []
        
        # Detect people if bboxes not provided and detector is available
        if bboxes is None and self.detector is not None:
            # Use RTMDet to detect all people
            det_result = inference_detector(self.detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            
            # Filter person detections (class 0) by score threshold
            person_mask = np.logical_and(
                pred_instance.labels == self.det_cat_id,
                pred_instance.scores > self.det_score_thr
            )
            
            if np.any(person_mask):
                bboxes = pred_instance.bboxes[person_mask]  # (N, 4) in xyxy format
                scores = pred_instance.scores[person_mask]
                
                # Apply NMS to remove overlapping detections
                if len(bboxes) > 1:
                    # Combine bboxes and scores for NMS
                    bboxes_with_scores = np.concatenate([bboxes, scores[:, None]], axis=1)
                    keep_indices = nms(bboxes_with_scores, self.det_nms_thr)
                    bboxes = bboxes[keep_indices]
            else:
                # No people detected
                return []
        
        # If still no bboxes (detector not available or no detections), use entire image
        if bboxes is None or len(bboxes) == 0:
            h, w = frame.shape[:2]
            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
        
        # Ensure bboxes format is correct
        if len(bboxes.shape) == 1:
            bboxes = bboxes[None, :]  # Add batch dimension
        
        # Only take first 4 values (xyxy format)
        if bboxes.shape[1] > 4:
            bboxes = bboxes[:, :4]
        
        # Inference
        results = inference_topdown(
            model=self.model,
            img=frame,
            bboxes=bboxes,
            bbox_format='xyxy'
        )
        
        # Extract keypoints
        keypoints_list = self._extract_body_keypoints(results, num_body_kpts=17)
        
        return keypoints_list
    
    def estimate_top2(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate poses for top 2 players (badminton typically has 2 players)
        
        Args:
            frame: Video frame
            
        Returns:
            (player1_keypoints, player2_keypoints), each is (17, 3) or None
        """
        # Estimate first (using entire image)
        poses = self.estimate(frame)
        
        if len(poses) >= 2:
            # Sort by y coordinate (assume lower player is player1, upper is player2)
            # Calculate average y coordinate for each player
            player_positions = []
            for i, pose in enumerate(poses):
                valid_kpts = pose[pose[:, 2] > 0.5]  # Keypoints with confidence > 0.5
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
