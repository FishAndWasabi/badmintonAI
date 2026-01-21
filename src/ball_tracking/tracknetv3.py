"""TrackNetv3 model for ball tracking"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
import os
from pathlib import Path

# Add TrackNetv3 path
tracknetv3_path = Path(__file__).parent.parent.parent / "models" / "ball_tracking" / "TrackNetv3"
sys.path.insert(0, str(tracknetv3_path))

from model import TrackNet
from utils.general import get_model
from src.utils.registry import BALL_TRACKERS


@BALL_TRACKERS.register_module(name='tracknetv3')
class TrackNetv3Tracker:
    """TrackNetv3 badminton ball tracker"""
    
    def __init__(self, model_path: Optional[str] = None,
                 input_size: Tuple[int, int] = (288, 512),
                 confidence_threshold: float = 0.5):
        """
        Initialize TrackNetv3 tracker
        
        Args:
            model_path: Model path
            input_size: Input image size (height, width)
            confidence_threshold: Confidence threshold
        """
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            
            # Get parameters from checkpoint
            if 'param_dict' in ckpt:
                seq_len = ckpt['param_dict']['seq_len']
                bg_mode = ckpt['param_dict'].get('bg_mode', '')
            else:
                # Use default values if param_dict is missing
                seq_len = 8  # Default sequence length
                bg_mode = 'concat'  # Default background mode
                print("Warning: param_dict missing in checkpoint, using default parameters")
            
            # Create model
            self.model = get_model('TrackNet', seq_len, bg_mode)
            
            # Load weights
            if 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
            else:
                # If checkpoint is directly a state_dict
                self.model.load_state_dict(ckpt)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.seq_len = seq_len
            self.bg_mode = bg_mode
            
            # Initialize frame buffer (for sequence input)
            self.frame_buffer = []
            self.background_frame = None  # For background mode
            
            print(f"✓ TrackNetv3 model loaded successfully")
            print(f"  Sequence length: {seq_len}, Background mode: {bg_mode}")
        else:
            if model_path:
                raise FileNotFoundError(f"TrackNetv3 model file not found: {model_path}")
            else:
                raise ValueError("TrackNetv3 model path not specified")
    
    def preprocess_sequence(self, frames: list) -> torch.Tensor:
        """
        Preprocess sequence of frames
        
        Args:
            frames: List of frames
            
        Returns:
            Preprocessed tensor (1, C, H, W), where C depends on bg_mode
        """
        processed_frames = []
        
        for frame in frames:
            # Resize
            resized = cv2.resize(frame, (self.input_size[1], self.input_size[0]))
            
            # Convert to RGB
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                rgb = resized
            
            # Normalize
            normalized = rgb.astype(np.float32) / 255.0
            processed_frames.append(normalized)
        
        # Process according to bg_mode
        if self.bg_mode == 'subtract':
            tensor = self._preprocess_subtract(processed_frames)
        elif self.bg_mode == 'subtract_concat':
            tensor = self._preprocess_subtract_concat(processed_frames)
        elif self.bg_mode == 'concat':
            tensor = self._preprocess_concat(processed_frames)
        else:
            # Default: RGB sequence only
            tensor = self._preprocess_rgb_only(processed_frames)
        
        return tensor.to(self.device)
    
    def _preprocess_subtract(self, processed_frames: list) -> torch.Tensor:
        """Preprocess with background subtraction"""
        if self.background_frame is None:
            self.background_frame = processed_frames[0]
        diff_frames = [np.abs(f - self.background_frame) for f in processed_frames]
        sequence = np.stack(diff_frames, axis=0)  # (seq_len, H, W)
        tensor = torch.from_numpy(sequence).unsqueeze(0)  # (1, seq_len, H, W)
        return tensor
    
    def _preprocess_subtract_concat(self, processed_frames: list) -> torch.Tensor:
        """Preprocess with RGB + difference frames"""
        if self.background_frame is None:
            self.background_frame = processed_frames[0]
        diff_frames = [np.abs(f - self.background_frame) for f in processed_frames]
        rgb_stack = np.stack(processed_frames, axis=0).transpose(0, 3, 1, 2)  # (seq_len, 3, H, W)
        diff_stack = np.stack(diff_frames, axis=0).transpose(0, 3, 1, 2)  # (seq_len, 3, H, W)
        sequence = np.concatenate([rgb_stack, diff_stack], axis=1)  # (seq_len, 6, H, W)
        sequence = sequence.reshape(-1, sequence.shape[2], sequence.shape[3])  # (seq_len*6, H, W)
        tensor = torch.from_numpy(sequence).unsqueeze(0)  # (1, seq_len*6, H, W)
        return tensor
    
    def _preprocess_concat(self, processed_frames: list) -> torch.Tensor:
        """Preprocess with background frame concatenation"""
        if self.background_frame is None:
            self.background_frame = processed_frames[0]
        frames_with_bg = [self.background_frame] + processed_frames
        sequence = np.stack(frames_with_bg, axis=0).transpose(0, 3, 1, 2)  # (seq_len+1, 3, H, W)
        sequence = sequence.reshape(-1, sequence.shape[2], sequence.shape[3])  # ((seq_len+1)*3, H, W)
        tensor = torch.from_numpy(sequence).unsqueeze(0)  # (1, (seq_len+1)*3, H, W)
        return tensor
    
    def _preprocess_rgb_only(self, processed_frames: list) -> torch.Tensor:
        """Preprocess RGB sequence only"""
        sequence = np.stack(processed_frames, axis=0).transpose(0, 3, 1, 2)  # (seq_len, 3, H, W)
        sequence = sequence.reshape(-1, sequence.shape[2], sequence.shape[3])  # (seq_len*3, H, W)
        tensor = torch.from_numpy(sequence).unsqueeze(0)  # (1, seq_len*3, H, W)
        return tensor
    
    def postprocess(self, output: torch.Tensor, original_size: Tuple[int, int], frame_idx: int = -1) -> Optional[Tuple[float, float, float]]:
        """
        Postprocess model output
        
        Args:
            output: Model output, shape (1, seq_len, H, W) or (1, seq_len, 3, H, W)
            original_size: Original image size (height, width)
            frame_idx: Frame index to extract (-1 means last frame)
            
        Returns:
            (x, y, confidence) or None
        """
        if output is None:
            return None
        
        # Convert output to numpy
        if isinstance(output, torch.Tensor):
            output = output.cpu().detach().numpy()
        
        # TrackNet output shape: (batch, seq_len, H, W) or (batch, seq_len, 3, H, W)
        if len(output.shape) == 4:
            # (batch, seq_len, H, W)
            if frame_idx < 0:
                frame_idx = output.shape[1] + frame_idx  # Convert to positive index
            heatmap = output[0, frame_idx]  # Extract heatmap for specified frame
        elif len(output.shape) == 5:
            # (batch, seq_len, 3, H, W) - multi-channel output, take first channel
            if frame_idx < 0:
                frame_idx = output.shape[1] + frame_idx
            heatmap = output[0, frame_idx, 0]  # Take first channel
        else:
            # Other formats, try to take first
            heatmap = output[0] if len(output.shape) >= 2 else output
        
        # Find maximum position
        max_val = float(np.max(heatmap))
        if max_val < self.confidence_threshold:
            return None
        
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y_pred, x_pred = max_pos
        
        # Scale to original image size
        scale_x = original_size[1] / self.input_size[1]
        scale_y = original_size[0] / self.input_size[0]
        
        x = float(x_pred * scale_x)
        y = float(y_pred * scale_y)
        
        return (x, y, max_val)
    
    def track(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Track badminton ball
        
        Args:
            frame: Video frame
            
        Returns:
            (x, y, confidence) or None
        """
        if self.model is None:
            return None
        
        original_size = (frame.shape[0], frame.shape[1])
        
        # Update frame buffer
        self.frame_buffer.append(frame.copy())
        
        # Maintain buffer size as seq_len
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)
        
        # If buffer is not full, fill with current frame
        if len(self.frame_buffer) < self.seq_len:
            while len(self.frame_buffer) < self.seq_len:
                self.frame_buffer.insert(0, self.frame_buffer[0] if self.frame_buffer else frame)
        
        # Preprocess sequence
        input_tensor = self.preprocess_sequence(self.frame_buffer)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess (take prediction result from last frame)
        result = self.postprocess(output, original_size, frame_idx=-1)
        
        return result
    
    def reset(self):
        """Reset tracker (clear frame buffer)"""
        self.frame_buffer.clear()
        self.background_frame = None