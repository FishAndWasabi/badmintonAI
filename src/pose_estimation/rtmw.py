"""RTMW model for player pose estimation"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
import sys
import os

# 尝试导入RTMW（如果已克隆）
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../models/pose_estimation/RTMW'))
    # 这里需要根据RTMW的实际代码结构进行调整
except:
    pass


class RTMWPoseEstimator:
    """RTMW姿态估计器"""
    
    def __init__(self, model_path: Optional[str] = None,
                 input_size: Tuple[int, int] = (256, 192)):
        """
        初始化RTMW姿态估计器
        
        Args:
            model_path: 模型路径
            input_size: 输入图像尺寸 (height, width)
        """
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型（需要根据RTMW的实际实现进行调整）
        if model_path and os.path.exists(model_path):
            try:
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
            except:
                print(f"警告: 无法加载RTMW模型 {model_path}，使用占位实现")
                self.model = None
        else:
            print("警告: RTMW模型路径未指定或不存在，使用占位实现")
            self.model = None
    
    def preprocess(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            frame: 输入帧
            bbox: 边界框 (x1, y1, x2, y2)，如果为None则使用整帧
            
        Returns:
            预处理后的张量
        """
        if bbox:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
        
        # 调整大小
        resized = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
        
        # 转换为RGB
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = resized
        
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        
        # 转换为张量
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def postprocess(self, output: torch.Tensor, 
                   original_size: Tuple[int, int],
                   bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        后处理模型输出
        
        Args:
            output: 模型输出
            original_size: 原始图像尺寸 (height, width)
            bbox: 边界框（如果使用了裁剪）
            
        Returns:
            关键点数组 (17, 3) - (x, y, confidence)
        """
        if output is None:
            return np.zeros((17, 3))
        
        # 将输出转换为numpy
        if isinstance(output, torch.Tensor):
            output = output.cpu().detach().numpy()
        
        # 假设输出是关键点坐标
        if len(output.shape) == 3:
            keypoints = output[0]  # (17, 2) or (17, 3)
        else:
            keypoints = output
        
        # 缩放到原始图像尺寸
        if bbox:
            x1, y1, x2, y2 = bbox
            scale_x = (x2 - x1) / self.input_size[1]
            scale_y = (y2 - y1) / self.input_size[0]
            offset_x = x1
            offset_y = y1
        else:
            scale_x = original_size[1] / self.input_size[1]
            scale_y = original_size[0] / self.input_size[0]
            offset_x = 0
            offset_y = 0
        
        # 缩放关键点
        if keypoints.shape[1] == 2:
            # 添加置信度
            keypoints = np.concatenate([keypoints, np.ones((keypoints.shape[0], 1))], axis=1)
        
        keypoints[:, 0] = keypoints[:, 0] * scale_x + offset_x
        keypoints[:, 1] = keypoints[:, 1] * scale_y + offset_y
        
        return keypoints
    
    def estimate(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        估计视频帧中所有玩家的姿态
        
        Args:
            frame: 视频帧
            
        Returns:
            关键点列表
        """
        if self.model is None:
            # 占位实现：返回空列表
            return []
        
        original_size = (frame.shape[0], frame.shape[1])
        
        # 预处理
        input_tensor = self.preprocess(frame)
        
        # 推理
        with torch.no_grad():
            try:
                output = self.model(input_tensor)
            except Exception as e:
                print(f"RTMW推理错误: {e}")
                return []
        
        # 后处理
        keypoints = self.postprocess(output, original_size)
        
        return [keypoints] if keypoints is not None else []
