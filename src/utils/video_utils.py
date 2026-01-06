"""Video processing utilities"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional


def load_video(video_path: str) -> Tuple[cv2.VideoCapture, dict]:
    """
    加载视频文件
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        video_capture: OpenCV视频捕获对象
        video_info: 视频信息字典（fps, width, height, frame_count）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_info = {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count
    }
    
    return cap, video_info


def frame_generator(video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    视频帧生成器
    
    Args:
        video_path: 视频文件路径
        
    Yields:
        frame_idx: 帧索引
        frame: 视频帧（BGR格式）
    """
    cap, _ = load_video(video_path)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1
    
    cap.release()


def save_video(frames: list, output_path: str, fps: float, size: Tuple[int, int]):
    """
    保存视频文件
    
    Args:
        frames: 帧列表
        output_path: 输出路径
        fps: 帧率
        size: 视频尺寸 (width, height)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
