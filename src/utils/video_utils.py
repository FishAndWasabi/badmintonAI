"""Video processing utilities for online video streams"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional


def load_video_stream(stream_url: str) -> Tuple[cv2.VideoCapture, dict]:
    """
    Load online video stream (RTSP, RTMP, HTTP, etc.)
    
    Args:
        stream_url: Video stream URL (e.g., rtsp://, rtmp://, http://, or camera index)
        
    Returns:
        video_capture: OpenCV video capture object
        video_info: Video info dictionary (fps, width, height, frame_count=None for streams)
    """
    # Set buffer size to reduce latency for streams
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time processing
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video stream: {stream_url}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # For streams, frame_count is typically -1 or 0 (unknown)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = None
    
    video_info = {
        'fps': fps if fps > 0 else 30.0,  # Default to 30 if unknown
        'width': width,
        'height': height,
        'frame_count': frame_count  # None for live streams
    }
    
    return cap, video_info


def frame_generator(stream_url: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Video stream frame generator (real-time processing)
    
    Args:
        stream_url: Video stream URL
        
    Yields:
        frame_idx: Frame index
        frame: Video frame (BGR format)
    """
    cap, _ = load_video_stream(stream_url)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # For streams, try to continue reading (stream might be temporarily unavailable)
            continue
        yield frame_idx, frame
        frame_idx += 1
    
    cap.release()


def save_video(frames: list, output_path: str, fps: float, size: Tuple[int, int]):
    """
    Save video file
    
    Args:
        frames: List of frames
        output_path: Output path
        fps: Frame rate
        size: Video size (width, height)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
