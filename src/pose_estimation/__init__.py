"""Pose estimation module"""

# Import classes to trigger registration and make them available
from .yolo_pose import YOLOPoseEstimator  # noqa: F401
from .rtmw import RTMWPoseEstimator  # noqa: F401

__all__ = [
    'YOLOPoseEstimator',
    'RTMWPoseEstimator',
]
