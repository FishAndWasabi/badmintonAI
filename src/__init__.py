"""
Badminton AI Analysis System
羽毛球AI分析系统
"""

__version__ = "0.1.0"

# Import modules to trigger registration (MMDetection style)
from . import ball_tracking  # noqa: F401
from . import pose_estimation  # noqa: F401
from . import shot_classification  # noqa: F401

__all__ = [
    '__version__',
    'ball_tracking',
    'pose_estimation',
    'shot_classification',
]
