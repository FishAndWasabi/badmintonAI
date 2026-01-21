"""Shot classification module"""

# Import classes to trigger registration and make them available
from .rule_based import RuleBasedClassifier, ShotInfo  # noqa: F401
from .classifier import MLShotClassifier  # noqa: F401

__all__ = [
    'RuleBasedClassifier',
    'MLShotClassifier',
    'ShotInfo',
]
