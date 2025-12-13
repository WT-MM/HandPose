"""Hand pose tracking with support for multiple backends."""

from .base import (
    EPS,
    BaseHandTracker,
    FingerJoints,
    Handedness,
    HandStructure,
)
from .hamer import HaMeRTracker
from .mediapipe import MediaPipeTracker

# Backward compatibility: HandTracker is an alias for MediaPipeTracker
HandTracker = MediaPipeTracker

__all__ = [
    "BaseHandTracker",
    "HandStructure",
    "FingerJoints",
    "Handedness",
    "MediaPipeTracker",
    "HandTracker",
    "EPS",
    "HaMeRTracker",
]
