"""Base classes and primitives for hand tracking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

Handedness = Literal["Left", "Right"]

EPS = 1e-6


@dataclass
class FingerJoints:
    """Represents joints of a single finger in a kinematic chain."""

    mcp: np.ndarray  # 3D position of MCP joint (metacarpophalangeal)
    tip: np.ndarray  # 3D position of fingertip
    pip: np.ndarray | None = None  # 3D position of PIP joint (proximal interphalangeal), None if not available
    dip: np.ndarray | None = None  # 3D position of DIP joint (distal interphalangeal), None if not available
    ip: np.ndarray | None = None  # 3D position of IP joint (interphalangeal, for thumb), None if not available

    # Joint positions in parent frame (for retargeting)
    mcp_local: np.ndarray | None = None  # MCP relative to wrist/palm
    pip_local: np.ndarray | None = None  # PIP relative to MCP
    dip_local: np.ndarray | None = None  # DIP relative to PIP
    ip_local: np.ndarray | None = None  # IP relative to MCP (thumb)
    tip_local: np.ndarray | None = None  # Tip relative to parent joint

    def get_joint(self, joint_type: str) -> np.ndarray | None:
        """Get joint position by type."""
        joint_map = {
            "mcp": self.mcp,
            "pip": self.pip,
            "dip": self.dip,
            "ip": self.ip,
            "tip": self.tip,
        }
        return joint_map.get(joint_type)

    def get_joint_local(self, joint_type: str) -> np.ndarray | None:
        """Get joint position in parent frame by type."""
        joint_map = {
            "mcp": self.mcp_local,
            "pip": self.pip_local,
            "dip": self.dip_local,
            "ip": self.ip_local,
            "tip": self.tip_local,
        }
        return joint_map.get(joint_type)


@dataclass
class HandStructure:
    """Structured hand representation optimized for retargeting.

    Represents the hand as a kinematic tree with fingers as chains.
    All positions are in the hand root frame (wrist at origin).
    """

    wrist_pose: np.ndarray  # 4x4 transformation matrix (hand root in camera frame)
    wrist_position: np.ndarray  # 3D wrist position in hand frame (should be [0,0,0])

    # Fingers as kinematic chains
    thumb: FingerJoints
    index: FingerJoints
    middle: FingerJoints
    ring: FingerJoints
    pinky: FingerJoints

    # Metadata
    handedness: Handedness
    confidence: float
    timestamp: float

    # Optional 2D landmark locations in the *input image pixel coordinates*
    # Shape: (21, 2), dtype float
    landmarks_2d: np.ndarray | None = None

    def get_finger(self, finger_name: str) -> FingerJoints:
        """Get finger by name."""
        finger_map = {
            "thumb": self.thumb,
            "index": self.index,
            "middle": self.middle,
            "ring": self.ring,
            "pinky": self.pinky,
        }
        return finger_map[finger_name]

    def get_all_joints(self, joint_type: str) -> dict[str, np.ndarray]:
        """Get all joints of a specific type across all fingers."""
        joints = {}
        for finger_name in ["thumb", "index", "middle", "ring", "pinky"]:
            finger = self.get_finger(finger_name)
            joint = finger.get_joint(joint_type)
            if joint is not None:
                joints[finger_name] = joint
        return joints

    def get_all_joints_local(self, joint_type: str) -> dict[str, np.ndarray]:
        """Get all joints in parent frame of a specific type across all fingers."""
        joints = {}
        for finger_name in ["thumb", "index", "middle", "ring", "pinky"]:
            finger = self.get_finger(finger_name)
            joint = finger.get_joint_local(joint_type)
            if joint is not None:
                joints[finger_name] = joint
        return joints


class BaseHandTracker(ABC):
    """Abstract base class for hand trackers."""

    def __init__(self, smoothing_factor: float = 0.0) -> None:
        """Initialize base tracker.

        Args:
            smoothing_factor: Exponential smoothing factor for 3D landmarks (0.0 = no smoothing, 1.0 = max smoothing).
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self.previous_landmarks: dict[str, np.ndarray] = {}

    @abstractmethod
    def detect_hands(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray | None = None,
        camera_matrix: np.ndarray | None = None,
        depth_scale: float = 1.0,
        timestamp: float = 0.0,
    ) -> list[HandStructure]:
        """Detect hands in RGB image and estimate 3D pose.

        Args:
            rgb_image: BGR image (HxWx3) - will be converted to RGB internally
            depth_image: Optional depth image (HxW) in meters
            camera_matrix: Camera intrinsic matrix (3x3)
            depth_scale: Scale factor for depth values
            timestamp: Timestamp of the frame in milliseconds

        Returns:
            List of detected HandStructure objects
        """
        pass

    @abstractmethod
    def visualize(
        self,
        image: np.ndarray,
        hand_structures: list[HandStructure],
        camera_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Visualize hand detections on image.

        Each tracker can implement this differently:
        - MediaPipe: Draw 2D landmarks
        - HaMeR: Render mesh or project to 2D

        Args:
            image: Input image (BGR format)
            hand_structures: List of detected HandStructure objects
            camera_matrix: Optional camera matrix for 3D projection

        Returns:
            Image with visualizations overlaid
        """
        pass

    def _compute_wrist_pose(self, joints_3d: np.ndarray) -> np.ndarray:
        """Compute wrist pose (hand root) as a 4x4 transformation matrix.

        Uses stable palm landmarks that don't move when fingers open/close.

        Args:
            joints_3d: Array of 3D joint positions (at least wrist and MCP joints)

        Returns:
            4x4 transformation matrix representing hand root frame
        """
        # Extract key joints (assuming standard ordering: wrist, thumb_mcp, index_mcp, etc.)
        wrist = joints_3d[0]
        index_mcp = joints_3d[5] if len(joints_3d) > 5 else joints_3d[1]
        middle_mcp = joints_3d[9] if len(joints_3d) > 9 else joints_3d[2]
        ring_mcp = joints_3d[13] if len(joints_3d) > 13 else joints_3d[3]
        pinky_mcp = joints_3d[17] if len(joints_3d) > 17 else joints_3d[4]

        palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0

        x_axis = palm_center - wrist
        x_axis = x_axis / (np.linalg.norm(x_axis) + EPS)

        palm_vec1 = pinky_mcp - index_mcp
        palm_vec2 = middle_mcp - ring_mcp
        palm_sideways = (palm_vec1 + palm_vec2) / 2.0
        palm_sideways = palm_sideways / (np.linalg.norm(palm_sideways) + EPS)

        z_axis = palm_sideways
        z_axis = z_axis / (np.linalg.norm(z_axis) + EPS)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + EPS)

        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + EPS)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + EPS)

        transform = np.eye(4)
        transform[:3, 0] = x_axis
        transform[:3, 1] = y_axis
        transform[:3, 2] = z_axis
        transform[:3, 3] = wrist

        return transform

    def _apply_smoothing(self, landmarks_3d: np.ndarray, handedness: str) -> np.ndarray:
        """Apply exponential smoothing to 3D landmarks.

        Args:
            landmarks_3d: Array of current 3D landmarks
            handedness: Hand label ('Left' or 'Right') used as key for tracking

        Returns:
            Smoothed array of landmarks
        """
        if self.smoothing_factor <= 0.0:
            self.previous_landmarks[handedness] = landmarks_3d.copy()
            return landmarks_3d

        alpha = 1.0 - self.smoothing_factor

        if handedness in self.previous_landmarks:
            smoothed = alpha * landmarks_3d + (1.0 - alpha) * self.previous_landmarks[handedness]
        else:
            smoothed = landmarks_3d.copy()

        self.previous_landmarks[handedness] = smoothed.copy()
        return smoothed

    def _compute_3d_landmarks_from_depth(
        self, landmarks_2d: np.ndarray, depth_image: np.ndarray, camera_matrix: np.ndarray, depth_scale: float
    ) -> np.ndarray:
        """Compute 3D landmarks using depth information.

        Args:
            landmarks_2d: Array of 2D landmarks
            depth_image: Depth image
            camera_matrix: Camera intrinsic matrix
            depth_scale: Depth scale factor

        Returns:
            Array of 3D landmarks in camera frame
        """
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        landmarks_3d = []

        for x, y in landmarks_2d:
            x_int, y_int = int(round(x)), int(round(y))
            x_int = max(0, min(x_int, depth_image.shape[1] - 1))
            y_int = max(0, min(y_int, depth_image.shape[0] - 1))

            window_size = 5
            x_min = max(0, x_int - window_size // 2)
            x_max = min(depth_image.shape[1], x_int + window_size // 2 + 1)
            y_min = max(0, y_int - window_size // 2)
            y_max = min(depth_image.shape[0], y_int + window_size // 2 + 1)

            depth_window = depth_image[y_min:y_max, x_min:x_max]
            depth = np.median(depth_window[depth_window > 0]) if np.any(depth_window > 0) else 0
            depth = depth * depth_scale

            if depth > 0:
                z = depth
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                landmarks_3d.append([x_3d, y_3d, z])
            else:
                landmarks_3d.append([0, 0, 0])

        return np.array(landmarks_3d)
