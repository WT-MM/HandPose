"""Retargeting human hand poses to ORCA robot hand."""

import json
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np

# Type alias for handedness
Handedness = Literal["Left", "Right"]


@dataclass
class ORCAHandConfig:
    """Configuration for ORCA hand."""

    # Joint limits (in radians)
    joint_limits: dict[str, tuple[float, float]]

    # Link lengths (in meters)
    link_lengths: dict[str, float]

    # Finger mapping from human to robot
    finger_mapping: dict[str, str]

    # Scale factor (robot hand size / human hand size)
    scale_factor: float = 0.7

    @classmethod
    def default_config(cls) -> Self:
        joint_limits = {
            # Thumb joints (CMC = carpometacarpal, MCP = metacarpophalangeal, IP = interphalangeal)
            "thumb_cmc_abd": (-1.082, 0.0),  # right_thumb_abd
            "thumb_cmc_flex": (-0.873, 0.873),  # right_thumb_mcp
            "thumb_mcp": (-0.794, 1.230),  # right_thumb_pip
            "thumb_ip": (-0.854, 1.450),  # right_thumb_dip
            # Index finger
            "index_mcp_abd": (-1.046, 0.246),  # right_index_abd
            "index_mcp_flex": (-0.349, 1.658),  # right_index_mcp
            "index_pip": (-0.349, 1.885),  # right_index_pip
            "index_dip": (0, 1.57),  # Not in ORCA
            # Middle finger
            "middle_mcp_abd": (-0.646, 0.646),  # right_middle_abd
            "middle_mcp_flex": (-0.349, 1.588),  # right_middle_mcp
            "middle_pip": (-0.349, 1.867),  # right_middle_pip
            "middle_dip": (0, 1.57),
            # Ring finger
            "ring_mcp_abd": (-0.476, 0.806),  # right_ring_abd
            "ring_mcp_flex": (-0.349, 1.588),  # right_ring_mcp
            "ring_pip": (-0.349, 1.867),  # right_ring_pip
            "ring_dip": (0, 1.57),
            # Pinky finger
            "pinky_mcp_abd": (-0.122, 1.169),  # right_pinky_abd
            "pinky_mcp_flex": (-0.349, 1.710),  # right_pinky_mcp
            "pinky_pip": (-0.349, 1.885),  # right_pinky_pip
            "pinky_dip": (0, 1.57),
        }

        link_lengths = {
            # Thumb
            "thumb_proximal": 0.038,
            "thumb_middle": 0.032,
            "thumb_distal": 0.027,
            # Index
            "index_proximal": 0.043,
            "index_middle": 0.028,
            "index_distal": 0.020,
            # Middle
            "middle_proximal": 0.048,
            "middle_middle": 0.032,
            "middle_distal": 0.022,
            # Ring
            "ring_proximal": 0.045,
            "ring_middle": 0.029,
            "ring_distal": 0.021,
            # Pinky
            "pinky_proximal": 0.037,
            "pinky_middle": 0.024,
            "pinky_distal": 0.018,
        }

        finger_mapping = {"thumb": "thumb", "index": "index", "middle": "middle", "ring": "ring", "pinky": "pinky"}

        return cls(
            joint_limits=joint_limits, link_lengths=link_lengths, finger_mapping=finger_mapping, scale_factor=0.7
        )


class ORCAHandRetargeting:
    """Retargets human hand poses to ORCA robot hand."""

    # MediaPipe landmark indices (same as in HandTracker)
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self, config: ORCAHandConfig | None = None) -> None:
        self.config = config or ORCAHandConfig.default_config()

    def retarget_pose(self, hand_landmarks_3d: np.ndarray, handedness: Handedness = "Right") -> dict[str, float]:
        """Retarget human hand pose to ORCA hand joint angles.

        Args:
            hand_landmarks_3d: 21x3 array of hand landmarks in hand frame
            handedness: 'Left' or 'Right'

        Returns:
            Dictionary of joint names to joint angles (radians)
        """
        joint_angles = {}

        # Mirror if left handed.
        if handedness == "Left":
            landmarks = self._mirror_hand(hand_landmarks_3d)
        else:
            landmarks = hand_landmarks_3d.copy()

        thumb_joints = self._retarget_thumb(landmarks)
        joint_angles.update(thumb_joints)

        index_joints = self._retarget_finger(
            landmarks,
            [self.INDEX_FINGER_MCP, self.INDEX_FINGER_PIP, self.INDEX_FINGER_DIP, self.INDEX_FINGER_TIP],
            "index",
        )
        joint_angles.update(index_joints)

        middle_joints = self._retarget_finger(
            landmarks,
            [self.MIDDLE_FINGER_MCP, self.MIDDLE_FINGER_PIP, self.MIDDLE_FINGER_DIP, self.MIDDLE_FINGER_TIP],
            "middle",
        )
        joint_angles.update(middle_joints)

        ring_joints = self._retarget_finger(
            landmarks, [self.RING_FINGER_MCP, self.RING_FINGER_PIP, self.RING_FINGER_DIP, self.RING_FINGER_TIP], "ring"
        )
        joint_angles.update(ring_joints)

        pinky_joints = self._retarget_finger(
            landmarks, [self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP], "pinky"
        )
        joint_angles.update(pinky_joints)

        # Apply joint limits
        joint_angles = self._apply_joint_limits(joint_angles)

        return joint_angles

    def _mirror_hand(self, landmarks: np.ndarray) -> np.ndarray:
        """Mirror hand landmarks for left-to-right conversion."""
        mirrored = landmarks.copy()
        mirrored[:, 0] = -mirrored[:, 0]  # Flip X axis
        return mirrored

    def _retarget_thumb(self, landmarks: np.ndarray) -> dict[str, float]:
        """Retarget thumb joints."""
        wrist = landmarks[self.WRIST]
        cmc = landmarks[self.THUMB_CMC]
        mcp = landmarks[self.THUMB_MCP]
        ip = landmarks[self.THUMB_IP]
        tip = landmarks[self.THUMB_TIP]

        # CMC abduction (angle in XY plane relative to palm center)
        # Use palm center as reference for consistent abduction computation
        index_mcp = landmarks[self.INDEX_FINGER_MCP]
        middle_mcp = landmarks[self.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks[self.RING_FINGER_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0

        vec_to_cmc = cmc - wrist
        vec_to_palm_center = palm_center - wrist

        # Project onto XY plane and compute angle
        vec_to_cmc_xy = vec_to_cmc[:2]
        vec_to_palm_center_xy = vec_to_palm_center[:2]

        abd_angle = self._angle_between_vectors(vec_to_palm_center_xy, vec_to_cmc_xy)

        # CMC flexion (angle from palm plane)
        cmc_flex = self._compute_joint_angle(wrist, cmc, mcp)

        # MCP joint
        mcp_angle = self._compute_joint_angle(cmc, mcp, ip)

        # IP joint
        ip_angle = self._compute_joint_angle(mcp, ip, tip)

        return {
            "thumb_cmc_abd": abd_angle,
            "thumb_cmc_flex": cmc_flex,
            "thumb_mcp": mcp_angle,
            "thumb_ip": ip_angle,
        }

    def _retarget_finger(self, landmarks: np.ndarray, joint_indices: list[int], finger_name: str) -> dict[str, float]:
        """Retarget a finger (index, middle, ring, or pinky)."""
        wrist = landmarks[self.WRIST]
        mcp = landmarks[joint_indices[0]]
        pip = landmarks[joint_indices[1]]
        dip = landmarks[joint_indices[2]]
        tip = landmarks[joint_indices[3]]

        # MCP abduction (relative to palm center direction)
        # Use the proximal phalanx (MCP to PIP) to measure finger direction
        index_mcp = landmarks[self.INDEX_FINGER_MCP]
        middle_mcp = landmarks[self.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks[self.RING_FINGER_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0
        
        # Use proximal phalanx direction (MCP to PIP) instead of metacarpal (wrist to MCP)
        vec_to_finger = pip - mcp
        vec_to_palm_center = palm_center - wrist
        
        # Reference direction: from wrist to palm center (forward along hand)
        # Abduction is the angle from this reference to the finger direction (proximal phalanx)
        abd_angle = self._angle_between_vectors(vec_to_palm_center[:2], vec_to_finger[:2])

        # MCP flexion
        mcp_flex = self._compute_joint_angle(wrist, mcp, pip)

        # PIP joint
        pip_angle = self._compute_joint_angle(mcp, pip, dip)

        # DIP joint
        dip_angle = self._compute_joint_angle(pip, dip, tip)

        return {
            f"{finger_name}_mcp_abd": abd_angle,
            f"{finger_name}_mcp_flex": mcp_flex,
            f"{finger_name}_pip": pip_angle,
            f"{finger_name}_dip": dip_angle,
        }

    def _compute_joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Compute joint angle at p2 formed by p1-p2-p3.

        Args:
            p1: First point (proximal)
            p2: Joint point
            p3: Third point (distal)

        Returns:
            Joint angle in radians
        """
        v1 = p1 - p2
        v2 = p3 - p2

        # Compute angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Return flexion angle (π - angle gives flexion from straight)
        return np.pi - angle

    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute signed angle between two 2D vectors."""
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

        # Normalize to [-π, π]
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi

        return angle

    def _apply_joint_limits(self, joint_angles: dict[str, float]) -> dict[str, float]:
        """Apply joint limits to retargeted angles."""
        limited_angles = {}

        for joint_name, angle in joint_angles.items():
            if joint_name in self.config.joint_limits:
                min_angle, max_angle = self.config.joint_limits[joint_name]
                limited_angles[joint_name] = np.clip(angle, min_angle, max_angle)
            else:
                limited_angles[joint_name] = angle

        return limited_angles

    def retarget_trajectory(self, trajectory_landmarks: np.ndarray, handedness: Handedness = "Right") -> np.ndarray:
        """Retarget a full trajectory.

        Args:
            trajectory_landmarks: Tx21x3 array of hand landmarks over time
            handedness: 'Left' or 'Right'

        Returns:
            TxN array of joint angles over time (N = number of joints)
        """
        n_frames = trajectory_landmarks.shape[0]

        # Get joint names from first frame
        first_angles = self.retarget_pose(trajectory_landmarks[0], handedness)
        joint_names = sorted(first_angles.keys())
        n_joints = len(joint_names)

        # Retarget each frame
        trajectory_angles = np.zeros((n_frames, n_joints))

        for i in range(n_frames):
            angles = self.retarget_pose(trajectory_landmarks[i], handedness)
            for j, joint_name in enumerate(joint_names):
                trajectory_angles[i, j] = angles[joint_name]

        return trajectory_angles

    def export_for_simulation(self, joint_angles: dict[str, float], format: str = "json") -> str:
        """Export joint angles in simulation-compatible format.

        Args:
            joint_angles: Dictionary of joint angles
            format: Output format ('json', 'array')

        Returns:
            Formatted string
        """
        if format == "json":
            return json.dumps(joint_angles, indent=2)
        elif format == "array":
            # Return as ordered array for direct use in simulation
            joint_names = sorted(joint_angles.keys())
            values = [joint_angles[name] for name in joint_names]
            return str(values)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_config(self, filepath: str) -> None:
        """Save ORCA hand configuration."""
        config_dict = {
            "joint_limits": self.config.joint_limits,
            "link_lengths": self.config.link_lengths,
            "finger_mapping": self.config.finger_mapping,
            "scale_factor": self.config.scale_factor,
        }

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, filepath: str) -> Self:
        """Load ORCA hand configuration."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)

        config = ORCAHandConfig(
            joint_limits=config_dict["joint_limits"],
            link_lengths=config_dict["link_lengths"],
            finger_mapping=config_dict["finger_mapping"],
            scale_factor=config_dict["scale_factor"],
        )

        return cls(config)
