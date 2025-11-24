"""Hand pose tracking using MediaPipe and depth information."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Type alias for handedness
Handedness = Literal["Left", "Right"]

# Download URL for hand landmarker model
HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

EPS = 1e-6

class SimpleLandmark:
    def __init__(self, x: float, y: float, w: int, h: int) -> None:
        self.x = x / w
        self.y = y / h


@dataclass
class HandPose:
    """Represents a detected hand pose."""

    landmarks_2d: np.ndarray  # 21x2 array of 2D keypoints
    landmarks_3d: np.ndarray  # 21x3 array of 3D keypoints in camera frame
    handedness: Handedness  # 'Left' or 'Right'
    confidence: float
    wrist_pose: np.ndarray  # 4x4 transformation matrix (hand root)
    timestamp: float


class HandTracker:
    """Tracks hand pose from RGB and RGB-D cameras."""

    # MediaPipe hand landmark indices
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

    def __init__(
        self,
        model_path: str | None = None,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_factor: float = 0.0,
    ) -> None:
        """Initialize hand tracker using modern MediaPipe HandLandmarker API.

        Args:
            model_path: Path to hand_landmarker.task model file. If None, downloads default model.
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            smoothing_factor: Exponential smoothing factor for 3D landmarks (0.0 = no smoothing, 1.0 = max smoothing).
                              Higher values = more smoothing but more lag.
        """
        # Download model if not provided
        if model_path is None:
            model_path = self._download_model()

        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # Use VIDEO mode for continuous tracking
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.mp_hands = mp.solutions.hands  # Keep for connection constants
        
        # Smoothing state
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))  # Clamp to [0, 1]
        self.previous_landmarks: dict[str, np.ndarray] = {}  # Store previous smoothed landmarks per hand

    def _download_model(self) -> str:
        """Download the hand landmarker model if not already cached."""
        import urllib.request

        # Cache model in user's home directory
        cache_dir = Path.home() / ".cache" / "handpose"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "hand_landmarker.task"

        if not model_path.exists():
            print(f"Downloading hand landmarker model to {model_path}...")
            urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, model_path)
            print("✓ Model downloaded")

        return str(model_path)

    def detect_hands(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray | None = None,
        camera_matrix: np.ndarray | None = None,
        depth_scale: float = 1.0,
        timestamp: float = 0.0,
    ) -> list[HandPose]:
        """Detect hands in RGB image and estimate 3D pose using modern MediaPipe API.

        Args:
            rgb_image: BGR image (HxWx3) - will be converted to RGB internally
            depth_image: Optional depth image (HxW) in meters
            camera_matrix: Camera intrinsic matrix (3x3)
            depth_scale: Scale factor for depth values
            timestamp: Timestamp of the frame in milliseconds

        Returns:
            List of detected HandPose objects
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Convert timestamp to milliseconds for MediaPipe
        timestamp_ms = int(timestamp * 1000)

        # Detect hands using modern API
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_poses = []

        if results.hand_landmarks:
            for i, (hand_landmarks, hand_world_landmarks, handedness_list) in enumerate(
                zip(results.hand_landmarks, results.hand_world_landmarks, results.handedness)
            ):
                # Extract 2D landmarks (normalized coordinates)
                h, w = rgb_image.shape[:2]
                landmarks_2d = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks])

                # Extract or estimate 3D landmarks
                if depth_image is not None and camera_matrix is not None:
                    landmarks_3d = self._compute_3d_landmarks_from_depth(
                        landmarks_2d, depth_image, camera_matrix, depth_scale
                    )
                else:
                    # Use MediaPipe's world landmarks (metric 3D coordinates)
                    landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks])

                # Get handedness
                hand_label = handedness_list[0].category_name
                confidence = handedness_list[0].score

                # Apply smoothing to 3D landmarks
                landmarks_3d = self._apply_smoothing(landmarks_3d, hand_label)

                # Compute wrist pose (hand root transformation)
                wrist_pose = self._compute_wrist_pose(landmarks_3d)


                hand_pose = HandPose(
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    handedness=hand_label,
                    confidence=confidence,
                    wrist_pose=wrist_pose,
                    timestamp=timestamp,
                )

                hand_poses.append(hand_pose)

        return hand_poses

    def _compute_3d_landmarks_from_depth(
        self, landmarks_2d: np.ndarray, depth_image: np.ndarray, camera_matrix: np.ndarray, depth_scale: float
    ) -> np.ndarray:
        """Compute 3D landmarks using depth information.

        TODO: test this function.

        Args:
            landmarks_2d: 21x2 array of 2D landmarks
            depth_image: Depth image
            camera_matrix: Camera intrinsic matrix
            depth_scale: Depth scale factor

        Returns:
            21x3 array of 3D landmarks in camera frame
        """
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        landmarks_3d = []

        for x, y in landmarks_2d:
            # Sample depth around the landmark (for robustness)
            x_int, y_int = int(round(x)), int(round(y))

            # Clamp to image bounds
            x_int = max(0, min(x_int, depth_image.shape[1] - 1))
            y_int = max(0, min(y_int, depth_image.shape[0] - 1))

            # Get depth value with median filtering for robustness
            window_size = 5
            x_min = max(0, x_int - window_size // 2)
            x_max = min(depth_image.shape[1], x_int + window_size // 2 + 1)
            y_min = max(0, y_int - window_size // 2)
            y_max = min(depth_image.shape[0], y_int + window_size // 2 + 1)

            depth_window = depth_image[y_min:y_max, x_min:x_max]
            depth = np.median(depth_window[depth_window > 0]) if np.any(depth_window > 0) else 0
            depth = depth * depth_scale

            if depth > 0:
                # Backproject to 3D
                z = depth
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                landmarks_3d.append([x_3d, y_3d, z])
            else:
                # If no depth, use zero or estimate
                landmarks_3d.append([0, 0, 0])

        return np.array(landmarks_3d)

    def _compute_wrist_pose(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """Compute wrist pose (hand root) as a 4x4 transformation matrix.

        Uses stable palm landmarks that don't move when fingers open/close.

        Args:
            landmarks_3d: 21x3 array of 3D landmarks

        Returns:
            4x4 transformation matrix representing hand root frame
        """
        # Use wrist as origin
        wrist = landmarks_3d[self.WRIST]

        # Use stable palm landmarks (MCP joints are at the base of fingers)
        # These move less when fingers open/close compared to PIP/DIP/TIP
        index_mcp = landmarks_3d[self.INDEX_FINGER_MCP]
        middle_mcp = landmarks_3d[self.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks_3d[self.RING_FINGER_MCP]
        pinky_mcp = landmarks_3d[self.PINKY_MCP]

        # Compute palm center as average of MCP joints (more stable than individual points)
        palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0

        # X-axis: from wrist towards palm center (forward along hand)
        x_axis = palm_center - wrist
        x_axis = x_axis / (np.linalg.norm(x_axis) + EPS)

        # Compute palm normal using multiple MCP points for stability
        # Use index-pinky line and middle-ring line to define palm plane
        palm_vec1 = pinky_mcp - index_mcp  # Sideways across palm
        palm_vec2 = middle_mcp - ring_mcp  # Another sideways vector

        # Average the two vectors for more stability
        palm_sideways = (palm_vec1 + palm_vec2) / 2.0
        palm_sideways = palm_sideways / (np.linalg.norm(palm_sideways) + EPS)

        # Z-axis: sideways direction (thumb to pinky)
        z_axis = palm_sideways
        z_axis = z_axis / (np.linalg.norm(z_axis) + EPS)

        # Y-axis: perpendicular to palm (out of palm, using right-hand rule)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + EPS)

        # Recompute X and Z to ensure orthogonality
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + EPS)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + EPS)

        # Construct transformation matrix
        transform = np.eye(4)
        transform[:3, 0] = x_axis
        transform[:3, 1] = y_axis
        transform[:3, 2] = z_axis
        transform[:3, 3] = wrist

        return transform

    def _apply_smoothing(self, landmarks_3d: np.ndarray, handedness: str) -> np.ndarray:
        """Apply exponential smoothing to 3D landmarks.

        Args:
            landmarks_3d: 21x3 array of current 3D landmarks
            handedness: Hand label ('Left' or 'Right') used as key for tracking

        Returns:
            Smoothed 21x3 array of landmarks
        """
        if self.smoothing_factor <= 0.0:
            # No smoothing, just store current landmarks
            self.previous_landmarks[handedness] = landmarks_3d.copy()
            return landmarks_3d

        alpha = 1.0 - self.smoothing_factor

        if handedness in self.previous_landmarks:
            smoothed = alpha * landmarks_3d + (1.0 - alpha) * self.previous_landmarks[handedness]
        else:
            smoothed = landmarks_3d.copy()

        self.previous_landmarks[handedness] = smoothed.copy()

        return smoothed

    def landmarks_in_hand_frame(self, hand_pose: HandPose) -> np.ndarray:
        """Transform landmarks to hand root coordinate frame.

        Args:
            hand_pose: HandPose object

        Returns:
            21x3 array of landmarks in hand root frame
        """
        # Get inverse of wrist pose
        wrist_pose_inv = np.linalg.inv(hand_pose.wrist_pose)

        # Transform landmarks
        landmarks_homogeneous = np.hstack([hand_pose.landmarks_3d, np.ones((len(hand_pose.landmarks_3d), 1))])

        landmarks_hand_frame = (wrist_pose_inv @ landmarks_homogeneous.T).T

        return landmarks_hand_frame[:, :3]

    def draw_landmarks(self, image: np.ndarray, hand_poses: list[HandPose]) -> np.ndarray:
        """Draw hand landmarks on image.

        Args:
            image: Input image
            hand_poses: list of HandPose objects

        Returns:
            Image with drawn landmarks
        """
        annotated_image = image.copy()

        for hand_pose in hand_poses:
            # Convert landmarks to MediaPipe format
            landmarks = []
            h, w = image.shape[:2]

            for x, y in hand_pose.landmarks_2d:
                landmarks.append(SimpleLandmark(x, y, w, h))

            # Draw connections manually for simplicity
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]

                start_point = tuple(hand_pose.landmarks_2d[start_idx].astype(int))
                end_point = tuple(hand_pose.landmarks_2d[end_idx].astype(int))

                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

            # Draw landmarks
            for x, y in hand_pose.landmarks_2d:
                cv2.circle(annotated_image, (int(x), int(y)), 4, (0, 0, 255), -1)

            # Add handedness label
            wrist_2d = hand_pose.landmarks_2d[self.WRIST].astype(int)
            label = f"{hand_pose.handedness} ({hand_pose.confidence:.2f})"
            cv2.putText(
                annotated_image,
                label,
                (wrist_2d[0], wrist_2d[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return annotated_image

    def __del__(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, "landmarker"):
            self.landmarker.close()
