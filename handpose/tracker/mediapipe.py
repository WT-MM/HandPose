"""MediaPipe-based hand tracker implementation."""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base import BaseHandTracker, FingerJoints, HandStructure


class MediaPipeTracker(BaseHandTracker):
    """MediaPipe-based hand tracker."""

    HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

    # MediaPipe landmark indices
    WRIST = 0
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
        """Initialize MediaPipe hand tracker.

        Args:
            model_path: Path to hand_landmarker.task model file. If None, downloads default model.
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            smoothing_factor: Exponential smoothing factor for 3D landmarks.
        """
        super().__init__(smoothing_factor=smoothing_factor)

        if model_path is None:
            model_path = self._download_model()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.mp_hands = mp.solutions.hands

    def _download_model(self) -> str:
        """Download the hand landmarker model if not already cached."""
        cache_dir = Path.home() / ".cache" / "handpose"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "hand_landmarker.task"

        if not model_path.exists():
            print(f"Downloading hand landmarker model to {model_path}...")
            urllib.request.urlretrieve(self.HAND_LANDMARKER_MODEL_URL, model_path)
            print("✓ Model downloaded")

        return str(model_path)

    def detect_hands(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray | None = None,
        camera_matrix: np.ndarray | None = None,
        depth_scale: float = 1.0,
        timestamp: float = 0.0,
    ) -> list[HandStructure]:
        """Detect hands using MediaPipe and return HandStructure."""
        image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(timestamp * 1000)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        structures = []

        if results.hand_landmarks:
            for hand_landmarks, hand_world_landmarks, handedness_list in zip(
                results.hand_landmarks, results.hand_world_landmarks, results.handedness
            ):
                h, w = rgb_image.shape[:2]
                landmarks_2d = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks])

                if depth_image is not None and camera_matrix is not None:
                    landmarks_3d = self._compute_3d_landmarks_from_depth(
                        landmarks_2d, depth_image, camera_matrix, depth_scale
                    )
                else:
                    landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks])

                hand_label = handedness_list[0].category_name
                confidence = handedness_list[0].score

                landmarks_3d = self._apply_smoothing(landmarks_3d, hand_label)
                wrist_pose = self._compute_wrist_pose(landmarks_3d)

                # Transform landmarks to hand frame
                wrist_pose_inv = np.linalg.inv(wrist_pose)
                landmarks_homogeneous = np.hstack([landmarks_3d, np.ones((len(landmarks_3d), 1))])
                landmarks_hand_frame = (wrist_pose_inv @ landmarks_homogeneous.T).T[:, :3]

                # Extract finger joints directly
                thumb = FingerJoints(
                    mcp=landmarks_hand_frame[self.THUMB_MCP],
                    ip=landmarks_hand_frame[self.THUMB_IP],
                    tip=landmarks_hand_frame[self.THUMB_TIP],
                    mcp_local=landmarks_hand_frame[self.THUMB_MCP] - landmarks_hand_frame[self.WRIST],
                    ip_local=landmarks_hand_frame[self.THUMB_IP] - landmarks_hand_frame[self.THUMB_MCP],
                    tip_local=landmarks_hand_frame[self.THUMB_TIP] - landmarks_hand_frame[self.THUMB_IP],
                )

                index = FingerJoints(
                    mcp=landmarks_hand_frame[self.INDEX_FINGER_MCP],
                    pip=landmarks_hand_frame[self.INDEX_FINGER_PIP],
                    dip=landmarks_hand_frame[self.INDEX_FINGER_DIP],
                    tip=landmarks_hand_frame[self.INDEX_FINGER_TIP],
                    mcp_local=landmarks_hand_frame[self.INDEX_FINGER_MCP] - landmarks_hand_frame[self.WRIST],
                    pip_local=landmarks_hand_frame[self.INDEX_FINGER_PIP] - landmarks_hand_frame[self.INDEX_FINGER_MCP],
                    dip_local=landmarks_hand_frame[self.INDEX_FINGER_DIP] - landmarks_hand_frame[self.INDEX_FINGER_PIP],
                    tip_local=landmarks_hand_frame[self.INDEX_FINGER_TIP] - landmarks_hand_frame[self.INDEX_FINGER_DIP],
                )

                middle = FingerJoints(
                    mcp=landmarks_hand_frame[self.MIDDLE_FINGER_MCP],
                    pip=landmarks_hand_frame[self.MIDDLE_FINGER_PIP],
                    dip=landmarks_hand_frame[self.MIDDLE_FINGER_DIP],
                    tip=landmarks_hand_frame[self.MIDDLE_FINGER_TIP],
                    mcp_local=landmarks_hand_frame[self.MIDDLE_FINGER_MCP] - landmarks_hand_frame[self.WRIST],
                    pip_local=landmarks_hand_frame[self.MIDDLE_FINGER_PIP]
                    - landmarks_hand_frame[self.MIDDLE_FINGER_MCP],
                    dip_local=landmarks_hand_frame[self.MIDDLE_FINGER_DIP]
                    - landmarks_hand_frame[self.MIDDLE_FINGER_PIP],
                    tip_local=landmarks_hand_frame[self.MIDDLE_FINGER_TIP]
                    - landmarks_hand_frame[self.MIDDLE_FINGER_DIP],
                )

                ring = FingerJoints(
                    mcp=landmarks_hand_frame[self.RING_FINGER_MCP],
                    pip=landmarks_hand_frame[self.RING_FINGER_PIP],
                    dip=landmarks_hand_frame[self.RING_FINGER_DIP],
                    tip=landmarks_hand_frame[self.RING_FINGER_TIP],
                    mcp_local=landmarks_hand_frame[self.RING_FINGER_MCP] - landmarks_hand_frame[self.WRIST],
                    pip_local=landmarks_hand_frame[self.RING_FINGER_PIP] - landmarks_hand_frame[self.RING_FINGER_MCP],
                    dip_local=landmarks_hand_frame[self.RING_FINGER_DIP] - landmarks_hand_frame[self.RING_FINGER_PIP],
                    tip_local=landmarks_hand_frame[self.RING_FINGER_TIP] - landmarks_hand_frame[self.RING_FINGER_DIP],
                )

                pinky = FingerJoints(
                    mcp=landmarks_hand_frame[self.PINKY_MCP],
                    pip=landmarks_hand_frame[self.PINKY_PIP],
                    dip=landmarks_hand_frame[self.PINKY_DIP],
                    tip=landmarks_hand_frame[self.PINKY_TIP],
                    mcp_local=landmarks_hand_frame[self.PINKY_MCP] - landmarks_hand_frame[self.WRIST],
                    pip_local=landmarks_hand_frame[self.PINKY_PIP] - landmarks_hand_frame[self.PINKY_MCP],
                    dip_local=landmarks_hand_frame[self.PINKY_DIP] - landmarks_hand_frame[self.PINKY_PIP],
                    tip_local=landmarks_hand_frame[self.PINKY_TIP] - landmarks_hand_frame[self.PINKY_DIP],
                )

                structure = HandStructure(
                    wrist_pose=wrist_pose,
                    wrist_position=landmarks_hand_frame[self.WRIST],
                    thumb=thumb,
                    index=index,
                    middle=middle,
                    ring=ring,
                    pinky=pinky,
                    handedness=hand_label,
                    confidence=confidence,
                    timestamp=timestamp,
                )
                structure.landmarks_2d = landmarks_2d.astype(np.float32)
                structures.append(structure)

        return structures

    def visualize(
        self,
        image: np.ndarray,
        hand_structures: list[HandStructure],
        camera_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Visualize MediaPipe hand detections using 2D landmarks."""
        annotated_image = image.copy()

        for structure in hand_structures:
            joints_2d = getattr(structure, "landmarks_2d", None)

            if joints_2d is None:
                # Fallback to old behavior if 2D isn't available
                joints_3d_camera = self._structure_to_camera_frame(structure)
                if camera_matrix is not None:
                    joints_2d = self._project_3d_to_2d(joints_3d_camera, camera_matrix)
                else:
                    h, w = image.shape[:2]
                    joints_2d = np.zeros((21, 2), dtype=np.float32)
                    for i, joint_3d in enumerate(joints_3d_camera):
                        if joint_3d[2] > 1e-6:
                            joints_2d[i] = [
                                joint_3d[0] / joint_3d[2] * w / 2 + w / 2,
                                joint_3d[1] / joint_3d[2] * h / 2 + h / 2,
                            ]

            # Draw connections
            for start_idx, end_idx in self.mp_hands.HAND_CONNECTIONS:
                p0 = tuple(joints_2d[start_idx].astype(int))
                p1 = tuple(joints_2d[end_idx].astype(int))
                cv2.line(annotated_image, p0, p1, (0, 255, 0), 2)

            # Draw keypoints
            for x, y in joints_2d:
                cv2.circle(annotated_image, (int(x), int(y)), 4, (0, 0, 255), -1)

            # Label
            wrist = joints_2d[self.WRIST].astype(int)
            cv2.putText(
                annotated_image,
                f"{structure.handedness} ({structure.confidence:.2f})",
                (wrist[0], wrist[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return annotated_image

    def _structure_to_camera_frame(self, structure: HandStructure) -> np.ndarray:
        """Convert HandStructure joints to camera frame."""
        # Reconstruct 21 joints from HandStructure
        joints_hand_frame = np.array(
            [
                structure.wrist_position,
                structure.thumb.mcp,  # CMC (approximate)
                structure.thumb.mcp,
                structure.thumb.ip,
                structure.thumb.tip,
                structure.index.mcp,
                structure.index.pip,
                structure.index.dip,
                structure.index.tip,
                structure.middle.mcp,
                structure.middle.pip,
                structure.middle.dip,
                structure.middle.tip,
                structure.ring.mcp,
                structure.ring.pip,
                structure.ring.dip,
                structure.ring.tip,
                structure.pinky.mcp,
                structure.pinky.pip,
                structure.pinky.dip,
                structure.pinky.tip,
            ]
        )

        # Transform to camera frame
        joints_homogeneous = np.hstack([joints_hand_frame, np.ones((21, 1))])
        joints_camera = (structure.wrist_pose @ joints_homogeneous.T).T[:, :3]
        return joints_camera

    def _project_3d_to_2d(self, joints_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Project 3D joints to 2D using camera matrix."""
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        x_2d = (joints_3d[:, 0] * fx / joints_3d[:, 2]) + cx
        y_2d = (joints_3d[:, 1] * fy / joints_3d[:, 2]) + cy

        return np.stack([x_2d, y_2d], axis=1)

    def __del__(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, "landmarker"):
            self.landmarker.close()
