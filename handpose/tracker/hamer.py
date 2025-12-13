"""HaMeR-based hand tracker implementation."""

import os
from typing import Any

import cv2
import numpy as np
import torch  # type: ignore[import-not-found]

from .base import EPS, BaseHandTracker, FingerJoints, Handedness, HandStructure


class HaMeRTracker(BaseHandTracker):
    """HaMeR-based hand tracker."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        smoothing_factor: float = 0.0,
        conf_threshold: float = 0.3,
    ) -> None:
        """Initialize HaMeR hand tracker.

        Args:
            model_path: Path to HaMeR model checkpoint. If None, downloads default model.
            device: Device to run inference on ('cuda' or 'cpu'). If None, auto-detects.
            smoothing_factor: Exponential smoothing factor for 3D landmarks.
            conf_threshold: Confidence threshold for hand detection.
        """
        super().__init__(smoothing_factor=smoothing_factor)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold

        # IMPORTANT: pyrender OffscreenRenderer wants this key to be ABSENT
        # os.environ.pop("PYOPENGL_PLATFORM", None)

        # Import *after* popping env var
        import hamer.models.hamer as hamer_hamer
        from hamer.configs import CACHE_DIR_HAMER
        from hamer.models import download_models, load_hamer

        class _NoopMeshRenderer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
                pass

            def __call__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
                return None

            def render(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
                return None

        # Disable renderer construction
        hamer_hamer.MeshRenderer = _NoopMeshRenderer

        if model_path is None:
            ckpt = f"{CACHE_DIR_HAMER}/hamer_ckpts/checkpoints/hamer.ckpt"
            if not os.path.exists(ckpt):
                download_models(CACHE_DIR_HAMER)
            model_path = ckpt

        self.model, self.model_cfg = load_hamer(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._last_predictions: list[Any] = []  # Store raw predictions for visualization

        # Debug: Check if BBOX_SHAPE was set by load_hamer
        if hasattr(self.model_cfg.MODEL, "BBOX_SHAPE"):
            self._needs_wide_input = True
        else:
            self._needs_wide_input = False

    def detect_hands(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray | None = None,
        camera_matrix: np.ndarray | None = None,
        depth_scale: float = 1.0,
        timestamp: float = 0.0,
    ) -> list[HandStructure]:
        """Detect hands using HaMeR and return HandStructure."""
        h, w = rgb_image.shape[:2]
        image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Preprocess image for HaMeR
        batch = self._preprocess_image(image_rgb)

        structures: list[HandStructure] = []
        self._last_predictions = []  # Store raw predictions for visualization

        with torch.no_grad():
            # HaMeR model expects a batch dict, not just a tensor
            output = self.model(batch)

            # HaMeR returns a dict with keys: pred_vertices, pred_keypoints_3d,
            # pred_keypoints_2d, pred_mano_params, etc.
            if not isinstance(output, dict):
                return structures

            # Extract predictions from batch output
            batch_size = output["pred_vertices"].shape[0]

            for n in range(batch_size):
                # Extract data for this hand
                vertices = output["pred_vertices"][n].detach().cpu().numpy()
                keypoints_3d = output["pred_keypoints_3d"][n].detach().cpu().numpy()
                keypoints_2d = output["pred_keypoints_2d"][n].detach().cpu().numpy()
                mano_params_dict = output["pred_mano_params"]

                # Get handedness from batch (if available) or infer from model
                is_right = batch.get("right", torch.ones(1, device=self.device))[n].item() if "right" in batch else 1.0
                handedness: Handedness = "Right" if is_right > 0.5 else "Left"

                # Get confidence (HaMeR doesn't provide explicit confidence, use a default)
                confidence = 1.0

                # Store raw prediction for visualization
                self._last_predictions.append(
                    {
                        "vertices": vertices,
                        "keypoints_3d": keypoints_3d,
                        "keypoints_2d": keypoints_2d,
                        "mano": mano_params_dict,
                        "is_right": is_right,
                    }
                )

                # MANO outputs 21 joints (wrist + 4 per finger)
                # Use keypoints_3d directly - it's already in the correct format
                joints_3d = keypoints_3d

                # Convert MANO joints directly to HandStructure
                structure = self._mano_to_hand_structure(
                    joints_3d, vertices, keypoints_2d, handedness, confidence, timestamp, h, w
                )
                if structure:
                    structures.append(structure)

        return structures

    def _preprocess_image(self, image: np.ndarray) -> dict[str, Any]:
        """Preprocess image for HaMeR model.

        Returns a batch dict compatible with HaMeR's forward method.
        HaMeR expects images of size IMAGE_SIZE x IMAGE_SIZE (typically 256x256).
        For ViT backbone, the model crops [:, :, :, 32:-32] internally, so we just
        provide a 256x256 image and let the model handle the cropping.
        """
        # Get image size from config (default 256 for ViT backbone, 224 for others)
        img_size = getattr(self.model_cfg.MODEL, "IMAGE_SIZE", 256)

        # For ViT backbone, model crops [:, :, :, 32:-32] internally
        # The crop removes 32 pixels from each side of width: 256 - 64 = 192
        # Looking at vitdet_dataset.py, it creates square 256x256 patches
        # So we should provide 256x256 input, which gets cropped to 256x192
        # Use the flag set during initialization
        if self._needs_wide_input:
            # ViT expects square 256x256 input, which it crops to 256x192 internally
            target_width = img_size  # 256
            target_height = img_size  # 256
        else:
            # Non-ViT: square image
            target_width = img_size
            target_height = img_size

        # Resize maintaining aspect ratio, then pad to target size
        h, w = image.shape[:2]
        scale = min(target_height / h, target_width / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(image, (new_w, new_h))

        # Pad to target size (center padding)
        pad_h = target_height - new_h
        pad_w = target_width - new_w
        img_padded = cv2.copyMakeBorder(
            img_resized,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_w // 2,
            pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        # Convert BGR to RGB (image is already RGB from detect_hands, but be safe)
        # Make a copy to avoid negative strides when converting to tensor
        if len(img_padded.shape) == 3 and img_padded.shape[2] == 3:
            img_padded = img_padded[:, :, ::-1].copy()  # BGR to RGB, make copy
        else:
            img_padded = img_padded.copy()

        # Convert to tensor (HWC -> CHW) - values are in [0, 255]
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float()  # (3, H, W)

        # Get normalization from config or use defaults
        # HaMeR uses mean/std scaled by 255 (values are in [0, 255] range)
        if hasattr(self.model_cfg.MODEL, "IMAGE_MEAN"):
            mean = torch.tensor(self.model_cfg.MODEL.IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1) * 255.0
        else:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1) * 255.0

        if hasattr(self.model_cfg.MODEL, "IMAGE_STD"):
            std = torch.tensor(self.model_cfg.MODEL.IMAGE_STD, dtype=torch.float32).view(3, 1, 1) * 255.0
        else:
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1) * 255.0

        # Normalize: (img - mean) / std
        img_tensor = (img_tensor - mean) / std

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

        # Move to device
        img_tensor = img_tensor.to(self.device)

        # Create batch dict (HaMeR expects this format)
        batch = {
            "img": img_tensor,
            "right": torch.ones(1, device=self.device, dtype=torch.float32),  # Assume right hand by default
        }

        return batch

    def _extract_mano_data(
        self, pred: Any, rgb_image: np.ndarray, camera_matrix: np.ndarray | None, h: int, w: int  # noqa: ANN401
    ) -> tuple[np.ndarray | None, Any | None, np.ndarray | None, Handedness, float]:
        """Extract MANO data from prediction."""
        # Extract vertices
        vertices = None
        if isinstance(pred, dict):
            vertices = pred.get("vertices") or pred.get("verts")
            if vertices is not None and torch.is_tensor(vertices):
                vertices = vertices.cpu().numpy()
                if len(vertices.shape) > 2:
                    vertices = vertices[0]
        elif hasattr(pred, "vertices"):
            vertices = pred.vertices
            if torch.is_tensor(vertices):
                vertices = vertices.cpu().numpy()
                if len(vertices.shape) > 2:
                    vertices = vertices[0]
        elif hasattr(pred, "verts"):
            vertices = pred.verts
            if torch.is_tensor(vertices):
                vertices = vertices.cpu().numpy()
                if len(vertices.shape) > 2:
                    vertices = vertices[0]

        # Get MANO parameters
        mano_params = None
        if isinstance(pred, dict):
            mano_params = pred.get("mano")
        elif hasattr(pred, "mano"):
            mano_params = pred.mano

        # Extract 2D keypoints
        keypoints_2d = None
        if isinstance(pred, dict):
            keypoints_2d = pred.get("keypoints_2d") or pred.get("kp_2d")
            if keypoints_2d is not None and torch.is_tensor(keypoints_2d):
                keypoints_2d = keypoints_2d.cpu().numpy()
                if len(keypoints_2d.shape) > 2:
                    keypoints_2d = keypoints_2d[0]
        elif hasattr(pred, "keypoints_2d"):
            keypoints_2d = pred.keypoints_2d
            if torch.is_tensor(keypoints_2d):
                keypoints_2d = keypoints_2d.cpu().numpy()
                if len(keypoints_2d.shape) > 2:
                    keypoints_2d = keypoints_2d[0]
        elif hasattr(pred, "kp_2d"):
            keypoints_2d = pred.kp_2d
            if torch.is_tensor(keypoints_2d):
                keypoints_2d = keypoints_2d.cpu().numpy()
                if len(keypoints_2d.shape) > 2:
                    keypoints_2d = keypoints_2d[0]

        if keypoints_2d is None and vertices is not None:
            if camera_matrix is not None:
                keypoints_2d = self._project_3d_to_2d(vertices, camera_matrix)
            else:
                keypoints_2d = vertices[:, :2]

        if keypoints_2d is not None and keypoints_2d.max() <= 1.0:
            keypoints_2d[:, 0] *= w
            keypoints_2d[:, 1] *= h

        # Determine handedness
        handedness: Handedness = "Right"
        if isinstance(pred, dict):
            handedness_val = pred.get("handedness") or pred.get("is_right")
            if handedness_val is not None:
                if isinstance(handedness_val, bool):
                    handedness = "Right" if handedness_val else "Left"
                elif isinstance(handedness_val, (int, float)):
                    handedness = "Right" if handedness_val > 0.5 else "Left"
        elif hasattr(pred, "handedness"):
            handedness_val = pred.handedness
            if torch.is_tensor(handedness_val):
                handedness_val = handedness_val.cpu().numpy()
            if isinstance(handedness_val, (int, float, np.ndarray)):
                handedness = "Right" if (float(handedness_val) > 0.5) else "Left"
        elif hasattr(pred, "is_right"):
            is_right = pred.is_right
            if torch.is_tensor(is_right):
                is_right = is_right.cpu().numpy()
            if isinstance(is_right, bool):
                handedness = "Right" if is_right else "Left"
            elif isinstance(is_right, (int, float)):
                handedness = "Right" if is_right > 0.5 else "Left"

        # Get confidence
        confidence = 1.0
        if isinstance(pred, dict):
            conf_val = pred.get("conf") or pred.get("confidence")
            if conf_val is not None:
                if torch.is_tensor(conf_val):
                    confidence = float(conf_val.cpu().numpy())
                else:
                    confidence = float(conf_val)
        elif hasattr(pred, "conf"):
            conf_val = pred.conf
            if torch.is_tensor(conf_val):
                confidence = float(conf_val.cpu().numpy())
            else:
                confidence = float(conf_val)
        elif hasattr(pred, "confidence"):
            conf_val = pred.confidence
            if torch.is_tensor(conf_val):
                confidence = float(conf_val.cpu().numpy())
            else:
                confidence = float(conf_val)

        return vertices, mano_params, keypoints_2d, handedness, confidence

    def _compute_wrist_pose_mano16(self, joints_3d: np.ndarray, handedness: Handedness) -> np.ndarray:
        """Compute a stable wrist pose for MANO-style 16-joint arrays.

        Expected 16-joint layout (after our conversion):
          0 wrist
          1-3 thumb (mcp, ip, tip)
          4-6 index (mcp, pip, dip)
          7-9 middle (mcp, pip, dip)
          10-12 ring (mcp, pip, dip)
          13-15 pinky (mcp, pip, dip)

        This *must not* reuse BaseHandTracker._compute_wrist_pose(), which assumes
        MediaPipe/21-joint indexing.
        """
        wrist = joints_3d[0]
        index_mcp = joints_3d[4]
        middle_mcp = joints_3d[7]
        ring_mcp = joints_3d[10]
        pinky_mcp = joints_3d[13]

        palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4.0

        # x: wrist -> palm center
        x_axis = palm_center - wrist
        x_axis = x_axis / (np.linalg.norm(x_axis) + EPS)

        # z: index -> pinky (sideways across the palm)
        z_axis = pinky_mcp - index_mcp
        z_axis = z_axis / (np.linalg.norm(z_axis) + EPS)

        # Make left/right consistent (avoid mirrored frames)
        if handedness == "Left":
            z_axis = -z_axis

        # y: completes right-handed frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + EPS)

        # Re-orthonormalize
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + EPS)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + EPS)

        t = np.eye(4)
        t[:3, 0] = x_axis
        t[:3, 1] = y_axis
        t[:3, 2] = z_axis
        t[:3, 3] = wrist
        return t

    def _mano_to_hand_structure(
        self,
        joints_3d: np.ndarray | None,
        vertices: np.ndarray | None,
        keypoints_2d: np.ndarray | None,
        handedness: Handedness,
        confidence: float,
        timestamp: float,
        h: int,
        w: int,
    ) -> HandStructure | None:
        """Convert MANO joints directly to HandStructure.

        MANO 16 format: wrist(0), thumb(1-3), index(4-6), middle(7-9), ring(10-12), pinky(13-15)
        """
        if joints_3d is None or joints_3d.shape[0] < 16:
            return None

        # HaMeR/MANO outputs 21 joints (wrist + 4 per finger)
        # We need to convert to 16 joints for our structure (wrist + 3 per finger)
        if joints_3d.shape[0] == 21:
            # Extract: wrist(0), thumb(2-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
            # Convert to 16: wrist(0), thumb(2-4), index(5-7), middle(9-11), ring(13-15), pinky(17-19)
            joints_3d = np.array(
                [
                    joints_3d[0],  # Wrist
                    joints_3d[2],  # Thumb MCP
                    joints_3d[3],  # Thumb IP
                    joints_3d[4],  # Thumb tip
                    joints_3d[5],  # Index MCP
                    joints_3d[6],  # Index PIP
                    joints_3d[7],  # Index DIP (use DIP, not tip)
                    joints_3d[9],  # Middle MCP
                    joints_3d[10],  # Middle PIP
                    joints_3d[11],  # Middle DIP
                    joints_3d[13],  # Ring MCP
                    joints_3d[14],  # Ring PIP
                    joints_3d[15],  # Ring DIP
                    joints_3d[17],  # Pinky MCP
                    joints_3d[18],  # Pinky PIP
                    joints_3d[19],  # Pinky DIP
                ]
            )
        elif joints_3d.shape[0] != 16:
            return None

        # Apply smoothing
        joints_3d = self._apply_smoothing(joints_3d, handedness)

        # Compute wrist pose (need at least wrist + a few joints)
        wrist_pose = self._compute_wrist_pose_mano16(joints_3d, handedness)

        # Transform to hand frame
        wrist_pose_inv = np.linalg.inv(wrist_pose)
        joints_homogeneous = np.hstack([joints_3d, np.ones((len(joints_3d), 1))])
        joints_hand_frame = (wrist_pose_inv @ joints_homogeneous.T).T[:, :3]

        # Extract joints directly from MANO structure
        # MANO 16: wrist(0), thumb(1-3), index(4-6), middle(7-9), ring(10-12), pinky(13-15)
        wrist = joints_hand_frame[0]

        # Thumb: joints 1-3 (MCP, IP, tip)
        thumb = FingerJoints(
            mcp=joints_hand_frame[1],
            ip=joints_hand_frame[2],
            tip=joints_hand_frame[3],
            mcp_local=joints_hand_frame[1] - wrist,
            ip_local=joints_hand_frame[2] - joints_hand_frame[1],
            tip_local=joints_hand_frame[3] - joints_hand_frame[2],
        )

        # Index: joints 4-6 (MCP, PIP, DIP/tip)
        index = FingerJoints(
            mcp=joints_hand_frame[4],
            pip=joints_hand_frame[5],
            dip=joints_hand_frame[6],
            tip=joints_hand_frame[6],  # MANO doesn't separate DIP and tip
            mcp_local=joints_hand_frame[4] - wrist,
            pip_local=joints_hand_frame[5] - joints_hand_frame[4],
            dip_local=joints_hand_frame[6] - joints_hand_frame[5],
            tip_local=joints_hand_frame[6] - joints_hand_frame[5],  # Same as DIP
        )

        # Middle: joints 7-9
        middle = FingerJoints(
            mcp=joints_hand_frame[7],
            pip=joints_hand_frame[8],
            dip=joints_hand_frame[9],
            tip=joints_hand_frame[9],
            mcp_local=joints_hand_frame[7] - wrist,
            pip_local=joints_hand_frame[8] - joints_hand_frame[7],
            dip_local=joints_hand_frame[9] - joints_hand_frame[8],
            tip_local=joints_hand_frame[9] - joints_hand_frame[8],
        )

        # Ring: joints 10-12
        ring = FingerJoints(
            mcp=joints_hand_frame[10],
            pip=joints_hand_frame[11],
            dip=joints_hand_frame[12],
            tip=joints_hand_frame[12],
            mcp_local=joints_hand_frame[10] - wrist,
            pip_local=joints_hand_frame[11] - joints_hand_frame[10],
            dip_local=joints_hand_frame[12] - joints_hand_frame[11],
            tip_local=joints_hand_frame[12] - joints_hand_frame[11],
        )

        # Pinky: joints 13-15
        pinky = FingerJoints(
            mcp=joints_hand_frame[13],
            pip=joints_hand_frame[14],
            dip=joints_hand_frame[15],
            tip=joints_hand_frame[15],
            mcp_local=joints_hand_frame[13] - wrist,
            pip_local=joints_hand_frame[14] - joints_hand_frame[13],
            dip_local=joints_hand_frame[15] - joints_hand_frame[14],
            tip_local=joints_hand_frame[15] - joints_hand_frame[14],
        )

        return HandStructure(
            wrist_pose=wrist_pose,
            wrist_position=wrist,
            thumb=thumb,
            index=index,
            middle=middle,
            ring=ring,
            pinky=pinky,
            handedness=handedness,
            confidence=confidence,
            timestamp=timestamp,
        )

    def _extract_mano_joints(self, vertices: np.ndarray, mano_params: Any | None = None) -> np.ndarray:  # noqa: ANN401
        """Extract joint positions from MANO vertices using joint regressor.

        Note: This is a fallback method. In detect_hands(), we use pred_keypoints_3d
        directly from the model output, which already contains the joints.
        """
        # Try to use MANO joint regressor if available
        if mano_params is not None:
            # mano_params might be a dict or an object with J_regressor
            if isinstance(mano_params, dict):
                j_regressor = None
                # Try to get J_regressor from MANO model if available
                if hasattr(self.model, "mano") and hasattr(self.model.mano, "J_regressor"):
                    j_regressor = self.model.mano.J_regressor
            elif hasattr(mano_params, "J_regressor"):
                j_regressor = mano_params.J_regressor
            else:
                j_regressor = None

            if j_regressor is not None:
                # Use MANO's joint regressor to get joints from vertices
                if torch.is_tensor(vertices):
                    vertices_tensor = vertices
                else:
                    vertices_tensor = torch.from_numpy(vertices).float()

                if len(vertices_tensor.shape) == 2:
                    vertices_tensor = vertices_tensor.unsqueeze(0)  # type: ignore[attr-defined]  # Add batch dim

                if torch.is_tensor(j_regressor):
                    joints = torch.matmul(j_regressor, vertices_tensor)
                    joints = joints.squeeze(0).cpu().numpy()
                else:
                    joints = np.matmul(j_regressor, vertices_tensor.cpu().numpy())  # type: ignore[attr-defined]
                    if len(joints.shape) > 2:
                        joints = joints[0]

                # MANO typically outputs 16 joints (wrist + 3 per finger)
                # Some models may output 21 joints - we'll handle both in _mano_to_hand_structure
                if joints.shape[0] == 16 or joints.shape[0] == 21:
                    return joints

        # Fallback: extract joints from vertices using known MANO structure
        return self._extract_joints_from_vertices(vertices)

    def _extract_joints_from_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Extract 21 joints from MANO vertices using approximate vertex indices."""
        # MANO has 778 vertices with known structure
        # These are approximate indices - actual implementation should use joint regressor
        if vertices.shape[0] >= 778:
            # Known MANO vertex indices for key joints (approximate)
            # These indices are based on MANO's vertex ordering
            joint_vertex_indices = [
                0,  # Wrist (root)
                744,  # Thumb CMC
                728,  # Thumb MCP
                712,  # Thumb IP
                696,  # Thumb tip
                316,  # Index MCP
                333,  # Index PIP
                350,  # Index DIP
                367,  # Index tip
                412,  # Middle MCP
                429,  # Middle PIP
                446,  # Middle DIP
                463,  # Middle tip
                508,  # Ring MCP
                525,  # Ring PIP
                542,  # Ring DIP
                559,  # Ring tip
                604,  # Pinky MCP
                621,  # Pinky PIP
                638,  # Pinky DIP
                655,  # Pinky tip
            ]
            # Clamp indices to valid range
            joint_vertex_indices = [min(i, vertices.shape[0] - 1) for i in joint_vertex_indices]
            return vertices[joint_vertex_indices]
        else:
            # Fallback: sample evenly if not enough vertices
            indices = np.linspace(0, vertices.shape[0] - 1, 21, dtype=int)
            return vertices[indices]

    def _project_3d_to_2d(self, vertices_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Project 3D vertices to 2D using camera matrix."""
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Project
        x_2d = (vertices_3d[:, 0] * fx / vertices_3d[:, 2]) + cx
        y_2d = (vertices_3d[:, 1] * fy / vertices_3d[:, 2]) + cy

        return np.stack([x_2d, y_2d], axis=1)

    def visualize(
        self,
        image: np.ndarray,
        hand_structures: list[HandStructure],
        camera_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Visualize HaMeR hand detections.

        For now, uses simple joint projection. In the future, could use
        hamer.utils.renderer.Renderer for mesh visualization.
        """
        return self._simple_visualize(image, hand_structures, camera_matrix)

    def _simple_visualize(
        self,
        image: np.ndarray,
        hand_structures: list[HandStructure],
        camera_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simple fallback visualization using joint projection."""
        h, w = image.shape[:2]
        annotated_image = image.copy()

        for structure in hand_structures:
            joints_3d_camera = self._structure_to_camera_frame(structure)
            if camera_matrix is not None:
                joints_2d = self._project_joints_3d_to_2d(joints_3d_camera, camera_matrix)
            else:
                joints_2d = np.zeros((21, 2))
                for i, joint_3d in enumerate(joints_3d_camera):
                    if joint_3d[2] > 0:
                        joints_2d[i] = [
                            joint_3d[0] / joint_3d[2] * w / 2 + w / 2,
                            joint_3d[1] / joint_3d[2] * h / 2 + h / 2,
                        ]

            connections = [
                (0, 2),
                (2, 3),
                (3, 4),
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),
            ]

            for start_idx, end_idx in connections:
                if start_idx < len(joints_2d) and end_idx < len(joints_2d):
                    start_point = tuple(joints_2d[start_idx].astype(int))
                    end_point = tuple(joints_2d[end_idx].astype(int))
                    cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

            for joint_2d in joints_2d:
                cv2.circle(annotated_image, (int(joint_2d[0]), int(joint_2d[1])), 4, (0, 0, 255), -1)

            if len(joints_2d) > 0:
                wrist_2d = joints_2d[0].astype(int)
                label = f"{structure.handedness} ({structure.confidence:.2f})"
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

    def _structure_to_camera_frame(self, structure: HandStructure) -> np.ndarray:
        """Convert HandStructure joints to camera frame."""
        joints_hand_frame = np.array(
            [
                structure.wrist_position,
                structure.thumb.mcp,
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

        joints_homogeneous = np.hstack([joints_hand_frame, np.ones((21, 1))])
        joints_camera = (structure.wrist_pose @ joints_homogeneous.T).T[:, :3]
        return joints_camera

    def _project_joints_3d_to_2d(self, joints_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Project 3D joints to 2D using camera matrix."""
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        x_2d = (joints_3d[:, 0] * fx / joints_3d[:, 2]) + cx
        y_2d = (joints_3d[:, 1] * fy / joints_3d[:, 2]) + cy

        return np.stack([x_2d, y_2d], axis=1)
