"""ROS2 publisher for HandPose tracking and IK retargeting.

This script captures video from a camera, runs HaMeR hand tracking,
performs IK retargeting to ORCA hand joints, and publishes JointState
messages on /joint_states for use with the tactile_dex simulation pipeline.

IMPORTANT: This publisher outputs raw IK joint angles WITHOUT applying
REF_OFFSETS. The tactile_dex simulation applies offsets internally.
"""

import argparse
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

# Remove system ROS2 paths from sys.path to prefer conda-installed ROS2 packages
# This prevents conflicts when system ROS2 (Python 3.10) is incompatible with conda env (Python 3.11)
sys.path = [
    p for p in sys.path
    if not (p.startswith("/opt/ros/") and "python3.10" in p)
]

import cv2
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from handpose.ik_retargeting import (
    FINGER_TARGET_BODIES,
    ORCA_JOINT_NAMES,
    ORCAHandIKConfig,
    ORCAHandIKRetargeting,
)
from handpose.tracker.hamer import HaMeRTracker


def inject_target_bodies(mjcf_path: Path) -> str:
    """Injects mocap bodies and tip sites into the MJCF XML string for IK targeting.
    
    This matches the approach in live_demo_ik.py to ensure tip sites exist for all fingers.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if worldbody is None:
        raise ValueError("Could not find worldbody in MJCF")

    # Convert relative paths to absolute paths
    # MuJoCo can't resolve relative paths when loading from string
    model_dir = mjcf_path.parent
    for asset in root.findall(".//asset"):
        for mesh in asset.findall("mesh"):
            file_attr = mesh.get("file")
            if file_attr and not Path(file_attr).is_absolute():
                # Convert relative path to absolute
                abs_path = (model_dir / file_attr).resolve()
                mesh.set("file", str(abs_path))

    # Add tip sites for all fingers (required for tip tracking)
    tip_site_specs = {
        "thumb": ("right_thumb_dp", np.array([0.0, 0.0, 0.018])),
        "index": ("right_index_ip", np.array([0.0, 0.0, 0.020])),
        "middle": ("right_middle_ip", np.array([0.0, 0.0, 0.022])),
        "ring": ("right_ring_ip", np.array([0.0, 0.0, 0.021])),
        "pinky": ("right_pinky_ip", np.array([0.0, 0.0, 0.018])),
    }

    for finger, (parent_body, offset) in tip_site_specs.items():
        body_elem = root.find(f".//body[@name='{parent_body}']")
        if body_elem is None:
            continue
        site_name = f"right_{finger}_tip_site"
        if body_elem.find(f"./site[@name='{site_name}']") is not None:
            continue
        site = ET.SubElement(body_elem, "site")
        site.set("name", site_name)
        site.set("pos", " ".join(f"{value:.5f}" for value in offset))
        site.set("size", "0.0025")
        site.set("rgba", "1 0.6 0.2 0.8")

    return ET.tostring(root, encoding="unicode")


class HandPoseROS2Publisher(Node):
    """ROS2 node that publishes hand joint states from camera tracking."""

    def __init__(
        self,
        camera: int = 0,
        fps: int = 30,
        model_path: str = "models/orca_hand_fixed.mjcf",
        scale: float = 1.0,
        viz: bool = False,
        smoothing: float = 0.0,
        conf_threshold: float = 0.3,
        hold_last: bool = False,
        width: int = 1280,
        height: int = 720,
        joint_smoothing: float = 1.0,
        target_joint_types: tuple[str, ...] = ("tip", "ip"),
    ) -> None:
        """Initialize the HandPose ROS2 publisher.

        Args:
            camera: Camera device index
            fps: Target publishing rate (Hz)
            model_path: Path to MuJoCo MJCF model file
            scale: Scale factor for hand size (robot/human ratio)
            viz: Enable OpenCV visualization window
            smoothing: Exponential smoothing factor (0.0 = no smoothing)
            conf_threshold: Confidence threshold for hand detection
            hold_last: If True, publish last valid joint values when hand is not detected
            width: Camera frame width
            height: Camera frame height
            joint_smoothing: Smoothing factor for joint positions (0.0-1.0). Default: 1.0 (no smoothing)
            target_joint_types: Tuple of joint types to target for IK (tip, ip, pip, mcp). Default: ("tip", "ip")
        """
        super().__init__("handpose_hamer_publisher")

        # ROS2 publisher
        self.pub = self.create_publisher(JointState, "/joint_states", qos_profile_sensor_data)

        # Camera setup
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera opened: {actual_width}x{actual_height}")

        # Hand tracker
        self.tracker = HaMeRTracker(smoothing_factor=smoothing, conf_threshold=conf_threshold)

        # Load MuJoCo model
        model_file = Path(model_path)
        if not model_file.is_absolute():
            # Resolve relative to HandPose root
            model_file = Path(__file__).parent.parent / model_path
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.get_logger().info(f"Loading MuJoCo model from {model_file}")
        # Inject tip sites into MJCF (required for tip tracking, matches live_demo_ik.py)
        xml_string = inject_target_bodies(model_file)
        self.model = mujoco.MjModel.from_xml_string(xml_string)

        # IK retargeting
        # Use the same wrist_offset_palm as live_demo_ik.py to match USD model coordinate frame
        # The default MJCF offset [0.002, -0.00144, -0.03872] may not match the USD model
        ik_cfg = ORCAHandIKConfig(
            scale_factor=scale,
            target_joint_types=target_joint_types,
            wrist_offset_palm=np.array([0.000, 0.0, -0.05]),  # Match live_demo_ik.py
        )
        self.ik = ORCAHandIKRetargeting(self.model, config=ik_cfg)

        # Initialize joint position smoothing state and MuJoCo data for forward kinematics
        self.data = mujoco.MjData(self.model)
        self.smoothed_qpos = self.data.qpos.copy()
        self.joint_smoothing = joint_smoothing

        # State tracking
        self.last_joint_vals = None
        self.hold_last = hold_last
        self.viz = viz

        # Statistics
        self.frame_count = 0
        self.hand_detected_count = 0
        self.last_stats_time = time.time()
        self.publish_count = 0

        # Timer for publishing
        self.dt = 1.0 / fps
        self.timer = self.create_timer(self.dt, self.tick)

        self.get_logger().info(
            f"HandPose ROS2 Publisher initialized: "
            f"camera={camera}, fps={fps}, model={model_file.name}, scale={scale}"
        )

    def tick(self) -> None:
        """Timer callback: capture frame, track hand, publish joint states."""
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Failed to read camera frame")
            return

        # Track hands
        timestamp = time.time()
        hands = self.tracker.detect_hands(frame, timestamp=timestamp)

        # Select best hand (prefer Right, otherwise highest confidence)
        hand = None
        if hands:
            right_hands = [h for h in hands if h.handedness == "Right"]
            if right_hands:
                hand = max(right_hands, key=lambda h: h.confidence)
            else:
                hand = max(hands, key=lambda h: h.confidence)

        # Solve IK if hand detected
        joint_vals = None
        if hand is not None:
            try:
                # Update IK solver configuration with current robot state (required for proper IK solving)
                # This matches the approach in live_demo_ik.py
                self.data.qpos[:] = self.smoothed_qpos
                mujoco.mj_forward(self.model, self.data)
                self.ik.configuration.update(self.data.qpos)
                
                # Solve IK
                qpos = self.ik.solve(hand)
                
                # Apply joint position smoothing (always applied; joint_smoothing=1.0 means no smoothing)
                if not np.any(np.isnan(qpos)) and not np.any(np.isinf(qpos)):
                    # Exponential moving average: smoothed = joint_smoothing * new + (1 - joint_smoothing) * old
                    self.smoothed_qpos = self.joint_smoothing * qpos + (1.0 - self.joint_smoothing) * self.smoothed_qpos
                    qpos = self.smoothed_qpos
                joint_vals = qpos[self.ik.joint_indices]  # Extract only ORCA joints
                self.last_joint_vals = joint_vals.copy()
                self.hand_detected_count += 1
            except Exception as e:
                self.get_logger().warn(f"IK solve failed: {e}", throttle_duration_sec=1.0)
        elif self.hold_last and self.last_joint_vals is not None:
            # Use last valid values
            joint_vals = self.last_joint_vals.copy()

        # Publish joint state
        if joint_vals is not None:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(ORCA_JOINT_NAMES)
            msg.position = [float(x) for x in joint_vals]
            self.pub.publish(msg)
            self.publish_count += 1

        # Visualization
        if self.viz and hand is not None:
            vis_frame = self.tracker.visualize(frame, [hand])
            cv2.imshow("HandPose Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("Visualization window closed by user")
                self.viz = False
                cv2.destroyAllWindows()

        # Statistics logging (every 1 second)
        self.frame_count += 1
        now = time.time()
        if now - self.last_stats_time >= 1.0:
            fps_actual = self.frame_count / (now - self.last_stats_time)
            pub_rate = self.publish_count / (now - self.last_stats_time)
            detection_rate = self.hand_detected_count / self.frame_count if self.frame_count > 0 else 0.0

            status = "OK" if hand is not None else "NO HAND"
            handedness = hand.handedness if hand is not None else "N/A"
            conf = hand.confidence if hand is not None else 0.0

            if joint_vals is not None:
                joint_min = float(np.min(joint_vals))
                joint_max = float(np.max(joint_vals))
                joint_info = f"joints: [{joint_min:.3f}, {joint_max:.3f}]"
            else:
                joint_info = "joints: N/A"

            self.get_logger().info(
                f"[stats] {status} | "
                f"fps: {fps_actual:.1f} | "
                f"pub_rate: {pub_rate:.1f} Hz | "
                f"detection: {detection_rate:.1%} | "
                f"hand: {handedness} ({conf:.2f}) | "
                f"{joint_info}"
            )

            # Reset counters
            self.frame_count = 0
            self.hand_detected_count = 0
            self.publish_count = 0
            self.last_stats_time = now

    def destroy_node(self) -> None:
        """Cleanup on shutdown."""
        self.get_logger().info("Shutting down HandPose publisher...")
        if self.cap.isOpened():
            self.cap.release()
        if self.viz:
            cv2.destroyAllWindows()
        super().destroy_node()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ROS2 publisher for HandPose tracking and IK retargeting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=30, help="Target publishing rate (Hz)")
    parser.add_argument(
        "--model",
        type=str,
        default="models/orca_hand_fixed.mjcf",
        help="Path to MuJoCo MJCF model file",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Hand scale factor (robot/human)")
    parser.add_argument("--viz", action="store_true", help="Enable OpenCV visualization window")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Exponential smoothing factor (0.0 = no smoothing)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for hand detection",
    )
    parser.add_argument(
        "--hold-last",
        action="store_true",
        help="Publish last valid joint values when hand is not detected",
    )
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width")
    parser.add_argument("--height", type=int, default=720, help="Camera frame height")
    parser.add_argument(
        "--joint-smoothing",
        type=float,
        default=1.0,
        help="Smoothing factor for joint positions (0.0-1.0). Lower values = more smoothing, higher = less smoothing. Default: 1.0 (no smoothing)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="tip, ip",
        help="Comma-separated joint targets (tip,ip,pip,mcp). Default: tip, ip",
    )

    args = parser.parse_args()

    # Parse and validate target joint types
    raw_targets = [part.strip().lower() for part in args.targets.split(",")]
    target_joints = tuple(dict.fromkeys(jt for jt in raw_targets if jt))
    if not target_joints:
        target_joints = ("tip",)
    supported_targets = {"tip", "ip", "pip", "mcp"}
    invalid = [jt for jt in target_joints if jt not in supported_targets]
    if invalid:
        parser.error(f"Unsupported target joint types: {', '.join(invalid)}")

    # Validate joint smoothing value
    if not (0.0 < args.joint_smoothing <= 1.0):
        parser.error("--joint-smoothing must be between 0.0 and 1.0 (exclusive of 0.0)")

    # Initialize ROS2
    rclpy.init()

    try:
        node = HandPoseROS2Publisher(
            camera=args.camera,
            fps=args.fps,
            model_path=args.model,
            scale=args.scale,
            viz=args.viz,
            smoothing=args.smoothing,
            conf_threshold=args.conf_threshold,
            hold_last=args.hold_last,
            width=args.width,
            height=args.height,
            joint_smoothing=args.joint_smoothing,
            target_joint_types=target_joints,
        )
        rclpy.spin(node)
        node.destroy_node()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
