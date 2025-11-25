"""Live IK Retargeting with ROS2 Joint State Publisher.

Publishes retargeted joint angles to /joint_states for ORCA hand control.
"""

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import rclpy
from askin import KeyboardController
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from handpose import HandTracker, ORCAHandIKRetargeting
from handpose.ik_retargeting import ORCAHandIKConfig


class JointStatePublisher(Node):
    """Publishes MuJoCo qpos → /joint_states at given rate."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, rate_hz: float = 100.0):
        super().__init__("handpose_joint_state_pub")
        self.model = model
        self.data = data

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(JointState, "/joint_states", qos)

        # Collect joint names and addresses (skip free joints)
        self.joint_names = []
        self.joint_addrs = []
        for j_id in range(model.njnt):
            if model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if name:  # Only add if name exists
                self.joint_names.append(name)
                self.joint_addrs.append(model.jnt_qposadr[j_id])

        self.create_timer(1.0 / rate_hz, self._publish)

    def _publish(self):
        """Publish current joint states."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(self.data.qpos[addr]) for addr in self.joint_addrs]
        self.pub.publish(msg)


async def main_async(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cap: cv2.VideoCapture,
    tracker: HandTracker,
    ik_solver: ORCAHandIKRetargeting,
    joint_pub: JointStatePublisher,
) -> None:
    """Async main loop with keyboard handling."""
    running = True

    async def key_handler(key: str) -> None:
        nonlocal running
        if key == "q":
            running = False
            print("[Main] Quitting...")

    # Initialize keyboard controller
    controller = KeyboardController(key_handler=key_handler, timeout=0.01)
    await controller.start()

    start_time = time.time()

    print("\nStarting hand tracking and ROS2 publishing...")
    print("Press 'q' to quit.")

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time() - start_time
        hand_poses = tracker.detect_hands(frame, timestamp=timestamp)

        if hand_poses:
            pose = hand_poses[0]
            landmarks_hand = tracker.landmarks_in_hand_frame(pose)

            # Update configuration with current robot state
            mujoco.mj_forward(model, data)
            ik_solver.configuration.update(data.qpos)

            # Solve IK for new joint angles
            target_q = ik_solver.solve(landmarks_hand)

            # Apply to simulation
            data.qpos[:] = target_q

            # Joint states are published automatically by the publisher timer

        # Step physics
        mujoco.mj_step(model, data)

        # Small sleep to prevent busy-waiting
        await asyncio.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand tracking with ROS2 joint state publishing")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", type=str, default="orca_hand.mjcf", help="MuJoCo model file")
    parser.add_argument("--scale", type=float, default=1.0, help="Robot/Human hand scale factor")
    parser.add_argument("--rate", type=float, default=100.0, help="Joint state publish rate (Hz)")
    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init(args=sys.argv[1:])

    # Setup paths and model
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "models" / args.model if not Path(args.model).is_absolute() else Path(args.model)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        rclpy.shutdown()
        return

    # Load MuJoCo Model
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Initialize Tracker and IK
    print("Initializing Hand Tracker...")
    tracker = HandTracker(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    print("Initializing IK Solver...")
    ik_config = ORCAHandIKConfig(
        scale_factor=args.scale,
        wrist_offset_palm=np.array([0.000, 0.0, -0.05]),
    )
    ik_solver = ORCAHandIKRetargeting(model, config=ik_config)

    # Setup camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create ROS2 joint state publisher
    print(f"Creating ROS2 joint state publisher (rate: {args.rate} Hz)...")
    joint_pub = JointStatePublisher(model, data, rate_hz=args.rate)

    # Start ROS2 executor in background thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(joint_pub)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        # Run async main loop
        asyncio.run(main_async(model, data, cap, tracker, ik_solver, joint_pub))
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()

