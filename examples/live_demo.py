#!/usr/bin/env python3
"""Live hand tracking and retargeting demo.

Real-time hand pose tracking from camera with MuJoCo visualization of the ORCA hand.
This demo tracks finger joints only (no root position/rotation).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import threading
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from askin import KeyboardController

from handpose import HandTracker, ORCAHandRetargeting


class LiveRetargetingDemo:
    """Live hand retargeting demo with OpenCV camera feed and MuJoCo visualization."""

    def __init__(self, camera_id: int = 0, model_path: str = "models/orca_hand.mjcf") -> None:
        """Initialize the demo.

        Args:
            camera_id: Camera device ID
            model_path: Path to MuJoCo ORCA hand model
        """
        # Initialize hand tracker
        print("Initializing hand tracker...")
        self.hand_tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Initialize retargeting
        print("Initializing retargeting...")
        self.retargeting = ORCAHandRetargeting()

        # Load MuJoCo model
        print(f"Loading MuJoCo model from {model_path}...")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Setup joint mapping
        self.mujoco_joint_qpos_addrs, self.retargeted_joint_names = self._setup_joint_mapping()

        # Open camera
        print(f"Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # State
        self.running = True
        self.current_joint_angles: np.ndarray | None = None
        self.lock = threading.Lock()

    def _setup_joint_mapping(self) -> tuple[np.ndarray, list[str]]:
        """Setup mapping from retargeted joint names to MuJoCo qpos addresses."""
        # Mapping from our retargeted names to ORCA hand joint names
        joint_mapping = {
            # Thumb
            "thumb_cmc_flex": "right_thumb_mcp",
            "thumb_cmc_abd": "right_thumb_abd",
            "thumb_mcp": "right_thumb_pip",
            "thumb_ip": "right_thumb_dip",
            # Index
            "index_mcp_abd": "right_index_abd",
            "index_mcp_flex": "right_index_mcp",
            "index_pip": "right_index_pip",
            # Middle
            "middle_mcp_abd": "right_middle_abd",
            "middle_mcp_flex": "right_middle_mcp",
            "middle_pip": "right_middle_pip",
            # Ring
            "ring_mcp_abd": "right_ring_abd",
            "ring_mcp_flex": "right_ring_mcp",
            "ring_pip": "right_ring_pip",
            # Pinky
            "pinky_mcp_abd": "right_pinky_abd",
            "pinky_mcp_flex": "right_pinky_mcp",
            "pinky_pip": "right_pinky_pip",
        }

        # Get retargeted joint names
        dummy_landmarks = np.zeros((21, 3))
        dummy_angles = self.retargeting.retarget_pose(dummy_landmarks, "Right")
        all_retargeted_names = sorted(dummy_angles.keys())

        # Map to MuJoCo qpos addresses
        qpos_addrs = []
        mapped_names = []

        for retargeted_name in all_retargeted_names:
            orca_name = joint_mapping.get(retargeted_name)
            if orca_name:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, orca_name)
                if joint_id >= 0:
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    qpos_addrs.append(qpos_adr)
                    mapped_names.append(retargeted_name)

        print(f"✓ Mapped {len(qpos_addrs)} joints to MuJoCo model")
        return np.array(qpos_addrs), mapped_names

    def update_mujoco(self) -> None:
        """Update MuJoCo visualization in a separate thread."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("MuJoCo viewer started")
            while self.running and viewer.is_running():
                with self.lock:
                    if self.current_joint_angles is not None:
                        # Set finger joint positions
                        self.data.qpos[self.mujoco_joint_qpos_addrs] = self.current_joint_angles

                # Step simulation
                mujoco.mj_forward(self.model, self.data)

                # Sync viewer
                viewer.sync()

                # Small sleep
                time.sleep(0.005)

    async def key_handler(self, key: str) -> None:
        """Handle keyboard input."""
        if key == "q":
            self.running = False
            print("\nQuitting...")

    async def run(self) -> None:
        """Main loop."""
        print("\n=== Controls ===")
        print("'q' - Quit")
        print("================\n")

        # Initialize keyboard controller
        controller = KeyboardController(key_handler=self.key_handler, timeout=0.01)
        await controller.start()

        # Start MuJoCo viewer in separate thread
        mujoco_thread = threading.Thread(target=self.update_mujoco, daemon=True)
        mujoco_thread.start()

        # Give MuJoCo time to start
        await asyncio.sleep(1.0)

        frame_count = 0
        fps_start_time = time.time()
        fps: float = 0.0

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                timestamp = time.time()

                # Detect hands
                hand_poses = self.hand_tracker.detect_hands(frame, timestamp=timestamp)

                # Process first detected hand
                if hand_poses:
                    hand_pose = hand_poses[0]

                    # Get landmarks in hand frame
                    landmarks_hand_frame = self.hand_tracker.landmarks_in_hand_frame(hand_pose)

                    # Retarget to ORCA hand
                    joint_angles_dict = self.retargeting.retarget_pose(landmarks_hand_frame, hand_pose.handedness)

                    # Apply offsets to match ORCA's reference positions
                    # Transform: orca_angle = retargeted_angle + orca_ref
                    orca_ref_offsets = {
                        # Thumb (Keep these! Thumb calculation is still based on static bones)
                        "thumb_cmc_flex": 0.0,      # right_thumb_mcp ref=0.0000
                        "thumb_cmc_abd": -0.7330,   # right_thumb_abd ref=-0.7330
                        "thumb_mcp": -0.5850,        # right_thumb_pip ref=-0.5850
                        "thumb_ip": -0.5048,         # right_thumb_dip ref=-0.5048
                        # Fingers (CHANGE THESE TO 0.0)
                        # The new dynamic calculation (pip-mcp) naturally finds the zero/neutral point,
                        # so we no longer need these calibration offsets.
                        "index_mcp_abd": -0.4/2,       # WAS -0.4000
                        "index_mcp_flex": 0.0,      # right_index_mcp ref=0.0000
                        "index_pip": 0.0,          # right_index_pip ref=0.0000
                        "middle_mcp_abd": 0.0,      # WAS 0.0 (No change)
                        "middle_mcp_flex": 0.0,     # right_middle_mcp ref=0.0000
                        "middle_pip": 0.0,          # right_middle_pip ref=0.0000
                        "ring_mcp_abd": 0.17/2,        # WAS 0.1700
                        "ring_mcp_flex": 0.0,       # right_ring_mcp ref=0.0000
                        "ring_pip": 0.0,            # right_ring_pip ref=0.0000
                        "pinky_mcp_abd": 0.5233/2,       # WAS 0.5233
                        "pinky_mcp_flex": 0.0,      # right_pinky_mcp ref=0.0000
                        "pinky_pip": 0.0,           # right_pinky_pip ref=0.0000
                    }

                    # Convert to array in correct order
                    joint_angles_array = np.zeros(len(self.mujoco_joint_qpos_addrs))
                    for i, name in enumerate(self.retargeted_joint_names):
                        if name in joint_angles_dict:
                            angle = joint_angles_dict[name]
                            # Apply offset if available
                            offset = orca_ref_offsets.get(name, 0.0)
                            joint_angles_array[i] = angle + offset

                    # Update MuJoCo state
                    with self.lock:
                        self.current_joint_angles = joint_angles_array

                    # Draw hand landmarks
                    frame = self.hand_tracker.draw_landmarks(frame, hand_poses)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30.0 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time

                # Display info
                info_text = f"FPS: {fps:.1f} | Hands: {len(hand_poses)}"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show frame (skip on macOS with mjpython due to GLFW conflicts)
                try:
                    cv2.imshow("Hand Tracking", frame)
                    cv2.waitKey(1)  # Just to keep window responsive
                except (cv2.error, Exception):
                    # OpenCV window not available (macOS + mjpython)
                    # Just print FPS to console occasionally
                    if frame_count % 30 == 0:
                        print(f"FPS: {fps:.1f} | Hands: {len(hand_poses)}")

                await asyncio.sleep(0.001)

        finally:
            # Cleanup
            await controller.stop()
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nDemo finished.")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live hand tracking and retargeting demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--model", type=str, default="models/orca_hand.mjcf", help="Path to MuJoCo model")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    # Create and run demo
    demo = LiveRetargetingDemo(camera_id=args.camera, model_path=str(model_path))
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
