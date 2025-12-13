"""Live hand tracking and retargeting demo.

Real-time hand pose tracking from camera with MuJoCo visualization of the ORCA hand.
This demo tracks finger joints only (no root position/rotation).
"""

import argparse
import asyncio
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Protocol, cast

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from askin import KeyboardController

from handpose import ORCAHandRetargeting
from handpose.tracker import HandStructure, MediaPipeTracker


class Flag(Protocol):
    value: int


def hand_structure_to_landmarks(structure: HandStructure) -> np.ndarray:
    """Convert HandStructure to 21x3 landmarks array for retargeting (MediaPipe format).

    Returns landmarks in hand frame (wrist at origin) in MediaPipe order.
    """
    landmarks = np.zeros((21, 3))

    # Wrist (index 0)
    landmarks[0] = structure.wrist_position

    # Thumb: CMC(1), MCP(2), IP(3), tip(4)
    landmarks[1] = structure.thumb.mcp  # CMC approximate
    landmarks[2] = structure.thumb.mcp
    landmarks[3] = structure.thumb.ip if structure.thumb.ip is not None else structure.thumb.mcp
    landmarks[4] = structure.thumb.tip

    # Index: MCP(5), PIP(6), DIP(7), tip(8)
    landmarks[5] = structure.index.mcp
    landmarks[6] = structure.index.pip if structure.index.pip is not None else structure.index.mcp
    landmarks[7] = structure.index.dip if structure.index.dip is not None else structure.index.tip
    landmarks[8] = structure.index.tip

    # Middle: MCP(9), PIP(10), DIP(11), tip(12)
    landmarks[9] = structure.middle.mcp
    landmarks[10] = structure.middle.pip if structure.middle.pip is not None else structure.middle.mcp
    landmarks[11] = structure.middle.dip if structure.middle.dip is not None else structure.middle.tip
    landmarks[12] = structure.middle.tip

    # Ring: MCP(13), PIP(14), DIP(15), tip(16)
    landmarks[13] = structure.ring.mcp
    landmarks[14] = structure.ring.pip if structure.ring.pip is not None else structure.ring.mcp
    landmarks[15] = structure.ring.dip if structure.ring.dip is not None else structure.ring.tip
    landmarks[16] = structure.ring.tip

    # Pinky: MCP(17), PIP(18), DIP(19), tip(20)
    landmarks[17] = structure.pinky.mcp
    landmarks[18] = structure.pinky.pip if structure.pinky.pip is not None else structure.pinky.mcp
    landmarks[19] = structure.pinky.dip if structure.pinky.dip is not None else structure.pinky.tip
    landmarks[20] = structure.pinky.tip

    return landmarks


def dual_window_process(frame_queue: mp.Queue, running_flag: Flag, window_name: str, scale: float) -> None:
    """Display frames in a separate process to avoid GLFW/OpenCV conflicts."""
    while True:
        if not running_flag.value and frame_queue.empty():
            break

        try:
            frame = frame_queue.get(timeout=0.05)
        except Empty:
            if not running_flag.value:
                break
            continue

        display_frame = frame
        if scale != 1.0:
            new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            display_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running_flag.value = 0
            break

    cv2.destroyAllWindows()


class LiveRetargetingDemo:
    """Live hand retargeting demo with OpenCV camera feed and MuJoCo visualization."""

    def __init__(
        self,
        camera_id: int = 0,
        model_path: str = "models/orca_hand.mjcf",
        dual_view: bool = False,
        dual_scale: float = 1.0,
    ) -> None:
        """Initialize the demo.

        Args:
            camera_id: Camera device ID
            model_path: Path to MuJoCo ORCA hand model
            dual_view: If True, show the OpenCV camera window alongside MuJoCo
            dual_scale: Scale factor for the OpenCV window when dual_view is enabled
        """
        # Initialize hand tracker
        print("Initializing hand tracker...")
        self.hand_tracker = MediaPipeTracker()

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
        self.dual_view = dual_view
        self.dual_scale = max(0.25, dual_scale)
        self.frame_queue: "mp.Queue | None" = None
        self.display_flag: Flag | None = None
        self.display_process: mp.Process | None = None

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
            if self.dual_view and self.display_flag is not None:
                self.display_flag.value = 0
            print("\nQuitting...")

    async def run(self) -> None:
        """Main loop."""
        print("\n=== Controls ===")
        print("'q' - Quit")
        print("================\n")

        if self.dual_view:
            self.frame_queue = mp.Queue(maxsize=2)
            self.display_flag = cast(Flag, mp.Value("i", 1))
            self.display_process = mp.Process(
                target=dual_window_process,
                args=(self.frame_queue, self.display_flag, "Hand Tracking", self.dual_scale),
                daemon=True,
            )
            self.display_process.start()

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

                if self.dual_view and self.display_flag is not None and not self.display_flag.value:
                    self.running = False
                    break

                timestamp = time.time()

                hand_poses = self.hand_tracker.detect_hands(frame, timestamp=timestamp)

                # Process first detected hand
                if hand_poses:
                    hand_pose = hand_poses[0]

                    # Get landmarks in hand frame
                    landmarks_hand_frame = hand_structure_to_landmarks(hand_pose)

                    # Retarget to ORCA hand
                    joint_angles_dict = self.retargeting.retarget_pose(landmarks_hand_frame, hand_pose.handedness)

                    # Apply offsets to match ORCA's reference positions
                    # Transform: orca_angle = retargeted_angle + orca_ref
                    orca_ref_offsets = {
                        # Thumb Offsets
                        "thumb_cmc_abd": 0.0,
                        "thumb_cmc_flex": 0.0,
                        "thumb_mcp": -0.2,
                        "thumb_ip": -0.2,
                        "index_mcp_abd": 0.0,
                        "index_mcp_flex": 0.0,
                        "index_pip": 0.0,
                        "middle_mcp_abd": 0.0,
                        "middle_mcp_flex": 0.0,
                        "middle_pip": 0.0,
                        "ring_mcp_abd": 0.0,
                        "ring_mcp_flex": 0.0,
                        "ring_pip": 0.0,
                        "pinky_mcp_abd": 0.0,
                        "pinky_mcp_flex": 0.0,
                        "pinky_pip": 0.0,
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
                    frame = self.hand_tracker.visualize(frame, hand_poses)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30.0 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time

                # Display info
                info_text = f"FPS: {fps:.1f} | Hands: {len(hand_poses)}"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if self.dual_view and self.frame_queue is not None and self.display_flag is not None:
                    if not self.display_flag.value:
                        self.running = False
                        break
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except Exception:
                        pass

                await asyncio.sleep(0.001)

        finally:
            # Cleanup
            await controller.stop()
            self.running = False
            self.cap.release()
            if self.dual_view and self.display_flag is not None:
                self.display_flag.value = 0
                if self.display_process is not None:
                    self.display_process.join(timeout=1.0)
            print("\nDemo finished.")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live hand tracking and retargeting demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--model", type=str, default="models/orca_hand.mjcf", help="Path to MuJoCo model")
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Show OpenCV camera window alongside the MuJoCo viewer.",
    )
    parser.add_argument(
        "--dual-scale",
        type=float,
        default=0.5,
        help="Scale factor for the dual OpenCV window (e.g., 0.5 for half size).",
    )
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
    demo = LiveRetargetingDemo(
        camera_id=args.camera,
        model_path=str(model_path),
        dual_view=args.dual,
        dual_scale=args.dual_scale,
    )
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
