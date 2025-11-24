"""Live demo with OpenCV in subprocess to display both windows on macOS.

Runs MuJoCo in main process (mjpython) and OpenCV in a subprocess.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import multiprocessing as mp
import time
from queue import Empty

import mujoco
import mujoco.viewer
import numpy as np
from askin import KeyboardController

from handpose import ORCAHandRetargeting


def opencv_subprocess_main(frame_queue: mp.Queue, running_flag: mp.Value, camera_id: int) -> None:  # type: ignore[valid-type]
    """OpenCV subprocess - runs in regular Python."""
    import cv2

    from handpose import HandTracker, ORCAHandRetargeting

    # Initialize
    hand_tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    retargeting = ORCAHandRetargeting()

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[OpenCV] Process started")

    frame_count = 0
    fps_start_time = time.time()
    fps: float = 0.0

    while running_flag.value:  # type: ignore[attr-defined]
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time()

        # Detect hands
        hand_poses = hand_tracker.detect_hands(frame, timestamp=timestamp)

        # Process and send joint angles
        if hand_poses:
            hand_pose = hand_poses[0]
            landmarks_hand_frame = hand_tracker.landmarks_in_hand_frame(hand_pose)
            joint_angles_dict = retargeting.retarget_pose(landmarks_hand_frame, hand_pose.handedness)

            # Apply offsets to match ORCA's reference positions
            # Transform: orca_angle = retargeted_angle + orca_ref
            orca_ref_offsets = {
                # Thumb Offsets
                # cmc_abd: Keep your tuned value, it looked fine for spread.
                "thumb_cmc_abd": -0.146,
                # cmc_flex: Reset to 0.0. The new math drives this positively now.
                # Add a small value (e.g. 0.1) only if the thumb looks too "tucked in" at neutral.
                "thumb_cmc_flex": 0.0,
                # mcp: FIX - Reduced from -0.873 to -0.2.
                # This restores your ability to flex the knuckle fully.
                "thumb_mcp": -0.2,
                "thumb_ip": -0.2,
                # Keep fingers as they were (working)
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

            # Apply offsets to joint angles
            for joint_name in joint_angles_dict:
                offset = orca_ref_offsets.get(joint_name, 0.0)
                joint_angles_dict[joint_name] = joint_angles_dict[joint_name] + offset

            # Send to main process
            try:
                frame_queue.put_nowait(("angles", joint_angles_dict))
            except Exception:
                pass

            # Draw landmarks
            frame = hand_tracker.draw_landmarks(frame, hand_poses)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()

        # Display
        info_text = f"FPS: {fps:.1f} | Hands: {len(hand_poses)}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Hand Tracking", frame)
        cv2.waitKey(1)  # Keep window responsive

    cap.release()
    cv2.destroyAllWindows()
    print("[OpenCV] Process finished")


async def main_async(
    model_path: Path,
    camera_id: int,
    running_flag: mp.Value,  # type: ignore[valid-type]
    frame_queue: mp.Queue,
    qpos_addrs: np.ndarray,
    mapped_names: list[str],
) -> None:
    """Async main loop with keyboard handling."""

    # Keyboard handler
    async def key_handler(key: str) -> None:
        if key == "q":
            running_flag.value = 0  # type: ignore[attr-defined]
            print("[Main] Quitting...")

    # Initialize keyboard controller in main process
    controller = KeyboardController(key_handler=key_handler, timeout=0.01)
    await controller.start()

    # Load MuJoCo model (in main process with mjpython)
    print("[MuJoCo] Loading model...")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    try:
        # Run MuJoCo viewer in main process
        print("[MuJoCo] Starting viewer...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while running_flag.value and viewer.is_running():  # type: ignore[attr-defined]
                # Get latest joint angles from OpenCV process
                try:
                    msg_type, joint_angles_dict = frame_queue.get_nowait()
                    if msg_type == "angles":
                        joint_angles_array = np.array([joint_angles_dict.get(name, 0.0) for name in mapped_names])
                        data.qpos[qpos_addrs] = joint_angles_array
                except Empty:
                    pass

                # Step simulation
                mujoco.mj_forward(model, data)
                viewer.sync()
                await asyncio.sleep(0.005)
    finally:
        await controller.stop()


def main() -> None:
    """Main entry point - runs MuJoCo in main process (mjpython)."""
    parser = argparse.ArgumentParser(description="Live demo with both windows")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--model", type=str, default="models/orca_hand.mjcf", help="MuJoCo model")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return

    print("\n=== Dual Window Demo ===")
    print("OpenCV window: Camera feed")
    print("MuJoCo window: ORCA hand")
    print("\nControls: 'q' - Quit")
    print("========================\n")

    # Create shared queue and flag
    frame_queue: mp.Queue[tuple[str, dict[str, float]]] = mp.Queue(maxsize=2)
    running_flag = mp.Value("i", 1)

    # Start OpenCV in subprocess
    opencv_proc = mp.Process(target=opencv_subprocess_main, args=(frame_queue, running_flag, args.camera))
    opencv_proc.start()

    # Give OpenCV time to start
    time.sleep(0.5)

    # Setup joint mapping (need to load model temporarily to get joint info)
    print("[MuJoCo] Setting up joint mapping...")
    temp_model = mujoco.MjModel.from_xml_path(str(model_path))
    retargeting = ORCAHandRetargeting()
    dummy_landmarks = np.zeros((21, 3))
    dummy_angles = retargeting.retarget_pose(dummy_landmarks, "Right")
    retargeted_joint_names = sorted(dummy_angles.keys())

    joint_mapping = {
        "thumb_cmc_flex": "right_thumb_mcp",
        "thumb_cmc_abd": "right_thumb_abd",
        "thumb_mcp": "right_thumb_pip",
        "thumb_ip": "right_thumb_dip",
        "index_mcp_abd": "right_index_abd",
        "index_mcp_flex": "right_index_mcp",
        "index_pip": "right_index_pip",
        "middle_mcp_abd": "right_middle_abd",
        "middle_mcp_flex": "right_middle_mcp",
        "middle_pip": "right_middle_pip",
        "ring_mcp_abd": "right_ring_abd",
        "ring_mcp_flex": "right_ring_mcp",
        "ring_pip": "right_ring_pip",
        "pinky_mcp_abd": "right_pinky_abd",
        "pinky_mcp_flex": "right_pinky_mcp",
        "pinky_pip": "right_pinky_pip",
    }

    qpos_addrs = []
    mapped_names = []
    for name in retargeted_joint_names:
        orca_name = joint_mapping.get(name)
        if orca_name:
            joint_id = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, orca_name)
            if joint_id >= 0:
                qpos_adr = temp_model.jnt_qposadr[joint_id]
                qpos_addrs.append(qpos_adr)
                mapped_names.append(name)

    qpos_addrs = np.array(qpos_addrs)
    print(f"[MuJoCo] Mapped {len(qpos_addrs)} joints")

    # Run async main loop
    try:
        asyncio.run(main_async(model_path, args.camera, running_flag, frame_queue, qpos_addrs, mapped_names))
    finally:
        # Cleanup
        running_flag.value = 0
        opencv_proc.join(timeout=2)
        if opencv_proc.is_alive():
            opencv_proc.terminate()

    print("\n[Main] Demo finished")


if __name__ == "__main__":
    main()
