"""Offline IK Retargeting Demo using Mink (recorded to hdf5 file)."""

import argparse
import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import colorlogging
import cv2
import h5py
import mujoco
import numpy as np
from askin import KeyboardController

from handpose import ORCAHandIKRetargeting
from handpose.ik_retargeting import ORCA_JOINT_NAMES, ORCAHandIKConfig
from handpose.tracker import BaseHandTracker
from handpose.tracker.hamer import HaMeRTracker
from handpose.tracker.mediapipe import MediaPipeTracker

logger = logging.getLogger(__name__)


def load_mjcf_model(mjcf_path: Path) -> str:
    """Load MJCF model and convert relative paths to absolute paths."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

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

    return ET.tostring(root, encoding="unicode")


def get_joint_names_from_model(model: mujoco.MjModel) -> list[str]:
    """Extract joint names from MuJoCo model."""
    joint_names = []
    for i in range(model.njnt):
        # Get joint name by index
        name_start = model.name_jointadr[i]
        if name_start >= 0:
            name_bytes = model.names[name_start:].split(b"\x00")[0]
            joint_name = name_bytes.decode("utf-8")
            joint_names.append(joint_name)
        elif i < len(ORCA_JOINT_NAMES):
            # Fallback: use ORCA joint names if available
            joint_names.append(ORCA_JOINT_NAMES[i])
        else:
            joint_names.append(f"joint_{i}")
    return joint_names


async def main_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cap: cv2.VideoCapture,
    tracker: BaseHandTracker,
    ik_solver: ORCAHandIKRetargeting,
    output_path: Path,
    joint_names: list[str],
    target_fps: float = 30.0,
    show_preview: bool = True,
) -> None:
    """Main loop: capture video, run IK, save joint states to HDF5.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        cap: The video capture.
        tracker: The hand tracker.
        ik_solver: The IK solver.
        output_path: The path to the output HDF5 file.
        joint_names: The names of the joints.
        target_fps: The target frequency in Hz.
        show_preview: Whether to show the preview window.
    """
    # Initialize data storage
    frame_count = 0
    start_time = time.time()
    target_dt = 1.0 / target_fps if target_fps > 0 else 0.0

    # Frequency monitoring
    fps_check_interval = 30  # Check FPS every N frames
    frame_times: list[float] = []
    last_fps_check_frame = 0

    running = True

    async def key_handler(key: str) -> None:
        nonlocal running
        if key == "q":
            running = False
            logger.info("Quitting...")

    # Initialize keyboard controller
    controller = KeyboardController(key_handler=key_handler, timeout=0.01)
    await controller.start()

    logger.info("Recording to %s", output_path)
    logger.info("Target frequency: %f Hz", target_fps)
    logger.info("Press 'q' to quit and save.")

    try:
        with h5py.File(output_path, "w") as h5_file:
            # Create datasets (we'll resize as we go)
            max_frames = 10000  # Initial estimate, will resize if needed
            qpos_dset = h5_file.create_dataset(
                "qpos",
                shape=(max_frames, model.nq),
                maxshape=(None, model.nq),
                dtype=np.float32,
                chunks=(100, model.nq),
            )
            timestamp_dset = h5_file.create_dataset(
                "timestamps",
                shape=(max_frames,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=(100,),
            )

            # Store metadata
            h5_file.attrs["joint_names"] = [name.encode("utf-8") for name in joint_names]
            h5_file.attrs["n_joints"] = model.nq
            h5_file.attrs["model_path"] = str(output_path)
            h5_file.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            h5_file.attrs["target_fps"] = target_fps

            last_frame_time = time.time()

            while running and cap.isOpened():
                loop_start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = time.time() - start_time
                hand_structures = tracker.detect_hands(frame, timestamp=timestamp)

                if hand_structures:
                    structure = hand_structures[0]

                    # --- IK SOLVE ---
                    # 1. Update the configuration object with current robot state
                    mujoco.mj_forward(model, data)
                    ik_solver.configuration.update(data.qpos)

                    # 2. Solve for new qpos
                    target_q = ik_solver.solve(structure)

                    # 3. Save valid joint states (with NaN guard)
                    if not np.any(np.isnan(target_q)) and not np.any(np.isinf(target_q)):
                        # Resize datasets if needed
                        if frame_count >= max_frames:
                            max_frames = int(max_frames * 1.5)
                            qpos_dset.resize((max_frames, model.nq))
                            timestamp_dset.resize((max_frames,))

                        # Store data
                        qpos_dset[frame_count] = target_q.astype(np.float32)
                        timestamp_dset[frame_count] = timestamp
                        frame_count += 1
                    else:
                        logger.warning("IK produced NaNs on frame %d. Skipping.", frame_count)

                # Optional preview window
                if show_preview:
                    frame = tracker.visualize(frame, hand_structures)
                    # Calculate current FPS for display (use time since last frame)
                    frame_dt = time.time() - last_frame_time
                    current_fps = 1.0 / frame_dt if frame_dt > 0 else 0.0
                    info_text = f"Frames: {frame_count} | Hands: {len(hand_structures)} | FPS: {current_fps:.1f}"
                    cv2.putText(
                        frame, info_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    cv2.imshow("Hand Tracking (Press 'q' to quit)", frame)
                    # Use non-blocking waitKey to avoid stalls
                    cv2.waitKey(1)

                # Frequency control: maintain target FPS
                loop_elapsed = time.time() - loop_start_time
                if target_dt > 0 and loop_elapsed < target_dt:
                    time.sleep(target_dt - loop_elapsed)

                # Track frame times for frequency monitoring (time between frames)
                current_time = time.time()
                if frame_count > 0:
                    frame_dt = current_time - last_frame_time
                    frame_times.append(frame_dt)

                last_frame_time = current_time

                # Check actual frequency periodically and warn if too low
                if frame_count - last_fps_check_frame >= fps_check_interval:
                    if len(frame_times) >= fps_check_interval:
                        # Calculate average FPS over the last interval
                        avg_frame_time = sum(frame_times[-fps_check_interval:]) / fps_check_interval
                        actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

                        if actual_fps < target_fps:
                            logger.warning(
                                "Actual frequency (%f Hz) is lower than target (%f Hz).",
                                actual_fps,
                                target_fps,
                            )

                        last_fps_check_frame = frame_count

                        # Keep only recent frame times to avoid memory growth
                        if len(frame_times) > fps_check_interval * 2:
                            frame_times = frame_times[-fps_check_interval:]

                # Small async sleep to allow keyboard events to be processed
                await asyncio.sleep(0.001)

            # Resize datasets to actual size
            if frame_count > 0:
                qpos_dset.resize((frame_count, model.nq))
                timestamp_dset.resize((frame_count,))

                # Calculate final statistics
                total_time = time.time() - start_time
                actual_fps = frame_count / total_time if total_time > 0 else 0.0

                # Store actual FPS in metadata
                h5_file.attrs["actual_fps"] = actual_fps
                h5_file.attrs["total_frames"] = frame_count
                h5_file.attrs["total_time"] = total_time

                logger.info("Saved %d frames to %s", frame_count, output_path)
                logger.info("Actual frequency: %f Hz (target: %f Hz)", actual_fps, target_fps)

                if actual_fps < target_fps:
                    logger.warning(
                        "Final frequency (%f Hz) was lower than target (%f Hz). \
                    Consider reducing --fps for future recordings.",
                        actual_fps,
                        target_fps,
                    )
            else:
                logger.warning("No valid frames recorded!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if show_preview:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="orca_hand.mjcf")
    parser.add_argument("--scale", type=float, default=1.3, help="Robot/Human hand scale factor")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--targets",
        type=str,
        default="tip, ip",
        help="Comma-separated joint targets (tip,ip,pip,mcp). Default: tip.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["hamer", "mp"],
        default="hamer",
        help="Hand tracker to use: 'hamer' for HaMeR or 'mp' for MediaPipe (default: hamer)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="retargeted_joints.hdf5",
        help="Output HDF5 file path for saving joint states (default: retargeted_joints.hdf5)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV preview window",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help=(
            "Target frequency in Hz for recording (default: 30.0). Warning will be shown if actual frequency is lower."
        ),
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    colorlogging.configure(level=level)

    raw_targets = [part.strip().lower() for part in args.targets.split(",")]
    target_joints = tuple(dict.fromkeys(jt for jt in raw_targets if jt))
    if not target_joints:
        target_joints = ("tip",)
    supported_targets = {"tip", "ip", "pip", "mcp"}
    invalid = [jt for jt in target_joints if jt not in supported_targets]
    if invalid:
        parser.error(f"Unsupported target joint types: {', '.join(invalid)}")

    # 1. Setup paths and model
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "models" / args.model if not Path(args.model).is_absolute() else Path(args.model)

    if not model_path.exists():
        logger.error("Model not found at %s", model_path)
        return

    # 2. Load MuJoCo Model
    print("Loading MuJoCo model...")
    xml_string = load_mjcf_model(model_path)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Get joint names from model
    joint_names = get_joint_names_from_model(model)
    print(f"Found {len(joint_names)} joints in model")

    print(f"Initializing Hand Tracker ({args.tracker})...")
    tracker: BaseHandTracker
    if args.tracker == "hamer":
        tracker = HaMeRTracker(smoothing_factor=0.0, conf_threshold=0.3)
    else:
        tracker = MediaPipeTracker(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    print("Initializing IK Solver (Mink)...")

    ik_config = ORCAHandIKConfig(
        scale_factor=args.scale,
        wrist_offset_palm=np.array([0.000, 0.0, -0.05]),
        target_joint_types=target_joints,
    )

    ik_solver = ORCAHandIKRetargeting(model, config=ik_config)

    # 5. Setup camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 6. Setup output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    # 7. Run main loop
    asyncio.run(
        main_loop(
            model,
            data,
            cap,
            tracker,
            ik_solver,
            output_path,
            joint_names,
            target_fps=args.fps,
            show_preview=not args.no_preview,
        )
    )

    cap.release()


"""
Run with:
    python examples/offline_retargeting.py --model orca_hand_fixed.mjcf --scale 1.0 --output my_joints.hdf5 --fps 30.0
"""
if __name__ == "__main__":
    main()
