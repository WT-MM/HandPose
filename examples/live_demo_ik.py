"""Live IK Retargeting Demo using Mink."""

import argparse
import asyncio
import multiprocessing as mp
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from queue import Empty
from typing import Protocol, cast

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from askin import KeyboardController

from handpose import ORCAHandIKRetargeting
from handpose.ik_retargeting import FINGER_TARGET_BODIES, MP_LANDMARK_INDICES, ORCAHandIKConfig
from handpose.tracker import BaseHandTracker, HandStructure
from handpose.tracker.hamer import HaMeRTracker
from handpose.tracker.mediapipe import MediaPipeTracker


class Flag(Protocol):
    value: int


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


def inject_target_bodies(mjcf_path: Path) -> str:
    """Injects mocap bodies into the MJCF XML string for visualization."""
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

    # Add mocap bodies for all keypoints (MCP, PIP, TIP, and IP for thumb)
    # Use different colors for different joint types
    joint_colors = {
        "mcp": "0 1 0 0.7",  # Green for MCP
        "pip": "0 0 1 0.7",  # Blue for PIP
        "ip": "1 1 0 0.7",  # Yellow for IP (thumb only)
        "tip": "1 0 0 0.7",  # Red for TIP
    }

    for finger, joints in FINGER_TARGET_BODIES.items():
        for joint_type in joints.keys():
            body = ET.SubElement(worldbody, "body")
            body.set("name", f"target_{finger}_{joint_type}")
            body.set("mocap", "true")
            body.set("pos", "0 0 0")

            # Add a visual sphere (smaller than before)
            geom = ET.SubElement(body, "geom")
            geom.set("type", "sphere")
            geom.set("size", "0.003")  # 3mm radius (smaller)
            geom.set("rgba", joint_colors.get(joint_type, "0.5 0.5 0.5 0.7"))
            geom.set("contype", "0")  # No collision
            geom.set("conaffinity", "0")

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


def _build_mp_label_lookup(target_joint_types: set[str]) -> dict[int, list[str]]:
    """Map MediaPipe landmark indices to ORCA joint labels."""
    mp_to_orca: dict[int, list[str]] = {}
    for finger, mp_mapping in MP_LANDMARK_INDICES.items():
        for joint_type, mp_idx in mp_mapping.items():
            if joint_type not in target_joint_types:
                continue
            labels = mp_to_orca.setdefault(mp_idx, [])
            label = f"{finger}_{joint_type}"
            if label not in labels:
                labels.append(label)
    return mp_to_orca


def hand_structure_to_landmarks(structure: HandStructure) -> np.ndarray:
    """Convert HandStructure to 21x3 landmarks array for IK (MediaPipe format).

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


def annotate_orca_labels(
    frame: np.ndarray,
    structure: HandStructure | None,
    label_lookup: dict[int, list[str]],
    camera_matrix: np.ndarray | None = None,
) -> None:
    """Overlay ORCA joint labels near the hand landmarks we target."""
    if structure is None:
        return

    # Project 3D landmarks to 2D for annotation
    landmarks_3d = hand_structure_to_landmarks(structure)

    # Transform to camera frame
    landmarks_homo = np.hstack([landmarks_3d, np.ones((landmarks_3d.shape[0], 1))])
    landmarks_camera = (structure.wrist_pose @ landmarks_homo.T).T[:, :3]

    if camera_matrix is not None:
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        landmarks_2d = np.zeros((landmarks_3d.shape[0], 2))
        for i, pt in enumerate(landmarks_camera):
            if pt[2] > 0:
                landmarks_2d[i] = [
                    (pt[0] * fx / pt[2]) + cx,
                    (pt[1] * fy / pt[2]) + cy,
                ]
    else:
        # Fallback: simple projection
        h, w = frame.shape[:2]
        landmarks_2d = np.zeros((landmarks_3d.shape[0], 2))
        for i, pt in enumerate(landmarks_camera):
            if pt[2] > 0:
                landmarks_2d[i] = [
                    pt[0] / pt[2] * w / 2 + w / 2,
                    pt[1] / pt[2] * h / 2 + h / 2,
                ]

    for mp_idx, labels in label_lookup.items():
        if mp_idx >= landmarks_2d.shape[0]:
            continue
        x, y = landmarks_2d[mp_idx]
        label_text = "/".join(labels)
        text_pos = (int(x) + 5, max(12, int(y) - 5))
        cv2.putText(
            frame,
            label_text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


async def main_async(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cap: cv2.VideoCapture,
    tracker: BaseHandTracker,
    ik_solver: ORCAHandIKRetargeting,
    target_body_ids: dict[str, dict[str, int]],
    frame_queue: "mp.Queue | None",
    display_flag: Flag | None,
    label_lookup: dict[int, list[str]],
    camera_matrix: np.ndarray | None = None,
) -> None:
    """Async main loop with keyboard handling."""
    running = True

    async def key_handler(key: str) -> None:
        nonlocal running
        if key == "q":
            running = False
            if display_flag is not None:
                display_flag.value = 0
            print("[Main] Quitting...")

    # Initialize keyboard controller
    controller = KeyboardController(key_handler=key_handler, timeout=0.01)
    await controller.start()

    start_time = time.time()

    frame_count = 0
    fps_start_time = time.time()
    fps: float = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while running and viewer.is_running() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if display_flag is not None and not display_flag.value:
                running = False
                break

            timestamp = time.time() - start_time
            hand_structures = tracker.detect_hands(frame, timestamp=timestamp)

            if hand_structures:
                structure = hand_structures[0]
                landmarks_hand = hand_structure_to_landmarks(structure)

                # --- IK SOLVE ---
                # 1. Update the configuration object with current robot state
                mujoco.mj_forward(model, data)
                ik_solver.configuration.update(data.qpos)

                # 2. Solve for new qpos
                target_q = ik_solver.solve(landmarks_hand)

                # 3. Apply to simulation (with NaN guard)
                if not np.any(np.isnan(target_q)) and not np.any(np.isinf(target_q)):
                    data.qpos[:] = target_q
                    data.qvel[:] = 0
                else:
                    print(f"Warning: IK produced NaNs on frame {frame_count}. Skipping update.")

                # --- VISUALIZATION ---
                # Update the mocap bodies to match the IK targets for all keypoints
                targets = ik_solver.compute_target_positions(landmarks_hand)

                # Get palm position and compute wrist position (consistent with IK solver)
                palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
                p_palm = data.xpos[palm_id]
                r_palm = data.xmat[palm_id].reshape(3, 3)

                # Use the same wrist offset as the IK solver
                wrist_offset_world = r_palm @ ik_solver.config.wrist_offset_palm
                p_wrist = p_palm + wrist_offset_world

                for finger, finger_targets in targets.items():
                    if finger not in target_body_ids:
                        continue

                    for joint_type, target_pos_mp in finger_targets.items():
                        if joint_type not in target_body_ids[finger]:
                            continue

                        rel = target_pos_mp - p_wrist
                        rot_vec = ik_solver.config.coord_transform @ rel
                        final_target = p_wrist + rot_vec

                        # Update Mocap body position
                        mocap_id = target_body_ids[finger][joint_type]
                        mocap_idx = model.body_mocapid[mocap_id]
                        if mocap_idx >= 0:
                            data.mocap_pos[mocap_idx] = final_target

            # Draw landmarks (include all detected hands)
            frame = tracker.visualize(frame, hand_structures, camera_matrix)
            if hand_structures:
                annotate_orca_labels(frame, hand_structures[0], label_lookup, camera_matrix)

            # Calculate FPS (every 30 frames)
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30.0 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time

            # Overlay simple stats (matches dual demo style)
            info_text = f"FPS: {fps:.1f} | Hands: {len(hand_structures)}"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            mujoco.mj_step(model, data)
            viewer.sync()

            if frame_queue is not None and display_flag is not None:
                if not display_flag.value:
                    running = False
                    break
                try:
                    frame_queue.put_nowait(frame.copy())
                except Exception:
                    pass

            await asyncio.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="orca_hand.mjcf")
    parser.add_argument("--scale", type=float, default=1.3, help="Robot/Human hand scale factor")
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
    parser.add_argument(
        "--targets",
        type=str,
        default="tip",
        help="Comma-separated joint targets (tip,ip,pip,mcp). Default: tip.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["mp", "hamer"],
        default="mp",
        help="Hand tracker to use: 'mp' for MediaPipe or 'hamer' for HaMeR (default: mp)",
    )
    args = parser.parse_args()

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
        print(f"Error: Model not found at {model_path}")
        return

    # 2. Inject visualization bodies (Mocap)
    print("Injecting visualization targets...")
    xml_string = inject_target_bodies(model_path)

    # 3. Load MuJoCo Model
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

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

    allowed_joint_types = set(target_joints)

    # 5. Helper to map finger names and joint types to mocap body IDs
    target_body_ids: dict[str, dict[str, int]] = {}
    for finger, joints in FINGER_TARGET_BODIES.items():
        for joint_type in joints.keys():
            if joint_type not in allowed_joint_types:
                continue
            finger_dict = target_body_ids.setdefault(finger, {})
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target_{finger}_{joint_type}")
            finger_dict[joint_type] = bid

    # 6. Main Loop
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_label_lookup = _build_mp_label_lookup(set(target_joints))

    # Simple camera matrix (can be calibrated properly)
    camera_matrix = None  # TODO: Add proper camera calibration

    frame_queue: "mp.Queue | None" = None
    display_flag: Flag | None = None
    display_process: mp.Process | None = None
    if args.dual:
        frame_queue = mp.Queue(maxsize=2)
        display_flag = cast(Flag, mp.Value("i", 1))
        display_process = mp.Process(
            target=dual_window_process,
            args=(frame_queue, display_flag, "Hand Tracking", max(0.25, args.dual_scale)),
            daemon=True,
        )
        display_process.start()

    print("\nStarting simulation...")
    print("Press 'q' to quit (or close viewer).")

    # Run async main loop
    asyncio.run(
        main_async(
            model,
            data,
            cap,
            tracker,
            ik_solver,
            target_body_ids,
            frame_queue,
            display_flag,
            mp_label_lookup,
            camera_matrix,
        )
    )

    cap.release()
    if display_flag is not None:
        display_flag.value = 0
        if display_process is not None:
            display_process.join(timeout=1.0)


"""
Run with `mjpython examples/live_demo_ik.py --model orca_hand_fixed.mjcf --scale 1.0`
"""
if __name__ == "__main__":
    main()
