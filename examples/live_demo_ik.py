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


def inject_target_bodies(mjcf_path: Path, target_joint_types: tuple[str, ...]) -> str:
    """Inject mocap bodies for tracker sites and ensure IK target sites exist.

    Args:
        mjcf_path: Path to MJCF model file
        target_joint_types: Tuple of joint types to visualize (tip, ip, pip, mcp)

    Returns:
        Modified MJCF XML string
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if worldbody is None:
        raise ValueError("Could not find worldbody in MJCF")

    # Convert relative paths to absolute paths
    model_dir = mjcf_path.parent
    for asset in root.findall(".//asset"):
        for mesh in asset.findall("mesh"):
            file_attr = mesh.get("file")
            if file_attr and not Path(file_attr).is_absolute():
                abs_path = (model_dir / file_attr).resolve()
                mesh.set("file", str(abs_path))

    # All IK target sites use blue for consistency
    ik_site_color = "0 0 1 0.9"  # Blue for all IK targets

    # Tip site offsets: [x, y, z] where z is upward direction
    tip_site_specs = {
        "thumb": ("right_thumb_dp", np.array([0.0, 0.0, 0.025])),
        "index": ("right_index_ip", np.array([0.0, 0.0, 0.035])),
        "middle": ("right_middle_ip", np.array([0.0, 0.0, 0.037])),
        "ring": ("right_ring_ip", np.array([0.0, 0.0, 0.036])),
        "pinky": ("right_pinky_ip", np.array([0.0, 0.0, 0.032])),
    }

    if "tip" in target_joint_types:
        tip_sites_created = []
        for finger, (parent_body, offset) in tip_site_specs.items():
            body_elem = root.find(f".//body[@name='{parent_body}']")
            if body_elem is None:
                print(f"Warning: Parent body '{parent_body}' not found for {finger} tip site")
                continue
            site_name = f"right_{finger}_tip_site"
            existing_site = body_elem.find(f"./site[@name='{site_name}']")
            if existing_site is not None:
                # Update existing site - make it larger and blue
                existing_site.set("type", "sphere")  # Ensure type is set
                existing_site.set("size", "0.005")
                existing_site.set("rgba", ik_site_color)
                tip_sites_created.append(f"{finger} (updated)")
            else:
                # Create new tip site
                site = ET.SubElement(body_elem, "site")
                site.set("name", site_name)
                site.set("type", "sphere")  # Explicit type for visibility
                site.set("pos", " ".join(f"{value:.5f}" for value in offset))
                site.set("size", "0.005")
                site.set("rgba", ik_site_color)
                tip_sites_created.append(f"{finger} (created)")
        print(f"Tip sites: {', '.join(tip_sites_created)}")

    # Add visual markers for IK target bodies (MCP, PIP, IP) that might not have visual geoms
    allowed_types = set(target_joint_types)
    markers_added = []
    for finger_name, bodies in FINGER_TARGET_BODIES.items():
        for joint_type, (frame_type, frame_name) in bodies.items():
            if joint_type not in allowed_types:
                continue

            if frame_type == "body":
                # Add a visual site to the body for visibility
                body_elem = root.find(f".//body[@name='{frame_name}']")
                if body_elem is not None:
                    # Check if marker site already exists
                    marker_site_name = f"ik_marker_{frame_name}"
                    existing_marker = body_elem.find(f"./site[@name='{marker_site_name}']")
                    if existing_marker is not None:
                        # Update existing marker
                        existing_marker.set("size", "0.004")  # Same size as tip sites
                        existing_marker.set("rgba", ik_site_color)
                        markers_added.append(f"{finger_name}_{joint_type} (updated)")
                    else:
                        # Create new marker site
                        site = ET.SubElement(body_elem, "site")
                        site.set("name", marker_site_name)
                        site.set("pos", "0 0 0")  # At body origin
                        site.set("size", "0.004")  # Same size as tip sites for consistency
                        site.set("rgba", ik_site_color)  # Blue for all IK targets
                        markers_added.append(f"{finger_name}_{joint_type} (created)")
                else:
                    print(f"Warning: Body '{frame_name}' not found in MJCF for {finger_name}_{joint_type}")
            elif frame_type == "site":
                # This is a tip site - find and update it (should already be handled above, but double-check)
                found = False
                for body in root.findall(".//body"):
                    site_elem = body.find(f"./site[@name='{frame_name}']")
                    if site_elem is not None:
                        site_elem.set("size", "0.005")
                        site_elem.set("rgba", ik_site_color)
                        found = True
                        markers_added.append(f"{finger_name}_{joint_type} (site updated)")
                        break
                if not found:
                    print(f"Warning: Site '{frame_name}' not found in MJCF for {finger_name}_{joint_type}")

    print(f"Added/updated {len(markers_added)} IK target markers: {', '.join(markers_added)}")

    # Add mocap bodies for tracker sites (MediaPipe landmarks)
    # All tracker sites use red for consistency - these represent the actual tracked hand
    tracker_color = "1 0 0 0.9"  # Red for all tracker sites (actual hand)

    allowed_types = set(target_joint_types)
    for finger_name, mp_mapping in MP_LANDMARK_INDICES.items():
        for joint_type, mp_idx in mp_mapping.items():
            if joint_type not in allowed_types:
                continue

            # Create mocap body for tracker site
            mocap_body = ET.SubElement(worldbody, "body")
            mocap_body.set("name", f"tracker_{finger_name}_{joint_type}")
            mocap_body.set("mocap", "true")
            mocap_body.set("pos", "0 0 0")

            # Add visual sphere for tracker site (larger, red)
            geom = ET.SubElement(mocap_body, "geom")
            geom.set("type", "sphere")
            geom.set("size", "0.005")  # 5mm radius (larger than IK targets for visibility)
            geom.set("rgba", tracker_color)  # Red for all tracker sites
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

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

    # If the tracker provided pixel-space landmarks, use them directly.
    lm2d = getattr(structure, "landmarks_2d", None)
    if lm2d is not None and lm2d.shape[0] >= 21:
        landmarks_2d = lm2d
    else:
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
    joint_smoothing: float = 1.0,
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

    # Initialize joint position smoothing state
    smoothed_qpos = data.qpos.copy()

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

                # --- IK SOLVE ---
                # 1. Update the configuration object with current robot state
                mujoco.mj_forward(model, data)
                ik_solver.configuration.update(data.qpos)

                # 2. Solve for new qpos
                target_q = ik_solver.solve(structure)

                # 3. Apply joint position smoothing (always applied; joint_smoothing=1.0 means no smoothing)
                if not np.any(np.isnan(target_q)) and not np.any(np.isinf(target_q)):
                    # Exponential moving average: smoothed = joint_smoothing * new + (1 - joint_smoothing) * old
                    smoothed_qpos = joint_smoothing * target_q + (1.0 - joint_smoothing) * smoothed_qpos
                    target_q = smoothed_qpos

                # 4. Apply to simulation (with NaN guard)
                if not np.any(np.isnan(target_q)) and not np.any(np.isinf(target_q)):
                    data.qpos[:] = target_q
                    data.qvel[:] = 0
                else:
                    print(f"Warning: IK produced NaNs on frame {frame_count}. Skipping update.")

                # --- VISUALIZATION ---
                # Update the mocap bodies to match the IK targets for all keypoints
                targets = ik_solver.compute_target_positions(structure)

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
    parser.add_argument("--model", type=str, default="orca_hand_fixed.mjcf")
    parser.add_argument("--scale", type=float, default=1.1, help="Robot/Human hand scale factor")
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
        help="Comma-separated joint targets (tip,ip,pip,mcp). Default: tip, ip",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["mp", "hamer"],
        default="mp",
        help="Hand tracker to use: 'mp' for MediaPipe or 'hamer' for HaMeR (default: mp)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Exponential smoothing factor for HaMeR tracker (0.0 = no smoothing)",
    )
    parser.add_argument(
        "--joint-smoothing",
        type=float,
        default=0.7,
        help="Smoothing factor for joint positions (0.0-1.0). \
        Lower values = more smoothing, higher = less smoothing. Default: 0.7",
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
    model_file = Path(args.model)
    if not model_file.is_absolute():
        model_file = Path(__file__).parent.parent / args.model
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # 2. Inject visualization bodies (Mocap)
    print(f"Loading model from {model_file}")
    xml_string = inject_target_bodies(model_file, target_joints)

    # 3. Load MuJoCo Model
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    print(f"Initializing Hand Tracker ({args.tracker})...")
    tracker: BaseHandTracker
    if args.tracker == "hamer":
        tracker = HaMeRTracker(smoothing_factor=args.smoothing, conf_threshold=0.3)
    else:
        tracker = MediaPipeTracker(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    print("Initializing IK Solver (Mink)...")

    ik_config = ORCAHandIKConfig(
        scale_factor=args.scale,
        target_joint_types=target_joints,
        lm_damping=0.5,
        ik_iterations=15,
    )

    ik_solver = ORCAHandIKRetargeting(model, config=ik_config)

    # Verify tip sites exist in the loaded model
    if "tip" in target_joints:
        print("\nVerifying tip sites in loaded model:")
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            site_name = f"right_{finger}_tip_site"
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                site_pos = data.site(site_id).xpos.copy()
                print(f"  ✓ {site_name}: found at position [{site_pos[0]:.4f}, {site_pos[1]:.4f}, {site_pos[2]:.4f}]")
            else:
                print(f"  ✗ {site_name}: NOT FOUND in model!")

    allowed_joint_types = set(target_joints)

    # 5. Helper to map finger names and joint types to mocap body IDs
    # Note: These use "tracker_" prefix to match the visualization bodies created by inject_target_bodies
    target_body_ids: dict[str, dict[str, int]] = {}
    for finger, joints in FINGER_TARGET_BODIES.items():
        for joint_type in joints.keys():
            if joint_type not in allowed_joint_types:
                continue
            finger_dict = target_body_ids.setdefault(finger, {})
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tracker_{finger}_{joint_type}")
            if bid < 0:
                print(f"Warning: Mocap body 'tracker_{finger}_{joint_type}' not found")
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

    # Validate joint smoothing value
    if not (0.0 < args.joint_smoothing <= 1.0):
        parser.error("--joint-smoothing must be between 0.0 and 1.0 (exclusive of 0.0)")

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
            joint_smoothing=args.joint_smoothing,
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
