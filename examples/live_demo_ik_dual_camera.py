"""Live IK Retargeting Demo using Mink with dual camera inputs.

This example takes two camera inputs and merges/decides between MediaPipe outputs
before running inverse kinematics.
"""

import argparse
import asyncio
import multiprocessing as mp
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from queue import Empty
from typing import Literal, Protocol, cast

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from askin import KeyboardController

from handpose import HandPose, HandTracker, ORCAHandIKRetargeting
from handpose.ik_retargeting import FINGER_TARGET_BODIES, MP_LANDMARK_INDICES, ORCAHandIKConfig


class Flag(Protocol):
    value: int


MergeStrategy = Literal["confidence", "average", "primary", "best_hand"]


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


def merge_hand_poses(
    poses_cam1: list[HandPose],
    poses_cam2: list[HandPose],
    strategy: MergeStrategy = "confidence",
) -> HandPose | None:
    """Merge or select hand poses from two cameras with rotation-invariant averaging."""

    def _to_local(pose: HandPose) -> np.ndarray:
        """Convert 3D landmarks to local hand frame (wrist at origin)."""
        wrist_inv = np.linalg.inv(pose.wrist_pose)
        ones = np.ones((pose.landmarks_3d.shape[0], 1))
        lm_homo = np.hstack([pose.landmarks_3d, ones])
        lm_local = (wrist_inv @ lm_homo.T).T
        return lm_local[:, :3]

    def _to_global(local_landmarks: np.ndarray, wrist_pose: np.ndarray) -> np.ndarray:
        """Convert local hand landmarks back to world/camera frame."""
        ones = np.ones((local_landmarks.shape[0], 1))
        lm_homo = np.hstack([local_landmarks, ones])
        lm_world = (wrist_pose @ lm_homo.T).T
        return lm_world[:, :3]

    if not poses_cam1 and not poses_cam2:
        return None

    if strategy == "confidence":
        all_poses = poses_cam1 + poses_cam2
        if not all_poses:
            return None
        return max(all_poses, key=lambda p: p.confidence)

    elif strategy == "average":
        if not poses_cam1:
            return poses_cam2[0] if poses_cam2 else None
        if not poses_cam2:
            return poses_cam1[0] if poses_cam1 else None

        pose1 = poses_cam1[0]
        pose2 = poses_cam2[0]

        lm1_local = _to_local(pose1)
        lm2_local = _to_local(pose2)

        total_conf = pose1.confidence + pose2.confidence
        if total_conf == 0:
            w1 = w2 = 0.5
        else:
            w1 = pose1.confidence / total_conf
            w2 = pose2.confidence / total_conf

        lm_local_avg = (w1 * lm1_local) + (w2 * lm2_local)

        if pose1.confidence >= pose2.confidence:
            ref_wrist_pose = pose1.wrist_pose
            ref_handedness = pose1.handedness
            ref_2d = pose1.landmarks_2d
        else:
            ref_wrist_pose = pose2.wrist_pose
            ref_handedness = pose2.handedness
            ref_2d = pose2.landmarks_2d

        lm_world_merged = _to_global(lm_local_avg, ref_wrist_pose)

        merged_pose = HandPose(
            landmarks_2d=ref_2d,
            landmarks_3d=lm_world_merged,
            handedness=ref_handedness,
            confidence=(pose1.confidence + pose2.confidence) / 2.0,
            wrist_pose=ref_wrist_pose,
            timestamp=(pose1.timestamp + pose2.timestamp) / 2.0,
        )

        return merged_pose

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def inject_target_bodies(mjcf_path: Path) -> str:
    """Injects mocap bodies into the MJCF XML string for visualization."""
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

    # Add mocap bodies for all keypoints
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

            geom = ET.SubElement(body, "geom")
            geom.set("type", "sphere")
            geom.set("size", "0.003")
            geom.set("rgba", joint_colors.get(joint_type, "0.5 0.5 0.5 0.7"))
            geom.set("contype", "0")
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


def annotate_orca_labels(frame: np.ndarray, pose: HandPose | None, label_lookup: dict[int, list[str]]) -> None:
    """Overlay ORCA joint labels near the MediaPipe landmarks we target."""
    if pose is None:
        return

    landmarks = pose.landmarks_2d
    for mp_idx, labels in label_lookup.items():
        if mp_idx >= landmarks.shape[0]:
            continue
        x, y = landmarks[mp_idx]
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


def draw_camera_labels(frame: np.ndarray, camera_name: str, hand_count: int) -> None:
    """Draw camera label and hand count on frame."""
    label_text = f"{camera_name}: {hand_count} hand(s)"
    cv2.putText(
        frame,
        label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


async def main_async(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cap1: cv2.VideoCapture,
    cap2: cv2.VideoCapture,
    tracker1: HandTracker,
    tracker2: HandTracker,
    ik_solver: ORCAHandIKRetargeting,
    target_body_ids: dict[str, dict[str, int]],
    frame_queue: "mp.Queue | None",
    display_flag: Flag | None,
    label_lookup: dict[int, list[str]],
    merge_strategy: MergeStrategy,
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

    print("Launching MuJoCo viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while running and viewer.is_running() and cap1.isOpened() and cap2.isOpened():
            # Read frames from both cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                print("Warning: Failed to read from one or both cameras")
                break

            if display_flag is not None and not display_flag.value:
                running = False
                break

            timestamp = time.time() - start_time

            # Detect hands from both cameras
            hand_poses_cam1 = tracker1.detect_hands(frame1, timestamp=timestamp)
            hand_poses_cam2 = tracker2.detect_hands(frame2, timestamp=timestamp)

            # Merge/decide on hand poses
            merged_pose = merge_hand_poses(hand_poses_cam1, hand_poses_cam2, strategy=merge_strategy)

            # Debug output (print every 30 frames to avoid spam)
            if frame_count % 30 == 0:
                cam1_conf = hand_poses_cam1[0].confidence if len(hand_poses_cam1) > 0 else 0.0
                cam2_conf = hand_poses_cam2[0].confidence if len(hand_poses_cam2) > 0 else 0.0
                merged_conf = merged_pose.confidence if merged_pose else 0.0
                print(
                    f"Frame {frame_count}: "
                    f"Cam1: {len(hand_poses_cam1)} hands (conf: {cam1_conf:.2f}), "
                    f"Cam2: {len(hand_poses_cam2)} hands (conf: {cam2_conf:.2f}), "
                    f"Merged: {'Yes' if merged_pose else 'No'} (conf: {merged_conf:.2f})"
                )

            # Draw landmarks on both camera frames
            frame1 = tracker1.draw_landmarks(frame1, hand_poses_cam1)
            frame2 = tracker2.draw_landmarks(frame2, hand_poses_cam2)

            # Add camera labels
            draw_camera_labels(frame1, "Camera 1", len(hand_poses_cam1))
            draw_camera_labels(frame2, "Camera 2", len(hand_poses_cam2))

            # Process merged pose for IK
            if merged_pose:
                landmarks_hand = tracker1.landmarks_in_hand_frame(merged_pose)

                mujoco.mj_forward(model, data)
                ik_solver.configuration.update(data.qpos)
                target_q = ik_solver.solve(landmarks_hand)

                if not np.any(np.isnan(target_q)) and not np.any(np.isinf(target_q)):
                    data.qpos[:] = target_q
                    data.qvel[:] = 0
                else:
                    print(f"Warning: IK produced NaNs on frame {frame_count}. Skipping update.")

                targets = ik_solver.compute_target_positions(landmarks_hand)
                palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
                p_palm = data.xpos[palm_id]
                r_palm = data.xmat[palm_id].reshape(3, 3)
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

                annotate_orca_labels(frame1, merged_pose, label_lookup)

            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            if h1 != h2:
                scale = h1 / h2
                new_w2 = int(w2 * scale)
                frame2_resized = cv2.resize(frame2, (new_w2, h1))
            else:
                frame2_resized = frame2

            combined_frame = np.hstack([frame1, frame2_resized])

            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30.0 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time

            # Overlay stats
            cam1_count = len(hand_poses_cam1)
            cam2_count = len(hand_poses_cam2)
            merged_info = "Merged" if merged_pose else "None"
            info_text = f"FPS: {fps:.1f} | Cam1: {cam1_count} | Cam2: {cam2_count} | Merged: {merged_info}"
            cv2.putText(
                combined_frame,
                info_text,
                (10, combined_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if merged_pose:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            viewer.sync()

            if frame_queue is not None and display_flag is not None:
                if not display_flag.value:
                    running = False
                    break
                try:
                    frame_queue.put_nowait(combined_frame.copy())
                except Exception:
                    pass

            await asyncio.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live IK Retargeting Demo with dual camera inputs",
    )
    parser.add_argument("--camera1", type=int, default=0, help="First camera device ID (default: 0)")
    parser.add_argument("--camera2", type=int, default=1, help="Second camera device ID (default: 1)")
    parser.add_argument("--model", type=str, default="orca_hand_fixed.mjcf", help="MuJoCo model file")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Robot/Human hand scale factor (default: 1.0)",
    )
    parser.add_argument(
        "--merge",
        type=str,
        choices=["confidence", "average"],
        default="confidence",
        help="Strategy for merging hand poses from two cameras (default: confidence)",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Show OpenCV camera window alongside the MuJoCo viewer",
    )
    parser.add_argument(
        "--dual-scale",
        type=float,
        default=0.5,
        help="Scale factor for the dual OpenCV window (default: 0.5)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="tip, ip",
        help="Comma-separated joint targets (tip,ip,pip,mcp). Default: tip, ip.",
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

    # Setup paths and model
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "models" / args.model if not Path(args.model).is_absolute() else Path(args.model)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Inject visualization bodies
    print("Injecting visualization targets...")
    xml_string = inject_target_bodies(model_path)

    # Load MuJoCo Model
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    print("Initializing Hand Trackers...")
    tracker1 = HandTracker(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    tracker2 = HandTracker(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    print("Initializing IK Solver (Mink)...")
    ik_config = ORCAHandIKConfig(
        scale_factor=args.scale,
        wrist_offset_palm=np.array([0.000, 0.0, -0.05]),
        target_joint_types=target_joints,
    )

    ik_solver = ORCAHandIKRetargeting(model, config=ik_config)

    allowed_joint_types = set(target_joints)

    # Map finger names and joint types to mocap body IDs
    target_body_ids: dict[str, dict[str, int]] = {}
    for finger, joints in FINGER_TARGET_BODIES.items():
        for joint_type in joints.keys():
            if joint_type not in allowed_joint_types:
                continue
            finger_dict = target_body_ids.setdefault(finger, {})
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target_{finger}_{joint_type}")
            finger_dict[joint_type] = bid

    # Open cameras
    print(f"Opening cameras {args.camera1} and {args.camera2}...")
    cap1 = cv2.VideoCapture(args.camera1)
    cap2 = cv2.VideoCapture(args.camera2)

    if not cap1.isOpened():
        print(f"Error: Could not open camera {args.camera1}")
        return

    if not cap2.isOpened():
        print(f"Error: Could not open camera {args.camera2}")
        cap1.release()
        return

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_label_lookup = _build_mp_label_lookup(set(target_joints))

    frame_queue: "mp.Queue | None" = None
    display_flag: Flag | None = None
    display_process: mp.Process | None = None
    if args.dual:
        frame_queue = mp.Queue(maxsize=2)
        display_flag = cast(Flag, mp.Value("i", 1))
        display_process = mp.Process(
            target=dual_window_process,
            args=(frame_queue, display_flag, "Hand Tracking (Dual Camera)", max(0.25, args.dual_scale)),
            daemon=True,
        )
        display_process.start()

    print(f"\nMerge strategy: {args.merge}")
    print("Starting simulation...")
    print("Press 'q' to quit (or close viewer).")
    print(f"Model has {model.nq} DOF, {model.nbody} bodies")

    # Run async main loop
    asyncio.run(
        main_async(
            model,
            data,
            cap1,
            cap2,
            tracker1,
            tracker2,
            ik_solver,
            target_body_ids,
            frame_queue,
            display_flag,
            mp_label_lookup,
            args.merge,
        )
    )

    cap1.release()
    cap2.release()
    if display_flag is not None:
        display_flag.value = 0
        if display_process is not None:
            display_process.join(timeout=1.0)


"""
Run with:
mjpython examples/live_demo_ik_dual_camera.py --camera1 0 --camera2 1 --merge average --dual
"""
if __name__ == "__main__":
    main()
