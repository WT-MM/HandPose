"""Record video, process with HAMER, visualize in MuJoCo, and concatenate videos.

This script:
1. Listens for key press ('r' or 's') to start recording
2. Records video from camera
3. Saves the recorded video
4. Processes video with HAMER frame by frame
5. Opens MuJoCo and visualizes the processed hand tracking
6. Records the MuJoCo visualization
7. Concatenates both videos together
"""

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import mujoco
import mujoco.viewer
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from handpose.ik_retargeting import (
    FINGER_TARGET_BODIES,
    MP_LANDMARK_INDICES,
    ORCAHandIKConfig,
    ORCAHandIKRetargeting,
)
from handpose.tracker.base import HandStructure
from handpose.tracker.hamer import HaMeRTracker


def inject_visualization_bodies(mjcf_path: Path, target_joint_types: tuple[str, ...]) -> str:
    """Inject mocap bodies for tracker sites and ensure IK target sites exist."""
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

    # IK target sites use blue
    ik_site_color = "0 0 1 0.9"

    # Tip site offsets
    tip_site_specs = {
        "thumb": ("right_thumb_dp", np.array([0.0, 0.0, 0.025])),
        "index": ("right_index_ip", np.array([0.0, 0.0, 0.035])),
        "middle": ("right_middle_ip", np.array([0.0, 0.0, 0.037])),
        "ring": ("right_ring_ip", np.array([0.0, 0.0, 0.036])),
        "pinky": ("right_pinky_ip", np.array([0.0, 0.0, 0.032])),
    }

    if "tip" in target_joint_types:
        for finger, (parent_body, offset) in tip_site_specs.items():
            body_elem = root.find(f".//body[@name='{parent_body}']")
            if body_elem is None:
                continue
            site_name = f"right_{finger}_tip_site"
            existing_site = body_elem.find(f"./site[@name='{site_name}']")
            if existing_site is not None:
                existing_site.set("type", "sphere")
                existing_site.set("size", "0.005")
                existing_site.set("rgba", ik_site_color)
            else:
                site = ET.SubElement(body_elem, "site")
                site.set("name", site_name)
                site.set("type", "sphere")
                site.set("pos", " ".join(f"{value:.5f}" for value in offset))
                site.set("size", "0.005")
                site.set("rgba", ik_site_color)

    # Add visual markers for IK target bodies
    allowed_types = set(target_joint_types)
    for finger_name, bodies in FINGER_TARGET_BODIES.items():
        for joint_type, (frame_type, frame_name) in bodies.items():
            if joint_type not in allowed_types:
                continue

            if frame_type == "body":
                body_elem = root.find(f".//body[@name='{frame_name}']")
                if body_elem is not None:
                    marker_site_name = f"ik_marker_{frame_name}"
                    existing_marker = body_elem.find(f"./site[@name='{marker_site_name}']")
                    if existing_marker is not None:
                        existing_marker.set("size", "0.004")
                        existing_marker.set("rgba", ik_site_color)
                    else:
                        site = ET.SubElement(body_elem, "site")
                        site.set("name", marker_site_name)
                        site.set("pos", "0 0 0")
                        site.set("size", "0.004")
                        site.set("rgba", ik_site_color)

    # Add mocap bodies for tracker sites (red)
    tracker_color = "1 0 0 0.9"
    # Add wrist tracker body
    wrist_body = ET.SubElement(worldbody, "body")
    wrist_body.set("name", "tracker_wrist")
    wrist_body.set("mocap", "true")
    wrist_body.set("pos", "0 0 0")
    wrist_geom = ET.SubElement(wrist_body, "geom")
    wrist_geom.set("type", "sphere")
    wrist_geom.set("size", "0.005")
    wrist_geom.set("rgba", tracker_color)
    wrist_geom.set("contype", "0")
    wrist_geom.set("conaffinity", "0")

    for finger_name, mp_mapping in MP_LANDMARK_INDICES.items():
        for joint_type, mp_idx in mp_mapping.items():
            if joint_type not in allowed_types:
                continue

            mocap_body = ET.SubElement(worldbody, "body")
            mocap_body.set("name", f"tracker_{finger_name}_{joint_type}")
            mocap_body.set("mocap", "true")
            mocap_body.set("pos", "0 0 0")

            geom = ET.SubElement(mocap_body, "geom")
            geom.set("type", "sphere")
            geom.set("size", "0.005")
            geom.set("rgba", tracker_color)
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    # Add skeleton lines (cylinders) between joints for skeleton overlay
    skeleton_color = "0 1 0 0.8"  # Green skeleton lines
    skeleton_connections = [
        # Thumb chain
        ("tracker_thumb_mcp", "tracker_thumb_ip"),
        ("tracker_thumb_ip", "tracker_thumb_tip"),
        # Index chain
        ("tracker_index_mcp", "tracker_index_pip"),
        ("tracker_index_pip", "tracker_index_tip"),
        # Middle chain
        ("tracker_middle_mcp", "tracker_middle_pip"),
        ("tracker_middle_pip", "tracker_middle_tip"),
        # Ring chain
        ("tracker_ring_mcp", "tracker_ring_pip"),
        ("tracker_ring_pip", "tracker_ring_tip"),
        # Pinky chain
        ("tracker_pinky_mcp", "tracker_pinky_pip"),
        ("tracker_pinky_pip", "tracker_pinky_tip"),
    ]

    # Add wrist connections (from wrist to each finger MCP)
    wrist_connections = [
        ("tracker_index_mcp", "tracker_wrist"),
        ("tracker_middle_mcp", "tracker_wrist"),
        ("tracker_ring_mcp", "tracker_wrist"),
        ("tracker_pinky_mcp", "tracker_wrist"),
        ("tracker_thumb_mcp", "tracker_wrist"),
    ]

    # Create skeleton line geoms (will be positioned dynamically)
    for start_joint, end_joint in skeleton_connections + wrist_connections:
        # Create a mocap body for the skeleton line
        line_body = ET.SubElement(worldbody, "body")
        line_body.set("name", f"skeleton_line_{start_joint}_{end_joint}")
        line_body.set("mocap", "true")
        line_body.set("pos", "0 0 0")

        # Add cylinder geom for the line
        line_geom = ET.SubElement(line_body, "geom")
        line_geom.set("type", "cylinder")
        line_geom.set("size", "0.001 0.02")  # Thin cylinder (radius, half-height) - will be updated dynamically
        line_geom.set("rgba", skeleton_color)
        line_geom.set("contype", "0")
        line_geom.set("conaffinity", "0")

    return ET.tostring(root, encoding="unicode")


def hand_structure_to_landmarks(structure: HandStructure) -> np.ndarray:
    """Convert HandStructure to 21x3 landmarks array (MediaPipe format)."""
    landmarks = np.zeros((21, 3))

    landmarks[0] = structure.wrist_position
    landmarks[1] = structure.thumb.mcp
    landmarks[2] = structure.thumb.mcp
    landmarks[3] = structure.thumb.ip if structure.thumb.ip is not None else structure.thumb.mcp
    landmarks[4] = structure.thumb.tip
    landmarks[5] = structure.index.mcp
    landmarks[6] = structure.index.pip if structure.index.pip is not None else structure.index.mcp
    landmarks[7] = structure.index.dip if structure.index.dip is not None else structure.index.tip
    landmarks[8] = structure.index.tip
    landmarks[9] = structure.middle.mcp
    landmarks[10] = structure.middle.pip if structure.middle.pip is not None else structure.middle.mcp
    landmarks[11] = structure.middle.dip if structure.middle.dip is not None else structure.middle.tip
    landmarks[12] = structure.middle.tip
    landmarks[13] = structure.ring.mcp
    landmarks[14] = structure.ring.pip if structure.ring.pip is not None else structure.ring.mcp
    landmarks[15] = structure.ring.dip if structure.ring.dip is not None else structure.ring.tip
    landmarks[16] = structure.ring.tip
    landmarks[17] = structure.pinky.mcp
    landmarks[18] = structure.pinky.pip if structure.pinky.pip is not None else structure.pinky.mcp
    landmarks[19] = structure.pinky.dip if structure.pinky.dip is not None else structure.pinky.tip
    landmarks[20] = structure.pinky.tip

    return landmarks


def get_joint_position_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    hand_structure: HandStructure,
    ik_solver: ORCAHandIKRetargeting,
    target_joint_types: tuple[str, ...],
) -> np.ndarray | None:
    """Get world position of a joint by name."""
    # Check if it's a mocap body we can query directly
    if joint_name.startswith("tracker_"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint_name)
        if body_id >= 0:
            mocap_idx = model.body_mocapid[body_id]
            if mocap_idx >= 0 and mocap_idx < model.nmocap:
                return data.mocap_pos[mocap_idx].copy()

    # Fallback: calculate from hand structure
    if joint_name in ("wrist", "tracker_wrist"):
        t_palm = ik_solver.configuration.get_transform_frame_to_world("right_palm", "body")
        p_palm = t_palm.translation()
        r_palm = t_palm.rotation().as_matrix()
        wrist_offset_world = r_palm @ ik_solver.config.wrist_offset_palm
        return p_palm + wrist_offset_world

    return None


def update_skeleton_lines(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_structure: HandStructure,
    target_joint_types: tuple[str, ...],
    ik_solver: ORCAHandIKRetargeting,
) -> None:
    """Update skeleton line positions and orientations between joints."""
    skeleton_connections = [
        ("tracker_thumb_mcp", "tracker_thumb_ip"),
        ("tracker_thumb_ip", "tracker_thumb_tip"),
        ("tracker_index_mcp", "tracker_index_pip"),
        ("tracker_index_pip", "tracker_index_tip"),
        ("tracker_middle_mcp", "tracker_middle_pip"),
        ("tracker_middle_pip", "tracker_middle_tip"),
        ("tracker_ring_mcp", "tracker_ring_pip"),
        ("tracker_ring_pip", "tracker_ring_tip"),
        ("tracker_pinky_mcp", "tracker_pinky_pip"),
        ("tracker_pinky_pip", "tracker_pinky_tip"),
        ("tracker_index_mcp", "tracker_wrist"),
        ("tracker_middle_mcp", "tracker_wrist"),
        ("tracker_ring_mcp", "tracker_wrist"),
        ("tracker_pinky_mcp", "tracker_wrist"),
        ("tracker_thumb_mcp", "tracker_wrist"),
    ]

    for start_joint, end_joint in skeleton_connections:
        p1 = get_joint_position_world(model, data, start_joint, hand_structure, ik_solver, target_joint_types)
        p2 = get_joint_position_world(model, data, end_joint, hand_structure, ik_solver, target_joint_types)

        if p1 is None or p2 is None:
            continue

        # Calculate midpoint and direction
        midpoint = (p1 + p2) / 2.0
        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length < 1e-6:
            continue

        direction = direction / length

        # Create rotation quaternion to align cylinder with direction
        # Default cylinder axis is [0, 0, 1], we want it along direction
        z_axis = np.array([0, 0, 1])
        if np.abs(np.dot(direction, z_axis)) > 0.99:
            # Parallel to z-axis, use identity
            quat = np.array([1, 0, 0, 0])
        else:
            # Calculate rotation axis and angle
            rot_axis = np.cross(z_axis, direction)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            cos_angle = np.dot(z_axis, direction)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            quat = np.array([
                np.cos(angle / 2),
                rot_axis[0] * np.sin(angle / 2),
                rot_axis[1] * np.sin(angle / 2),
                rot_axis[2] * np.sin(angle / 2),
            ])

        # Update mocap body position and orientation
        line_body_name = f"skeleton_line_{start_joint}_{end_joint}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, line_body_name)
        if body_id >= 0:
            mocap_idx = model.body_mocapid[body_id]
            if mocap_idx >= 0 and mocap_idx < model.nmocap:
                data.mocap_pos[mocap_idx] = midpoint
                data.mocap_quat[mocap_idx] = quat


def update_tracker_sites(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_structure: HandStructure,
    target_joint_types: tuple[str, ...],
    ik_solver: ORCAHandIKRetargeting,
) -> None:
    """Update mocap body positions for tracker sites based on hand structure."""
    landmarks_hand_frame = hand_structure_to_landmarks(hand_structure)

    t_palm = ik_solver.configuration.get_transform_frame_to_world("right_palm", "body")
    p_palm = t_palm.translation()
    r_palm = t_palm.rotation().as_matrix()

    wrist_offset_world = r_palm @ ik_solver.config.wrist_offset_palm
    p_wrist = p_palm + wrist_offset_world

    scale = ik_solver.config.scale_factor
    coord_transform = ik_solver.config.coord_transform

    allowed_types = set(target_joint_types)
    for finger_name, mp_mapping in MP_LANDMARK_INDICES.items():
        for joint_type, mp_idx in mp_mapping.items():
            if joint_type not in allowed_types:
                continue

            landmark_pos_hand_frame = landmarks_hand_frame[mp_idx]
            scaled_vec = landmark_pos_hand_frame * scale
            target_pos_world = p_wrist + scaled_vec

            rel_vec = target_pos_world - p_wrist
            transformed_vec = coord_transform @ rel_vec
            final_pos = p_wrist + transformed_vec

            mocap_body_name = f"tracker_{finger_name}_{joint_type}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
            if body_id >= 0:
                mocap_idx = model.body_mocapid[body_id]
                if mocap_idx >= 0 and mocap_idx < model.nmocap:
                    data.mocap_pos[mocap_idx] = final_pos

    # Update wrist tracker position
    wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tracker_wrist")
    if wrist_body_id >= 0:
        wrist_mocap_idx = model.body_mocapid[wrist_body_id]
        if wrist_mocap_idx >= 0 and wrist_mocap_idx < model.nmocap:
            data.mocap_pos[wrist_mocap_idx] = p_wrist

    # Update skeleton lines after updating tracker sites
    update_skeleton_lines(model, data, hand_structure, target_joint_types, ik_solver)


def record_camera_video(cap: cv2.VideoCapture, output_path: Path, fps: float = 30.0) -> bool:
    """Record video from camera until 'q' is pressed.

    Returns True if recording was successful, False otherwise.
    """
    print(f"\n{'='*70}")
    print("VIDEO RECORDING")
    print(f"{'='*70}")
    print("Press 'r' or 's' to START recording")
    print("Press 'q' to STOP recording and save")
    print(f"{'='*70}\n")

    # Set camera FPS if possible
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Wait for start key
    recording = False
    out = None
    frame_count = 0
    start_time = None
    frame_time = 1.0 / fps  # Time per frame in seconds
    target_time = time.time() + frame_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            return False

        display_frame = frame.copy()
        if not recording:
            cv2.putText(
                display_frame,
                "Press 'r' or 's' to START recording",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            elapsed = time.time() - start_time
            cv2.putText(
                display_frame,
                f"RECORDING... {elapsed:.1f}s ({frame_count} frames) - Press 'q' to STOP",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Camera Feed - Recording", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") and recording:
            break
        elif (key == ord("r") or key == ord("s")) and not recording:
            recording = True
            # Get frame dimensions
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"Error: Failed to open video writer for {output_path}")
                cv2.destroyAllWindows()
                return False
            start_time = time.time()
            print(f"Recording started at {fps} FPS. Saving to: {output_path}")

        if recording and out is not None:
            # Write frame and control timing
            out.write(frame)
            frame_count += 1

            # Sleep to maintain target FPS
            sleep_time = max(0, target_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)
            target_time += frame_time


    if out is not None:
        out.release()
        elapsed = time.time() - start_time if start_time else 0
        print(f"Recording stopped. Saved {frame_count} frames ({elapsed:.2f}s) to {output_path}")
        print(f"Actual FPS: {frame_count / elapsed:.2f} (target: {fps:.2f})")

    cv2.destroyAllWindows()
    return recording


def process_video_with_hamer(
    video_path: Path,
    tracker: HaMeRTracker,
    ik_solver: ORCAHandIKRetargeting,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_joint_types: tuple[str, ...],
) -> list[tuple[np.ndarray, Optional[HandStructure]]]:
    """Process video with HAMER and return frames with hand structures."""
    print(f"\n{'='*70}")
    print("PROCESSING VIDEO WITH HAMER")
    print(f"{'='*70}")
    print(f"Processing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {frame_count} frames at {fps} FPS")

    processed_frames = []
    timestamp = 0.0
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use previous frame's solution as warm start for better IK convergence
        # Only reset on first frame
        if frame_idx == 0:
            mujoco.mj_resetData(model, data)

        timestamp = frame_idx * dt
        hand_structures = tracker.detect_hands(frame, timestamp=timestamp)

        hand_structure = hand_structures[0] if hand_structures else None

        if hand_structure:
            # Update IK solver with current state (warm start from previous frame)
            mujoco.mj_forward(model, data)
            ik_solver.configuration.update(data.qpos)

            # Solve IK
            qpos = ik_solver.solve(hand_structure)
            if not np.any(np.isnan(qpos)) and not np.any(np.isinf(qpos)):
                data.qpos[:] = qpos
                data.qvel[:] = 0

            # Update tracker site positions
            update_tracker_sites(model, data, hand_structure, target_joint_types, ik_solver)

        # Forward kinematics
        mujoco.mj_forward(model, data)

        processed_frames.append((frame.copy(), hand_structure))
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    print(f"Processing complete. Processed {len(processed_frames)} frames.")
    return processed_frames


def record_mujoco_visualization(
    processed_frames: list[tuple[np.ndarray, Optional[HandStructure]]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    ik_solver: Optional[ORCAHandIKRetargeting] = None,
    target_joint_types: tuple[str, ...] = ("tip",),
) -> None:
    """Record MuJoCo visualization from processed frames."""
    print(f"\n{'='*70}")
    print("RECORDING MUJOCO VISUALIZATION")
    print(f"{'='*70}")
    print(f"Recording {len(processed_frames)} frames to: {output_path}")

    # Create renderer
    # The model should have framebuffer settings in MJCF for larger resolutions
    try:
        renderer = mujoco.Renderer(model, height=height, width=width, max_geom=10000)
    except ValueError as e:
        # Fallback to smaller resolution if framebuffer is too small
        print(f"Warning: {e}")
        print("Falling back to 640x480 resolution")
        width = 640
        height = 480
        renderer = mujoco.Renderer(model, height=height, width=width, max_geom=10000)

    # Setup camera
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.6  # Further back to fit entire hand
    cam.azimuth = -20
    cam.elevation = 35
    cam.lookat[:] = [0.04, 0.0, 0.08]  # Look at palm area

    # Setup options - enable transparency and joint visualization for skeleton overlay
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Make meshes transparent
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True  # Show joints for skeleton overlay

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process each frame
    for frame_idx, (frame, hand_structure) in enumerate(processed_frames):
        # Only reset on first frame
        if frame_idx == 0:
            mujoco.mj_resetData(model, data)

        if hand_structure is not None and ik_solver is not None:
            # Update IK solver configuration
            mujoco.mj_forward(model, data)
            ik_solver.configuration.update(data.qpos)

            # Solve IK
            qpos = ik_solver.solve(hand_structure)
            if not np.any(np.isnan(qpos)) and not np.any(np.isinf(qpos)):
                data.qpos[:] = qpos
                data.qvel[:] = 0

            # Update tracker site positions
            update_tracker_sites(model, data, hand_structure, target_joint_types, ik_solver)

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Update scene with current state
        renderer.update_scene(data, camera=cam, scene_option=option)

        # Render to numpy array (returns RGB, shape: (height, width, 3))
        rgb_frame = renderer.render()

        # Convert RGB to BGR for OpenCV
        # Note: MuJoCo Renderer already returns correct orientation, no need to flip
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        out.write(bgr_frame)

        if (frame_idx + 1) % 10 == 0:
            print(f"Rendered {frame_idx + 1}/{len(processed_frames)} frames...")

    out.release()
    renderer.close()
    print(f"MuJoCo visualization recording complete. Saved to: {output_path}")


def concatenate_videos(video1_path: Path, video2_path: Path, output_path: Path) -> None:
    """Concatenate two videos side by side."""
    print(f"\n{'='*70}")
    print("CONCATENATING VIDEOS")
    print(f"{'='*70}")
    print(f"Concatenating {video1_path} and {video2_path}")
    print(f"Output: {output_path}")

    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))

    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Failed to open one or both videos")

    # Get properties
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    fps = min(fps1, fps2) if fps1 > 0 and fps2 > 0 else 30.0

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use the maximum height and sum of widths for side-by-side
    max_height = max(height1, height2)
    total_width = width1 + width2

    # Resize videos to same height if needed
    if height1 != height2:
        scale1 = max_height / height1
        scale2 = max_height / height2
        width1 = int(width1 * scale1)
        width2 = int(width2 * scale2)
        total_width = width1 + width2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (total_width, max_height))

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Resize if needed
        if height1 != max_height:
            frame1 = cv2.resize(frame1, (width1, max_height))
        if height2 != max_height:
            frame2 = cv2.resize(frame2, (width2, max_height))

        # Concatenate side by side
        concatenated = np.hstack([frame1, frame2])
        out.write(concatenated)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Concatenated {frame_count} frames...")

    cap1.release()
    cap2.release()
    out.release()
    print(f"Concatenation complete. Saved {frame_count} frames to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record video, process with HAMER, visualize in MuJoCo, and concatenate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--model",
        type=str,
        default="models/orca_hand_fixed.mjcf",
        help="Path to MuJoCo MJCF model file",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="tip",
        help="Comma-separated joint targets (tip,ip,pip,mcp)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Hand scale factor (robot/human)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="recordings",
        help="Output directory for videos",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to input video file (if provided, skips camera recording and uses this video)",
    )
    args = parser.parse_args()

    # Parse target joint types
    raw_targets = [part.strip().lower() for part in args.targets.split(",")]
    target_joints = tuple(dict.fromkeys(jt for jt in raw_targets if jt))
    if not target_joints:
        target_joints = ("tip",)
    supported_targets = {"tip", "ip", "pip", "mcp"}
    invalid = [jt for jt in target_joints if jt not in supported_targets]
    if invalid:
        parser.error(f"Unsupported target joint types: {', '.join(invalid)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    timestamp = int(time.time())
    if args.video_path:
        # Use input video name as base for output files
        input_video_name = Path(args.video_path).stem
        camera_video_path = Path(args.video_path)  # Use input video directly
        mujoco_video_path = output_dir / f"mujoco_{input_video_name}_{timestamp}.mp4"
        final_video_path = output_dir / f"concatenated_{input_video_name}_{timestamp}.mp4"
    else:
        camera_video_path = output_dir / f"camera_recording_{timestamp}.mp4"
        mujoco_video_path = output_dir / f"mujoco_recording_{timestamp}.mp4"
        final_video_path = output_dir / f"concatenated_{timestamp}.mp4"

    # Load model
    model_file = Path(args.model)
    if not model_file.is_absolute():
        model_file = Path(__file__).parent.parent / args.model
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading model from {model_file}")
    xml_string = inject_visualization_bodies(model_file, target_joints)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Initialize tracker and IK solver
    tracker = HaMeRTracker()
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

    # Step 1: Record camera video or use provided video file
    if args.video_path:
        camera_video_path = Path(args.video_path)
        if not camera_video_path.exists():
            raise FileNotFoundError(f"Video file not found: {camera_video_path}")
        print(f"Using existing video: {camera_video_path}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {args.camera}")

        if not record_camera_video(cap, camera_video_path, args.fps):
            print("Recording cancelled or failed.")
            cap.release()
            return

        cap.release()

    # Step 2: Process video with HAMER
    processed_frames = process_video_with_hamer(
        camera_video_path,
        tracker,
        ik_solver,
        model,
        data,
        target_joints,
    )

    # Get video FPS (use input video FPS if provided, otherwise use args.fps)
    if args.video_path:
        cap = cv2.VideoCapture(str(camera_video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if video_fps > 0:
            fps = video_fps
        else:
            fps = args.fps
    else:
        fps = args.fps

    # Step 3: Record MuJoCo visualization
    record_mujoco_visualization(
        processed_frames,
        model,
        data,
        mujoco_video_path,
        fps,
        ik_solver=ik_solver,
        target_joint_types=target_joints,
    )

    # Step 4: Concatenate videos
    concatenate_videos(camera_video_path, mujoco_video_path, final_video_path)

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}")
    print(f"Camera video: {camera_video_path}")
    print(f"MuJoCo video: {mujoco_video_path}")
    print(f"Concatenated video: {final_video_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
