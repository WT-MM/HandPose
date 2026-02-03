"""Visualize tracker sites and IK target sites on ORCA hand model.

This script overlays:
1. Tracker sites (from MediaPipe/HaMeR hand tracking) - shown as colored spheres
2. IK target sites (on ORCA hand model) - shown as different colored spheres

Both are color-coded and labeled for easy comparison.
"""

import argparse
import multiprocessing as mp
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from queue import Empty
from typing import Protocol

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
from handpose.tracker.mediapipe import MediaPipeTracker


def inject_visualization_bodies(mjcf_path: Path, target_joint_types: tuple[str, ...]) -> str:
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


def hand_structure_to_landmarks(structure: HandStructure) -> np.ndarray:
    """Convert HandStructure to 21x3 landmarks array (MediaPipe format).

    Returns landmarks in hand frame (wrist at origin).
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


def update_tracker_sites(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_structure: HandStructure,
    target_joint_types: tuple[str, ...],
    ik_solver: ORCAHandIKRetargeting,
) -> None:
    """Update mocap body positions for tracker sites based on hand structure.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        hand_structure: Tracked hand structure
        target_joint_types: Joint types to visualize
        ik_solver: IK solver (for coordinate transformation)
    """
    landmarks_hand_frame = hand_structure_to_landmarks(hand_structure)

    # Get palm transform to convert hand frame to world frame
    t_palm = ik_solver.configuration.get_transform_frame_to_world("right_palm", "body")
    p_palm = t_palm.translation()
    r_palm = t_palm.rotation().as_matrix()

    # Wrist offset in palm frame
    wrist_offset_world = r_palm @ ik_solver.config.wrist_offset_palm
    p_wrist = p_palm + wrist_offset_world

    # Scale factor
    scale = ik_solver.config.scale_factor

    # Coordinate transformation
    coord_transform = ik_solver.config.coord_transform

    allowed_types = set(target_joint_types)
    for finger_name, mp_mapping in MP_LANDMARK_INDICES.items():
        for joint_type, mp_idx in mp_mapping.items():
            if joint_type not in allowed_types:
                continue

            # Get landmark position in hand frame
            landmark_pos_hand_frame = landmarks_hand_frame[mp_idx]

            # Scale to robot size
            scaled_vec = landmark_pos_hand_frame * scale

            # Transform to world frame (anchor to wrist)
            target_pos_world = p_wrist + scaled_vec

            # Apply coordinate transformation
            rel_vec = target_pos_world - p_wrist
            transformed_vec = coord_transform @ rel_vec
            final_pos = p_wrist + transformed_vec

            # Update mocap body position
            mocap_body_name = f"tracker_{finger_name}_{joint_type}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
            if body_id >= 0:
                # Get the mocap index for this body (mocap bodies have mocapid >= 0)
                mocap_idx = model.body_mocapid[body_id]
                if mocap_idx >= 0 and mocap_idx < model.nmocap:
                    data.mocap_pos[mocap_idx] = final_pos


def visualize_ik_target_sites(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_joint_types: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Get positions of IK target sites on ORCA hand.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_joint_types: Joint types to visualize

    Returns:
        Dictionary mapping finger names to joint_type -> position
    """
    ik_positions = {}
    allowed_types = set(target_joint_types)

    for finger_name, bodies in FINGER_TARGET_BODIES.items():
        finger_positions = {}
        for joint_type, (frame_type, frame_name) in bodies.items():
            if joint_type not in allowed_types:
                continue

            obj_type = mujoco.mjtObj.mjOBJ_BODY if frame_type == "body" else mujoco.mjtObj.mjOBJ_SITE
            frame_id = mujoco.mj_name2id(model, obj_type, frame_name)

            if frame_id >= 0:
                if obj_type == mujoco.mjtObj.mjOBJ_SITE:
                    pos = data.site(frame_id).xpos.copy()
                else:
                    pos = data.body(frame_id).xpos.copy()
                finger_positions[joint_type] = pos

        if finger_positions:
            ik_positions[finger_name] = finger_positions

    return ik_positions


class Flag(Protocol):
    value: int


def display_feed_process(frame_queue: mp.Queue, running_flag: Flag, window_name: str) -> None:
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

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running_flag.value = 0
            break

    cv2.destroyAllWindows()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize tracker sites and IK target sites on ORCA hand",
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
        "--tracker",
        type=str,
        default="hamer",
        choices=["hamer", "mediapipe"],
        help="Hand tracking backend",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="tip, ip",
        help="Comma-separated joint targets (tip,ip,pip,mcp). Default: tip, ip",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Hand scale factor (robot/human)",
    )
    parser.add_argument(
        "--show-feed",
        action="store_true",
        help="Show camera feed with hand tracking overlay",
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

    # Initialize tracker
    if args.tracker == "hamer":
        tracker = HaMeRTracker()
    else:
        tracker = MediaPipeTracker()

    # Initialize IK solver
    ik_config = ORCAHandIKConfig(
        scale_factor=args.scale,
        target_joint_types=target_joints,
        lm_damping=0.5
    )
    ik_solver = ORCAHandIKRetargeting(model, config=ik_config)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera}")

    print("\n" + "=" * 70)
    print("VISUALIZATION GUIDE:")
    print("=" * 70)
    print("TRACKER SITES (larger RED spheres - actual tracked hand from camera):")
    print("  Red spheres = All tracker sites (thumb, index, middle, ring, pinky)")
    print("               These represent the actual human hand position")
    print("\nIK TARGET SITES (smaller BLUE spheres - on ORCA hand model):")
    print("  Blue spheres = All IK target sites (thumb, index, middle, ring, pinky)")
    print("                These represent where the robot hand should move")
    print("\nNOTE: RED spheres (tracker) should align with BLUE spheres (IK targets)")
    print("      when IK is solving correctly. RED = actual hand, BLUE = robot targets.")
    print("\nPress 'q' in MuJoCo viewer to quit")
    print("=" * 70 + "\n")

    start_time = time.time()

    # Set up multiprocessing for display window if requested
    display_process: mp.Process | None = None
    frame_queue: mp.Queue | None = None
    running_flag: Flag | None = None

    if args.show_feed:
        print("Starting camera feed display in separate process...")
        frame_queue = mp.Queue(maxsize=2)  # Small buffer to avoid lag
        running_flag = mp.Value("i", 1)
        display_process = mp.Process(
            target=display_feed_process,
            args=(frame_queue, running_flag, "Hand Tracking Feed"),
        )
        display_process.start()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = time.time() - start_time
                hand_structures = tracker.detect_hands(frame, timestamp=timestamp)

                # Send frame to display process if enabled
                if args.show_feed and frame_queue is not None and running_flag is not None:
                    if hand_structures:
                        vis_frame = tracker.visualize(frame, hand_structures)
                    else:
                        vis_frame = frame.copy()
                    try:
                        frame_queue.put_nowait(vis_frame)
                    except Exception:
                        pass  # Queue full, skip this frame

                if hand_structures:
                    hand = hand_structures[0]

                    # Update IK solver configuration
                    mujoco.mj_forward(model, data)
                    ik_solver.configuration.update(data.qpos)

                    # Solve IK
                    qpos = ik_solver.solve(hand)
                    if not np.any(np.isnan(qpos)) and not np.any(np.isinf(qpos)):
                        data.qpos[:] = qpos
                        data.qvel[:] = 0

                    # Update tracker site positions
                    update_tracker_sites(model, data, hand, target_joints, ik_solver)

                # Forward kinematics to update IK target site positions
                mujoco.mj_forward(model, data)

                # Get IK target positions (sites are automatically rendered by MuJoCo)
                ik_positions = visualize_ik_target_sites(model, data, target_joints)

                # Print current positions for debugging (first frame only)
                if hand_structures:
                    print("\nCurrent IK Target Positions:")
                    for finger, positions in ik_positions.items():
                        for joint_type, pos in positions.items():
                            print(f"  {finger}_{joint_type}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

                # Sync viewer
                viewer.sync()

    finally:
        # Clean up display process
        if running_flag is not None:
            running_flag.value = 0
        if display_process is not None:
            display_process.join(timeout=1.0)
            if display_process.is_alive():
                display_process.terminate()
        cap.release()
        print("\nVisualization closed.")


if __name__ == "__main__":
    main()
