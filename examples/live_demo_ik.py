"""Live IK Retargeting Demo using Mink."""

import argparse
import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from askin import KeyboardController

from handpose import HandTracker, ORCAHandIKRetargeting
from handpose.ik_retargeting import FINGER_TARGET_BODIES, ORCAHandIKConfig


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
        for joint_type, body_name in joints.items():
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

    return ET.tostring(root, encoding="unicode")


async def main_async(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cap: cv2.VideoCapture,
    tracker: HandTracker,
    ik_solver: ORCAHandIKRetargeting,
    target_body_ids: dict[str, dict[str, int]],
) -> None:
    """Async main loop with keyboard handling."""
    import time

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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while running and viewer.is_running() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = time.time() - start_time
            hand_poses = tracker.detect_hands(frame, timestamp=timestamp)

            if hand_poses:
                pose = hand_poses[0]
                landmarks_hand = tracker.landmarks_in_hand_frame(pose)

                # --- IK SOLVE ---
                # 1. Update the configuration object with current robot state
                mujoco.mj_forward(model, data)
                ik_solver.configuration.update(data.qpos)

                # 2. Solve for new qpos
                target_q = ik_solver.solve(landmarks_hand)

                # 3. Apply to simulation
                data.qpos[:] = target_q

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

                # Draw landmarks on frame
                tracker.draw_landmarks(frame, [pose])

            mujoco.mj_step(model, data)
            viewer.sync()

            try:
                cv2.imshow("Tracker", frame)
            except (cv2.error, Exception):
                pass

            await asyncio.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="orca_hand.mjcf")
    parser.add_argument("--scale", type=float, default=1.3, help="Robot/Human hand scale factor")
    args = parser.parse_args()

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

    print("Initializing Hand Tracker...")
    tracker = HandTracker(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    print("Initializing IK Solver (Mink)...")

    ik_config = ORCAHandIKConfig(
        scale_factor=args.scale,
        wrist_offset_palm=np.array([0.000, 0.0, -0.05]),
    )

    ik_solver = ORCAHandIKRetargeting(model, config=ik_config)

    # 5. Helper to map finger names and joint types to mocap body IDs
    target_body_ids: dict[str, dict[str, int]] = {}
    for finger, joints in FINGER_TARGET_BODIES.items():
        target_body_ids[finger] = {}
        for joint_type in joints.keys():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target_{finger}_{joint_type}")
            target_body_ids[finger][joint_type] = bid

    # 6. Main Loop
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nStarting simulation...")
    print("Press 'q' to quit (or close viewer).")

    # Run async main loop
    asyncio.run(main_async(model, data, cap, tracker, ik_solver, target_body_ids))

    cap.release()
    cv2.destroyAllWindows()


"""
Run with `mjpython examples/live_demo_ik.py --model orca_hand_fixed.mjcf --scale 1.0`
"""
if __name__ == "__main__":
    main()
