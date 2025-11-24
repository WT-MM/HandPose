"""Live IK Retargeting Demo using Mink."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from handpose import FINGERTIP_BODIES, HandTracker, ORCAHandIKRetargeting


def inject_target_bodies(mjcf_path: Path) -> str:
    """Injects mocap bodies into the MJCF XML string for visualization.
    Converts relative paths to absolute paths so MuJoCo can find assets.
    Returns the modified XML string.
    """
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

    # Add a mocap body for each finger target
    for finger in FINGERTIP_BODIES.keys():
        body = ET.SubElement(worldbody, "body")
        body.set("name", f"target_{finger}")
        body.set("mocap", "true")
        body.set("pos", "0 0 0")

        # Add a visual sphere
        geom = ET.SubElement(body, "geom")
        geom.set("type", "sphere")
        geom.set("size", "0.01")  # 1cm radius
        geom.set("rgba", "1 0 0 0.6")  # Red, semi-transparent
        geom.set("contype", "0")  # No collision
        geom.set("conaffinity", "0")

    return ET.tostring(root, encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="models/orca_hand.mjcf")
    parser.add_argument("--scale", type=float, default=1.3, help="Robot/Human hand scale factor")
    args = parser.parse_args()

    # 1. Setup paths and model
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model if not Path(args.model).is_absolute() else Path(args.model)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # 2. Inject visualization bodies (Mocap)
    print("Injecting visualization targets...")
    xml_string = inject_target_bodies(model_path)

    # 3. Load MuJoCo Model
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # 4. Initialize Tracker and IK
    print("Initializing Hand Tracker...")
    tracker = HandTracker(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    print("Initializing IK Solver (Mink)...")
    # Note: scale=1.0 means robot hand is same size as human.
    # ORCA is quite large, so 1.0-1.3 is often appropriate depending on the user.
    ik_solver = ORCAHandIKRetargeting(model, scale_factor=args.scale)

    # 5. Helper to map finger names to mocap body IDs
    target_body_ids = {}
    for finger in FINGERTIP_BODIES.keys():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"target_{finger}")
        target_body_ids[finger] = bid

    # 6. Main Loop
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nStarting simulation...")
    print("Press 'q' in OpenCV window to quit (or close viewer).")

    import time
    start_time = time.time()
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- VISION ---
            # MediaPipe requires monotonically increasing timestamps
            timestamp = time.time() - start_time
            hand_poses = tracker.detect_hands(frame, timestamp=timestamp)

            if hand_poses:
                pose = hand_poses[0]
                landmarks_hand = tracker.landmarks_in_hand_frame(pose)

                # --- IK SOLVE ---
                # 1. Update the configuration object with current robot state
                # Need to call mj_forward first to ensure kinematics are up to date
                mujoco.mj_forward(model, data)
                ik_solver.configuration.update(data.qpos)

                # 2. Solve for new qpos
                target_q = ik_solver.solve(landmarks_hand)

                # 3. Apply to simulation
                # In a real robot, we would send velocity/position commands.
                # In sim, we can just set qpos directly for "teleportation" control
                # or use a PD controller. For immediate viz, setting qpos is snappiest.
                data.qpos[:] = target_q

                # --- VISUALIZATION ---
                # Update the mocap bodies to match the IK targets so we can verify alignment
                # We need to get the targets calculated inside the solver
                # (Recalculating here for visualization simplicity, though slightly inefficient)
                targets = ik_solver.compute_target_positions(landmarks_hand)

                # Get Palm transform to apply the same rotation we did in IK
                # (This replicates the logic in ik_retargeting.py to show exactly where IK is aiming)
                palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
                p_palm = data.xpos[palm_id]

                for finger, target_pos_mp in targets.items():
                    # Replicate rotation logic from ik_retargeting.py
                    rel = target_pos_mp - p_palm
                    # Rotation: MP(x,y,z) -> Robot(z, -y, x)
                    rot_vec = np.array([rel[2], -rel[1], rel[0]])
                    final_target = p_palm + rot_vec

                    # Update Mocap body position
                    mocap_id = target_body_ids[finger]
                    # Mocap bodies are controlled via data.mocap_pos
                    # We need to find which mocap index corresponds to this body
                    # model.body_mocapid maps body ID to mocap ID
                    mocap_idx = model.body_mocapid[mocap_id]
                    if mocap_idx >= 0:
                        data.mocap_pos[mocap_idx] = final_target

                # Draw landmarks on frame
                tracker.draw_landmarks(frame, [pose])

            # --- PHYSICS & RENDER ---
            mujoco.mj_step(model, data)
            viewer.sync()

            # Show OpenCV frame (handle macOS conflicts gracefully)
            try:
                cv2.imshow("Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except (cv2.error, Exception) as e:
                # OpenCV window not available (macOS + mjpython conflict)
                # Just continue without displaying
                pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""

### How to Run
Make sure you have `mink` installed:
```bash
pip install mink
```

Then run:
```bash
python live_demo_ik.py --model models/orca_hand.mjcf --scale 1.3
```
"""
