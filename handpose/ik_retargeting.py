"""Inverse Kinematics Retargeting using Mink."""

from typing import Dict

import mink
import mujoco
import numpy as np

# Joint names for the ORCA hand
# Note: We exclude the wrist/root joints as we generally want to solve for fingers relative to palm
ORCA_JOINT_NAMES = [
    # Thumb
    "right_thumb_abd",
    "right_thumb_mcp",
    "right_thumb_pip",
    "right_thumb_dip",
    # Index
    "right_index_abd",
    "right_index_mcp",
    "right_index_pip",
    # Middle
    "right_middle_abd",
    "right_middle_mcp",
    "right_middle_pip",
    # Ring
    "right_ring_abd",
    "right_ring_mcp",
    "right_ring_pip",
    # Pinky
    "right_pinky_abd",
    "right_pinky_mcp",
    "right_pinky_pip",
]

# Map finger names to the End-Effector body names in the MJCF
# Based on the provided MJCF, the distal bodies are named:
FINGERTIP_BODIES = {
    "thumb": "right_thumb_dp",
    "index": "right_index_ip",  # ORCA index/middle/ring/pinky end at 'ip' body which contains the tip
    "middle": "right_middle_ip",
    "ring": "right_ring_ip",
    "pinky": "right_pinky_ip",
}


class ORCAHandIKRetargeting:
    def __init__(self, model: mujoco.MjModel, scale_factor: float = 0.8):
        """Args:
        model: The MuJoCo model.
        scale_factor: Ratio of Robot Hand Size / Human Hand Size.
                      Adjust this to match your hand size to the robot.
        """
        self.model = model
        self.configuration = mink.Configuration(model)
        self.scale_factor = scale_factor

        # Identify the qpos indices for the finger joints we want to control
        joint_indices_list = []
        for name in ORCA_JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                # qpos address
                qpos_adr = model.jnt_qposadr[jnt_id]
                joint_indices_list.append(qpos_adr)
        self.joint_indices = np.array(joint_indices_list)

        # Create End Effector tasks for each fingertip
        self.tasks = {}
        for finger_name, body_name in FINGERTIP_BODIES.items():
            # We target position only (3 degrees of freedom per finger task)
            # This allows the finger to rotate naturally to reach the point
            task = mink.FrameTask(
                frame_name=body_name,
                frame_type="body",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,  # Levenberg-Marquardt damping for stability
            )
            self.tasks[finger_name] = task

        # We also need a posture task to encourage the hand to stay close to a "neutral" pose
        # when not reaching for extremes. This prevents weird internal configurations.
        self.posture_task = mink.PostureTask(model, cost=1e-2, lm_damping=1.0)
        # Set a reasonable neutral pose (e.g., slightly curled)
        # You could tune these values to look natural
        self.target_pose = np.zeros(model.nq)
        # Example: slightly curl fingers
        # (This is optional, keeping it 0.0 is fine for a start)
        self.posture_task.set_target(self.target_pose)

    def compute_target_positions(self, landmarks_hand_frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Converts normalized MediaPipe landmarks (relative to wrist) into
        Robot Root Frame target positions using the scale factor.
        """
        # MediaPipe Indices
        WRIST = 0
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20

        # Map MP indices to finger names
        tips = {
            "thumb": THUMB_TIP,
            "index": INDEX_TIP,
            "middle": MIDDLE_TIP,
            "ring": RING_TIP,
            "pinky": PINKY_TIP,
        }

        targets = {}

        # Wrist position in hand frame is usually (0,0,0) or very close
        wrist_pos = landmarks_hand_frame[WRIST]

        for name, tip_idx in tips.items():
            # Vector from Wrist to Tip
            tip_pos = landmarks_hand_frame[tip_idx]
            rel_vec = tip_pos - wrist_pos

            # Scale to robot size
            scaled_vec = rel_vec * self.scale_factor

            # The Robot "Wrist" (right_wrist joint) is at the origin of the 'right_palm' body.
            # However, our IK is solving in World Frame generally.
            # Assuming the Robot Base is at (0,0,0) or we are just setting qpos,
            # we need to define where the "Wrist" is in the robot model.

            # In orca_hand.mjcf, 'right_palm' is offset from 'right_tower'.
            # A simple approach:
            # We want the fingertip *relative to the palm* to match.
            # So Target = Robot_Palm_Pos + Scaled_Vector

            # Let's assume the palm is roughly at the simulation origin or calculate it.
            # For robustness, we will perform IK assuming the palm is fixed at its current location
            # in the configuration.

            # Get current transform of the palm (frame of reference)
            T_palm = self.configuration.get_transform_frame_to_world("right_palm", "body")
            p_palm = T_palm.translation()
            R_palm = T_palm.rotation().as_matrix()

            # Transform the relative vector (which is in a "visual" frame aligned with MP)
            # MediaPipe Hand Frame:
            # +X: Side (Thumb to Pinky approx)
            # +Y: Out of palm (Normal) ??? No, MediaPipe is weird.
            # We did a robust basis computation in tracker.py.
            # If `landmarks_hand_frame` comes from `tracker.py`, it's already aligned:
            # X: Side, Y: Normal, Z: Forward (depending on implementation).

            # Actually, `tracker.py` `_compute_wrist_pose` defines:
            # X: Wrist -> Palm Center (Forward)
            # Y: Out of Palm (Normal)
            # Z: Side (Thumb -> Pinky) -- Wait, standard MP is often different.

            # Let's rely on the visual correlation.
            # If the visualization looks rotated, we apply a rotation matrix here.

            # Standard ORCA Palm:
            # Z-axis usually points UP (along fingers) or Forward.
            # In the MJCF: right_palm pos="...".

            # Simple Identity mapping for start:
            target_pos = p_palm + scaled_vec

            targets[name] = target_pos

        return targets

    def solve(self, landmarks_hand_frame: np.ndarray) -> np.ndarray:
        """Solves IK for the given hand landmarks.
        Returns the full qpos array for the robot.
        """
        # 1. Calculate Target Positions
        targets = self.compute_target_positions(landmarks_hand_frame)

        # 2. Update Tasks
        active_tasks: list[mink.Task] = [self.posture_task]

        for name, task in self.tasks.items():
            if name in targets:
                # We need to rotate the vector to match MuJoCo's coordinate system if needed.
                # MediaPipe (tracker.py):
                #   X: Forward (Wrist -> Fingers)
                #   Y: Up/Normal (Out of palm)
                #   Z: Side (Thumb -> Pinky) (Left handed coord system often)

                # ORCA MuJoCo:
                #   Looking at `right_palm` body:
                #   Usually Z is "up" (along the arm/hand length) or Y is.
                #   Let's map:
                #   MP X (Forward) -> Robot Z (Up/Length)
                #   MP Z (Side)    -> Robot X (Side)
                #   MP Y (Normal)  -> Robot Y (Normal)

                raw_target = targets[name]

                # Get palm center to do relative math again for rotation
                T_palm = self.configuration.get_transform_frame_to_world("right_palm", "body")
                p_palm = T_palm.translation()
                rel = raw_target - p_palm

                # Re-map axes:
                # MP [x, y, z] -> [z, x, y] (Example guess, usually requires tuning)
                # Let's try a direct mapping first, then rotate if the hand looks scrambled.
                # Based on previous `retargeting.py` logic, MP frames were slightly different.

                # Let's try:
                # Robot Z is usually the long axis.
                # Robot X is side.
                # Robot Y is thickness.

                # Input `landmarks_hand_frame` from `tracker.py`:
                # Row 0 (Wrist) is origin (0,0,0).
                # Row 9 (Middle MCP) is roughly (Length, 0, 0) if X is forward.

                mp_vec = rel  # This is (x, y, z) from MP

                # Mapping MP (X-Forward, Y-Normal, Z-Side) to Robot (Z-Up/Forward, X-Side, Y-Normal)
                # target_local = [mp_z, mp_y, mp_x] ?
                # Let's start with a standard permutation:
                # Robot X = MP Z (Side)
                # Robot Y = MP -Y (Normal, maybe flipped)
                # Robot Z = MP X (Forward)

                # Apply rotation
                rot_vec = np.array([mp_vec[2], -mp_vec[1], mp_vec[0]])

                final_target = p_palm + rot_vec

                # FrameTask.set_target expects an SE3 transform
                # We only care about position, so create SE3 from translation
                target_se3 = mink.SE3.from_translation(final_target)
                task.set_target(target_se3)
                active_tasks.append(task)

        # 3. Solve IK
        # Solve for velocity limits (mink uses lie algebra, solving for delta q)
        # We perform one step of IK integration

        # Note: In a live loop, we update the configuration from `data.qpos` each frame.
        # But here we are just calculating the target qpos.

        # Mink solves for dq. q_next = q_curr + dq * dt
        # We can iterate a few times for convergence if the jump is large,
        # but for live tracking, 1 step per frame is usually enough if FPS is high.

        # solve_ik signature: (configuration, tasks, dt, solver, damping)
        # Use smaller dt for smoother convergence (0.01-0.1 is typical)
        # For live tracking, we want small steps to avoid overshooting
        dt = 0.05  # 50ms timestep for smooth IK
        
        # Use 'daqp' if available, otherwise try other available solvers
        solver = "daqp"
        try:
            vel = mink.solve_ik(
                self.configuration, active_tasks, dt, solver, 1e-3
            )
        except Exception:
            # Fallback: try to find any available solver
            import qpsolvers
            available = qpsolvers.available_solvers
            if available and available[0] != "daqp":
                solver = available[0]
            elif len(available) > 1:
                solver = available[1]  # Try second available solver
            else:
                solver = "osqp"  # Default fallback
            try:
                vel = mink.solve_ik(
                    self.configuration, active_tasks, dt, solver, 1e-3
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to solve IK with any available solver. "
                    f"Available: {qpsolvers.available_solvers}, Error: {e}"
                ) from e

        # Integrate velocity to update configuration
        # For live tracking, we can iterate a few times for better convergence
        # but for real-time, one step is usually sufficient
        self.configuration.integrate_inplace(vel, dt)

        # Return the new configuration qpos
        return self.configuration.q
