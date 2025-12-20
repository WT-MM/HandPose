"""Inverse Kinematics Retargeting using Mink."""

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from handpose.tracker.base import HandStructure

# Joint names for the ORCA hand
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

# Map finger names to frames (body or site) used for IK targeting
# Each entry is (frame_type, frame_name)
FINGER_TARGET_BODIES = {
    "thumb": {
        "mcp": ("body", "right_thumb_mp"),
        "pip": ("body", "right_thumb_pp"),
        "ip": ("body", "right_thumb_ip"),
        "tip": ("site", "right_thumb_tip_site"),
    },
    "index": {
        "mcp": ("body", "right_index_mp"),
        "pip": ("body", "right_index_pp"),
        "tip": ("site", "right_index_tip_site"),
    },
    "middle": {
        "mcp": ("body", "right_middle_mp"),
        "pip": ("body", "right_middle_pp"),
        "tip": ("site", "right_middle_tip_site"),
    },
    "ring": {
        "mcp": ("body", "right_ring_mp"),
        "pip": ("body", "right_ring_pp"),
        "tip": ("site", "right_ring_tip_site"),
    },
    "pinky": {
        "mcp": ("body", "right_pinky_mp"),
        "pip": ("body", "right_pinky_pp"),
        "tip": ("site", "right_pinky_tip_site"),
    },
}

# MediaPipe landmark indices for each joint type
MP_LANDMARK_INDICES = {
    "thumb": {
        "mcp": 2,  # THUMB_MCP
        "pip": 2,  # Map robot PIP to MP MCP to avoid over-folding
        "ip": 3,  # THUMB_IP (explicit target to drive the yellow sphere)
        "tip": 4,  # THUMB_TIP
    },
    "index": {
        "mcp": 5,  # INDEX_FINGER_MCP
        "pip": 6,  # INDEX_FINGER_PIP
        "tip": 8,  # INDEX_FINGER_TIP
    },
    "middle": {
        "mcp": 9,  # MIDDLE_FINGER_MCP
        "pip": 10,  # MIDDLE_FINGER_PIP
        "tip": 12,  # MIDDLE_FINGER_TIP
    },
    "ring": {
        "mcp": 13,  # RING_FINGER_MCP
        "pip": 14,  # RING_FINGER_PIP
        "tip": 16,  # RING_FINGER_TIP
    },
    "pinky": {
        "mcp": 17,  # PINKY_MCP
        "pip": 18,  # PINKY_PIP
        "tip": 20,  # PINKY_TIP
    },
}


@dataclass
class ORCAHandIKConfig:
    """Configuration for ORCA hand IK retargeting."""

    # Scale factor: Ratio of Robot Hand Size / Human Hand Size
    scale_factor: float = 1.0

    # Wrist joint offset from palm center (in palm frame, meters)
    # Default from MJCF: right_wrist joint pos="0.002 -0.00144 -0.03872"
    wrist_offset_palm: np.ndarray | None = None

    # IK solver parameters
    dt: float = 0.05  # Timestep for IK integration (seconds)
    damping: float = 1e-2  # Levenberg-Marquardt damping
    solver: str = "daqp"  # QP solver
    ik_iterations: int = 5  # Number of IK passes to perform before returning (for better convergence)

    # Task costs
    position_cost: float = 3.0  # Cost for position tracking
    orientation_cost: float = 0.0  # Cost for orientation tracking
    posture_cost: float = 1e-4  # Cost for posture task (keeps hand near neutral)

    # Coordinate frame transformation
    # MediaPipe to Robot coordinate mapping: [MP_X, MP_Y, MP_Z] -> [Robot_X, Robot_Y, Robot_Z]
    # Default: MP (X=Forward, Y=Normal, Z=Side) -> Robot (Z=Up, X=Side, Y=Normal)
    coord_transform: np.ndarray | None = None
    target_joint_types: tuple[str, ...] = ("tip",)

    def __post_init__(self) -> None:
        """Initialize default values if None."""
        if self.wrist_offset_palm is None:
            # Default from MJCF: right_wrist joint position relative to palm
            self.wrist_offset_palm = np.array([0.002, -0.00144, -0.03872])

        if self.coord_transform is None:
            # Default transformation matrix: MP [x, y, z] -> Robot [z, -y, x]
            # This maps: X(Forward) -> Z(Up), Y(Normal) -> -Y(Normal), Z(Side) -> X(Side)
            self.coord_transform = np.array(
                [
                    [0, 0, 1],  # Robot X = MP Z
                    [0, -1, 0],  # Robot Y = -MP Y
                    [1, 0, 0],  # Robot Z = MP X
                ]
            )

    @classmethod
    def default_config(cls) -> "ORCAHandIKConfig":
        """Create default configuration."""
        return cls()


class ORCAHandIKRetargeting:
    def __init__(
        self,
        model: mujoco.MjModel,
        config: ORCAHandIKConfig = ORCAHandIKConfig.default_config(),
    ) -> None:
        """Initialize IK retargeting for ORCA hand.

        Args:
            model: The MuJoCo model.
            config: IK configuration. If None, uses default config.
        """
        self.model = model
        self.configuration = mink.Configuration(model)

        # Initialize config
        if config is None:
            config = ORCAHandIKConfig.default_config()

        self.config = config

        # Identify the qpos indices for the finger joints we want to control
        joint_indices_list = []
        for name in ORCA_JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                # qpos address
                qpos_adr = model.jnt_qposadr[jnt_id]
                joint_indices_list.append(qpos_adr)
        self.joint_indices = np.array(joint_indices_list)

        # Create IK tasks for multiple keypoints per finger
        self.tasks = {}
        allowed_types = set(self.config.target_joint_types)
        for finger_name, bodies in FINGER_TARGET_BODIES.items():
            finger_tasks = {}
            for joint_type, body_name in bodies.items():
                if joint_type not in allowed_types:
                    continue
                frame_type, frame_name = body_name
                obj_type = mujoco.mjtObj.mjOBJ_BODY if frame_type == "body" else mujoco.mjtObj.mjOBJ_SITE
                if mujoco.mj_name2id(model, obj_type, frame_name) < 0:
                    continue
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type=frame_type,
                    position_cost=self.config.position_cost,
                    orientation_cost=self.config.orientation_cost,
                    lm_damping=self.config.damping,
                )
                finger_tasks[joint_type] = task
            if finger_tasks:
                self.tasks[finger_name] = finger_tasks

        if not self.tasks:
            raise ValueError(
                "No IK tasks were created. Ensure target_joint_types exist in the model "
                "(tip tracking requires fingertip sites to be injected)."
            )

        # We also need a posture task to encourage the hand to stay close to a "neutral" pose
        # when not reaching for extremes. This prevents weird internal configurations.
        self.posture_task = mink.PostureTask(
            model,
            cost=self.config.posture_cost,
            lm_damping=self.config.damping,
        )

        self.target_pose = np.zeros(model.nq)
        self.posture_task.set_target(self.target_pose)

    def _hand_structure_to_landmarks(self, hand_structure: HandStructure) -> np.ndarray:
        """Convert HandStructure to 21x3 landmarks array (MediaPipe format).

        Args:
            hand_structure: HandStructure from tracker

        Returns:
            landmarks_hand_frame: 21x3 array of landmarks in hand frame (wrist at origin)
        """
        landmarks = np.zeros((21, 3))

        # Wrist (index 0)
        landmarks[0] = hand_structure.wrist_position

        # Thumb: CMC(1), MCP(2), IP(3), tip(4)
        landmarks[1] = hand_structure.thumb.mcp  # CMC approximate
        landmarks[2] = hand_structure.thumb.mcp
        landmarks[3] = hand_structure.thumb.ip if hand_structure.thumb.ip is not None else hand_structure.thumb.mcp
        landmarks[4] = hand_structure.thumb.tip

        # Index: MCP(5), PIP(6), DIP(7), tip(8)
        landmarks[5] = hand_structure.index.mcp
        landmarks[6] = hand_structure.index.pip if hand_structure.index.pip is not None else hand_structure.index.mcp
        landmarks[7] = hand_structure.index.dip if hand_structure.index.dip is not None else hand_structure.index.tip
        landmarks[8] = hand_structure.index.tip

        # Middle: MCP(9), PIP(10), DIP(11), tip(12)
        landmarks[9] = hand_structure.middle.mcp
        landmarks[10] = (
            hand_structure.middle.pip if hand_structure.middle.pip is not None else hand_structure.middle.mcp
        )
        landmarks[11] = (
            hand_structure.middle.dip if hand_structure.middle.dip is not None else hand_structure.middle.tip
        )
        landmarks[12] = hand_structure.middle.tip

        # Ring: MCP(13), PIP(14), DIP(15), tip(16)
        landmarks[13] = hand_structure.ring.mcp
        landmarks[14] = hand_structure.ring.pip if hand_structure.ring.pip is not None else hand_structure.ring.mcp
        landmarks[15] = hand_structure.ring.dip if hand_structure.ring.dip is not None else hand_structure.ring.tip
        landmarks[16] = hand_structure.ring.tip

        # Pinky: MCP(17), PIP(18), DIP(19), tip(20)
        landmarks[17] = hand_structure.pinky.mcp
        landmarks[18] = hand_structure.pinky.pip if hand_structure.pinky.pip is not None else hand_structure.pinky.mcp
        landmarks[19] = hand_structure.pinky.dip if hand_structure.pinky.dip is not None else hand_structure.pinky.tip
        landmarks[20] = hand_structure.pinky.tip

        return landmarks

    def compute_target_positions(self, hand_input: HandStructure | np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """Converts hand input (HandStructure or landmarks array) into Robot Root Frame target positions.

        Args:
            hand_input: Either a HandStructure from tracker or a 21x3 numpy array of landmarks
                in hand frame (wrist at origin, MediaPipe format).

        Returns:
            Dictionary mapping finger names to dictionaries of joint_type -> target position
        """
        # Convert HandStructure to landmarks array if needed

        if isinstance(hand_input, HandStructure):
            landmarks_hand_frame = self._hand_structure_to_landmarks(hand_input)
        else:
            landmarks_hand_frame = hand_input

        wrist = 0
        targets = {}

        # Wrist position in hand frame is (0,0,0) - MediaPipe uses wrist as origin
        wrist_pos = landmarks_hand_frame[wrist]

        # Get current transform of the palm (frame of reference)
        t_palm = self.configuration.get_transform_frame_to_world("right_palm", "body")
        p_palm = t_palm.translation()
        r_palm = t_palm.rotation().as_matrix()

        # Wrist joint offset in palm frame (configurable)
        # Transform wrist offset from palm frame to world frame
        wrist_offset_world = r_palm @ self.config.wrist_offset_palm

        # Wrist joint position in world frame
        p_wrist = p_palm + wrist_offset_world

        allowed_types = set(self.config.target_joint_types)

        for finger_name, landmark_indices in MP_LANDMARK_INDICES.items():
            finger_targets = {}

            for joint_type, mp_idx in landmark_indices.items():
                if joint_type not in allowed_types:
                    continue
                # Vector from Wrist to joint (in MediaPipe hand frame)
                joint_pos = landmarks_hand_frame[mp_idx]
                rel_vec = joint_pos - wrist_pos

                # Scale to robot size
                scaled_vec = rel_vec * self.config.scale_factor

                # Anchor to wrist joint position (not palm center!)
                # Coordinate frame transformation will be applied in solve()
                target_pos = p_wrist + scaled_vec
                finger_targets[joint_type] = target_pos

            if finger_targets:
                targets[finger_name] = finger_targets

        return targets

    def solve(self, hand_input: HandStructure | np.ndarray) -> np.ndarray:
        """Solves IK for the given hand input.

        Args:
            hand_input: Either a HandStructure from tracker or a 21x3 numpy array of landmarks
                in hand frame (wrist at origin, MediaPipe format).

        Returns:
            The full qpos array for the robot.
        """
        targets = self.compute_target_positions(hand_input)

        # Use configurable parameters
        solver = self.config.solver
        dt = self.config.dt
        damping = self.config.damping

        # Perform multiple IK iterations for better convergence
        for iteration in range(self.config.ik_iterations):
            # Build active tasks list (posture task + finger tasks)
            active_tasks: list[mink.Task] = [self.posture_task]

            # Get palm transform and compute wrist position (consistent with compute_target_positions)
            # Recompute each iteration in case configuration changed
            t_palm = self.configuration.get_transform_frame_to_world("right_palm", "body")
            p_palm = t_palm.translation()
            r_palm = t_palm.rotation().as_matrix()

            # Wrist joint offset in palm frame (from config)
            wrist_offset_world = r_palm @ self.config.wrist_offset_palm
            p_wrist = p_palm + wrist_offset_world

            # Update task targets based on current wrist position
            for finger_name, finger_targets in targets.items():
                if finger_name not in self.tasks:
                    continue

                finger_task_dict = self.tasks[finger_name]

                for joint_type, raw_target in finger_targets.items():
                    if joint_type not in finger_task_dict:
                        continue

                    task = finger_task_dict[joint_type]

                    # Coordinate frame transformation:
                    # raw_target is already in world frame, anchored to wrist
                    # Get relative vector from current wrist position
                    rel = raw_target - p_wrist
                    mp_vec = rel  # This is (x, y, z) from MP in world frame

                    # Apply configurable coordinate transformation
                    # Default: MP (X=Forward, Y=Normal, Z=Side) -> Robot (Z=Up, X=Side, Y=Normal)
                    rot_vec = self.config.coord_transform @ mp_vec
                    final_target = p_wrist + rot_vec

                    # FrameTask.set_target expects an SE3 transform
                    target_se3 = mink.SE3.from_translation(final_target)
                    task.set_target(target_se3)
                    active_tasks.append(task)

            # Solve IK
            vel = mink.solve_ik(self.configuration, active_tasks, dt, solver, damping)

            # Integrate velocity to update configuration
            self.configuration.integrate_inplace(vel, dt)

        # Return the new configuration qpos
        return self.configuration.q
