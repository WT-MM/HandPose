# Quick Start Guide: HandPose ROS2 Integration

This guide provides a quick reference for using HandPose with the tactile_dex data collection pipeline.

## Prerequisites

1. ROS2 installed and sourced: `source /opt/ros/<distro>/setup.bash`
2. HandPose installed: `cd HandPose && pip install -e .`
3. Camera connected and accessible

## Three-Terminal Setup

### Terminal 1: HandPose Publisher
```bash
cd HandPose
python examples/handpose_ros2_publisher.py \
    --camera 0 \
    --model models/orca_hand_fixed.mjcf \
    --scale 1.0 \
    --fps 30 \
    --viz  # Optional: enable visualization window
```

**What it does:**
- Captures video from camera (default 1280x720)
- Detects hand using HaMeR tracker
- Performs IK retargeting to ORCA hand joints
- Publishes raw IK joint angles to `/joint_states` at 30 Hz
- **IMPORTANT**: Does NOT apply REF_OFFSETS (these are applied in tactile_dex)

### Terminal 2: Simulation (Isaac Lab)
```bash
cd tactile_dex
./isaaclab.sh -p orca_tomato_replay.py \
    --delta_topic object_delta_world \
    --rot_topic object_rot_delta_world \
    --joint_topic /joint_states
```

**What it does:**
- Subscribes to `/joint_states` from HandPose publisher
- Applies REF_OFFSETS and REF_GAIN to joint values
- Clamps joints to URDF limits (with limit offsets)
- Applies special limits for `right_thumb_abd` (-20° to +42°)
- Visualizes hand and object in simulation
- Publishes:
  - `/sim_joint_states`: Actual joint angles from simulation
  - `/object_in_wrist`: Object pose relative to wrist frame
  - `/teleop_joint_states_raw`: Raw values from HandPose (diagnostic)
  - `/teleop_joint_states_postproc`: After offset+gain+clamp (diagnostic)

### Terminal 3: Data Recording
```bash
cd tactile_dex
python record_hand_can_yolanda.py record next grasp_task --notes "vision_demo"
```

**What it does:**
- Records `/sim_joint_states` and `/object_in_wrist` to rosbag2
- Creates episode directories with metadata

## Verification

Check that topics are being published:
```bash
# List all topics
ros2 topic list

# Echo joint states
ros2 topic echo /joint_states

# Check publishing rate
ros2 topic hz /joint_states
```

## Troubleshooting

### Camera Issues

**Camera not found:**
- Check device: `ls /dev/video*` (Linux) or check System Preferences (macOS)
- Try different index: `--camera 1`
- On macOS, you may need to grant camera permissions

**No hand detected:**
- Ensure hand is fully visible in frame
- Check lighting conditions (avoid backlighting)
- Lower confidence threshold: `--conf-threshold 0.2`
- Try enabling visualization: `--viz` to see what the tracker sees

**Low frame rate:**
- Reduce camera resolution: `--width 640 --height 480`
- Lower FPS: `--fps 15` (minimum 10 Hz recommended)
- Disable visualization if enabled
- Check CPU/GPU usage

### ROS2 Communication Issues

**Joint states not received:**
- Verify ROS2 is sourced: `source /opt/ros/<distro>/setup.bash`
- Check topic exists: `ros2 topic list`
- Verify publisher is running: `ros2 topic hz /joint_states`
- Check topic names match: publisher uses `/joint_states`, simulator subscribes to `--joint_topic`

**Topics not visible:**
- Ensure both nodes are in the same ROS2 domain (check `ROS_DOMAIN_ID`)
- Verify network configuration if running on different machines

### Hand Pose Issues

**Hand looks "pre-bent" or "twisted" at rest:**
- **This indicates offsets are being applied twice!**
- Check that HandPose publisher does NOT apply REF_OFFSETS (it should output raw IK)
- Verify `orca_tomato_replay.py` applies offsets exactly once

**Hand barely moves or moves incorrectly:**
- Check that joint names match between publisher and simulator
- Verify scale factor (`--scale`) matches your hand size
- Enable diagnostic topics: `ros2 topic echo /teleop_joint_states_raw`
- Check for joint clamping warnings (enable with `LOG_JOINT_STATS=1`)

**Jittery or unstable hand motion:**
- Enable smoothing: `--smoothing 0.3` (0.0 = no smoothing)
- Reduce camera resolution to improve frame rate
- Check that camera is stable and not shaking

**Joint limits being hit frequently:**
- Enable joint stats: `LOG_JOINT_STATS=1 ./isaaclab.sh -p orca_tomato_replay.py ...`
- Check diagnostic topic: `ros2 topic echo /teleop_joint_states_postproc`
- Verify URDF limits are loaded correctly (check console output)

### Integration Notes

**Joint Offsets:**
- HandPose publisher outputs **raw IK joint angles** without offsets
- `orca_tomato_replay.py` applies REF_OFFSETS internally (as per PDF requirements)
- This prevents double-application of offsets

**Joint Limits:**
- Limits are parsed from `orca_hand_model/new_hand1.urdf`
- Limit offsets are applied before clamping (same as REF_OFFSETS)
- Special case: `right_thumb_abd` uses custom limits (-20° to +42°)

**Timestamps:**
- All JointState messages use ROS system time (`node.get_clock().now()`)
- Ensures proper synchronization across topics

## Stopping

Press `Ctrl+C` in each terminal to stop gracefully.


