# HandPose

Real-time hand tracking and retargeting to ORCA robot hand using MediaPipe and MuJoCo.

## Installation

```bash
make install
```
Or manually:
```bash
pip install -e ".[dev]"
pip install -e ".[real]"  # adds ROS2 dependencies (rclpy)
```

## Running the Demos

### IK Solution (Simulation)

Uses inverse kinematics (mink) to solve for joint angles:

```bash
make ik
```

**Controls:**
- `q` - Quit

### IK Solution with ROS2 (Hardware)

Publishes joint states to `/joint_states` for ORCA hand control:

```bash
mjpython examples/live_demo_ik_ros2.py --model orca_hand.mjcf --scale 1.0 --rate 100
```

**Requirements:**
- ROS2 installed and sourced
- `rclpy` package: `pip install rclpy`

**Publishes:**
- `/joint_states` (sensor_msgs/JointState) - Joint angles for ORCA hand

**Controls:**
- `q` - Quit

**New to ROS2?** See [docs/ROS2_SYSTEM_OVERVIEW.md](docs/ROS2_SYSTEM_OVERVIEW.md) for a complete explanation of how the system works.

### Manual Retargeting Solution

Uses direct geometric mapping from hand landmarks to joint angles.

```bash
mjpython examples/live_demo.py --camera 0
```

**Controls:**
- `q` - Quit

## General structure:

```
handpose/
├── examples/
│   ├── live_demo_ik.py
│   ├── live_demo_dual.py
│   └── ...
├── handpose/ <--- Library code
│   ├── __init__.py
│   ├── ik_retargeting.py
│   ├── retargeting.py
│   └── tracker.py
```

## Developing
Please keep the codebase well-formatted and linted.

```bash
make format
make static-checks
```
