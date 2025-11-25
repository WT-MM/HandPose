# HandPose

Real-time hand tracking and retargeting to ORCA robot hand using MediaPipe and MuJoCo.

## Installation

```bash
make install 
```
Or manually:
```bash
pip install -e ".[dev]"
```

## Running the Demos

### IK Solution

Uses inverse kinematics (mink) to solve for joint angles:

```bash
make ik
```

**Controls:**
- `q` - Quit

### Manual Retargeting Solution

Uses direct geometric mapping from hand landmarks to joint angles.

```bash
mjpython examples/live_demo_dual.py --camera 0
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
