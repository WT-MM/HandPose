# HandPose

Real-time hand tracking and retargeting to robot hands using MediaPipe and MuJoCo.

## Features

- **Real-time hand tracking** using MediaPipe
- **Retargeting to ORCA hand** with 17 DOF finger joints
- **Live MuJoCo visualization** showing retargeted hand in real-time
- **Simple, clean codebase** with minimal dependencies

## Installation

```bash
# Clone the repository
cd /Users/wesleymaa/Columbia/RoboPIL/HandPose

# Install dependencies
pip install -e .

# Or install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Single Window Demo (mjpython)

```bash
make run
# or
.venv/bin/mjpython examples/live_demo.py --camera 0
```

Shows MuJoCo window only (OpenCV conflicts with GLFW on macOS).

**Controls:**
- `q` - Quit

### Dual Window Demo (multiprocessing)

```bash
.venv/bin/mjpython examples/live_demo_opencv_subprocess.py --camera 0
```

Shows **both windows** by running OpenCV in a subprocess:
- **OpenCV window**: Camera feed with hand landmarks  
- **MuJoCo window**: Retargeted ORCA hand in real-time

**Controls:**
- `q` in OpenCV window - Quit

**Note:** Requires mjpython for MuJoCo window on macOS

## Project Structure

```
HandPose/
├── handpose/
│   ├── __init__.py
│   ├── hand_tracker.py      # MediaPipe hand tracking
│   ├── retargeting.py       # ORCA hand retargeting
│   └── requirements.txt
├── examples/
│   ├── live_demo.py                    # Live tracking (MuJoCo only)
│   └── live_demo_dual.py  # Live tracking (both windows)
├── models/
│   ├── orca_hand.mjcf       # ORCA hand MuJoCo model
│   └── assets/              # Mesh files
├── pyproject.toml
├── setup.py
└── README.md
```

## How It Works

1. **Hand Tracking**: MediaPipe detects 21 hand landmarks in 3D
2. **Coordinate Transform**: Converts landmarks to hand-relative frame
3. **Retargeting**: Maps human hand pose to 17 ORCA hand joint angles
4. **Visualization**: Displays retargeted pose in MuJoCo in real-time

## Joint Mapping

The system maps human hand joints to ORCA hand joints:

**Thumb (4 joints):**
- CMC flexion → MCP
- CMC abduction → ABD
- MCP flexion → PIP
- IP flexion → DIP

**Fingers (3 joints each):**
- MCP abduction → ABD
- MCP flexion → MCP
- PIP flexion → PIP

Note: DIP joints are not currently used in the ORCA hand model.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- MediaPipe for hand tracking
- ORCA Hand for robot hand model
- MuJoCo for physics simulation

