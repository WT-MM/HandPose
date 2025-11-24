# Quick Start Guide

## Installation

```bash
cd /Users/wesleymaa/Columbia/RoboPIL/HandPose

make install
```

## Run the Demo

```bash
make run
```

## What to Expect

1. Two windows will open:
   - **OpenCV**: Camera feed with green hand landmarks
   - **MuJoCo**: ORCA robot hand mimicking your movements

2. Move your hand in front of the camera

3. The ORCA hand in MuJoCo will mirror your finger movements in real-time

4. Press `q` in the OpenCV window to quit

## Troubleshooting

### Camera not detected
- Check camera permissions
- Try different camera IDs: `--camera 1` or `--camera 2`

### Poor hand tracking  
- Ensure good lighting
- Keep hand clearly visible
- Avoid fast movements

### MuJoCo window issues on macOS
- This is normal due to OpenGL/GLFW conflicts
- The demo will still work without the OpenCV window

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
make format

# Run lints
make lint

# Run tests
make test
```

