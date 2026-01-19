"""Test script to verify handpose_ros2_publisher.py structure without ROS2."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports that don't require ROS2
print("Testing imports...")
try:
    import cv2
    print("✓ cv2")
except ImportError as e:
    print(f"✗ cv2: {e}")

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import mujoco
    print("✓ mujoco")
except ImportError as e:
    print(f"✗ mujoco: {e}")

try:
    from handpose.ik_retargeting import ORCAHandIKRetargeting, ORCAHandIKConfig, ORCA_JOINT_NAMES
    print(f"✓ handpose.ik_retargeting ({len(ORCA_JOINT_NAMES)} joints)")
except ImportError as e:
    print(f"✗ handpose.ik_retargeting: {e}")

try:
    from handpose.tracker.hamer import HaMeRTracker
    print("✓ handpose.tracker.hamer")
except ImportError as e:
    print(f"✗ handpose.tracker.hamer: {e}")

# Test ROS2 imports (will fail, but that's expected)
print("\nTesting ROS2 imports (expected to fail on macOS)...")
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    print("✓ ROS2 packages available!")
except ImportError as e:
    print(f"✗ ROS2 not available: {e}")
    print("  (This is expected if ROS2 is not installed)")

# Test that the script can be parsed
print("\nTesting script structure...")
try:
    with open("examples/handpose_ros2_publisher.py", "r") as f:
        code = f.read()
    
    # Check for key components
    checks = [
        ("HandPoseROS2Publisher class", "class HandPoseROS2Publisher"),
        ("HaMeRTracker usage", "HaMeRTracker"),
        ("ORCAHandIKRetargeting usage", "ORCAHandIKRetargeting"),
        ("JointState publishing", "JointState"),
        ("ORCA_JOINT_NAMES usage", "ORCA_JOINT_NAMES"),
        ("Camera capture", "cv2.VideoCapture"),
        ("ROS2 timer", "create_timer"),
        ("Command line args", "argparse"),
        ("Main function", "def main()"),
    ]
    
    for name, pattern in checks:
        if pattern in code:
            print(f"✓ {name}")
        else:
            print(f"✗ {name} (missing: {pattern})")
            
except Exception as e:
    print(f"✗ Error reading script: {e}")

# Test ORCA_JOINT_NAMES structure
print("\nTesting ORCA joint names...")
try:
    from handpose.ik_retargeting import ORCA_JOINT_NAMES
    print(f"Total joints: {len(ORCA_JOINT_NAMES)}")
    print("Joint names:")
    for i, name in enumerate(ORCA_JOINT_NAMES, 1):
        print(f"  {i:2d}. {name}")
    
    # Verify expected joints
    expected_joints = [
        "right_thumb_abd", "right_thumb_mcp", "right_thumb_pip", "right_thumb_dip",
        "right_index_abd", "right_index_mcp", "right_index_pip",
        "right_middle_abd", "right_middle_mcp", "right_middle_pip",
        "right_ring_abd", "right_ring_mcp", "right_ring_pip",
        "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip",
    ]
    
    missing = set(expected_joints) - set(ORCA_JOINT_NAMES)
    if missing:
        print(f"\n⚠ Missing expected joints: {missing}")
    else:
        print("\n✓ All expected joints present")
        
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("Structure test complete!")
print("="*60)

