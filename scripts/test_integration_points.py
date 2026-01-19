"""Test integration points between HandPose and tactile_dex."""

import sys
from pathlib import Path

print("="*60)
print("Testing Integration Points")
print("="*60)

# Test 1: Verify joint names match
print("\n1. Testing joint name consistency...")
try:
    # From HandPose
    sys.path.insert(0, str(Path(__file__).parent))
    from handpose.ik_retargeting import ORCA_JOINT_NAMES as handpose_joints
    
    # Expected joints (from tactile_dex expectations)
    expected_joints = [
        "right_thumb_abd", "right_thumb_mcp", "right_thumb_pip", "right_thumb_dip",
        "right_index_abd", "right_index_mcp", "right_index_pip",
        "right_middle_abd", "right_middle_mcp", "right_middle_pip",
        "right_ring_abd", "right_ring_mcp", "right_ring_pip",
        "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip",
    ]
    
    if list(handpose_joints) == expected_joints:
        print("✓ Joint names match exactly")
    else:
        print("⚠ Joint name mismatch:")
        print(f"  HandPose: {list(handpose_joints)}")
        print(f"  Expected: {expected_joints}")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Verify REF_OFFSETS keys match joint names
print("\n2. Testing REF_OFFSETS consistency...")
try:
    # Read from tactile_dex script
    tactile_dex_path = Path(__file__).parent.parent / "tactile_dex" / "orca_tomato_replay.py"
    with open(tactile_dex_path, "r") as f:
        code = f.read()
    
    # Extract REF_OFFSETS (simple parsing)
    import re
    offsets_match = re.search(r'REF_OFFSETS\s*=\s*\{([^}]+)\}', code, re.DOTALL)
    if offsets_match:
        offsets_str = offsets_match.group(1)
        offset_joints = re.findall(r'"([^"]+)"', offsets_str)
        print(f"✓ Found REF_OFFSETS for joints: {offset_joints}")
        
        # Check if all offset joints are in ORCA_JOINT_NAMES
        missing = set(offset_joints) - set(handpose_joints)
        if missing:
            print(f"⚠ Joints in REF_OFFSETS but not in ORCA_JOINT_NAMES: {missing}")
        else:
            print("✓ All REF_OFFSETS joints are in ORCA_JOINT_NAMES")
    else:
        print("✗ Could not find REF_OFFSETS in script")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Verify topic names
print("\n3. Testing ROS2 topic names...")
try:
    publisher_path = Path(__file__).parent / "examples" / "handpose_ros2_publisher.py"
    with open(publisher_path, "r") as f:
        pub_code = f.read()
    
    # Check publisher topic
    if '"/joint_states"' in pub_code or "'/joint_states'" in pub_code:
        print("✓ Publisher uses /joint_states")
    else:
        print("✗ Publisher topic not found")
    
    # Check subscriber topic in tactile_dex
    if 'joint_topic' in code and 'default="/joint_states"' in code:
        print("✓ Subscriber defaults to /joint_states")
    else:
        print("⚠ Subscriber topic configuration may differ")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Verify no double offset application
print("\n4. Testing offset application logic...")
try:
    # Check publisher doesn't apply offsets
    if "REF_OFFSETS" not in pub_code:
        print("✓ Publisher does NOT apply REF_OFFSETS (correct)")
    else:
        print("⚠ Publisher may apply REF_OFFSETS (should be in tactile_dex only)")
    
    # Check tactile_dex applies offsets
    if "REF_OFFSETS.get" in code and "val = val + REF_OFFSETS.get" in code:
        print("✓ tactile_dex applies REF_OFFSETS (correct)")
    else:
        print("✗ tactile_dex may not apply REF_OFFSETS")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Verify timestamp usage
print("\n5. Testing timestamp usage...")
try:
    # Check publisher uses ROS time
    if "get_clock().now().to_msg()" in pub_code:
        print("✓ Publisher uses ROS system time")
    else:
        print("⚠ Publisher may not use ROS time")
    
    # Check tactile_dex uses ROS time
    if "get_clock().now().to_msg()" in code:
        print("✓ tactile_dex uses ROS system time")
    else:
        print("⚠ tactile_dex may not use ROS time")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Verify model path handling
print("\n6. Testing model path handling...")
try:
    # Check publisher handles relative paths
    if "Path(__file__).parent.parent" in pub_code or "Path(model_path)" in pub_code:
        print("✓ Publisher handles relative model paths")
    else:
        print("⚠ Publisher model path handling unclear")
        
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("Integration point tests complete!")
print("="*60)

