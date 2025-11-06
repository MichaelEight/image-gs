#!/usr/bin/env python3
"""
Test script to verify the video generation feature is properly integrated.
This script tests the configuration without actually running training.
"""

import sys
import os

# Test 1: Import the configuration module
print("=" * 80)
print("TEST 1: Importing configuration module")
print("=" * 80)
try:
    from quick_start.config import TrainingConfig, set_config
    print("✓ Successfully imported TrainingConfig and set_config")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create a config with video parameters
print("\n" + "=" * 80)
print("TEST 2: Creating configuration with video parameters")
print("=" * 80)
try:
    config = set_config(
        input_filenames="test.png",
        gaussians=[1000],
        steps=[100],
        make_training_video=True,
        video_iterations=50
    )
    print(f"✓ Config created successfully")
    print(f"  - make_training_video: {config.make_training_video}")
    print(f"  - video_iterations: {config.video_iterations}")

    # Verify the values
    assert config.make_training_video == True, "make_training_video should be True"
    assert config.video_iterations == 50, "video_iterations should be 50"
    print("✓ Config values are correct")
except Exception as e:
    print(f"✗ Failed to create config: {e}")
    sys.exit(1)

# Test 3: Test default values
print("\n" + "=" * 80)
print("TEST 3: Testing default values")
print("=" * 80)
try:
    config_default = set_config(
        input_filenames="test.png",
        gaussians=[1000],
        steps=[100]
    )
    print(f"✓ Config with defaults created successfully")
    print(f"  - make_training_video: {config_default.make_training_video}")
    print(f"  - video_iterations: {config_default.video_iterations}")

    # Verify default values
    assert config_default.make_training_video == False, "Default make_training_video should be False"
    assert config_default.video_iterations == 50, "Default video_iterations should be 50"
    print("✓ Default values are correct")
except Exception as e:
    print(f"✗ Failed with defaults: {e}")
    sys.exit(1)

# Test 4: Test validation
print("\n" + "=" * 80)
print("TEST 4: Testing validation")
print("=" * 80)
try:
    # This should raise an error for invalid video_iterations
    try:
        invalid_config = set_config(
            input_filenames="test.png",
            gaussians=[1000],
            steps=[100],
            video_iterations=-10  # Invalid
        )
        print("✗ Validation failed: Should have rejected negative video_iterations")
        sys.exit(1)
    except ValueError as e:
        print(f"✓ Validation works: {e}")
except Exception as e:
    print(f"✗ Unexpected error during validation test: {e}")
    sys.exit(1)

# Test 5: Check if model.py has the video attributes
print("\n" + "=" * 80)
print("TEST 5: Checking model.py integration")
print("=" * 80)
try:
    # Read model.py to verify video attributes exist
    with open("model.py", "r") as f:
        model_content = f.read()

    required_strings = [
        "make_training_video",
        "video_iterations",
        "video_frames",
        "_generate_training_video"
    ]

    for req in required_strings:
        if req in model_content:
            print(f"✓ Found '{req}' in model.py")
        else:
            print(f"✗ Missing '{req}' in model.py")
            sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check model.py: {e}")
    sys.exit(1)

# Test 6: Check if default.yaml has the video parameters
print("\n" + "=" * 80)
print("TEST 6: Checking default.yaml configuration")
print("=" * 80)
try:
    with open("cfgs/default.yaml", "r") as f:
        yaml_content = f.read()

    if "make_training_video" in yaml_content:
        print("✓ Found 'make_training_video' in default.yaml")
    else:
        print("✗ Missing 'make_training_video' in default.yaml")
        sys.exit(1)

    if "video_iterations" in yaml_content:
        print("✓ Found 'video_iterations' in default.yaml")
    else:
        print("✗ Missing 'video_iterations' in default.yaml")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check default.yaml: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nThe video generation feature has been successfully integrated.")
print("\nTo use it, set the following parameters in your training config:")
print("  - make_training_video=True    # Enable video generation")
print("  - video_iterations=50         # Capture frame every 50 iterations")
print("\nThe training video will be saved as 'training_video.mp4' in the output folder.")
print("=" * 80)
