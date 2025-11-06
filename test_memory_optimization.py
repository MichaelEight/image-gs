#!/usr/bin/env python3
"""
Test script to verify the memory optimization features are properly integrated.
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

# Test 2: Create a config with memory optimization parameters
print("\n" + "=" * 80)
print("TEST 2: Creating configuration with memory optimization parameters")
print("=" * 80)
try:
    config = set_config(
        input_filenames="test.png",
        gaussians=[1000],
        steps=[100],
        use_amp=True,
        use_gradient_checkpointing=True,
        video_save_to_disk=True
    )
    print(f"✓ Config created successfully")
    print(f"  - use_amp: {config.use_amp}")
    print(f"  - use_gradient_checkpointing: {config.use_gradient_checkpointing}")
    print(f"  - video_save_to_disk: {config.video_save_to_disk}")

    # Verify the values
    assert config.use_amp == True, "use_amp should be True"
    assert config.use_gradient_checkpointing == True, "use_gradient_checkpointing should be True"
    assert config.video_save_to_disk == True, "video_save_to_disk should be True"
    print("✓ Config values are correct")
except Exception as e:
    print(f"✗ Failed to create config: {e}")
    sys.exit(1)

# Test 3: Test default values
print("\n" + "=" * 80)
print("TEST 3: Testing default values for memory optimization")
print("=" * 80)
try:
    config_default = set_config(
        input_filenames="test.png",
        gaussians=[1000],
        steps=[100]
    )
    print(f"✓ Config with defaults created successfully")
    print(f"  - use_amp: {config_default.use_amp}")
    print(f"  - use_gradient_checkpointing: {config_default.use_gradient_checkpointing}")
    print(f"  - video_save_to_disk: {config_default.video_save_to_disk}")

    # Verify default values match cfgs/default.yaml
    assert config_default.use_amp == True, "Default use_amp should be True"
    assert config_default.use_gradient_checkpointing == False, "Default use_gradient_checkpointing should be False"
    assert config_default.video_save_to_disk == True, "Default video_save_to_disk should be True"
    print("✓ Default values are correct")
except Exception as e:
    print(f"✗ Failed with defaults: {e}")
    sys.exit(1)

# Test 4: Test disabling memory optimizations
print("\n" + "=" * 80)
print("TEST 4: Testing disabled memory optimization")
print("=" * 80)
try:
    config_disabled = set_config(
        input_filenames="test.png",
        gaussians=[1000],
        steps=[100],
        use_amp=False,
        use_gradient_checkpointing=False,
        video_save_to_disk=False
    )
    print(f"✓ Config with disabled optimizations created")
    print(f"  - use_amp: {config_disabled.use_amp}")
    print(f"  - use_gradient_checkpointing: {config_disabled.use_gradient_checkpointing}")
    print(f"  - video_save_to_disk: {config_disabled.video_save_to_disk}")

    assert config_disabled.use_amp == False, "use_amp should be False when disabled"
    assert config_disabled.use_gradient_checkpointing == False, "use_gradient_checkpointing should be False"
    assert config_disabled.video_save_to_disk == False, "video_save_to_disk should be False"
    print("✓ Disabled optimization values are correct")
except Exception as e:
    print(f"✗ Failed with disabled optimizations: {e}")
    sys.exit(1)

# Test 5: Check if model.py has the memory optimization attributes
print("\n" + "=" * 80)
print("TEST 5: Checking model.py integration")
print("=" * 80)
try:
    # Read model.py to verify memory optimization attributes exist
    with open("model.py", "r") as f:
        model_content = f.read()

    required_strings = [
        "use_amp",
        "use_gradient_checkpointing",
        "video_save_to_disk",
        "GradScaler",
        "autocast",
        "_save_video_frame_to_disk",
        "_generate_training_video_from_disk"
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

# Test 6: Check if default.yaml has the memory optimization parameters
print("\n" + "=" * 80)
print("TEST 6: Checking default.yaml configuration")
print("=" * 80)
try:
    with open("cfgs/default.yaml", "r") as f:
        yaml_content = f.read()

    required_yaml_params = [
        "use_amp",
        "use_gradient_checkpointing",
        "video_save_to_disk"
    ]

    for param in required_yaml_params:
        if param in yaml_content:
            print(f"✓ Found '{param}' in default.yaml")
        else:
            print(f"✗ Missing '{param}' in default.yaml")
            sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check default.yaml: {e}")
    sys.exit(1)

# Test 7: Check training.py integration
print("\n" + "=" * 80)
print("TEST 7: Checking training.py integration")
print("=" * 80)
try:
    with open("quick_start/training.py", "r") as f:
        training_content = f.read()

    required_training_strings = [
        "use_amp",
        "use_gradient_checkpointing",
        "video_save_to_disk",
        "--use_amp",
        "--use_gradient_checkpointing",
        "--video_save_to_disk"
    ]

    for req in required_training_strings:
        if req in training_content:
            print(f"✓ Found '{req}' in training.py")
        else:
            print(f"✗ Missing '{req}' in training.py")
            sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check training.py: {e}")
    sys.exit(1)

# Test 8: Check config.py dataclass integration
print("\n" + "=" * 80)
print("TEST 8: Checking config.py dataclass fields")
print("=" * 80)
try:
    from quick_start.config import TrainingConfig
    import inspect

    # Get TrainingConfig annotations
    annotations = TrainingConfig.__annotations__

    required_fields = [
        "use_amp",
        "use_gradient_checkpointing",
        "video_save_to_disk"
    ]

    for field in required_fields:
        if field in annotations:
            print(f"✓ Found '{field}' in TrainingConfig dataclass")
        else:
            print(f"✗ Missing '{field}' in TrainingConfig dataclass")
            sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check config.py dataclass: {e}")
    sys.exit(1)

# Test 9: Test configuration with all features combined
print("\n" + "=" * 80)
print("TEST 9: Testing combined configuration (8K image scenario)")
print("=" * 80)
try:
    config_8k = set_config(
        input_filenames="large_8k_image.png",
        gaussians=[30000],
        steps=[10000],
        use_progressive=True,
        make_training_video=True,
        video_iterations=100,
        eval_steps=100,
        use_amp=True,
        use_gradient_checkpointing=True,
        video_save_to_disk=True
    )
    print(f"✓ 8K scenario config created successfully")
    print(f"  - Gaussians: {config_8k.gaussians}")
    print(f"  - Steps: {config_8k.steps}")
    print(f"  - use_amp: {config_8k.use_amp}")
    print(f"  - use_gradient_checkpointing: {config_8k.use_gradient_checkpointing}")
    print(f"  - video_save_to_disk: {config_8k.video_save_to_disk}")
    print(f"  - make_training_video: {config_8k.make_training_video}")

    # Verify all values
    assert config_8k.gaussians == [30000]
    assert config_8k.steps == [10000]
    assert config_8k.use_amp == True
    assert config_8k.use_gradient_checkpointing == True
    assert config_8k.video_save_to_disk == True
    assert config_8k.make_training_video == True
    print("✓ 8K scenario values are correct")
except Exception as e:
    print(f"✗ Failed with 8K scenario: {e}")
    sys.exit(1)

# Test 10: Check documentation exists
print("\n" + "=" * 80)
print("TEST 10: Checking documentation files")
print("=" * 80)
try:
    doc_file = "MEMORY_OPTIMIZATION_DOCUMENTATION.md"
    if os.path.exists(doc_file):
        print(f"✓ Found '{doc_file}'")
        with open(doc_file, "r") as f:
            doc_content = f.read()
            doc_keywords = ["AMP", "Gradient Checkpointing", "Disk-Based Video", "8K"]
            for keyword in doc_keywords:
                if keyword in doc_content:
                    print(f"  ✓ Documentation contains '{keyword}'")
                else:
                    print(f"  ⚠ Documentation missing '{keyword}'")
    else:
        print(f"⚠ Documentation file '{doc_file}' not found (optional)")
except Exception as e:
    print(f"⚠ Warning checking documentation: {e}")

# All tests passed
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nMemory optimization features have been successfully integrated.")
print("\nFeatures available:")
print("  1. Automatic Mixed Precision (AMP) - ~40-50% memory savings")
print("  2. Gradient Checkpointing - ~30-40% additional savings (slower)")
print("  3. Disk-Based Video Storage - Saves video frames to disk")
print("\nUsage in quick-start.ipynb:")
print("  config = set_config(")
print("      input_filenames='image_8k.png',")
print("      gaussians=[30000],")
print("      steps=[10000],")
print("      use_amp=True,                      # Enable AMP (recommended)")
print("      use_gradient_checkpointing=False,  # Enable only if needed")
print("      video_save_to_disk=True            # Save to disk (recommended)")
print("  )")
print("\nRecommended for 8K+ images:")
print("  - Always enable: use_amp=True")
print("  - Always enable: video_save_to_disk=True")
print("  - Use sparingly: use_gradient_checkpointing=True (only if OOM persists)")
print("\nSee MEMORY_OPTIMIZATION_DOCUMENTATION.md for detailed usage guide.")
print("=" * 80)
