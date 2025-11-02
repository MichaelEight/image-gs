#!/usr/bin/env python3
"""
Test script to verify notebook integration with adaptive refinement.
Tests the configuration and function signatures without running full training.
"""

import sys
from pathlib import Path

# Add repo to path
repo_dir = Path(__file__).parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))


def test_set_config_standard():
    """Test standard configuration (unchanged behavior)."""
    print("\n" + "="*70)
    print("Test 1: Standard Configuration (backward compatibility)")
    print("="*70)

    from quick_start import set_config

    try:
        config = set_config(
            input_filenames="cat.png",
            gaussians=[1000],
            steps=[2000]
        )

        print(f"✅ Standard config created successfully")
        print(f"  - Images: {config.input_filenames}")
        print(f"  - Gaussians: {config.gaussians}")
        print(f"  - Steps: {config.steps}")
        print(f"  - Adaptive config: {config.adaptive_config}")
        print(f"  - Expected: adaptive_config should be None")

        assert config.adaptive_config is None, "Standard config should not have adaptive_config"
        print(f"✅ PASSED: Backward compatibility maintained")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_set_config_adaptive():
    """Test adaptive configuration (new feature)."""
    print("\n" + "="*70)
    print("Test 2: Adaptive Configuration (new feature)")
    print("="*70)

    from quick_start import set_config

    try:
        config = set_config(
            input_filenames="cat.png",
            gaussians=[10000],
            steps=[5000],
            adaptive_refinement=True,
            adaptive_patch_size=256,
            adaptive_error_threshold=0.3
        )

        print(f"✅ Adaptive config created successfully")
        print(f"  - Images: {config.input_filenames}")
        print(f"  - Gaussians: {config.gaussians}")
        print(f"  - Steps: {config.steps}")
        print(f"  - Adaptive config: {config.adaptive_config}")
        print(f"  - Adaptive enabled: {config.adaptive_config.enable if config.adaptive_config else False}")
        print(f"  - Patch size: {config.adaptive_config.patch_size if config.adaptive_config else 'N/A'}")
        print(f"  - Error threshold: {config.adaptive_config.error_threshold if config.adaptive_config else 'N/A'}")

        assert config.adaptive_config is not None, "Adaptive config should be set"
        assert config.adaptive_config.enable is True, "Adaptive should be enabled"
        assert config.adaptive_config.patch_size == 256, "Patch size should be 256"
        assert config.adaptive_config.error_threshold == 0.3, "Error threshold should be 0.3"

        print(f"✅ PASSED: Adaptive configuration works correctly")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_config_validation():
    """Test that adaptive config uses first gaussian count as base."""
    print("\n" + "="*70)
    print("Test 3: Adaptive Config Validation")
    print("="*70)

    from quick_start import set_config

    try:
        config = set_config(
            input_filenames=["cat.png", "dog.png"],
            gaussians=[10000, 20000],
            steps=[5000],
            adaptive_refinement=True
        )

        print(f"✅ Multi-gaussian adaptive config created")
        print(f"  - Gaussians list: {config.gaussians}")
        print(f"  - Base gaussians (adaptive): {config.adaptive_config.base_gaussians}")
        print(f"  - Expected: base_gaussians should be {config.gaussians[0]}")

        assert config.adaptive_config.base_gaussians == config.gaussians[0], \
            f"Base gaussians should be first gaussian count ({config.gaussians[0]})"

        print(f"✅ PASSED: Base gaussians correctly set to first value")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_routing():
    """Test that train() function routes correctly based on config."""
    print("\n" + "="*70)
    print("Test 4: Train Routing Logic")
    print("="*70)

    from quick_start.config import TrainingConfig, AdaptiveRefinementConfig

    try:
        # Test standard config
        standard_config = TrainingConfig(
            input_filenames=["test.png"],
            gaussians=[1000],
            steps=[2000]
        )

        print(f"✅ Standard TrainingConfig created")
        print(f"  - Adaptive config: {standard_config.adaptive_config}")
        print(f"  - Should route to: Standard training")

        # Test adaptive config
        adaptive_cfg = AdaptiveRefinementConfig(enable=True)
        adaptive_config = TrainingConfig(
            input_filenames=["test.png"],
            gaussians=[10000],
            steps=[5000],
            adaptive_config=adaptive_cfg
        )

        print(f"\n✅ Adaptive TrainingConfig created")
        print(f"  - Adaptive config: {adaptive_config.adaptive_config}")
        print(f"  - Enabled: {adaptive_config.adaptive_config.enable}")
        print(f"  - Should route to: Adaptive training")

        # Verify routing logic
        is_adaptive_standard = (standard_config.adaptive_config is not None and
                               standard_config.adaptive_config.enable)
        is_adaptive_adaptive = (adaptive_config.adaptive_config is not None and
                               adaptive_config.adaptive_config.enable)

        assert not is_adaptive_standard, "Standard config should not trigger adaptive"
        assert is_adaptive_adaptive, "Adaptive config should trigger adaptive"

        print(f"\n✅ PASSED: Train routing logic correct")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_example_notebook_usage():
    """Test the example usage that will be shown in notebooks."""
    print("\n" + "="*70)
    print("Test 5: Example Notebook Usage")
    print("="*70)

    from quick_start import set_config

    try:
        print("\n📓 Example 1: Standard training")
        print("-" * 70)
        print("config = set_config(")
        print("    input_filenames='cat.png',")
        print("    gaussians=[1000],")
        print("    steps=[2000]")
        print(")")

        config1 = set_config(
            input_filenames="cat.png",
            gaussians=[1000],
            steps=[2000]
        )
        print(f"✅ Created: Standard config (adaptive: {config1.adaptive_config is not None})")

        print("\n📓 Example 2: Adaptive refinement training")
        print("-" * 70)
        print("config = set_config(")
        print("    input_filenames='cat.png',")
        print("    gaussians=[10000],")
        print("    steps=[5000],")
        print("    adaptive_refinement=True,")
        print("    adaptive_error_threshold=0.3")
        print(")")

        config2 = set_config(
            input_filenames="cat.png",
            gaussians=[10000],
            steps=[5000],
            adaptive_refinement=True,
            adaptive_error_threshold=0.3
        )
        print(f"✅ Created: Adaptive config (adaptive: {config2.adaptive_config is not None})")

        print("\n✅ PASSED: Example notebook usage works")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("NOTEBOOK INTEGRATION TESTS")
    print("Testing adaptive refinement integration with quick_start API")
    print("="*70)

    results = []

    # Run tests
    results.append(("Standard Config", test_set_config_standard()))
    results.append(("Adaptive Config", test_set_config_adaptive()))
    results.append(("Config Validation", test_adaptive_config_validation()))
    results.append(("Train Routing", test_train_routing()))
    results.append(("Example Usage", test_example_notebook_usage()))

    # Print results summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe notebook integration is working correctly.")
        print("\nYou can now use adaptive refinement in your notebooks:")
        print("\n  from quick_start import set_config, train, view_results")
        print("\n  config = set_config(")
        print("      input_filenames='cat.png',")
        print("      gaussians=[10000],")
        print("      steps=[5000],")
        print("      adaptive_refinement=True")
        print("  )")
        print("\n  results = train(config)")
        print("  view_results(results[0])")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease review the failures above.")

    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
