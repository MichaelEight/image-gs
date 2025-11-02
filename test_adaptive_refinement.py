"""
Test script for adaptive patch-based refinement feature.

This script demonstrates how to use the adaptive refinement feature
and verifies the implementation works correctly.
"""

import sys
import os

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quick_start.config import AdaptiveRefinementConfig


def test_config_creation():
    """Test that AdaptiveRefinementConfig can be created and validated."""
    print("Testing AdaptiveRefinementConfig creation...")

    # Test valid config
    try:
        config = AdaptiveRefinementConfig(
            enable=True,
            patch_size=256,
            overlap=32,
            psnr_weight=0.7,
            ssim_weight=0.3,
            error_threshold=0.3,
            base_gaussians=10000,
            refinement_gaussian_increment=2000,
            max_refinement_iterations=3,
            max_gaussians_per_patch=20000
        )
        print("[OK] Valid config created successfully")
        print(f"  Patch size: {config.patch_size}")
        print(f"  Overlap: {config.overlap}")
        print(f"  Error threshold: {config.error_threshold}")
    except Exception as e:
        print(f"[FAIL] Failed to create valid config: {e}")
        return False

    # Test invalid config (weights don't sum to 1)
    try:
        invalid_config = AdaptiveRefinementConfig(
            enable=True,
            psnr_weight=0.5,
            ssim_weight=0.3  # Sum is 0.8, not 1.0
        )
        print("[FAIL] Invalid config should have raised ValueError")
        return False
    except ValueError as e:
        print(f"[OK] Invalid config correctly rejected: {e}")

    return True


def test_patch_utilities():
    """Test patch utility functions."""
    print("\nTesting patch utilities...")

    import torch
    from utils.patch_utils import (
        split_into_patches,
        reconstruct_from_patches,
        calculate_patch_error,
        create_blend_weight_map
    )

    # Create test image
    test_image = torch.rand(3, 512, 512)  # RGB image

    # Test splitting
    patches, coordinates = split_into_patches(test_image, patch_size=256, overlap=32)
    print(f"[OK] Image split into {len(patches)} patches")
    print(f"  Coordinates count: {len(coordinates)}")

    # Test blend weight map
    weights = create_blend_weight_map(patch_size=256, overlap=32)
    print(f"[OK] Blend weight map created: shape {weights.shape}")

    # Test reconstruction
    reconstructed = reconstruct_from_patches(
        patches, coordinates, test_image.shape, overlap=32
    )
    print(f"[OK] Image reconstructed: shape {reconstructed.shape}")

    # Check reconstruction quality
    diff = torch.abs(test_image - reconstructed).mean().item()
    print(f"  Reconstruction error: {diff:.6f}")

    if diff < 0.01:
        print("[OK] Reconstruction quality is good")
    else:
        print("[WARN] Reconstruction has noticeable error")

    # Test error calculation
    patch1 = torch.rand(3, 256, 256)
    patch2 = patch1 + 0.1  # Add some noise
    metrics = calculate_patch_error(patch1, patch2)
    print(f"[OK] Error metrics calculated:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  Combined error: {metrics['combined_error']:.4f}")

    return True


def test_patch_manager():
    """Test PatchManager class."""
    print("\nTesting PatchManager...")

    import torch
    from quick_start.patch_manager import PatchManager

    # Create test setup
    image_shape = (3, 512, 512)
    manager = PatchManager(
        image_shape=image_shape,
        patch_size=256,
        overlap=32,
        error_threshold=0.3,
        psnr_weight=0.7,
        ssim_weight=0.3
    )

    print(f"[OK] PatchManager created")
    print(f"  Total patches: {len(manager.patches)}")
    print(f"  Grid: {manager.n_rows}x{manager.n_cols}")

    # Test image splitting
    test_image = torch.rand(*image_shape)
    patches = manager.split_image(test_image)
    print(f"[OK] Image split: {len(patches)} patches")

    # Test reconstruction
    reconstructed = manager.reconstruct_image(patches)
    print(f"[OK] Image reconstructed: shape {reconstructed.shape}")

    # Test patch evaluation
    gt_patches = patches
    rendered_patches = [p + torch.randn_like(p) * 0.05 for p in patches]  # Add noise
    stats = manager.evaluate_patches(rendered_patches, gt_patches)
    print(f"[OK] Patches evaluated:")
    print(f"  Mean PSNR: {stats['mean_psnr']:.2f} dB")
    print(f"  Mean SSIM: {stats['mean_ssim']:.4f}")
    print(f"  Patches needing refinement: {stats['num_needing_refinement']}/{stats['num_patches']}")

    # Test visualization
    viz = manager.create_visualization(test_image, show_errors=True)
    print(f"[OK] Visualization created: shape {viz.shape}")

    return True


def print_usage_instructions():
    """Print instructions for using the adaptive refinement feature."""
    print("\n" + "="*80)
    print("HOW TO USE ADAPTIVE REFINEMENT")
    print("="*80)
    print("\n1. Using the command line:")
    print("   python main.py --adaptive_refinement True")
    print("   (Configure other parameters in cfgs/default.yaml)")
    print("\n2. Using the Python API:")
    print("""
   from quick_start.config import AdaptiveRefinementConfig
   from quick_start.training import train_adaptive_wrapper

   config = AdaptiveRefinementConfig(
       enable=True,
       patch_size=256,
       overlap=32,
       error_threshold=0.3,
       base_gaussians=10000,
       refinement_gaussian_increment=2000,
       max_refinement_iterations=3
   )

   output_dir = train_adaptive_wrapper("your_image.png", config)
   print(f"Results saved to: {output_dir}")
    """)
    print("\n3. Output structure:")
    print("""
   output/session_N/{image}-adaptive-{gaussians}/
   ├── base_training/
   │   ├── model.pt              # Base model checkpoint
   │   └── rendered.jpg          # Base rendered image
   ├── enhanced/
   │   ├── rendered.jpg          # Enhanced final image
   │   └── patch_summary.json    # Per-patch statistics
   ├── comparison.jpg            # 3-way comparison (original, base, enhanced)
   └── other/
       ├── patches/
       │   ├── patch_0_0/        # Individual patch results
       │   │   ├── before.jpg
       │   │   ├── after.jpg
       │   │   ├── error_before.jpg
       │   │   ├── error_after.jpg
       │   │   └── metrics.json
       │   └── ...
       └── patch_grid_visualization.jpg
    """)
    print("="*80)


def main():
    """Run all tests."""
    print("="*80)
    print("ADAPTIVE REFINEMENT FEATURE TEST")
    print("="*80)

    all_passed = True

    # Test 1: Config creation
    if not test_config_creation():
        all_passed = False

    # Test 2: Patch utilities
    try:
        if not test_patch_utilities():
            all_passed = False
    except Exception as e:
        print(f"[FAIL] Patch utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: Patch manager
    try:
        if not test_patch_manager():
            all_passed = False
    except Exception as e:
        print(f"[FAIL] Patch manager test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Print results
    print("\n" + "="*80)
    if all_passed:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("="*80)

    # Print usage instructions
    print_usage_instructions()

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
