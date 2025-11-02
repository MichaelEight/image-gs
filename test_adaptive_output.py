#!/usr/bin/env python3
"""
Test script to verify enhanced adaptive patching output functionality.
Tests the new visualization and summary features without running full training.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Import the visualization modules
from quick_start.patch_manager import PatchManager, PatchInfo
from quick_start.adaptive_visualization import (
    create_tile_score_heatmap,
    format_adaptive_summary_text
)


def create_test_data():
    """Create synthetic test data for visualization."""
    # Create a simple test image (3, 256, 256)
    image_shape = (3, 256, 256)

    # Create patch manager with test parameters
    patch_manager = PatchManager(
        image_shape=image_shape,
        patch_size=128,
        overlap=16,
        error_threshold=0.3,
        psnr_weight=0.7,
        ssim_weight=0.3,
        device='cpu'
    )

    # Simulate some patch errors (some above threshold, some below)
    np.random.seed(42)
    for i, patch in enumerate(patch_manager.patches):
        # Create varied error scores
        base_error = 0.2 + np.random.random() * 0.4  # 0.2 to 0.6
        patch.combined_error = base_error
        patch.psnr = 25.0 + np.random.random() * 10.0  # 25-35 dB
        patch.ssim = 0.7 + np.random.random() * 0.25  # 0.7-0.95
        patch.needs_refinement = base_error > patch_manager.error_threshold

        # Simulate refinement for patches above threshold
        if patch.needs_refinement:
            patch.refinement_iteration = np.random.randint(1, 4)
            patch.total_gaussians = 10000 + patch.refinement_iteration * 2000

    # Create patch refinement details
    patch_refinement_details = []
    for patch in patch_manager.patches:
        if patch.needs_refinement:
            detail = {
                'patch_id': patch.patch_id,
                'row': patch.row,
                'col': patch.col,
                'before': {
                    'psnr': patch.psnr - 2.0,
                    'ssim': patch.ssim - 0.05,
                    'combined_error': patch.combined_error + 0.1
                },
                'after': {
                    'psnr': patch.psnr,
                    'ssim': patch.ssim,
                    'combined_error': patch.combined_error
                },
                'improvements': {
                    'psnr': 2.0,
                    'ssim': 0.05,
                    'error_reduction': 0.1
                },
                'iterations': patch.refinement_iteration,
                'gaussians_added': patch.refinement_iteration * 2000,
                'final_gaussians': patch.total_gaussians,
                'training_time_seconds': 45.3 * patch.refinement_iteration,
                'threshold_met': patch.combined_error <= patch_manager.error_threshold
            }
            patch_refinement_details.append(detail)

    # Create patch statistics
    errors = [p.combined_error for p in patch_manager.patches]
    psnrs = [p.psnr for p in patch_manager.patches]
    ssims = [p.ssim for p in patch_manager.patches]

    patch_stats = {
        'total_patches': len(patch_manager.patches),
        'patches_refined': len(patch_refinement_details),
        'patches_still_needing_refinement': sum(1 for p in patch_manager.patches if p.needs_refinement),
        'mean_error': float(np.mean(errors)),
        'max_error': float(np.max(errors)),
        'min_error': float(np.min(errors)),
        'mean_psnr': float(np.mean(psnrs)),
        'mean_ssim': float(np.mean(ssims)),
        'total_gaussians': sum(p.total_gaussians for p in patch_manager.patches if p.total_gaussians > 0),
        'mean_gaussians_per_patch': 12000.0
    }

    return patch_manager, patch_refinement_details, patch_stats


def test_tile_score_heatmap():
    """Test the tile score heatmap visualization."""
    print("\n" + "="*60)
    print("Test 1: Tile Score Heatmap")
    print("="*60)

    patch_manager, _, _ = create_test_data()

    # Create output directory
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)

    # Create heatmap
    print("\nCreating tile score heatmap...")
    try:
        heatmap_path = output_dir / "test_tile_scores_heatmap.png"
        fig = create_tile_score_heatmap(
            patch_manager=patch_manager,
            error_threshold=0.3,
            output_path=str(heatmap_path)
        )
        plt.close(fig)
        print(f"✅ SUCCESS: Heatmap created at {heatmap_path}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summary_text():
    """Test the enhanced summary text generation."""
    print("\n" + "="*60)
    print("Test 2: Enhanced Summary Text")
    print("="*60)

    _, patch_refinement_details, patch_stats = create_test_data()

    # Create summary text
    print("\nGenerating summary text...")
    try:
        summary_text = format_adaptive_summary_text(
            base_psnr=28.5,
            base_ssim=0.85,
            enhanced_psnr=31.2,
            enhanced_ssim=0.89,
            patch_stats=patch_stats,
            training_time=180.5,
            base_gaussians=10000,
            patch_refinement_details=patch_refinement_details
        )

        # Save summary
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        summary_path = output_dir / "test_summary.txt"

        with open(summary_path, 'w') as f:
            f.write(summary_text)

        print(f"✅ SUCCESS: Summary text created at {summary_path}")
        print("\nPreview of summary:")
        print("-" * 60)
        lines = summary_text.split('\n')
        for line in lines[:30]:  # Show first 30 lines
            print(line)
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structures():
    """Test that data structures are created correctly."""
    print("\n" + "="*60)
    print("Test 3: Data Structure Validation")
    print("="*60)

    patch_manager, patch_refinement_details, patch_stats = create_test_data()

    print(f"\n✓ Patch manager created")
    print(f"  - Grid dimensions: {patch_manager.n_rows} × {patch_manager.n_cols}")
    print(f"  - Total patches: {len(patch_manager.patches)}")
    print(f"  - Patches needing refinement: {sum(1 for p in patch_manager.patches if p.needs_refinement)}")

    print(f"\n✓ Patch refinement details created")
    print(f"  - Number of refined patches: {len(patch_refinement_details)}")
    if patch_refinement_details:
        print(f"  - Sample detail keys: {list(patch_refinement_details[0].keys())}")

    print(f"\n✓ Patch statistics created")
    print(f"  - Total patches: {patch_stats['total_patches']}")
    print(f"  - Patches refined: {patch_stats['patches_refined']}")
    print(f"  - Mean error: {patch_stats['mean_error']:.4f}")
    print(f"  - Mean PSNR: {patch_stats['mean_psnr']:.2f} dB")
    print(f"  - Mean SSIM: {patch_stats['mean_ssim']:.4f}")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ADAPTIVE PATCHING OUTPUT TESTS")
    print("="*60)

    results = []

    # Run tests
    results.append(("Data Structures", test_data_structures()))
    results.append(("Tile Score Heatmap", test_tile_score_heatmap()))
    results.append(("Summary Text", test_summary_text()))

    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nGenerated test outputs in ./test_output/:")
        print("  - test_tile_scores_heatmap.png")
        print("  - test_summary.txt")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
