"""
Adaptive patch-based refinement training workflow.

This module orchestrates the complete adaptive refinement process:
1. Base training on full image
2. Patch extraction and error analysis
3. Iterative refinement of poor-quality patches
4. Reconstruction and comparison visualization
"""

import os
import torch
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

from model import GaussianSplatting2D
from quick_start.config import AdaptiveRefinementConfig
from quick_start.patch_manager import PatchManager, PatchInfo
from utils.image_utils import save_image
from utils.patch_utils import calculate_patch_error


def train_adaptive(
    args,
    adaptive_config: AdaptiveRefinementConfig,
    workspace_dir: str,
    session_name: str
) -> Dict:
    """
    Main adaptive refinement training workflow.

    Args:
        args: Training arguments
        adaptive_config: Adaptive refinement configuration
        workspace_dir: Workspace directory path
        session_name: Session name for output organization

    Returns:
        Dictionary with training summary and paths
    """
    print("\n" + "="*80)
    print("ADAPTIVE PATCH-BASED REFINEMENT")
    print("="*80)

    # Setup output directories
    output_base = Path(workspace_dir) / "output" / session_name
    image_name = Path(args.input_path).stem
    run_name = f"{image_name}-adaptive-{adaptive_config.base_gaussians}"
    output_dir = output_base / run_name

    base_dir = output_dir / "base_training"
    enhanced_dir = output_dir / "enhanced"
    patches_dir = output_dir / "other" / "patches"
    comparison_dir = output_dir

    for d in [base_dir, enhanced_dir, patches_dir, comparison_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Base training: {adaptive_config.base_gaussians} gaussians")
    print(f"Patch size: {adaptive_config.patch_size}x{adaptive_config.patch_size}")
    print(f"Overlap: {adaptive_config.overlap}px")
    print(f"Error threshold: {adaptive_config.error_threshold}")

    # ========== PHASE 1: Base Training ==========
    print("\n" + "-"*80)
    print("PHASE 1: Base Training")
    print("-"*80)

    # Update args for base training
    args.num_gaussians = adaptive_config.base_gaussians
    args.log_dir = str(base_dir / "logs")
    os.makedirs(args.log_dir, exist_ok=True)

    # Create and train base model
    base_model = GaussianSplatting2D(args)

    if not args.evaluate:
        print(f"Training base model with {adaptive_config.base_gaussians} gaussians...")
        start_time = time.time()
        base_model.optimize()
        base_training_time = time.time() - start_time
        print(f"Base training complete in {base_training_time:.2f}s")

        # Save base model
        base_model_path = base_dir / "model.pt"
        base_model.save(str(base_model_path))
        print(f"Base model saved to {base_model_path}")

    # Render base image
    print("Rendering base image...")
    base_rendered = base_model._render_images()
    base_rendered_path = base_dir / "rendered.jpg"
    save_image(base_rendered, str(base_rendered_path), gamma=args.gamma)
    print(f"Base rendered image saved to {base_rendered_path}")

    # Get ground truth
    gt_images = base_model.gt_images
    image_shape = gt_images.shape  # (C, H, W)

    # ========== PHASE 2: Patch Analysis ==========
    print("\n" + "-"*80)
    print("PHASE 2: Patch Analysis")
    print("-"*80)

    # Create patch manager
    patch_manager = PatchManager(
        image_shape=image_shape,
        patch_size=adaptive_config.patch_size,
        overlap=adaptive_config.overlap,
        error_threshold=adaptive_config.error_threshold,
        psnr_weight=adaptive_config.psnr_weight,
        ssim_weight=adaptive_config.ssim_weight,
        device=args.device
    )

    print(f"Image split into {len(patch_manager.patches)} patches ({patch_manager.n_rows}x{patch_manager.n_cols})")

    # Split images into patches
    gt_patches = patch_manager.split_image(gt_images)
    base_patches = patch_manager.split_image(base_rendered)

    # Evaluate base patches
    print("Evaluating base patch quality...")
    base_stats = patch_manager.evaluate_patches(base_patches, gt_patches, gamma=args.gamma)

    print(f"\nBase Model Statistics:")
    print(f"  Mean PSNR: {base_stats['mean_psnr']:.2f} dB")
    print(f"  Mean SSIM: {base_stats['mean_ssim']:.4f}")
    print(f"  Mean Error: {base_stats['mean_error']:.4f}")
    print(f"  Patches needing refinement: {base_stats['num_needing_refinement']}/{base_stats['num_patches']} ({base_stats['refinement_ratio']*100:.1f}%)")

    # Save patch visualization
    patch_viz = patch_manager.create_visualization(gt_images, show_errors=True)
    patch_viz_path = output_dir / "other" / "patch_grid_visualization.jpg"
    save_image(patch_viz, str(patch_viz_path), gamma=1.0)
    print(f"Patch visualization saved to {patch_viz_path}")

    # Get patches needing refinement
    patches_to_refine = patch_manager.get_patches_needing_refinement()

    if len(patches_to_refine) == 0:
        print("\nNo patches need refinement! Base model is already excellent.")
        return {
            'output_dir': str(output_dir),
            'base_model_path': str(base_model_path),
            'patches_refined': 0,
            'base_stats': base_stats
        }

    print(f"\nPatches to refine: {len(patches_to_refine)}")

    # ========== PHASE 3: Iterative Patch Refinement ==========
    print("\n" + "-"*80)
    print("PHASE 3: Iterative Patch Refinement")
    print("-"*80)

    # Extract base model state for initialization
    base_model_state = {
        'xy': base_model.xy.detach().clone(),
        'scale': base_model.scale.detach().clone(),
        'rot': base_model.rot.detach().clone(),
        'feat': base_model.feat.detach().clone()
    }

    refined_patches_dict = {}  # Store refined patches

    total_refinement_time = 0
    patches_successfully_refined = 0

    for idx, patch_info in enumerate(patches_to_refine, 1):
        print(f"\n[{idx}/{len(patches_to_refine)}] Refining {patch_info.patch_id}")
        print(f"  Initial Error: {patch_info.combined_error:.4f} (threshold: {adaptive_config.error_threshold})")
        print(f"  Initial PSNR: {patch_info.psnr:.2f} dB, SSIM: {patch_info.ssim:.4f}")

        # Get patch data
        patch_idx = patch_manager.patches.index(patch_info)
        gt_patch = gt_patches[patch_idx]
        base_patch = base_patches[patch_idx]

        # Get patch bounds
        bounds = patch_manager.get_patch_bounds(patch_info)

        # Create patch output directory
        patch_output_dir = patches_dir / patch_info.patch_id
        patch_output_dir.mkdir(parents=True, exist_ok=True)

        # Save before images
        save_image(base_patch, str(patch_output_dir / "before.jpg"), gamma=args.gamma)
        save_image(gt_patch, str(patch_output_dir / "ground_truth.jpg"), gamma=args.gamma)

        # Calculate and save before error map
        before_metrics = calculate_patch_error(base_patch, gt_patch,
                                               adaptive_config.psnr_weight,
                                               adaptive_config.ssim_weight, args.gamma)
        before_error_map = torch.abs(base_patch - gt_patch).mean(dim=0, keepdim=True)
        save_image(before_error_map.repeat(3, 1, 1), str(patch_output_dir / "error_before.jpg"), gamma=1.0)

        # Iterative refinement
        current_gaussians = adaptive_config.base_gaussians
        best_patch = base_patch
        best_error = patch_info.combined_error
        iteration = 0

        for iteration in range(1, adaptive_config.max_refinement_iterations + 1):
            print(f"  Iteration {iteration}: Training with {current_gaussians} gaussians...")

            # Create new model for patch training
            patch_args = type('Args', (), {})()
            for attr in dir(args):
                if not attr.startswith('_'):
                    setattr(patch_args, attr, getattr(args, attr))

            patch_args.num_gaussians = current_gaussians
            patch_args.log_dir = str(patch_output_dir / f"logs_iter{iteration}")
            patch_args.evaluate = False
            os.makedirs(patch_args.log_dir, exist_ok=True)

            # Train patch
            patch_model = GaussianSplatting2D(patch_args)

            iter_start = time.time()
            patch_result = patch_model.train_patch(
                gt_patch,
                current_gaussians,
                max_steps=args.max_steps // 2,  # Use half steps for patches
                bounds=bounds,
                base_model_state=base_model_state,
                log_prefix=f"{patch_info.patch_id}_iter{iteration}"
            )
            iter_time = time.time() - iter_start
            total_refinement_time += iter_time

            # Render refined patch
            refined_patch = patch_model._render_images()

            # Calculate error
            refined_metrics = calculate_patch_error(
                refined_patch, gt_patch,
                adaptive_config.psnr_weight,
                adaptive_config.ssim_weight,
                args.gamma
            )

            print(f"    Result: Error {refined_metrics['combined_error']:.4f}, PSNR {refined_metrics['psnr']:.2f}, SSIM {refined_metrics['ssim']:.4f} ({iter_time:.1f}s)")

            # Check if improved
            if refined_metrics['combined_error'] < best_error:
                best_patch = refined_patch
                best_error = refined_metrics['combined_error']

            # Check if threshold met
            if refined_metrics['combined_error'] <= adaptive_config.error_threshold:
                print(f"    ✓ Threshold met!")
                patches_successfully_refined += 1
                break

            # Check gaussian limit
            if current_gaussians >= adaptive_config.max_gaussians_per_patch:
                print(f"    ! Max gaussians reached")
                break

            # Add more gaussians for next iteration
            current_gaussians += adaptive_config.refinement_gaussian_increment

        # Update patch manager with final results
        final_metrics = calculate_patch_error(best_patch, gt_patch,
                                              adaptive_config.psnr_weight,
                                              adaptive_config.ssim_weight, args.gamma)
        patch_manager.update_patch_refinement(
            patch_info.patch_id,
            iteration,
            current_gaussians,
            final_metrics['psnr'],
            final_metrics['ssim'],
            final_metrics['combined_error']
        )

        # Save refined patch
        refined_patches_dict[patch_idx] = best_patch
        save_image(best_patch, str(patch_output_dir / "after.jpg"), gamma=args.gamma)

        # Save after error map
        after_error_map = torch.abs(best_patch - gt_patch).mean(dim=0, keepdim=True)
        save_image(after_error_map.repeat(3, 1, 1), str(patch_output_dir / "error_after.jpg"), gamma=1.0)

        # Save patch metrics
        patch_metrics = {
            'patch_id': patch_info.patch_id,
            'before': before_metrics,
            'after': final_metrics,
            'iterations': iteration,
            'final_gaussians': current_gaussians,
            'training_time_seconds': total_refinement_time
        }
        with open(patch_output_dir / "metrics.json", 'w') as f:
            json.dump(patch_metrics, f, indent=2)

    # ========== PHASE 4: Reconstruction ==========
    print("\n" + "-"*80)
    print("PHASE 4: Reconstruction")
    print("-"*80)

    # Create enhanced patches list (base + refined)
    enhanced_patches = []
    for idx in range(len(base_patches)):
        if idx in refined_patches_dict:
            enhanced_patches.append(refined_patches_dict[idx])
        else:
            enhanced_patches.append(base_patches[idx])

    # Reconstruct enhanced image
    print("Reconstructing enhanced image from patches...")
    enhanced_image = patch_manager.reconstruct_image(enhanced_patches)
    enhanced_path = enhanced_dir / "rendered.jpg"
    save_image(enhanced_image, str(enhanced_path), gamma=args.gamma)
    print(f"Enhanced image saved to {enhanced_path}")

    # ========== PHASE 5: Comparison ==========
    print("\n" + "-"*80)
    print("PHASE 5: Comparison and Summary")
    print("-"*80)

    # Create 3-image comparison
    print("Creating comparison visualization...")
    comparison_image = create_comparison_image(gt_images, base_rendered, enhanced_image)
    comparison_path = comparison_dir / "comparison.jpg"
    save_image(comparison_image, str(comparison_path), gamma=args.gamma)
    print(f"Comparison image saved to {comparison_path}")

    # Calculate final statistics
    from utils.image_utils import get_psnr
    from utils.ssim import fused_ssim

    # Gamma correction for metrics
    gt_corrected = torch.pow(torch.clamp(gt_images, 0.0, 1.0), 1.0 / args.gamma)
    base_corrected = torch.pow(torch.clamp(base_rendered, 0.0, 1.0), 1.0 / args.gamma)
    enhanced_corrected = torch.pow(torch.clamp(enhanced_image, 0.0, 1.0), 1.0 / args.gamma)

    base_psnr = get_psnr(base_corrected, gt_corrected).item()
    base_ssim = fused_ssim(base_corrected.unsqueeze(0), gt_corrected.unsqueeze(0)).item()

    enhanced_psnr = get_psnr(enhanced_corrected, gt_corrected).item()
    enhanced_ssim = fused_ssim(enhanced_corrected.unsqueeze(0), gt_corrected.unsqueeze(0)).item()

    # Save patch summary
    patch_manager.save_summary(str(enhanced_dir / "patch_summary.json"))

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBase Model:")
    print(f"  PSNR: {base_psnr:.2f} dB")
    print(f"  SSIM: {base_ssim:.4f}")
    print(f"  Gaussians: {adaptive_config.base_gaussians}")

    print(f"\nEnhanced Model:")
    print(f"  PSNR: {enhanced_psnr:.2f} dB (+{enhanced_psnr - base_psnr:.2f})")
    print(f"  SSIM: {enhanced_ssim:.4f} (+{enhanced_ssim - base_ssim:.4f})")

    print(f"\nRefinement Statistics:")
    print(f"  Patches refined: {len(patches_to_refine)}")
    print(f"  Patches meeting threshold: {patches_successfully_refined}")
    print(f"  Total refinement time: {total_refinement_time:.2f}s")

    stats = patch_manager.get_statistics()
    print(f"  Total gaussians used: {stats['total_gaussians']}")
    print(f"  Avg gaussians per patch: {stats['mean_gaussians_per_patch']:.0f}")

    print(f"\nOutput directory: {output_dir}")
    print("="*80 + "\n")

    return {
        'output_dir': str(output_dir),
        'base_model_path': str(base_model_path),
        'enhanced_path': str(enhanced_path),
        'comparison_path': str(comparison_path),
        'patches_refined': len(patches_to_refine),
        'patches_successful': patches_successfully_refined,
        'base_psnr': base_psnr,
        'base_ssim': base_ssim,
        'enhanced_psnr': enhanced_psnr,
        'enhanced_ssim': enhanced_ssim,
        'total_refinement_time': total_refinement_time
    }


def create_comparison_image(original: torch.Tensor, base: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
    """
    Create side-by-side comparison of original, base, and enhanced images.

    Args:
        original: Original image (C, H, W)
        base: Base training result (C, H, W)
        enhanced: Enhanced result (C, H, W)

    Returns:
        Comparison image (C, H, W*3)
    """
    return torch.cat([original, base, enhanced], dim=2)
