"""
Adaptive patch-based refinement training workflow.

This module orchestrates the complete adaptive refinement process:
1. Base training on full image
2. Patch extraction and error analysis
3. Iterative refinement of poor-quality patches
4. Reconstruction and comparison visualization
"""

import os
import shutil
import torch
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

# Safe matplotlib configuration (no LaTeX dependencies)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import matplotlib.pyplot as plt

from model import GaussianSplatting2D
from quick_start.config import AdaptiveRefinementConfig
from quick_start.patch_manager import PatchManager, PatchInfo
from quick_start.adaptive_visualization import (
    view_adaptive_results,
    create_enhanced_comparison_image,
    create_flip_error_map
)
from utils.image_utils import save_image
from utils.misc_utils import get_latest_ckpt_step
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

    if not args.eval:
        print(f"Training base model with {adaptive_config.base_gaussians} gaussians...")
        start_time = time.time()
        base_model.optimize()
        base_training_time = time.time() - start_time
        print(f"Base training complete in {base_training_time:.2f}s")

        # Copy base model checkpoint
        # The checkpoint is automatically saved during optimize() in ckpt_dir
        base_model_path = base_dir / "model.pt"
        ckpt_dir = Path(base_model.ckpt_dir)
        latest_step = get_latest_ckpt_step(str(ckpt_dir))
        if latest_step > 0:
            src_ckpt = ckpt_dir / f"ckpt_step-{latest_step}.pt"
            shutil.copy2(src_ckpt, base_model_path)
            print(f"Base model checkpoint copied to {base_model_path}")
        else:
            print("Warning: No checkpoint found to copy")

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
    print(f"\n🔄 ADAPTIVE PATCHING ACTIVATED")
    print(f"   Refining {len(patches_to_refine)} patches that exceed error threshold")
    print(f"   Configuration:")
    print(f"     - Error threshold: {adaptive_config.error_threshold}")
    print(f"     - Max iterations per patch: {adaptive_config.max_refinement_iterations}")
    print(f"     - Gaussian increment: +{adaptive_config.refinement_gaussian_increment} per iteration")
    print(f"     - Max gaussians per patch: {adaptive_config.max_gaussians_per_patch}")

    # Extract base model state for initialization
    base_model_state = {
        'xy': base_model.xy.detach().clone(),
        'scale': base_model.scale.detach().clone(),
        'rot': base_model.rot.detach().clone(),
        'feat': base_model.feat.detach().clone()
    }

    refined_patches_dict = {}  # Store refined patches
    patch_refinement_details = []  # Track detailed metrics for summary

    total_refinement_time = 0
    patches_successfully_refined = 0

    for idx, patch_info in enumerate(patches_to_refine, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(patches_to_refine)}] Processing {patch_info.patch_id}")
        print(f"  Location: Row {patch_info.row}, Col {patch_info.col}")
        print(f"  Coordinates: {patch_info.coordinates}")
        print(f"  Initial Error: {patch_info.combined_error:.4f} (threshold: {adaptive_config.error_threshold})")
        print(f"  Initial PSNR: {patch_info.psnr:.2f} dB, SSIM: {patch_info.ssim:.4f}")
        print(f"{'='*60}")

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
        # Use FLIP error map for better perceptual error visualization
        before_error_map = create_flip_error_map(base_patch, gt_patch, args.gamma)
        # Convert to RGB for visualization
        before_error_rgb = plt.cm.magma(before_error_map / 0.2)[:, :, :3]  # Normalize to 0-0.2 range
        before_error_tensor = torch.from_numpy(before_error_rgb).permute(2, 0, 1).float()
        save_image(before_error_tensor, str(patch_output_dir / "error_before.jpg"), gamma=1.0)

        # Iterative refinement
        patch_start_time = time.time()
        current_gaussians = adaptive_config.base_gaussians
        best_patch = base_patch
        best_error = patch_info.combined_error
        iteration = 0
        gaussians_added = 0

        print(f"\n  Starting refinement iterations...")
        for iteration in range(1, adaptive_config.max_refinement_iterations + 1):
            print(f"\n  📊 Iteration {iteration}/{adaptive_config.max_refinement_iterations}")
            print(f"     Training with {current_gaussians} gaussians...")

            # Create new model for patch training
            patch_args = type('Args', (), {})()
            for attr in dir(args):
                if not attr.startswith('_'):
                    setattr(patch_args, attr, getattr(args, attr))

            patch_args.num_gaussians = current_gaussians
            patch_args.log_dir = str(patch_output_dir / f"logs_iter{iteration}")
            patch_args.eval = False
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

            # Calculate improvements
            psnr_improvement = refined_metrics['psnr'] - before_metrics['psnr']
            ssim_improvement = refined_metrics['ssim'] - before_metrics['ssim']
            error_reduction = before_metrics['combined_error'] - refined_metrics['combined_error']

            print(f"     Results:")
            print(f"       Error: {refined_metrics['combined_error']:.4f} (Δ {-error_reduction:+.4f})")
            print(f"       PSNR:  {refined_metrics['psnr']:.2f} dB (Δ {psnr_improvement:+.2f})")
            print(f"       SSIM:  {refined_metrics['ssim']:.4f} (Δ {ssim_improvement:+.4f})")
            print(f"       Time:  {iter_time:.1f}s")

            # Check if improved
            if refined_metrics['combined_error'] < best_error:
                best_patch = refined_patch
                best_error = refined_metrics['combined_error']

            # Check if threshold met
            if refined_metrics['combined_error'] <= adaptive_config.error_threshold:
                print(f"\n     ✅ SUCCESS: Threshold met!")
                print(f"        Target: {adaptive_config.error_threshold:.4f}")
                print(f"        Achieved: {refined_metrics['combined_error']:.4f}")
                patches_successfully_refined += 1
                break

            # Check gaussian limit
            if current_gaussians >= adaptive_config.max_gaussians_per_patch:
                print(f"\n     ⚠️  Max gaussians limit reached ({adaptive_config.max_gaussians_per_patch})")
                break

            # Add more gaussians for next iteration
            current_gaussians += adaptive_config.refinement_gaussian_increment
            gaussians_added += adaptive_config.refinement_gaussian_increment

        # Update patch manager with final results
        patch_total_time = time.time() - patch_start_time
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

        # Save after error map using FLIP
        after_error_map = create_flip_error_map(best_patch, gt_patch, args.gamma)
        # Convert to RGB for visualization
        after_error_rgb = plt.cm.magma(after_error_map / 0.2)[:, :, :3]  # Normalize to 0-0.2 range
        after_error_tensor = torch.from_numpy(after_error_rgb).permute(2, 0, 1).float()
        save_image(after_error_tensor, str(patch_output_dir / "error_after.jpg"), gamma=1.0)

        # Calculate total improvements
        total_psnr_improvement = final_metrics['psnr'] - before_metrics['psnr']
        total_ssim_improvement = final_metrics['ssim'] - before_metrics['ssim']
        total_error_reduction = before_metrics['combined_error'] - final_metrics['combined_error']

        # Print patch summary
        print(f"\n  📈 Patch Summary:")
        print(f"     Iterations completed: {iteration}")
        print(f"     Gaussians added: {gaussians_added} (final: {current_gaussians})")
        print(f"     Total time: {patch_total_time:.1f}s")
        print(f"     Quality improvement:")
        print(f"       PSNR:  {before_metrics['psnr']:.2f} → {final_metrics['psnr']:.2f} dB ({total_psnr_improvement:+.2f})")
        print(f"       SSIM:  {before_metrics['ssim']:.4f} → {final_metrics['ssim']:.4f} ({total_ssim_improvement:+.4f})")
        print(f"       Error: {before_metrics['combined_error']:.4f} → {final_metrics['combined_error']:.4f} ({-total_error_reduction:+.4f})")

        threshold_status = "✅ Met" if final_metrics['combined_error'] <= adaptive_config.error_threshold else "⚠️ Not met"
        print(f"     Threshold status: {threshold_status}")

        # Save patch metrics
        patch_metrics = {
            'patch_id': patch_info.patch_id,
            'row': patch_info.row,
            'col': patch_info.col,
            'before': before_metrics,
            'after': final_metrics,
            'improvements': {
                'psnr': total_psnr_improvement,
                'ssim': total_ssim_improvement,
                'error_reduction': total_error_reduction
            },
            'iterations': iteration,
            'gaussians_added': gaussians_added,
            'final_gaussians': current_gaussians,
            'training_time_seconds': patch_total_time,
            'threshold_met': final_metrics['combined_error'] <= adaptive_config.error_threshold
        }
        patch_refinement_details.append(patch_metrics)

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

    # Calculate final statistics first (needed for visualizations)
    from utils.image_utils import get_psnr
    from fused_ssim import fused_ssim

    # Gamma correction for metrics
    gt_corrected = torch.pow(torch.clamp(gt_images, 0.0, 1.0), 1.0 / args.gamma)
    base_corrected = torch.pow(torch.clamp(base_rendered, 0.0, 1.0), 1.0 / args.gamma)
    enhanced_corrected = torch.pow(torch.clamp(enhanced_image, 0.0, 1.0), 1.0 / args.gamma)

    base_psnr = get_psnr(base_corrected, gt_corrected).item()
    base_ssim = fused_ssim(base_corrected.unsqueeze(0), gt_corrected.unsqueeze(0)).item()

    enhanced_psnr = get_psnr(enhanced_corrected, gt_corrected).item()
    enhanced_ssim = fused_ssim(enhanced_corrected.unsqueeze(0), gt_corrected.unsqueeze(0)).item()

    # Create 3-image comparison (simple concatenation for backward compatibility)
    print("Creating simple comparison image...")
    comparison_image = create_comparison_image(gt_images, base_rendered, enhanced_image)
    comparison_path = comparison_dir / "comparison.jpg"
    save_image(comparison_image, str(comparison_path), gamma=args.gamma)
    print(f"Comparison image saved to {comparison_path}")

    # Create enhanced comparison with matplotlib (with error maps)
    print("Creating enhanced comparison visualization...")
    try:
        comparison_fig = create_enhanced_comparison_image(
            gt_images, base_rendered, enhanced_image,
            base_psnr, base_ssim, enhanced_psnr, enhanced_ssim,
            gamma=args.gamma
        )
        comparison_enhanced_path = comparison_dir / "comparison_detailed.png"
        comparison_fig.savefig(str(comparison_enhanced_path), dpi=150, bbox_inches='tight')
        plt.close(comparison_fig)
        print(f"Enhanced comparison saved to {comparison_enhanced_path}")
    except Exception as e:
        print(f"Warning: Could not create enhanced comparison: {e}")

    # Save patch summary
    patch_manager.save_summary(str(enhanced_dir / "patch_summary.json"))

    # Get patch statistics
    stats = patch_manager.get_statistics()

    # Create comprehensive summary visualization
    print("Creating comprehensive summary visualization...")
    try:
        view_adaptive_results(
            gt_image=gt_images,
            base_image=base_rendered,
            enhanced_image=enhanced_image,
            patch_grid_viz=patch_viz,
            base_psnr=base_psnr,
            base_ssim=base_ssim,
            enhanced_psnr=enhanced_psnr,
            enhanced_ssim=enhanced_ssim,
            patch_stats=stats,
            training_time=total_refinement_time,
            base_gaussians=adaptive_config.base_gaussians,
            output_dir=str(output_dir),
            gamma=args.gamma,
            patch_refinement_details=patch_refinement_details,
            patch_manager=patch_manager,
            error_threshold=adaptive_config.error_threshold
        )
    except Exception as e:
        print(f"Warning: Could not create summary visualization: {e}")
        import traceback
        traceback.print_exc()

        # Print summary manually
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
