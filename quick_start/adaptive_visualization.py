"""
Visualization module for adaptive patch-based refinement results.

This module provides comprehensive visualization functions for adaptive training,
including error maps, comparisons, and summary reports.
"""

import os
import numpy as np
import torch

# Safe matplotlib configuration (no LaTeX dependencies)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_flip_error_map(image: torch.Tensor, gt_image: torch.Tensor, gamma: float = 2.2):
    """
    Create FLIP-based perceptual error map.

    Args:
        image: Rendered image (C, H, W)
        gt_image: Ground truth image (C, H, W)
        gamma: Gamma correction value

    Returns:
        Error map as numpy array (H, W)
    """
    try:
        import flip_evaluator

        # Convert to numpy and apply gamma correction
        img_np = torch.pow(torch.clamp(image, 0.0, 1.0), 1.0 / gamma).cpu().numpy()
        gt_np = torch.pow(torch.clamp(gt_image, 0.0, 1.0), 1.0 / gamma).cpu().numpy()

        # Transpose to HWC format for FLIP
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
            gt_np = np.transpose(gt_np, (1, 2, 0))

        # Handle grayscale
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
            gt_np = np.repeat(gt_np, 3, axis=2)

        # Compute FLIP error
        error_map = flip_evaluator.compute_flip(gt_np, img_np)

        return error_map

    except (ImportError, Exception):
        # Fallback to simple L1 difference if FLIP not available
        diff = torch.abs(image - gt_image).mean(dim=0)
        return diff.cpu().numpy()


def create_simple_error_map(image: torch.Tensor, gt_image: torch.Tensor) -> np.ndarray:
    """
    Create simple L1 error map.

    Args:
        image: Rendered image (C, H, W)
        gt_image: Ground truth image (C, H, W)

    Returns:
        Error map as numpy array (H, W)
    """
    diff = torch.abs(image - gt_image).mean(dim=0)
    return diff.cpu().numpy()


def _tensor_to_numpy(tensor: torch.Tensor, gamma: float = 2.2) -> np.ndarray:
    """
    Convert tensor to numpy array for visualization.

    Args:
        tensor: Image tensor (C, H, W)
        gamma: Gamma correction value

    Returns:
        Numpy array in (H, W, C) format
    """
    # Apply gamma correction
    img = torch.pow(torch.clamp(tensor, 0.0, 1.0), 1.0 / gamma)
    img_np = img.cpu().numpy()

    # Handle grayscale
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)

    # Transpose to HWC
    img_np = np.transpose(img_np, (1, 2, 0))

    return np.clip(img_np, 0, 1)


def create_adaptive_summary_figure(
    gt_image: torch.Tensor,
    base_image: torch.Tensor,
    enhanced_image: torch.Tensor,
    patch_grid_viz: torch.Tensor,
    base_psnr: float,
    base_ssim: float,
    enhanced_psnr: float,
    enhanced_ssim: float,
    gamma: float = 2.2
):
    """
    Create comprehensive 6-panel summary figure for adaptive training.

    Args:
        gt_image: Ground truth image tensor (C, H, W)
        base_image: Base training result tensor (C, H, W)
        enhanced_image: Enhanced result tensor (C, H, W)
        patch_grid_viz: Patch grid visualization tensor (C, H, W)
        base_psnr: Base model PSNR
        base_ssim: Base model SSIM
        enhanced_psnr: Enhanced model PSNR
        enhanced_ssim: Enhanced model SSIM
        gamma: Gamma correction value

    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy
    gt_np = _tensor_to_numpy(gt_image, gamma)
    base_np = _tensor_to_numpy(base_image, gamma)
    enhanced_np = _tensor_to_numpy(enhanced_image, gamma)
    grid_np = _tensor_to_numpy(patch_grid_viz, gamma=1.0)  # No gamma for visualization

    # Create error maps
    base_error = create_flip_error_map(base_image, gt_image, gamma)
    enhanced_error = create_flip_error_map(enhanced_image, gt_image, gamma)

    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Row 2: Error maps and patch grid
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Plot images
    ax1.imshow(gt_np)
    ax1.set_title('Ground Truth\n(Original)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(base_np)
    ax2.set_title(f'Base Training\nPSNR: {base_psnr:.2f} dB | SSIM: {base_ssim:.4f}',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3.imshow(enhanced_np)
    improvement_psnr = enhanced_psnr - base_psnr
    improvement_ssim = enhanced_ssim - base_ssim
    ax3.set_title(f'Enhanced (Adaptive)\nPSNR: {enhanced_psnr:.2f} dB (+{improvement_psnr:.2f}) | SSIM: {enhanced_ssim:.4f} (+{improvement_ssim:.4f})',
                  fontsize=14, fontweight='bold', color='green')
    ax3.axis('off')

    # Plot error maps
    im4 = ax4.imshow(base_error, cmap='magma', vmin=0, vmax=0.2)
    ax4.set_title('Base Error Map\n(FLIP Metric)', fontsize=12)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Error')

    im5 = ax5.imshow(enhanced_error, cmap='magma', vmin=0, vmax=0.2)
    ax5.set_title('Enhanced Error Map\n(FLIP Metric)', fontsize=12)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='Error')

    # Plot patch grid
    ax6.imshow(grid_np)
    ax6.set_title('Patch Refinement Grid\nGreen = Good | Red = Refined', fontsize=12)
    ax6.axis('off')

    # Add main title
    fig.suptitle('Adaptive Patch-Based Refinement Results',
                fontsize=16, fontweight='bold', y=0.98)

    return fig


def create_tile_score_heatmap(
    patch_manager,
    error_threshold: float,
    output_path: str = None
):
    """
    Create standalone heatmap matrix showing error scores for all patches.

    Args:
        patch_manager: PatchManager instance with patch information
        error_threshold: Error threshold value to mark on visualization
        output_path: Optional path to save the heatmap

    Returns:
        Matplotlib figure
    """
    # Get patch grid dimensions
    n_rows = patch_manager.n_rows
    n_cols = patch_manager.n_cols

    # Create error matrix
    error_matrix = np.zeros((n_rows, n_cols))
    for patch_info in patch_manager.patches:
        error_matrix[patch_info.row, patch_info.col] = patch_info.combined_error

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.5), max(8, n_rows * 1.2)))

    # Create heatmap
    im = ax.imshow(error_matrix, cmap='RdYlGn_r', vmin=0, vmax=1.0, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Combined Error Score', fontsize=12, fontweight='bold')

    # Mark threshold on colorbar
    cbar.ax.axhline(y=error_threshold, color='blue', linewidth=2, linestyle='--')
    cbar.ax.text(1.5, error_threshold, f'Threshold: {error_threshold:.3f}',
                 va='center', ha='left', fontsize=10, fontweight='bold', color='blue')

    # Set ticks
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(np.arange(n_cols), fontsize=10)
    ax.set_yticklabels(np.arange(n_rows), fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    # Annotate cells with error values and patch IDs
    for patch_info in patch_manager.patches:
        row, col = patch_info.row, patch_info.col
        error = patch_info.combined_error

        # Choose text color based on error value
        text_color = 'white' if error > 0.5 else 'black'

        # Add error value
        ax.text(col, row, f'{error:.3f}',
                ha='center', va='center', fontsize=10,
                color=text_color, fontweight='bold')

        # Add patch ID below error value
        ax.text(col, row + 0.3, f'{patch_info.patch_id}',
                ha='center', va='center', fontsize=7,
                color=text_color, style='italic')

        # Add border for patches needing refinement
        if patch_info.needs_refinement:
            rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(rect)

    # Labels
    ax.set_xlabel('Column', fontsize=14, fontweight='bold')
    ax.set_ylabel('Row', fontsize=14, fontweight='bold')
    ax.set_title('Patch Error Score Heatmap (Base Training)\nRed Border = Needs Refinement',
                fontsize=16, fontweight='bold', pad=20)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label=f'Below threshold (<{error_threshold:.3f})'),
        Patch(facecolor='red', alpha=0.5, label=f'Above threshold (≥{error_threshold:.3f})'),
        Patch(facecolor='none', edgecolor='red', linewidth=3, label='Refined in adaptive training')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1),
             fontsize=11, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Tile score heatmap saved: {output_path}")

    return fig


def format_adaptive_summary_text(
    base_psnr: float,
    base_ssim: float,
    enhanced_psnr: float,
    enhanced_ssim: float,
    patch_stats: dict,
    training_time: float,
    base_gaussians: int,
    patch_refinement_details: list = None
) -> str:
    """
    Format detailed text summary for adaptive training.

    Args:
        base_psnr: Base model PSNR
        base_ssim: Base model SSIM
        enhanced_psnr: Enhanced model PSNR
        enhanced_ssim: Enhanced model SSIM
        patch_stats: Dictionary with patch statistics
        training_time: Total refinement time in seconds
        base_gaussians: Number of base gaussians
        patch_refinement_details: List of per-patch refinement details

    Returns:
        Formatted text summary
    """
    improvement_psnr = enhanced_psnr - base_psnr
    improvement_ssim = enhanced_ssim - base_ssim

    summary = []
    summary.append("="*80)
    summary.append("ADAPTIVE PATCH-BASED REFINEMENT SUMMARY")
    summary.append("="*80)
    summary.append("")

    # Adaptive Patching Status
    summary.append("ADAPTIVE PATCHING STATUS")
    summary.append("-"*80)
    if patch_refinement_details and len(patch_refinement_details) > 0:
        summary.append("✅ ACTIVATED - Adaptive patching was used")
        summary.append(f"   Refined {len(patch_refinement_details)} patches")
    else:
        summary.append("❌ NOT ACTIVATED - No patches needed refinement")
        summary.append("   Base model quality was sufficient for all patches")
    summary.append("")

    # Quality Metrics
    summary.append("QUALITY METRICS")
    summary.append("-"*80)
    summary.append(f"{'Metric':<20} {'Base Model':<20} {'Enhanced Model':<20} {'Improvement':<20}")
    summary.append("-"*80)
    summary.append(f"{'PSNR (dB)':<20} {base_psnr:<20.2f} {enhanced_psnr:<20.2f} {improvement_psnr:+<20.2f}")
    summary.append(f"{'SSIM':<20} {base_ssim:<20.4f} {enhanced_ssim:<20.4f} {improvement_ssim:+<20.4f}")
    summary.append("")

    # Patch Statistics
    summary.append("PATCH STATISTICS")
    summary.append("-"*80)
    summary.append(f"Total Patches: {patch_stats.get('total_patches', 0)}")
    summary.append(f"Patches Refined: {patch_stats.get('patches_refined', 0)}")
    summary.append(f"Patches Still Needing Refinement: {patch_stats.get('patches_still_needing_refinement', 0)}")
    summary.append(f"Mean Error: {patch_stats.get('mean_error', 0):.4f}")
    summary.append(f"Max Error: {patch_stats.get('max_error', 0):.4f}")
    summary.append(f"Min Error: {patch_stats.get('min_error', 0):.4f}")
    summary.append("")

    # Gaussian Usage
    summary.append("GAUSSIAN USAGE")
    summary.append("-"*80)
    summary.append(f"Base Model Gaussians: {base_gaussians}")
    summary.append(f"Total Gaussians (All Patches): {patch_stats.get('total_gaussians', 0)}")
    summary.append(f"Mean Gaussians per Patch: {patch_stats.get('mean_gaussians_per_patch', 0):.0f}")
    summary.append("")

    # Training Time
    summary.append("TRAINING TIME")
    summary.append("-"*80)
    summary.append(f"Refinement Time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    summary.append("")

    # Per-Patch Metrics
    if patch_stats.get('mean_psnr', 0) > 0:
        summary.append("PER-PATCH QUALITY")
        summary.append("-"*80)
        summary.append(f"Mean PSNR: {patch_stats.get('mean_psnr', 0):.2f} dB")
        summary.append(f"Mean SSIM: {patch_stats.get('mean_ssim', 0):.4f}")
        summary.append("")

    # Detailed Per-Patch Breakdown
    if patch_refinement_details and len(patch_refinement_details) > 0:
        summary.append("ADAPTIVE PATCHING DETAILS - PER-PATCH BREAKDOWN")
        summary.append("="*80)
        summary.append("")

        # Table header
        header = f"{'Patch ID':<12} {'Location':<10} {'Before (PSNR/SSIM/Error)':<28} {'After (PSNR/SSIM/Error)':<28} {'Improvement':<18} {'Gaussians':<12} {'Time (s)':<10}"
        summary.append(header)
        summary.append("-"*len(header))

        total_gaussians_added = 0
        total_time = 0

        for detail in patch_refinement_details:
            patch_id = detail['patch_id']
            location = f"R{detail['row']}C{detail['col']}"

            before = detail['before']
            after = detail['after']
            before_str = f"{before['psnr']:.1f}/{before['ssim']:.3f}/{before['combined_error']:.3f}"
            after_str = f"{after['psnr']:.1f}/{after['ssim']:.3f}/{after['combined_error']:.3f}"

            improvements = detail['improvements']
            improvement_str = f"+{improvements['psnr']:.2f}/+{improvements['ssim']:.3f}"

            gaussians_str = f"+{detail['gaussians_added']} ({detail['final_gaussians']})"
            time_str = f"{detail['training_time_seconds']:.1f}"

            # Threshold indicator
            threshold_mark = "✓" if detail.get('threshold_met', False) else "✗"

            row_str = f"{patch_id:<12} {location:<10} {before_str:<28} {after_str:<28} {improvement_str:<18} {gaussians_str:<12} {time_str:<10} {threshold_mark}"
            summary.append(row_str)

            total_gaussians_added += detail['gaussians_added']
            total_time += detail['training_time_seconds']

        summary.append("-"*len(header))

        # Summary row
        patches_met = sum(1 for d in patch_refinement_details if d.get('threshold_met', False))
        avg_improvement_psnr = sum(d['improvements']['psnr'] for d in patch_refinement_details) / len(patch_refinement_details)
        avg_improvement_ssim = sum(d['improvements']['ssim'] for d in patch_refinement_details) / len(patch_refinement_details)

        summary.append(f"\nSUMMARY:")
        summary.append(f"  Total patches refined: {len(patch_refinement_details)}")
        summary.append(f"  Patches meeting threshold: {patches_met}/{len(patch_refinement_details)}")
        summary.append(f"  Total gaussians added: {total_gaussians_added}")
        summary.append(f"  Average PSNR improvement: +{avg_improvement_psnr:.2f} dB")
        summary.append(f"  Average SSIM improvement: +{avg_improvement_ssim:.4f}")
        summary.append(f"  Total refinement time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        summary.append("")
        summary.append(f"Legend: ✓ = Threshold met, ✗ = Threshold not met")
        summary.append("")

    summary.append("="*80)

    return "\n".join(summary)


def view_adaptive_results(
    gt_image: torch.Tensor,
    base_image: torch.Tensor,
    enhanced_image: torch.Tensor,
    patch_grid_viz: torch.Tensor,
    base_psnr: float,
    base_ssim: float,
    enhanced_psnr: float,
    enhanced_ssim: float,
    patch_stats: dict,
    training_time: float,
    base_gaussians: int,
    output_dir: str,
    gamma: float = 2.2,
    patch_refinement_details: list = None,
    patch_manager = None,
    error_threshold: float = 0.3
):
    """
    Create and save comprehensive visualization for adaptive training results.

    Args:
        gt_image: Ground truth image tensor
        base_image: Base training result tensor
        enhanced_image: Enhanced result tensor
        patch_grid_viz: Patch grid visualization tensor
        base_psnr: Base model PSNR
        base_ssim: Base model SSIM
        enhanced_psnr: Enhanced model PSNR
        enhanced_ssim: Enhanced model SSIM
        patch_stats: Dictionary with patch statistics
        training_time: Total refinement time
        base_gaussians: Number of base gaussians
        output_dir: Output directory path
        gamma: Gamma correction value
        patch_refinement_details: List of per-patch refinement details
        patch_manager: PatchManager instance for creating heatmap
        error_threshold: Error threshold value
    """
    print("\nGenerating comprehensive visualization...")

    # Create summary figure
    fig = create_adaptive_summary_figure(
        gt_image, base_image, enhanced_image, patch_grid_viz,
        base_psnr, base_ssim, enhanced_psnr, enhanced_ssim, gamma
    )

    # Save summary figure
    summary_path = os.path.join(output_dir, "summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Summary visualization saved: {summary_path}")

    # Create tile score heatmap if patch manager provided
    if patch_manager is not None:
        heatmap_path = os.path.join(output_dir, "tile_scores_heatmap.png")
        heatmap_fig = create_tile_score_heatmap(
            patch_manager,
            error_threshold,
            output_path=heatmap_path
        )
        plt.close(heatmap_fig)

    # Create and save text summary
    summary_text = format_adaptive_summary_text(
        base_psnr, base_ssim, enhanced_psnr, enhanced_ssim,
        patch_stats, training_time, base_gaussians,
        patch_refinement_details=patch_refinement_details
    )

    summary_text_path = os.path.join(output_dir, "summary.txt")
    with open(summary_text_path, 'w') as f:
        f.write(summary_text)
    print(f"[OK] Summary text saved: {summary_text_path}")

    # Print summary to console
    print("\n" + summary_text)


def create_enhanced_comparison_image(
    gt_image: torch.Tensor,
    base_image: torch.Tensor,
    enhanced_image: torch.Tensor,
    base_psnr: float,
    base_ssim: float,
    enhanced_psnr: float,
    enhanced_ssim: float,
    gamma: float = 2.2
):
    """
    Create rich comparison visualization with error maps.

    Args:
        gt_image: Ground truth image tensor (C, H, W)
        base_image: Base training result tensor (C, H, W)
        enhanced_image: Enhanced result tensor (C, H, W)
        base_psnr: Base model PSNR
        base_ssim: Base model SSIM
        enhanced_psnr: Enhanced model PSNR
        enhanced_ssim: Enhanced model SSIM
        gamma: Gamma correction value

    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy
    gt_np = _tensor_to_numpy(gt_image, gamma)
    base_np = _tensor_to_numpy(base_image, gamma)
    enhanced_np = _tensor_to_numpy(enhanced_image, gamma)

    # Create error maps
    base_error = create_simple_error_map(base_image, gt_image)
    enhanced_error = create_simple_error_map(enhanced_image, gt_image)

    # Create figure
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15, height_ratios=[3, 1])

    # Top row: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Bottom row: Error maps
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Plot images
    ax1.imshow(gt_np)
    ax1.set_title('Original', fontsize=16, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(base_np)
    ax2.set_title(f'Base Training\n{base_psnr:.2f} dB / {base_ssim:.4f}',
                  fontsize=16, fontweight='bold')
    ax2.axis('off')

    ax3.imshow(enhanced_np)
    improvement = enhanced_psnr - base_psnr
    ax3.set_title(f'Enhanced\n{enhanced_psnr:.2f} dB (+{improvement:.2f}) / {enhanced_ssim:.4f}',
                  fontsize=16, fontweight='bold', color='green')
    ax3.axis('off')

    # Error maps (smaller)
    ax4.axis('off')  # Empty for original

    im5 = ax5.imshow(base_error, cmap='hot', vmin=0, vmax=0.2)
    ax5.set_title('Base Error', fontsize=10)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    im6 = ax6.imshow(enhanced_error, cmap='hot', vmin=0, vmax=0.2)
    ax6.set_title('Enhanced Error', fontsize=10)
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    fig.suptitle('Adaptive Refinement Comparison', fontsize=18, fontweight='bold', y=0.98)

    return fig
