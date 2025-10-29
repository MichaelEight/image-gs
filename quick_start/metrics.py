"""Metrics loading and plotting functions for Image-GS."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from .utils import get_paths


def load_metrics(output_folder: str) -> pd.DataFrame:
    """
    Load training metrics from CSV file.

    Args:
        output_folder: Name of folder in output/ with training results

    Returns:
        pandas.DataFrame with columns:
            step, total_loss, l1_loss, l2_loss, ssim_loss, psnr, ssim,
            num_gaussians, num_bytes, render_time_accum, total_time_accum

    Raises:
        FileNotFoundError: If output folder or metrics CSV not found

    Example:
        >>> df = load_metrics("cat-5000-3500-20251027_143052")
        >>> print(df[['step', 'psnr', 'ssim']])
    """
    _, _, _, OUTPUT_DIR = get_paths()

    output_path = os.path.join(OUTPUT_DIR, output_folder)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output folder not found: {output_path}")

    metrics_csv_path = os.path.join(output_path, "metrics.csv")
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv_path}")

    # Load CSV
    df = pd.read_csv(metrics_csv_path)

    # Convert empty strings to NaN for numeric columns
    numeric_cols = ['total_loss', 'l1_loss', 'l2_loss', 'ssim_loss']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_metrics(output_folder: str, save_plot: bool = True, show_plot: bool = True) -> None:
    """
    Plot comprehensive training metrics over iterations.

    Creates a multi-panel visualization showing:
    - Loss curves (total and components)
    - Quality metrics (PSNR, SSIM)
    - Model growth (size, gaussian count)
    - Timing information

    Args:
        output_folder: Name of folder in output/ with training results
        save_plot: Save plot as metrics_plot.png (default: True)
        show_plot: Display plot inline (default: True)

    Example:
        >>> plot_metrics("cat-5000-3500-20251027_143052")
    """
    # Load metrics
    df = load_metrics(output_folder)

    _, _, _, OUTPUT_DIR = get_paths()
    output_path = os.path.join(OUTPUT_DIR, output_folder)

    # Parse config from folder name
    parts = output_folder.rsplit("-", 3)
    if len(parts) >= 3:
        base_name = parts[0]
        num_gaussians = int(parts[1])
        max_steps = int(parts[2])
    else:
        base_name = output_folder
        num_gaussians = "?"
        max_steps = "?"

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Title
    fig.suptitle(f'Training Metrics: {base_name} (G={num_gaussians}, Steps={max_steps})',
                 fontsize=16, fontweight='bold', y=0.995)

    # 1. Loss curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['step'], df['total_loss'], 'k-', linewidth=2, label='Total Loss')
    if 'l1_loss' in df.columns and df['l1_loss'].notna().any():
        ax1.plot(df['step'], df['l1_loss'], '--', alpha=0.7, label='L1 Loss')
    if 'l2_loss' in df.columns and df['l2_loss'].notna().any():
        ax1.plot(df['step'], df['l2_loss'], '--', alpha=0.7, label='L2 Loss')
    if 'ssim_loss' in df.columns and df['ssim_loss'].notna().any():
        ax1.plot(df['step'], df['ssim_loss'], '--', alpha=0.7, label='SSIM Loss')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. PSNR (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['step'], df['psnr'], 'b-', linewidth=2)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Peak Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # Add final value annotation
    final_psnr = df['psnr'].iloc[-1]
    ax2.axhline(y=final_psnr, color='b', linestyle='--', alpha=0.3)
    ax2.text(0.98, 0.02, f'Final: {final_psnr:.2f} dB',
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. SSIM (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['step'], df['ssim'], 'g-', linewidth=2)
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('SSIM', fontsize=12)
    ax3.set_title('Structural Similarity Index', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # Add final value annotation
    final_ssim = df['ssim'].iloc[-1]
    ax3.axhline(y=final_ssim, color='g', linestyle='--', alpha=0.3)
    ax3.text(0.98, 0.02, f'Final: {final_ssim:.4f}',
             transform=ax3.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. Model size (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    # Convert bytes to KB
    df['size_kb'] = df['num_bytes'] / 1024
    ax4.plot(df['step'], df['size_kb'], 'r-', linewidth=2)
    ax4.set_xlabel('Training Step', fontsize=12)
    ax4.set_ylabel('Model Size (KB)', fontsize=12)
    ax4.set_title('Model Size Growth', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    # Add final value annotation
    final_size = df['size_kb'].iloc[-1]
    ax4.text(0.98, 0.98, f'Final: {final_size:.2f} KB',
             transform=ax4.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5. Gaussian count (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df['step'], df['num_gaussians'], 'm-', linewidth=2)
    ax5.set_xlabel('Training Step', fontsize=12)
    ax5.set_ylabel('Number of Gaussians', fontsize=12)
    ax5.set_title('Gaussian Count', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    # Add final value annotation
    final_gaussians = df['num_gaussians'].iloc[-1]
    ax5.text(0.98, 0.98, f'Final: {int(final_gaussians):,}',
             transform=ax5.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 6. Timing (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(df['step'], df['total_time_accum'], 'c-', linewidth=2, label='Total Time')
    ax6.plot(df['step'], df['render_time_accum'], 'orange', linewidth=2, label='Render Time')
    ax6.set_xlabel('Training Step', fontsize=12)
    ax6.set_ylabel('Time (seconds)', fontsize=12)
    ax6.set_title('Accumulated Time', fontsize=14, fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    # Add final value annotation
    final_time = df['total_time_accum'].iloc[-1]
    ax6.text(0.98, 0.02, f'Total: {final_time:.1f}s ({final_time/60:.1f}m)',
             transform=ax6.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save plot
    if save_plot:
        plot_path = os.path.join(output_path, "metrics_plot.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: metrics_plot.png")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    if save_plot:
        print(f"📁 Location: {output_path}")
