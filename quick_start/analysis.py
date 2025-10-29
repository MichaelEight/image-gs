"""Results analysis and visualization functions for Image-GS."""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Any, Optional

from .utils import get_paths, format_size, format_ratio


def _load_training_results(output_folder: str) -> Dict[str, Any]:
    """
    Load training results and parse configuration.

    Args:
        output_folder: Output folder name

    Returns:
        Dictionary containing loaded results and metadata
    """
    _, _, _, OUTPUT_DIR = get_paths()

    output_path = os.path.join(OUTPUT_DIR, output_folder)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output folder not found: {output_path}")

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

    # File paths
    model_path = os.path.join(output_path, "model.pt")
    rendered_path = os.path.join(output_path, "rendered.jpg")
    gt_path = os.path.join(output_path, "other", "ground_truth.jpg")
    initial_model_path = os.path.join(output_path, "initial_model.pt")

    # Load images
    gt_img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
    render_img = np.array(Image.open(rendered_path)).astype(np.float32) / 255.0

    # File sizes
    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else None
    gt_size = os.path.getsize(gt_path)
    render_size = os.path.getsize(rendered_path)

    # Check for initial model
    has_initial_model = os.path.exists(initial_model_path)
    initial_model_size = os.path.getsize(initial_model_path) if has_initial_model else None

    return {
        'output_path': output_path,
        'output_folder': output_folder,
        'base_name': base_name,
        'num_gaussians': num_gaussians,
        'max_steps': max_steps,
        'gt_img': gt_img,
        'render_img': render_img,
        'model_size': model_size,
        'gt_size': gt_size,
        'render_size': render_size,
        'has_initial_model': has_initial_model,
        'initial_model_path': initial_model_path if has_initial_model else None,
        'initial_model_size': initial_model_size
    }


def _calculate_quality_metrics(gt_img: np.ndarray, render_img: np.ndarray) -> Dict[str, Any]:
    """
    Calculate quality metrics between ground truth and rendered images.

    Args:
        gt_img: Ground truth image array
        render_img: Rendered image array

    Returns:
        Dictionary containing quality metrics
    """
    # Calculate difference
    diff = np.abs(gt_img - render_img)
    diff_gray = np.mean(diff, axis=2)

    # Basic statistics
    mean_diff = np.mean(diff_gray)
    max_diff = np.max(diff_gray)
    std_diff = np.std(diff_gray)

    # Pixel-level analysis
    pix_1pct = np.sum(diff_gray > 0.01) / diff_gray.size * 100
    pix_5pct = np.sum(diff_gray > 0.05) / diff_gray.size * 100
    pix_10pct = np.sum(diff_gray > 0.10) / diff_gray.size * 100

    return {
        'diff': diff,
        'diff_gray': diff_gray,
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'std_diff': std_diff,
        'pix_1pct': pix_1pct,
        'pix_5pct': pix_5pct,
        'pix_10pct': pix_10pct
    }


def _render_initial_gaussians(initial_model_path: str, input_filename: str) -> Optional[np.ndarray]:
    """
    Render initial Gaussians to visualize starting point.

    Args:
        initial_model_path: Path to initial model checkpoint
        input_filename: Input image filename

    Returns:
        Rendered image array, or None if rendering fails
    """
    import torch

    ROOT_WORKSPACE, REPO_DIR, INPUT_DIR, _ = get_paths()
    os.chdir(REPO_DIR)

    # Add repo to path if not already there
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    try:
        from model import GaussianSplatting2D
        from utils.misc_utils import load_cfg

        # Load initial checkpoint to get number of Gaussians
        checkpoint = torch.load(initial_model_path, weights_only=False)
        num_gaussians = checkpoint['state_dict']['xy'].shape[0]

        # Create temporary config for rendering
        args = load_cfg()
        args.input_path = f"images/{input_filename}"
        args.num_gaussians = num_gaussians
        args.eval = True
        args.ckpt_file = ""
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Create temporary directory for initial render
        temp_dir = os.path.join(REPO_DIR, "results", "temp_initial_render")
        os.makedirs(temp_dir, exist_ok=True)
        args.log_dir = temp_dir

        # Initialize model
        model = GaussianSplatting2D(args)

        # Load initial checkpoint state
        model.load_state_dict(checkpoint['state_dict'])

        # Render
        render_img = model.render()

        # Convert to numpy for visualization
        if torch.is_tensor(render_img):
            render_img = render_img.detach().cpu().permute(1, 2, 0).numpy()

        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        return render_img

    except Exception as e:
        print(f"Warning: Could not render initial Gaussians: {e}")
        return None


def _create_results_visualization(
    gt_img: np.ndarray,
    render_img: np.ndarray,
    diff_gray: np.ndarray,
    mean_diff: float,
    initial_img: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Create 3-panel or 4-panel comparison visualization.

    Args:
        gt_img: Ground truth image
        render_img: Rendered image
        diff_gray: Grayscale difference map
        mean_diff: Mean difference value
        initial_img: Optional initial Gaussians render

    Returns:
        Matplotlib figure
    """
    if initial_img is not None:
        # 4-panel layout with initial render
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()

        axes[0].imshow(initial_img)
        axes[0].set_title("Initial Gaussians (Starting Point)", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(render_img)
        axes[1].set_title("Final Gaussians (After Training)", fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(gt_img)
        axes[2].set_title("Ground Truth (Target)", fontsize=14, fontweight='bold')
        axes[2].axis('off')

        im = axes[3].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.2)
        axes[3].set_title(f"Difference Map (Final vs GT)\nMean: {mean_diff:.4f} ({mean_diff*100:.2f}%)", fontsize=14, fontweight='bold')
        axes[3].axis('off')

        cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Difference', rotation=270, labelpad=20)
    else:
        # Original 3-panel layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(gt_img)
        axes[0].set_title("Ground Truth", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(render_img)
        axes[1].set_title("2D Gaussians (Rendered)", fontsize=14, fontweight='bold')
        axes[1].axis('off')

        im = axes[2].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.2)
        axes[2].set_title(f"Difference Map\nMean: {mean_diff:.4f} ({mean_diff*100:.2f}%)", fontsize=14, fontweight='bold')
        axes[2].axis('off')

        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Difference', rotation=270, labelpad=20)

    plt.tight_layout()

    return fig


def _format_summary_text(results_data: Dict[str, Any], metrics_data: Dict[str, Any]) -> str:
    """
    Format summary text with tables.

    Args:
        results_data: Results metadata
        metrics_data: Quality metrics

    Returns:
        Formatted summary text
    """
    # Extract data
    output_folder = results_data['output_folder']
    base_name = results_data['base_name']
    num_gaussians = results_data['num_gaussians']
    max_steps = results_data['max_steps']
    gt_img = results_data['gt_img']
    model_size = results_data['model_size']
    gt_size = results_data['gt_size']
    render_size = results_data['render_size']
    has_initial_model = results_data['has_initial_model']
    initial_model_size = results_data['initial_model_size']

    mean_diff = metrics_data['mean_diff']
    max_diff = metrics_data['max_diff']
    std_diff = metrics_data['std_diff']
    pix_1pct = metrics_data['pix_1pct']
    pix_5pct = metrics_data['pix_5pct']
    pix_10pct = metrics_data['pix_10pct']

    # Image info
    height, width, channels = gt_img.shape
    total_pixels = width * height

    # Get initial Gaussian count if available
    initial_num_gaussians = None
    if has_initial_model:
        try:
            import torch
            checkpoint = torch.load(results_data['initial_model_path'], weights_only=False)
            if 'state_dict' in checkpoint and 'xy' in checkpoint['state_dict']:
                initial_num_gaussians = checkpoint['state_dict']['xy'].shape[0]
        except:
            pass

    # Build summary
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("IMAGE-GS TRAINING SUMMARY")
    summary_lines.append("=" * 100)
    summary_lines.append("")

    # Table 1: Training Configuration
    summary_lines.append("TRAINING CONFIGURATION")
    summary_lines.append("-" * 100)
    summary_lines.append(f"{'Output Folder':<25} {'Base Name':<20} {'Gaussians':<15} {'Steps':<15}")
    summary_lines.append(f"{output_folder:<25} {base_name:<20} {num_gaussians:<15} {max_steps:<15}")
    summary_lines.append("")

    # Table 1b: Initialization Information
    summary_lines.append("INITIALIZATION")
    summary_lines.append("-" * 100)
    if has_initial_model:
        summary_lines.append(f"{'Method':<30} {'Custom checkpoint (pretrained)':<70}")
        summary_lines.append(f"{'Initial Model Size':<30} {format_size(initial_model_size):<70}")
        if initial_num_gaussians:
            summary_lines.append(f"{'Initial Gaussian Count':<30} {f'{initial_num_gaussians:,}':<70}")
            if initial_num_gaussians == num_gaussians:
                summary_lines.append(f"{'Match Status':<30} {'Exact match (all Gaussians loaded)':<70}")
            elif initial_num_gaussians < num_gaussians:
                summary_lines.append(f"{'Match Status':<30} {f'Partial (loaded {initial_num_gaussians}, initialized {num_gaussians - initial_num_gaussians} more)':<70}")
            else:
                summary_lines.append(f"{'Match Status':<30} {f'Partial (loaded first {num_gaussians} of {initial_num_gaussians})':<70}")
    else:
        summary_lines.append(f"{'Method':<30} {'Default (random/gradient-based)':<70}")
        summary_lines.append(f"{'Source':<30} {'Generated from scratch':<70}")
    summary_lines.append("")

    # Table 2: Image & File Information
    summary_lines.append("IMAGE & FILE INFORMATION")
    summary_lines.append("-" * 100)
    summary_lines.append(f"{'Metric':<30} {'Value':<30} {'Metric':<30} {'Value':<30}")
    summary_lines.append(f"{'Resolution':<30} {f'{width} x {height} px':<30} {'Total Pixels':<30} {f'{total_pixels:,}':<30}")
    summary_lines.append(f"{'Channels':<30} {channels:<30} {'Ground Truth Size':<30} {format_size(gt_size):<30}")
    summary_lines.append(f"{'Model Size':<30} {format_size(model_size):<30} {'Rendered Size':<30} {format_size(render_size):<30}")
    summary_lines.append("")

    # Table 3: Compression Analysis
    if model_size:
        uncompressed_size = total_pixels * channels
        bpp = (model_size * 8) / total_pixels
        compression_vs_gt = gt_size / model_size
        compression_vs_raw = uncompressed_size / model_size

        summary_lines.append("COMPRESSION ANALYSIS")
        summary_lines.append("-" * 100)
        summary_lines.append(f"{'Metric':<40} {'Value':<30} {'Note':<30}")
        summary_lines.append(f"{'Compression vs Original (JPG)':<40} {format_ratio(gt_size, model_size):<30} {'Model is {:.1f}% of original'.format((model_size/gt_size)*100):<30}")
        summary_lines.append(f"{'Compression vs Raw (uncompressed)':<40} {format_ratio(uncompressed_size, model_size):<30} {f'{format_size(uncompressed_size)} -> {format_size(model_size)}':<30}")
        summary_lines.append(f"{'Bits Per Pixel (bpp)':<40} {f'{bpp:.4f} bpp':<30} {'Lower is better':<30}")
        summary_lines.append("")

    # Table 4: Quality Metrics
    summary_lines.append("QUALITY METRICS")
    summary_lines.append("-" * 100)
    summary_lines.append(f"{'Metric':<30} {'Value':<30} {'Metric':<30} {'Value':<30}")
    summary_lines.append(f"{'Mean Difference':<30} {f'{mean_diff:.6f} ({mean_diff*100:.2f}%)':<30} {'Max Difference':<30} {f'{max_diff:.6f} ({max_diff*100:.2f}%)':<30}")
    summary_lines.append(f"{'Std Deviation':<30} {f'{std_diff:.6f}':<30} {'Pixels > 1% diff':<30} {f'{pix_1pct:.2f}%':<30}")
    summary_lines.append(f"{'Pixels > 5% diff':<30} {f'{pix_5pct:.2f}%':<30} {'Pixels > 10% diff':<30} {f'{pix_10pct:.2f}%':<30}")
    summary_lines.append("")

    # Files saved
    summary_lines.append("FILES SAVED")
    summary_lines.append("-" * 100)
    summary_lines.append(f"{'File':<25} {'Description':<75}")
    summary_lines.append(f"{'📄 summary.txt':<25} {'This summary (text format)':<75}")
    summary_lines.append(f"{'📊 summary.png':<25} {'Visual comparison (3-panel image)':<75}")
    summary_lines.append(f"{'📈 metrics.csv':<25} {'Training metrics over iterations (CSV format)':<75}")
    summary_lines.append(f"{'📉 metrics_plot.png':<25} {'Training metrics visualization (6-panel plot)':<75}")
    summary_lines.append(f"{'🧠 model.pt':<25} {'Trained 2D Gaussian model (PyTorch checkpoint)':<75}")
    if has_initial_model:
        summary_lines.append(f"{'🔷 initial_model.pt':<25} {'Initial Gaussian model used for initialization':<75}")
    summary_lines.append(f"{'🖼️  rendered.jpg':<25} {'Rendered output from model':<75}")
    summary_lines.append(f"{'📁 other/':<25} {'Training logs and additional files':<75}")
    summary_lines.append("=" * 100)

    return "\n".join(summary_lines)


def _save_summary_files(output_path: str, fig: plt.Figure, summary_text: str) -> None:
    """
    Save visualization and summary text to files.

    Args:
        output_path: Path to output directory
        fig: Matplotlib figure
        summary_text: Formatted summary text
    """
    # Save visualization
    summary_img_path = os.path.join(output_path, "summary.png")
    fig.savefig(summary_img_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Save summary text
    summary_txt_path = os.path.join(output_path, "summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write(summary_text)

    print(summary_text)
    print()
    print(f"💾 Saved: summary.txt")
    print(f"💾 Saved: summary.png")
    print()
    print(f"📁 Full path: {output_path}")


def view_results(output_folder: str) -> None:
    """
    View and analyze results from a training run.

    This function loads training results, calculates quality metrics,
    creates visualizations, and saves summary files.

    Args:
        output_folder: Name of folder in output/ (e.g., "cat-10000-5000-20251027_143052")

    Example:
        >>> view_results("cat-5000-3500-20251027_143052")
    """
    results_data = _load_training_results(output_folder)
    metrics_data = _calculate_quality_metrics(results_data['gt_img'], results_data['render_img'])

    # Try to render initial Gaussians if available
    initial_img = None
    if results_data['has_initial_model']:
        print("Rendering initial Gaussians...")
        # Get input filename from base_name
        input_filename = f"{results_data['base_name']}.png"  # Assume PNG, might need adjustment
        initial_img = _render_initial_gaussians(results_data['initial_model_path'], input_filename)
        if initial_img is not None:
            print("✓ Initial Gaussians rendered successfully")
        else:
            print("⚠️  Could not render initial Gaussians (will use 3-panel layout)")

    fig = _create_results_visualization(
        results_data['gt_img'],
        results_data['render_img'],
        metrics_data['diff_gray'],
        metrics_data['mean_diff'],
        initial_img
    )
    summary_text = _format_summary_text(results_data, metrics_data)
    _save_summary_files(results_data['output_path'], fig, summary_text)
