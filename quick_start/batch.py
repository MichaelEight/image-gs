"""Batch training comparison functions for Image-GS."""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Any

from .utils import get_paths


def _collect_batch_metrics(output_folders: List[str]) -> List[Dict[str, Any]]:
    """
    Collect metrics from all batch training results.

    Args:
        output_folders: List of output folder names

    Returns:
        List of dictionaries containing metrics for each run
    """
    _, _, _, OUTPUT_DIR = get_paths()

    results = []

    for folder in output_folders:
        output_path = os.path.join(OUTPUT_DIR, folder)

        # Parse config
        parts = folder.rsplit("-", 3)
        if len(parts) < 3:
            continue

        base_name = parts[0]
        num_gaussians = int(parts[1])
        max_steps = int(parts[2])

        # File paths
        model_path = os.path.join(output_path, "model.pt")
        rendered_path = os.path.join(output_path, "rendered.jpg")
        gt_path = os.path.join(output_path, "other", "ground_truth.jpg")

        if not all([os.path.exists(model_path), os.path.exists(rendered_path), os.path.exists(gt_path)]):
            continue

        # Get sizes
        model_size = os.path.getsize(model_path)
        gt_size = os.path.getsize(gt_path)

        # Load images and calculate metrics
        gt_img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        render_img = np.array(Image.open(rendered_path)).astype(np.float32) / 255.0

        diff = np.abs(gt_img - render_img)
        diff_gray = np.mean(diff, axis=2)

        mean_diff = np.mean(diff_gray)
        max_diff = np.max(diff_gray)

        # Calculate compression
        height, width, channels = gt_img.shape
        total_pixels = width * height
        bpp = (model_size * 8) / total_pixels
        compression_ratio = gt_size / model_size

        results.append({
            'folder': folder,
            'gaussians': num_gaussians,
            'steps': max_steps,
            'model_size_kb': model_size / 1024,
            'compression': compression_ratio,
            'bpp': bpp,
            'mean_diff': mean_diff,
            'max_diff': max_diff
        })

    # Sort by gaussians, then steps
    results.sort(key=lambda x: (x['gaussians'], x['steps']))

    return results


def _print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """
    Print comparison table for batch results.

    Args:
        results: List of result dictionaries
    """
    print("=" * 120)
    print("BATCH TRAINING COMPARISON")
    print("=" * 120)
    print(f"{'Gaussians':<12} {'Steps':<8} {'Model Size':<14} {'Compression':<14} {'BPP':<10} {'Mean Diff':<14} {'Max Diff':<14}")
    print("-" * 120)

    for r in results:
        print(f"{r['gaussians']:<12} {r['steps']:<8} {r['model_size_kb']:>10.2f} KB  {r['compression']:>10.2f}x  {r['bpp']:>8.4f}  {r['mean_diff']:>10.6f}  {r['max_diff']:>10.6f}")

    print("=" * 120)


def _print_highlights(results: List[Dict[str, Any]]) -> None:
    """
    Print highlights showing best compression and quality.

    Args:
        results: List of result dictionaries
    """
    best_compression = max(results, key=lambda x: x['compression'])
    best_quality = min(results, key=lambda x: x['mean_diff'])

    print()
    print("HIGHLIGHTS:")
    print(f"  Best Compression: G={best_compression['gaussians']}, S={best_compression['steps']} -> {best_compression['compression']:.2f}x")
    print(f"  Best Quality:     G={best_quality['gaussians']}, S={best_quality['steps']} -> Mean Diff={best_quality['mean_diff']:.6f}")
    print("=" * 120)


def compare_batch_results(output_folders: List[str]) -> List[Dict[str, Any]]:
    """
    Compare metrics across all trained models.

    This function collects metrics from multiple training runs and displays
    a comparison table showing model size, compression ratio, quality metrics,
    and performance highlights.

    Args:
        output_folders: List of output folder names from batch training

    Returns:
        List of dictionaries containing metrics for each run

    Example:
        >>> results = train(config)  # Returns list of folder names
        >>> comparison = compare_batch_results(results)
    """
    results = _collect_batch_metrics(output_folders)

    if not results:
        print("No results to compare")
        return []

    _print_comparison_table(results)
    _print_highlights(results)

    return results
