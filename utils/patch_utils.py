"""
Patch utilities for adaptive Gaussian splatting refinement.

This module provides functions for:
- Splitting images into overlapping patches
- Reconstructing images from patches with blending
- Calculating per-patch error metrics
- Visualizing patch grids and error distributions
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from .image_utils import get_psnr
from fused_ssim import fused_ssim


def split_into_patches(
    image: torch.Tensor,
    patch_size: int,
    overlap: int,
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
    """
    Split an image into overlapping patches.

    Args:
        image: Input image tensor of shape (C, H, W) or (1, C, H, W)
        patch_size: Size of each square patch
        overlap: Number of pixels to overlap between adjacent patches
        device: Device to place patches on

    Returns:
        patches: List of patch tensors, each of shape (C, patch_size, patch_size)
        coordinates: List of (x_start, y_start, x_end, y_end) coordinates for each patch
    """
    # Handle batch dimension
    if image.dim() == 4:
        image = image.squeeze(0)

    C, H, W = image.shape
    stride = patch_size - overlap

    patches = []
    coordinates = []

    # Calculate number of patches in each dimension
    n_patches_h = (H - overlap + stride - 1) // stride
    n_patches_w = (W - overlap + stride - 1) // stride

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate patch boundaries
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + patch_size, H)
            x_end = min(x_start + patch_size, W)

            # Extract patch
            patch = image[:, y_start:y_end, x_start:x_end]

            # Pad if necessary (for edge patches)
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                pad_h = patch_size - patch.shape[1]
                pad_w = patch_size - patch.shape[2]

                # Choose padding mode based on patch size
                # Reflect padding requires padding size < input dimension
                # Use replicate if patch is too small for reflect
                if pad_h < patch.shape[1] and pad_w < patch.shape[2]:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    # For very small patches, use replicate padding
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')

            patches.append(patch.to(device))
            coordinates.append((x_start, y_start, x_end, y_end))

    return patches, coordinates


def create_blend_weight_map(
    patch_size: int,
    overlap: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create a blending weight map for smooth patch transitions.
    Uses linear blending in the overlap regions.

    Args:
        patch_size: Size of the square patch
        overlap: Overlap size in pixels
        device: Device to create tensor on

    Returns:
        weight_map: Tensor of shape (patch_size, patch_size) with values in [0, 1]
    """
    weight_map = torch.ones(patch_size, patch_size, device=device)

    if overlap == 0:
        return weight_map

    # Create linear ramps for blending in overlap regions
    ramp = torch.linspace(0, 1, overlap, device=device)

    # Top edge
    weight_map[:overlap, :] = weight_map[:overlap, :] * ramp.unsqueeze(1)

    # Bottom edge
    weight_map[-overlap:, :] = weight_map[-overlap:, :] * ramp.flip(0).unsqueeze(1)

    # Left edge
    weight_map[:, :overlap] = weight_map[:, :overlap] * ramp.unsqueeze(0)

    # Right edge
    weight_map[:, -overlap:] = weight_map[:, -overlap:] * ramp.flip(0).unsqueeze(0)

    return weight_map


def reconstruct_from_patches(
    patches: List[torch.Tensor],
    coordinates: List[Tuple[int, int, int, int]],
    output_shape: Tuple[int, int, int],
    overlap: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Reconstruct an image from overlapping patches with blending.

    Args:
        patches: List of patch tensors, each of shape (C, patch_size, patch_size)
        coordinates: List of (x_start, y_start, x_end, y_end) for each patch
        output_shape: (C, H, W) shape of the output image
        overlap: Overlap size used when creating patches
        device: Device for reconstruction

    Returns:
        reconstructed: Reconstructed image tensor of shape (C, H, W)
    """
    C, H, W = output_shape
    reconstructed = torch.zeros(C, H, W, device=device)
    weight_sum = torch.zeros(1, H, W, device=device)

    if len(patches) == 0:
        return reconstructed

    patch_size = patches[0].shape[1]
    blend_weights = create_blend_weight_map(patch_size, overlap, device)

    for patch, (x_start, y_start, x_end, y_end) in zip(patches, coordinates):
        # Get actual patch dimensions (may be smaller than patch_size for edge patches)
        actual_h = y_end - y_start
        actual_w = x_end - x_start

        # Crop patch and weights if needed
        patch_cropped = patch[:, :actual_h, :actual_w]
        weights_cropped = blend_weights[:actual_h, :actual_w]

        # Add weighted patch to reconstruction
        reconstructed[:, y_start:y_end, x_start:x_end] += patch_cropped * weights_cropped
        weight_sum[:, y_start:y_end, x_start:x_end] += weights_cropped

    # Normalize by weight sum to get final blended result
    weight_sum = torch.clamp(weight_sum, min=1e-6)
    reconstructed = reconstructed / weight_sum

    return reconstructed


def calculate_patch_error(
    patch_rendered: torch.Tensor,
    patch_gt: torch.Tensor,
    psnr_weight: float = 0.7,
    ssim_weight: float = 0.3,
    gamma: float = 2.2
) -> Dict[str, float]:
    """
    Calculate weighted error metric for a patch.

    Args:
        patch_rendered: Rendered patch tensor (C, H, W)
        patch_gt: Ground truth patch tensor (C, H, W)
        psnr_weight: Weight for PSNR in combined metric
        ssim_weight: Weight for SSIM in combined metric
        gamma: Gamma correction value

    Returns:
        Dictionary containing:
            - 'psnr': PSNR value
            - 'ssim': SSIM value
            - 'combined_error': Weighted combination
            - 'combined_score': Quality score (higher is better)
    """
    # Apply gamma correction
    rendered_corrected = torch.pow(torch.clamp(patch_rendered, 0.0, 1.0), 1.0 / gamma)
    gt_corrected = torch.pow(torch.clamp(patch_gt, 0.0, 1.0), 1.0 / gamma)

    # Calculate PSNR
    psnr = get_psnr(rendered_corrected, gt_corrected).item()

    # Calculate SSIM (needs batch dimension)
    ssim = fused_ssim(
        rendered_corrected.unsqueeze(0),
        gt_corrected.unsqueeze(0)
    ).item()

    # Combined score (higher is better)
    # Normalize PSNR to roughly [0, 1] range (assuming typical PSNR range 20-50)
    psnr_normalized = (psnr - 20.0) / 30.0
    psnr_normalized = max(0.0, min(1.0, psnr_normalized))

    combined_score = psnr_weight * psnr_normalized + ssim_weight * ssim

    # Error (lower is better) - inverse of score
    combined_error = 1.0 - combined_score

    return {
        'psnr': psnr,
        'ssim': ssim,
        'combined_score': combined_score,
        'combined_error': combined_error,
        'psnr_normalized': psnr_normalized
    }


def create_patch_grid_visualization(
    image: torch.Tensor,
    coordinates: List[Tuple[int, int, int, int]],
    errors: List[float] = None,
    error_threshold: float = None
) -> np.ndarray:
    """
    Create a visualization showing patch boundaries and optionally error values.

    Args:
        image: Image tensor (C, H, W)
        coordinates: List of (x_start, y_start, x_end, y_end) for each patch
        errors: Optional list of error values for each patch
        error_threshold: Optional threshold to highlight problematic patches

    Returns:
        Visualization as numpy array (H, W, 3) in RGB format, values in [0, 255]
    """
    # Convert to numpy and ensure RGB format
    if image.dim() == 4:
        image = image.squeeze(0)

    img_np = image.cpu().detach().numpy()

    # Handle grayscale
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)

    # Convert to HWC format and scale to [0, 255]
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # Draw patch boundaries
    overlay = img_np.copy()

    for idx, (x_start, y_start, x_end, y_end) in enumerate(coordinates):
        # Determine color based on error if provided
        if errors is not None and error_threshold is not None:
            if errors[idx] > error_threshold:
                color = (255, 0, 0)  # Red for high error
            else:
                color = (0, 255, 0)  # Green for acceptable
        else:
            color = (255, 255, 0)  # Yellow for neutral

        # Draw rectangle
        overlay[y_start:y_start+2, x_start:x_end] = color  # Top
        overlay[y_end-2:y_end, x_start:x_end] = color      # Bottom
        overlay[y_start:y_end, x_start:x_start+2] = color  # Left
        overlay[y_start:y_end, x_end-2:x_end] = color      # Right

    # Blend overlay with original
    result = (0.7 * img_np + 0.3 * overlay).astype(np.uint8)

    return result


def get_patch_bounds_for_gaussians(
    coordinates: Tuple[int, int, int, int],
    image_width: int,
    image_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert pixel coordinates to normalized bounds [0, 1] for gaussian initialization.

    Args:
        coordinates: (x_start, y_start, x_end, y_end) in pixels
        image_width: Full image width
        image_height: Full image height

    Returns:
        (x_min, y_min, x_max, y_max) in normalized coordinates [0, 1]
    """
    x_start, y_start, x_end, y_end = coordinates

    x_min = x_start / image_width
    y_min = y_start / image_height
    x_max = x_end / image_width
    y_max = y_end / image_height

    return (x_min, y_min, x_max, y_max)


def extract_patch_gaussians(
    xy: torch.Tensor,
    bounds: Tuple[float, float, float, float],
    margin: float = 0.1
) -> torch.Tensor:
    """
    Extract indices of gaussians that fall within patch bounds (with margin).

    Args:
        xy: Gaussian positions tensor (N, 2) in normalized coordinates [0, 1]
        bounds: (x_min, y_min, x_max, y_max) in normalized coordinates
        margin: Additional margin to include gaussians near patch boundaries

    Returns:
        Boolean mask tensor of shape (N,) indicating which gaussians are in patch
    """
    x_min, y_min, x_max, y_max = bounds

    # Add margin
    x_min = max(0.0, x_min - margin)
    y_min = max(0.0, y_min - margin)
    x_max = min(1.0, x_max + margin)
    y_max = min(1.0, y_max + margin)

    # Check which gaussians fall within bounds
    mask = (
        (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) &
        (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
    )

    return mask
