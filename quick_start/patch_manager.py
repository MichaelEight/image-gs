"""
Patch manager for coordinating adaptive refinement.

This module manages the patch-based refinement workflow:
- Splitting images into patches
- Tracking patch errors and quality
- Coordinating per-patch training
- Managing refinement iterations
"""

import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from utils.patch_utils import (
    split_into_patches,
    reconstruct_from_patches,
    calculate_patch_error,
    create_patch_grid_visualization,
    get_patch_bounds_for_gaussians
)


@dataclass
class PatchInfo:
    """Information about a single patch."""
    patch_id: str  # e.g., "patch_0_0"
    row: int
    col: int
    coordinates: Tuple[int, int, int, int]  # (x_start, y_start, x_end, y_end)
    psnr: float = 0.0
    ssim: float = 0.0
    combined_score: float = 0.0
    combined_error: float = 1.0
    needs_refinement: bool = False
    refinement_iteration: int = 0
    total_gaussians: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'patch_id': self.patch_id,
            'row': self.row,
            'col': self.col,
            'coordinates': self.coordinates,
            'psnr': float(self.psnr),
            'ssim': float(self.ssim),
            'combined_score': float(self.combined_score),
            'combined_error': float(self.combined_error),
            'needs_refinement': bool(self.needs_refinement),
            'refinement_iteration': int(self.refinement_iteration),
            'total_gaussians': int(self.total_gaussians)
        }


class PatchManager:
    """
    Manages the adaptive refinement process for patches.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # (C, H, W)
        patch_size: int = 256,
        overlap: int = 32,
        error_threshold: float = 0.3,
        psnr_weight: float = 0.7,
        ssim_weight: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize patch manager.

        Args:
            image_shape: Shape of the full image (C, H, W)
            patch_size: Size of each square patch
            overlap: Overlap between patches
            error_threshold: Error threshold for refinement
            psnr_weight: Weight for PSNR in combined metric
            ssim_weight: Weight for SSIM in combined metric
            device: Device to use
        """
        self.image_shape = image_shape
        self.C, self.H, self.W = image_shape
        self.patch_size = patch_size
        self.overlap = overlap
        self.error_threshold = error_threshold
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.device = device

        # Patch tracking
        self.patches: List[PatchInfo] = []
        self.patch_dict: Dict[str, PatchInfo] = {}

        # Coordinate grid
        self.coordinates: List[Tuple[int, int, int, int]] = []
        self.n_rows = 0
        self.n_cols = 0

        # Initialize patches
        self._initialize_patches()

    def _initialize_patches(self):
        """Initialize patch grid and create PatchInfo objects."""
        # Create dummy image to get coordinates
        dummy_image = torch.zeros(self.image_shape, device=self.device)
        _, self.coordinates = split_into_patches(
            dummy_image, self.patch_size, self.overlap, self.device
        )

        # Calculate grid dimensions
        stride = self.patch_size - self.overlap
        self.n_rows = (self.H - self.overlap + stride - 1) // stride
        self.n_cols = (self.W - self.overlap + stride - 1) // stride

        # Create PatchInfo for each patch
        for idx, coords in enumerate(self.coordinates):
            row = idx // self.n_cols
            col = idx % self.n_cols
            patch_id = f"patch_{row}_{col}"

            patch_info = PatchInfo(
                patch_id=patch_id,
                row=row,
                col=col,
                coordinates=coords
            )

            self.patches.append(patch_info)
            self.patch_dict[patch_id] = patch_info

    def split_image(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Split image into patches.

        Args:
            image: Image tensor (C, H, W)

        Returns:
            List of patch tensors
        """
        patches, _ = split_into_patches(
            image, self.patch_size, self.overlap, self.device
        )
        return patches

    def reconstruct_image(self, patches: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct image from patches.

        Args:
            patches: List of patch tensors

        Returns:
            Reconstructed image tensor (C, H, W)
        """
        return reconstruct_from_patches(
            patches, self.coordinates, self.image_shape, self.overlap, self.device
        )

    def evaluate_patches(
        self,
        rendered_patches: List[torch.Tensor],
        gt_patches: List[torch.Tensor],
        gamma: float = 2.2
    ) -> Dict[str, float]:
        """
        Evaluate error for all patches.

        Args:
            rendered_patches: List of rendered patch tensors
            gt_patches: List of ground truth patch tensors
            gamma: Gamma correction value

        Returns:
            Dictionary with summary statistics
        """
        assert len(rendered_patches) == len(self.patches), \
            f"Patch count mismatch: {len(rendered_patches)} vs {len(self.patches)}"

        errors = []
        psnrs = []
        ssims = []
        num_needing_refinement = 0

        for patch_info, rendered, gt in zip(self.patches, rendered_patches, gt_patches):
            # Calculate error metrics
            metrics = calculate_patch_error(
                rendered, gt, self.psnr_weight, self.ssim_weight, gamma
            )

            # Update patch info
            patch_info.psnr = metrics['psnr']
            patch_info.ssim = metrics['ssim']
            patch_info.combined_score = metrics['combined_score']
            patch_info.combined_error = metrics['combined_error']
            patch_info.needs_refinement = metrics['combined_error'] > self.error_threshold

            errors.append(metrics['combined_error'])
            psnrs.append(metrics['psnr'])
            ssims.append(metrics['ssim'])

            if patch_info.needs_refinement:
                num_needing_refinement += 1

        return {
            'mean_error': float(sum(errors) / len(errors)),
            'max_error': float(max(errors)),
            'min_error': float(min(errors)),
            'mean_psnr': float(sum(psnrs) / len(psnrs)),
            'mean_ssim': float(sum(ssims) / len(ssims)),
            'num_patches': len(self.patches),
            'num_needing_refinement': num_needing_refinement,
            'refinement_ratio': num_needing_refinement / len(self.patches)
        }

    def get_patches_needing_refinement(self) -> List[PatchInfo]:
        """
        Get list of patches that need refinement.

        Returns:
            List of PatchInfo objects for patches exceeding error threshold
        """
        return [p for p in self.patches if p.needs_refinement]

    def get_patch_bounds(self, patch_info: PatchInfo) -> Tuple[float, float, float, float]:
        """
        Get normalized bounds for a patch.

        Args:
            patch_info: PatchInfo object

        Returns:
            (x_min, y_min, x_max, y_max) in normalized coordinates [0, 1]
        """
        return get_patch_bounds_for_gaussians(
            patch_info.coordinates, self.W, self.H
        )

    def update_patch_refinement(
        self,
        patch_id: str,
        iteration: int,
        gaussians: int,
        psnr: float,
        ssim: float,
        combined_error: float
    ):
        """
        Update patch refinement status.

        Args:
            patch_id: Patch identifier
            iteration: Current refinement iteration
            gaussians: Total gaussians used
            psnr: Updated PSNR
            ssim: Updated SSIM
            combined_error: Updated combined error
        """
        patch = self.patch_dict[patch_id]
        patch.refinement_iteration = iteration
        patch.total_gaussians = gaussians
        patch.psnr = psnr
        patch.ssim = ssim
        patch.combined_error = combined_error
        patch.needs_refinement = combined_error > self.error_threshold

    def create_visualization(
        self,
        image: torch.Tensor,
        show_errors: bool = True
    ) -> torch.Tensor:
        """
        Create patch grid visualization.

        Args:
            image: Image to visualize (C, H, W)
            show_errors: Whether to color patches by error

        Returns:
            Visualization tensor
        """
        import numpy as np

        errors = [p.combined_error for p in self.patches] if show_errors else None

        viz = create_patch_grid_visualization(
            image, self.coordinates, errors, self.error_threshold
        )

        return torch.from_numpy(viz).permute(2, 0, 1).float() / 255.0

    def save_summary(self, output_path: str):
        """
        Save patch summary to JSON file.

        Args:
            output_path: Path to save JSON file
        """
        summary = {
            'image_shape': self.image_shape,
            'patch_size': self.patch_size,
            'overlap': self.overlap,
            'error_threshold': self.error_threshold,
            'psnr_weight': self.psnr_weight,
            'ssim_weight': self.ssim_weight,
            'n_rows': self.n_rows,
            'n_cols': self.n_cols,
            'total_patches': len(self.patches),
            'patches': [p.to_dict() for p in self.patches]
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def load_summary(self, input_path: str):
        """
        Load patch summary from JSON file.

        Args:
            input_path: Path to JSON file
        """
        with open(input_path, 'r') as f:
            summary = json.load(f)

        # Update configuration
        self.image_shape = tuple(summary['image_shape'])
        self.C, self.H, self.W = self.image_shape
        self.patch_size = summary['patch_size']
        self.overlap = summary['overlap']
        self.error_threshold = summary['error_threshold']
        self.psnr_weight = summary['psnr_weight']
        self.ssim_weight = summary['ssim_weight']
        self.n_rows = summary['n_rows']
        self.n_cols = summary['n_cols']

        # Reconstruct patches
        self.patches = []
        self.patch_dict = {}

        for p_dict in summary['patches']:
            patch_info = PatchInfo(
                patch_id=p_dict['patch_id'],
                row=p_dict['row'],
                col=p_dict['col'],
                coordinates=tuple(p_dict['coordinates']),
                psnr=p_dict['psnr'],
                ssim=p_dict['ssim'],
                combined_score=p_dict['combined_score'],
                combined_error=p_dict['combined_error'],
                needs_refinement=p_dict['needs_refinement'],
                refinement_iteration=p_dict['refinement_iteration'],
                total_gaussians=p_dict['total_gaussians']
            )
            self.patches.append(patch_info)
            self.patch_dict[patch_info.patch_id] = patch_info

        # Reconstruct coordinates
        self.coordinates = [p.coordinates for p in self.patches]

    def get_statistics(self) -> Dict:
        """
        Get summary statistics about patches.

        Returns:
            Dictionary with statistics
        """
        errors = [p.combined_error for p in self.patches]
        psnrs = [p.psnr for p in self.patches]
        ssims = [p.ssim for p in self.patches]
        gaussians = [p.total_gaussians for p in self.patches]

        num_refined = sum(1 for p in self.patches if p.refinement_iteration > 0)
        num_needing = sum(1 for p in self.patches if p.needs_refinement)

        return {
            'total_patches': len(self.patches),
            'patches_refined': num_refined,
            'patches_still_needing_refinement': num_needing,
            'mean_error': float(sum(errors) / len(errors)) if errors else 0.0,
            'max_error': float(max(errors)) if errors else 0.0,
            'min_error': float(min(errors)) if errors else 0.0,
            'mean_psnr': float(sum(psnrs) / len(psnrs)) if psnrs else 0.0,
            'mean_ssim': float(sum(ssims) / len(ssims)) if ssims else 0.0,
            'total_gaussians': sum(gaussians),
            'mean_gaussians_per_patch': float(sum(gaussians) / len(gaussians)) if gaussians else 0.0
        }
