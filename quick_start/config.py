"""Configuration management for Image-GS training."""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class AdaptiveRefinementConfig:
    """
    Configuration for adaptive patch-based refinement.

    Attributes:
        enable: Enable adaptive refinement feature
        patch_size: Size of each square patch in pixels
        overlap: Overlap between adjacent patches in pixels
        psnr_weight: Weight for PSNR in combined error metric (0-1)
        ssim_weight: Weight for SSIM in combined error metric (0-1)
        error_threshold: Combined error threshold for refinement (0-1, lower is stricter)
        base_gaussians: Number of gaussians for base training
        refinement_gaussian_increment: Gaussians to add per refinement iteration
        max_refinement_iterations: Maximum refinement iterations per patch
        max_gaussians_per_patch: Safety limit on gaussians per patch
    """
    enable: bool = False
    patch_size: int = 256
    overlap: int = 32
    psnr_weight: float = 0.7
    ssim_weight: float = 0.3
    error_threshold: float = 0.3  # Combined error threshold
    base_gaussians: int = 10000
    refinement_gaussian_increment: int = 2000
    max_refinement_iterations: int = 3
    max_gaussians_per_patch: int = 20000

    def __post_init__(self):
        """Validate configuration."""
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive: {self.patch_size}")

        if self.overlap < 0 or self.overlap >= self.patch_size:
            raise ValueError(f"overlap must be in [0, patch_size): {self.overlap}")

        if not (0 <= self.psnr_weight <= 1):
            raise ValueError(f"psnr_weight must be in [0, 1]: {self.psnr_weight}")

        if not (0 <= self.ssim_weight <= 1):
            raise ValueError(f"ssim_weight must be in [0, 1]: {self.ssim_weight}")

        if abs(self.psnr_weight + self.ssim_weight - 1.0) > 0.01:
            raise ValueError(f"psnr_weight + ssim_weight must equal 1.0")

        if not (0 <= self.error_threshold <= 1):
            raise ValueError(f"error_threshold must be in [0, 1]: {self.error_threshold}")

        if self.base_gaussians <= 0:
            raise ValueError(f"base_gaussians must be positive: {self.base_gaussians}")

        if self.refinement_gaussian_increment <= 0:
            raise ValueError(f"refinement_gaussian_increment must be positive")

        if self.max_refinement_iterations <= 0:
            raise ValueError(f"max_refinement_iterations must be positive")

        if self.max_gaussians_per_patch < self.base_gaussians:
            raise ValueError(f"max_gaussians_per_patch must be >= base_gaussians")


@dataclass
class TrainingConfig:
    """
    Configuration for Image-GS training.

    Attributes:
        input_filenames: List of input image filenames (e.g., ["cat.png", "dog.png"])
        gaussians: List of Gaussian counts to train (e.g., [5000, 10000])
        steps: List of training step counts (e.g., [3500, 5000])
        use_progressive: Enable progressive optimization (recommended: True)
        init_gaussian_file: Path to initial Gaussian checkpoint relative to workspace
                           (e.g., "output/session_1/cat-5000-3500/model.pt"). None for random init.
        allow_partial: Allow partial initialization if Gaussian counts don't match
    """
    input_filenames: List[str]
    gaussians: List[int]
    steps: List[int]
    use_progressive: bool = True
    init_gaussian_file: Optional[str] = None
    allow_partial: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.input_filenames:
            raise ValueError("input_filenames list cannot be empty")

        for filename in self.input_filenames:
            if not filename:
                raise ValueError("input_filenames cannot contain empty strings")

        if not self.gaussians:
            raise ValueError("gaussians list cannot be empty")

        if not self.steps:
            raise ValueError("steps list cannot be empty")

        for g in self.gaussians:
            if g <= 0:
                raise ValueError(f"Invalid Gaussian count: {g} (must be positive)")

        for s in self.steps:
            if s <= 0:
                raise ValueError(f"Invalid step count: {s} (must be positive)")

        if self.init_gaussian_file is not None and not self.init_gaussian_file.endswith('.pt'):
            raise ValueError(f"init_gaussian_file must be a .pt file: {self.init_gaussian_file}")


def set_config(
    input_filenames: Union[str, List[str]],
    gaussians: List[int],
    steps: List[int],
    use_progressive: bool = True,
    init_gaussian_file: Optional[str] = None,
    allow_partial: bool = False
) -> TrainingConfig:
    """
    Create and validate a training configuration.

    Supports both single and multiple image training. All combinations of
    images × gaussians × steps will be trained sequentially.

    Args:
        input_filenames: Single filename (str) or list of filenames (List[str])
                        e.g., "cat.png" or ["cat.png", "dog.png"]
        gaussians: List of Gaussian counts to train (e.g., [5000, 10000])
        steps: List of training step counts (e.g., [3500, 5000])
        use_progressive: Enable progressive optimization (default: True)
        init_gaussian_file: Path to initial Gaussian checkpoint relative to repo root.
                           None for random initialization (default: None)
        allow_partial: Allow partial initialization if counts don't match (default: False)

    Returns:
        TrainingConfig object

    Raises:
        ValueError: If configuration is invalid

    Examples:
        >>> # Single image
        >>> config = set_config(
        ...     input_filenames="cat.png",
        ...     gaussians=[5000],
        ...     steps=[3500]
        ... )

        >>> # Multiple images
        >>> config = set_config(
        ...     input_filenames=["cat.png", "dog.png"],
        ...     gaussians=[1000, 5000],
        ...     steps=[2000, 3500]
        ... )
        >>> # This trains 8 models: 2 images × 2 gaussians × 2 steps
    """
    # Convert single filename to list
    if isinstance(input_filenames, str):
        input_filenames = [input_filenames]

    return TrainingConfig(
        input_filenames=input_filenames,
        gaussians=gaussians,
        steps=steps,
        use_progressive=use_progressive,
        init_gaussian_file=init_gaussian_file,
        allow_partial=allow_partial
    )
