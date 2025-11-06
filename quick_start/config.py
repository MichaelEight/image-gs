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
        adaptive_config: Optional adaptive refinement configuration. None for standard training.
        make_training_video: Generate video showing training progression (default: False)
        video_iterations: Capture frame every N iterations for video (default: 50)
        eval_steps: Evaluate metrics every N iterations (default: 100)
    """
    input_filenames: List[str]
    gaussians: List[int]
    steps: List[int]
    use_progressive: bool = True
    init_gaussian_file: Optional[str] = None
    allow_partial: bool = False
    adaptive_config: Optional[AdaptiveRefinementConfig] = None
    make_training_video: bool = False
    video_iterations: int = 50
    eval_steps: int = 100

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

        if self.video_iterations <= 0:
            raise ValueError(f"video_iterations must be positive: {self.video_iterations}")

        if self.eval_steps <= 0:
            raise ValueError(f"eval_steps must be positive: {self.eval_steps}")


def set_config(
    input_filenames: Union[str, List[str]],
    gaussians: List[int],
    steps: List[int],
    use_progressive: bool = True,
    init_gaussian_file: Optional[str] = None,
    allow_partial: bool = False,
    # Video generation parameters
    make_training_video: bool = False,
    video_iterations: int = 50,
    eval_steps: int = 100,
    # Adaptive refinement parameters
    adaptive_refinement: bool = False,
    adaptive_patch_size: int = 256,
    adaptive_overlap: int = 32,
    adaptive_psnr_weight: float = 0.7,
    adaptive_ssim_weight: float = 0.3,
    adaptive_error_threshold: float = 0.3,
    adaptive_base_gaussians: Optional[int] = None,
    adaptive_refinement_increment: int = 2000,
    adaptive_max_iterations: int = 3,
    adaptive_max_gaussians_per_patch: int = 20000
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

        # Video Generation Parameters (opt-in)
        make_training_video: Generate training progress video (default: False)
        video_iterations: Capture frame every N iterations (default: 50)
        eval_steps: Evaluate metrics every N iterations (default: 100)
                   Note: Set eval_steps = video_iterations for smoother videos

        # Adaptive Refinement Parameters (opt-in)
        adaptive_refinement: Enable adaptive patch-based refinement (default: False)
        adaptive_patch_size: Size of each square patch in pixels (default: 256)
        adaptive_overlap: Overlap between patches in pixels (default: 32)
        adaptive_psnr_weight: Weight for PSNR in error metric 0-1 (default: 0.7)
        adaptive_ssim_weight: Weight for SSIM in error metric 0-1 (default: 0.3)
        adaptive_error_threshold: Error threshold for refinement 0-1 (default: 0.3)
        adaptive_base_gaussians: Gaussians for base training (default: uses gaussians[0])
        adaptive_refinement_increment: Gaussians added per iteration (default: 2000)
        adaptive_max_iterations: Max refinement iterations per patch (default: 3)
        adaptive_max_gaussians_per_patch: Safety limit on gaussians (default: 20000)

    Returns:
        TrainingConfig object

    Raises:
        ValueError: If configuration is invalid

    Examples:
        >>> # Standard training
        >>> config = set_config(
        ...     input_filenames="cat.png",
        ...     gaussians=[5000],
        ...     steps=[3500]
        ... )

        >>> # Adaptive refinement training
        >>> config = set_config(
        ...     input_filenames="cat.png",
        ...     gaussians=[10000],
        ...     steps=[5000],
        ...     adaptive_refinement=True,
        ...     adaptive_error_threshold=0.3
        ... )

        >>> # Multiple images with adaptive refinement
        >>> config = set_config(
        ...     input_filenames=["cat.png", "dog.png"],
        ...     gaussians=[10000],
        ...     steps=[5000],
        ...     adaptive_refinement=True
        ... )
    """
    # Convert single filename to list
    if isinstance(input_filenames, str):
        input_filenames = [input_filenames]

    # Create adaptive config if enabled
    adaptive_config = None
    if adaptive_refinement:
        # Use first gaussian count as base if not specified
        if adaptive_base_gaussians is None:
            adaptive_base_gaussians = gaussians[0]

        adaptive_config = AdaptiveRefinementConfig(
            enable=True,
            patch_size=adaptive_patch_size,
            overlap=adaptive_overlap,
            psnr_weight=adaptive_psnr_weight,
            ssim_weight=adaptive_ssim_weight,
            error_threshold=adaptive_error_threshold,
            base_gaussians=adaptive_base_gaussians,
            refinement_gaussian_increment=adaptive_refinement_increment,
            max_refinement_iterations=adaptive_max_iterations,
            max_gaussians_per_patch=adaptive_max_gaussians_per_patch
        )

    return TrainingConfig(
        input_filenames=input_filenames,
        gaussians=gaussians,
        steps=steps,
        use_progressive=use_progressive,
        init_gaussian_file=init_gaussian_file,
        allow_partial=allow_partial,
        adaptive_config=adaptive_config,
        make_training_video=make_training_video,
        video_iterations=video_iterations,
        eval_steps=eval_steps
    )
