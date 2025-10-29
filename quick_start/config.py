"""Configuration management for Image-GS training."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for Image-GS training.

    Attributes:
        input_filename: Name of input image file (e.g., "cat.png")
        gaussians: List of Gaussian counts to train (e.g., [5000, 10000])
        steps: List of training step counts (e.g., [3500, 5000])
        use_progressive: Enable progressive optimization (recommended: True)
        init_gaussian_file: Path to initial Gaussian checkpoint relative to workspace
                           (e.g., "input/pretrained.pt"). None for random init.
        allow_partial: Allow partial initialization if Gaussian counts don't match
    """
    input_filename: str
    gaussians: List[int]
    steps: List[int]
    use_progressive: bool = True
    init_gaussian_file: Optional[str] = None
    allow_partial: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.input_filename:
            raise ValueError("input_filename cannot be empty")

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
    input_filename: str,
    gaussians: List[int],
    steps: List[int],
    use_progressive: bool = True,
    init_gaussian_file: Optional[str] = None,
    allow_partial: bool = False
) -> TrainingConfig:
    """
    Create and validate a training configuration.

    Args:
        input_filename: Name of input image file (e.g., "cat.png")
        gaussians: List of Gaussian counts to train (e.g., [5000, 10000])
        steps: List of training step counts (e.g., [3500, 5000])
        use_progressive: Enable progressive optimization (default: True)
        init_gaussian_file: Path to initial Gaussian checkpoint relative to workspace.
                           None for random initialization (default: None)
        allow_partial: Allow partial initialization if counts don't match (default: False)

    Returns:
        TrainingConfig object

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> config = set_config(
        ...     input_filename="cat.png",
        ...     gaussians=[5000, 10000],
        ...     steps=[3500],
        ...     use_progressive=True
        ... )
    """
    return TrainingConfig(
        input_filename=input_filename,
        gaussians=gaussians,
        steps=steps,
        use_progressive=use_progressive,
        init_gaussian_file=init_gaussian_file,
        allow_partial=allow_partial
    )
