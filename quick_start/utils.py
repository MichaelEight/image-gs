"""Utility functions for path management and formatting."""

import os
from typing import Tuple, Optional


def get_paths() -> Tuple[str, str, str, str]:
    """
    Get workspace paths for the Image-GS project.

    Assumes we are already inside the image-gs repository directory.

    Returns:
        Tuple of (ROOT_WORKSPACE, REPO_DIR, INPUT_DIR, OUTPUT_DIR)
        - ROOT_WORKSPACE: Current directory (should be image-gs/)
        - REPO_DIR: Same as ROOT_WORKSPACE (we ARE the repo)
        - INPUT_DIR: image-gs/input/
        - OUTPUT_DIR: image-gs/output/
    """
    ROOT_WORKSPACE = os.getcwd()
    REPO_DIR = ROOT_WORKSPACE  # We are already in the repository
    INPUT_DIR = os.path.join(ROOT_WORKSPACE, "input")
    OUTPUT_DIR = os.path.join(ROOT_WORKSPACE, "output")

    # Create directories if they don't exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return ROOT_WORKSPACE, REPO_DIR, INPUT_DIR, OUTPUT_DIR


def format_size(size_bytes: Optional[int]) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes, or None

    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    if size_bytes is None:
        return "N/A"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def format_ratio(num: Optional[float], denom: Optional[float]) -> str:
    """
    Format a ratio as a multiplier (e.g., "2.50x").

    Args:
        num: Numerator
        denom: Denominator

    Returns:
        Formatted ratio string
    """
    if num is None or denom is None or denom == 0:
        return "N/A"
    return f"{num / denom:.2f}x"
