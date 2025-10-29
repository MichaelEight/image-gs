"""Installation and setup functions for Image-GS."""

import sys
import subprocess
import os
from typing import List


def install_system_deps() -> None:
    """Install system dependencies via apt-get."""
    print("Installing system dependencies...\n")

    commands = [
        "apt-get update -qq",
        "apt-get install -y -qq build-essential git wget curl",
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=True)

    print("✓ System dependencies installed")


def upgrade_pip() -> None:
    """Upgrade pip, setuptools, and wheel."""
    print("Upgrading pip...\n")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "-q"],
        capture_output=True
    )
    print("✓ pip upgraded")


def install_pytorch() -> None:
    """Install PyTorch with CUDA 12.1 support for RTX 4090."""
    print("Installing PyTorch 2.4.1 with CUDA 12.1...\n")

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])

    import torch
    print(f"\n✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")


def install_python_deps() -> None:
    """Install Python dependencies."""
    print("Installing Python dependencies...\n")

    dependencies = [
        "flip-evaluator",
        "lpips==0.1.4",
        "matplotlib==3.9.2",
        "numpy<2.1",
        "opencv-python==4.12.0.88",
        "pytorch-msssim==1.0.0",
        "scikit-image==0.24.0",
        "scipy==1.13.1",
        "torchmetrics==1.5.2",
        "jaxtyping",
        "rich>=12",
        "pyyaml==6.0",
        "ninja",
    ]

    for dep in dependencies:
        subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q"], capture_output=True)

    print("✓ Python dependencies installed")


def clone_repository() -> None:
    """Clone the Image-GS repository."""
    ROOT_WORKSPACE = os.getcwd()
    REPO_DIR = os.path.join(ROOT_WORKSPACE, "image-gs")

    if os.path.exists(REPO_DIR):
        print(f"Repository already exists at {REPO_DIR}")
        os.chdir(REPO_DIR)
        subprocess.run(["git", "pull"], capture_output=True)
        os.chdir(ROOT_WORKSPACE)
    else:
        print(f"Cloning repository to {REPO_DIR}...\n")
        subprocess.run(["git", "clone", "https://github.com/MichaelEight/image-gs", REPO_DIR])
        # subprocess.run(["git", "clone", "https://github.com/NYU-ICL/image-gs.git", REPO_DIR]) # ORIGINAL

    print(f"\n✓ Repository: {REPO_DIR}")


def install_fused_ssim() -> None:
    """Install fused-ssim extension."""
    print("Installing fused-ssim...\n")

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/rahul-goel/fused-ssim.git",
        "--no-build-isolation", "-q"
    ])

    print("✓ fused-ssim installed")


def install_gsplat() -> None:
    """Install gsplat CUDA extension."""
    print("Installing gsplat CUDA extension...\n")
    print("This will take 5-10 minutes.\n")

    ROOT_WORKSPACE = os.getcwd()
    REPO_DIR = os.path.join(ROOT_WORKSPACE, "image-gs")
    gsplat_dir = os.path.join(REPO_DIR, "gsplat")

    original_dir = os.getcwd()
    os.chdir(gsplat_dir)

    # Uninstall any existing installation
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "gsplat", "-q"])
    subprocess.run([sys.executable, "-m", "pip", "cache", "purge", "-q"])

    # Regular install (not editable)
    subprocess.run([sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"])

    os.chdir(original_dir)
    print("\n✓ gsplat installed")


def verify_installation() -> None:
    """Verify that all components are installed correctly."""
    print("Verifying installation...\n")

    # Force reimport
    for mod in list(sys.modules.keys()):
        if 'gsplat' in mod:
            del sys.modules[mod]

    errors = []

    try:
        import torch
        assert torch.cuda.is_available()
        print("✓ PyTorch with CUDA")
    except Exception as e:
        errors.append(f"PyTorch: {e}")

    try:
        from fused_ssim import fused_ssim
        print("✓ fused_ssim")
    except Exception as e:
        errors.append(f"fused_ssim: {e}")

    try:
        from gsplat import (
            project_gaussians_2d_scale_rot,
            rasterize_gaussians_no_tiles,
            rasterize_gaussians_sum,
        )
        print("✓ gsplat CUDA extensions")
    except Exception as e:
        errors.append(f"gsplat: {e}")

    try:
        ROOT_WORKSPACE = os.getcwd()
        REPO_DIR = os.path.join(ROOT_WORKSPACE, "image-gs")
        os.chdir(REPO_DIR)
        sys.path.insert(0, REPO_DIR)
        from model import GaussianSplatting2D
        from utils.misc_utils import load_cfg
        os.chdir(ROOT_WORKSPACE)
        print("✓ Image-GS modules")
    except Exception as e:
        errors.append(f"Image-GS: {e}")

    if errors:
        print(f"\n⚠️  {len(errors)} error(s):")
        for err in errors:
            print(f"  {err}")
    else:
        print("\n✅ All components verified!")


def setup() -> None:
    """
    Complete setup workflow for Image-GS.

    This function orchestrates all installation steps in the correct order:
    1. Install system dependencies
    2. Upgrade pip
    3. Install PyTorch
    4. Install Python dependencies
    5. Clone repository
    6. Install fused-ssim
    7. Install gsplat
    8. Verify installation
    """
    install_system_deps()
    upgrade_pip()
    install_pytorch()
    install_python_deps()
    clone_repository()
    install_fused_ssim()
    install_gsplat()
    verify_installation()

    print("\n" + "=" * 80)
    print("✅ SETUP COMPLETE")
    print("=" * 80)
