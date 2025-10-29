"""Training workflow functions for Image-GS."""

import os
import sys
import shutil
import glob
import itertools
from datetime import datetime
from typing import List, Optional, Tuple

from .config import TrainingConfig
from .utils import get_paths


def _setup_training_environment(
    input_filename: str,
    num_gaussians: int,
    max_steps: int,
    init_gaussian_file: Optional[str] = None,
    allow_partial: bool = False
) -> Tuple[str, str, Optional[str]]:
    """
    Setup training environment: validate input, create directories, copy files.

    Args:
        input_filename: Input image filename
        num_gaussians: Number of Gaussians
        max_steps: Maximum training steps
        init_gaussian_file: Path to initial Gaussian file relative to workspace
        allow_partial: Allow partial initialization

    Returns:
        Tuple of (output_folder, output_path, init_checkpoint_path)
    """
    ROOT_WORKSPACE, REPO_DIR, INPUT_DIR, OUTPUT_DIR = get_paths()

    # Change to repository directory
    os.chdir(REPO_DIR)

    # Validate input
    input_path = os.path.join(INPUT_DIR, input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Validate initial Gaussian file if provided
    init_source_path = None
    if init_gaussian_file is not None:
        init_source_path = os.path.join(ROOT_WORKSPACE, init_gaussian_file)
        if not os.path.exists(init_source_path):
            raise FileNotFoundError(f"Initial Gaussian file not found: {init_source_path}")
        if not init_source_path.endswith('.pt'):
            raise ValueError(f"Initial Gaussian file must be a .pt file: {init_source_path}")

    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(input_filename)[0]
    output_folder = f"{base_name}-{num_gaussians}-{max_steps}-{timestamp}"
    output_path = os.path.join(OUTPUT_DIR, output_folder)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "other"), exist_ok=True)

    # Copy input to media/images/
    media_input_path = os.path.join(REPO_DIR, "media", "images", input_filename)
    os.makedirs(os.path.join(REPO_DIR, "media", "images"), exist_ok=True)
    shutil.copy2(input_path, media_input_path)

    # Copy initial Gaussian file if provided
    init_checkpoint_path = None
    if init_source_path is not None:
        init_checkpoint_path = os.path.join(output_path, "initial_model.pt")
        shutil.copy2(init_source_path, init_checkpoint_path)
        print(f"✓ Copied initial Gaussian file: {init_gaussian_file}")
        print(f"  → {init_checkpoint_path}")

    return output_folder, output_path, init_checkpoint_path


def _build_training_command(
    input_filename: str,
    output_folder: str,
    num_gaussians: int,
    max_steps: int,
    use_progressive: bool,
    init_checkpoint_path: Optional[str] = None,
    allow_partial: bool = False
) -> str:
    """
    Build the training command string.

    Args:
        input_filename: Input image filename
        output_folder: Output folder name
        num_gaussians: Number of Gaussians
        max_steps: Maximum training steps
        use_progressive: Enable progressive optimization
        init_checkpoint_path: Path to initial checkpoint
        allow_partial: Allow partial initialization

    Returns:
        Command string to execute
    """
    prog_flag = "" if use_progressive else "--disable_prog_optim"
    temp_exp_name = f"temp/{output_folder}"

    # Add initialization flags if checkpoint provided
    init_flags = ""
    if init_checkpoint_path is not None:
        init_flags = f"--init_from_checkpoint --init_checkpoint_path=\"{init_checkpoint_path}\""
        if allow_partial:
            init_flags += " --init_partial"

    cmd = f"""
    {sys.executable} main.py \
      --input_path="images/{input_filename}" \
      --exp_name="{temp_exp_name}" \
      --num_gaussians={num_gaussians} \
      --max_steps={max_steps} \
      --quantize \
      {prog_flag} \
      {init_flags} \
      --device="cuda:0"
    """

    return cmd


def _run_training(
    cmd: str,
    input_filename: str,
    num_gaussians: int,
    max_steps: int,
    use_progressive: bool,
    output_folder: str,
    init_checkpoint_path: Optional[str] = None,
    allow_partial: bool = False
) -> None:
    """
    Print training info and execute training command.

    Args:
        cmd: Command to execute
        input_filename: Input image filename
        num_gaussians: Number of Gaussians
        max_steps: Maximum training steps
        use_progressive: Progressive optimization enabled
        output_folder: Output folder name
        init_checkpoint_path: Path to initial checkpoint
        allow_partial: Allow partial initialization
    """
    ROOT_WORKSPACE, _, _, _ = get_paths()

    print("=" * 80)
    print(f"🚀 TRAINING: {input_filename}")
    print("=" * 80)
    print(f"Gaussians:   {num_gaussians}")
    print(f"Steps:       {max_steps}")
    print(f"Progressive: {use_progressive}")
    if init_checkpoint_path is not None:
        print(f"Init from:   Custom checkpoint")
        print(f"             {init_checkpoint_path}")
        print(f"Partial:     {allow_partial}")
    else:
        print(f"Init from:   Default (random)")
    print(f"Output:      output/{output_folder}/")
    print(f"Time est:    ~{max_steps * 0.002:.1f}-{max_steps * 0.005:.1f} minutes")
    print("=" * 80)
    print()

    os.system(cmd)

    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"📁 Output folder: output/{output_folder}/")
    print()


def _organize_training_outputs(output_folder: str, output_path: str) -> None:
    """
    Copy training outputs to organized output folder.

    Args:
        output_folder: Output folder name
        output_path: Full path to output folder
    """
    _, REPO_DIR, _, _ = get_paths()

    temp_exp_name = f"temp/{output_folder}"
    result_base = os.path.join(REPO_DIR, "results", temp_exp_name)
    run_dirs = [d for d in os.listdir(result_base) if os.path.isdir(os.path.join(result_base, d))]
    latest_run = sorted(run_dirs)[-1]
    result_dir = os.path.join(result_base, latest_run)

    # 1. Model checkpoint
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "ckpt_step-*.pt"))
    if ckpt_files:
        latest_ckpt = sorted(ckpt_files)[-1]
        shutil.copy2(latest_ckpt, os.path.join(output_path, "model.pt"))

    # 2. Rendered image
    renders = glob.glob(os.path.join(result_dir, "render_res-*.jpg"))
    if renders:
        shutil.copy2(renders[0], os.path.join(output_path, "rendered.jpg"))

    # 3. Ground truth image
    gts = glob.glob(os.path.join(result_dir, "gt_res-*.jpg"))
    if gts:
        shutil.copy2(gts[0], os.path.join(output_path, "other", "ground_truth.jpg"))

    # 4. Training log
    log_file = os.path.join(result_dir, "log_train.txt")
    if os.path.exists(log_file):
        shutil.copy2(log_file, os.path.join(output_path, "other", "log_train.txt"))

    # 5. Metrics CSV
    metrics_csv = os.path.join(result_dir, "metrics.csv")
    if os.path.exists(metrics_csv):
        shutil.copy2(metrics_csv, os.path.join(output_path, "metrics.csv"))

    # 6. Copy all other files to "other" subdirectory
    for item in os.listdir(result_dir):
        item_path = os.path.join(result_dir, item)
        if os.path.isfile(item_path):
            # Skip files we already copied
            if item not in ["log_train.txt", "metrics.csv"] and not item.startswith("render_res-") and not item.startswith("gt_res-"):
                shutil.copy2(item_path, os.path.join(output_path, "other", item))


def train_single(
    input_filename: str,
    num_gaussians: int,
    max_steps: int,
    use_progressive: bool = True,
    init_gaussian_file: Optional[str] = None,
    allow_partial: bool = False
) -> str:
    """
    Train a single Image-GS model.

    Args:
        input_filename: Input image filename (from input/ directory)
        num_gaussians: Number of Gaussians
        max_steps: Training steps
        use_progressive: Enable progressive optimization
        init_gaussian_file: Path to initial Gaussian file relative to workspace
        allow_partial: Allow partial initialization

    Returns:
        Output folder name (e.g., "cat-5000-3500-20251027_143052")
    """
    output_folder, output_path, init_checkpoint_path = _setup_training_environment(
        input_filename, num_gaussians, max_steps, init_gaussian_file, allow_partial
    )
    cmd = _build_training_command(
        input_filename, output_folder, num_gaussians, max_steps, use_progressive,
        init_checkpoint_path, allow_partial
    )
    _run_training(
        cmd, input_filename, num_gaussians, max_steps, use_progressive, output_folder,
        init_checkpoint_path, allow_partial
    )
    _organize_training_outputs(output_folder, output_path)

    return output_folder


def train(config: TrainingConfig) -> List[str]:
    """
    Train Image-GS models with batch support.

    This function trains all combinations of Gaussians × Steps specified in the config.

    Args:
        config: Training configuration object

    Returns:
        List of output folder names

    Example:
        >>> config = set_config(
        ...     input_filename="cat.png",
        ...     gaussians=[5000, 10000],
        ...     steps=[3500, 5000]
        ... )
        >>> results = train(config)  # Trains 4 models (2 × 2)
    """
    # Generate all combinations
    combinations = list(itertools.product(config.gaussians, config.steps))
    total_combinations = len(combinations)

    print("=" * 80)
    print(f"BATCH TRAINING: {total_combinations} combination(s)")
    print("=" * 80)
    print()

    # Store output folders
    output_folders = []

    # Train each combination
    for idx, (num_gaussians, max_steps) in enumerate(combinations, 1):
        print(f"{'═' * 80}")
        print(f"TRAINING {idx}/{total_combinations}")
        print(f"{'═' * 80}")
        print(f"Parameters: Gaussians={num_gaussians}, Steps={max_steps}")
        print()

        # Train
        start_time = datetime.now()
        output_folder = train_single(
            input_filename=config.input_filename,
            num_gaussians=num_gaussians,
            max_steps=max_steps,
            use_progressive=config.use_progressive,
            init_gaussian_file=config.init_gaussian_file,
            allow_partial=config.allow_partial
        )
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds() / 60

        output_folders.append(output_folder)

        print(f"⏱️  Elapsed time: {elapsed:.2f} minutes")
        print(f"📁 Output: {output_folder}")
        print()

    print("=" * 80)
    print("✅ ALL TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total runs:     {total_combinations}")
    print(f"Output folders: {len(output_folders)}")
    print()
    print("Output folders list:")
    for i, folder in enumerate(output_folders, 1):
        print(f"  {i}. {folder}")
    print("=" * 80)

    return output_folders
