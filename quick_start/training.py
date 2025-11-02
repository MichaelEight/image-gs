"""Training workflow functions for Image-GS."""

import os
import sys
import shutil
import glob
import itertools
from typing import List, Optional, Tuple

from .config import TrainingConfig, AdaptiveRefinementConfig
from .utils import get_paths


def _create_session_folder() -> Tuple[str, str]:
    """
    Create a new session folder for training outputs.

    Finds the next available session number (session_1, session_2, etc.)

    Returns:
        Tuple of (session_name, session_path)
    """
    _, _, _, OUTPUT_DIR = get_paths()

    # Find next available session number
    session_num = 1
    while True:
        session_name = f"session_{session_num}"
        session_path = os.path.join(OUTPUT_DIR, session_name)
        if not os.path.exists(session_path):
            os.makedirs(session_path, exist_ok=True)
            return session_name, session_path
        session_num += 1


def _setup_training_environment(
    session_path: str,
    input_filename: str,
    num_gaussians: int,
    max_steps: int,
    init_gaussian_file: Optional[str] = None,
    allow_partial: bool = False
) -> Tuple[str, str, Optional[str]]:
    """
    Setup training environment: validate input, create directories, copy files.

    Args:
        session_path: Path to session folder
        input_filename: Input image filename
        num_gaussians: Number of Gaussians
        max_steps: Maximum training steps
        init_gaussian_file: Path to initial Gaussian file relative to repo root
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

    # Create output folder (no timestamp)
    base_name = os.path.splitext(input_filename)[0]
    output_folder = f"{base_name}-{num_gaussians}-{max_steps}"
    output_path = os.path.join(session_path, output_folder)

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
    print(f"Output:      {output_folder}/")
    print(f"Time est:    ~{max_steps * 0.002:.1f}-{max_steps * 0.005:.1f} minutes")
    print("=" * 80)
    print()

    os.system(cmd)

    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"📁 Output folder: {output_folder}/")
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

    if not os.path.exists(result_base):
        raise FileNotFoundError(f"Results directory not found: {result_base}")

    run_dirs = [d for d in os.listdir(result_base) if os.path.isdir(os.path.join(result_base, d))]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {result_base}")

    latest_run = sorted(run_dirs)[-1]
    result_dir = os.path.join(result_base, latest_run)

    # 1. Model checkpoint
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
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
    session_path: str,
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
        session_path: Path to session folder
        input_filename: Input image filename (from input/ directory)
        num_gaussians: Number of Gaussians
        max_steps: Training steps
        use_progressive: Enable progressive optimization
        init_gaussian_file: Path to initial Gaussian file relative to repo root
        allow_partial: Allow partial initialization

    Returns:
        Output folder name (e.g., "cat-5000-3500")
    """
    output_folder, output_path, init_checkpoint_path = _setup_training_environment(
        session_path, input_filename, num_gaussians, max_steps, init_gaussian_file, allow_partial
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

    This function trains all combinations of images × Gaussians × Steps
    specified in the config. All results are saved in a single session folder.

    Supports both standard training and adaptive refinement training based on
    the config.adaptive_config setting.

    Args:
        config: Training configuration object

    Returns:
        List of output folder names (relative to session folder)

    Example:
        >>> # Standard training
        >>> config = set_config(
        ...     input_filenames=["cat.png", "dog.png"],
        ...     gaussians=[1000, 5000],
        ...     steps=[2000, 3500]
        ... )
        >>> results = train(config)  # Trains 8 models (2×2×2)

        >>> # Adaptive refinement training
        >>> config = set_config(
        ...     input_filenames="cat.png",
        ...     gaussians=[10000],
        ...     steps=[5000],
        ...     adaptive_refinement=True
        ... )
        >>> results = train(config)  # Uses adaptive patching
    """
    # Check if adaptive refinement is enabled
    if config.adaptive_config is not None and config.adaptive_config.enable:
        return _train_adaptive_batch(config)
    else:
        return _train_standard_batch(config)


def _train_standard_batch(config: TrainingConfig) -> List[str]:
    """
    Train standard Image-GS models (original behavior).

    Args:
        config: Training configuration object

    Returns:
        List of output folder names (relative to session folder)
    """
    # Create session folder
    session_name, session_path = _create_session_folder()

    # Generate all combinations: images × gaussians × steps
    combinations = list(itertools.product(config.input_filenames, config.gaussians, config.steps))
    total_combinations = len(combinations)

    print("=" * 80)
    print(f"BATCH TRAINING SESSION: {session_name}")
    print("=" * 80)
    print(f"Images:      {len(config.input_filenames)} ({', '.join(config.input_filenames)})")
    print(f"Gaussians:   {config.gaussians}")
    print(f"Steps:       {config.steps}")
    print(f"Total runs:  {total_combinations} ({len(config.input_filenames)}×{len(config.gaussians)}×{len(config.steps)})")
    print(f"Session:     {session_name}")
    print("=" * 80)
    print()

    # Store output folders
    output_folders = []

    # Train each combination
    for idx, (input_filename, num_gaussians, max_steps) in enumerate(combinations, 1):
        print(f"{'═' * 80}")
        print(f"TRAINING {idx}/{total_combinations}")
        print(f"{'═' * 80}")
        print(f"Image: {input_filename}, Gaussians: {num_gaussians}, Steps: {max_steps}")
        print()

        # Train
        output_folder = train_single(
            session_path=session_path,
            input_filename=input_filename,
            num_gaussians=num_gaussians,
            max_steps=max_steps,
            use_progressive=config.use_progressive,
            init_gaussian_file=config.init_gaussian_file,
            allow_partial=config.allow_partial
        )

        # Store as session_name/folder_name for easy reference
        full_output_ref = f"{session_name}/{output_folder}"
        output_folders.append(full_output_ref)

        print(f"📁 Output: {full_output_ref}")
        print()

    print("=" * 80)
    print("✅ ALL TRAINING COMPLETE")
    print("=" * 80)
    print(f"Session:    {session_name}")
    print(f"Total runs: {total_combinations}")
    print()
    print("Output folders:")
    for i, folder in enumerate(output_folders, 1):
        print(f"  {i}. {folder}")
    print("=" * 80)

    return output_folders


def _train_adaptive_batch(config: TrainingConfig) -> List[str]:
    """
    Train with adaptive refinement for all image/step combinations.

    Args:
        config: Training configuration object with adaptive_config

    Returns:
        List of output folder names (relative to session folder)
    """
    from .adaptive_training import train_adaptive
    import argparse

    # Create session folder
    session_name, session_path = _create_session_folder()

    ROOT_WORKSPACE, REPO_DIR, INPUT_DIR, OUTPUT_DIR = get_paths()

    # For adaptive training, we only use the first gaussian count (base gaussians)
    # and generate combinations of images × steps
    combinations = list(itertools.product(config.input_filenames, config.steps))
    total_combinations = len(combinations)

    print("=" * 80)
    print(f"ADAPTIVE REFINEMENT SESSION: {session_name}")
    print("=" * 80)
    print(f"Mode:        Adaptive Patch-Based Refinement")
    print(f"Images:      {len(config.input_filenames)} ({', '.join(config.input_filenames)})")
    print(f"Base Gaussians: {config.adaptive_config.base_gaussians}")
    print(f"Steps:       {config.steps}")
    print(f"Patch Size:  {config.adaptive_config.patch_size}x{config.adaptive_config.patch_size}")
    print(f"Error Threshold: {config.adaptive_config.error_threshold}")
    print(f"Total runs:  {total_combinations} ({len(config.input_filenames)}×{len(config.steps)})")
    print(f"Session:     {session_name}")
    print("=" * 80)
    print()

    # Store output folders
    output_folders = []

    # Train each combination
    for idx, (input_filename, max_steps) in enumerate(combinations, 1):
        print(f"{'═' * 80}")
        print(f"ADAPTIVE TRAINING {idx}/{total_combinations}")
        print(f"{'═' * 80}")
        print(f"Image: {input_filename}, Steps: {max_steps}")
        print()

        # Change to repository directory
        os.chdir(REPO_DIR)

        # Validate and prepare input
        input_path = os.path.join(INPUT_DIR, input_filename)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        # Copy input to media/images/
        media_input_path = os.path.join(REPO_DIR, "media", "images", input_filename)
        os.makedirs(os.path.join(REPO_DIR, "media", "images"), exist_ok=True)
        shutil.copy2(input_path, media_input_path)

        # Create args object for adaptive training
        args = argparse.Namespace()
        args.input_path = f"images/{input_filename}"
        args.num_gaussians = config.adaptive_config.base_gaussians
        args.max_steps = max_steps
        args.device = "cuda:0"
        args.quantize = True
        args.disable_prog_optim = not config.use_progressive
        args.evaluate = False
        args.gamma = 1.0
        args.pos_bits = 16
        args.scale_bits = 16
        args.rot_bits = 16
        args.feat_bits = 16
        args.data_root = "media"

        # Run adaptive training
        result = train_adaptive(
            args,
            config.adaptive_config,
            workspace_dir=ROOT_WORKSPACE,
            session_name=session_name
        )

        # Extract folder name from output_dir
        # output_dir format: workspace/output/session_N/image-adaptive-gaussians
        output_dir_parts = result['output_dir'].split(os.sep)
        # Find the part after session_name
        try:
            session_idx = output_dir_parts.index(session_name)
            output_folder = output_dir_parts[session_idx + 1]
        except (ValueError, IndexError):
            # Fallback: use last part of path
            output_folder = os.path.basename(result['output_dir'])

        # Store as session_name/folder_name
        full_output_ref = f"{session_name}/{output_folder}"
        output_folders.append(full_output_ref)

        print(f"\n📁 Output: {full_output_ref}")
        print()

    print("=" * 80)
    print("✅ ALL ADAPTIVE TRAINING COMPLETE")
    print("=" * 80)
    print(f"Session:    {session_name}")
    print(f"Total runs: {total_combinations}")
    print()
    print("Output folders:")
    for i, folder in enumerate(output_folders, 1):
        print(f"  {i}. {folder}")
    print("=" * 80)

    return output_folders


def train_adaptive_wrapper(
    input_filename: str,
    adaptive_config: AdaptiveRefinementConfig,
    max_steps: int = 10000
) -> str:
    """
    Wrapper for adaptive refinement training with simplified interface.

    Args:
        input_filename: Input image filename
        adaptive_config: Adaptive refinement configuration
        max_steps: Maximum training steps for base and patch training

    Returns:
        Path to output directory

    Example:
        >>> from quick_start.config import AdaptiveRefinementConfig
        >>> config = AdaptiveRefinementConfig(
        ...     enable=True,
        ...     patch_size=256,
        ...     overlap=32,
        ...     error_threshold=0.3,
        ...     base_gaussians=10000
        ... )
        >>> output = train_adaptive_wrapper("cat.png", config)
    """
    from .adaptive_training import train_adaptive
    import argparse

    ROOT_WORKSPACE, REPO_DIR, INPUT_DIR, OUTPUT_DIR = get_paths()

    # Change to repository directory
    os.chdir(REPO_DIR)

    # Validate input
    input_path = os.path.join(INPUT_DIR, input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Copy input to media/images/
    media_input_path = os.path.join(REPO_DIR, "media", "images", input_filename)
    os.makedirs(os.path.join(REPO_DIR, "media", "images"), exist_ok=True)
    shutil.copy2(input_path, media_input_path)

    # Create session folder
    session_name, session_path = _create_session_folder()

    # Load default config
    sys.path.insert(0, REPO_DIR)
    from utils.misc_utils import load_cfg

    parser = argparse.ArgumentParser()
    parser = load_cfg(cfg_path="cfgs/default.yaml", parser=parser)
    args = parser.parse_args([])

    # Update args for adaptive training
    args.input_path = f"images/{input_filename}"
    args.max_steps = max_steps
    args.num_gaussians = adaptive_config.base_gaussians

    # Run adaptive training
    result = train_adaptive(
        args,
        adaptive_config,
        workspace_dir=ROOT_WORKSPACE,
        session_name=session_name
    )

    return result['output_dir']
