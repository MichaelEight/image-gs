"""
Image-GS Quick Start API

This package provides a simplified API for training and analyzing 2D Gaussian Splatting models.

Quick Start:
    >>> from quick_start import setup, set_config, train, view_results
    >>>
    >>> # 1. Setup environment (run once)
    >>> setup()
    >>>
    >>> # 2. Configure training
    >>> config = set_config(
    ...     input_filename="cat.png",
    ...     gaussians=[5000],
    ...     steps=[3500]
    ... )
    >>>
    >>> # 3. Train models
    >>> results = train(config)
    >>>
    >>> # 4. View results
    >>> view_results(results[0])

Main Functions:
    - setup(): Install all dependencies and verify setup
    - verify_setup(): Verify installation without reinstalling
    - set_config(): Create training configuration
    - train(): Train models (supports batch training)
    - view_results(): Visualize and analyze training results
    - compare_batch_results(): Compare multiple training runs
    - plot_metrics(): Plot detailed training metrics
"""

from .setup import (
    setup,
    verify_installation as verify_setup,
    install_system_deps,
    upgrade_pip,
    install_pytorch,
    install_python_deps,
    clone_repository,
    install_fused_ssim,
    install_gsplat,
)

from .config import (
    set_config,
    TrainingConfig,
)

from .training import (
    train,
    train_single,
)

from .analysis import (
    view_results,
)

from .metrics import (
    load_metrics,
    plot_metrics,
)

from .batch import (
    compare_batch_results,
)

from .utils import (
    get_paths,
    format_size,
    format_ratio,
)

__all__ = [
    # Main user-facing API
    'setup',
    'verify_setup',
    'set_config',
    'train',
    'view_results',
    'compare_batch_results',
    'plot_metrics',
    # Configuration
    'TrainingConfig',
    # Advanced functions
    'train_single',
    'load_metrics',
    # Individual setup functions (for debugging)
    'install_system_deps',
    'upgrade_pip',
    'install_pytorch',
    'install_python_deps',
    'clone_repository',
    'install_fused_ssim',
    'install_gsplat',
    # Utilities
    'get_paths',
    'format_size',
    'format_ratio',
]

__version__ = '1.0.0'