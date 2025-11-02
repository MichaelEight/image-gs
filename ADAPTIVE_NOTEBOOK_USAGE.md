# Using Adaptive Refinement in Jupyter Notebooks

This guide shows how to use the adaptive patch-based refinement feature in your Jupyter notebooks.

## Quick Start

### Standard Training (Unchanged)

```python
from quick_start import set_config, train, view_results

# Configure standard training
config = set_config(
    input_filenames="cat.png",
    gaussians=[1000],
    steps=[2000]
)

# Train
results = train(config)

# View results
view_results(results[0])
```

### Adaptive Refinement Training (New!)

```python
from quick_start import set_config, train, view_results

# Configure adaptive refinement training
config = set_config(
    input_filenames="cat.png",
    gaussians=[10000],           # Base gaussians for initial training
    steps=[5000],                # Training steps
    adaptive_refinement=True,    # Enable adaptive patching
    adaptive_error_threshold=0.3 # Quality threshold (lower = stricter)
)

# Train with adaptive refinement
results = train(config)

# View results (includes detailed adaptive patching info)
view_results(results[0])
```

## Configuration Options

### Basic Adaptive Parameters

```python
config = set_config(
    input_filenames="image.png",
    gaussians=[10000],
    steps=[5000],

    # Adaptive refinement
    adaptive_refinement=True,           # Enable/disable (default: False)
    adaptive_error_threshold=0.3        # Error threshold (default: 0.3)
)
```

### Advanced Adaptive Parameters

```python
config = set_config(
    input_filenames="image.png",
    gaussians=[10000],
    steps=[5000],

    # Adaptive refinement settings
    adaptive_refinement=True,
    adaptive_patch_size=256,            # Patch size in pixels (default: 256)
    adaptive_overlap=32,                # Overlap between patches (default: 32)
    adaptive_error_threshold=0.3,       # Error threshold (default: 0.3)
    adaptive_base_gaussians=10000,      # Base gaussians (default: uses gaussians[0])
    adaptive_refinement_increment=2000, # Gaussians added per iteration (default: 2000)
    adaptive_max_iterations=3,          # Max refinement iterations (default: 3)
    adaptive_psnr_weight=0.7,           # PSNR weight in error metric (default: 0.7)
    adaptive_ssim_weight=0.3            # SSIM weight in error metric (default: 0.3)
)
```

## Output Files

### Standard Training Outputs

```
output/session_N/image-gaussians-steps/
├── model.pt           # Trained model
├── rendered.jpg       # Rendered output
├── summary.txt        # Training summary
├── summary.png        # Visual comparison
├── metrics.csv        # Training metrics
└── other/             # Additional files
```

### Adaptive Refinement Outputs

```
output/session_N/image-adaptive-gaussians/
├── base_training/
│   ├── model.pt              # Base model
│   └── rendered.jpg          # Base render
├── enhanced/
│   ├── rendered.jpg          # Enhanced final image
│   └── patch_summary.json    # Patch statistics
├── summary.txt               # Comprehensive summary with per-patch breakdown
├── summary.png               # 6-panel visualization
├── tile_scores_heatmap.png   # Patch error scores matrix
├── comparison.jpg            # 3-way comparison
├── comparison_detailed.png   # Enhanced comparison with error maps
└── other/
    ├── patches/              # Individual patch results
    │   ├── patch_0_0/
    │   │   ├── before.jpg
    │   │   ├── after.jpg
    │   │   ├── error_before.jpg
    │   │   ├── error_after.jpg
    │   │   └── metrics.json
    │   └── ...
    └── patch_grid_visualization.jpg
```

## Enhanced Output Features

When using adaptive refinement, you get:

### 1. Real-time Console Output

```
🔄 ADAPTIVE PATCHING ACTIVATED
   Refining 3 patches that exceed error threshold

============================================================
[1/3] Processing patch_0_1
  Location: Row 0, Col 1
  Initial Error: 0.3542 (threshold: 0.3)
  Initial PSNR: 28.3 dB, SSIM: 0.8421
============================================================

  📊 Iteration 1/3
     Training with 10000 gaussians...
     Results:
       Error: 0.3124 (Δ -0.0418)
       PSNR:  29.1 dB (Δ +0.8)
       SSIM:  0.8567 (Δ +0.0146)
       Time:  45.3s

     ✅ SUCCESS: Threshold met!

  📈 Patch Summary:
     Iterations completed: 1
     Gaussians added: 2000 (final: 12000)
     Total time: 45.3s
     Quality improvement:
       PSNR:  28.3 → 29.1 dB (+0.8)
       SSIM:  0.8421 → 0.8567 (+0.0146)
       Error: 0.3542 → 0.3124 (-0.0418)
     Threshold status: ✅ Met
```

### 2. Enhanced Summary Text

The `summary.txt` file includes:

- **Adaptive Patching Status**: Whether adaptive patching was used
- **Per-Patch Breakdown Table**: Details for each refined patch
  - Before/after quality metrics (PSNR, SSIM, Error)
  - Improvements achieved
  - Gaussians added
  - Training time
  - Threshold status (✓ Met / ✗ Not met)
- **Summary Statistics**:
  - Total patches refined
  - Success rate
  - Average improvements
  - Total training time

### 3. Tile Score Heatmap

A visual matrix showing error scores for all patches:

- **Color coding**: Green (low error) to Red (high error)
- **Annotations**: Error scores and patch IDs in each cell
- **Threshold line**: Marked on the colorbar
- **Red borders**: Highlight patches that were refined
- **Legend**: Explains the visualization

## Examples

### Example 1: Single Image with Adaptive Refinement

```python
from quick_start import set_config, train, view_results

config = set_config(
    input_filenames="cat.png",
    gaussians=[10000],
    steps=[5000],
    adaptive_refinement=True,
    adaptive_error_threshold=0.3
)

results = train(config)
view_results(results[0])
```

### Example 2: Multiple Images with Adaptive Refinement

```python
from quick_start import set_config, train, view_results

config = set_config(
    input_filenames=["cat.png", "dog.png"],
    gaussians=[10000],
    steps=[5000],
    adaptive_refinement=True
)

results = train(config)

# View each result
for folder in results:
    view_results(folder)
```

### Example 3: Fine-tuning Adaptive Parameters

```python
from quick_start import set_config, train, view_results

config = set_config(
    input_filenames="complex_image.png",
    gaussians=[15000],
    steps=[7000],
    adaptive_refinement=True,
    adaptive_patch_size=128,        # Smaller patches for finer detail
    adaptive_error_threshold=0.25,  # Stricter quality requirement
    adaptive_max_iterations=5       # More refinement attempts
)

results = train(config)
view_results(results[0])
```

## When to Use Adaptive Refinement

Use **adaptive refinement** when:

- ✅ You want the highest quality output
- ✅ Your image has complex regions with fine details
- ✅ Standard training produces noticeable artifacts in some areas
- ✅ You're willing to spend more training time for better quality

Use **standard training** when:

- ✅ You need fast results
- ✅ Your image is relatively simple or uniform
- ✅ The quality from standard training is sufficient
- ✅ You're doing initial experiments or tests

## Tips

1. **Start with default parameters**: The defaults work well for most images
2. **Adjust threshold for quality**: Lower threshold = stricter quality = more patches refined
3. **Monitor console output**: Real-time feedback shows which patches are being refined
4. **Check the heatmap**: `tile_scores_heatmap.png` shows which areas had quality issues
5. **Review the summary**: `summary.txt` has detailed per-patch statistics

## Troubleshooting

### No patches are being refined

- **Possible cause**: Error threshold too high
- **Solution**: Lower `adaptive_error_threshold` (try 0.2 or 0.25)

### Too many patches being refined (very slow)

- **Possible cause**: Error threshold too low or patches too small
- **Solution**: Increase `adaptive_error_threshold` or increase `adaptive_patch_size`

### Out of memory errors

- **Possible cause**: Too many gaussians per patch
- **Solution**: Reduce `adaptive_max_gaussians_per_patch` or increase `adaptive_patch_size`

## Backward Compatibility

All existing notebook code continues to work without changes:

```python
# This still works exactly as before
config = set_config(
    input_filenames="cat.png",
    gaussians=[1000, 5000],
    steps=[2000, 3500]
)

results = train(config)
view_results(results[0])
```

Adaptive refinement is completely opt-in - you must explicitly set `adaptive_refinement=True`.
