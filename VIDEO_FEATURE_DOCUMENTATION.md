# Training Video Generation Feature

## Overview

The training video generation feature allows users to create a side-by-side comparison video showing the progression of Gaussian Splatting training. The video displays the ground truth image on the left and the rendered image on the right, with training metrics (PSNR, SSIM, step number) overlaid on each frame.

## Features

- **Side-by-side comparison**: Ground truth vs. rendered image
- **Metrics overlay**: Step number, PSNR, and SSIM displayed on each frame
- **Configurable frame rate**: Capture frames every N iterations
- **High-quality output**: MP4 format with H.264 codec at 30 FPS
- **Memory efficient**: Frames stored as CPU tensors to avoid GPU memory issues

## Usage

### Quick Start Notebook

In `quick-start.ipynb`, set the following parameters in Step 3 (Configuration):

```python
from quick_start import set_config

config = set_config(
    input_filenames="cat.png",
    gaussians=[1000],
    steps=[2000],
    make_training_video=True,    # Enable video generation
    video_iterations=50,         # Capture frame every 50 iterations
    eval_steps=50                # Evaluate every 50 steps (match video_iterations!)
)
```

### Parameters

- **`make_training_video`** (bool, default: `False`):
  - Enable or disable training video generation
  - When `True`, the system captures frames during training and generates a video at the end

- **`video_iterations`** (int, default: `50`):
  - Number of training iterations between frame captures
  - Lower values = more frames = smoother video but larger file size
  - Higher values = fewer frames = choppier video but smaller file size
  - Recommended values: 25-100 depending on total training steps

- **`eval_steps`** (int, default: `100`):
  - Number of training iterations between metric evaluations (PSNR, SSIM)
  - **IMPORTANT**: Frames can only be captured during evaluation steps
  - **Best Practice**: Set `eval_steps = video_iterations` for optimal video smoothness
  - Lower values = more frequent evaluation = slightly slower training but smoother videos
  - Higher values = less frequent evaluation = faster training but choppier videos

### Example Configurations

#### High Detail (Many Frames)
```python
config = set_config(
    input_filenames="image.png",
    gaussians=[5000],
    steps=[10000],
    make_training_video=True,
    video_iterations=25,  # Capture every 25 steps = ~400 frames
    eval_steps=25         # Match video_iterations for smooth video
)
```

#### Balanced (Recommended)
```python
config = set_config(
    input_filenames="image.png",
    gaussians=[5000],
    steps=[10000],
    make_training_video=True,
    video_iterations=50,  # Capture every 50 steps = ~200 frames
    eval_steps=50         # Match video_iterations for smooth video
)
```

#### Compact (Fewer Frames)
```python
config = set_config(
    input_filenames="image.png",
    gaussians=[5000],
    steps=[10000],
    make_training_video=True,
    video_iterations=100,  # Capture every 100 steps = ~100 frames
    eval_steps=100         # Match video_iterations for smooth video
)
```

## Output

### Video Location

The generated video is saved in the same directory as the training output:

```
output/
└── session_1/
    └── cat-1000-2000/
        ├── model.pt
        ├── rendered.jpg
        ├── metrics.csv
        ├── training_video.mp4  ← Generated video here
        └── other/
            └── ...
```

### Video Specifications

- **Format**: MP4 (H.264 codec)
- **Frame Rate**: 30 FPS
- **Resolution**: 2× input image width (side-by-side layout)
- **Aspect Ratio**: Same as input image
- **Quality**: Standard (codec: 'mp4v')

### Video Content

Each frame contains:
1. **Left side**: Ground truth image (original)
2. **Right side**: Rendered image at current training step
3. **Labels**: "Ground Truth" and "Rendered" text overlays
4. **Metrics** (displayed on right side):
   - Step number
   - PSNR (dB)
   - SSIM (0-1)

## Technical Details

### Frame Capture

Frames are captured at the following points:
1. **Initial state** (step 0): Before any training
2. **Every N iterations**: Where N = `video_iterations`
3. **Final state**: After training completes

### Implementation Details

The feature integrates into the training loop at `model.py:580-588`:

```python
# Capture frame for video if enabled
if self.make_training_video and self.step % self.video_iterations == 0:
    images = self._render_images(upsample=False)
    images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0/self.gamma)
    self.video_frames.append({
        'render': images.clone(),
        'step': self.step,
        'psnr': self.psnr_curr,
        'ssim': self.ssim_curr
    })
```

### Video Generation

After training completes, the `_generate_training_video()` method (model.py:679-772):
1. Processes all captured frames
2. Applies gamma correction
3. Converts from RGB to BGR (OpenCV format)
4. Creates side-by-side layout
5. Adds text overlays with metrics
6. Writes frames to MP4 file

## Performance Considerations

### Memory Usage

- Frames are stored as **CPU tensors** to avoid GPU memory issues
- Memory usage = `(H × W × C × 4 bytes) × num_frames`
- Example: 512×512 RGB image, 200 frames = ~150 MB

### Training Speed Impact

- **Evaluation frequency**: Setting lower `eval_steps` increases evaluation frequency
- More evaluations = more PSNR/SSIM calculations = slightly slower training
- Frame capture itself has minimal overhead (happens during evaluation)
- Video generation occurs after training completes (no impact on training speed)
- **Recommendation**: Balance between video smoothness and training speed
  - For fast training: `eval_steps=100` (default)
  - For smooth videos: `eval_steps=video_iterations`

### File Size Estimates

For a 512×512 image:
- **25 iterations**: ~400 frames, ~15 MB video, ~13 seconds @ 30 FPS
- **50 iterations**: ~200 frames, ~8 MB video, ~7 seconds @ 30 FPS
- **100 iterations**: ~100 frames, ~4 MB video, ~3 seconds @ 30 FPS

## Troubleshooting

### Video Not Generated

1. **Check if feature is enabled**: `make_training_video=True`
2. **Verify OpenCV is installed**: `pip install opencv-python`
3. **Check training completed**: Video generates after training finishes
4. **Check eval_steps**: Frames can only be captured at evaluation steps
5. **Look for error messages**: Check training logs for video generation errors

### Video Has Fewer Frames Than Expected

- **Problem**: Set `video_iterations=50` but video only has frames every 100 iterations
- **Cause**: `eval_steps` is larger than `video_iterations`
- **Solution**: Set `eval_steps = video_iterations` or lower
- **Example**: If `video_iterations=50`, set `eval_steps=50` (not 100)

### Video Quality Issues

- **Video too choppy**: Decrease `video_iterations` (more frames)
- **File too large**: Increase `video_iterations` (fewer frames)
- **Colors look wrong**: This is normal - OpenCV uses BGR format internally

### Memory Errors

- **GPU out of memory**: Frames are stored on CPU, check system RAM
- **System RAM full**: Increase `video_iterations` to capture fewer frames
- **Large images**: Consider downsampling input image

## Examples

### Standard Training with Video

```python
from quick_start import set_config, train

config = set_config(
    input_filenames="cat.png",
    gaussians=[5000],
    steps=[3500],
    make_training_video=True,
    video_iterations=50,
    eval_steps=50  # Match video_iterations!
)

results = train(config)
# Video saved to: output/session_X/cat-5000-3500/training_video.mp4
```

### Batch Training with Videos

```python
config = set_config(
    input_filenames=["cat.png", "dog.png"],
    gaussians=[1000, 5000],
    steps=[2000],
    make_training_video=True,
    video_iterations=50,
    eval_steps=50  # Match video_iterations!
)

results = train(config)
# Generates 4 videos (2 images × 2 gaussian counts)
# - output/session_X/cat-1000-2000/training_video.mp4
# - output/session_X/cat-5000-2000/training_video.mp4
# - output/session_X/dog-1000-2000/training_video.mp4
# - output/session_X/dog-5000-2000/training_video.mp4
```

### Adaptive Refinement with Video

```python
config = set_config(
    input_filenames="cat.png",
    gaussians=[10000],
    steps=[5000],
    make_training_video=True,
    video_iterations=100,
    eval_steps=100,  # Match video_iterations!
    adaptive_refinement=True,
    adaptive_error_threshold=0.3
)

results = train(config)
# Video shows adaptive patching refinement process
```

## Integration Points

The feature integrates with the following files:

1. **`quick_start/config.py`**: Configuration dataclass and validation
2. **`model.py`**: Frame capture and video generation logic
3. **`quick_start/training.py`**: Parameter passing through training pipeline
4. **`cfgs/default.yaml`**: Default configuration values
5. **`quick-start.ipynb`**: User-facing interface

## Future Enhancements

Potential improvements for future versions:

- Support for different video codecs (H.265, VP9)
- Adjustable video quality settings
- Option to include additional visualizations (gaussian positions, error maps)
- Support for multi-view comparison (e.g., 2×2 grid)
- Progress bar during video generation
- Ability to customize text overlay style and position

## Version Information

- **Feature added**: 2025-11-06
- **Compatible with**: Image-GS v1.0.0+
- **Dependencies**: opencv-python >= 4.0

## Support

For issues or questions about the video generation feature:
1. Check this documentation first
2. Verify all tests pass: `python test_video_feature.py`
3. Report issues at: https://github.com/MichaelEight/image-gs/issues
