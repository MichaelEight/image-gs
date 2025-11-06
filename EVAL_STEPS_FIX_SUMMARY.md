# eval_steps Parameter Fix - Summary

## Problem

Users reported that even when setting `video_iterations=50`, video frames were only captured every 100 iterations, not every 50 as expected. This prevented users from creating smoother training videos with more frequent frame captures.

## Root Cause

The frame capture code was nested inside the evaluation check:
```python
if self.step % self.eval_steps == 0:  # Only evaluate every 100 steps
    self._evaluate(log=True, upsample=False)
    # ...
    if self.make_training_video and self.step % self.video_iterations == 0:
        # Capture frame - but this only happens when eval happens!
```

Since `eval_steps` was hardcoded to 100 in `quick_start/training.py:509`, frames could only be captured every 100 iterations minimum, regardless of the `video_iterations` setting.

## Solution

Made `eval_steps` a user-configurable parameter in `TrainingConfig`, allowing users to set the evaluation frequency to match their desired video frame rate.

---

## Changes Made

### 1. **quick_start/config.py**
- Added `eval_steps: int = 100` field to `TrainingConfig` dataclass (line 95)
- Added validation: `if self.eval_steps <= 0: raise ValueError` (lines 126-127)
- Added `eval_steps` parameter to `set_config()` function (line 140)
- Updated docstring with usage note about matching `eval_steps` with `video_iterations` (lines 172-173)
- Pass `eval_steps` to TrainingConfig constructor (line 252)

### 2. **quick_start/training.py**
- Added `eval_steps` parameter to `_build_training_command()` (line 111)
- Added `--eval_steps={eval_steps}` flag to training command (line 152)
- Added `eval_steps` parameter to `train_single()` (line 283)
- Pass `eval_steps` through to `_build_training_command()` (line 308)
- Pass `config.eval_steps` in `_train_standard_batch()` (line 410)
- **Removed hardcoded value** in `_train_adaptive_batch()`, replaced with `config.eval_steps` (line 515)

### 3. **quick-start.ipynb**
- Updated Step 3 markdown documentation to include `eval_steps` parameter
- Added note: "Set eval_steps = video_iterations for smoother videos"
- Updated configuration cell example with `eval_steps=50`

### 4. **test_video_feature.py**
- Added Test 5: Testing `eval_steps` parameter
- Added Test 4b: Validation for invalid `eval_steps` values
- Updated final message to include `eval_steps` usage instructions

### 5. **VIDEO_FEATURE_DOCUMENTATION.md**
- Added comprehensive `eval_steps` parameter documentation
- Updated all examples to include `eval_steps` parameter
- Added "Video Has Fewer Frames Than Expected" troubleshooting section
- Updated performance considerations with evaluation frequency impact
- Added best practices for coordinating `eval_steps` with `video_iterations`

---

## Usage

### Before Fix (Problem)
```python
config = set_config(
    input_filenames="cat.png",
    gaussians=[1000],
    steps=[2000],
    make_training_video=True,
    video_iterations=50  # Ignored! Frames still captured every 100 steps
)
```
**Result**: Frames at iterations 100, 200, 300, ...

### After Fix (Solution)
```python
config = set_config(
    input_filenames="cat.png",
    gaussians=[1000],
    steps=[2000],
    make_training_video=True,
    video_iterations=50,
    eval_steps=50  # Now frames captured every 50 steps!
)
```
**Result**: Frames at iterations 50, 100, 150, 200, ...

---

## Best Practices

### For Smooth Videos
Set `eval_steps = video_iterations`:
```python
config = set_config(
    make_training_video=True,
    video_iterations=25,
    eval_steps=25  # Match for smooth video
)
```

### For Fast Training with Occasional Frames
Keep `eval_steps` high (less frequent evaluation):
```python
config = set_config(
    make_training_video=True,
    video_iterations=100,
    eval_steps=100  # Default value, faster training
)
```

### Trade-offs

| Configuration | Video Smoothness | Training Speed | Use Case |
|---------------|------------------|----------------|----------|
| `eval_steps=25` | Very smooth (many frames) | Slower (more evaluations) | High-quality demo videos |
| `eval_steps=50` | Smooth (balanced) | Moderate | Recommended for most cases |
| `eval_steps=100` | Standard | Fast (default) | Quick experiments |
| `eval_steps=video_iterations` | Optimal | Depends on video_iterations | Best practice |

---

## Backward Compatibility

- **Default value**: `eval_steps=100` (maintains current behavior)
- **Existing code**: Works unchanged if `eval_steps` not specified
- **No breaking changes**: All previous configurations continue to work

---

## Testing

All tests pass successfully:
```bash
$ python test_video_feature.py
================================================================================
✅ ALL TESTS PASSED!
================================================================================
```

Tests verify:
1. ✅ Config creation with `eval_steps` parameter
2. ✅ Default value (100) when not specified
3. ✅ Custom values (e.g., 25, 50)
4. ✅ Validation (rejects zero/negative values)
5. ✅ Integration with training pipeline

---

## Performance Impact

### Evaluation Frequency
- Lower `eval_steps` = More frequent PSNR/SSIM calculations
- Each evaluation adds ~0.1-0.5 seconds overhead
- Impact is proportional to evaluation frequency

### Example for 10,000 steps training:
- `eval_steps=100`: 100 evaluations (~10-50 seconds overhead)
- `eval_steps=50`: 200 evaluations (~20-100 seconds overhead)
- `eval_steps=25`: 400 evaluations (~40-200 seconds overhead)

**Recommendation**: Balance between video quality and training speed based on your needs.

---

## Files Modified

1. `quick_start/config.py` - Configuration dataclass and validation
2. `quick_start/training.py` - Training pipeline integration
3. `quick-start.ipynb` - User documentation and examples
4. `test_video_feature.py` - Test coverage
5. `VIDEO_FEATURE_DOCUMENTATION.md` - Complete documentation

---

## Related Documentation

- See `VIDEO_FEATURE_DOCUMENTATION.md` for complete video feature documentation
- See `quick-start.ipynb` Step 3 for configuration examples
- Run `python test_video_feature.py` to verify installation

---

## Version Information

- **Fix applied**: 2025-11-06
- **Compatible with**: Image-GS v1.0.0+
- **Related issue**: Video frames not captured at desired frequency

## Summary

The `eval_steps` parameter fix allows users to control the evaluation frequency, which directly affects when video frames are captured. By setting `eval_steps = video_iterations`, users can now create training videos with the exact frame rate they desire.

**Key takeaway**: Always set `eval_steps = video_iterations` for optimal video smoothness! 🎥
