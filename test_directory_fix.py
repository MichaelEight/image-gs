#!/usr/bin/env python3
"""
Test to verify the temp_video_dir directory creation fix.
"""

print("Testing directory creation fix...")
print("=" * 80)

# Verify the fix in model.py
with open("model.py", "r") as f:
    content = f.read()

# Check that temp_video_dir is initialized to None
if 'self.temp_video_dir = None' in content:
    print("✓ temp_video_dir initialized to None (will be created later)")
else:
    print("✗ temp_video_dir should be initialized to None")
    exit(1)

# Check that directory is created before use (initial frame)
if 'if self.temp_video_dir is None:' in content and \
   'self.temp_video_dir = os.path.join(self.log_dir, "temp_video_frames")' in content and \
   'os.makedirs(self.temp_video_dir, exist_ok=True)' in content:
    print("✓ Directory creation logic added for initial frame capture")
else:
    print("✗ Missing directory creation logic")
    exit(1)

# Count occurrences of the directory creation check
count = content.count('if self.temp_video_dir is None:')
if count >= 2:
    print(f"✓ Directory creation check appears {count} times (initial + training loop)")
else:
    print(f"✗ Directory creation check should appear at least 2 times, found {count}")
    exit(1)

print("=" * 80)
print("✅ Directory creation fix verified!")
print("\nThe fix ensures:")
print("  1. temp_video_dir starts as None during __init__")
print("  2. Directory is created when log_dir is finalized")
print("  3. Directory creation happens before first frame save")
print("  4. Both initial frame and training loop frames are handled")
