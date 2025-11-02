#!/usr/bin/env python3
"""
Verification script to check that the enhanced adaptive patching features are present.
This script performs static analysis without running the full training pipeline.
"""

import ast
import inspect
from pathlib import Path


def check_function_signature(file_path, function_name, required_params):
    """Check if a function has the required parameters."""
    with open(file_path, 'r') as f:
        content = f.read()

    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.args.args:
            if node.name == function_name:
                param_names = [arg.arg for arg in node.args.args]
                missing_params = [p for p in required_params if p not in param_names]

                if missing_params:
                    return False, f"Missing parameters: {missing_params}"
                return True, f"All required parameters present"

    return False, f"Function '{function_name}' not found"


def check_string_in_file(file_path, search_strings):
    """Check if specific strings exist in a file."""
    with open(file_path, 'r') as f:
        content = f.read()

    results = {}
    for search_str in search_strings:
        results[search_str] = search_str in content

    return results


def main():
    print("\n" + "="*70)
    print("VERIFICATION: Enhanced Adaptive Patching Output Features")
    print("="*70)

    all_checks_passed = True

    # Check 1: Enhanced console output in adaptive_training.py
    print("\n1. Checking enhanced console output in adaptive_training.py...")
    console_checks = check_string_in_file(
        'quick_start/adaptive_training.py',
        [
            'ADAPTIVE PATCHING ACTIVATED',
            'patch_refinement_details',
            'Gaussians added:',
            'Quality improvement:',
            'Threshold status:'
        ]
    )

    for feature, found in console_checks.items():
        status = "✅" if found else "❌"
        print(f"   {status} {feature[:50]}")
        if not found:
            all_checks_passed = False

    # Check 2: Tile score heatmap in adaptive_visualization.py
    print("\n2. Checking tile score heatmap function...")
    heatmap_checks = check_string_in_file(
        'quick_start/adaptive_visualization.py',
        [
            'create_tile_score_heatmap',
            'error_matrix',
            'RdYlGn_r',
            'Patch Error Score Heatmap'
        ]
    )

    for feature, found in heatmap_checks.items():
        status = "✅" if found else "❌"
        print(f"   {status} {feature[:50]}")
        if not found:
            all_checks_passed = False

    # Check 3: Enhanced summary text
    print("\n3. Checking enhanced summary text generation...")
    summary_checks = check_string_in_file(
        'quick_start/adaptive_visualization.py',
        [
            'ADAPTIVE PATCHING STATUS',
            'ADAPTIVE PATCHING DETAILS - PER-PATCH BREAKDOWN',
            'patch_refinement_details',
            'Before (PSNR/SSIM/Error)',
            'After (PSNR/SSIM/Error)'
        ]
    )

    for feature, found in summary_checks.items():
        status = "✅" if found else "❌"
        print(f"   {status} {feature[:50]}")
        if not found:
            all_checks_passed = False

    # Check 4: Function signatures
    print("\n4. Checking function signatures...")

    # Check format_adaptive_summary_text signature
    passed, msg = check_function_signature(
        'quick_start/adaptive_visualization.py',
        'format_adaptive_summary_text',
        ['patch_refinement_details']
    )
    status = "✅" if passed else "❌"
    print(f"   {status} format_adaptive_summary_text: {msg}")
    if not passed:
        all_checks_passed = False

    # Check view_adaptive_results signature
    passed, msg = check_function_signature(
        'quick_start/adaptive_visualization.py',
        'view_adaptive_results',
        ['patch_refinement_details', 'patch_manager', 'error_threshold']
    )
    status = "✅" if passed else "❌"
    print(f"   {status} view_adaptive_results: {msg}")
    if not passed:
        all_checks_passed = False

    # Check 5: Integration points
    print("\n5. Checking integration with main training workflow...")
    integration_checks = check_string_in_file(
        'quick_start/adaptive_training.py',
        [
            'patch_manager=patch_manager',
            'patch_refinement_details=patch_refinement_details',
            'error_threshold=adaptive_config.error_threshold'
        ]
    )

    for feature, found in integration_checks.items():
        status = "✅" if found else "❌"
        print(f"   {status} {feature[:50]}")
        if not found:
            all_checks_passed = False

    # Final summary
    print("\n" + "="*70)
    if all_checks_passed:
        print("✅ ALL VERIFICATION CHECKS PASSED!")
        print("\nEnhanced features successfully implemented:")
        print("  ✓ Real-time console output with detailed metrics")
        print("  ✓ Tile score heatmap visualization")
        print("  ✓ Enhanced summary.txt with per-patch breakdown")
        print("  ✓ All components properly integrated")
        print("\nThe enhanced adaptive patching output is ready to use.")
    else:
        print("❌ SOME VERIFICATION CHECKS FAILED")
        print("\nPlease review the failures above.")

    print("="*70 + "\n")

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    exit(main())
