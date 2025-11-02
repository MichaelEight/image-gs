import argparse

import torch

from model import GaussianSplatting2D
from utils.misc_utils import load_cfg


def get_gaussian_cfg(args):
    gaussian_cfg = f"num-{args.num_gaussians:d}"
    if args.disable_inverse_scale:
        gaussian_cfg += f"_scale-{args.init_scale:.1f}"
    else:
        gaussian_cfg += f"_inv-scale-{args.init_scale:.1f}"
    if not args.quantize:
        args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits = 32, 32, 32, 32
    min_bits = min(args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits)
    max_bits = max(args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits)
    if min_bits < 4 or max_bits > 32:
        raise ValueError(
            f"Bit precision must be between 4 and 32 but got: {args.pos_bits:d}, {args.scale_bits:d}, {args.rot_bits:d}, {args.feat_bits:d}")
    gaussian_cfg += f"_bits-{args.pos_bits:d}-{args.scale_bits:d}-{args.rot_bits:d}-{args.feat_bits:d}"
    if not args.disable_topk_norm:
        gaussian_cfg += f"_top-{args.topk:d}"
    gaussian_cfg += f"_{args.init_mode[0]}-{args.init_random_ratio:.1f}"
    return gaussian_cfg


def get_log_dir(args):
    gaussian_cfg = get_gaussian_cfg(args)
    loss_cfg = f"l1-{args.l1_loss_ratio:.1f}_l2-{args.l2_loss_ratio:.1f}_ssim-{args.ssim_loss_ratio:.1f}"
    folder = f"{gaussian_cfg}_{loss_cfg}"
    if args.downsample:
        folder += f"_ds-{args.downsample_ratio:.1f}"
    if not args.disable_lr_schedule:
        folder += f"_decay-{args.max_decay_times:d}-{args.decay_ratio:.1f}"
    if not args.disable_prog_optim:
        folder += "_prog"
    return f"{args.log_root}/{args.exp_name}/{folder}"


def main(args):
    # Check if adaptive refinement is enabled
    if hasattr(args, 'adaptive_refinement') and args.adaptive_refinement:
        from quick_start.adaptive_training import train_adaptive
        from quick_start.config import AdaptiveRefinementConfig

        # Create adaptive refinement config from args
        adaptive_config = AdaptiveRefinementConfig(
            enable=True,
            patch_size=args.adaptive_patch_size,
            overlap=args.adaptive_overlap,
            psnr_weight=args.adaptive_psnr_weight,
            ssim_weight=args.adaptive_ssim_weight,
            error_threshold=args.adaptive_error_threshold,
            base_gaussians=args.adaptive_base_gaussians,
            refinement_gaussian_increment=args.adaptive_refinement_increment,
            max_refinement_iterations=args.adaptive_max_iterations,
            max_gaussians_per_patch=args.adaptive_max_gaussians_per_patch
        )

        # Run adaptive training
        result = train_adaptive(
            args,
            adaptive_config,
            workspace_dir=args.data_root,
            session_name=args.exp_name
        )

        print(f"\nAdaptive refinement complete!")
        print(f"Results saved to: {result['output_dir']}")
    else:
        # Standard training/rendering
        args.log_dir = get_log_dir(args)
        ImageGS = GaussianSplatting2D(args)
        if args.eval:
            ImageGS.render(render_height=args.render_height)
        else:
            ImageGS.optimize()


if __name__ == "__main__":
    torch.hub.set_dir("models/torch")
    parser = argparse.ArgumentParser()
    parser = load_cfg(cfg_path="cfgs/default.yaml", parser=parser)
    arguments = parser.parse_args()
    main(arguments)
