import csv
import logging
import math
import os
import sys
import warnings
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim
from lpips import LPIPS
from pytorch_msssim import MS_SSIM
from torchvision.transforms.functional import gaussian_blur

from gsplat import (
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
    rasterize_gaussians_sum,
)
from utils.flip import LDRFLIPLoss
from utils.image_utils import (
    compute_image_gradients,
    get_grid,
    get_psnr,
    load_images,
    save_error_maps,
    save_image,
    separate_image_channels,
    visualize_added_gaussians,
    visualize_gaussian_footprint,
    visualize_gaussian_position,
)
from utils.misc_utils import clean_dir, get_latest_ckpt_step, save_cfg, set_random_seed
from utils.quantization_utils import ste_quantize
from utils.saliency_utils import get_smap

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")


class GaussianSplatting2D(nn.Module):
    def __init__(self, args):
        super(GaussianSplatting2D, self).__init__()
        self.evaluate = args.eval
        set_random_seed(seed=args.seed)
        self.device = args.device
        self.dtype = torch.float32
        self._init_logging(args)
        self._init_target(args)
        self._init_bit_precision(args)
        self._init_gaussians(args)
        self._init_loss(args)
        self._init_optimization(args)
        # Initialization
        if self.evaluate:
            self.ckpt_file = args.ckpt_file
            self._load_model()
        elif args.init_from_checkpoint:
            self._load_pretrained_gaussians(args)
        else:
            self._init_pos_scale_feat(args)

    def _init_logging(self, args):
        self.log_dir = args.log_dir
        self.log_level = args.log_level
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.train_dir = os.path.join(self.log_dir, "train")
        self.eval_dir = os.path.join(self.log_dir, "eval")
        self.save_image_format = args.save_image_format
        self.save_plot_format = args.save_plot_format
        self.vis_gaussians = args.vis_gaussians
        self.save_image_steps = args.save_image_steps
        self.save_ckpt_steps = args.save_ckpt_steps
        self.eval_steps = args.eval_steps
        # Video generation settings
        self.make_training_video = getattr(args, 'make_training_video', False)
        self.video_iterations = getattr(args, 'video_iterations', 50)
        self.video_frames = []  # Store frames for video generation
        if not self.evaluate:
            clean_dir(path=self.log_dir)
            os.makedirs(self.log_dir, exist_ok=False)
            os.makedirs(self.ckpt_dir, exist_ok=False)
            os.makedirs(self.train_dir, exist_ok=False)
        else:
            os.makedirs(self.eval_dir, exist_ok=True)
        self._gen_logger(args)
        if not self.evaluate:
            save_cfg(path=f"{self.log_dir}/cfg_train.yaml", args=args)

    def _gen_logger(self, args):
        log_fname = "log_train"
        if self.evaluate:
            log_fname = "log_eval"
        log_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(level=log_level)
        self.worklog = logging.getLogger("Image-GS Logger")
        self.worklog.propagate = False
        datefmt = "%Y/%m/%d %H:%M:%S"
        fileHandler = logging.FileHandler(f"{self.log_dir}/{log_fname}.txt", mode="a", encoding="utf8")
        fileHandler.setFormatter(logging.Formatter(fmt="[{asctime}] {message}", datefmt=datefmt, style="{"))
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter(fmt="\x1b[32m[{asctime}] \x1b[0m{message}", datefmt=datefmt, style="{"))
        self.worklog.handlers = [fileHandler, consoleHandler]
        action = "rendering" if self.evaluate else "optimizing"
        self.worklog.info(f"Start {action} {args.num_gaussians:d} Gaussians for '{args.input_path}'")
        self.worklog.info("***********************************************")

    def _init_target(self, args):
        self.gamma = args.gamma
        self.downsample = args.downsample
        if self.downsample:
            self.downsample_ratio = float(args.downsample_ratio)
        self.block_h, self.block_w = 16, 16  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        self._load_target_images(path=os.path.join(args.data_root, args.input_path))
        if self.downsample:
            self.gt_images_upsampled = self.gt_images
            self.img_h_upsampled, self.img_w_upsampled = self.img_h, self.img_w
            self.tile_bounds_upsampled = self.tile_bounds
            self._load_target_images(path=os.path.join(args.data_root, args.input_path), downsample_ratio=self.downsample_ratio)
            if not self.evaluate:
                path = f"{self.log_dir}/gt_upsample-{self.downsample_ratio:.1f}_res-{self.img_h_upsampled:d}x{self.img_w_upsampled:d}"
                self._separate_and_save_images(images=self.gt_images_upsampled, channels=self.input_channels, path=path)
        self.num_pixels = self.img_h * self.img_w
        if not self.evaluate:
            path = f"{self.log_dir}/gt_res-{self.img_h:d}x{self.img_w:d}"
            self._separate_and_save_images(images=self.gt_images, channels=self.input_channels, path=path)

    def _load_target_images(self, path, downsample_ratio=None):
        self.gt_images, self.input_channels, self.image_fnames, self.bit_depths = load_images(
            load_path=path, downsample_ratio=downsample_ratio, gamma=self.gamma)
        self.gt_images = torch.from_numpy(self.gt_images).to(dtype=self.dtype, device=self.device)
        self.img_h, self.img_w = self.gt_images.shape[1:]
        self.tile_bounds = ((self.img_w + self.block_w - 1) // self.block_w, (self.img_h + self.block_h - 1) // self.block_h, 1)

    def _separate_and_save_images(self, images, channels, path):
        images_sep = separate_image_channels(images=images, input_channels=channels)
        for idx, image in enumerate(images_sep, 1):
            suffix = "" if len(images_sep) == 1 else f"_{idx:d}"
            save_image(image, f"{path}{suffix}.{self.save_image_format}", gamma=self.gamma)

    def _init_bit_precision(self, args):
        self.quantize = args.quantize
        self.pos_bits = args.pos_bits
        self.scale_bits = args.scale_bits
        self.rot_bits = args.rot_bits
        self.feat_bits = args.feat_bits

    def _init_gaussians(self, args):
        self.num_gaussians = args.num_gaussians
        self.total_num_gaussians = args.num_gaussians
        self.disable_prog_optim = args.disable_prog_optim
        if not self.disable_prog_optim and not self.evaluate:
            self.initial_ratio = args.initial_ratio
            self.add_times = args.add_times
            self.add_steps = args.add_steps
            self.num_gaussians = math.ceil(self.initial_ratio * self.total_num_gaussians)
            self.max_add_num = math.ceil(float(self.total_num_gaussians-self.num_gaussians) / self.add_times)
            min_steps = self.add_steps * self.add_times + args.post_min_steps
            if args.max_steps < min_steps:
                self.worklog.info(f"Max steps ({args.max_steps:d}) is too small for progressive optimization. Resetting to {min_steps:d}")
                args.max_steps = min_steps
        self.topk = args.topk  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        self.eps = 1e-7 if args.disable_tiles else 1e-4  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        self.init_scale = args.init_scale
        self.disable_topk_norm = args.disable_topk_norm
        self.disable_inverse_scale = args.disable_inverse_scale
        self.disable_color_init = args.disable_color_init
        self.xy = nn.Parameter(torch.rand(self.num_gaussians, 2, dtype=self.dtype, device=self.device), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(self.num_gaussians, 2, dtype=self.dtype, device=self.device), requires_grad=True)
        self.rot = nn.Parameter(torch.zeros(self.num_gaussians, 1, dtype=self.dtype, device=self.device), requires_grad=True)
        self.feat_dim = sum(self.input_channels)
        self.feat = nn.Parameter(torch.rand(self.num_gaussians, self.feat_dim, dtype=self.dtype, device=self.device), requires_grad=True)
        self.vis_feat = nn.Parameter(torch.rand_like(self.feat), requires_grad=False)  # Only used for Gaussian ID visualization
        self._log_compression_rate()

    def _log_compression_rate(self):
        bytes_uncompressed = 0.0
        curr_channel = 0
        for num_channels, bit_depth in zip(self.input_channels, self.bit_depths):
            bytes_uncompressed += float(self.gt_images[curr_channel:curr_channel+num_channels].numel()) * (bit_depth / 8.0)
            curr_channel += num_channels
        bpp_uncompressed = 0.0
        for num_channels, bit_depth in zip(self.input_channels, self.bit_depths):
            bpp_uncompressed += float(num_channels) * bit_depth
        bppc_uncompressed = bpp_uncompressed / self.feat_dim
        self.worklog.info(f"Uncompressed: {bytes_uncompressed/1e3:.2f} KB | {bpp_uncompressed:.3f} bpp | {bppc_uncompressed:.3f} bppc")
        bits_compressed = (2*self.pos_bits + 2*self.scale_bits + self.rot_bits + self.feat_dim*self.feat_bits) * self.total_num_gaussians
        bytes_compressed = bits_compressed / 8.0
        bpp_compressed = float(bits_compressed) / self.num_pixels
        bppc_compressed = bpp_compressed / self.feat_dim
        self.num_bytes = bytes_compressed
        self.worklog.info(f"Compressed: {bytes_compressed/1e3:.2f} KB | {bpp_compressed:.3f} bpp | {bppc_compressed:.3f} bppc")
        self.worklog.info(f"Compression rate: {bpp_uncompressed/bpp_compressed:.2f}x | {100.0*bpp_compressed/bpp_uncompressed:.2f}%")
        self.worklog.info("***********************************************")

    def _init_loss(self, args):
        self.l1_loss = None
        self.l2_loss = None
        self.ssim_loss = None
        self.l1_loss_ratio = args.l1_loss_ratio
        self.l2_loss_ratio = args.l2_loss_ratio
        self.ssim_loss_ratio = args.ssim_loss_ratio

    def _init_optimization(self, args):
        self.disable_tiles = args.disable_tiles
        self.start_step = 1
        self.max_steps = args.max_steps
        self.pos_lr = args.pos_lr
        self.scale_lr = args.scale_lr
        self.rot_lr = args.rot_lr
        self.feat_lr = args.feat_lr
        self.optimizer = torch.optim.Adam([{'params': self.xy, 'lr': self.pos_lr},
                                           {'params': self.scale, 'lr': self.scale_lr},
                                           {'params': self.rot, 'lr': self.rot_lr},
                                           {'params': self.feat, 'lr': self.feat_lr}])
        self.disable_lr_schedule = args.disable_lr_schedule
        if not self.disable_lr_schedule:
            self.decay_ratio = args.decay_ratio
            self.check_decay_steps = args.check_decay_steps
            self.max_decay_times = args.max_decay_times
            self.decay_threshold = args.decay_threshold

    def _init_pos_scale_feat(self, args):
        self.init_mode = args.init_mode
        self.init_random_ratio = args.init_random_ratio
        self.pixel_xy = get_grid(h=self.img_h, w=self.img_w).to(dtype=self.dtype, device=self.device).reshape(-1, 2)
        with torch.no_grad():
            # Position
            if self.init_mode == 'gradient':
                self._compute_gmap()
                self.xy.copy_(self._sample_pos(prob=self.image_gradients))
            elif self.init_mode == 'saliency':
                self.smap_filter_size = args.smap_filter_size
                self._compute_smap(path="models")
                self.xy.copy_(self._sample_pos(prob=self.saliency))
            else:
                selected = np.random.choice(self.num_pixels, self.num_gaussians, replace=False, p=None)
                self.xy.copy_(self.pixel_xy.detach().clone()[selected])
            # Scale
            self.scale.fill_(self.init_scale if self.disable_inverse_scale else 1.0/self.init_scale)
            # Feature
            if not self.disable_color_init:
                self.feat.copy_(self._get_target_features(positions=self.xy).detach().clone())

    def _sample_pos(self, prob):
        num_random = round(self.init_random_ratio*self.num_gaussians)
        selected_random = np.random.choice(self.num_pixels, num_random, replace=False, p=None)
        selected_other = np.random.choice(self.num_pixels, self.num_gaussians-num_random, replace=False, p=prob)
        return torch.cat([self.pixel_xy.detach().clone()[selected_random], self.pixel_xy.detach().clone()[selected_other]], dim=0)

    def _compute_gmap(self):
        gy, gx = compute_image_gradients(np.power(self.gt_images.detach().cpu().clone().numpy(), 1.0/self.gamma))
        g_norm = np.hypot(gy, gx).astype(np.float32)
        g_norm = g_norm / g_norm.max()
        save_image(g_norm, f"{self.log_dir}/gmap_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}")
        g_norm = np.power(g_norm.reshape(-1), 2.0)
        self.image_gradients = g_norm / g_norm.sum()
        self.worklog.info("Image gradient map successfully saved")
        self.worklog.info("***********************************************")

    def _compute_smap(self, path):
        smap = get_smap(torch.pow(self.gt_images.detach().clone(), 1.0/self.gamma), path, self.smap_filter_size)
        save_image(smap, f"{self.log_dir}/smap_res-{self.img_h:d}x{self.img_w:d}.{self.save_image_format}")
        self.saliency = (smap / smap.sum()).reshape(-1)
        self.worklog.info("Saliency map successfully saved")
        self.worklog.info("***********************************************")

    def _get_target_features(self, positions):
        with torch.no_grad():
            # gt_images [1, C, H, W]; positions [1, 1, P, 2]; top-left [-1, -1]; bottom-right [1, 1]
            target_features = F.grid_sample(self.gt_images.unsqueeze(0), positions[None, None, ...] * 2.0 - 1.0, align_corners=False)
            target_features = target_features[0, :, 0, :].permute(1, 0)  # [P, C]
        return target_features

    def _load_model(self):
        if self.ckpt_file != "":
            ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_file)
        else:
            latest_step = get_latest_ckpt_step(self.ckpt_dir)
            if latest_step == -1:
                raise FileNotFoundError(f"No checkpoint found in '{self.ckpt_dir}'")
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_step-{latest_step:d}.pt")
        checkpoint = torch.load(ckpt_path, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.start_step = checkpoint["step"]+1
        self.worklog.info(f"Checkpoint '{ckpt_path}' successfully loaded")
        self.worklog.info("***********************************************")

    def _load_pretrained_gaussians(self, args):
        """Load pretrained Gaussians from checkpoint file for initialization."""
        ckpt_path = args.init_checkpoint_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: '{ckpt_path}'")

        self.worklog.info(f"Loading pretrained Gaussians from '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, weights_only=False)
        pretrained_state = checkpoint['state_dict']

        # Get number of Gaussians in checkpoint
        pretrained_num_gaussians = pretrained_state['xy'].shape[0]

        with torch.no_grad():
            if pretrained_num_gaussians != self.num_gaussians:
                if args.init_partial:
                    # Partial loading: load what fits, initialize remaining
                    num_to_load = min(pretrained_num_gaussians, self.num_gaussians)
                    self.worklog.info(f"Partial initialization: loading {num_to_load} of {pretrained_num_gaussians} Gaussians")

                    # Copy pretrained Gaussians
                    self.xy[:num_to_load].copy_(pretrained_state['xy'][:num_to_load])
                    self.scale[:num_to_load].copy_(pretrained_state['scale'][:num_to_load])
                    self.rot[:num_to_load].copy_(pretrained_state['rot'][:num_to_load])
                    self.feat[:num_to_load].copy_(pretrained_state['feat'][:num_to_load])

                    # Initialize remaining Gaussians if needed
                    if self.num_gaussians > pretrained_num_gaussians:
                        self.worklog.info(f"Initializing remaining {self.num_gaussians - num_to_load} Gaussians")
                        self._init_pos_scale_feat_range(args, num_to_load, self.num_gaussians)
                else:
                    # Strict matching: raise error if sizes don't match
                    raise ValueError(
                        f"Checkpoint has {pretrained_num_gaussians} Gaussians but model expects {self.num_gaussians}. "
                        f"Set init_partial=True to allow partial initialization."
                    )
            else:
                # Exact match: load all Gaussians
                self.xy.copy_(pretrained_state['xy'])
                self.scale.copy_(pretrained_state['scale'])
                self.rot.copy_(pretrained_state['rot'])
                self.feat.copy_(pretrained_state['feat'])
                self.worklog.info(f"Loaded all {pretrained_num_gaussians} Gaussians")

        self.worklog.info("Pretrained Gaussians successfully loaded")
        self.worklog.info("***********************************************")

    def _init_pos_scale_feat_range(self, args, start_idx, end_idx):
        """Initialize positions, scales, and features for a range of Gaussians."""
        self.init_mode = args.init_mode
        self.init_random_ratio = args.init_random_ratio

        if not hasattr(self, 'pixel_xy'):
            self.pixel_xy = get_grid(h=self.img_h, w=self.img_w).to(dtype=self.dtype, device=self.device).reshape(-1, 2)

        num_to_init = end_idx - start_idx

        with torch.no_grad():
            # Position
            if self.init_mode == 'gradient':
                if not hasattr(self, 'image_gradients'):
                    self._compute_gmap()
                # Sample positions for the range
                sampled_pos = self._sample_pos_n(prob=self.image_gradients, n=num_to_init)
                self.xy[start_idx:end_idx].copy_(sampled_pos)
            elif self.init_mode == 'saliency':
                self.smap_filter_size = args.smap_filter_size
                if not hasattr(self, 'saliency'):
                    self._compute_smap(path="models")
                sampled_pos = self._sample_pos_n(prob=self.saliency, n=num_to_init)
                self.xy[start_idx:end_idx].copy_(sampled_pos)
            else:
                # Random sampling
                selected = np.random.choice(self.num_pixels, num_to_init, replace=False, p=None)
                self.xy[start_idx:end_idx].copy_(self.pixel_xy.detach().clone()[selected])

            # Scale
            scale_value = self.init_scale if self.disable_inverse_scale else 1.0/self.init_scale
            self.scale[start_idx:end_idx].fill_(scale_value)

            # Feature
            if not self.disable_color_init:
                self.feat[start_idx:end_idx].copy_(
                    self._get_target_features(positions=self.xy[start_idx:end_idx]).detach().clone()
                )

    def _sample_pos_n(self, prob, n):
        """Sample n positions based on probability distribution."""
        prob_normalized = prob / prob.sum()
        selected = np.random.choice(
            self.num_pixels,
            n,
            replace=False,
            p=prob_normalized.cpu().numpy()
        )
        selected_random = np.random.choice(
            self.num_pixels,
            n,
            replace=False,
            p=None
        )
        selected_flag = torch.rand(n) < self.init_random_ratio
        selected_tensor = torch.from_numpy(selected).to(device=self.device)
        selected_random_tensor = torch.from_numpy(selected_random).to(device=self.device)
        selected_final = torch.where(selected_flag, selected_random_tensor, selected_tensor)
        return self.pixel_xy[selected_final].detach().clone()

    def _save_model(self):
        if self.quantize:
            self._quantize()
        psnr, ssim = self._evaluate(log=False, upsample=False)
        self._evaluate_extra()
        ckpt_data = {"step": self.step,
                     "psnr": psnr,
                     "ssim": ssim,
                     "lpips": self.lpips_final,
                     "flip": self.flip_final,
                     "msssim": self.msssim_final,
                     "bytes": self.num_bytes,
                     "time": self.total_time_accum,
                     "state_dict": self.state_dict(),
                     "optim_state_dict": self.optimizer.state_dict()}
        save_path = f"{self.ckpt_dir}/ckpt_step-{self.step:d}.pt"
        torch.save(ckpt_data, save_path)
        self.worklog.info(f"Checkpoint 'ckpt_step-{self.step:d}.pt' successfully saved")
        self.worklog.info(
            f"PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | LPIPS: {self.lpips_final:.4f} | FLIP: {self.flip_final:.4f} | MS-SSIM: {self.msssim_final:.4f}")
        self.worklog.info("***********************************************")

    def _quantize(self):
        with torch.no_grad():
            self.xy.copy_(ste_quantize(self.xy, self.pos_bits))
            self.scale.copy_(ste_quantize(self.scale, self.scale_bits))
            self.rot.copy_(ste_quantize(self.rot, self.rot_bits))
            self.feat.copy_(ste_quantize(self.feat, self.feat_bits))

    def render(self, render_height=None):
        img_h, img_w = self.img_h, self.img_w
        if render_height is not None:
            img_h, img_w = render_height, round((float(render_height)/img_h)*img_w)
        tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)
        upsample_ratio = float(img_h) / self.img_h
        with torch.no_grad():
            num_prep_runs = 2
            for _ in range(num_prep_runs):
                self.forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark=True)
            images, render_time = self.forward(img_h, img_w, tile_bounds, upsample_ratio)
            path = f"{self.eval_dir}/render_upsample-{upsample_ratio:.1f}_res-{img_h:d}x{img_w:d}"
            self._separate_and_save_images(images=images, channels=self.input_channels, path=path)
        self.worklog.info(f"Step: {self.start_step-1:d} | Time: {render_time:.6f} s")
        self.worklog.info(f"Rendering at resolution ({img_h:d}, {img_w:d}) completed")
        self.worklog.info("***********************************************")

    def benchmark_render_time(self, num_reps, render_height=None):
        img_h, img_w = self.img_h, self.img_w
        if render_height is not None:
            img_h, img_w = render_height, round((float(render_height)/img_h)*img_w)
        tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)
        upsample_ratio = float(img_h) / self.img_h
        with torch.no_grad():
            render_time_all = np.zeros(num_reps, dtype=np.float32)
            num_prep_runs = 2
            for _ in range(num_prep_runs):
                self.forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark=True)
            for rid in range(num_reps):
                render_time = self.forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark=True)
                render_time_all[rid] = render_time
        return render_time_all

    def forward(self, img_h, img_w, tile_bounds, upsample_ratio=None, benchmark=False):
        scale = self._get_scale(upsample_ratio=upsample_ratio)
        xy, rot, feat = self.xy, self.rot, self.feat
        if self.quantize:
            xy, scale, rot, feat = ste_quantize(xy, self.pos_bits), ste_quantize(
                scale, self.scale_bits), ste_quantize(rot, self.rot_bits), ste_quantize(feat, self.feat_bits)
        begin = perf_counter()
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp
        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = xy, radii, conics, num_tiles_hit, feat, img_h, img_w, self.block_h, self.block_w, enable_topk_norm
            out_image = rasterize_gaussians_sum(*tmp)
        else:
            tmp = xy, conics, feat, img_h, img_w
            out_image = rasterize_gaussians_no_tiles(*tmp)
        render_time = perf_counter() - begin
        if benchmark:
            return render_time
        out_image = out_image.view(-1, img_h, img_w, self.feat_dim).permute(0, 3, 1, 2).contiguous()
        return out_image.squeeze(dim=0), render_time

    def _get_scale(self, upsample_ratio=None):
        scale = self.scale
        if not self.disable_inverse_scale:
            scale = 1.0 / scale
        if upsample_ratio is not None:
            scale = upsample_ratio * scale
        return scale

    def _visualize_gaussian_id(self, img_h, img_w, tile_bounds, upsample_ratio=None):
        scale = self._get_scale(upsample_ratio=upsample_ratio)
        xy, rot, feat = self.xy, self.rot, self.feat
        if self.quantize:
            xy, scale, rot, feat = ste_quantize(xy, self.pos_bits), ste_quantize(
                scale, self.scale_bits), ste_quantize(rot, self.rot_bits), ste_quantize(feat, self.feat_bits)
        feat = self.vis_feat * feat.norm(dim=-1, keepdim=True)
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp
        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = xy, radii, conics, num_tiles_hit, feat, img_h, img_w, self.block_h, self.block_w, enable_topk_norm
            out_image = rasterize_gaussians_sum(*tmp)
        else:
            tmp = xy, conics, feat, img_h, img_w
            out_image = rasterize_gaussians_no_tiles(*tmp)
        out_image = out_image.view(-1, img_h, img_w, self.feat_dim).permute(0, 3, 1, 2).contiguous()
        return out_image.squeeze(dim=0)

    def optimize(self):
        self.psnr_curr, self.ssim_curr = 0.0, 0.0
        self.best_psnr, self.best_ssim = 0.0, 0.0
        self.decay_times, self.no_improvement_steps = 0, 0
        self.render_time_accum, self.total_time_accum = 0.0, 0.0
        self.lpips_final, self.flip_final, self.msssim_final = 1.0, 1.0, 0.0

        # Initialize CSV file for metric logging
        self.metrics_csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.metrics_csv_file = open(self.metrics_csv_path, 'w', newline='')
        self.metrics_csv_writer = csv.writer(self.metrics_csv_file)

        # Write CSV header
        self.metrics_csv_writer.writerow([
            'step', 'total_loss', 'l1_loss', 'l2_loss', 'ssim_loss',
            'psnr', 'ssim', 'num_gaussians', 'num_bytes',
            'render_time_accum', 'total_time_accum'
        ])
        self.metrics_csv_file.flush()

        self.step = 0
        with torch.no_grad():
            self._log_images(log_final=False, plot_gaussians=self.vis_gaussians)
            # Capture initial frame for video if enabled
            if self.make_training_video:
                images = self._render_images(upsample=False)
                images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0/self.gamma)
                psnr, ssim = get_psnr(images, torch.pow(self.gt_images, 1.0/self.gamma)).item(), \
                             fused_ssim(images.unsqueeze(0), torch.pow(self.gt_images, 1.0/self.gamma).unsqueeze(0)).item()
                self.video_frames.append({
                    'render': images.clone(),
                    'step': 0,
                    'psnr': psnr,
                    'ssim': ssim
                })
        for step in range(self.start_step, self.max_steps+1):
            self.step = step
            self.optimizer.zero_grad()
            # Rendering
            images, render_time = self.forward(self.img_h, self.img_w, self.tile_bounds)
            self.render_time_accum += render_time
            # Optimization
            begin = perf_counter()
            self._get_total_loss(images)
            self.total_loss.backward()
            self.optimizer.step()
            self.total_time_accum += (perf_counter() - begin + render_time)
            # Logging
            terminate = False
            with torch.no_grad():
                if self.step % self.eval_steps == 0:
                    self._evaluate(log=True, upsample=False)
                    # Log metrics to CSV
                    self.metrics_csv_writer.writerow([
                        self.step,
                        self.total_loss.item() if self.total_loss is not None else '',
                        self.l1_loss.item() if self.l1_loss is not None else '',
                        self.l2_loss.item() if self.l2_loss is not None else '',
                        self.ssim_loss.item() if self.ssim_loss is not None else '',
                        self.psnr_curr,
                        self.ssim_curr,
                        self.num_gaussians,
                        self.num_bytes,
                        self.render_time_accum,
                        self.total_time_accum
                    ])
                    self.metrics_csv_file.flush()
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
                    if not self.disable_lr_schedule and self.num_gaussians == self.total_num_gaussians:
                        terminate = self._lr_schedule()
                if self.step % self.save_image_steps == 0:
                    self._log_images(log_final=False, plot_gaussians=self.vis_gaussians)
                if self.step % self.save_ckpt_steps == 0 and self.num_gaussians == self.total_num_gaussians:
                    self._save_model()
                if not self.disable_prog_optim and self.step % self.add_steps == 0 and self.num_gaussians < self.total_num_gaussians:
                    self._add_gaussians(self.max_add_num, plot_gaussians=self.vis_gaussians)
                if terminate:
                    break
        with torch.no_grad():
            self._log_images(log_final=True, plot_gaussians=self.vis_gaussians)
            self._save_model()
            # Capture final frame for video if enabled
            if self.make_training_video:
                # Evaluate to get final metrics
                psnr, ssim = self._evaluate(log=False, upsample=False)
                images = self._render_images(upsample=False)
                images = torch.pow(torch.clamp(images, 0.0, 1.0), 1.0/self.gamma)
                # Only add if this step wasn't already captured
                if len(self.video_frames) == 0 or self.video_frames[-1]['step'] != self.step:
                    self.video_frames.append({
                        'render': images.clone(),
                        'step': self.step,
                        'psnr': psnr,
                        'ssim': ssim
                    })

        # Close CSV file
        self.metrics_csv_file.close()
        self.worklog.info(f"Metrics saved to: {self.metrics_csv_path}")

        # Generate training video if enabled
        if self.make_training_video and len(self.video_frames) > 0:
            self._generate_training_video()

        self.worklog.info("Optimization completed")
        self.worklog.info("***********************************************")
        self.worklog.info(f"Mean scale: {self._get_scale().mean().item():.4f} (pixel) | {self.scale.mean().item():.4f} (raw)")
        self.worklog.info("***********************************************")
        return self.psnr_curr, self.ssim_curr

    def _get_total_loss(self, images):
        self.total_loss = 0
        if self.l1_loss_ratio > 1e-7:
            self.l1_loss = self.l1_loss_ratio * F.l1_loss(images, self.gt_images)
            self.total_loss += self.l1_loss
        else:
            self.l1_loss = None
        if self.l2_loss_ratio > 1e-7:
            self.l2_loss = self.l2_loss_ratio * F.mse_loss(images, self.gt_images)
            self.total_loss += self.l2_loss
        else:
            self.l2_loss = None
        if self.ssim_loss_ratio > 1e-7:
            self.ssim_loss = self.ssim_loss_ratio * (1 - fused_ssim(images.unsqueeze(0), self.gt_images.unsqueeze(0)))
            self.total_loss += self.ssim_loss
        else:
            self.ssim_loss = None

    def _evaluate(self, log=True, upsample=False):
        if upsample:  # Do not log performance metrics for upsampled images
            log = False
        images = torch.pow(torch.clamp(self._render_images(upsample=upsample), 0.0, 1.0), 1.0/self.gamma)
        gt_images = torch.pow(self.gt_images_upsampled if upsample else self.gt_images, 1.0/self.gamma)
        psnr = get_psnr(images, gt_images).item()
        ssim = fused_ssim(images.unsqueeze(0), gt_images.unsqueeze(0)).item()
        if log:
            self.psnr_curr, self.ssim_curr = psnr, ssim
            loss_results = f"Loss: {self.total_loss.item():.4f}"
            loss_results += f", L1: {self.l1_loss.item():.4f}" if self.l1_loss is not None else ""
            loss_results += f", L2: {self.l2_loss.item():.4f}" if self.l2_loss is not None else ""
            loss_results += f", SSIM: {self.ssim_loss.item():.4f}" if self.ssim_loss is not None else ""
            time_results = f"Total: {self.total_time_accum:.2f} s | Render: {self.render_time_accum:.2f} s"
            self.worklog.info(f"Step: {self.step:d} | {time_results} | {loss_results} | PSNR: {self.psnr_curr:.2f} | SSIM: {self.ssim_curr:.4f}")
        return psnr, ssim

    def _evaluate_extra(self):
        images = torch.pow(torch.clamp(self._render_images(upsample=False), 0.0, 1.0), 1.0/self.gamma)[None, ...]
        gt_images = torch.pow(self.gt_images, 1.0/self.gamma)[None, ...]
        msssim_metric = MS_SSIM(data_range=1.0, size_average=True, channel=self.feat_dim).to(device=self.device).eval()
        self.msssim_final = msssim_metric(images, gt_images).item()
        lpips_metric = LPIPS(net='alex').to(device=self.device).eval()
        flip_metric = LDRFLIPLoss().to(device=self.device).eval()
        num_channels = 1 if self.feat_dim < 3 else 3
        self.lpips_final = lpips_metric(images[:, :num_channels], gt_images[:, :num_channels]).item()
        if self.feat_dim >= 3:
            self.flip_final = flip_metric(images[:, :3], gt_images[:, :3]).item()

    def _log_images(self, log_final=False, plot_gaussians=False):
        images = self._render_images(upsample=False)
        if log_final:
            path = f"{self.log_dir}/render_res-{self.img_h:d}x{self.img_w:d}"
            self._separate_and_save_images(images=images, channels=self.input_channels, path=path)
        psnr, ssim = self._evaluate(log=False, upsample=False)
        path = f"{self.train_dir}/render_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
        self._separate_and_save_images(images=images, channels=self.input_channels, path=path)
        if plot_gaussians:
            path = f"{self.train_dir}/flip-error_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            save_error_maps(path, images, self.gt_images, channels=self.input_channels,
                            gamma=self.gamma, save_image_format=self.save_image_format)
            # path = f"{self.train_dir}/gaussian-footprint_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            # visualize_gaussian_footprint(path, self.xy, self._get_scale(), self.rot, self.feat, self.img_h,
            #                     self.img_w, self.input_channels, alpha=0.8, gamma=self.gamma, save_image_format=self.save_plot_format)
            path = f"{self.train_dir}/gaussian-position_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            every_n = max(1, self.total_num_gaussians // 1000)
            size = 1.5 * (self.img_h * self.img_w) / 1e4
            visualize_gaussian_position(path, images, self.xy, self.input_channels, color="#c0b1fc", size=size,
                                        every_n=every_n, alpha=0.9, gamma=self.gamma, save_image_format=self.save_plot_format)
            images = self._visualize_gaussian_id(self.img_h, self.img_w, self.tile_bounds)
            path = f"{self.train_dir}/gaussian-id_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{self.img_h:d}x{self.img_w:d}"
            self._separate_and_save_images(images=images, channels=self.input_channels, path=path)
        if self.downsample:
            images = self._render_images(upsample=True)
            psnr, ssim = self._evaluate(log=False, upsample=True)
            img_h, img_w = self.img_h_upsampled, self.img_w_upsampled
            path = f"{self.train_dir}/render_upsample-{self.downsample_ratio:.1f}_step-{self.step:d}_psnr-{psnr:.2f}_ssim-{ssim:.4f}_res-{img_h:d}x{img_w:d}"
            self._separate_and_save_images(images=images, channels=self.input_channels, path=path)

    def _render_images(self, upsample=False):
        if upsample:
            images, _ = self.forward(self.img_h_upsampled, self.img_w_upsampled, self.tile_bounds_upsampled, upsample_ratio=self.downsample_ratio)
        else:
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds)
        return images

    def _generate_training_video(self):
        """
        Generate a training progress video from captured frames.
        Creates a side-by-side comparison video with ground truth on the left
        and rendered image on the right, with metrics overlay.
        """
        import cv2

        self.worklog.info(f"Generating training video from {len(self.video_frames)} frames...")

        # Video output path
        video_path = os.path.join(self.log_dir, "training_video.mp4")

        # Video parameters
        fps = 30

        # Get dimensions from first frame
        first_frame_data = self.video_frames[0]
        render_tensor = first_frame_data['render']

        # Convert tensor to numpy array (H, W, C) and scale to 0-255
        render_np = render_tensor.detach().cpu().permute(1, 2, 0).numpy()
        h, w = render_np.shape[:2]

        # Side-by-side width (ground truth + rendered)
        frame_width = w * 2
        frame_height = h

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

        # Get ground truth image
        gt_np = self.gt_images.detach().cpu().permute(1, 2, 0).numpy()
        gt_np = np.clip(gt_np ** (1.0 / self.gamma), 0, 1)

        # Convert ground truth to uint8
        if gt_np.shape[2] == 1:
            # Grayscale
            gt_uint8 = (gt_np * 255).astype(np.uint8)
            gt_uint8 = cv2.cvtColor(gt_uint8, cv2.COLOR_GRAY2BGR)
        else:
            # RGB - convert to BGR for OpenCV
            gt_uint8 = (gt_np * 255).astype(np.uint8)
            gt_uint8 = cv2.cvtColor(gt_uint8, cv2.COLOR_RGB2BGR)

        # Process each frame
        for frame_data in self.video_frames:
            render_tensor = frame_data['render']
            step = frame_data['step']
            psnr = frame_data['psnr']
            ssim = frame_data['ssim']

            # Convert rendered image to numpy
            render_np = render_tensor.detach().cpu().permute(1, 2, 0).numpy()
            render_np = np.clip(render_np ** (1.0 / self.gamma), 0, 1)

            # Convert to uint8
            if render_np.shape[2] == 1:
                # Grayscale
                render_uint8 = (render_np * 255).astype(np.uint8)
                render_uint8 = cv2.cvtColor(render_uint8, cv2.COLOR_GRAY2BGR)
            else:
                # RGB - convert to BGR for OpenCV
                render_uint8 = (render_np * 255).astype(np.uint8)
                render_uint8 = cv2.cvtColor(render_uint8, cv2.COLOR_RGB2BGR)

            # Create side-by-side frame
            side_by_side = np.hstack([gt_uint8, render_uint8])

            # Add text overlay with metrics
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 255)  # White text

            # Add labels
            cv2.putText(side_by_side, "Ground Truth", (10, 30), font, font_scale, color, thickness)
            cv2.putText(side_by_side, "Rendered", (w + 10, 30), font, font_scale, color, thickness)

            # Add metrics on the right side
            metrics_y = h - 70
            cv2.putText(side_by_side, f"Step: {step}", (w + 10, metrics_y), font, font_scale, color, thickness)
            cv2.putText(side_by_side, f"PSNR: {psnr:.2f} dB", (w + 10, metrics_y + 25), font, font_scale, color, thickness)
            cv2.putText(side_by_side, f"SSIM: {ssim:.4f}", (w + 10, metrics_y + 50), font, font_scale, color, thickness)

            # Write frame to video
            video.write(side_by_side)

        # Release video writer
        video.release()

        self.worklog.info(f"Training video saved to: {video_path}")
        self.worklog.info(f"Video contains {len(self.video_frames)} frames at {fps} FPS ({len(self.video_frames)/fps:.2f} seconds)")

    def _lr_schedule(self):
        if (self.psnr_curr <= self.best_psnr + 100*self.decay_threshold or self.ssim_curr <= self.best_ssim + self.decay_threshold):
            self.no_improvement_steps += self.eval_steps
            if self.no_improvement_steps >= self.check_decay_steps:
                self.no_improvement_steps = 0
                self.decay_times += 1
                if self.decay_times > self.max_decay_times:
                    return True
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= self.decay_ratio
                self.worklog.info(f"Learning rate decayed by {self.decay_ratio:.1f}")
                self.worklog.info("***********************************************")
            return False
        else:
            self.best_psnr = self.psnr_curr
            self.best_ssim = self.ssim_curr
            self.no_improvement_steps = 0
            return False

    def _add_gaussians(self, add_num, plot_gaussians=False):
        add_num = min(add_num, self.max_add_num, self.total_num_gaussians-self.num_gaussians)
        if add_num <= 0:
            return
        raw_images = self._render_images(upsample=False)
        images = torch.pow(torch.clamp(raw_images, 0.0, 1.0), 1.0/self.gamma)
        gt_images = torch.pow(self.gt_images, 1.0/self.gamma)
        kernel_size = round(np.sqrt(self.img_h * self.img_w) // 400)
        if kernel_size >= 1:
            kernel_size = max(3, kernel_size)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            gt_images = gaussian_blur(img=gt_images, kernel_size=kernel_size)
        diff_map = (gt_images - images).detach().clone()
        error_map = torch.pow(torch.abs(diff_map).mean(dim=0).reshape(-1), 2.0)
        sample_prob = (error_map / error_map.sum()).cpu().numpy()
        selected = np.random.choice(self.num_pixels, add_num, replace=False, p=sample_prob)
        # New Gaussians
        new_xy = self.pixel_xy.detach().clone()[selected]
        new_scale = torch.ones(add_num, 2, dtype=self.dtype, device=self.device)
        init_scale = self.init_scale
        new_scale.fill_(init_scale if self.disable_inverse_scale else 1.0/init_scale)
        new_rot = torch.zeros(add_num, 1, dtype=self.dtype, device=self.device)
        new_feat = diff_map.permute(1, 2, 0).reshape(-1, self.feat_dim)[selected]
        new_vis_feat = torch.rand_like(new_feat)
        # Old Gaussians
        old_xy = self.xy.detach().clone()
        old_scale = self.scale.detach().clone()
        old_rot = self.rot.detach().clone()
        old_feat = self.feat.detach().clone()
        old_vis_feat = self.vis_feat.detach().clone()
        # Update trainable parameters
        self.num_gaussians += add_num
        all_xy = torch.cat([old_xy, new_xy], dim=0)
        all_scale = torch.cat([old_scale, new_scale], dim=0)
        all_rot = torch.cat([old_rot, new_rot], dim=0)
        all_feat = torch.cat([old_feat, new_feat], dim=0)
        all_vis_feat = torch.cat([old_vis_feat, new_vis_feat], dim=0)
        self.xy = nn.Parameter(all_xy, requires_grad=True)
        self.scale = nn.Parameter(all_scale, requires_grad=True)
        self.rot = nn.Parameter(all_rot, requires_grad=True)
        self.feat = nn.Parameter(all_feat, requires_grad=True)
        self.vis_feat = nn.Parameter(all_vis_feat, requires_grad=False)
        # Plot Gaussians
        if plot_gaussians:
            path = f"{self.train_dir}/add-gaussians_step-{self.step:d}_num-{self.num_gaussians:d}_res-{self.img_h:d}x{self.img_w:d}"
            every_n = max(1, self.total_num_gaussians // 2000)
            size = (self.img_h * self.img_w) / 1e4
            visualize_added_gaussians(path, raw_images, old_xy, new_xy, self.input_channels, size=size, every_n=every_n,
                                      alpha=0.8, gamma=self.gamma, save_image_format=self.save_plot_format)
        # Update optimizer
        self.optimizer = torch.optim.Adam([{'params': self.xy, 'lr': self.pos_lr},
                                           {'params': self.scale, 'lr': self.scale_lr},
                                           {'params': self.rot, 'lr': self.rot_lr},
                                           {'params': self.feat, 'lr': self.feat_lr}])
        self.worklog.info(f"Step: {self.step:d} | Adding {add_num:d} Gaussians ({self.num_gaussians-add_num:d} -> {self.num_gaussians:d})")
        self.worklog.info("***********************************************")

    # ========== Patch-based Training Methods ==========

    def set_patch_target(self, patch_gt: torch.Tensor):
        """
        Set a patch as the training target (for patch-specific training).

        Args:
            patch_gt: Ground truth patch tensor (C, H, W)
        """
        self.gt_images = patch_gt.to(dtype=self.dtype, device=self.device)
        self.img_h, self.img_w = self.gt_images.shape[1:]
        self.tile_bounds = ((self.img_w + self.block_w - 1) // self.block_w,
                           (self.img_h + self.block_h - 1) // self.block_h, 1)
        self.num_pixels = self.img_h * self.img_w
        self.pixel_xy = get_grid(h=self.img_h, w=self.img_w).to(dtype=self.dtype, device=self.device).reshape(-1, 2)

    def init_gaussians_in_patch_bounds(
        self,
        num_gaussians: int,
        bounds: tuple = None,
        init_from_base: bool = False,
        base_xy: torch.Tensor = None,
        base_scale: torch.Tensor = None,
        base_rot: torch.Tensor = None,
        base_feat: torch.Tensor = None
    ):
        """
        Initialize gaussians within patch bounds, optionally from base model.

        Args:
            num_gaussians: Number of gaussians to initialize
            bounds: (x_min, y_min, x_max, y_max) in normalized coordinates [0, 1]
            init_from_base: Whether to initialize from base model gaussians
            base_xy: Base model xy positions
            base_scale: Base model scales
            base_rot: Base model rotations
            base_feat: Base model features
        """
        if init_from_base and base_xy is not None:
            # Extract gaussians from base model that fall within patch
            from utils.patch_utils import extract_patch_gaussians

            mask = extract_patch_gaussians(base_xy, bounds, margin=0.1)
            num_from_base = mask.sum().item()

            if num_from_base > 0:
                # Use base model gaussians that fall in patch
                self.xy = nn.Parameter(base_xy[mask].detach().clone(), requires_grad=True)
                self.scale = nn.Parameter(base_scale[mask].detach().clone(), requires_grad=True)
                self.rot = nn.Parameter(base_rot[mask].detach().clone(), requires_grad=True)
                self.feat = nn.Parameter(base_feat[mask].detach().clone(), requires_grad=True)
                self.vis_feat = nn.Parameter(torch.rand_like(self.feat), requires_grad=False)

                self.num_gaussians = num_from_base
                self.worklog.info(f"Initialized {num_from_base} gaussians from base model")

                # Add additional gaussians if needed
                if num_gaussians > num_from_base:
                    add_num = num_gaussians - num_from_base
                    self._add_random_gaussians_in_bounds(add_num, bounds)
            else:
                # No base gaussians in patch, initialize randomly
                self.num_gaussians = 0
                self._add_random_gaussians_in_bounds(num_gaussians, bounds)
        else:
            # Random initialization within bounds
            self.num_gaussians = 0
            self._add_random_gaussians_in_bounds(num_gaussians, bounds)

    def _add_random_gaussians_in_bounds(self, num_gaussians: int, bounds: tuple = None):
        """
        Add randomly initialized gaussians, optionally constrained to bounds.

        Args:
            num_gaussians: Number of gaussians to add
            bounds: Optional (x_min, y_min, x_max, y_max) in normalized coordinates
        """
        # Generate random positions
        if bounds is not None:
            x_min, y_min, x_max, y_max = bounds
            new_xy = torch.rand(num_gaussians, 2, dtype=self.dtype, device=self.device)
            new_xy[:, 0] = x_min + new_xy[:, 0] * (x_max - x_min)
            new_xy[:, 1] = y_min + new_xy[:, 1] * (y_max - y_min)
        else:
            new_xy = torch.rand(num_gaussians, 2, dtype=self.dtype, device=self.device)

        # Initialize other parameters
        new_scale = torch.ones(num_gaussians, 2, dtype=self.dtype, device=self.device)
        new_scale.fill_(self.init_scale if self.disable_inverse_scale else 1.0/self.init_scale)
        new_rot = torch.zeros(num_gaussians, 1, dtype=self.dtype, device=self.device)
        new_feat = torch.rand(num_gaussians, self.feat_dim, dtype=self.dtype, device=self.device)
        new_vis_feat = torch.rand_like(new_feat)

        if self.num_gaussians == 0:
            # First initialization
            self.xy = nn.Parameter(new_xy, requires_grad=True)
            self.scale = nn.Parameter(new_scale, requires_grad=True)
            self.rot = nn.Parameter(new_rot, requires_grad=True)
            self.feat = nn.Parameter(new_feat, requires_grad=True)
            self.vis_feat = nn.Parameter(new_vis_feat, requires_grad=False)
            self.num_gaussians = num_gaussians
        else:
            # Add to existing gaussians
            self.xy = nn.Parameter(torch.cat([self.xy.detach(), new_xy], dim=0), requires_grad=True)
            self.scale = nn.Parameter(torch.cat([self.scale.detach(), new_scale], dim=0), requires_grad=True)
            self.rot = nn.Parameter(torch.cat([self.rot.detach(), new_rot], dim=0), requires_grad=True)
            self.feat = nn.Parameter(torch.cat([self.feat.detach(), new_feat], dim=0), requires_grad=True)
            self.vis_feat = nn.Parameter(torch.cat([self.vis_feat.detach(), new_vis_feat], dim=0), requires_grad=False)
            self.num_gaussians += num_gaussians

        self.worklog.info(f"Added {num_gaussians} random gaussians (total: {self.num_gaussians})")

    def train_patch(
        self,
        patch_gt: torch.Tensor,
        num_gaussians: int,
        max_steps: int,
        bounds: tuple = None,
        base_model_state: dict = None,
        log_prefix: str = "patch"
    ) -> dict:
        """
        Train a single patch.

        Args:
            patch_gt: Ground truth patch tensor (C, H, W)
            num_gaussians: Number of gaussians to use
            max_steps: Number of training steps
            bounds: Patch bounds in normalized coordinates
            base_model_state: Optional base model state dict to initialize from
            log_prefix: Prefix for logging

        Returns:
            Dictionary with trained model state and metrics
        """
        # Set patch as target
        self.set_patch_target(patch_gt)

        # Initialize gaussians
        if base_model_state is not None:
            self.init_gaussians_in_patch_bounds(
                num_gaussians,
                bounds,
                init_from_base=True,
                base_xy=base_model_state['xy'],
                base_scale=base_model_state['scale'],
                base_rot=base_model_state['rot'],
                base_feat=base_model_state['feat']
            )
        else:
            self.init_gaussians_in_patch_bounds(num_gaussians, bounds)

        # Reinitialize optimizer for patch training
        self.optimizer = torch.optim.Adam([
            {'params': self.xy, 'lr': self.pos_lr},
            {'params': self.scale, 'lr': self.scale_lr},
            {'params': self.rot, 'lr': self.rot_lr},
            {'params': self.feat, 'lr': self.feat_lr}
        ])

        # Training loop
        self.worklog.info(f"Starting patch training: {log_prefix} | {num_gaussians} gaussians | {max_steps} steps")

        for step in range(max_steps):
            self.step = step

            # Forward pass
            images, _ = self.forward(self.img_h, self.img_w, self.tile_bounds)

            # Calculate loss
            self._get_total_loss(images)

            # Backward pass
            self.optimizer.zero_grad()
            self.total_loss.backward()
            self.optimizer.step()

            # Log periodically
            if step % 100 == 0:
                psnr, ssim = self._evaluate(log=False)
                self.worklog.info(f"{log_prefix} Step {step}/{max_steps} | Loss: {self.total_loss.item():.6f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

        # Final evaluation
        psnr, ssim = self._evaluate(log=False)
        self.worklog.info(f"{log_prefix} Training complete | Final PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

        # Return state and metrics
        return {
            'xy': self.xy.detach().clone(),
            'scale': self.scale.detach().clone(),
            'rot': self.rot.detach().clone(),
            'feat': self.feat.detach().clone(),
            'psnr': psnr,
            'ssim': ssim,
            'num_gaussians': self.num_gaussians
        }
