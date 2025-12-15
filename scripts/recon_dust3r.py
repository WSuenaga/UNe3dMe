#!/usr/bin/env python3
import torch

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # gradio
DUST3R_DIR = os.path.join(PROJECT_ROOT, "models", "dust3r")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(DUST3R_DIR)
sys.path.append(os.path.join(DUST3R_DIR, "dust3r"))
sys.path.insert(0, os.path.join(DUST3R_DIR, "dust3r"))

from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.demo import get_args_parser, set_print_with_timestamp, get_reconstructed_scene

import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()

    parser.add_argument("--outdir", type=str, required=True, help="Directory to save the output")
    parser.add_argument("--filelist", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"], help="Schedule type")
    parser.add_argument("--niter", type=int, default=300, help="Number of iterations for alignment")
    parser.add_argument("--min_conf_thr", type=float, default=3.0, help="Minimum confidence threshold")
    parser.add_argument("--as_pointcloud", action="store_true", help="Export as pointcloud instead of mesh")
    parser.add_argument("--mask_sky", action="store_true", help="Mask sky in the depth maps")
    parser.add_argument("--clean_depth", action="store_true", help="Clean up the depth maps")
    parser.add_argument("--transparent_cams", action="store_true", help="Make cameras transparent in output")
    parser.add_argument("--cam_size", type=float, default=0.05, help="Camera size in the scene")
    parser.add_argument("--scenegraph_type", type=str, default="complete", choices=["complete", "swin", "oneref"],
                        help="Scene graph type for image pairing")
    parser.add_argument("--winsize", type=int, default=1, help="Window size for 'swin' mode")
    parser.add_argument("--refid", type=int, default=0, help="Reference image ID for 'oneref' mode")
       
    args = parser.parse_args()
    set_print_with_timestamp()

    weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # 三次元再構築結果はoutdirに生成される.
    scene, outfile, imgs = get_reconstructed_scene(args.outdir, 
                                                   model, 
                                                   args.device, 
                                                   args.silent,
                                                   args.image_size,
                                                   args.filelist,
                                                   args.schedule,
                                                   args.niter,
                                                   args.min_conf_thr,
                                                   args.as_pointcloud,
                                                   args.mask_sky,
                                                   args.clean_depth,
                                                   args.transparent_cams,
                                                   args.cam_size,
                                                   args.scenegraph_type,
                                                   args.winsize,
                                                   args.refid)
    
    # 保存先フォルダ作成
    render_dir = os.path.join(args.outdir, "render")
    os.makedirs(render_dir, exist_ok=True)

    # imgsの内容を順番に保存
    for idx, img in enumerate(imgs):
        save_path = os.path.join(render_dir, f"img_{idx:03d}.png")
        plt.imsave(save_path, img)

    print(f"Saved {len(imgs)} images to {render_dir}")