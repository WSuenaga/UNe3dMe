#!/usr/bin/env python3
import os
import torch

from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.demo import get_args_parser, set_print_with_timestamp, get_reconstructed_scene

import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()
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