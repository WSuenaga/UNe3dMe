# This script is based on or incorporates code from Depth-Anything-3 
# (Copyright (c) 2025 ByteDance Ltd.), used under the Apache License 2.0.

import argparse
import sys
import os
import numpy as np
import torch

# ==========================================================
# Add DA3 src to path
# ==========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_ROOT = os.path.dirname(CURRENT_DIR)

DA3_SRC_PATH = os.path.abspath(os.path.join(DEMO_ROOT, "models", "Depth-Anything-3", "src"))
sys.path.append(DA3_SRC_PATH)

# ==========================================================
# Import public APIs
# ==========================================================
try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.export.glb import export_to_glb
    from depth_anything_3.utils.export.gs import export_to_gs_video
    from depth_anything_3.utils.memory import cleanup_cuda_memory
except ImportError as e:
    print(f"Error: Could not import DA3 modules. Make sure the path is correct: {DA3_SRC_PATH}")
    print(e)
    sys.exit(1)

# ==========================================================
# Argument parser
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Depth Anything 3 CLI (os.path version)")

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--process_res_method", default="upper_bound_resize")
    parser.add_argument("--save_percentage", type=float, default=30.0)
    parser.add_argument("--num_max_points", type=int, default=1_000_000)

    parser.add_argument("--filter_black_bg", action="store_true")
    parser.add_argument("--filter_white_bg", action="store_true")

    parser.add_argument("--infer_gs", action="store_true")
    parser.add_argument("--gs_trj_mode", default="extend")
    parser.add_argument("--gs_video_quality", default="high")

    parser.add_argument("--ref_view_strategy", default="saddle_balanced")

    return parser.parse_args()

# ==========================================================
# Main
# ==========================================================
def main():
    args = parse_args()

    input_root = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_dir)
    model_dir = "depth-anything/da3-giant"

    os.makedirs(output_root, exist_ok=True)

    # imagesフォルダ探索
    image_dir = os.path.join(input_root, "images")
    if not os.path.exists(image_dir):
        image_dir = input_root

    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    ])

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_dir}")

    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        prediction = model.inference(
            image_paths,
            export_dir=None,
            process_res_method=args.process_res_method,
            infer_gs=args.infer_gs,
            ref_view_strategy=args.ref_view_strategy,
        )

    # Export GLB
    export_to_glb(
        prediction,
        filter_black_bg=args.filter_black_bg,
        filter_white_bg=args.filter_white_bg,
        export_dir=output_root,
        conf_thresh_percentile=args.save_percentage,
        num_max_points=args.num_max_points,
        show_cameras=True,
    )

    if args.infer_gs:
        mode_mapping = {"extend": "extend", "smooth": "interpolate_smooth"}
        export_to_gs_video(
            prediction,
            export_dir=output_root,
            chunk_size=4,
            trj_mode=mode_mapping.get(args.gs_trj_mode, "extend"),
            enable_tqdm=True,
            vis_depth="hcat",
            video_quality=args.gs_video_quality,
            output_name="gs"
        )

    # Save cache
    save_path = os.path.join(output_root, "predictions.npz")

    save_dict = {
        "depths": np.round(prediction.depth, 6) if prediction.depth is not None else None,
        "conf": np.round(prediction.conf, 2) if prediction.conf is not None else None,
        "extrinsics": prediction.extrinsics,
        "intrinsics": prediction.intrinsics
    }

    save_dict = {k: v for k, v in save_dict.items() if v is not None}
    np.savez_compressed(save_path, **save_dict)

    cleanup_cuda_memory()
    print(f"Inference completed. Saved to: {output_root}")

if __name__ == "__main__":
    main()
