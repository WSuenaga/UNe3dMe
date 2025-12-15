import os
import sys
import argparse
import torch
from huggingface_hub import hf_hub_download

# --- パス設定 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # gradio
SPLATT3R_DIR = os.path.join(PROJECT_ROOT, "models", "splatt3r")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(SPLATT3R_DIR)
sys.path.insert(0, SPLATT3R_DIR)

sys.path.append(os.path.join(SPLATT3R_DIR, "src", "mast3r_src"))
sys.path.append(os.path.join(SPLATT3R_DIR, "src", "mast3r_src", "dust3r"))
sys.path.append(os.path.join(SPLATT3R_DIR, "src", "pixelsplat_src"))

from demo import get_reconstructed_scene

import main


def main_cli():
    parser = argparse.ArgumentParser(
        description="Splatt3R CLI Demo — run 3D Gaussian Splat reconstruction from one or two images."
    )

    parser.add_argument("--image1", type=str, required=True, help="Path to first input image")
    parser.add_argument("--image2", type=str, default=None, help="Optional second image")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory for .ply file")
    parser.add_argument("--image_size", type=int, default=512, help="Image resize size (default: 512)")
    parser.add_argument("--silent", action="store_true", help="Disable verbose logging")
    parser.add_argument("--ios_mode", action="store_true", help="Enable iOS-style single-image gallery mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run model on")

    parser.add_argument("--model_name", type=str, default="brandonsmart/splatt3r_v1.0", help="Hugging Face model name")
    parser.add_argument("--checkpoint", type=str, default="epoch=19-step=1200.ckpt", help="Checkpoint filename on Hugging Face")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading model: {args.model_name}/{args.checkpoint}")
    weights_path = hf_hub_download(repo_id=args.model_name, filename=args.checkpoint)

    model = main.MAST3RGaussians.load_from_checkpoint(weights_path, args.device)
    model.eval()

    filelist = [args.image1] if args.image2 is None else [args.image1, args.image2]
    print(f"Running reconstruction on {len(filelist)} image(s)...")

    plyfile = get_reconstructed_scene(
        outdir=args.outdir,
        model=model,
        device=args.device,
        silent=args.silent,
        image_size=args.image_size,
        ios_mode=args.ios_mode,
        filelist=filelist
    )

    print(f"Reconstruction finished. Output saved to:\n{plyfile}")


if __name__ == "__main__":
    main_cli()