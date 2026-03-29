from __future__ import annotations

import argparse
import os
import sys

import torch
from huggingface_hub import hf_hub_download

def msg(lang, jp_msg, en_msg):
    """
    言語コードに応じて日本語または英語のメッセージを返す．

    Args:
        lang (str): 言語コード．`"jp"` のとき日本語，それ以外は英語を返す．
        jp_msg (str): 日本語メッセージ．
        en_msg (str): 英語メッセージ．

    Returns:
        str: 選択された言語のメッセージ．
    """
    return jp_msg if lang == "jp" else en_msg

# パス設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

SPLATT3R_DIR = os.path.join(PROJECT_ROOT, "models", "splatt3r")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(SPLATT3R_DIR)
sys.path.insert(0, SPLATT3R_DIR)

sys.path.append(os.path.join(SPLATT3R_DIR, "src", "mast3r_src"))
sys.path.append(os.path.join(SPLATT3R_DIR, "src", "mast3r_src", "dust3r"))
sys.path.append(os.path.join(SPLATT3R_DIR, "src", "pixelsplat_src"))

from demo import get_reconstructed_scene
import main as splatt3r_main


def parse_args():
    """
    CLI 引数を解釈する．

    Returns:
        argparse.Namespace: 解釈済みの引数．
    """
    parser = argparse.ArgumentParser(
        description="Splatt3R CLI Demo - run 3D Gaussian Splat reconstruction from one or two images."
    )

    parser.add_argument("--lang", type=str, default="jp", choices=["jp", "en"], help="Log language")
    parser.add_argument("--image1", type=str, required=True, help="Path to first input image")
    parser.add_argument("--image2", type=str, default=None, help="Optional second image")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory for .ply file")
    parser.add_argument("--image_size", type=int, default=512, help="Image resize size")
    parser.add_argument("--silent", action="store_true", help="Disable verbose logging")
    parser.add_argument("--ios_mode", action="store_true", help="Enable iOS-style single-image gallery mode")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="brandonsmart/splatt3r_v1.0",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="epoch=19-step=1200.ckpt",
        help="Checkpoint filename on Hugging Face",
    )

    return parser.parse_args()


def build_filelist(image1: str, image2: str | None) -> list[str]:
    """
    入力画像のリストを作成する．

    Args:
        image1 (str): 1 枚目の画像パス．
        image2 (str | None): 2 枚目の画像パス．

    Returns:
        list[str]: 再構成に使う画像パスの一覧．
    """
    return [image1] if image2 is None else [image1, image2]


def main_cli() -> None:
    """
    Splatt3R による 3D Gaussian Splat 再構成を実行する．
    """
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(
        msg(
            args.lang,
            f"モデルを読み込みます: {args.model_name}/{args.checkpoint}",
            f"Loading model: {args.model_name}/{args.checkpoint}",
        )
    )
    weights_path = hf_hub_download(repo_id=args.model_name, filename=args.checkpoint)

    model = splatt3r_main.MAST3RGaussians.load_from_checkpoint(weights_path, args.device)
    model.eval()

    filelist = build_filelist(args.image1, args.image2)
    print(
        msg(
            args.lang,
            f"{len(filelist)} 枚の画像で再構成を実行します．",
            f"Running reconstruction on {len(filelist)} image(s)...",
        )
    )

    plyfile = get_reconstructed_scene(
        outdir=args.outdir,
        model=model,
        device=args.device,
        silent=args.silent,
        image_size=args.image_size,
        ios_mode=args.ios_mode,
        filelist=filelist,
    )

    print(
        msg(
            args.lang,
            f"再構成が完了しました．出力先:\n{plyfile}",
            f"Reconstruction finished. Output saved to:\n{plyfile}",
        )
    )


if __name__ == "__main__":
    main_cli()