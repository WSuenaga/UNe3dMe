#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import torch

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DUST3R_DIR = os.path.join(PROJECT_ROOT, "models", "dust3r")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(DUST3R_DIR)
sys.path.append(os.path.join(DUST3R_DIR, "dust3r"))
sys.path.insert(0, os.path.join(DUST3R_DIR, "dust3r"))

from dust3r.demo import get_args_parser, get_reconstructed_scene, set_print_with_timestamp
from dust3r.model import AsymmetricCroCo3DStereo

torch.backends.cuda.matmul.allow_tf32 = True  # GPU が Ampere 以降かつ PyTorch 1.12 以降なら有効．


def parse_args():
    """
    CLI 引数を解釈する．

    Returns:
        argparse.Namespace: 解釈済みの引数．
    """
    parser = get_args_parser()

    parser.add_argument("--lang", type=str, default="jp", choices=["jp", "en"], help="Log language")
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
    parser.add_argument(
        "--scenegraph_type",
        type=str,
        default="complete",
        choices=["complete", "swin", "oneref"],
        help="Scene graph type for image pairing",
    )
    parser.add_argument("--winsize", type=int, default=1, help="Window size for 'swin' mode")
    parser.add_argument("--refid", type=int, default=0, help="Reference image ID for 'oneref' mode")

    return parser.parse_args()


def save_render_images(lang: str, outdir: str, imgs) -> str:
    """
    レンダー画像を保存する．

    Args:
        lang (str): ログ出力に使用する言語コード．
        outdir (str): 出力ディレクトリのパス．
        imgs: 保存対象の画像列．

    Returns:
        str: 画像保存先ディレクトリのパス．
    """
    # 保存先フォルダ作成
    render_dir = os.path.join(outdir, "render")
    os.makedirs(render_dir, exist_ok=True)

    # imgs の内容を順番に保存
    for idx, img in enumerate(imgs):
        save_path = os.path.join(render_dir, f"img_{idx:03d}.png")
        plt.imsave(save_path, img)

    print(
        msg(
            lang,
            f"{len(imgs)} 枚の画像を {render_dir} に保存しました．",
            f"Saved {len(imgs)} images to {render_dir}",
        )
    )
    return render_dir


def main() -> None:
    """
    DUST3R による三次元再構成を実行し，レンダー画像を保存する．
    """
    args = parse_args()
    set_print_with_timestamp()

    weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # 三次元再構築結果は outdir に生成される．
    scene, outfile, imgs = get_reconstructed_scene(
        args.outdir,
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
        args.refid,
    )

    save_render_images(args.lang, args.outdir, imgs)

    print(
        msg(
            args.lang,
            f"再構成が完了しました．出力ファイル: {outfile}",
            f"Reconstruction completed. Output file: {outfile}",
        )
    )


if __name__ == "__main__":
    main()