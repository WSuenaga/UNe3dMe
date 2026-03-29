# This script is based on or incorporates code from Depth-Anything-3
# (Copyright (c) 2025 ByteDance Ltd.), used under the Apache License 2.0.

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DA3_SRC_PATH = os.path.abspath(
    os.path.join(PROJECT_ROOT, "models", "Depth-Anything-3", "src")
)
if DA3_SRC_PATH not in sys.path:
    sys.path.insert(0, DA3_SRC_PATH)

import numpy as np
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

try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.export.glb import export_to_glb
    from depth_anything_3.utils.export.gs import export_to_gs_video
    from depth_anything_3.utils.memory import cleanup_cuda_memory
except ImportError as e:
    print(
        msg(
            "en",
            f"DA3 モジュールを import できませんでした．パスを確認してください: {DA3_SRC_PATH}",
            f"Could not import DA3 modules. Make sure the path is correct: {DA3_SRC_PATH}",
        )
    )
    print(e)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """
    CLI 引数を解釈する．

    Returns:
        argparse.Namespace: 解釈済みの引数．
    """
    parser = argparse.ArgumentParser(description="Depth Anything 3 CLI")

    parser.add_argument("--lang", default="jp", choices=["jp", "en"], help="Log language")
    parser.add_argument("--input_dir", required=True, help="Input directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")

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


def collect_image_paths(image_dir: str) -> list[str]:
    """
    指定ディレクトリから画像パスを収集する．

    Args:
        image_dir (str): 画像を探索するディレクトリ．

    Returns:
        list[str]: 見つかった画像パスの一覧．
    """
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    return sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]
    )


def main() -> None:
    """
    Depth Anything 3 を用いて推論を実行し，結果を保存する．
    """
    args = parse_args()

    input_root = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_dir)
    model_dir = "depth-anything/da3-giant"

    os.makedirs(output_root, exist_ok=True)

    image_dir = os.path.join(input_root, "images")
    if not os.path.exists(image_dir):
        image_dir = input_root

    image_paths = collect_image_paths(image_dir)
    if not image_paths:
        raise ValueError(
            msg(
                args.lang,
                f"{image_dir} に画像が見つかりませんでした．",
                f"No images found in {image_dir}",
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        msg(
            args.lang,
            f"モデルを読み込みます: {model_dir}",
            f"Loading model from: {model_dir}",
        )
    )

    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    print(msg(args.lang, "推論を実行します．", "Running inference..."))
    with torch.no_grad():
        prediction = model.inference(
            image_paths,
            export_dir=None,
            process_res_method=args.process_res_method,
            infer_gs=args.infer_gs,
            ref_view_strategy=args.ref_view_strategy,
        )

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
        mode_mapping = {
            "extend": "extend",
            "smooth": "interpolate_smooth",
        }
        export_to_gs_video(
            prediction,
            export_dir=output_root,
            chunk_size=4,
            trj_mode=mode_mapping.get(args.gs_trj_mode, "extend"),
            enable_tqdm=True,
            vis_depth="hcat",
            video_quality=args.gs_video_quality,
            output_name="gs",
        )

    save_path = os.path.join(output_root, "predictions.npz")
    save_dict = {
        "depths": np.round(prediction.depth, 6) if prediction.depth is not None else None,
        "conf": np.round(prediction.conf, 2) if prediction.conf is not None else None,
        "extrinsics": prediction.extrinsics,
        "intrinsics": prediction.intrinsics,
    }
    save_dict = {k: v for k, v in save_dict.items() if v is not None}
    np.savez_compressed(save_path, **save_dict)

    cleanup_cuda_memory()
    print(
        msg(
            args.lang,
            f"推論が完了しました．出力先: {output_root}",
            f"Inference completed. Saved to: {output_root}",
        )
    )


if __name__ == "__main__":
    main()