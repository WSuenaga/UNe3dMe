from __future__ import annotations

import argparse
import gc
import glob
import os
import sys

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

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(os.path.join(PROJECT_ROOT, "models", "vggt"))

from models.vggt.visual_util import predictions_to_glb
from models.vggt.vggt.models.vggt import VGGT
from models.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from models.vggt.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri


def run_vggt_reconstruction(
    lang,
    dataset,
    outdir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
    mode="crop",
    device=None,
):
    """
    VGGT による三次元再構成を実行し，GLB を出力する．

    Args:
        lang (str): ログ出力に使用する言語コード．
        dataset (str): 入力データセットのルートディレクトリ．
        outdir (str): 出力ディレクトリのパス．
        conf_thres (float, optional): 信頼度閾値．
        frame_filter (str, optional): 使用フレームのフィルタ条件．
        mask_black_bg (bool, optional): 黒背景を除外するかどうか．
        mask_white_bg (bool, optional): 白背景を除外するかどうか．
        show_cam (bool, optional): カメラ位置を表示するかどうか．
        mask_sky (bool, optional): 空領域を除外するかどうか．
        prediction_mode (str, optional): 可視化に使う予測モード．
        mode (str, optional): 画像前処理モード．
        device (str | None, optional): 使用デバイス．None のとき自動選択する．

    Returns:
        tuple[str, str]: 生成した GLB ファイルのパスとログ全文．
    """
    log_lines = []

    # デバイス設定
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and device == "cuda":
        raise ValueError(msg(lang, "CUDA が利用できません．", "CUDA is not available."))
    device = torch.device(device)

    os.makedirs(outdir, exist_ok=True)

    log_lines.append(msg(lang, "VGGT モデルを初期化します．", "Initializing VGGT model..."))
    model = VGGT()
    model_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
    model.eval().to(device)

    # 画像の読み込み
    image_dir = os.path.join(dataset, "images")
    image_names = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_names) == 0:
        raise ValueError(msg(lang, "画像が見つかりませんでした．", "No images found."))

    log_lines.append(
        msg(
            lang,
            f"{len(image_names)} 枚の画像を処理します．",
            f"Processing {len(image_names)} images...",
        )
    )
    images = load_and_preprocess_images(image_names, mode).to(device)

    # 推論
    log_lines.append(msg(lang, "推論を実行します．", "Running inference..."))
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    else:
        with torch.no_grad():
            predictions = model(images)

    # Pose Encoding 変換
    log_lines.append(
        msg(
            lang,
            "Pose encoding を外部・内部パラメータへ変換します．",
            "Converting pose encoding to extrinsic and intrinsic matrices...",
        )
    )
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 出力を NumPy に変換
    # すべての PyTorch テンソルを CPU 上の NumPy 配列に変換する．
    for key, value in list(predictions.items()):
        if isinstance(value, torch.Tensor):
            predictions[key] = value.cpu().numpy().squeeze(0)

    # 不要なデータを削除して軽量化する．
    predictions["pose_enc_list"] = None

    # Depth map から Point map を生成
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(
        depth_map,
        predictions["extrinsic"],
        predictions["intrinsic"],
    )
    predictions["world_points_from_depth"] = world_points

    # GLB ファイル出力
    glbfile = os.path.join(outdir, "scene.glb")
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=outdir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)
    log_lines.append(msg(lang, f"GLB 出力完了: {glbfile}", f"GLB exported: {glbfile}"))

    torch.cuda.empty_cache()
    gc.collect()

    return glbfile, "\n".join(log_lines)


def parse_args():
    """
    CLI 引数を解釈する．

    Returns:
        argparse.Namespace: 解釈済みの引数．
    """
    parser = argparse.ArgumentParser(description="VGGT Reconstruction Runner")
    parser.add_argument("--lang", default="jp", choices=["jp", "en"], help="Log language")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--conf-thres", type=float, default=3.0)
    parser.add_argument("--frame-filter", default="All")
    parser.add_argument("--mask-black-bg", action="store_true")
    parser.add_argument("--mask-white-bg", action="store_true")
    parser.add_argument("--show-cam", action="store_true")
    parser.add_argument("--mask-sky", action="store_true")
    parser.add_argument("--prediction-mode", default="Pointmap Regression")
    parser.add_argument("--mode", default="crop")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    """
    VGGT 再構成を実行して結果を表示する．
    """
    args = parse_args()

    glb_path, log_msg = run_vggt_reconstruction(
        lang=args.lang,
        dataset=args.image_dir,
        outdir=args.out_dir,
        conf_thres=args.conf_thres,
        frame_filter=args.frame_filter,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=args.show_cam,
        mask_sky=args.mask_sky,
        prediction_mode=args.prediction_mode,
        mode=args.mode,
        device=args.device,
    )

    print(log_msg)
    print(msg(args.lang, f"生成された GLB: {glb_path}", f"Generated GLB: {glb_path}"))


if __name__ == "__main__":
    main()