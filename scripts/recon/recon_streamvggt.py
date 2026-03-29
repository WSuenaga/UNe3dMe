from __future__ import annotations

import argparse
import gc
import glob
import os
import sys

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

STREAMVGGT_DIR = os.path.join(PROJECT_ROOT, "models", "StreamVGGT")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(STREAMVGGT_DIR)
sys.path.append(os.path.join(STREAMVGGT_DIR, "src"))

from visual_util import predictions_to_glb
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

# チェックポイントのパス
LOCAL_CKPT_PATH = os.path.join(STREAMVGGT_DIR, "ckpt", "checkpoints.pth")


def load_model(lang: str):
    """
    StreamVGGT モデルを読み込む．

    ローカルにチェックポイントがあればそれを使い，無ければ Hugging Face から取得する．

    Args:
        lang (str): ログ出力に使用する言語コード．

    Returns:
        StreamVGGT: 読み込み済みのモデル．
    """
    if os.path.exists(LOCAL_CKPT_PATH):
        print(
            msg(
                lang,
                f"ローカルチェックポイントを読み込みます: {LOCAL_CKPT_PATH}",
                f"Loading local checkpoint from {LOCAL_CKPT_PATH}",
            )
        )
        model = StreamVGGT()
        ckpt = torch.load(LOCAL_CKPT_PATH, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        del ckpt
        return model

    print(
        msg(
            lang,
            "ローカルチェックポイントが見つからないため，Hugging Face からダウンロードします．",
            "Local checkpoint not found, downloading from Hugging Face...",
        )
    )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="lch01/StreamVGGT",
        filename="checkpoints.pth",
        revision="main",
        force_download=True,
    )
    model = StreamVGGT()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    del ckpt
    return model


def run_model(lang: str, target_dir: str, model):
    """
    StreamVGGT モデルで推論を実行し，予測結果を返す．

    Args:
        lang (str): ログ出力に使用する言語コード．
        target_dir (str): 入力画像ディレクトリを含むルートディレクトリ．
        model: 読み込み済みの StreamVGGT モデル．

    Returns:
        tuple[dict, list[str]]: 予測結果辞書とログ文字列リスト．
    """
    logs = []

    # print と logs の両方に書き込む関数
    def log(text: str) -> None:
        print(text)
        logs.append(text)

    log(
        msg(
            lang,
            f"画像を処理します: {target_dir}",
            f"Processing images from {target_dir}",
        )
    )

    # デバイスの設定．
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError(
            msg(
                lang,
                "CUDA が利用できません．環境を確認してください．",
                "CUDA is not available. Check your environment.",
            )
        )

    log(msg(lang, "StreamVGGT モデルを初期化します．", "Initializing and loading StreamVGGT model..."))

    # モデルをデバイスに設定．
    model = model.to(device)
    model.eval()

    # 画像ファイル一覧取得
    # target_dir/images/ 以下の画像をすべて取得してソートする．
    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    log(msg(lang, f"{len(image_names)} 枚の画像を検出しました．", f"Found {len(image_names)} images"))

    # ファイルが無い場合はエラー
    if len(image_names) == 0:
        raise ValueError(
            msg(
                lang,
                "画像が見つかりませんでした．入力内容を確認してください．",
                "No images found. Check your upload.",
            )
        )

    # 画像を前処理してテンソルに変換
    images = load_and_preprocess_images(image_names).to(device)
    log(msg(lang, f"前処理後の画像 shape: {images.shape}", f"Preprocessed images shape: {images.shape}"))

    # predictions 辞書の初期化と images 保存
    predictions = {}
    # 前処理済み画像を格納しておく
    predictions["images"] = images  # (S, 3, H, W)
    log(msg(lang, f"画像 shape: {images.shape}", f"Images shape: {images.shape}"))

    # frames リストの構築
    # モデル入力フォーマットへ変換する．
    frames = []
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)
        frame = {"img": image}
        frames.append(frame)

    # 推論前のログと dtype 選択
    log(msg(lang, "推論を実行します．", "Running inference..."))
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # 推論の実行
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            output = model.inference(frames)

    # 推論出力の抽出
    all_pts3d = []
    all_conf = []
    all_depth = []
    all_depth_conf = []
    all_camera_pose = []

    for res in output.ress:
        all_pts3d.append(res["pts3d_in_other_view"].squeeze(0))
        all_conf.append(res["conf"].squeeze(0))
        all_depth.append(res["depth"].squeeze(0))
        all_depth_conf.append(res["depth_conf"].squeeze(0))
        all_camera_pose.append(res["camera_pose"].squeeze(0))

    # リストをテンソルスタックして predictions に格納
    predictions["world_points"] = torch.stack(all_pts3d, dim=0)  # (S, H, W, 3)
    predictions["world_points_conf"] = torch.stack(all_conf, dim=0)  # (S, H, W)
    predictions["depth"] = torch.stack(all_depth, dim=0)  # (S, H, W, 1)
    predictions["depth_conf"] = torch.stack(all_depth_conf, dim=0)  # (S, H, W)
    predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0)  # (S, 9)

    # 中間ログ
    log(msg(lang, f"World points shape: {predictions['world_points'].shape}", f"World points shape: {predictions['world_points'].shape}"))
    log(msg(lang, f"World points confidence shape: {predictions['world_points_conf'].shape}", f"World points confidence shape: {predictions['world_points_conf'].shape}"))
    log(msg(lang, f"Depth map shape: {predictions['depth'].shape}", f"Depth map shape: {predictions['depth'].shape}"))
    log(msg(lang, f"Depth confidence shape: {predictions['depth_conf'].shape}", f"Depth confidence shape: {predictions['depth_conf'].shape}"))
    log(msg(lang, f"Pose encoding shape: {predictions['pose_enc'].shape}", f"Pose encoding shape: {predictions['pose_enc'].shape}"))
    log(msg(lang, f"Images shape: {images.shape}", f"Images shape: {images.shape}"))

    # pose encoding を extrinsic / intrinsic に変換
    log(
        msg(
            lang,
            "Pose encoding を外部・内部パラメータへ変換します．",
            "Converting pose encoding to extrinsic and intrinsic matrices...",
        )
    )
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"],
        images.shape[-2:],
    )
    predictions["extrinsic"] = extrinsic.squeeze(0)  # (S, 3, 4)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None  # (S, 3, 3) or None
    log(msg(lang, f"Extrinsic shape: {predictions['extrinsic'].shape}", f"Extrinsic shape: {predictions['extrinsic'].shape}"))
    log(msg(lang, f"Intrinsic shape: {predictions['intrinsic'].shape}", f"Intrinsic shape: {predictions['intrinsic'].shape}"))

    # Torch テンソルを NumPy に変換
    for key, value in list(predictions.items()):
        if isinstance(value, torch.Tensor):
            predictions[key] = value.cpu().numpy()

    # depth map からワールド座標 point を生成
    log(msg(lang, "Depth map からワールド座標を計算します．", "Computing world points from depth map..."))
    predictions["world_points_from_depth"] = predictions["world_points"]

    # Clean up
    torch.cuda.empty_cache()

    return predictions, logs


def reconstruction(
    lang: str,
    model,
    input_dir: str,
    output_dir: str,
    conf_thres: float = 3.0,
    frame_filter: str = "All",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    prediction_mode: str = "Pointmap Regression",
):
    """
    StreamVGGT による再構成を実行し，GLB と予測結果を保存する．

    Args:
        lang (str): ログ出力に使用する言語コード．
        model: 読み込み済みの StreamVGGT モデル．
        input_dir (str): 入力ディレクトリ．
        output_dir (str): 出力ディレクトリ．
        conf_thres (float, optional): 信頼度閾値．
        frame_filter (str, optional): フレームフィルタ．
        mask_black_bg (bool, optional): 黒背景マスクを有効にするかどうか．
        mask_white_bg (bool, optional): 白背景マスクを有効にするかどうか．
        show_cam (bool, optional): カメラ位置を表示するかどうか．
        mask_sky (bool, optional): 空領域を除外するかどうか．
        prediction_mode (str, optional): 予測モード．

    Returns:
        tuple[str, str]: 生成した GLB パスとログ全文．
    """
    logs = []

    os.makedirs(output_dir, exist_ok=True)

    # メモリクリーン
    gc.collect()
    torch.cuda.empty_cache()

    # モデル実行
    logs.append(msg(lang, "run_model を実行します．", "Running run_model..."))
    print(msg(lang, "run_model を実行します．", "Running run_model..."))
    with torch.no_grad():
        predictions, run_logs = run_model(lang, input_dir, model)
        logs.extend(run_logs)

    # 推論結果を .npz で保存
    logs.append(msg(lang, "predictions.npz を保存します．", "Saving predictions.npz ..."))
    prediction_save_path = os.path.join(output_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    if frame_filter is None:
        frame_filter = "All"

    # 出力の GLB ファイル名
    glbfile = os.path.join(output_dir, "scene.glb")

    # GLB の生成
    logs.append(msg(lang, "GLB ファイルを生成します．", "Generating GLB file..."))
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=input_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    logs.append(msg(lang, f"GLB を出力しました: {glbfile}", f"GLB exported: {glbfile}"))

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    return glbfile, "\n".join(logs)


def parse_args():
    """
    CLI 引数を解釈する．

    Returns:
        argparse.Namespace: 解釈済みの引数．
    """
    parser = argparse.ArgumentParser(description="Run StreamVGGT 3D reconstruction without Gradio UI")
    parser.add_argument("--lang", type=str, default="jp", choices=["jp", "en"], help="Log language")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images/ folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--conf_thres", type=float, default=3.0, help="Confidence threshold")
    parser.add_argument("--frame_filter", type=str, default="All", help="Frame filter")
    parser.add_argument("--mask_black_bg", action="store_true", help="Enable black background masking")
    parser.add_argument("--mask_white_bg", action="store_true", help="Enable white background masking")
    parser.add_argument("--show_cam", action="store_true", help="Show camera positions")
    parser.add_argument("--mask_sky", action="store_true", help="Filter out sky")
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="Pointmap Regression",
        choices=["Pointmap Regression", "Depthmap and Camera Branch", "Pointmap Branch"],
        help="Prediction mode",
    )
    return parser.parse_args()


def main() -> None:
    """
    StreamVGGT の CLI 再構成処理を実行する．
    """
    args = parse_args()
    model = load_model(args.lang)

    # Run reconstruction
    print(msg(args.lang, "再構成を実行します．", "Running reconstruction..."))
    glb_path, log_msg = reconstruction(
        lang=args.lang,
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        conf_thres=args.conf_thres,
        frame_filter=args.frame_filter,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=args.show_cam,
        mask_sky=args.mask_sky,
        prediction_mode=args.prediction_mode,
    )

    print("-" * 56)
    print(msg(args.lang, "StreamVGGT CLI 再構成が完了しました．", "StreamVGGT CLI Reconstruction Finished"))
    print(msg(args.lang, f"ログ:\n{log_msg}", f"Log:\n{log_msg}"))
    if glb_path:
        print(msg(args.lang, f"生成された GLB: {glb_path}", f"Generated GLB: {glb_path}"))
    else:
        print(msg(args.lang, "失敗しました．入力ディレクトリを確認してください．", "Failed. Check your input directory."))
    print("-" * 56)


if __name__ == "__main__":
    main()