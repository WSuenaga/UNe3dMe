import os
import argparse
import glob
import torch
import gc

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # gradio
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(os.path.join(PROJECT_ROOT, "models", "vggt"))

from models.vggt.visual_util import predictions_to_glb
from models.vggt.vggt.models.vggt import VGGT
from models.vggt.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from models.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

def run_vggt_reconstruction(
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
    log_lines = []

    # --- デバイス設定 ---
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and device == "cuda":
        raise ValueError("CUDAが利用できません。")

    log_lines.append("VGGTモデルを初期化中...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval().to(device)

    # --- 画像の読み込み ---
    image_dir = os.path.join(dataset, "images")
    image_names = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_names) == 0:
        raise ValueError("画像が見つかりませんでした。")

    log_lines.append(f"{len(image_names)}枚の画像を処理中...")
    images = load_and_preprocess_images(image_names, mode).to(device)

    # --- 推論 ---
    log_lines.append("推論を実行中...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

    # --- Pose Encoding変換 ---
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 出力を NumPy に変換
    # すべてのPyTorchテンソルをCPU上のNumPy配列に変換
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    # 不要なデータを削除して軽量化
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Depth map から Point map を生成
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # GLBファイル出力
    # GLBファイル名
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
    log_lines.append(f"GLB出力完了: {glbfile}")

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT Reconstruction Runner")
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
    args = parser.parse_args()

    run_vggt_reconstruction(
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
