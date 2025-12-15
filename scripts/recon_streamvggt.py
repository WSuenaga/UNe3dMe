import torch
import numpy as np
import glob
import gc
import argparse

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # gradio
STREAMVGGT_DIR = os.path.join(PROJECT_ROOT, "models", "StreamVGGT")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
sys.path.append(STREAMVGGT_DIR)
sys.path.append(os.path.join(STREAMVGGT_DIR, "src"))

from visual_util import predictions_to_glb
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import unproject_depth_map_to_point_map

# チェックポイントのパス
local_ckpt_path = os.path.join(STREAMVGGT_DIR, "ckpt", "checkpoints.pth")

# チェックポイントが見つかればモデルに設定する．見つからなければHugging Faceからダウンロードする．
if os.path.exists(local_ckpt_path):
    print(f"Loading local checkpoint from {local_ckpt_path}")
    model = StreamVGGT()
    ckpt = torch.load(local_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    del ckpt
else:
    print("Local checkpoint not found, downloading from Hugging Face...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="lch01/StreamVGGT",
        filename="checkpoints.pth",
        revision="main",
        force_download=True
    )
    model = StreamVGGT()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval() 
    del ckpt

def run_model(target_dir, model):
    logs = []  # ← ログ保存用

    # print と logs の両方に書き込む関数
    def log(msg):
        print(msg)
        logs.append(msg)

    log(f"Processing images from {target_dir}")

    # デバイスの設定．
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    
    log("Initializing and loading StreamVGGT model...")

    # モデルをデバイスに設定．
    model = model.to(device)
    model.eval()

    # 画像ファイル一覧取得
    # target_dir/images/以下の画像をすべて取得してソート
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    log(f"Found {len(image_names)} images")
    # ファイルが無い場合はエラー
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    # 画像を前処理してテンソルに変換
    images = load_and_preprocess_images(image_names).to(device)
    log(f"Preprocessed images shape: {images.shape}")

    # predictions辞書の初期化とimages保存
    predictions = {}
    # 前処理済み画像を格納しておく
    predictions["images"] = images  # (S, 3, H, W)
    log(f"Images shape: {images.shape}")

    # framesリストの構築（モデル入力フォーマットへ変換）
    frames = []
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)
        frame = {
            "img": image
        }
        frames.append(frame)

    # 推論前のログ & dtype選択
    log("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # 推論の実行（混合精度）
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            output = model.inference(frames)

    # 推論出力の抽出（フレームごと）
    all_pts3d = []
    all_conf = []
    all_depth = []
    all_depth_conf = []
    all_camera_pose = []
    
    for res in output.ress:
        all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
        all_conf.append(res['conf'].squeeze(0))
        all_depth.append(res['depth'].squeeze(0))
        all_depth_conf.append(res['depth_conf'].squeeze(0))
        all_camera_pose.append(res['camera_pose'].squeeze(0))

    # リストをテンソルスタックしてpredictionsに格納
    predictions["world_points"] = torch.stack(all_pts3d, dim=0)  # (S, H, W, 3)
    predictions["world_points_conf"] = torch.stack(all_conf, dim=0)  # (S, H, W)
    predictions["depth"] = torch.stack(all_depth, dim=0)  # (S, H, W, 1)
    predictions["depth_conf"] = torch.stack(all_depth_conf, dim=0)  # (S, H, W)
    predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0)  # (S, 9)

    # 中間ログ（shape情報）
    log(f"World points shape: {predictions['world_points'].shape}")
    log(f"World points confidence shape: {predictions['world_points_conf'].shape}")
    log(f"Depth map shape: {predictions['depth'].shape}")
    log(f"Depth confidence shape: {predictions['depth_conf'].shape}")
    log(f"Pose encoding shape: {predictions['pose_enc'].shape}")
    log(f"Images shape: {images.shape}")

    # pose encoding → extrinsic/intrinsic に変換
    log("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"],
        images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic.squeeze(0)  # (S, 3, 4)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None  # (S, 3, 3) or None
    log(f"Extrinsic shape: {predictions['extrinsic'].shape}")
    log(f"Intrinsic shape: {predictions['intrinsic'].shape}")

    # TorchテンソルをNumPyに変換
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()  # remove batch dimension if needed

    # depth map からワールド座標 point を生成
    log("Computing world points from depth map...")
    #depth_map = predictions["depth"]  # (S, H, W, 1)
    #world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    #predictions["world_points_from_depth"] = world_points
    predictions["world_points_from_depth"] = predictions["world_points"]

    # Clean up
    torch.cuda.empty_cache()

    return predictions, logs

def reconstruction(
    input_dir,
    output_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    logs = [] 

    # メモリクリーン
    gc.collect()
    torch.cuda.empty_cache()

    # モデル実行
    logs.append("Running run_model...")
    print("Running run_model...")
    with torch.no_grad():
        predictions, run_logs = run_model(input_dir, model)
        logs.extend(run_logs)

    # 推論結果を.npzで保存  
    logs.append("Saving predictions.npz ...")
    prediction_save_path = os.path.join(output_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    if frame_filter is None:
        frame_filter = "All"

    # 出力のGLBファイル名
    glbfile = os.path.join(output_dir,"scene.glb",)

    # GLBの生成
    logs.append("Generating GLB file...")
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

    logs.append(f"GLB exported: {glbfile}")

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    return glbfile, "\n".join(logs)

def main():
    parser = argparse.ArgumentParser(description="Run StreamVGGT 3D reconstruction without Gradio UI")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing images/ folder")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing images/ folder")
    parser.add_argument("--conf_thres", type=float, default=3.0,
                        help="Confidence threshold")
    parser.add_argument("--frame_filter", type=str, default="All",
                        help="Frame filter (e.g., 'All' or '0:000001.png')")
    parser.add_argument("--mask_black_bg", action="store_true", help="Enable black background masking")
    parser.add_argument("--mask_white_bg", action="store_true", help="Enable white background masking")
    parser.add_argument("--show_cam", action="store_true", help="Show camera positions")
    parser.add_argument("--mask_sky", action="store_true", help="Filter out sky")
    parser.add_argument("--prediction_mode", type=str,
                        default="Pointmap Regression",
                        choices=["Pointmap Regression", "Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Prediction mode")
    args = parser.parse_args()

    # ---------------- Run reconstruction ----------------
    print("Running reconstruction...")
    glb_path, log_msg = reconstruction(
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

    print("--------------------------------------------------------")
    print("StreamVGGT CLI Reconstruction Finished")
    print("Log:", log_msg)
    if glb_path:
        print("Generated GLB:", glb_path)
    else:
        print("Failed. Check your target_dir.")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    main()