from __future__ import annotations

import argparse
import gc
import glob
import os
import sys

import matplotlib
import numpy as np
import torch
import trimesh

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

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "fast3r"))

from models.fast3r.fast3r.dust3r.inference_multiview import inference
from models.fast3r.fast3r.dust3r.utils.image import load_images
from models.fast3r.fast3r.models.fast3r import Fast3R
from models.fast3r.fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from models.fast3r.fast3r.viz.viser_visualizer import generate_ply_bytes, safe_color_conversion


def fast3r_output_to_glb(output_dict, camera_poses, out_dir, show_cam=True):
    """
    Fast3R の出力から GLB シーンを生成して保存する．

    Args:
        output_dict: Fast3R 推論結果を含む辞書．
        camera_poses: 推定されたカメラ姿勢列．
        out_dir: 出力ディレクトリのパス．
        show_cam: True のときカメラ位置をシーンへ追加する．

    Returns:
        str: 保存された GLB ファイルのパス．
    """
    preds = output_dict["preds"]
    views = output_dict["views"]
    S = len(preds)

    # 3D points and RGB colors
    vertices_3d = np.concatenate(
        [
            preds[i]["pts3d_in_other_view"].cpu().numpy().reshape(-1, 3)
            for i in range(S)
        ],
        axis=0,
    )
    colors_rgb = np.concatenate(
        [
            safe_color_conversion(views[i]["img"].cpu().numpy())
            .squeeze()
            .transpose(1, 2, 0)
            .reshape(-1, 3)
            for i in range(S)
        ],
        axis=0,
    )

    # Build 3D Scene
    scene_3d = trimesh.Scene()
    scene_3d.add_geometry(trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb))

    # Add cameras
    if show_cam:
        colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
        for i, pose in enumerate(camera_poses):
            cam_color = tuple((np.array(colormap(i / len(camera_poses))[:3]) * 255).astype(np.uint8))
            if isinstance(pose, torch.Tensor):
                pose_np = pose.cpu().numpy()
            else:
                pose_np = pose
            cam_to_world = np.linalg.inv(pose_np)
            cam_point = cam_to_world[:3, 3].reshape(1, 3)
            scene_3d.add_geometry(trimesh.PointCloud(cam_point, colors=np.array([cam_color])))

    # Export GLB
    os.makedirs(out_dir, exist_ok=True)
    glb_path = os.path.join(out_dir, "scene.glb")
    scene_3d.export(glb_path)

    return glb_path


def run_fast3r_reconstruction(
    lang,
    inpdir,
    outdir,
    image_size=512,
    dtype=torch.float32,
    niter_PnP=100,
    verbose=True,
    device=None,
):
    """
    Fast3R によるマルチビュー再構成を実行し，PLY と GLB を出力する．

    Args:
        lang (str): ログ出力に使用する言語コード．
        inpdir (str): 入力画像ディレクトリのパス．
        outdir (str): 出力ディレクトリのパス．
        image_size (int，optional): 入力画像のリサイズ解像度．
        dtype: 推論時の精度．
        niter_PnP (int，optional): PnP の反復回数．
        verbose (bool，optional): True のとき詳細ログを有効にする．
        device (str | None，optional): 使用デバイス．None のとき自動選択する．

    Returns:
        str: 実行ログ全文．
    """
    log_lines = []

    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and device == "cuda":
        raise ValueError(
            msg(
                lang,
                "CUDA が利用できません．",
                "CUDA is not available.",
            )
        )
    device = torch.device(device)
    log_lines.append(
        msg(
            lang,
            f"出力ディレクトリ: {outdir}",
            f"Output directory: {outdir}",
        )
    )

    # Load model
    log_lines.append(msg(lang, "Fast3R モデルを読み込みます．", "Loading Fast3R model..."))
    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
    model = model.to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval()
    lit_module.eval()

    # Load images
    image_names = sorted(glob.glob(os.path.join(inpdir, "*")))
    if len(image_names) == 0:
        raise ValueError(
            msg(
                lang,
                "入力ディレクトリに画像が見つかりませんでした．",
                "No images found in the input directory.",
            )
        )
    log_lines.append(
        msg(
            lang,
            f"{len(image_names)} 枚の画像を読み込みます．",
            f"Loading {len(image_names)} images...",
        )
    )
    images = load_images(image_names, size=image_size, verbose=verbose)

    # Run inference
    log_lines.append(msg(lang, "推論を実行します．", "Running inference..."))
    output_dict, profiling_info = inference(
        images,
        model,
        device,
        dtype=dtype,
        verbose=verbose,
        profiling=True,
    )

    # Estimate camera poses
    log_lines.append(msg(lang, "カメラ姿勢を推定します．", "Estimating camera poses..."))
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict["preds"],
        niter_PnP=niter_PnP,
        focal_length_estimation_method="first_view_from_global_head",
    )
    camera_poses = poses_c2w_batch[0]
    for view_idx, pose in enumerate(camera_poses):
        log_lines.append(
            msg(
                lang,
                f"カメラ姿勢 [{view_idx}]: {pose.shape}",
                f"Camera Pose [{view_idx}]: {pose.shape}",
            )
        )

    # Extract 3D points and RGB colors
    log_lines.append(msg(lang, "3D 点群を抽出します．", "Extracting 3D point cloud..."))
    all_points = []
    all_colors = []
    for view_idx, pred in enumerate(output_dict["preds"]):
        pts3d = pred["pts3d_in_other_view"].cpu().numpy().reshape(-1, 3)
        img_rgb = output_dict["views"][view_idx]["img"].cpu().numpy()
        img_rgb = safe_color_conversion(img_rgb)
        img_rgb = img_rgb.squeeze().transpose(1, 2, 0).reshape(-1, 3)
        all_points.append(pts3d)
        all_colors.append(img_rgb)
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Export PLY
    log_lines.append(msg(lang, "PLY ファイルを生成します．", "Generating PLY file..."))
    os.makedirs(outdir, exist_ok=True)
    ply_bytes = generate_ply_bytes(all_points, all_colors)
    ply_path = os.path.join(outdir, "scene.ply")
    with open(ply_path, "wb") as f:
        f.write(ply_bytes)
    log_lines.append(
        msg(
            lang,
            f"PLY ファイルを保存しました: {ply_path}",
            f"PLY file saved: {ply_path}",
        )
    )

    # Export GLB
    log_lines.append(msg(lang, "GLB ファイルを生成します．", "Generating GLB file..."))
    glb_path = fast3r_output_to_glb(output_dict, camera_poses, outdir)
    log_lines.append(
        msg(
            lang,
            f"GLB ファイルを保存しました: {glb_path}",
            f"GLB file saved: {glb_path}",
        )
    )

    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return "\n".join(log_lines)


def parse_args():
    """
    CLI 引数を解釈する．

    Returns:
        argparse.Namespace: 解釈済みの引数．
    """
    parser = argparse.ArgumentParser(description="Fast3R Multi-View Reconstruction Runner")
    parser.add_argument("--lang", default="jp", choices=["jp", "en"], help="Log language")
    parser.add_argument("--inpdir", required=True, help="Path to the input image directory")
    parser.add_argument("--outdir", required=True, help="Path to the output directory")
    parser.add_argument("--image_size", type=int, default=512, help="Resize resolution")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"], help="Inference precision")
    parser.add_argument("--niter_pnp", type=int, default=100, help="Number of PnP iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    """
    Fast3R による再構成を実行してログを出力する．
    """
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    log = run_fast3r_reconstruction(
        lang=args.lang,
        inpdir=args.inpdir,
        outdir=args.outdir,
        image_size=args.image_size,
        dtype=dtype,
        niter_PnP=args.niter_pnp,
        verbose=args.verbose,
        device=args.device,
    )

    print(log)


if __name__ == "__main__":
    main()