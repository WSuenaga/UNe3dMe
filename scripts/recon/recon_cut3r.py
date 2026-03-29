from __future__ import annotations

import os
import sys
import argparse
import gc

import imageio.v2 as iio
import matplotlib
import numpy as np
import torch
import trimesh

# --- パス設定．CUT3R を import ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "models", "CUT3R"))
sys.path.append(os.path.join(PROJECT_ROOT, "models", "CUT3R", "src"))

from models.CUT3R import demo
from models.CUT3R.add_ckpt_path import add_path_to_dust3r
from models.CUT3R.src.dust3r.inference import inference
from models.CUT3R.src.dust3r.model import ARCroco3DStereo
from models.CUT3R.src.dust3r.utils.camera import pose_encoding_to_camera

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


def run_inference(
    lang,
    inpdir,
    model_path,
    outdir,
    image_size=512,
    vis_threshold=1.5,
    show_cam=True,
    device="cuda",
):
    """
    CUT3R によるマルチビュー再構成を実行する．

    入力画像列を読み込み，3D 点群を推定し，PLY と GLB を出力する．
    あわせて，各視点の深度マップとカラー画像も保存する．

    Args:
        lang (str): ログ出力に使用する言語コード．
            `"jp"` のとき日本語，それ以外のとき英語を用いる想定である．
        inpdir (str): 入力画像ディレクトリのパス．
        model_path (str): 学習済み CUT3R モデルのパス．
        outdir (str): 出力先ディレクトリのパス．
        image_size (int, optional): 入力画像のリサイズサイズ．デフォルトは 512．
        vis_threshold (float, optional): 点群抽出に用いる信頼度閾値．デフォルトは 1.5．
        show_cam (bool, optional): True のとき GLB 内にカメラ位置を可視化する．
        device (str, optional): 推論デバイス．`"cuda"` または `"cpu"` を指定する．

    Returns:
        dict | None: 成功時は出力ファイル情報を含む辞書を返す．
            入力画像が見つからない場合は None を返す．
    """
    # デバイス設定
    if device == "cuda" and not torch.cuda.is_available():
        print(
            msg(
                lang,
                "⚠️ CUDA が利用できないため，CPU に切り替えます．",
                "⚠️ CUDA not available. Switching to CPU.",
            )
        )
        device = "cpu"

    os.makedirs(outdir, exist_ok=True)

    # モデル依存モジュール読み込み用に dust3r のパスを追加する．
    add_path_to_dust3r(model_path)

    # 入力画像の準備
    img_paths, _ = demo.parse_seq_path(inpdir)
    if not img_paths:
        print(
            msg(
                lang,
                f"❌ {inpdir} に画像が見つかりませんでした．パスを確認してください．",
                f"❌ No images found in {inpdir}. Please verify the path.",
            )
        )
        return None

    print(
        msg(
            lang,
            f"📸 {inpdir} で {len(img_paths)} 枚の画像を検出しました．",
            f"📸 Found {len(img_paths)} images in {inpdir}.",
        )
    )
    img_mask = [True] * len(img_paths)

    print(msg(lang, "🧩 入力ビューを準備しています．", "🧩 Preparing input views..."))
    views = demo.prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=image_size,
        revisit=1,
        update=True,
    )

    # モデルをロード
    print(
        msg(
            lang,
            f"🧠 モデルを読み込んでいます: {model_path}",
            f"🧠 Loading model from: {model_path}",
        )
    )
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.eval()

    # 推論実行
    print(msg(lang, "🚀 推論を実行しています．", "🚀 Running inference..."))
    with torch.no_grad():
        outputs, _ = inference(views, model, device)

    preds = outputs["pred"]
    views_out = outputs["views"]

    # 点群の抽出
    print(msg(lang, "🔍 点群を抽出しています．", "🔍 Extracting point clouds..."))
    pts3d_list, color_list = [], []

    for i, pred in enumerate(preds):
        # pts3d: [1, H, W, 3] -> [H, W, 3]
        pts3d = pred["pts3d_in_self_view"].cpu().squeeze(0).numpy()
        # conf: [1, H, W] -> [H, W]
        conf = pred["conf_self"].cpu().squeeze(0).numpy()
        # img: [1, 3, H, W] -> [H, W, 3]
        img = views_out[i]["img"].cpu().squeeze(0).permute(1, 2, 0).numpy()
        img = (0.5 * (img + 1.0)).clip(0, 1)

        # 信頼度に基づいて有効点を抽出する．
        valid_mask = conf > vis_threshold

        # 点群と色を一次元化して取り出す．
        pts3d_masked = pts3d[valid_mask]
        colors_masked = img[valid_mask]

        pts3d_list.append(pts3d_masked)
        color_list.append(colors_masked)

    pts3d_all = np.concatenate(pts3d_list, axis=0)
    colors_all = np.concatenate(color_list, axis=0)
    print(
        msg(
            lang,
            f"✅ {pts3d_all.shape[0]} 個の有効な 3D 点を生成しました．",
            f"✅ Generated {pts3d_all.shape[0]} valid 3D points.",
        )
    )

    # 点群の PLY 保存
    ply_path = os.path.join(outdir, "scene.ply")
    cloud = trimesh.PointCloud(vertices=pts3d_all, colors=(colors_all * 255).astype(np.uint8))
    cloud.export(ply_path)
    print(
        msg(
            lang,
            f"💾 PLY ファイルを保存しました: {ply_path}",
            f"💾 PLY file saved: {ply_path}",
        )
    )

    # カメラ姿勢の推定
    print(msg(lang, "📷 カメラ姿勢を推定しています．", "📷 Estimating camera poses..."))
    cam_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu().numpy()[0]
        for pred in preds
    ]

    # GLB 出力
    print(msg(lang, "🌈 GLB を出力しています．", "🌈 Exporting GLB ..."))
    scene = trimesh.Scene()
    scene.add_geometry(cloud)

    if show_cam:
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
        for i, pose in enumerate(cam_poses):
            cam_color = (np.array(cmap(i / len(cam_poses))[:3]) * 255).astype(np.uint8)
            cam_center = np.linalg.inv(pose)[:3, 3]
            cam_pc = trimesh.PointCloud([cam_center], colors=[cam_color])
            scene.add_geometry(cam_pc)

    glb_path = os.path.join(outdir, "scene.glb")
    scene.export(glb_path)
    print(
        msg(
            lang,
            f"💾 GLB ファイルを保存しました: {glb_path}",
            f"💾 GLB file saved: {glb_path}",
        )
    )

    # 深度マップ・カラー画像の保存
    depth_dir = os.path.join(outdir, "depth")
    color_dir = os.path.join(outdir, "color")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)

    for i, pred in enumerate(preds):
        depth = pred["pts3d_in_self_view"][..., 2].cpu().numpy()
        color_img = (
            0.5 * (views_out[i]["img"].cpu().squeeze().permute(1, 2, 0).numpy() + 1.0) * 255
        ).astype(np.uint8)
        np.save(os.path.join(depth_dir, f"{i:03d}.npy"), depth)
        iio.imwrite(os.path.join(color_dir, f"{i:03d}.png"), color_img)

    # メモリ解放
    torch.cuda.empty_cache()
    gc.collect()
    print(msg(lang, "🎉 推論が正常に完了しました．", "🎉 Inference completed successfully."))

    return {
        "ply_path": ply_path,
        "glb_path": glb_path,
        "num_points": pts3d_all.shape[0],
        "output_dir": outdir,
    }


def main() -> None:
    """
    CLI 引数を解釈し，CUT3R 推論を実行する．
    """
    parser = argparse.ArgumentParser(description="CUT3R Multi-View Reconstruction Runner")
    parser.add_argument("--lang", default="jp", choices=["jp", "en"], help="Log language")
    parser.add_argument("--inpdir", required=True, help="Input image directory")
    parser.add_argument("--model_path", required=True, help="Path to CUT3R pretrained .pth model")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--image_size", type=int, default=512, help="Resize size")
    parser.add_argument("--vis_threshold", type=float, default=1.5, help="Confidence threshold")
    parser.add_argument("--no_cam", action="store_true", help="Do not visualize camera points in GLB")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    args = parser.parse_args()

    result = run_inference(
        lang=args.lang,
        inpdir=args.inpdir,
        model_path=args.model_path,
        outdir=args.outdir,
        image_size=args.image_size,
        vis_threshold=args.vis_threshold,
        show_cam=not args.no_cam,
        device=args.device,
    )

    print(
        msg(
            args.lang,
            "=== ✅ CUT3R 再構成が完了しました ===",
            "=== ✅ CUT3R Reconstruction Completed ===",
        )
    )
    print(result)


if __name__ == "__main__":
    main()