import os
import time
import json
import glob
import shutil
import random
import string
import zipfile
import tempfile
import traceback
import subprocess
import platform
from datetime import datetime

import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

import torch
import torchvision.transforms as T
import lpips
import piq
from torch_fidelity import calculate_metrics

import gradio as gr
from tqdm import tqdm

# subprocessのshellフラグの設定
SHELL_FLAG = platform.system() == "Windows"

# --- ディレクトリ内の画像パスをリスト化するメソッド ---
def get_imagelist(dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    files = sorted([f for ext in exts for f in glob.glob(os.path.join(dir, ext))])
    return files

# --- 評価指標ロードメソッド ---
def load_json_nerfstudio(json_path, psnr_key, ssim_key, lpips_key):
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})

    return [[results.get(psnr_key),results.get(ssim_key),results.get(lpips_key)]]

# 入力画像を1つのディレクトリにまとめるメソッド
def copy_images(image_paths, parent_path, name):
    if name == "":
        # 英数字8文字のランダム文字列
        name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    # 出力ディレクトリの作成
    output_path = os.path.join(parent_path, name, "images")
    os.makedirs(output_path, exist_ok=True)

    for img_path in image_paths:
        basename = os.path.basename(img_path)
        dst_path = os.path.join(output_path, basename)

        if os.path.exists(dst_path):
            print(f"{dst_path} は既に存在します")
            continue
        shutil.copy(img_path, dst_path)
    
    imagelist = get_imagelist(output_path)

    return output_path, output_path, imagelist, gr.Column(visible=True)

def remove_similar_images(input_dir: str, ssim_threshold: float = 0.8):
    """
    フォルダ内の類似画像をSSIMで判定し削除する
    ・縮小のみ
    """
    def preprocess(img, size=256):
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    images = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    if not images:
        return "0%", "0", "0"

    reference_path = os.path.join(input_dir, images[0])
    ref_raw = cv2.imread(reference_path)
    if ref_raw is None:
        return "0%", "0", "0"

    reference_img = preprocess(ref_raw)
    original_count = len(images)

    for img_name in tqdm(images[1:], desc="Removing similar images"):
        img_path = os.path.join(input_dir, img_name)
        raw = cv2.imread(img_path)
        if raw is None:
            continue

        current_img = preprocess(raw)

        ssim_val = ssim(
            reference_img,
            current_img,
            channel_axis=2,  # ← カラーSSIM
            data_range=255
        )

        if ssim_val < ssim_threshold:
            reference_img = current_img
        else:
            os.remove(img_path)

    selected_count = len([f for f in os.listdir(input_dir) if f.endswith(".png")])
    rejected_count = original_count - selected_count
    compression_rate = f"{(selected_count / original_count * 100):.3g}%"

    return compression_rate, str(selected_count), str(rejected_count)

def extract_frames_with_filter(video, parent_path, fps, remove_similar, ssim_threshold):
    video_name = os.path.splitext(os.path.basename(video))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_dir = os.path.join(parent_path, video_name, timestamp)
    output_path = os.path.join(dataset_dir, "images")
    os.makedirs(output_path, exist_ok=True)

    global SHELL_FLAG
    command = [
        "ffmpeg",
        "-i", video,
        "-vf", f"fps={fps}",
        os.path.join(output_path, "%04d.png")
    ]
    subprocess.run(command, check=True, shell=SHELL_FLAG)

    # フレーム抽出後の画像リスト
    extracted_images = sorted([
        f for f in os.listdir(output_path) if f.endswith(".png")
    ])

    if remove_similar:
        comp_rate, sel_images_num, rej_images_num = remove_similar_images(output_path, ssim_threshold)
    else:
        comp_rate = "100%"
        sel_images_num = len(extracted_images)  
        rej_images_num = "0枚"

    # フルパスで返す
    imagelist = sorted([
        os.path.join(output_path, f)
        for f in os.listdir(output_path) if f.endswith(".png")
    ])

    return output_path, output_path, comp_rate, sel_images_num, rej_images_num, imagelist, gr.Column(visible=True)

# データセットロードメソッド
def unzip_dataset(zip_file, datasets_parent):
    if zip_file is None:
        return None, None, "❌ ZIP が指定されていません"

    try:
        # ZIPデータの読み込み
        if hasattr(zip_file, "read"):
            data = zip_file.read()
        elif isinstance(zip_file, (bytes, bytearray, memoryview)):
            data = bytes(zip_file)
        elif isinstance(zip_file, str):
            with open(zip_file, "rb") as f:
                data = f.read()
        else:
            return None, None, f"❌ 想定外の入力型です: {type(zip_file)}"

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as f:
            f.write(data)
            zip_path = f.name

        try:
            if not zipfile.is_zipfile(zip_path):
                with open(zip_path, "rb") as f:
                    print("file head:", f.read(8))
                return None, None, "❌ 指定されたファイルは ZIP として認識できません"

            # 展開先パスを決定
            basename = os.path.splitext(os.path.basename(getattr(zip_file, "name", "dataset.zip")))[0]
            dataset_path = os.path.join(datasets_parent, basename)
            os.makedirs(dataset_path, exist_ok=True)

            # 解凍処理
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dataset_path)

            # ディレクトリ確認
            image_path = os.path.join(dataset_path, "images")
            colmap_path = os.path.join(dataset_path, "colmap")

            has_images = os.path.isdir(image_path)
            has_colmap = os.path.isdir(colmap_path)

            if not has_images and not has_colmap:
                return None, None, "⚠️ ZIP内に 'images' または 'colmap' ディレクトリが見つかりません"

            return (
                image_path if has_images else None,
                colmap_path if has_colmap else None,
                f"✅ 解凍しました: {dataset_path}"
            )

        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    except Exception as e:
        return None, None, f"❌ 解凍中にエラーが発生しました: {e}"

def zip_dataset(dataset):
    dirname = os.path.dirname(dataset)
    dataset_path = os.path.abspath(dirname)

    if not os.path.isdir(dataset_path):
        raise ValueError("dataset_path はディレクトリである必要があります")

    # データセット名（ZIP の名前）
    zip_path = dataset_path + ".zip"

    # 含めるフォルダ（存在確認）
    include_images = os.path.isdir(os.path.join(dataset_path, "images"))
    include_colmap = os.path.isdir(os.path.join(dataset_path, "colmap"))

    # ZIP 作成
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

        # images があれば追加
        if include_images:
            images_dir = os.path.join(dataset_path, "images")
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    full = os.path.join(root, file)
                    arcname = os.path.relpath(full, dataset_path)
                    zipf.write(full, arcname)

        # colmap があれば追加
        if include_colmap:
            colmap_dir = os.path.join(dataset_path, "colmap")
            for root, dirs, files in os.walk(colmap_dir):
                for file in files:
                    full = os.path.join(root, file)
                    arcname = os.path.relpath(full, dataset_path)
                    zipf.write(full, arcname)

    return zip_path

# --- colmap実行デバイスを判断するメソッド ---
def detect_colmap_gpu():
    try:
        result = subprocess.run(
            ["colmap", "feature_extractor", "-h"],
            capture_output=True,
            text=True
        )
        if "use_gpu" in result.stdout:
            import torch
            return "1" if torch.cuda.is_available() else "0"
        return "0"
    except Exception:
        return "0"

# --- colmap用画像リサイズメソッド ---
def make_multiscale_images(
    src_dir,
    out_root,
    scales=(2, 4, 8),
    exts=(".jpg", ".jpeg", ".png")
):
    img_files = [
        f for f in os.listdir(src_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]

    for s in scales:
        out_dir = os.path.join(out_root, f"images_{s}")
        os.makedirs(out_dir, exist_ok=True)

        for f in img_files:
            src = os.path.join(src_dir, f)
            dst = os.path.join(out_dir, f)

            with Image.open(src) as im:
                w, h = im.size
                im.resize(
                    (w // s, h // s),
                    Image.LANCZOS
                ).save(dst)
    
# --- colmap 実行メソッド ---
def run_colmap(exe_mode, image_dataset, rebuild):
    if not image_dataset:
        return "", "画像データセットがセットされていません", gr.Column(visible=False)

    global SHELL_FLAG
    all_logs = []

    dataset = os.path.dirname(image_dataset)
    out_dir = os.path.join(dataset, "colmap")
    input_dir = os.path.join(out_dir, "input")
    distorted_dir = os.path.join(out_dir, "distorted")

    # ===== rebuild 処理 =====
    if rebuild and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        all_logs.append("colmap ディレクトリを削除しました．")

    if not rebuild and os.path.exists(out_dir):
        all_logs.append("既に COLMAP 処理済みです．")
        return out_dir, "\n".join(all_logs), gr.Column(visible=True)

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(distorted_dir, exist_ok=True)
    os.makedirs(os.path.join(distorted_dir, "sparse"), exist_ok=True)

    # ===== images → colmap/input =====
    for f in os.listdir(image_dataset):
        src = os.path.join(image_dataset, f)
        if (
            os.path.isfile(src)
            and os.path.splitext(f)[1].lower()
            in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        ):
            shutil.copy2(src, input_dir)

    use_gpu = detect_colmap_gpu()

    def run(cmd, cwd=None):
        all_logs.append(" ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                shell=SHELL_FLAG
            )
            if result.stdout:
                all_logs.append(result.stdout.strip())
            if result.stderr:
                all_logs.append("【STDERR】")
                all_logs.append(result.stderr.strip())
            if result.returncode != 0:
                all_logs.append(f"コマンドが異常終了しました（コード: {result.returncode}）")
                return False
            return True
        except Exception as e:
            all_logs.append(f"コマンド実行中に例外が発生しました: {e}")
            return False

    try:
        if exe_mode == "local":

            # ===== COLMAP =====
            run([
                "colmap", "feature_extractor",
                "--database_path", os.path.join(distorted_dir, "database.db"),
                "--image_path", input_dir,
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "OPENCV",
                "--SiftExtraction.use_gpu", use_gpu
            ])

            run([
                "colmap", "exhaustive_matcher",
                "--database_path", os.path.join(distorted_dir, "database.db"),
                "--SiftMatching.use_gpu", use_gpu
            ])

            run([
                "colmap", "mapper",
                "--database_path", os.path.join(distorted_dir, "database.db"),
                "--image_path", input_dir,
                "--output_path", os.path.join(distorted_dir, "sparse"),
                "--Mapper.ba_global_function_tolerance", "1e-6"
            ])

            run([
                "colmap", "image_undistorter",
                "--image_path", input_dir,
                "--input_path", os.path.join(distorted_dir, "sparse", "0"),
                "--output_path", out_dir,
                "--output_type", "COLMAP"
            ])

            all_logs.append("COLMAP 変換完了．")

            # ===== sparse/0 =====
            sparse_root = os.path.join(out_dir, "sparse")
            sparse0 = os.path.join(sparse_root, "0")
            os.makedirs(sparse0, exist_ok=True)

            for name in ["cameras.bin", "images.bin", "points3D.bin"]:
                src = os.path.join(sparse_root, name)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(sparse0, name))

            # ===== GS マルチスケール =====
            undistorted_images = os.path.join(out_dir, "images")
            make_multiscale_images(
                src_dir=undistorted_images,
                out_root=out_dir,
                scales=(2, 4, 8)
            )

            # ===== Nerfstudio =====
            colmap_model_path = os.path.join(out_dir, "sparse", "0")

            run([
                "conda", "run", "-n", "nerfstudio",
                "ns-process-data", "images",
                "--data", input_dir,
                "--output-dir", out_dir,
                "--skip-colmap",
                "--colmap-model-path", colmap_model_path,
                "--skip-image-processing",
                "--camera-type", "perspective",
                "--same-dimensions"
            ], cwd=os.path.join("models", "nerfstudio"))

            all_logs.append("Nerfstudio データ変換完了．")

        elif exe_mode == "slurm":

            # ===== sbatch 実行 =====
            sbatch_script = os.path.join("scripts", "run_colmap.sh")

            cmd = [
                "sbatch",
                sbatch_script,
                input_dir,
                out_dir,
                distorted_dir,
                use_gpu
            ]

            run(cmd, cwd="./")
            all_logs.append("SLURM ジョブを投入しました．")

        return out_dir, "\n".join(all_logs), gr.Column(visible=True)

    except Exception as e:
        all_logs.append(f"エラー: {e}")
        return "", "\n".join(all_logs), gr.Column(visible=False)


# --- 評価計算メソッド ---
def evaluate_all_metrics(method_name, gt_dir, render_dir, output_dir):
    start_time = time.perf_counter()
    log_lines = []
    returncode = 0
    summary_list = None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_lines.append(f"[INFO] device = {device}\n")

        to_tensor = T.ToTensor()
        lpips_fn = lpips.LPIPS(net="vgg").to(device)

        def load_image(path):
            img = Image.open(path).convert("RGB")
            return to_tensor(img).unsqueeze(0).to(device)

        # ------------------------
        # Per-image metrics
        # ------------------------
        per_image = []
        gt_files = sorted(os.listdir(gt_dir))

        for fname in tqdm(gt_files, desc="Evaluating image pairs", ncols=80):
            gt_path = os.path.join(gt_dir, fname)
            pred_path = os.path.join(render_dir, fname)

            if not os.path.isfile(gt_path):
                continue
            if not os.path.exists(pred_path):
                log_lines.append(f"[WARN] missing: {fname}\n")
                continue

            gt = load_image(gt_path)
            pred = load_image(pred_path)

            _, _, h, w = pred.shape
            if h < 161 or w < 161:
                ms_ssim_val = float("nan")
            else:
                ms_ssim_val = piq.multi_scale_ssim(pred, gt, data_range=1.0).item()

            metrics = {
                "image": fname,
                "psnr": piq.psnr(pred, gt, data_range=1.0).item(),
                "ssim": piq.ssim(pred, gt, data_range=1.0).item(),
                "ms_ssim": ms_ssim_val,
                "lpips": lpips_fn(pred, gt).item(),
                "fsim": piq.fsim(pred, gt, data_range=1.0).item(),
                "vif": piq.vif_p(pred, gt, data_range=1.0).item(),
                "brisque": getattr(piq, "brisque", lambda x: float("nan"))(pred).item()
            }
            per_image.append(metrics)
            log_lines.append(f"[INFO] evaluated {fname}\n")  # ここで逐次ログに追加

        if len(per_image) == 0:
            raise RuntimeError("No valid image pairs found.")

        metric_order = [
            "psnr", "ssim", "ms_ssim", "lpips",
            "fsim", "vif", "brisque", "fid"
        ]

        summary_dict = {
            k: float(np.mean([m[k] for m in per_image]))
            for k in metric_order
            if k in per_image[0]
        }

        # ------------------------
        # FID
        # ------------------------
        metrics = calculate_metrics(
            input1=gt_dir,
            input2=render_dir,
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=False
        )
        summary_dict["fid"] = float(metrics["frechet_inception_distance"])

        # ------------------------
        # Output
        # ------------------------
        summary_list = [[method_name] + [round(summary_dict.get(k, float("nan")), 3) for k in metric_order]]

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics_per_image.json"), "w", encoding="utf-8") as f:
            json.dump(per_image, f, indent=2)

        with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        log_lines.append("[INFO] metric evaluation finished successfully\n")

    except Exception as e:
        returncode = 1
        log_lines.append("[ERROR] metric evaluation failed\n")
        log_lines.append(str(e) + "\n")
        log_lines.append(traceback.format_exc())
        summary_list = None

    # 最終ログを行ごとに整形
    full_log = "".join(log_lines)
    status = "✅ Success" if returncode == 0 else "❌ Failed"
    run_time = time.perf_counter() - start_time

    return run_time, status, full_log, summary_list