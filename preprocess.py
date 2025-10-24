# 標準ライブラリ
import glob
import os
import random
import shutil
import string
import subprocess
import zipfile
# サードパーティライブラリ
import cv2
import gradio as gr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# 指定したディレクトリ内の画像パスをリスト化するメソッド
def get_imagelist(dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    files = sorted([f for ext in exts for f in glob.glob(os.path.join(dir, ext))])
    return files

# 入力画像を1つのディレクトリにまとめるメソッド
def copy_images(image_paths, parent_path, name):
    if name == "":
        # 英数字8文字のランダム文字列
        name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    # 出力ディレクトリの作成
    output_path = os.path.join(parent_path, name)
    os.makedirs(output_path, exist_ok=True)

    for img_path in image_paths:
        basename = os.path.basename(img_path)
        dst_path = os.path.join(output_path, basename)

        if os.path.exists(dst_path):
            print(f"{dst_path} は既に存在します")
            continue
        shutil.copy(img_path, dst_path)
    
    imagelist = get_imagelist(output_path)
    return output_path, gr.Column(visible=True), output_path, imagelist

def remove_similar_images(input_dir: str, ssim_threshold: float = 0.95):
    """
    フォルダ内の類似画像をSSIMで判定し削除する
    削除枚数と圧縮率(残存率)を返す
    """
    def compute_ssim(img1, img2):
        return ssim(img1, img2, data_range=img2.max() - img2.min(), channel_axis=-1)

    images = sorted(os.listdir(input_dir))
    if not images:
        print("入力フォルダに画像がありません")
        return 0, 0.0

    reference_path = os.path.join(input_dir, images[0])
    reference_img = cv2.imread(reference_path)
    if reference_img is None:
        print(f"基準画像の読み込みに失敗: {images[0]}")
        return 0, 0.0

    original_count = len([f for f in images if f.endswith(".png")])

    # 最初の画像は残す
    for img_name in tqdm(images[1:], desc="Removing similar images"):
        img_path = os.path.join(input_dir, img_name)
        current_img = cv2.imread(img_path)
        if current_img is None:
            continue

        ssim_val = compute_ssim(reference_img, current_img)
        if ssim_val < ssim_threshold:
            reference_img = current_img
        else:
            os.remove(img_path)

    remaining_count = len([f for f in os.listdir(input_dir) if f.endswith(".png")])
    deleted_count = original_count - remaining_count

    compression_rate = 0.0
    if original_count > 0:
        compression_rate = (remaining_count / original_count) * 100
        compression_rate = float(f"{compression_rate:.3g}")
    return f"{compression_rate}%", f"{deleted_count}枚", 

def extract_frames_with_filter(video, parent_path, fps, remove_similar, ssim_threshold):
    video_name = os.path.splitext(os.path.basename(video))[0]
    output_path = os.path.join(parent_path, video_name)
    
    # ディレクトリが既に存在し、画像が存在する場合はスキップ
    existing_images = sorted([
        f for f in os.listdir(output_path) if f.endswith(".png")
    ]) if os.path.exists(output_path) else []

    if existing_images:
        print(f"ディレクトリ {output_path} は既に存在します。抽出済みの画像を使用します。")
        comp_rate, del_images_num = "100%", "0枚"  # 過去結果の概算（必要に応じて変更）
    else:
        os.makedirs(output_path, exist_ok=True)

        command = [
            "ffmpeg",
            "-i", video,
            "-vf", f"fps={fps}",
            os.path.join(output_path, "%04d.png")
        ]
        subprocess.run(command, check=True)

        if remove_similar:
            comp_rate, del_images_num = remove_similar_images(output_path, ssim_threshold)
        else:
            # 削除しない場合の値
            comp_rate = "100%"
            del_images_num = "0枚"

    # フルパスで返す
    imagelist = sorted([
        os.path.join(output_path, f)
        for f in os.listdir(output_path) if f.endswith(".png")
    ])

    return output_path, gr.Column(visible=True), output_path, comp_rate, del_images_num, imagelist

def unzip_dataset(zip_file, datasets_parent):
    # zipファイル名（拡張子なし）をデータセット名にする
    basename = os.path.splitext(os.path.basename(zip_file.name))[0]
    output_path = os.path.join(datasets_parent, basename)
    
    if os.path.exists(output_path) and os.listdir(output_path):
        print(f"既に展開済み: {output_path}")
        return output_path, gr.Column(visible=True)

    # 解凍
    with zipfile.ZipFile(zip_file.name, "r") as zip_ref:
        zip_ref.extractall(output_path)
    print(f"解凍しました: {output_path}")

    return output_path, gr.Column(visible=True)

def create_images_dir(dataset):
    if dataset == "":
        return "データセットがセットされていません", gr.Column(visible=False)
    
    # imagesフォルダを作成
    images_dir = os.path.join(dataset, "images")
    if os.path.exists(images_dir):
        return "前処理済みです", gr.Column(visible=True)
    else:
        os.makedirs(images_dir, exist_ok=True)

    # dataset直下の画像を images にコピー
    copied = 0
    for file in os.listdir(dataset):
        file_path = os.path.join(dataset, file)
        if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy2(file_path, os.path.join(images_dir, file))
            copied += 1

    if copied == 0:
        return "画像ファイルが見つかりませんでした", gr.Column(visible=False)

    log = f"{copied} 枚の画像を images フォルダにコピーしました"

    return log, gr.Column(visible=True)

# ---nerfstudio_COLMAP実行メソッド---
def run_nscolmap(dataset):
    if dataset == "":
        return "データセットがセットされていません", gr.Column(visible=False)
    
    # nsフォルダを作成
    ns_dir = os.path.join(dataset, "ns")
    if os.path.exists(ns_dir):
        return "前処理済みです", gr.Column(visible=True)
    else:
        os.makedirs(ns_dir, exist_ok=True)

    # COLMAP実行コマンド
    cmd = [
        "conda", "run", "-n", "nerfstudio", "ns-process-data", "images",
        "--data", dataset,
        "--output-dir", ns_dir
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    # 標準出力とエラーを結合
    error_output = result.stderr.strip()

    # returncodeが0以外ならエラー
    if result.returncode != 0:
        log = f"前処理に失敗しました\n\nエラー内容:\n{error_output}"
        return log, gr.Column(visible=False)

    log = "前処理が完了しました"

    return log, gr.Column(visible=True)

# ---3dgs_COLMAP実行メソッド---
def run_gscolmap(dataset):
    if dataset == "":
        return "データセットがセットされていません" 
    
    # gsフォルダを作成
    gs_dir = os.path.join(dataset, "gs")
    if os.path.exists(gs_dir):
        return "前処理済みです", gr.Column(visible=True)
    else:
        os.makedirs(gs_dir, exist_ok=True)

    # input フォルダ作成
    input_dir = os.path.join(gs_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # dataset直下の画像を input にコピー
    for file in os.listdir(dataset):
        file_path = os.path.join(dataset, file)
        if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy2(file_path, os.path.join(input_dir, file))

    # COLMAP実行コマンド
    script_path = "./models/gaussian-splatting/convert.py"
    cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", script_path,
        "--source_path", gs_dir,
        "--resize"
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    # 標準出力とエラーを結合
    error_output = result.stderr.strip()

    # returncode が 0 以外ならエラーとして返す
    if result.returncode != 0:
        log = f"前処理に失敗しました\n\nエラー内容:\n{error_output}"
        return log, gr.Column(visible=False)
    log = "前処理が完了しました"

    return log, gr.Column(visible=True)

# ---mip-splatting_COLMAP実行メソッド---
def run_mscolmap(dataset):
    if dataset == "":
        return "データセットがセットされていません" 
    
    # msフォルダを作成
    ms_dir = os.path.join(dataset, "ms")
    if os.path.exists(ms_dir):
        return "前処理済みです", gr.Column(visible=True)
    else:
        os.makedirs(ms_dir, exist_ok=True)

    # input フォルダ作成
    input_dir = os.path.join(ms_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # dataset直下の画像を input にコピー
    for file in os.listdir(dataset):
        file_path = os.path.join(dataset, file)
        if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy2(file_path, os.path.join(input_dir, file))

    # COLMAP実行コマンド
    script_path = "./models/mip-splatting/convert.py"
    cmd = [
        "conda", "run", "-n", "mip-splatting", "python", script_path,
        "--source_path", ms_dir,
        "--resize"
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    # 標準出力とエラーを結合
    error_output = result.stderr.strip()

    # returncode が 0 以外ならエラーとして返す
    if result.returncode != 0:
        log = f"前処理に失敗しました\n\nエラー内容:\n{error_output}"
        return log, gr.Column(visible=False)
    log = "前処理が完了しました"

    return log, gr.Column(visible=True)

# ---mip-splatting_COLMAP実行メソッド---
def run_4dgscolmap(dataset):
    if dataset == "":
        return "データセットがセットされていません" 
    
    # msフォルダを作成
    gs4d_dir = os.path.join(dataset, "4dgs")
    if os.path.exists(gs4d_dir):
        return "前処理済みです", gr.Column(visible=True)
    else:
        os.makedirs(gs4d_dir, exist_ok=True)

    # input フォルダ作成
    input_dir = os.path.join(gs4d_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    # dataset直下の画像を input にコピー
    for file in os.listdir(dataset):
        file_path = os.path.join(dataset, file)
        if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy2(file_path, os.path.join(input_dir, file))

    # COLMAP実行コマンド
    script_path = "./models/4DGaussians/convert.py"
    cmd = [
        "conda", "run", "-n", "Gaussians4D", "python", script_path,
        "--source_path", gs4d_dir,
        "--resize"
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    # 標準出力とエラーを結合
    error_output = result.stderr.strip()

    # returncode が 0 以外ならエラーとして返す
    if result.returncode != 0:
        log = f"前処理に失敗しました\n\nエラー内容:\n{error_output}"
        return log, gr.Column(visible=False)
    log = "前処理が完了しました"

    return log, gr.Column(visible=True)