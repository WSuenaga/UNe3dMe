# 標準ライブラリ
import glob
import os
import random
import shutil
import string
import subprocess
import platform
import zipfile
# サードパーティライブラリ
import cv2
import gradio as gr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# subprocessのshellフラグの設定
SHELL_FLAG = platform.system() == "Windows"

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

    dataset_dir = output_path = os.path.join(parent_path, name)

    return dataset_dir, gr.Column(visible=True), dataset_dir, imagelist

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

    selected_count = len([f for f in os.listdir(input_dir) if f.endswith(".png")])
    rejected_count = original_count - selected_count

    compression_rate = 0.0
    if original_count > 0:
        compression_rate = (selected_count / original_count) * 100
        compression_rate = float(f"{compression_rate:.3g}")

    return f"{compression_rate}%", f"{selected_count}枚", f"{rejected_count}枚"

def extract_frames_with_filter(video, parent_path, fps, remove_similar, ssim_threshold):
    video_name = os.path.splitext(os.path.basename(video))[0]
    output_path = os.path.join(parent_path, video_name, "images")
    
    # ディレクトリが既に存在し、画像が存在する場合はスキップ
    if os.path.exists(output_path):
        existing_images = sorted([
            f for f in os.listdir(output_path) if f.endswith(".png")
        ])
    else:
        existing_images = []

    if existing_images:
        print(f"ディレクトリ {output_path} は既に存在します。抽出済みの画像を使用します。")
        comp_rate = ""
        sel_images_num = len(existing_images) 
        rej_images_num = "" 
    else:
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

    dataset_dir = os.path.join(parent_path, video_name)

    return dataset_dir, gr.Column(visible=True), dataset_dir, comp_rate, sel_images_num, rej_images_num, imagelist

def unzip_dataset(zip_file, datasets_parent):
    # zipファイル名（拡張子なし）をデータセット名にする
    basename = os.path.splitext(os.path.basename(zip_file.name))[0]
    dataset_path = os.path.join(datasets_parent, basename)
    
    if os.path.exists(dataset_path) and os.listdir(dataset_path):
        print(f"既に展開済み: {dataset_path}")
        return dataset_path, gr.Column(visible=True)

    # 解凍
    with zipfile.ZipFile(zip_file.name, "r") as zip_ref:
        zip_ref.extractall(dataset_path)
    print(f"解凍しました: {dataset_path}")

    return dataset_path, gr.Column(visible=True)

def zip_dataset(dataset):
    dataset_path = os.path.abspath(dataset)

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

def run_colmap(dataset, rebuild):
    if dataset == "":
        return "データセットがセットされていません", gr.Column(visible=False)

    global SHELL_FLAG
    all_logs = []

    images_dir = os.path.join(dataset, "images")
    out_dir = os.path. join(dataset, "colmap")
    input_dir = os.path.join(out_dir, "input")

    # --- rebuild フラグに応じた colmap ディレクトリ処理 ---
    if rebuild and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        all_logs.append("colmapディレクトリを削除しました．")
    
    if not rebuild and os.path.exists(out_dir):
        all_logs.append("処理済みです．")
        return "\n".join(all_logs), gr.Column(visible=True)
    
    # colmap/input ディレクトリ作成
    os.makedirs(input_dir, exist_ok=True)

    for image_file in os.listdir(images_dir):
        full_image_path = os.path.join(images_dir, image_file)
        # ファイルかつ画像形式か確認
        if os.path.isfile(full_image_path) and os.path.splitext(image_file)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            shutil.copy(full_image_path, input_dir)

    # --- COLMAP実行 ---
    script_path = "convert.py"
    cmd_colmap = [
        "conda", "run", "-n", "gaussian_splatting", "python", script_path,
        "--source_path", out_dir,
        "--resize"
    ]

    print("Running:", " ".join(cmd_colmap))
    cwd_colmap = os.path.join("models", "gaussian-splatting")
    result = subprocess.run(
        cmd_colmap,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd_colmap,
        shell=SHELL_FLAG
    )

    stdout_colmap = result.stdout.strip()
    stderr_colmap = result.stderr.strip()
    all_logs.append("【COLMAP実行ログ】")
    all_logs.append(stdout_colmap)
    if stderr_colmap:
        all_logs.append("【COLMAPエラー】")
        all_logs.append(stderr_colmap)

    if result.returncode != 0:
        return "COLMAP変換に失敗しました\n\n" + "\n".join(all_logs), gr.Column(visible=False)

    # --- ns-process-data 実行 ---
    colmap_model_path = os.path.join(out_dir, "sparse", "0")
    cmd_ns = [
        "conda", "run", "-n", "nerfstudio",
        "ns-process-data", "images",
        "--data", input_dir,
        "--output-dir", out_dir,
        "--skip-colmap",
        "--colmap-model-path", colmap_model_path,
        "--skip-image-processing",
        "--camera-type", "perspective",
        "--same-dimensions"
    ]

    print("Running:", " ".join(cmd_ns))
    cwd_nerfstudio = os.path.join("models", "nerfstudio")
    result_ns = subprocess.run(
        cmd_ns,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd_nerfstudio,
        shell=SHELL_FLAG
    )

    stdout_ns = result_ns.stdout.strip()
    stderr_ns = result_ns.stderr.strip()
    all_logs.append("\n【Nerfstudio変換ログ】")
    all_logs.append(stdout_ns)
    if stderr_ns:
        all_logs.append("【Nerfstudioエラー】")
        all_logs.append(stderr_ns)

    if result_ns.returncode != 0:
        return "ns-process-data images 実行に失敗しました\n\n" + "\n".join(all_logs), gr.Column(visible=False)

    return "\n".join(all_logs), gr.Column(visible=True)
