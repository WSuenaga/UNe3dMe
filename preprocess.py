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
    output_path = os.path.join(parent_path, name, "input")
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

    remaining_count = len([f for f in os.listdir(input_dir) if f.endswith(".png")])
    deleted_count = original_count - remaining_count

    compression_rate = 0.0
    if original_count > 0:
        compression_rate = (remaining_count / original_count) * 100
        compression_rate = float(f"{compression_rate:.3g}")
    return f"{compression_rate}%", f"{deleted_count}枚", 

def extract_frames_with_filter(video, parent_path, fps, remove_similar, ssim_threshold):
    video_name = os.path.splitext(os.path.basename(video))[0]
    output_path = os.path.join(parent_path, video_name, "input")
    
    # ディレクトリが既に存在し、画像が存在する場合はスキップ
    existing_images = sorted([
        f for f in os.listdir(output_path) if f.endswith(".png")
    ]) if os.path.exists(output_path) else []

    if existing_images:
        print(f"ディレクトリ {output_path} は既に存在します。抽出済みの画像を使用します。")
        comp_rate, del_images_num = "100%", "0枚"  # 過去結果の概算（必要に応じて変更）
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

    dataset_dir = output_path = os.path.join(parent_path, video_name)

    return dataset_dir, gr.Column(visible=True), dataset_dir, comp_rate, del_images_num, imagelist

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

# COLMAP実行メソッド
def run_colmap(dataset, force_rebuild):
    if dataset == "":
        return "データセットがセットされていません", gr.Column(visible=False)

    global SHELL_FLAG
    all_logs = []

    images_dir = os.path.join(dataset, "images")
    transforms_path = os.path.join(dataset, "transforms.json")

    # --- force_rebuild=True: input以外を削除 ---
    if force_rebuild:
        all_logs.append("再構築フラグが有効なため，input以外の既存データを削除して再実行します．")
        for item in os.listdir(dataset):
            item_path = os.path.join(dataset, item)

            # inputディレクトリは削除しない
            if os.path.basename(item_path) == "input":
                all_logs.append(f"保持: {item_path}")
                continue

            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    all_logs.append(f"削除: {item_path}")
                else:
                    os.remove(item_path)
                    all_logs.append(f"削除: {item_path}")
            except Exception as e:
                all_logs.append(f"削除エラー: {item_path} ({e})")
    # --- force_rebuild=Falseかつ処理済みの場合はスキップ ---
    elif os.path.exists(images_dir) and os.path.exists(transforms_path):
        msg = "既存の images/ および transforms.json が見つかったため，処理をスキップしました．"
        all_logs.append(msg)
        return "\n".join(all_logs), gr.Column(visible=True)

    # --- COLMAP実行 ---
    script_path = "convert.py"
    cmd_colmap = [
        "conda", "run", "-n", "gaussian_splatting", "python", script_path,
        "--source_path", dataset,
        "--resize"
    ]

    print("Running:", " ".join(cmd_colmap))
    result = subprocess.run(
        cmd_colmap,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd="./models/gaussian-splatting/",
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
        return (
            "COLMAP変換に失敗しました\n\n" + "\n".join(all_logs),
            gr.Column(visible=False)
        )

    # --- ns-process-data 実行 ---
    input_path = os.path.join(dataset, "input")
    colmap_model_path = os.path.join(dataset, "sparse", "0")
    cmd_ns = [
        "conda", "run", "-n", "nerfstudio",
        "ns-process-data", "images",
        "--data", input_path,
        "--output-dir", dataset,
        "--skip-colmap",
        "--colmap-model-path", colmap_model_path,
        "--skip-image-processing",
        "--camera-type", "perspective",
        "--same-dimensions"
    ]

    print("Running:", " ".join(cmd_ns))
    result_ns = subprocess.run(
        cmd_ns,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd="./models/nerfstudio/",
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
        return (
            "ns-process-data images 実行に失敗しました\n\n" + "\n".join(all_logs),
            gr.Column(visible=False)
        )

    all_logs.append("\nすべての処理が正常に完了しました。")
    return "\n".join(all_logs), gr.Column(visible=True)