import glob
import json
import os
import platform
import random
import shutil
import socket
import string
import subprocess
import tempfile
import time
import traceback
import zipfile
from datetime import datetime

import cv2
import gradio as gr
import lpips
import numpy as np
import piq
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch_fidelity import calculate_metrics
from tqdm import tqdm

# subprocessのshellフラグの設定
SHELL_FLAG = platform.system() == "Windows"

# 保存先一時ディレクトリのパス（main.py 実行時に設定される）
TMPDIR = ""


# =========================
# 共通関数
# =========================

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


def get_imagelist(directory):
    """
    ディレクトリ内の画像ファイルパスを収集し，ソートして返す．

    対応拡張子は `.png`, `.jpg`, `.jpeg`, `.webp`．

    Args:
        directory (str): 画像を探索するディレクトリのパス．

    Returns:
        list[str]: 画像ファイルのパス一覧．
    """
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    files = sorted(
        [f for ext in exts for f in glob.glob(os.path.join(directory, ext))]
    )
    return files


# =========================
# 画像データセットの作成
# =========================

def copy_images(lang, image_paths, parent_path, name):
    """
    指定した画像をコピーして画像データセットを作成する．

    Args:
        lang (str): 言語コード．
        image_paths (list[str]): コピー元画像パスの一覧．
        parent_path (str): データセットの親ディレクトリ．
        name (str): データセット名．空文字の場合はランダム文字列を付与する．

    Returns:
        tuple[str, str, list[str]]:
            - output_path: 出力先の画像ディレクトリ．
            - log_text: 実行ログ．
            - imagelist: 出力後の画像一覧．
    """
    logs = []

    if not image_paths:
        log_text = msg(lang, "画像が指定されていません．", "No images were specified.")
        return "", log_text, []

    # データセット名が指定されていない場合はランダムな文字列を付与
    if name == "":
        name = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    # 出力ディレクトリを作成
    output_path = os.path.join(parent_path, name, "images")
    os.makedirs(output_path, exist_ok=True)

    # 画像を出力ディレクトリにコピー
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        dst_path = os.path.join(output_path, basename)

        if os.path.exists(dst_path):
            logs.append(msg(lang, f"{dst_path} は既に存在します．", f"{dst_path} already exists."))
            continue

        shutil.copy(img_path, dst_path)

    imagelist = get_imagelist(output_path)

    logs.append(msg(lang, f"📂 保存先: {output_path}", f"📂 Saved to: {output_path}"))

    log_text = "\n".join(logs)

    return output_path, log_text, imagelist


def remove_similar_images(input_dir, ssim_threshold):
    """
    SSIM を用いて類似した連続画像を削除する．

    先頭画像を基準画像とし，以降は直前に採用された画像とのみ比較する．
    SSIM がしきい値以上の画像は類似画像とみなして削除する．

    Args:
        input_dir (str): 比較対象画像が格納されたディレクトリ．
        ssim_threshold (float): 類似判定に用いる SSIM のしきい値．

    Returns:
        tuple[str, str, str]:
            - compression_rate: 元画像に対する残存率．
            - selected_count: 残った画像枚数．
            - rejected_count: 削除した画像枚数．
    """
    def preprocess(img, size=256):
        """
        アスペクト比を保ったまま画像を縮小する．

        Args:
            img (numpy.ndarray): 入力画像．
            size (int, optional): 長辺の目標サイズ．デフォルトは 256．

        Returns:
            numpy.ndarray: 縮小後の画像．
        """
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    # 切り出された画像群を読み込み
    images = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    if not images:
        return "0%", "0", "0"

    # 最初の画像を初期基準画像に設定
    reference_path = os.path.join(input_dir, images[0])
    ref_raw = cv2.imread(reference_path)
    if ref_raw is None:
        return "0%", "0", "0"
    reference_img = preprocess(ref_raw)

    # 切り出された画像総数
    original_count = len(images)

    # 類似画像を SSIM で判定し，削除（連続したフレーム同士のみ比較）
    for img_name in tqdm(images[1:], desc="Removing similar images"):
        img_path = os.path.join(input_dir, img_name)
        raw = cv2.imread(img_path)
        if raw is None:
            continue

        current_img = preprocess(raw)

        ssim_val = ssim(
            reference_img,
            current_img,
            channel_axis=2,  # カラー画像として比較
            data_range=255,
        )

        if ssim_val < ssim_threshold:
            reference_img = current_img
        else:
            os.remove(img_path)

    # 残った画像枚数
    selected_count = len([f for f in os.listdir(input_dir) if f.endswith(".png")])
    # 削除した画像枚数
    rejected_count = original_count - selected_count
    # 元画像に対する残存率
    compression_rate = f"{(selected_count / original_count * 100):.3g}%"

    return compression_rate, str(selected_count), str(rejected_count)


def extract_frames_with_filter(video, parent_path, fps, remove_similar, ssim_threshold):
    """
    動画からフレームを抽出し，必要に応じて類似画像を削除して画像データセットを作成する．

    Args:
        video (str): 入力動画ファイルのパス．
        parent_path (str): データセットの親ディレクトリ．
        fps (float | int): フレーム抽出感覚．
        remove_similar (bool): 類似画像削除を行うかどうか．
        ssim_threshold (float): 類似画像削除時に用いる SSIM のしきい値．

    Returns:
        tuple[str, str, str, str, str, list[str]]:
            - output_path: 出力先ディレクトリ（ログを含む）
            - output_path: 出力先ディレクトリ（ログを含む）
            - comp_rate: 圧縮率
            - sel_images_num: 残存画像枚数
            - rej_images_num: 削除画像枚数
            - imagelist: 最終画像のパスのリスト
    """
    global SHELL_FLAG

    video_name = os.path.splitext(os.path.basename(video))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join(parent_path, video_name, timestamp)

    # 出力ディレクトリを作成
    output_path = os.path.join(dataset_dir, "images")
    os.makedirs(output_path, exist_ok=True)

    # ffmpeg でフレームを抽出
    command = [
        "ffmpeg",
        "-i", video,
        "-vf", f"fps={fps}",
        os.path.join(output_path, "%04d.png"),
    ]
    subprocess.run(command, check=True, shell=SHELL_FLAG)

    # フレーム抽出後の画像リスト
    extracted_images = sorted([f for f in os.listdir(output_path) if f.endswith(".png")])

    # 類似画像削除オプション
    if remove_similar:
        comp_rate, sel_images_num, rej_images_num = remove_similar_images(output_path, ssim_threshold)
    else:
        comp_rate = "100%"
        sel_images_num = len(extracted_images)
        rej_images_num = "0"

    # 最終的な画像リスト
    imagelist = sorted(
        [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".png")]
    )

    return output_path, output_path, comp_rate, sel_images_num, rej_images_num, imagelist


# =========================
# COLMAPデータセットの作成
# =========================

def make_multiscale_images(
    src_dir,
    out_root,
    scales=(2, 4, 8),
    exts=(".jpg", ".jpeg", ".png"),
):
    """
    入力画像から複数解像度の縮小画像群を生成する．

    Args:
        src_dir (str): 元画像が格納されたディレクトリ．
        out_root (str): 縮小画像ディレクトリを作成する親ディレクトリ．
        scales (tuple[int, ...], optional): 縮小倍率．デフォルトは (2, 4, 8)．
        exts (tuple[str, ...], optional): 対象とする拡張子．デフォルトは (".jpg", ".jpeg", ".png")．

    Returns:
        None
    """
    scale_dirs = {}
    for s in scales:
        dst_dir = os.path.join(out_root, f"images_{s}")
        os.makedirs(dst_dir, exist_ok=True)
        scale_dirs[s] = dst_dir

    for entry in os.scandir(src_dir):
        if not entry.is_file():
            continue
        if os.path.splitext(entry.name)[1].lower() not in exts:
            continue

        with Image.open(entry.path) as im:
            w, h = im.size
            for s, dst_dir in scale_dirs.items():
                resized = im.resize((max(1, w // s), max(1, h // s)), Image.LANCZOS)
                resized.save(os.path.join(dst_dir, entry.name))


def run_colmap(lang, exe_mode, image_dataset, rebuild):
    """
    COLMAP 前処理全体をまとめて実行する．

    処理内容:
        1. 入力画像の準備
        2. feature extraction / matching / mapping
        3. image undistortion
        4. sparse model の正規化
        5. multiscale 画像生成
        6. nerfstudio 形式への変換

    Args:
        lang (str): 言語コード．
        exe_mode (str): 実行モード．"local" または "slurm"．
        image_dataset (str): 入力画像ディレクトリ．
        rebuild (bool): 既存の colmap ディレクトリを作り直すかどうか（強制再処理）．

    Returns:
        tuple[str, str, gr.Column]:
            - 出力パス（失敗時は空文字）．
            - ログ全文．
            - 表示制御用の gr.Column．
    """
    global SHELL_FLAG

    # 画面に返すログをここへ蓄積する
    all_logs = []

    # 入力画像として許可する拡張子
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    # multiscale 生成時に対象にする拡張子
    MULTISCALE_EXTS = (".jpg", ".jpeg", ".png")
    # nerfstudio 用に作る縮小画像ディレクトリの倍率
    MULTISCALES = (2, 4, 8)
    # COLMAP モデルとして最低限必要なバイナリ
    MODEL_BIN_NAMES = ("cameras.bin", "images.bin", "points3D.bin")

    # 入力チェック
    if not image_dataset:
        return (
            "",
            msg(lang, "画像データセットがセットされていません．", "No image dataset has been set."),
            gr.Column(visible=False),
        )

    # ディレクトリ構成
    dataset = os.path.dirname(image_dataset)
    out_dir = os.path.join(dataset, "colmap")
    input_dir = os.path.join(out_dir, "input")
    distorted_dir = os.path.join(out_dir, "distorted")
    mapper_sparse_dir = os.path.join(distorted_dir, "sparse")
    db_path = os.path.join(distorted_dir, "database.db")

    def log(jp_msg, en_msg):
        """
        現在の UI 言語に応じてログを追加する．

        Args:
            jp_msg (str): 日本語ログ．
            en_msg (str): 英語ログ．

        Returns:
            None
        """
        all_logs.append(msg(lang, jp_msg, en_msg))

    def finish(path="", visible=False, jp_msg=None, en_msg=None):
        """
        最終返却用の共通処理を行う．

        必要に応じて最後のログを追加し，返却形式を統一する．

        Args:
            path (str, optional): 返却する出力パス．
            visible (bool, optional): UI 表示状態．
            jp_msg (str | None, optional): 日本語ログ．
            en_msg (str | None, optional): 英語ログ．

        Returns:
            tuple[str, str, gr.Column]:
                - 出力パス．
                - ログ全文．
                - 表示制御用の gr.Column．
        """
        if jp_msg is not None:
            log(jp_msg, en_msg)
        return path, "\n".join(all_logs), gr.Column(visible=visible)

    def ok(path, jp_msg=None, en_msg=None):
        """
        成功時の返却を行う。

        Args:
            path (str): 出力パス．
            jp_msg (str | None, optional): 日本語ログ．
            en_msg (str | None, optional): 英語ログ．

        Returns:
            tuple[str, str, gr.Column]: 成功時の返却値．
        """
        return finish(path=path, visible=True, jp_msg=jp_msg, en_msg=en_msg)

    def ng(jp_msg, en_msg):
        """
        失敗時の返却を行う．

        Args:
            jp_msg (str): 日本語ログ．
            en_msg (str): 英語ログ．

        Returns:
            tuple[str, str, gr.Column]: 失敗時の返却値．
        """
        return finish(path="", visible=False, jp_msg=jp_msg, en_msg=en_msg)

    def stringify_cmd(cmd):
        """
        コマンドリストを 1 行の文字列に変換する．

        Args:
            cmd (list[str]): コマンド引数のリスト．

        Returns:
            str: ログ記録用のコマンド文字列．
        """
        return subprocess.list2cmdline([str(x) for x in cmd])

    def run_checked(cmd, cwd=None):
        """
        コマンドを実行し，成功可否を返す．

        処理内容:
            - 実行コマンドをログに残す．
            - stdout / stderr をログに残す．
            - 例外発生や returncode != 0 を失敗扱いにする．

        Args:
            cmd (list[str]): 実行コマンド．
            cwd (str | None, optional): 実行時の作業ディレクトリ．

        Returns:
            bool: 成功時は True，失敗時は False．
        """
        cmd = [str(x) for x in cmd]
        cmd_str = stringify_cmd(cmd)

        # 実行コマンドを記録
        all_logs.append(cmd_str)

        # コマンドを実行
        try:
            result = subprocess.run(
                cmd_str if SHELL_FLAG else cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                shell=SHELL_FLAG,
            )
        except Exception as e:
            log(
                f"コマンド実行中に例外が発生しました: {e}",
                f"An exception occurred while running the command: {e}",
            )
            return False

        # stdout をログへ追記
        if result.stdout:
            all_logs.append(result.stdout.strip())

        # stderr を区切って追記
        if result.stderr:
            log("【STDERR】", "[STDERR]")
            all_logs.append(result.stderr.strip())

        # 非 0 終了は失敗扱い
        if result.returncode != 0:
            log(
                f"コマンドが異常終了しました（コード: {result.returncode}）",
                f"The command exited with an error (code: {result.returncode})",
            )
            return False

        return True

    def read_help_text(subcommand):
        """
        `colmap <subcommand> -h` を実行して help テキスト全文を返す．

        Args:
            subcommand (str): COLMAP のサブコマンド名．

        Returns:
            str: help テキスト全文。取得失敗時は空文字．
        """
        try:
            result = subprocess.run(
                ["colmap", subcommand, "-h"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:
            log(
                f"COLMAP ヘルプ取得に失敗しました: {subcommand}: {e}",
                f"Failed to read COLMAP help: {subcommand}: {e}",
            )
            return ""

        return ((result.stdout or "") + "\n" + (result.stderr or "")).strip()

    def detect_colmap_capabilities():
        """
        COLMAP の GPU 関連オプションと CUDA 利用可否を検出する．

        Returns:
            dict[str, str | None]:
                - feature_extractor: feature 用 GPU オプション名または None
                - exhaustive_matcher: matcher 用 GPU オプション名または None
                - slurm_use_gpu: "1" または "0"
        """
        feature_help = read_help_text("feature_extractor")
        matcher_help = read_help_text("exhaustive_matcher")

        feature_gpu_opt = None
        matcher_gpu_opt = None

        # feature_extractor 側の GPU オプション名を判定
        if "--SiftExtraction.use_gpu" in feature_help:
            feature_gpu_opt = "--SiftExtraction.use_gpu"
        elif "--FeatureExtraction.use_gpu" in feature_help:
            feature_gpu_opt = "--FeatureExtraction.use_gpu"

        # exhaustive_matcher 側の GPU オプション名を判定
        if "--SiftMatching.use_gpu" in matcher_help:
            matcher_gpu_opt = "--SiftMatching.use_gpu"
        elif "--FeatureMatching.use_gpu" in matcher_help:
            matcher_gpu_opt = "--FeatureMatching.use_gpu"

        # slurm スクリプトへ渡す GPU 使用フラグ
        use_gpu = "0"

        # GPU オプションが見えていれば CUDA 利用可否を確認
        if feature_gpu_opt is not None or matcher_gpu_opt is not None or "use_gpu" in feature_help:
            try:
                import torch
                use_gpu = "1" if torch.cuda.is_available() else "0"
            except Exception as e:
                log(
                    f"torch の GPU 判定に失敗したため CPU 扱いにします: {e}",
                    f"Failed to detect CUDA via torch, falling back to CPU mode: {e}",
                )

        log(
            f"GPU option detect: feature={feature_gpu_opt}, matcher={matcher_gpu_opt}, use_gpu={use_gpu}",
            f"GPU option detect: feature={feature_gpu_opt}, matcher={matcher_gpu_opt}, use_gpu={use_gpu}",
        )

        return {
            "feature_extractor": feature_gpu_opt,
            "exhaustive_matcher": matcher_gpu_opt,
            "slurm_use_gpu": use_gpu,
        }

    def run_with_fallback(subcommand, gpu_opt):
        """
        feature_extractor / exhaustive_matcher を
        GPU → CPU → オプションなし の順で試行する．

        GPU オプション自体が存在しない COLMAP では
        オプションなしのみを試行する．

        Args:
            subcommand (str): 実行対象の COLMAP サブコマンド．
            gpu_opt (str | None): GPU 使用オプション名．

        Returns:
            bool: いずれかの試行が成功すれば True．
        """
        if subcommand == "feature_extractor":
            base_cmd = [
                "colmap", "feature_extractor",
                "--database_path", db_path,
                "--image_path", input_dir,
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "OPENCV",
            ]
        else:
            base_cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", db_path,
            ]

        if gpu_opt is None:
            trials = [("auto", None)]
            log(
                f"{subcommand}: GPU/CPU オプション未対応のため、オプションなしで実行します．",
                f"{subcommand}: GPU/CPU options are not supported, so running without the option.",
            )
        else:
            trials = [("gpu", "1"), ("cpu", "0"), ("auto", None)]

        try_names_jp = {
            "gpu": "GPUモードを試します．",
            "cpu": "CPUモードを試します．",
            "auto": "オプションなしを試します．",
        }
        try_names_en = {
            "gpu": "Trying GPU mode.",
            "cpu": "Trying CPU mode.",
            "auto": "Trying without the option.",
        }
        mode_names_jp = {
            "gpu": "GPUモード",
            "cpu": "CPUモード",
            "auto": "オプションなし",
        }
        mode_names_en = {
            "gpu": "GPU mode",
            "cpu": "CPU mode",
            "auto": "without the option",
        }

        # 順番に試し、1つ成功したら終了
        for mode, gpu_value in trials:
            log(
                f"{subcommand}: {try_names_jp[mode]}",
                f"{subcommand}: {try_names_en[mode]}",
            )

            cmd = list(base_cmd)

            if gpu_value is not None:
                cmd += [gpu_opt, gpu_value]

            if run_checked(cmd):
                log(
                    f"{subcommand}: {mode_names_jp[mode]}で成功しました．",
                    f"{subcommand}: Succeeded in {mode_names_en[mode]}.",
                )
                return True

            log(
                f"{subcommand}: {mode_names_jp[mode]}で失敗しました．",
                f"{subcommand}: Failed in {mode_names_en[mode]}.",
            )

        return False

    def prepare_directories():
        """
        出力ディレクトリを準備する．

        rebuild=True:
            既存 out_dir があれば削除して作り直す．

        rebuild=False:
            既に out_dir があれば「処理済み」として即返却する．

        Returns:
            None | tuple[str, str, gr.Column]:
                続行可能なら None、早期終了時は返却値．
        """
        if rebuild and os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            log(
                "colmap ディレクトリを削除しました．",
                "Deleted the colmap directory.",
            )

        if not rebuild and os.path.exists(out_dir):
            return ok(
                out_dir,
                "既に COLMAP 処理済みです．",
                "COLMAP processing has already been completed.",
            )

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(distorted_dir, exist_ok=True)
        os.makedirs(mapper_sparse_dir, exist_ok=True)
        return None

    def copy_input_images():
        """
        image_dataset 配下の画像ファイルを input_dir へコピーする．

        Returns:
            None | tuple[str, str, gr.Column]:
                コピー成功時は None、エラー時は返却値．
        """
        copied_count = 0

        for entry in os.scandir(image_dataset):
            if not entry.is_file():
                continue
            if os.path.splitext(entry.name)[1].lower() not in IMAGE_EXTS:
                continue

            shutil.copy2(entry.path, os.path.join(input_dir, entry.name))
            copied_count += 1

        if copied_count == 0:
            return ng(
                f"エラー: 入力画像が見つかりませんでした: {image_dataset}",
                f"Error: No input images were found: {image_dataset}",
            )

        return None

    def normalize_sparse_output():
        """
        image_undistorter 後の sparse 出力を out_dir/sparse/0 形式に正規化する．

        Returns:
            tuple[str | None, None | tuple[str, str, gr.Column]]:
                - final_sparse0_dir: 正規化後の sparse/0 ディレクトリ
                - err: エラー時の返却値
        """
        final_sparse_root = os.path.join(out_dir, "sparse")
        final_sparse0_dir = os.path.join(final_sparse_root, "0")
        os.makedirs(final_sparse0_dir, exist_ok=True)

        # root 直下にある bin を sparse/0/ へ移動
        for name in MODEL_BIN_NAMES:
            src = os.path.join(final_sparse_root, name)
            dst = os.path.join(final_sparse0_dir, name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)

        # 必須ファイルがそろっているか確認
        missing = [
            name for name in MODEL_BIN_NAMES
            if not os.path.exists(os.path.join(final_sparse0_dir, name))
        ]
        if missing:
            return None, ng(
                f"エラー: COLMAP model に必要なファイルが不足しています: {', '.join(missing)}",
                f"Error: Required files are missing in the COLMAP model: {', '.join(missing)}",
            )

        return final_sparse0_dir, None

    def run_local(caps):
        """
        ローカル環境で COLMAP から nerfstudio 変換までを連続実行する．

        Args:
            caps (dict[str, str | None]): GPU capability 情報．

        Returns:
            tuple[str, str, gr.Column]: 実行結果．
        """
        # 1. 特徴点抽出
        if not run_with_fallback("feature_extractor", caps["feature_extractor"]):
            return ng(
                "エラー: feature_extractor に失敗しました．",
                "Error: feature_extractor failed.",
            )

        # 2. 特徴点マッチング
        if not run_with_fallback("exhaustive_matcher", caps["exhaustive_matcher"]):
            return ng(
                "エラー: exhaustive_matcher に失敗しました．",
                "Error: exhaustive_matcher failed.",
            )

        # 3. mapper で sparse model を構築
        if not run_checked([
            "colmap", "mapper",
            "--database_path", db_path,
            "--image_path", input_dir,
            "--output_path", mapper_sparse_dir,
            "--Mapper.ba_global_function_tolerance", "1e-6",
        ]):
            return ng(
                "エラー: mapper に失敗しました．",
                "Error: mapper failed.",
            )

        distorted_sparse0_dir = os.path.join(mapper_sparse_dir, "0")
        if not os.path.isdir(distorted_sparse0_dir):
            return ng(
                f"エラー: sparse model が見つかりません: {distorted_sparse0_dir}",
                f"Error: sparse model was not found: {distorted_sparse0_dir}",
            )

        # 4. undistort 実行
        if not run_checked([
            "colmap", "image_undistorter",
            "--image_path", input_dir,
            "--input_path", distorted_sparse0_dir,
            "--output_path", out_dir,
            "--output_type", "COLMAP",
        ]):
            return ng(
                "エラー: image_undistorter に失敗しました．",
                "Error: image_undistorter failed.",
            )

        log(
            "COLMAP 変換完了．",
            "COLMAP conversion completed.",
        )

        # 5. sparse 出力形式を整える
        final_sparse0_dir, err = normalize_sparse_output()
        if err is not None:
            return err

        # 6. undistorted images ディレクトリ確認
        undistorted_images = os.path.join(out_dir, "images")
        if not os.path.isdir(undistorted_images):
            return ng(
                f"エラー: undistorted images が見つかりません: {undistorted_images}",
                f"Error: undistorted images were not found: {undistorted_images}",
            )

        # 7. nerfstudio 用 multiscale 画像生成
        make_multiscale_images(
            src_dir=undistorted_images,
            out_root=out_dir,
            scales=MULTISCALES,
            exts=MULTISCALE_EXTS,
        )

        # 8. nerfstudio 形式へ変換
        if not run_checked([
            "conda", "run", "--no-capture-output",
            "-n", "nerfstudio",
            "ns-process-data", "images",
            "--data", undistorted_images,
            "--output-dir", out_dir,
            "--skip-colmap",
            "--colmap-model-path", final_sparse0_dir,
            "--skip-image-processing",
            "--camera-type", "perspective",
            "--same-dimensions",
        ], cwd=os.path.join("models", "nerfstudio")):
            return ng(
                "エラー: Nerfstudio データ変換に失敗しました．",
                "Error: Nerfstudio data conversion failed.",
            )

        return ok(
            out_dir,
            "Nerfstudio データ変換完了．",
            "Nerfstudio data conversion completed.",
        )

    def run_slurm(caps):
        """
        SLURM ジョブとして COLMAP 処理を投入する．

        Args:
            caps (dict[str, str | None]): GPU capability 情報．

        Returns:
            tuple[str, str, gr.Column]: 実行結果．
        """
        sbatch_script = os.path.join("scripts", "run_colmap.sh")

        if not run_checked([
            "sbatch",
            sbatch_script,
            input_dir,
            out_dir,
            distorted_dir,
            caps["slurm_use_gpu"],
        ], cwd="./"):
            return ng(
                "エラー: SLURM ジョブ投入に失敗しました．",
                "Error: Failed to submit the SLURM job.",
            )

        return ok(
            out_dir,
            "SLURM ジョブを投入しました．",
            "Submitted the SLURM job.",
        )

    # メインフロー
    try:
        # 1. 出力先準備
        early = prepare_directories()
        if early is not None:
            return early

        # 2. 入力画像コピー
        copied = copy_input_images()
        if copied is not None:
            return copied

        # 3. COLMAP の GPU capability 判定
        caps = detect_colmap_capabilities()

        # 4. 実行モードに応じて処理を分岐
        return run_local(caps) if exe_mode == "local" else run_slurm(caps)

    except Exception as e:
        # 想定外の例外も UI へ返す
        return ng(f"エラー: {e}", f"Error: {e}")


# =========================
# データセットの展開，圧縮
# =========================

def unzip_dataset(lang, zip_file, datasets_parent):
    """
    ZIP ファイルを展開し，データセット内の images / colmap ディレクトリを検出して返す．

    Args:
        lang (str): UI 言語コード．
        zip_file: ZIP ファイル本体，バイト列，または ZIP ファイルパス．
        datasets_parent (str): 展開先の親ディレクトリ．

    Returns:
        tuple[str | None, str | None, str]:
            - images ディレクトリのパス．存在しない場合は None．
            - colmap ディレクトリのパス．存在しない場合は None．
            - 実行結果メッセージ．
    """
    if zip_file is None:
        return None, None, msg(
            lang,
            "❌ ZIP が指定されていません．",
            "❌ No ZIP file was specified.",
        )

    try:
        # ZIP データを読み込む．
        if hasattr(zip_file, "read"):
            data = zip_file.read()
        elif isinstance(zip_file, (bytes, bytearray, memoryview)):
            data = bytes(zip_file)
        elif isinstance(zip_file, str):
            with open(zip_file, "rb") as f:
                data = f.read()
        else:
            return None, None, msg(
                lang,
                f"❌ 想定外の入力型です: {type(zip_file)}",
                f"❌ Unexpected input type: {type(zip_file)}",
            )

        # 一時 ZIP ファイルとして保存する．
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as f:
            f.write(data)
            zip_path = f.name

        try:
            # ZIP として認識できるか確認する．
            if not zipfile.is_zipfile(zip_path):
                with open(zip_path, "rb") as f:
                    print("file head:", f.read(8))
                return None, None, msg(
                    lang,
                    "❌ 指定されたファイルは ZIP として認識できません．",
                    "❌ The specified file could not be recognized as a ZIP file.",
                )

            # 展開先パスを決定する．
            basename = os.path.splitext(
                os.path.basename(getattr(zip_file, "name", "dataset.zip"))
            )[0]
            dataset_path = os.path.join(datasets_parent, basename)
            os.makedirs(dataset_path, exist_ok=True)

            # ZIP を展開する．
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dataset_path)

            # 必要ディレクトリの有無を確認する．
            image_path = os.path.join(dataset_path, "images")
            colmap_path = os.path.join(dataset_path, "colmap")

            has_images = os.path.isdir(image_path)
            has_colmap = os.path.isdir(colmap_path)

            if not has_images and not has_colmap:
                return None, None, msg(
                    lang,
                    "⚠️ ZIP 内に 'images' または 'colmap' ディレクトリが見つかりません．",
                    "⚠️ Neither 'images' nor 'colmap' directory was found in the ZIP.",
                )

            return (
                image_path if has_images else None,
                colmap_path if has_colmap else None,
                msg(lang, f"✅ 解凍しました: {dataset_path}", f"✅ Extracted to: {dataset_path}"),
            )

        finally:
            # 一時 ZIP ファイルを削除する．
            if os.path.exists(zip_path):
                os.remove(zip_path)

    except Exception as e:
        return None, None, msg(
            lang,
            f"❌ 解凍中にエラーが発生しました: {e}",
            f"❌ An error occurred while extracting: {e}",
        )


def zip_dataset(lang, dataset):
    """
    データセット内の images / colmap ディレクトリを ZIP ファイルへ圧縮する．

    Args:
        lang (str): UI 言語コード．
        dataset (str): データセット内のパス．通常は images などの配下パスを想定する．

    Returns:
        str: 作成した ZIP ファイルのパス．

    Raises:
        ValueError: dataset_path がディレクトリでない場合．
    """
    dirname = os.path.dirname(dataset)
    dataset_path = os.path.abspath(dirname)

    if not os.path.isdir(dataset_path):
        raise ValueError(
            msg(
                lang,
                "dataset_path はディレクトリである必要があります．",
                "dataset_path must be a directory.",
            )
        )

    # データセット名を使って ZIP ファイル名を決定する．
    zip_path = dataset_path + ".zip"

    # 含めるディレクトリの存在を確認する．
    include_images = os.path.isdir(os.path.join(dataset_path, "images"))
    include_colmap = os.path.isdir(os.path.join(dataset_path, "colmap"))

    # ZIP を作成する．
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # images ディレクトリがあれば追加する．
        if include_images:
            images_dir = os.path.join(dataset_path, "images")
            for root, _, files in os.walk(images_dir):
                for file in files:
                    full = os.path.join(root, file)
                    arcname = os.path.relpath(full, dataset_path)
                    zipf.write(full, arcname)

        # colmap ディレクトリがあれば追加する．
        if include_colmap:
            colmap_dir = os.path.join(dataset_path, "colmap")
            for root, _, files in os.walk(colmap_dir):
                for file in files:
                    full = os.path.join(root, file)
                    arcname = os.path.relpath(full, dataset_path)
                    zipf.write(full, arcname)

    return zip_path


# =========================
# 再構築結果可視化用 Viewer
# =========================

def viewer(lang, viewer, outmodel, host="127.0.0.1", port=8080):
    """
    3D モデル用 Viewer を起動し，起動確認後にアクセス URL またはエラーメッセージを返す．

    Args:
        lang (str): UI 言語コード．
        viewer (str): 起動する viewer スクリプト名．
        outmodel (str): 表示対象の 3D モデルファイルパス．
        host (str, optional): Viewer を待ち受けるホスト．デフォルトは "127.0.0.1"．
        port (int, optional): Viewer を待ち受けるポート番号．デフォルトは 8080．

    Returns:
        str: 起動した Viewer の URL，またはエラーメッセージ．
    """
    global TMPDIR

    # 3D モデルの存在を確認する．
    if outmodel is None:
        return msg(
            lang,
            "ファイルが指定されていません．",
            "No file has been specified.",
        )

    outmodel = os.path.expanduser(str(outmodel))
    if not os.path.exists(outmodel):
        return msg(
            lang,
            f"ファイルが見つかりません: {outmodel}",
            f"File not found: {outmodel}",
        )

    if not os.path.isfile(outmodel):
        return msg(
            lang,
            f"ファイルではありません: {outmodel}",
            f"Not a file: {outmodel}",
        )

    # Viewer スクリプトの存在を確認する．
    viewer_script = os.path.join("scripts", "viewer", viewer)
    if not os.path.exists(viewer_script):
        return msg(
            lang,
            f"スクリプトが見つかりません: {viewer_script}",
            f"Script not found: {viewer_script}",
        )

    # 実行コマンドを構築する．
    cmd = [
        "python3",
        viewer_script,
        "--input", outmodel,
        "--host", str(host),
        "--port", str(port),
    ]

    # GaussianSplatting 用 viewer では重心を原点へ寄せるオプションを付与する．
    if viewer == "viewer_gaussian.py":
        cmd.append("--center")

    start_time = time.time()

    # ログファイルの保存先を準備する．
    log_dir = os.path.join(TMPDIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    log_path = os.path.join(log_dir, f"{timestamp}.log")

    # コマンドを 1 行の文字列へ整形する．
    cmd_str = " ".join(map(str, cmd))

    # 作業ディレクトリを設定する．
    workdir = "./"

    # Viewer を起動する．
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        log_file = open(log_path, "w", encoding="utf-8")

        header = (
            f"[{msg(lang, 'コマンド', 'COMMAND')}]\n"
            f"{cmd_str}\n"
            f"{'-' * 60}\n"
        )
        log_file.write(header)
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=workdir,
            shell=SHELL_FLAG,
        )
    except Exception as e:
        return msg(
            lang,
            f"viewer の起動に失敗しました: {e}",
            f"Failed to start the viewer: {e}",
        )

    # ソケットによる起動確認に使う接続先を決定する．
    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    # ユーザーへ返す URL 用のホストを決定する．
    display_host = "127.0.0.1" if host == "0.0.0.0" else host

    # 起動確認の待機条件を設定する．
    wait_timeout = 15      # 最大待機時間．
    stable_seconds = 3.0   # 連続接続が必要な秒数．
    start = time.time()    # 待機開始時刻．
    connected_since = None

    # 最大待機時間内で接続確認を繰り返す．
    while time.time() - start < wait_timeout:
        # 子プロセスがすでに終了していれば，起動失敗として扱う．
        if process.poll() is not None:
            log_file.close()
            return msg(
                lang,
                f"viewer は起動しましたが，すぐ終了しました．ログを確認してください: {log_path}",
                f"The viewer started but exited immediately. Please check the log: {log_path}",
            )

        # TCP ソケットで接続確認を行う．
        ok = False
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)

        try:
            ok = (s.connect_ex((check_host, int(port))) == 0)
        finally:
            s.close()

        # 一定時間連続で接続できた場合のみ起動完了とみなす．
        if ok:
            if connected_since is None:
                connected_since = time.time()
            elif time.time() - connected_since >= stable_seconds:
                log_file.flush()
                return f"http://{display_host}:{port}"
        else:
            connected_since = None

        time.sleep(0.2)

    # 起動確認は完了しなかったが，プロセスが動作中なら URL を返す．
    if process.poll() is None:
        log_file.flush()
        return msg(
            lang,
            f"http://{display_host}:{port} (起動確認は未完了)",
            f"http://{display_host}:{port} (startup confirmation not completed)",
        )

    # 接続確認できず，プロセスも終了していた場合は失敗として扱う．
    log_file.close()
    return msg(
        lang,
        f"viewer の起動に失敗しました．ログを確認してください: {log_path}",
        f"Failed to start the viewer. Please check the log: {log_path}",
    )


def viewer_nerfstudio(lang, outdir, method_name, host="127.0.0.1", port=8080):
    """
    拡張 Nerfstudio Viewer を起動し，起動確認後にアクセス URL またはエラーメッセージを返す．

    Args:
        lang (str): UI 言語コード．
        outdir (str): 再構築結果の出力ディレクトリ．
        method_name (str): 手法名．config.yml の探索に用いる．
        host (str, optional): Viewer を待ち受けるホスト．デフォルトは "127.0.0.1"．
        port (int, optional): Viewer を待ち受けるポート番号．デフォルトは 8080．

    Returns:
        str: 起動した Viewer の URL，またはエラーメッセージ．
    """
    global TMPDIR

    # config.yml のパスを構築し，存在を確認する．
    config_path = os.path.join(outdir, "results", method_name, "results", "config.yml")

    if not config_path:
        return msg(
            lang,
            "ファイルが指定されていません．",
            "No file has been specified.",
        )

    if not os.path.exists(config_path):
        return msg(
            lang,
            f"config.yml が見つかりません: {config_path}",
            f"config.yml not found: {config_path}",
        )

    if config_path is None:
        return msg(
            lang,
            "ファイルが指定されていません．",
            "No file has been specified.",
        )

    # Viewer スクリプトの存在を確認する．
    viewer_script = os.path.join("scripts", "viewer", "viewer_nerfstudio.py")
    if not os.path.exists(viewer_script):
        return msg(
            lang,
            f"スクリプトが見つかりません: {viewer_script}",
            f"Script not found: {viewer_script}",
        )

    # 実行コマンドを構築する．
    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", "nerfstudio", "python", viewer_script,
        "--load-config", config_path,
    ]

    start_time = time.time()

    # ログファイルの保存先を準備する．
    log_dir = os.path.join(TMPDIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    log_path = os.path.join(log_dir, f"{timestamp}.log")

    # コマンドを 1 行の文字列へ整形する．
    cmd_str = " ".join(map(str, cmd))

    # 作業ディレクトリを設定する．
    workdir = "./"

    # Viewer を起動する．
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        log_file = open(log_path, "w", encoding="utf-8")
        header = (
            f"[{msg(lang, 'コマンド', 'COMMAND')}]\n"
            f"{cmd_str}\n"
            f"{'-' * 60}\n"
        )
        log_file.write(header)
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=workdir,
            shell=SHELL_FLAG,
        )
    except Exception as e:
        return msg(
            lang,
            f"viewer の起動に失敗しました: {e}",
            f"Failed to start the viewer: {e}",
        )

    # ソケットによる起動確認に使う接続先を決定する．
    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    # ユーザーへ返す URL 用のホストを決定する．
    display_host = "127.0.0.1" if host == "0.0.0.0" else host

    # 起動確認の待機条件を設定する．
    wait_timeout = 15      # 最大待機時間．
    stable_seconds = 3.0   # 連続接続が必要な秒数．
    start = time.time()    # 待機開始時刻．
    connected_since = None

    # 最大待機時間内で接続確認を繰り返す．
    while time.time() - start < wait_timeout:
        # 子プロセスがすでに終了していれば，起動失敗として扱う．
        if process.poll() is not None:
            log_file.close()
            return msg(
                lang,
                f"viewer は起動しましたが，すぐ終了しました．ログを確認してください: {log_path}",
                f"The viewer started but exited immediately. Please check the log: {log_path}",
            )

        # TCP ソケットで接続確認を行う．
        ok = False
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)

        try:
            ok = (s.connect_ex((check_host, int(port))) == 0)
        finally:
            s.close()

        # 一定時間連続で接続できた場合のみ起動完了とみなす．
        if ok:
            if connected_since is None:
                connected_since = time.time()
            elif time.time() - connected_since >= stable_seconds:
                log_file.flush()
                return f"http://{display_host}:{port}"
        else:
            connected_since = None

        time.sleep(0.2)

    # 起動確認は完了しなかったが，プロセスが動作中なら URL を返す．
    if process.poll() is None:
        log_file.flush()
        return msg(
            lang,
            f"http://{display_host}:{port} (起動確認は未完了)",
            f"http://{display_host}:{port} (startup confirmation not completed)",
        )

    # 接続確認できず，プロセスも終了していた場合は失敗として扱う．
    log_file.close()
    return msg(
        lang,
        f"viewer の起動に失敗しました．ログを確認してください: {log_path}",
        f"Failed to start the viewer. Please check the log: {log_path}",
    )


# =========================
# 評価計算
# =========================

def evaluate_all_metrics(lang, method_name, gt_dir, render_dir, output_dir):
    """
    複数の画像評価指標を計算し，ログ，実行時間，集計結果を返す．

    Args:
        lang (str): UI 言語コード．
        method_name (str): 評価対象手法名．
        gt_dir (str): 正解画像ディレクトリ．
        render_dir (str): 生成画像ディレクトリ．
        output_dir (str): 評価結果の保存先ディレクトリ．

    Returns:
        tuple[str, str, str, list[list[object]] | None]:
            - 実行時間．
            - 実行結果ステータス．
            - ログ全文．
            - 集計結果リスト．失敗時は None．
    """
    start_time = time.time()

    log_lines = []       # ログを貯めるリスト．
    returncode = 0       # 実行結果コード．
    summary_list = None  # 評価結果を格納するリスト．

    def add_log(level, jp_text, en_text):
        """
        装飾付きログを追加する．

        Args:
            level (str): ログの種類（INFO，WARN 等）．
            jp_text (str): 日本語メッセージ．
            en_text (str): 英語メッセージ．

        Returns:
            None
        """
        log_lines.append(f"[{level}] {msg(lang, jp_text, en_text)}\n")

    def load_image(path, to_tensor, device):
        """
        画像を RGB で読み込み，Tensor 化して指定デバイスへ転送する．

        Args:
            path (str): 画像ファイルパス．
            to_tensor: PIL 画像を Tensor に変換する関数またはオブジェクト．
            device (str): 転送先デバイス．

        Returns:
            torch.Tensor: バッチ次元付きの画像 Tensor．
        """
        img = Image.open(path).convert("RGB")
        return to_tensor(img).unsqueeze(0).to(device)

    # 評価計算を実行する．
    try:
        # 使用デバイスを決定する．
        device = "cuda" if torch.cuda.is_available() else "cpu"
        add_log("INFO", f"device = {device}", f"device = {device}")

        # PIL 画像を PyTorch Tensor に変換するための前処理を用意する．
        to_tensor = T.ToTensor()
        # LPIPS 指標を計算するモデル（VGG）を初期化する．
        lpips_fn = lpips.LPIPS(net="vgg").to(device)

        # 各画像ごとの評価を行う．
        per_image = []  # 各画像ごとの評価結果を格納する．
        gt_files = sorted(os.listdir(gt_dir))

        for fname in tqdm(
            gt_files,
            desc=msg(lang, "画像ペアを評価中", "Evaluating image pairs"),
            ncols=80,
        ):
            gt_path = os.path.join(gt_dir, fname)
            pred_path = os.path.join(render_dir, fname)

            if not os.path.isfile(gt_path):
                continue

            if not os.path.exists(pred_path):
                add_log("WARN", f"ファイル欠損: {fname}", f"missing file: {fname}")
                continue

            # 画像を読み込む．
            gt = load_image(gt_path, to_tensor, device)
            pred = load_image(pred_path, to_tensor, device)

            # 画像サイズが小さすぎる場合は MS-SSIM を計算しない．
            _, _, h, w = pred.shape
            ms_ssim_val = (
                float("nan")
                if h < 161 or w < 161
                else piq.multi_scale_ssim(pred, gt, data_range=1.0).item()
            )

            # 各評価指標を計算する．
            per_image.append({
                "image": fname,
                "psnr": piq.psnr(pred, gt, data_range=1.0).item(),
                "ssim": piq.ssim(pred, gt, data_range=1.0).item(),
                "ms_ssim": ms_ssim_val,
                "lpips": lpips_fn(pred, gt).item(),
                "fsim": piq.fsim(pred, gt, data_range=1.0).item(),
                "vif": piq.vif_p(pred, gt, data_range=1.0).item(),
                "brisque": getattr(piq, "brisque", lambda x: float("nan"))(pred).item(),
            })
            add_log("INFO", f"評価完了: {fname}", f"evaluated: {fname}")

        # 有効な画像ペアが 1 枚もない場合は例外とする．
        if not per_image:
            raise RuntimeError(
                msg(
                    lang,
                    "有効な画像ペアが見つかりませんでした．",
                    "No valid image pairs found.",
                )
            )

        # 出力時の評価指標順を定義する．
        metric_order = [
            "psnr", "ssim", "ms_ssim", "lpips",
            "fsim", "vif", "brisque", "fid",
        ]

        # 全画像における平均値を計算する．
        summary_dict = {
            key: float(np.mean([m[key] for m in per_image]))
            for key in metric_order
            if key in per_image[0]
        }

        # FID を計算する．
        fid_metrics = calculate_metrics(
            input1=gt_dir,
            input2=render_dir,
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=False,
        )
        summary_dict["fid"] = float(fid_metrics["frechet_inception_distance"])

        # 出力用リストを作成する．
        summary_list = [
            [method_name] + [round(summary_dict.get(key, float("nan")), 3) for key in metric_order]
        ]

        # 評価結果を保存する．
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "metrics_per_image.json"), "w", encoding="utf-8") as f:
            json.dump(per_image, f, indent=2)

        with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        add_log("INFO", "メトリクス評価が正常に完了しました．", "Metric evaluation finished successfully.")

    except Exception as e:
        returncode = 1
        add_log("ERROR", "メトリクス評価に失敗しました．", "Metric evaluation failed.")
        log_lines.append(str(e) + "\n")
        log_lines.append(traceback.format_exc())
        summary_list = None

    end_time = time.time()

    # 実行時間を計算する．
    run_seconds = int(end_time - start_time)
    h, rem = divmod(run_seconds, 3600)
    m, s = divmod(rem, 60)
    run_time = f"{h:02d}:{m:02d}:{s:02d}"

    # 最終ログを結合する．
    full_log = "".join(log_lines)

    # 実行結果ステータスを判定する．
    status = (
        msg(lang, "✅ 成功", "✅ Success")
        if returncode == 0
        else msg(lang, "❌ 失敗", "❌ Failed")
    )

    return run_time, status, full_log, summary_list
