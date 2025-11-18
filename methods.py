import os
import sys
import glob
import json
import time
import shutil
import subprocess
import platform

import gradio as gr

from preprocess import get_imagelist

# subprocessのshellフラグの設定
SHELL_FLAG = platform.system() == "Windows"

def run_subprocess(cmd, workdir):
    """
    subprocess.run を共通化したメソッド
    Args:
        cmd (list[str]): 実行コマンド（リスト形式）
        workdir (str): 実行ディレクトリ
    Returns:
        run_time (str): 実行時間 (HHmmss)
        status (str): 実行ステータス（✅ 成功 / ❌ 失敗(returncode=xx)）
        log (str): コマンド＋標準出力/標準エラーをまとめたログ
    """
    # subprocess実行
    global SHELL_FLAG
    print("Running:", " ".join(map(str, cmd)))
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workdir,
            shell=SHELL_FLAG
        )
    except Exception as e:
        return "0時間0分0秒", "❌ 失敗 (Exception)", f"実行に失敗しました: {e}"
    end_time = time.time()

    # 実行時間の計算
    run_seconds = int(end_time - start_time)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    run_time = f"{hours}時間{minutes}分{seconds}秒"

    # ログの出力
    if result.returncode == 0:
        status = "✅ 成功"
        log = f"{result.stdout.strip()}"
    else:
        status = "❌ 失敗"
        log = f"{result.stderr.strip()}"

    return run_time, status, log
    
# --- ns-train呼び出しメソッド ---
def train_nerfstudio(dataset, outputs_dir, method_name, train_args=None):
    """
    Nerfstudio モデルを学習する関数
    """
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, method_name, name)
    os.makedirs(outdir, exist_ok=True)

    # データセットのパス
    dataset =os.path.join(dataset, "colmap")
    train_cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-train", method_name,
        "--output-dir", outdir,
        "--experiment-name", "results",
        "--timestamp", "results",
        "--vis", "viewer",
        "--viewer.quit-on-train-completion", "True"
    ]
    if train_args:
        train_cmd.extend(train_args)
    train_cmd.extend([
        "nerfstudio-data",
        "--data", dataset,
        "--downscale-factor", "1"
    ])

    workdir = "./"

    runtime, status, log = run_subprocess(train_cmd, workdir)

    return outdir, runtime, status, log, gr.Column(visible=True)

# --- ns-export呼び出しメソッド ---
def export_nerfstudio(dataset, outputs_dir, method_name, filetype, export_args=None):
    """
    Nerfstudio モデルをエクスポートする関数 (学習済みの config.yml 必須)
    """
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, method_name, name)
    config_path = os.path.join(outdir, "results", method_name, "results", "config.yml")

    export_cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-export", filetype,
        "--load-config", config_path,
        "--output-dir", outdir,
        "--normal-method", "open3d",
    ]
    if export_args:
        export_cmd.extend(export_args)

    run_time, success, log = run_subprocess(export_cmd)

    # 出力整理（フラット化）
    results_dir = os.path.join(outdir, "results", method_name, "results")
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            src = os.path.join(results_dir, item)
            dst = os.path.join(outdir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)

        # 空ディレクトリ削除
        nested_dirs = [
            os.path.join(outdir, "results", method_name),
            os.path.join(outdir, "results"),
        ]
        for d in nested_dirs:
            if os.path.exists(d) and not os.listdir(d):
                shutil.rmtree(d)
    return outdir, run_time, success, log

# --- ns-render呼び出しメソッド ---
def render_nerfstudio():
    return

"""
NeRF
"""
# --- 再構築メソッド ---
def recon_nerf(dataset, out_dir, iter):
    train_args = ["--max-num-iterations", f"{iter}",
                  "--viewer.websocket-port-default", "7007"]
    return train_nerfstudio(dataset, out_dir, "vanilla-nerf", train_args)
# --- 点群出力メソッド ---
def export_nerf(dataset, out_dir):
    export_args = ["--rgb-output-name", "rgb_fine", 
                   "--depth-output-name", "depth_fine"]
    return export_nerfstudio(dataset, out_dir, "vanilla-nerf", "pointcloud", export_args)

"""
Nerfacto
"""
# --- 再構築メソッド ---
def recon_nerfacto(dataset, out_dir, iter):
    train_args = ["--max-num-iterations", f"{iter}",
                  "--viewer.websocket-port-default", "7008"]
    return train_nerfstudio(dataset, out_dir, "nerfacto-huge", train_args)
# --- 点群出力メソッド ---
def export_nerfacto(dataset, out_dir):
    export_args = ["--rgb-output-name", "rgb", 
                   "--depth-output-name", "depth"]
    return export_nerfstudio(dataset, out_dir, "nerfacto", "pointcloud", export_args)

"""
mip-NeRF
"""
# --- 再構築メソッド ---
def recon_mipnerf(dataset, out_dir, iter):
    train_args = ["--max-num-iterations", f"{iter}",
                  "--viewer.websocket-port-default", "7009"]
    return train_nerfstudio(dataset, out_dir, "mipnerf", train_args)
# --- 点群出力メソッド ---
def export_mipnerf(dataset, out_dir):
    export_args = ["--rgb-output-name", "rgb_fine", 
                   "--depth-output-name", "depth_fine"]
    return export_nerfstudio(dataset, out_dir, "mipnerf", "pointcloud", export_args)

"""
SeaThru-NeRF
"""
# --- 再構築メソッド ---
def recon_stnerf(dataset, out_dir, iter):
    train_args = ["--max-num-iterations", f"{iter}",
                  "--viewer.websocket-port-default", "7010"]
    return train_nerfstudio(dataset, out_dir, "seathru-nerf", train_args)
# --- 点群出力メソッド ---
def export_stnerf(dataset, out_dir):
    export_args = ["--rgb-output-name", "rgb", 
                   "--depth-output-name", "depth"]
    return export_nerfstudio(dataset, out_dir, "seathru-nerf", "pointcloud", export_args)

"""
3DGS
"""
# --- 再構築メソッド--- 
def recon_3dgs(dataset, outputs_dir, sh_degree, data_device, lambde_dsiim, iterations,
             test_iteraion, save_iteration, 
             feature_lr, opacity_lr, scaling_lr, rotation_lr, position_lr_init,
             position_lr_final, position_lr_delay_mult, densify_from_iter,
             densify_until_iter, densify_grad_threshold, densification_interval,
             opacity_rest_interval, percent_dense):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "3dgs", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 再構築スクリプトパス
    script_path = "train.py"

    # データセットのパス
    dataset =os.path.join(dataset, "colmap")
    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", script_path,
        "--source_path", dataset,
        "--model_path", outdir,
        "--save_iterations", str(save_iteration),
        "--eval"
    ]

    # 実行ディレクトリ
    workdir = os.path.join("models", "gaussian-splatting")

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "point_cloud", f"iteration_{save_iteration}", "point_cloud.ply")

    return outdir, runtime, status, log, model_path, gr.Column(visible=True)

# --- 最大イテレーション数取得メソッド --- 
def _get_latest_iteration(model_path):
    """train/ または test ディレクトリから最新のours_xxxxx を見つけて番号を返す"""
    test_dir = os.path.join(model_path, "test")
    if not os.path.exists(test_dir):
        return None
    ours_dirs = [d for d in os.listdir(test_dir) if d.startswith("ours_")]
    if not ours_dirs:
        return None
    # 数値部分でソートして最新を取得
    latest = sorted(ours_dirs, key=lambda x: int(x.split("_")[1]))[-1]
    return latest.split("_")[1]

# --- レンダリング&評価メソッド ---
def render_eval_3dgs(model_path, skip_train, skip_test, iteration=None):
    # レンダリング
    render_script_path = os.path.join("models", "gaussian-splatting", "render.py")
    render_cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", render_script_path,
        "--model_path", model_path,
    ]
    if skip_train:
        render_cmd.append("--skip_train")
    if skip_test:
        render_cmd.append("--skip_test")
    if iteration is not None:
        render_cmd.extend(["--iteration", str(iteration)])

    print("Running:", " ".join(map(str, render_cmd)))
    render_result = subprocess.run(render_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", shell=SHELL_FLAG)

    if render_result.returncode != 0:
        error_output = render_result.stderr.strip()
        return f"レンダリングに失敗しました\n\nエラー内容:\n{error_output}", []
    
    # 評価
    eval_script_path = os.path.join("models", "gaussian-splatting", "metrics.py")
    eval_cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", eval_script_path,
        "--model_path", model_path,
    ]

    print("Running:", " ".join(map(str, eval_cmd)))
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, shell=SHELL_FLAG)

    if eval_result.returncode != 0:
        error_output = eval_result.stderr.strip()
        return f"レンダリングに失敗しました\n\nエラー内容:\n{error_output}", [], []

    # 評価結果の取得
    results_json = os.path.join(model_path, "results.json")
    values = []
    if os.path.exists(results_json):
        with open(results_json, "r") as f:
            results_data = json.load(f)
        # 最初の行の値だけ取り出す
        first_method = list(results_data.keys())[0]
        metrics = results_data[first_method]
        # PSNR, SSIM, LPIPS の順でリスト化
        values = [[metrics["PSNR"], metrics["SSIM"], metrics["LPIPS"]]]

    # 最新イテレーションを推測
    iter_str = str(iteration) if iteration is not None else _get_latest_iteration(model_path)
    test_dir = os.path.join(model_path, "test", f"ours_{iter_str}")
    gt_dir = os.path.join(test_dir, "gt")
    render_dir = os.path.join(test_dir, "renders")

    if not os.path.exists(render_dir) or not os.path.exists(gt_dir):
        return f"出力ディレクトリが見つかりません: {render_dir} または {gt_dir}", [], []

    # ソートして画像取得
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    render_images = sorted(glob.glob(os.path.join(render_dir, "*.png")))

    # ペアにして gallery 表示用のリストに変換
    gallery_images = []
    for gt_img, render_img in zip(gt_images, render_images):
        gallery_images.append(gt_img)
        gallery_images.append(render_img)
    return "レンダリングに成功しました", values, gallery_images

"""
Mip-Splatting
"""
# --- 再構築メソッド ---
def recon_mipSplatting(dataset, outputs_dir, save_iteration):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "mip-splatting", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 再構築スクリプトパス
    script_path ="train.py"

    # データセットのパス
    dataset =os.path.join(dataset, "colmap")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "mip-splatting", "python", script_path,
        "-s", dataset,
        "-m", outdir,
        "--save_iterations", str(save_iteration)
    ]

    # 実行ディレクトリ
    workdir = os.path.join("models", "mip-splatting")

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "point_cloud", f"iteration_{save_iteration}", "point_cloud.ply")

    return outdir, runtime, status, log, model_path

"""
Splatfacto
"""
# --- 再構築メソッド ---
def recon_sfacto(dataset, out_dir, iter):
    train_args = ["--max-num-iterations", f"{iter}",
                  "--viewer.websocket-port-default", "7011"]
    return train_nerfstudio(dataset, out_dir, "splatfacto-big", train_args)
# --- 点群出力メソッド ---
def export_sfacto(dataset, out_dir):
    export_args = ["--rgb-output-name", "rgb", 
                   "--depth-output-name", "depth"]
    return export_nerfstudio(dataset, out_dir, "splatfacto-big", "gaussian-splat", export_args)

"""
4D-Gaussians
"""
# --- 再構築メソッド ---
def recon_4dGaussians(dataset, outputs_dir, save_iteration):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "4D-Gaussians", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 再構築スクリプトパス
    script_path ="train.py"

    # データセットのパス
    dataset =os.path.join(dataset, "colmap")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "Gaussians4D", "python", script_path,
        "--source_path", dataset,
        "--model_path", outdir,
        "--save_iterations", str(save_iteration)
    ]

    # 実行ディレクトリ
    workdir = os.path.join("models", "4DGaussians")

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "point_cloud", f"iteration_{save_iteration}", "point_cloud.ply")

    return outdir, runtime, status, log, model_path

"""
DUSt3R
"""
# --- 再構築メソッド ---
def recon_dust3r(dataset, outputs_dir, schedule, niter, min_conf_thr, as_pointcloud, mask_sky, 
               clean_depth, transparent_cams, cam_size, scenegraph_type, winsize, refid):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "dust3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 使用モデル
    model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # 変数の定義
    image_size = 512
    device = "cuda"

    # 再構築スクリプト
    script_path = os.path.join("src", "recon_dust3r.py")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "dust3r", "python", script_path,
        "--model_name", model_name,
        "--device", device,
        "--outdir", outdir,
        "--image_size", str(image_size),
        "--filelist", dataset,
        "--schedule", schedule,
        "--niter", str(niter),
        "--min_conf_thr", str(min_conf_thr),
        "--cam_size", str(cam_size),
        "--scenegraph_type", scenegraph_type,
        "--winsize", str(winsize),
        "--refid", str(refid),
    ]
    if as_pointcloud:
        cmd.append("--as_pointcloud")
    if mask_sky:
        cmd.append("--mask_sky")
    if clean_depth:
        cmd.append("--clean_depth")
    if transparent_cams:
        cmd.append("--transparent_cams")

    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")
    # レンダリング画像のパス
    outimgs = os.path.join(outdir, "render")

    return outdir, runtime, status, log, model_path, outimgs

"""
MASt3R
"""
# --- 再構築メソッド ---
def recon_mast3r(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "mast3r", name)
    os.makedirs(outdir, exist_ok=True)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 使用モデル
    model = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    # 個々の画像のパスのリストを作成
    filelist = get_imagelist(dataset)

    # 再構築スクリプトパス
    script_path = os.path.join("src", "recon_mast3r.py")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "mast3r", "python", script_path,
        "--filelist", str(filelist),
        "--outdir", outdir,
        "--model_name", model,
        "--as_pointcloud"
    ]

    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
MonST3R
"""
# --- 再構築メソッド ---
def recon_monst3r(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "mast3r")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 再構築スクリプトパス
    script_path ="demo.py"

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "monst3r", "python", script_path,
        "--input_dir", dataset,
        "--output_dir", outdir,
        "--seq_name", name
    ]

    # 実行ディレクトリ
    workdir = os.path.join("models", "monst3r")

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, name, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
Easi3R
"""
# --- 再構築メソッド ---
def recon_easi3r(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "easi3r")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 再構築スクリプトパス
    script_path ="demo.py"

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "easi3r", "python", script_path,
        "--input_dir", dataset,
        "--output_dir", outdir,
        "--seq_name", name
    ]

    # 実行ディレクトリ
    workdir = os.path.join("models", "Easi3R")

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, name, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
MUSt3R
"""
# --- 再構築メソッド ---
def recon_must3r(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "must3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 再構築スクリプトパス
    script_path ="get_reconstruction.py"

    # 実行コマンド
    cmd = [
        "micromamba", "run", "-n", "must3r", "python", script_path,
        "--image_dir", dataset,
        "--output", outdir,
        "--weights", "ckpt/MUSt3R_512.pth",
        "--retrieval", "ckpt/MUSt3R_512_retrieval_trainingfree.pth",
        "--image_size", "512",
        "--file_type", "glb"
    ]

    # 実行ディレクトリ
    workdir = os.path.join("models", "must3r")

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene_1.05.glb")

    return outdir, runtime, status, log, model_path

"""
Fast3R
"""
# --- 再構築メソッド ---
def recon_fast3r(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "fast3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 再構築スクリプトパス
    script_path = os.path.join("src", "recon_fast3r.py")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "fast3r", "python", script_path,
        "--inpdir", dataset,
        "--outdir", outdir,
    ]

    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
Splatt3R
"""
# --- 再構築メソッド ---
def recon_splatt3r(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "splatt3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 再構築スクリプトパス
    script_path = os.path.join("src", "recon_splatt3r.py")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "splatt3r", "python", script_path,
        "--image1", dataset, 
        "--outdir", outdir
    ]

    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "gaussians.ply") 

    return outdir, runtime, status, log, model_path

"""
CUT3R
"""
# --- 再構築メソッド ---
def recon_cut3r(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "cutt3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 再構築スクリプトパス
    script_path = os.path.join("src", "recon_cut3r.py")

    # 使用モデルパス
    model_path = os.path.join("models", "CUT3R", "src", "cut3r_512_dpt_4_64.pth")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "cut3r", "python", script_path,
        "--inpdir", dataset, 
        "--outdir", outdir,
        "--model_path", model_path,
        "--image_size", "512",
        "--vis_threshold", "1.5",
        "--device", "cuda"
    ]

    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb") 

    return outdir, runtime, status, log, model_path

"""
WinT3R
"""
# --- 再構築メソッド ---
def recon_wint3r(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "wint3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")
    
    # 再構築スクリプトパス
    script_path = "recon.py"

    # checkpointパス
    ckpt_path = os.path.join("checkpoints", "pytorch_model.bin")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "wint3r", "python", script_path,
        "--data_path", dataset, 
        "--save_dir", outdir,
        "--inference_mode", "offline",
        "--ckpt", ckpt_path
    ]

    # 実行ディレクトリ
    workdir = os.path.join("models", "WinT3R")

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "recon.ply") 

    return outdir, runtime, status, log, model_path

"""
MoGe
"""
# --- 再構築メソッド ---
def recon_moge(dataset, outputs_dir, img_type):
    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "moge")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 再構築スクリプトパス
    if img_type=="標準画像":
        script_path = os.path.join("models", "MoGe", "moge", "scripts", "infer.py")
    elif img_type=="パノラマ画像":
        script_path = os.path.join("models", "MoGe", "moge", "scripts", "infer_panorama.py")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "MoGe", "python", script_path,
        "-i", dataset, 
        "-o", outdir,
        "--maps",
        "--glb",
        "--ply"
    ]

    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, name, "mesh.glb") 

    return outdir, runtime, status, log, model_path

"""
UniK3D
"""
# --- 再構築メソッド ---
def recon_unik3d(dataset, outputs_dir):
    # 出力ディレクトリの作成
    outdir = os.path.join(outputs_dir, "unik3d")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 再構築スクリプトパス
    script_path = os.path.join("models", "UniK3D", "scripts", "infer.py")
    # configファイルパス
    config_path = os.path.join("models", "UniK3D", "configs", "eval", "vitl.json")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "UniK3D", "python", script_path,
        "--input", dataset, 
        "--output", outdir,
        "--config-file", config_path,
        "--save",
        "--save-ply"
    ]

    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    #base_name = os.path.splitext(os.path.basename(dataset))[0]
    #model_path = os.path.join(outdir, base_name, "mesh.glb") 

    return outdir, runtime, status, log

"""
VGGT
"""
# --- 再構築メソッド ---
def recon_vggt(dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "vggt", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 再構築スクリプトパス
    script_path = os.path.join("src", "recon_vggt.py")

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "dust3r", "python", script_path,
        "--image-dir", dataset,
        "--out-dir", outdir,
        "--conf-thres", "3.0",
        "--frame-filter", "All",
        "--prediction-mode", "Pointmap Regression",
        "--mode", "crop",
        "--device", "cuda",
        "--show-cam"
    ]
    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path
"""
VGGDSfM
"""
# --- 実行メソッド ---
def recon_vggdsfm():

    cmd = []


    return 