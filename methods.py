import os
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
# 保存先一時ディレクトリ
TMPDIR = ""

# --- subprocess.Popen実行メソッド ---
def run_subprocess_popen(cmd, workdir, log_dir=None):
    global SHELL_FLAG
    global TMPDIR

    print("Running:", " ".join(map(str, cmd)))
    start_time = time.time()

    # ログ保存ディレクトリ
    log_dir = os.path.join(TMPDIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 実行開始時刻（ファイル名用）
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_time))
    log_path = os.path.join(log_dir, f"{timestamp}.log")

    # 実行コマンド文字列
    cmd_str = " ".join(map(str, cmd))

    # ログ内容を保持（戻り値用）
    log_lines = []

    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workdir,
            shell=SHELL_FLAG,
            bufsize=1
        )

        with open(log_path, "w", encoding="utf-8") as log_file:
            # --- ログ先頭：実行コマンド ---
            header = f"[COMMAND]\n{cmd_str}\n{'-'*60}\n"
            log_file.write(header)
            log_file.flush()
            log_lines.append(header)

            # --- stdout を逐次取得 ---
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
                log_lines.append(line)

        returncode = process.wait()

    except Exception as e:
        error_log = f"実行に失敗しました: {e}"
        return "0時間0分0秒", "❌ 失敗 (Exception)", error_log

    end_time = time.time()

    # 実行時間計算
    run_seconds = int(end_time - start_time)
    h, rem = divmod(run_seconds, 3600)
    m, s = divmod(rem, 60)
    run_time = f"{h}時間{m}分{s}秒"

    # ステータス
    if returncode == 0:
        status = "✅ Success"
    else:
        status = "❌ Failed"

    # ログ全文を1つの文字列に
    full_log = "".join(log_lines)

    return run_time, status, full_log
    
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

    # 実行コマンド
    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-train", method_name,
        "--output-dir", outdir,
        "--experiment-name", "results",
        "--timestamp", "results",
        "--vis", "viewer",
        "--viewer.quit-on-train-completion", "True"]
    if train_args:
        cmd.extend(train_args)
    cmd.extend([
        "nerfstudio-data",
        "--data", dataset,
        "--downscale-factor", "1"
    ])

    workdir = "./"

    runtime, status, log = run_subprocess_popen(cmd, workdir)

    return outdir, runtime, status, log, gr.Column(visible=True), gr.Column(visible=True)

# --- ns-train呼び出しメソッド（slurm） ---
def train_nerfstudio_slurm(dataset, outputs_dir, method_name, iter, port):
    """
    Nerfstudio モデルを学習する関数
    """
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, method_name, name)
    os.makedirs(outdir, exist_ok=True)

    # スクリプトパス
    sbatch_script = os.path.join("scripts", "recon_nerfstudio.sh")

    # データセットのパス
    dataset =os.path.join(dataset, "colmap")

    cmd = ["sbatch", f"--job-name={method_name}", sbatch_script, method_name, outdir, str(iter), str(port), dataset]

    workdir = "./"

    runtime, status, log = run_subprocess_popen(cmd, workdir)

    return outdir, runtime, status, log, gr.Column(visible=True)

# --- ns-export呼び出しメソッド ---
def export_nerfstudio(dataset, outputs_dir, method_name1, method_name2, filetype, export_args=None):
    """
    Nerfstudio モデルをエクスポートする関数
    (学習済みの config.yml 必須)
    """
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, method_name1, name)

    config_path = os.path.join(outdir, "results", method_name2, "results", "config.yml")

    export_cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-export", filetype,
        "--load-config", config_path,
        "--output-dir", outdir,
    ]
    if export_args:
        export_cmd.extend(export_args)

    workdir = "./"

    run_time, success, log = run_subprocess_popen(export_cmd, workdir)

    return outdir, run_time, success, log

# --- ns-export呼び出しメソッド ---
def export_nerfstudio_slurm(dataset, outputs_dir, method_name1, method_name2, filetype, export_args=None):
    """
    Nerfstudio モデルをエクスポートする関数
    (学習済みの config.yml 必須)
    """
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, method_name1, name)

    # 学習結果の config.yml（そのまま）
    config_path = os.path.join(
        outdir, "results", method_name2, "results", "config.yml"
    )

    export_cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-export", filetype,
        "--load-config", config_path,
        "--output-dir", outdir,
    ]
    if export_args:
        export_cmd.extend(export_args)

    workdir = "./"

    run_time, success, log = run_subprocess_popen(export_cmd, workdir)

    # -----------------------------
    # 出力整理（<method_name> 配下に集約）
    # -----------------------------
    src_results_dir = os.path.join(outdir, "results", method_name2, "results")
    dst_method_dir = os.path.join(outdir, method_name1)

    if os.path.exists(src_results_dir):
        os.makedirs(dst_method_dir, exist_ok=True)

        for item in os.listdir(src_results_dir):
            src = os.path.join(src_results_dir, item)
            dst = os.path.join(dst_method_dir, item)

            # 既存があれば削除（強制置換）
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)

            shutil.move(src, dst)

        # 空ディレクトリ削除（下から順に）
        cleanup_dirs = [
            os.path.join(outdir, "results", method_name2, "results"),
            os.path.join(outdir, "results", method_name2),
            os.path.join(outdir, "results"),
        ]

        for d in cleanup_dirs:
            if os.path.exists(d) and not os.listdir(d):
                shutil.rmtree(d)

    return outdir, run_time, success, log

# --- ns-eval呼び出しメソッド ---
def render_eval_nerfstudio(dataset, outputs_dir, method_name1, method_name2):
    """
    Nerfstudio モデルを評価する関数
    (学習済みの config.yml 必須)
    """
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, method_name1, name)
    renders = os.path.join(outdir, "renders")
    evals = os.path.join(outdir, "evals.json")
    os.makedirs(renders, exist_ok=True)

    config_path = os.path.join(outdir, "results", method_name2, "results", "config.yml")

    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-eval",
        "--load-config", config_path,
        "--output-path", evals,
        "--render-output-path", renders 
    ]

    workdir = "./"

    run_time, success, log = run_subprocess_popen(cmd, workdir)

    return outdir, run_time, success, log, evals, renders

# --- ns-eval呼び出しメソッド ---
def eval_nerfstudio_slurm(dataset, outputs_dir, method_name1, method_name2, eval_args=None):
    return

"""
Vanilla-NeRF
"""
# --- 再構築メソッド ---
def recon_vnerf(mode, dataset, out_dir, iter):
    port = 7007
    if mode == "local":
        train_args = ["--max-num-iterations", f"{iter}",
                      "--viewer.websocket-port-default", f"{port}"]
        return train_nerfstudio(dataset, out_dir, "vanilla-nerf", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(dataset, out_dir, "vanilla-nerf", iter, port)
    
# --- 点群出力メソッド ---
def export_vnerf(mode, dataset, out_dir):
    if mode == "local":
        export_args = ["--normal-method", "open3d",
                       "--rgb-output-name", "rgb_fine", 
                       "--depth-output-name", "depth_fine"]
        return export_nerfstudio(dataset, out_dir, "vanilla-nerf", "vanilla-nerf", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(dataset, out_dir, "vanilla-nerf")
    
# --- レンダリング&評価指標メソッド ---
def render_eval_vnerf(mode, dataset, out_dir):
    if mode == "local":
        return render_eval_nerfstudio(dataset, out_dir, "vanilla-nerf", "vanilla-nerf")
    elif mode == "slurm":
        return eval_nerfstudio_slurm(dataset, out_dir, "vanilla-nerf")
    
"""
Nerfacto
"""
# --- 再構築メソッド ---
def recon_nerfacto(mode, dataset, out_dir, iter):
    port = 7008
    if mode == "local":
        train_args = ["--max-num-iterations", f"{iter}",
                      "--viewer.websocket-port-default", f"{port}"]
        return train_nerfstudio(dataset, out_dir, "nerfacto-huge", train_args) 
    elif mode == "slurm":
        return train_nerfstudio_slurm(dataset, out_dir, "nerfacto-huge", iter, port)
    
# --- 点群出力メソッド ---
def export_nerfacto(mode, dataset, out_dir):
    if mode == "local":
        export_args = ["--normal-method", "open3d",
                       "--rgb-output-name", "rgb", 
                       "--depth-output-name", "depth"]
        return export_nerfstudio(dataset, out_dir, "nerfacto-huge", "nerfacto", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(dataset, out_dir, "nerfacto")

# --- レンダリング&評価指標メソッド ---
def render_eval_nerfacto(mode, dataset, out_dir):
    if mode == "local":
        return render_eval_nerfstudio(dataset, out_dir, "nerfacto-huge", "nerfacto")
    elif mode == "slurm":
        return eval_nerfstudio_slurm(dataset, out_dir, "nerfacto")

"""
mip-NeRF
"""
# --- 再構築メソッド ---
def recon_mipnerf(mode, dataset, out_dir, iter):
    port = 7009
    if mode == "local":
        train_args = ["--max-num-iterations", f"{iter}",
                      "--viewer.websocket-port-default", f"{port}"]
        return train_nerfstudio(dataset, out_dir, "mipnerf", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(dataset, out_dir, "mipnerf", iter, port)
    
# --- 点群出力メソッド ---
def export_mipnerf(mode, dataset, out_dir):
    if mode == "local":
        export_args = ["--normal-method", "open3d",
                       "--rgb-output-name", "rgb_fine", 
                       "--depth-output-name", "depth_fine"]
        return export_nerfstudio(dataset, out_dir, "mipnerf", "mipnerf", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(dataset, out_dir, "mipnerf")
    
# --- レンダリング&評価指標メソッド ---
def render_eval_mipnerf(mode, dataset, out_dir):
    if mode == "local":
        return render_eval_nerfstudio(dataset, out_dir, "mipnerf", "mipnerf")
    elif mode == "slurm":
        return eval_nerfstudio_slurm(dataset, out_dir, "mipnerf")

"""
SeaThru-NeRF
"""
# --- 再構築メソッド ---
def recon_stnerf(mode, dataset, out_dir, iter):
    port = 7010
    if mode == "local":
        train_args = ["--max-num-iterations", f"{iter}",
                      "--viewer.websocket-port-default", f"{port}"]
        return train_nerfstudio(dataset, out_dir, "seathru-nerf", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(dataset, out_dir, "seathru-nerf", iter, port)

# --- 点群出力メソッド ---
def export_stnerf(mode, dataset, out_dir):
    if mode == "local":
        export_args = ["--normal-method", "open3d",
                       "--rgb-output-name", "rgb", 
                       "--depth-output-name", "depth"]
        return export_nerfstudio(dataset, out_dir, "seathru-nerf", "seathru-nerf", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(dataset, out_dir, "seathru-nerf")
    
# --- レンダリング&評価指標メソッド ---
def render_eval_stnerf(mode, dataset, out_dir):
    if mode == "local":
        return render_eval_nerfstudio(dataset, out_dir, "seathru-nerf", "seathru-nerf")
    elif mode == "slurm":
        return eval_nerfstudio_slurm(dataset, out_dir, "seathru-nerf")

"""
Vanilla-GS
"""
# --- 再構築メソッド--- 
def recon_vgs(mode, dataset, outputs_dir, sh_degree, data_device, lambde_dsiim, iterations,
             test_iteraion, save_iter, 
             feature_lr, opacity_lr, scaling_lr, rotation_lr, position_lr_init,
             position_lr_final, position_lr_delay_mult, densify_from_iter,
             densify_until_iter, densify_grad_threshold, densification_interval,
             opacity_rest_interval, percent_dense):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "3dgs", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset =os.path.join(dataset, "colmap")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = "train.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "gaussian_splatting", "python", recon_script,
            "--source_path", dataset,
            "--model_path", outdir,
            "--iterations", str(save_iter),
            "--save_iterations", str(save_iter),
            "--eval"
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "gaussian-splatting")
    elif mode=="slurm":

        # sbatchスクリプトパス
        sbatch_script = os.path.join("scripts", "recon_vanillags.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "gaussian-splatting", "train.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, save_iter]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "point_cloud", f"iteration_{save_iter}", "point_cloud.ply")

    return outdir, runtime, status, log, model_path, gr.Column(visible=True)

# --- レンダリング&評価メソッド ---
def render_eval_3dgs(model_path, skip_train, skip_test, iteration):
    """
    3DGS のレンダリング & 評価
    """

    workdir = os.path.join("models", "gaussian-splatting")

    # =========================
    # Render
    # =========================
    render_script = "render.py"
    render_cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", render_script,
        "--model_path", model_path,
    ]

    if skip_train:
        render_cmd.append("--skip_train")
    if skip_test:
        render_cmd.append("--skip_test")
    if iteration is not None:
        render_cmd.extend(["--iteration", str(iteration)])

    runtime_r, status_r, log_r = run_subprocess_popen(render_cmd, workdir)

    if status_r != "✅ Success":
        return (
            runtime_r,
            "❌ Failed",
            "レンダリングに失敗しました\n\n" + log_r,
            [],
            [],
        )

    # =========================
    # Evaluation
    # =========================
    eval_script = "metrics.py"
    eval_cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", eval_script,
        "--model_path", model_path,
    ]

    runtime_e, status_e, log_e = run_subprocess_popen(eval_cmd, workdir)

    if status_e != "✅ Success":
        return (
            runtime_r + runtime_e,
            "❌ Failed",
            "評価に失敗しました\n\n" + log_e,
            [],
            [],
        )

    # =========================
    # Load metrics
    # =========================
    results_json = os.path.join(model_path, "results.json")
    values = []

    if os.path.exists(results_json):
        with open(results_json, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        first_method = list(results_data.keys())[0]
        metrics = results_data[first_method]
        values = [[
            metrics.get("PSNR"),
            metrics.get("SSIM"),
            metrics.get("LPIPS"),
        ]]

    # =========================
    # Load images
    # =========================
    test_dir = os.path.join(model_path, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt")
    render_dir = os.path.join(test_dir, "renders")

    if not os.path.exists(render_dir) or not os.path.exists(gt_dir):
        return (
            runtime_r + runtime_e,
            "❌ Failed",
            f"ディレクトリが見つかりません:\n{render_dir}\n{gt_dir}",
            values,
            [],
        )

    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    render_images = sorted(glob.glob(os.path.join(render_dir, "*.png")))

    gallery = []
    for gt_img, render_img in zip(gt_images, render_images):
        gallery.append(gt_img)
        gallery.append(render_img)

    # =========================
    # Summary
    # =========================
    runtime = runtime_r + runtime_e
    status = "✅ Success"
    log = (
        "===== Render =====\n"
        + log_r
        + "\n\n===== Eval =====\n"
        + log_e
    )

    return runtime, status, log, values, gallery

"""
Mip-Splatting
"""
# --- 再構築メソッド ---
def recon_mipSplatting(mode, dataset, outputs_dir, save_iter):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "mip-splatting", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset =os.path.join(dataset, "colmap")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script ="train.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "mip-splatting", "python", recon_script,
            "-s", dataset,
            "-m", outdir,
            "--iterations", str(save_iter),
            "--test_iterations", str(save_iter),
            "--save_iterations", str(save_iter),
            "--eval"
            ]
        
        # 実行ディレクトリ
        workdir = os.path.join("models", "mip-splatting")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_mipsplatting.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "mip-splatting", "train.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, save_iter]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "point_cloud", f"iteration_{save_iter}", "point_cloud.ply")

    return outdir, runtime, status, log, model_path, gr.Column(visible=True)

# --- レンダリング&評価メソッド ---
def render_eval_mips(model_path, skip_train, skip_test, iteration):
    workdir = os.path.join("models", "mip-splatting")

    # =========================
    # Render
    # =========================
    render_script = "render.py"
    render_cmd = [
        "conda", "run", "-n", "mip-splatting", "python", render_script,
        "-m", model_path
    ]

    if skip_train:
        render_cmd.append("--skip_train")
    if skip_test:
        render_cmd.append("--skip_test")
    if iteration is not None:
        render_cmd.extend(["--iteration", str(iteration)])

    runtime_r, status_r, log_r = run_subprocess_popen(render_cmd, workdir)

    if status_r != "✅ Success":
        return (
            runtime_r,
            "❌ Failed",
            "レンダリングに失敗しました\n\n" + log_r,
            [],
            [],
        )

    # =========================
    # Evaluation
    # =========================
    eval_script = "metrics.py"
    eval_cmd = [
        "conda", "run", "-n", "mip-splatting", "python", eval_script,
        "--model_path", model_path,
    ]

    runtime_e, status_e, log_e = run_subprocess_popen(eval_cmd, workdir)

    if status_e != "✅ Success":
        return (
            runtime_r + runtime_e,
            "❌ Failed",
            "評価に失敗しました\n\n" + log_e,
            [],
            [],
        )

    # =========================
    # Load metrics
    # =========================
    results_json = os.path.join(model_path, "results.json")
    values = []

    if os.path.exists(results_json):
        with open(results_json, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        first_method = list(results_data.keys())[0]
        metrics = results_data[first_method]
        values = [[
            metrics.get("PSNR"),
            metrics.get("SSIM"),
            metrics.get("LPIPS"),
        ]]

    # =========================
    # Load images
    # =========================
    test_dir = os.path.join(model_path, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt_-1")
    render_dir = os.path.join(test_dir, "test_preds_-1")

    if not os.path.exists(render_dir) or not os.path.exists(gt_dir):
        return (
            runtime_r + runtime_e,
            "❌ Failed",
            f"ディレクトリが見つかりません:\n{render_dir}\n{gt_dir}",
            values,
            [],
        )

    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    render_images = sorted(glob.glob(os.path.join(render_dir, "*.png")))

    gallery = []
    for gt_img, render_img in zip(gt_images, render_images):
        gallery.append(gt_img)
        gallery.append(render_img)

    # =========================
    # Summary
    # =========================
    runtime = runtime_r + runtime_e
    status = "✅ Success"
    log = (
        "===== Render =====\n"
        + log_r
        + "\n\n===== Eval =====\n"
        + log_e
    )

    return runtime, status, log, values, gallery

"""
Splatfacto
"""
# --- 再構築メソッド ---
def recon_sfacto(mode, dataset, out_dir, iter):
    port = 7011
    if mode == "local":
        train_args = ["--max-num-iterations", f"{iter}",
                      "--viewer.websocket-port-default", f"{port}"]
        return train_nerfstudio(dataset, out_dir, "splatfacto-big", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(dataset, out_dir, "splatfacto-big", iter, port)
    
# --- 点群出力メソッド ---
def export_sfacto(mode, dataset, out_dir):
    if mode == "local":
        export_args = []
        return export_nerfstudio(dataset, out_dir, "splatfacto-big", "splatfacto", "gaussian-splat", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(dataset, out_dir, "splatfacto-big")
    
# --- レンダリング&評価指標メソッド ---
def render_eval_sfacto(mode, dataset, out_dir):
    if mode == "local":
        return render_eval_nerfstudio(dataset, out_dir, "splatfacto-big", "splatfacto")
    elif mode == "slurm":
        return eval_nerfstudio_slurm(dataset, out_dir, "splatfacto-big")

"""
4D-Gaussians
"""
# --- 再構築メソッド ---
def recon_4dGaussians(mode, dataset, outputs_dir, save_iter):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "4D-Gaussians", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset =os.path.join(dataset, "colmap")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script ="train.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "Gaussians4D", "python", recon_script,
            "--source_path", dataset,
            "--model_path", outdir,
            "--iterations", str(save_iter),
            "--test_iterations", str(save_iter),
            "--save_iterations", str(save_iter),
            "--eval"
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "4DGaussians")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_4dgaussians.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "4DGaussians", "train.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, save_iter]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "point_cloud", f"iteration_{save_iter}", "point_cloud.ply")

    return outdir, runtime, status, log, model_path, gr.Column(visible=True)

# --- レンダリング&評価メソッド ---
def render_eval_4dgs(model_path, skip_train, skip_test, iteration):
    """
    3DGS のレンダリング & 評価
    """

    workdir = os.path.join("models", "4DGaussians")

    # =========================
    # Render
    # =========================
    render_script = "render.py"
    render_cmd = [
        "conda", "run", "-n", "Gaussians4D", "python", render_script,
        "-m", model_path,
    ]

    if skip_train:
        render_cmd.append("--skip_train")
    if skip_test:
        render_cmd.append("--skip_test")
    if iteration is not None:
        render_cmd.extend(["--iteration", str(iteration)])

    runtime_r, status_r, log_r = run_subprocess_popen(render_cmd, workdir)

    if status_r != "✅ Success":
        return (
            runtime_r,
            "❌ Failed",
            "レンダリングに失敗しました\n\n" + log_r,
            [],
            [],
        )

    # =========================
    # Evaluation
    # =========================
    eval_script = "metrics.py"
    eval_cmd = [
        "conda", "run", "-n", "Gaussians4D", "python", eval_script,
        "-m", model_path,
    ]

    runtime_e, status_e, log_e = run_subprocess_popen(eval_cmd, workdir)

    if status_e != "✅ Success":
        return (
            runtime_r + runtime_e,
            "❌ Failed",
            "評価に失敗しました\n\n" + log_e,
            [],
            [],
        )

    # =========================
    # Load metrics
    # =========================
    results_json = os.path.join(model_path, "results.json")
    values = []

    if os.path.exists(results_json):
        with open(results_json, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        first_method = next(iter(results_data))
        metrics = results_data[first_method]

        values = [[
            metrics.get("PSNR"),
            metrics.get("SSIM"),
            metrics.get("LPIPS-vgg"),
        ]]

    # =========================
    # Load images
    # =========================
    test_dir = os.path.join(model_path, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt")
    render_dir = os.path.join(test_dir, "renders")

    if not os.path.exists(render_dir) or not os.path.exists(gt_dir):
        return (
            runtime_r + runtime_e,
            "❌ Failed",
            f"ディレクトリが見つかりません:\n{render_dir}\n{gt_dir}",
            values,
            [],
        )

    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    render_images = sorted(glob.glob(os.path.join(render_dir, "*.png")))

    gallery = []
    for gt_img, render_img in zip(gt_images, render_images):
        gallery.append(gt_img)
        gallery.append(render_img)

    # =========================
    # Summary
    # =========================
    runtime = runtime_r + runtime_e
    status = "✅ Success"
    log = (
        "===== Render =====\n"
        + log_r
        + "\n\n===== Eval =====\n"
        + log_e
    )

    return runtime, status, log, values, gallery

"""
DUSt3R
"""
# --- 再構築メソッド ---
def recon_dust3r(mode, dataset, outputs_dir, schedule, niter, min_conf_thr, as_pointcloud, mask_sky, 
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

    if mode=="local":
        # 再構築スクリプト
        recon_script = os.path.join("scripts", "recon_dust3r.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "dust3r", "python", recon_script,
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
    elif mode=="slurm":
        # sbatchスクリプトパス
        sbatch_script = os.path.join("scripts", "recon_dust3r.sh")

        # 再構築スクリプト
        recon_script = os.path.join("scripts", "recon_dust3r.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, model_name, device, outdir, image_size, dataset]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")
    # レンダリング画像のパス
    outimgs = os.path.join(outdir, "render")

    return outdir, runtime, status, log, model_path, outimgs

"""
MASt3R
"""
# --- 再構築メソッド ---
def recon_mast3r(mode, dataset, outputs_dir):
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

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_mast3r.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "mast3r", "python", recon_script,
            "--filelist", str(filelist),
            "--outdir", outdir,
            "--model_name", model,
            "--as_pointcloud"
        ]

        # 実行ディレクトリ
        workdir = "./"
    elif mode=="slurm":
        # sbatchスクリプトパス
        sbatch_script = os.path.join("scripts", "recon_mast3r.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_mast3r.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, filelist, outdir, model]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
MonST3R
"""
# --- 再構築メソッド ---
def recon_monst3r(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "mast3r")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script ="demo.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "monst3r", "python", recon_script,
            "--input_dir", dataset,
            "--output_dir", outdir,
            "--seq_name", name
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "monst3r")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_mast3r.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "monst3r", "demo.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, name]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, name, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
Easi3R
"""
# --- 再構築メソッド ---
def recon_easi3r(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "easi3r")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script ="demo.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "easi3r", "python", recon_script,
            "--input_dir", dataset,
            "--output_dir", outdir,
            "--seq_name", name
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "Easi3R")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_easi3r.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "Easi3R", "demo.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, name]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, name, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
MUSt3R
"""
# --- 再構築メソッド ---
def recon_must3r(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "must3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script ="get_reconstruction.py"

        # 実行コマンド
        cmd = [
            "micromamba", "run", "-n", "must3r", "python", recon_script,
            "--image_dir", dataset,
            "--output", outdir,
            "--weights", "ckpt/MUSt3R_512.pth",
            "--retrieval", "ckpt/MUSt3R_512_retrieval_trainingfree.pth",
            "--image_size", "512",
            "--file_type", "glb"
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "must3r")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_must3r.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "must3r", "get_reconstruction.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene_1.05.glb")

    return outdir, runtime, status, log, model_path

"""
Fast3R
"""
# --- 再構築メソッド ---
def recon_fast3r(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "fast3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_fast3r.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "fast3r", "python", recon_script,
            "--inpdir", dataset,
            "--outdir", outdir,
        ]

        # 実行ディレクトリ
        workdir = "./"
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_fast3r.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_fast3r.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
Splatt3R
"""
# --- 再構築メソッド ---
def recon_splatt3r(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "splatt3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_splatt3r.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "splatt3r", "python", recon_script,
            "--image1", dataset, 
            "--outdir", outdir
        ]

        # 実行ディレクトリ
        workdir = "./"
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_splatt3r.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_splatt3r.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "gaussians.ply") 

    return outdir, runtime, status, log, model_path

"""
CUT3R
"""
# --- 再構築メソッド ---
def recon_cut3r(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "cutt3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # 使用モデルパス
    model_path = os.path.join("models", "CUT3R", "src", "cut3r_512_dpt_4_64.pth")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_cut3r.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "cut3r", "python", recon_script,
            "--inpdir", dataset, 
            "--outdir", outdir,
            "--model_path", model_path,
            "--image_size", "512",
            "--vis_threshold", "1.5",
            "--device", "cuda"
        ]

        # 実行ディレクトリ
        workdir = "./"
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_cut3r.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_cut3r.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, model_path]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb") 

    return outdir, runtime, status, log, model_path

"""
WinT3R
"""
# --- 再構築メソッド ---
def recon_wint3r(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "wint3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットのパス
    dataset = os.path.join(dataset, "images")

    # checkpointパス
    ckpt_path = os.path.join("checkpoints", "pytorch_model.bin")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = "recon.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "wint3r", "python", recon_script,
            "--data_path", dataset, 
            "--save_dir", outdir,
            "--inference_mode", "offline",
            "--ckpt", ckpt_path
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "WinT3R")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_wint3r.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "WinT3R", "recon.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, ckpt_path]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "recon.ply") 

    return outdir, runtime, status, log, model_path

"""
MoGe
"""
# --- 再構築メソッド ---
def recon_moge(mode, dataset, outputs_dir, img_type):
    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "moge")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 再構築スクリプトパス
    if img_type=="標準画像" or img_type=="Standard Image":
        recon_script = os.path.join("models", "MoGe", "moge", "scripts", "infer.py")
    elif img_type=="パノラマ画像" or img_type=="Panorama Image":
        recon_script = os.path.join("models", "MoGe", "moge", "scripts", "infer_panorama.py")

    if mode=="local":
        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "MoGe", "python", recon_script,
            "-i", dataset, 
            "-o", outdir,
            "--maps",
            "--glb",
            "--ply"
        ]

        # 実行ディレクトリ
        workdir = "./"
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_moge.sh")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, name, "mesh.glb") 

    return outdir, runtime, status, log, model_path

"""
UniK3D
"""
# --- 再構築メソッド ---
def recon_unik3d(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "unik3d")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "infer.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "UniK3D", "python", recon_script,
            "--input", dataset, 
            "--output", outdir,
            "--config-file", "configs/eval/vitl.json",
            "--save",
            "--save-ply"
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "UniK3D")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_unik3d.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "UniK3D", "scripts", "infer.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, f"{name}.ply") 

    return outdir, runtime, status, log, model_path

"""
VGGT
"""
# --- 再構築メソッド ---
def recon_vggt(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "vggt", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_vggt.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "vggt", "python", recon_script,
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
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_vggt.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_vggt.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
VGGSfM
"""
# --- 再構築メソッド ---
def recon_vggsfm(mode, dataset):
    outdir = os.path.join(dataset, "sparse")
    if mode=="local":
        # 再構築スクリプトパス
        recon_script = "demo.py"

        # 実行コマンド
        cmd = ["conda", "run", "-n", "vggsfm_tmp", "python", recon_script,
               f"SCENE_DIR={dataset}"]

        # 実行ディレクトリ
        workdir = os.path.join("models", "vggsfm")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_vggsfm.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "vggsfm", "demo.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    return outdir, runtime, status, log, gr.Column(visible=True)            
# --- 点群出力メソッド ---
def export_vggsfm(dataset, outputs_dir): # 軽量なのでlocalのみ
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "vggsfm", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # データセットのパス
    dataset = os.path.join(dataset, "sparse")

    # 出力ファイル
    model_path = os.path.join(outdir, "scene.ply")

    # 実行コマンド
    cmd = ["colmap",
           "model_converter",
           "--input_path", dataset,
           "--output_path", model_path,
           "--output_type", "ply"]
    
    # 実行ディレクトリ
    workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    return outdir, runtime, status, log, model_path


"""
VGGT-SLAM
"""
# --- 再構築メソッド ---
def recon_vggtslam(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "vggt-slam", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # データセットのパス
        dataset = os.path.join(dataset, "images")

        # 再構築スクリプトパス
        recon_script = "main.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "vggt-slam", "python", recon_script,
            "--image_folder", dataset,
            "--vis_map"
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "VGGT-SLAM")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_vggtslam.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "VGGT-SLAM", "main.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = ""

    return outdir, runtime, status, log, model_path

"""
StreamVGGT
"""
# --- 再構築メソッド ---
def recon_stmvggt(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "stmvggt", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_streamvggt.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "streamvggt", "python", recon_script,
            "--input_dir", dataset,
            "--output_dir", outdir,
            "--show_cam"
        ]
        # 実行ディレクトリ
        workdir = "./"
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_streamvggt.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon_streamvggt.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "scene.glb")

    return outdir, runtime, status, log, model_path

"""
FastVGGT
"""
# --- 再構築メソッド ---
def recon_fastvggt(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "fastvggt", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセット
    dataset = os.path.join(dataset, "images")

    # check pointのパス
    ckpt_path = os.path.join("ckpt", "model_tracker_fixed_e20.pt")

    if mode=="local":
    # 再構築スクリプトパス
        recon_script = os.path.join("eval", "eval_custom.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "fastvggt", "python", recon_script,
            "--data_path", dataset,
            "--output_path", outdir,
            "--ckpt_path", ckpt_path,
            "--plot"
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "FastVGGT")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_fastvggt.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "FastVGGT", "eval", "eval_custom.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir, ckpt_path]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "custom_dataset", "reconstructed_points.ply")

    return outdir, runtime, status, log, model_path

"""
Pi3
"""
# --- 再構築メソッド ---
def recon_pi3(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "pi3", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットパス
    dataset = os.path.join(dataset, "images")

    # 出力パス
    outdir = os.path.join(outdir, "recon.ply")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = "example.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "Pi3", "python", recon_script,
            "--data_path", dataset,
            "--save_path", outdir
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "Pi3")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_pi3.sh")

        # 再構築スクリプトパス
        recon_script = os.path.join("models", "Pi3", "example.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, recon_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    model_path = os.path.join(outdir, "recon.ply")

    return outdir, runtime, status, log, model_path