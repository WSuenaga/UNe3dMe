import os
import cv2
import time
import shutil
import subprocess
import platform

from local_backend import get_imagelist, evaluate_all_metrics

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
        return "", "❌ Failed (Exception)", error_log

    end_time = time.time()

    # 実行時間計算
    run_seconds = int(end_time - start_time)
    h, rem = divmod(run_seconds, 3600)
    m, s = divmod(rem, 60)
    run_time = f"{h:02d}:{m:02d}:{s:02d}"

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
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, method_name, name)
    os.makedirs(outdir, exist_ok=True)

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

    return outdir, runtime, status, log

# --- ns-train呼び出しメソッド（slurm） ---
def train_nerfstudio_slurm(dataset, outputs_dir, method_name, iter, port):
    """
    Nerfstudio モデルを学習する関数
    """
    dirname = os.path.dirname(dataset)
    name = os.path.basename(dirname)
    outdir = os.path.join(outputs_dir, method_name, name)
    os.makedirs(outdir, exist_ok=True)

    # スクリプトパス
    sbatch_script = os.path.join("scripts", "recon_nerfstudio.sh")

    cmd = ["sbatch", f"--job-name={method_name}", sbatch_script, method_name, outdir, str(iter), str(port), dataset]

    workdir = "./"

    runtime, status, log = run_subprocess_popen(cmd, workdir)

    return outdir, runtime, status, log

# --- ns-export呼び出しメソッド ---
def export_nerfstudio(outdir, method_name, filetype, export_args=None):
    """
    Nerfstudio モデルをエクスポートする関数
    (学習済みの config.yml 必須)
    """
    config_path = os.path.join(outdir, "results", method_name, "results", "config.yml")

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

    outmodel = os.path.join(outdir, "point_cloud.ply")

    return outdir, run_time, success, log, outmodel

# --- ns-export呼び出しメソッド ---
def export_nerfstudio_slurm(outdir, method_name, filetype, export_args=None):
    """
    Nerfstudio モデルをエクスポートする関数
    (学習済みの config.yml 必須)
    """
    config_path = os.path.join(outdir, "results", method_name, "results", "config.yml")

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

# --- ns-eval呼び出しメソッド ---
def render_eval_nerfstudio(outdir, method_name, gt_name, pred_name):
    """
    Nerfstudio モデルを評価する関数
    (学習済みの config.yml 必須)

    想定する ns-render の出力:
    1. 複数枚の場合:
       outdir/test/<name>/*.jpg
    2. 1枚だけの場合:
       outdir/test/<name>.jpg など
       -> outdir/test/<name>/<name>.jpg に移動して統一する
    """
    config_path = os.path.join(outdir, "results", method_name, "results", "config.yml")
    workdir = "./"

    # gt をレンダリング
    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-render", "dataset",
        "--load-config", config_path,
        "--rendered-output-names", gt_name,
        "--output-path", outdir,
    ]
    run_subprocess_popen(cmd, workdir)

    # pred をレンダリング
    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-render", "dataset",
        "--load-config", config_path,
        "--rendered-output-names", pred_name,
        "--output-path", outdir,
    ]
    run_subprocess_popen(cmd, workdir)

    test_dir = os.path.join(outdir, "test")
    exts = (".png", ".jpg", ".jpeg")

    def normalize_render_output(base_dir, name):
        """
        出力を必ず test/<name>/ 配下に統一する。
        戻り値:
            files: 画像ファイルパスのリスト
            dir_path: 評価用ディレクトリ
        """
        dir_path = os.path.join(base_dir, name)

        # 1) すでにディレクトリ形式
        if os.path.isdir(dir_path):
            files = sorted(
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.lower().endswith(exts)
            )
            if files:
                return files, dir_path

        # 2) 単一ファイル形式 -> test/<name>/ に移動
        for ext in exts:
            flat_file = os.path.join(base_dir, f"{name}{ext}")
            if os.path.isfile(flat_file):
                os.makedirs(dir_path, exist_ok=True)

                dst_file = os.path.join(dir_path, os.path.basename(flat_file))

                # すでに同名ファイルがあれば上書きのため削除
                if os.path.exists(dst_file):
                    os.remove(dst_file)

                shutil.move(flat_file, dst_file)
                return [dst_file], dir_path

        # 3) 見つからない
        return [], None

    gt_files, gt_dir = normalize_render_output(test_dir, gt_name)
    pred_files, pred_dir = normalize_render_output(test_dir, pred_name)

    if not gt_files:
        return outdir, "", "失敗", f"GT画像が見つかりません: {gt_name}", [], []
    if not pred_files:
        return outdir, "", "失敗", f"予測画像が見つかりません: {pred_name}", [], []

    if len(gt_files) != len(pred_files):
        full_log = (
            f"GT と予測の画像枚数が一致しません．\n"
            f"{gt_name}: {len(gt_files)} 枚\n"
            f"{pred_name}: {len(pred_files)} 枚"
        )
        return outdir, "", "失敗", full_log, [], []

    # 評価指標の計算
    run_time, status, full_log, summary_list = evaluate_all_metrics(
        method_name,
        gt_dir,
        pred_dir,
        outdir
    )

    # ギャラリーの作成
    gallery = []
    for gt, pred in zip(gt_files, pred_files):
        gallery.append(gt)
        gallery.append(pred)

    return outdir, run_time, status, full_log, summary_list, gallery

# --- ns-eval呼び出しメソッド ---
def render_eval_nerfstudio_slurm(outdir, method_name, gt_name, pred_name):
    """
    Nerfstudio モデルを評価する関数
    (学習済みの config.yml 必須)
    """
    config_path = os.path.join(outdir, "results", method_name, "results", "config.yml")

    workdir = "./"  

    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-render", "dataset",
        "--load-config", config_path,
        "--rendered-output-names", gt_name,
        "--output-path", outdir
    ]
    run_subprocess_popen(cmd, workdir)

    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-render", "dataset",
        "--load-config", config_path,
        "--rendered-output-names", pred_name,
        "--output-path", outdir
    ]
    run_subprocess_popen(cmd, workdir)

    gt_dir = os.path.join(outdir, "test", gt_name)
    pred_dir = os.path.join(outdir, "test", pred_name)

    # 評価指標の計算
    run_time, status, full_log, summary_list = evaluate_all_metrics(method_name, gt_dir, pred_dir, outdir)

    # ファイル名をソートして一致させる
    gt_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    pred_files = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # ギャラリーの作成
    gallery = []
    for gt, pred in zip(gt_files, pred_files):
        gallery.append(gt)
        gallery.append(pred)

    return outdir, run_time, status, full_log, summary_list, gallery

"""
Vanilla-NeRF
"""
# --- 再構築メソッド ---
def recon_vnerf(mode, dataset, outdir, iter):
    port = 7007
    if mode == "local":
        train_args = ["--max-num-iterations", f"{iter}",
                      "--viewer.websocket-port-default", f"{port}"]
        return train_nerfstudio(dataset, outdir, "vanilla-nerf", train_args)
    elif mode == "slurm":
        return train_nerfstudio_slurm(dataset, outdir, "vanilla-nerf", iter, port)
    
# --- 点群出力メソッド ---
def export_vnerf(mode, outdir):
    if mode == "local":
        export_args = ["--normal-method", "open3d",
                       "--rgb-output-name", "rgb_fine", 
                       "--depth-output-name", "depth_fine"]
        return export_nerfstudio(outdir, "vanilla-nerf", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(outdir, "vanilla-nerf", "pointcloud", export_args)
    
# --- レンダリング&評価指標メソッド ---
def render_eval_vnerf(mode, outdir):
    if mode == "local":
        return render_eval_nerfstudio(outdir, "vanilla-nerf", "rgb_fine", "gt-rgb")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(outdir, "vanilla-nerf", "rgb_fine", "gt-rgb")
    
"""
Nerfacto
"""
# --- 再構築メソッド ---
def recon_nerfacto(mode, dataset, outdir, iter):
    port = 7008
    if mode == "local":
        train_args = ["--max-num-iterations", f"{iter}",
                      "--viewer.websocket-port-default", f"{port}"]
        return train_nerfstudio(dataset, outdir, "nerfacto-huge", train_args) 
    elif mode == "slurm":
        return train_nerfstudio_slurm(dataset, outdir, "nerfacto-huge", iter, port)
    
# --- 点群出力メソッド ---
def export_nerfacto(mode, outdir):
    if mode == "local":
        export_args = ["--normal-method", "open3d",
                       "--rgb-output-name", "rgb", 
                       "--depth-output-name", "depth"]
        return export_nerfstudio(outdir, "nerfacto", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(outdir, "nerfacto", "pointcloud", export_args)

# --- レンダリング&評価指標メソッド ---
def render_eval_nerfacto(mode, outdir):
    if mode == "local":
        return render_eval_nerfstudio(outdir, "nerfacto", "rgb", "gt-rgb")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(outdir, "nerfacto", "rgb", "gt-rgb")

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
def export_mipnerf(mode, outdir):
    if mode == "local":
        export_args = ["--normal-method", "open3d",
                       "--rgb-output-name", "rgb_fine", 
                       "--depth-output-name", "depth_fine"]
        return export_nerfstudio(outdir, "mipnerf", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(outdir, "mipnerf", "pointcloud", export_args)
    
# --- レンダリング&評価指標メソッド ---
def render_eval_mipnerf(mode, outdir):
    if mode == "local":
        return render_eval_nerfstudio(outdir, "mipnerf", "rgb_fine", "gt-rgb")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(outdir, "mipnerf", "rgb_fine", "gt-rgb")

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
def export_stnerf(mode, outdir):
    if mode == "local":
        export_args = ["--normal-method", "open3d",
                       "--rgb-output-name", "rgb", 
                       "--depth-output-name", "depth"]
        return export_nerfstudio(outdir, "seathru-nerf", "pointcloud", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(outdir, "seathru-nerf", "pointcloud", export_args)
    
# --- レンダリング&評価指標メソッド ---
def render_eval_stnerf(mode, outdir):
    if mode == "local":
        return render_eval_nerfstudio(outdir, "seathru-nerf", "rgb", "gt-rgb")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(outdir, "seathru-nerf", "rgb", "gt-rgb")

"""
Vanilla-GS
"""
# --- 再構築メソッド--- 
def recon_vgs(mode, dataset, outputs_dir, save_iter):
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "3dgs", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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

    return outdir, runtime, status, log, model_path

# --- レンダリング&評価メソッド ---
def render_eval_3dgs(model_path, skip_train, skip_test, iteration):
    workdir = os.path.join("models", "gaussian-splatting")

    # レンダリング
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

    test_dir = os.path.join(model_path, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt")
    pred_dir = os.path.join(test_dir, "renders")

    # 評価指標の計算
    run_time, status, full_log, summary_list = evaluate_all_metrics("gaussian-splatting", gt_dir, pred_dir, model_path)

    # ファイル名をソートして一致させる
    gt_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    pred_files = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # ギャラリーの作成
    gallery = []
    for gt, pred in zip(gt_files, pred_files):
        gallery.append(gt)
        gallery.append(pred)

    return model_path, run_time, status, full_log, summary_list, gallery

"""
Mip-Splatting
"""
# --- 再構築メソッド ---
def recon_mipSplatting(mode, dataset, outputs_dir, save_iter):
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "mip-splatting", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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

    return outdir, runtime, status, log, model_path

# --- レンダリング&評価メソッド ---
def render_eval_mips(model_path, skip_train, skip_test, iteration):
    workdir = os.path.join("models", "mip-splatting")

    # レンダリング
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

    test_dir = os.path.join(model_path, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt_-1")
    pred_dir = os.path.join(test_dir, "test_preds_-1")

    # 評価指標の計算
    run_time, status, full_log, summary_list = evaluate_all_metrics("mip-splatting", gt_dir, pred_dir, model_path)

    # ファイル名をソートして一致させる
    gt_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    pred_files = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # ギャラリーの作成
    gallery = []
    for gt, pred in zip(gt_files, pred_files):
        gallery.append(gt)
        gallery.append(pred)

    return model_path, run_time, status, full_log, summary_list, gallery

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
def export_sfacto(mode, outdir):
    if mode == "local":
        export_args = []
        return export_nerfstudio(outdir, "splatfacto", "gaussian-splat", export_args)
    elif mode == "slurm":
        return export_nerfstudio_slurm(outdir, "splatfacto", "gaussian-splat", export_args)
    
# --- レンダリング&評価指標メソッド ---
def render_eval_sfacto(mode, outdir):
    if mode == "local":
        return render_eval_nerfstudio(outdir, "splatfacto", "rgb", "gt-rgb")
    elif mode == "slurm":
        return render_eval_nerfstudio_slurm(outdir, "splatfacto", "rgb", "gt-rgb")

"""
4D-Gaussians
"""
# --- 再構築メソッド ---
def recon_4dGaussians(mode, dataset, outputs_dir, save_iter):
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename(os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "4D-Gaussians", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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

    return outdir, runtime, status, log, model_path

# --- レンダリング&評価メソッド ---
def render_eval_4dgs(model_path, skip_train, skip_test, iteration):
    workdir = os.path.join("models", "4DGaussians")

    # レンダリング
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

    test_dir = os.path.join(model_path, "test", f"ours_{iteration}")
    gt_dir = os.path.join(test_dir, "gt")
    pred_dir = os.path.join(test_dir, "renders")

    # 評価指標の計算
    run_time, status, full_log, summary_list = evaluate_all_metrics("4d-gaussians", gt_dir, pred_dir, model_path)

    # ファイル名をソートして一致させる
    gt_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    pred_files = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # ギャラリーの作成
    gallery = []
    for gt, pred in zip(gt_files, pred_files):
        gallery.append(gt)
        gallery.append(pred)

    return model_path, run_time, status, full_log, summary_list, gallery

"""
DUSt3R
"""
# --- 再構築メソッド ---
def recon_dust3r(mode, dataset, outputs_dir, schedule, niter, min_conf_thr, as_pointcloud, mask_sky, 
               clean_depth, transparent_cams, cam_size, scenegraph_type, winsize, refid):
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename( os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "dust3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 使用モデル
    model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # 変数の定義
    image_size = 512
    device = "cuda"

    if mode=="local":
        # 再構築スクリプト
        recon_script = os.path.join("scripts", "recon", "recon_dust3r.py")

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
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename( os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "mast3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 使用モデル
    model = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    # 個々の画像のパスのリストを作成
    filelist = get_imagelist(dataset)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon", "recon_mast3r.py")

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
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename( os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "monst3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename( os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "easi3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename( os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "must3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename( os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "fast3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon", "recon_fast3r.py")

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
        recon_script = os.path.join("scripts", "recon", "recon_splatt3r.py")

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
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename( os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "cutt3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 使用モデルパス
    model_path = os.path.join("models", "CUT3R", "src", "cut3r_512_dpt_4_64.pth")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon", "recon_cut3r.py")

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
    # 入力ディレクトリ
    dataset =os.path.abspath(dataset)
    # 出力ディレクトリの作成
    name = os.path.basename( os.path.dirname(dataset))
    outdir = os.path.join(outputs_dir, "wint3r", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

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
VGGT
"""
# --- 再構築メソッド ---
def recon_vggt(mode, dataset, outputs_dir):
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "vggt", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon", "recon_vggt.py")

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
    dataset = os.path.dirname(dataset)
    # 出力先のパス
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

    return outdir, runtime, status, log
         
# --- 点群出力メソッド ---
def export_vggsfm(dataset, outputs_dir): # 軽量なのでlocalのみ
    # 出力ディレクトリの作成
    dirname = os.path.abspath(dataset)
    name = os.path.basename(os.path.dirname(dirname))
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
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))
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
    model_path = None

    return outdir, runtime, status, log, model_path

"""
StreamVGGT
"""
# --- 再構築メソッド ---
def recon_stmvggt(mode, dataset, outputs_dir):
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "stmvggt", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = os.path.join("scripts", "recon", "recon_streamvggt.py")

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
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))
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
    # 入力ディレクトリ
    dataset = os.path.dirname(os.path.abspath(dataset))
    # 出力ディレクトリの作成
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "pi3", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # データセットパス
    dataset = os.path.join(dataset, "images")

    # 再構築結果のパス
    model_path = os.path.join(outdir, "recon.ply")

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = "example.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "Pi3", "python", recon_script,
            "--data_path", dataset,
            "--save_path", model_path
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

    return outdir, runtime, status, log, model_path

"""
MoGe
"""
# --- 再構築メソッド ---
def recon_moge2(mode, dataset, outputs_dir, img_type):
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
Depth-Anything-V2
"""
# --- 画像推論メソッド ---
def run_image_da2(mode, dataset, outputs_dir, encoder):
    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "Depth-Anything-V2", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        recon_script = "run.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "DA2", "python", recon_script,
            "--img-path", dataset, 
            "--outdir", outdir,
            "--encoder", encoder
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "Depth-Anything-V2")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_da2.sh")

        # 推論スクリプトパス
        infer_script = os.path.join("models", "Depth-Anything-V2", "run.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, infer_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    return outdir, runtime, status, log, outdir

# --- 動画推論メソッド ---
def run_video_da2(mode, dataset, outputs_dir, encoder):
    # 出力ディレクトリの作成
    name = os.path.splitext(os.path.basename(dataset))[0]
    outdir = os.path.join(outputs_dir, "Depth-Anything-V2", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode=="local":
        # 再構築スクリプトパス
        infer_script = "run_video.py"

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "DA2", "python", infer_script,
            "--video-path", dataset, 
            "--outdir", outdir,
            "--encoder", encoder
        ]

        # 実行ディレクトリ
        workdir = os.path.join("models", "Depth-Anything-V2")
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_da2.sh")

        # 推論スクリプトパス
        infer_script = os.path.join("models", "Depth-Anything-V2", "run.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, infer_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 出力動画
    outvideo_path = os.path.join(outdir, f"{name}.mp4")

    return outdir, runtime, status, log, outvideo_path

"""
Depth-Anything-3
"""
def recon_da3(mode, dataset, outputs_dir):
    # 出力ディレクトリの作成
    dataset = os.path.dirname(dataset)
    name = os.path.basename(dataset)
    outdir = os.path.join(outputs_dir, "Depth-Anything-3", name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 内部関数（depth画像 → mp4生成）
    def images_to_video(image_dir, output_path, fps=5):
        if not os.path.exists(image_dir):
            return
        images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not images:
            return
        first = cv2.imread(os.path.join(image_dir, images[0]))
        if first is None:
            return
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for img in images:
            frame = cv2.imread(os.path.join(image_dir, img))
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            out.write(frame)
        out.release()

    if mode=="local":
        # 再構築スクリプトパス
        infer_script = os.path.join("scripts", "recon", "recon_da3.py")

        # 実行コマンド
        cmd = [
            "conda", "run", "-n", "DA3", "python", infer_script,
            "--input_dir", dataset, 
            "--output_dir", outdir,
            "--infer_gs"
        ]

        # 実行ディレクトリ
        workdir = "./"
    elif mode=="slurm":
        # sbatchスクリプト
        sbatch_script = os.path.join("scripts", "recon_da3.sh")

        # 推論スクリプトパス
        infer_script = os.path.join("scripts", "recon_da3.py")

        # 実行コマンド
        cmd = ["sbatch", sbatch_script, infer_script, dataset, outdir]

        # 実行ディレクトリ
        workdir = "./"

    # 推論実行
    runtime, status, log = run_subprocess_popen(cmd, workdir)

    # 再構築結果のパス
    outmodel = os.path.join(outdir, "scene.glb")

    # 出力画像ディレクトリ
    outimages_path = os.path.join(outdir, "depth_vis")

    # depth画像 → mp4生成
    outvideo_path = os.path.join(outdir, "scene.mp4")
    images_to_video(outimages_path, outvideo_path)

    # 出力gs動画ディレクトリ
    outGSvideo_path = os.path.join(outdir, "gs_video", "gs.mp4")

    return outdir, runtime, status, log, outmodel, outimages_path, outvideo_path, outGSvideo_path