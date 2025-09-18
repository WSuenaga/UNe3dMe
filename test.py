import os
import sys
import glob
import shutil
import subprocess
import time
import json
import gradio as gr

# 共通：リアルタイム実行関数
def run_command_realtime(cmd, cwd=None):
    """サブプロセスをリアルタイムで実行してログを標準出力に流す"""
    print("Running:", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )
    output_lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
    process.wait()
    return process.returncode, "".join(output_lines)

# ns-train呼び出しメソッド
def train_nerfstudio(dataset, outputs_dir, method_name, extra_args):
    # COLMAPでの前処理済みのデータセットへのパス
    dataset_path = os.path.join(dataset, "ns")
    # 出力ディレクトリの作成
    outdir = os.path.join(outputs_dir, method_name, os.path.basename(dataset))
    os.makedirs(outdir, exist_ok=True)

    # 共通ns-trainコマンド
    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-train",
        method_name,
        "--output-dir", outdir,
        "--experiment-name", "results",
        "--timestamp", "results",
        "--vis", "viewer",
        "--viewer.quit-on-train-completion", "True"
    ]
    # 追加オプションを適用
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([
        "nerfstudio-data",
        "--data", dataset_path])

    # subbprocess.runを用いてns-trainを実行
    print("Running:", " ".join(cmd))
    start_time = time.time()
    returncode, stderr_output = run_command_realtime(cmd)
    end_time = time.time()

    # 学習時間の算出
    run_seconds = int(end_time - start_time)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    run_time = f"{hours}時間{minutes}分{seconds}秒"

    # 学習結果の出力
    if returncode == 0:
        log = "学習に成功しました"
    else:
        log = f"学習に失敗しました\n\nエラー内容:\n{stderr_output}"

    # 点群の出力
    config_path = os.path.join(outdir, "ns", method_name, "results", "config.yml")
    cmd = [
        "conda", "run", "-n", "nerfstudio",
        "ns-export",
        "pointcloud",
        "--load-config", config_path,
        "--output-dir", outdir
    ]
    run_command_realtime(cmd)

    # nerfstudioのデフォルト出力のフラット化
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

        # 空になった中間ディレクトリを削除
        nested_dirs = [
            os.path.join(outdir, "results", method_name),
            os.path.join(outdir, "results")
        ]
        for d in nested_dirs:
            if os.path.exists(d) and not os.listdir(d):
                shutil.rmtree(d)

    # 3Dモデルへのパス
    model_path = os.path.join(outdir, "point_cloud.ply")
    outimgs = os.path.join(outdir, "render")

    return run_time, log, outdir, model_path

"""
NeRF
"""
# 実行メソッド
def recon_nerf(dataset, out_dir):
    extra_args = []
    return train_nerfstudio(dataset, out_dir, "vanilla-nerf", extra_args)

"""
Nerfacto
"""
# 実行メソッド
def recon_nerfacto(dataset, out_dir):
    extra_args = ["--pipeline.model.predict-normals", "True"]
    return train_nerfstudio(dataset, out_dir, "nerfacto-huge", extra_args)

"""
mip-NeRF
"""
# 実行メソッド
def recon_mipNeRF(dataset, out_dir):
    extra_args = ["--wandb.disable", "True"]
    return train_nerfstudio(dataset, out_dir, "mipnerf", extra_args)

"""
SeaThru-NeRF
"""
# 実行メソッド
def recon_seathruNerf(dataset, out_dir):
    extra_args = []
    return train_nerfstudio(dataset, out_dir, "seathru-nerf", extra_args)

"""
3DGS
"""
# 実行メソッド
def recon_3dgs(dataset, outputs_dir, sh_degree, data_device, lambde_dsiim, iterations,
             test_iteraion1, test_iteration2, save_iteration1, save_iteration2, 
             feature_lr, opacity_lr, scaling_lr, rotation_lr, position_lr_init,
             position_lr_final, position_lr_delay_mult, densify_from_iter,
             densify_until_iter, densify_grad_threshold, densification_interval,
             opacity_rest_interval, percent_dense):
    # 入力パスの指定
    inpdir = os.path.join(dataset, "gs")
    # 出力先の作成
    outdir = os.path.join(outputs_dir, "3dgs", os.path.basename(dataset))
    os.makedirs(outdir, exist_ok=True)

    # 3DGS学習コマンド
    script_path = "./models/gaussian-splatting/train.py"
    cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", script_path,
        "--source_path", inpdir,
        "--model_path", outdir,
        "--eval"
    ]

    print("Running:", " ".join(cmd))
    start_time = time.time()
    returncode, stderr_output = run_command_realtime(cmd)
    end_time = time.time() 

    #ログの取得
    if returncode == 0:
        log = "学習に成功しました"
    else:
        log = f"学習に失敗しました\n\nエラー内容:\n{stderr_output}"

    # 出力モデルパス
    model_path1 = os.path.join(outdir, "point_cloud", f"iteration_{save_iteration1}", "point_cloud.ply")
    model_path2 = os.path.join(outdir, "point_cloud", f"iteration_{save_iteration2}", "point_cloud.ply")

    #学習時間の取得
    run_seconds = int(end_time - start_time)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    run_time = f"{hours}時間{minutes}分{seconds}秒"

    return run_time, log, outdir, model_path1, model_path2, gr.Column(visible=True)

# 最大イテレーション数取得メソッド
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

# レンダリング&評価メソッド
def eval_3dgs(model_path, skip_train, skip_test, iteration=None):
    render_script_path = "./models/gaussian-splatting/render.py"
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

    ret, render_log = run_command_realtime(render_cmd)
    if ret != 0:
        return f"レンダリングに失敗しました\n\nエラー内容:\n{render_log}", []

    eval_script_path = "./models/gaussian-splatting/metrics.py"
    eval_cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", eval_script_path,
        "--model_path", model_path,
    ]
    ret, eval_log = run_command_realtime(eval_cmd)
    if ret != 0:
        return f"評価に失敗しました\n\nエラー内容:\n{eval_log}", [], []

    # 評価結果の取得
    results_json = os.path.join(model_path, "results.json")
    values = []
    if os.path.exists(results_json):
        with open(results_json, "r") as f:
            results_data = json.load(f)
        first_method = list(results_data.keys())[0]
        metrics = results_data[first_method]
        values = [[metrics["PSNR"], metrics["SSIM"], metrics["LPIPS"]]]

    iter_str = str(iteration) if iteration is not None else _get_latest_iteration(model_path)
    test_dir = os.path.join(model_path, "test", f"ours_{iter_str}")
    gt_dir = os.path.join(test_dir, "gt")
    render_dir = os.path.join(test_dir, "renders")

    if not os.path.exists(render_dir) or not os.path.exists(gt_dir):
        return f"出力ディレクトリが見つかりません: {render_dir} または {gt_dir}", [], []

    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    render_images = sorted(glob.glob(os.path.join(render_dir, "*.png")))

    gallery_images = []
    for gt_img, render_img in zip(gt_images, render_images):
        gallery_images.append(gt_img)
        gallery_images.append(render_img)

    return "レンダリングに成功しました", values, gallery_images

"""
Mip-Splatting
"""
# 実行メソッド
def recon_mipSplatting():
    return 

"""
Splatfacto
"""
# 実行メソッド
def recon_splatfacto(dataset, outputs_dir):
    dataset_path = os.path.join(dataset, "ns")
    outdir = os.path.join(outputs_dir, os.path.basename(dataset))
    os.makedirs(outdir, exist_ok=True)

    train_cmd = [
        "conda", "run", "-n", "nerfstudio", "ns-train", "splatfacto-big",
        "--data", dataset_path,
        "--output-dir", outdir,
        "--timestamp", "results",
        "--viewer.quit-on-train-completion", "True"
    ]
    start_time = time.time()
    ret, log_output = run_command_realtime(train_cmd)
    end_time = time.time()

    run_seconds = int(end_time - start_time)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    run_time = f"{hours}時間{minutes}分{seconds}秒"

    if ret == 0:
        log = "学習に成功しました"
    else:
        log = f"学習に失敗しました\n\nエラー内容:\n{log_output}"

    # 点群出力
    config_path = os.path.join(outdir, "ns", "splatfacto-big", "results", "config.yml")
    export_cmd = [
        "conda", "run", "-n", "nerfstudio", "ns-export", "gaussian-splat",
        "--load-config", config_path,
        "--output-dir", outdir
    ]
    run_command_realtime(export_cmd)

    results_dir = os.path.join(outdir, "ns", "gaussian-splat", "results")
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
        nested_dir = os.path.join(outdir, "ns")
        if os.path.exists(nested_dir):
            shutil.rmtree(nested_dir)

    return run_time, log, outdir

"""
4D-Gaussians
"""
# 実行メソッド
def recon_4dGaussians():
    return 

"""
DUSt3R
"""
# 実行メソッド
def recon_dust3r(dataset, outputs_dir, schedule, niter, min_conf_thr, as_pointcloud,
                 mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize, refid):

    outdir = os.path.join(outputs_dir, "dust3r", os.path.basename(dataset))
    os.makedirs(outdir, exist_ok=True)

    model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    image_size = 512
    device = "cuda"

    script_path = "./models/dust3r/reconstruct.py"
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
    if as_pointcloud: cmd.append("--as_pointcloud")
    if mask_sky: cmd.append("--mask_sky")
    if clean_depth: cmd.append("--clean_depth")
    if transparent_cams: cmd.append("--transparent_cams")

    start_time = time.time()
    ret, log_output = run_command_realtime(cmd)
    end_time = time.time()

    if ret == 0:
        log = "学習に成功しました"
    else:
        log = f"学習に失敗しました\n\nエラー内容:\n{log_output}"

    run_seconds = int(end_time - start_time)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    run_time = f"{hours}時間{minutes}分{seconds}秒"

    model_path = os.path.join(outdir, "scene.glb")
    outimgs = os.path.join(outdir, "render")

    return run_time, log, outdir, model_path, outimgs

"""
MASt3R
"""
# 実行メソッド
def recon_mast3r():
    return 

"""
Easi3R
"""
# 実行メソッド
def recon_easi3r():
    return 

"""
MUSt3R
"""
# 実行メソッド
def recon_must3r():
    return 

"""
Fast3R
"""
# 実行メソッド
def recon_fast3r():
    return 

"""
Splatt3R
"""
# 実行メソッド
def recon_splatt3r():
    return 

"""
MoGe
"""
# 実行メソッド
def recon_moge():
    return 

"""
UniK3D
"""
# 実行メソッド
def recon_unik3d():
    return 

"""
VGGD3D
"""
# 実行メソッド
def recon_vgg3d():
    return

"""
VGGDSfM
"""
# 実行メソッド
def recon_vggdsfm():
    return 