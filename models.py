import os
import glob
import subprocess
import time
import json
import pandas as pd
import gradio as gr

# ---NeRF実行メソッド---
def reconstruct_nerf():
    return 

# ---Nerfacto実行メソッド---
def reconstruct_nerfacto(dataset, outputs_dir):
    # 入力パスの指定
    inpdir = os.path.join(dataset, "ns")
    # 出力先の作成
    outdir = os.path.join(outputs_dir, "nerfacto")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, os.path.basename(dataset))

    # nerfacto学習コマンド
    cmd = [
        "conda", "run", "-n", "nerfstudio", "ns-train", "nerfacto",
        "--data", inpdir,
        "--output-dir", outdir,
        "--timestamp", "results",
        "--pipeline.model.predict-normals", "True"
        "--viewer.quit-on-train-completion", "True"
    ]

    print("Running:", " ".join(cmd))
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    #ログの取得
    #output_log = result.stdout.strip()
    error_output = result.stderr.strip()
    if result.returncode != 0:
        log = f"学習に失敗しました\n\nエラー内容:\n{error_output}"
    log = "学習に成功しました"

    # 点群の取得
    config_path = os.path.join(outdir, os.path.basename(dataset), "nerfacto", "results", "config.yml")
    # 点群出力コマンド
    cmd = [
        "conda", "run", "-n", "nerfstudio", "ns-export", "pointcloud",
        "--loda-config", config_path,
        "--output-dir", outpath
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("点群の生成に失敗しました\n\nエラー内容:\n{error_output}")
    print("点群の生成に成功しました")

    # 出力モデルパス
    model_path = os.path.join(outpath, "point_cloud.ply")

    #学習時間の取得
    run_seconds = int(end_time - start_time)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    run_time = f"{hours}時間{minutes}分{seconds}秒"

    return run_time, log, outpath, model_path

# ---mip-NeRF実行メソッド---
def reconstruct_mipNeRF():
    return 

# ---SeaThru-NeRF実行メソッド---
def reconstruct_seathruNerf():
    return 

# ---3DGS実行メソッド---
def reconstruct_3dgs(dataset, outputs_dir, sh_degree, data_device, lambde_dsiim, iterations,
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time() 

    #ログの取得
    #output_log = result.stdout.strip()
    error_output = result.stderr.strip()
    if result.returncode != 0:
        log = f"学習に失敗しました\n\nエラー内容:\n{error_output}"
    log = "学習に成功しました"

    # 出力モデルパス
    model_path1 = os.path.join(outdir, "point_cloud", f"iteration_{save_iteration1}", "point_cloud.ply")
    model_path2 = os.path.join(outdir, "point_cloud", f"iteration_{save_iteration2}", "point_cloud.ply")

    #学習時間の取得
    run_seconds = int(end_time - start_time)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    run_time = f"{hours}時間{minutes}分{seconds}秒"

    return run_time, log, outdir, model_path1, model_path2, gr.Column(visible=True)

# ---最新のイテレーション数取得メソッド---
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

# ---3DGSレンダリング・評価メソッド---
def eval_3dgs(model_path, skip_train, skip_test, iteration=None):
    # レンダリング
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

    print("Running:", " ".join(map(str, render_cmd)))
    render_result = subprocess.run(render_cmd, capture_output=True, text=True)

    if render_result.returncode != 0:
        error_output = render_result.stderr.strip()
        return f"レンダリングに失敗しました\n\nエラー内容:\n{error_output}", []
    
    # 評価
    eval_script_path = "./models/gaussian-splatting/metrics.py"
    eval_cmd = [
        "conda", "run", "-n", "gaussian_splatting", "python", eval_script_path,
        "--model_path", model_path,
    ]

    print("Running:", " ".join(map(str, eval_cmd)))
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)

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

# ---Mip-Splatting実行メソッド---
def reconstruct_mipSplatting():
    return 

# ---Splatfacto実行メソッド---
def reconstruct_splatfacto():
    return 

# ---4D-Gaussians実行メソッド---
def reconstruct_4dGaussians():
    return 

# ---DUSt3R実行メソッド---
def reconstruct_dust3r(dataset, outputs_dir, schedule, niter, min_conf_thr, as_pointcloud, mask_sky, 
               clean_depth, transparent_cams, cam_size, scenegraph_type, winsize, refid):
    # 出力先の作成
    outdir = os.path.join(outputs_dir, "dust3r", os.path.basename(dataset))
    os.makedirs(outdir, exist_ok=True)
    
    # 1. 変数準備
    model_name = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    image_size = 512
    device = "cuda"

    # 2. subprocess 呼び出し用コマンド組み立て（Conda環境 "dust3r"）
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

    if as_pointcloud:
        cmd.append("--as_pointcloud")
    if mask_sky:
        cmd.append("--mask_sky")
    if clean_depth:
        cmd.append("--clean_depth")
    if transparent_cams:
        cmd.append("--transparent_cams")

    # 実行
    print("Running:", " ".join(cmd))
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    #output_log = result.stdout.strip()
    error_output = result.stderr.strip()
    if result.returncode != 0:
        log = f"学習に失敗しました\n\nエラー内容:\n{error_output}"
    log = "学習に成功しました"

    # 出力の取得
    model_path = os.path.join(outdir, "scene.glb")
    outimgs = os.path.join(outdir, "render")

    # 学習時間の取得
    run_seconds = int(end_time - start_time)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    run_time = f"{hours}時間{minutes}分{seconds}秒"

    return run_time, log, outdir, model_path, outimgs

# ---MASt3R実行メソッド---
def reconstruct_mast3r():
    return 

# ---Easi3R実行メソッド---
def reconstruct_easi3r():
    return 

# ---MUSt3R実行メソッド---
def reconstruct_must3r():
    return 

# ---Fast3R実行メソッド---
def reconstruct_fast3r():
    return 

# ---Splatt3R実行メソッド---
def reconstruct_splatt3r():
    return 

# ---MoGe実行メソッド---
def reconstruct_moge():
    return 

# ---UniK3D実行メソッド---
def reconstruct_unik3d():
    return 

# ---VGGD3D実行メソッド---
def reconstruct_vgg3d():
    return

# ---VGGDSfM実行メソッド---
def reconstruct_vggdsfm():
    return 