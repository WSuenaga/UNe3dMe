import os
import glob
import subprocess
import shutil
import random
import string
import remove_similar_images
import methods
import gradio as gr


# 指定したフォルダ内の画像パスをリスト化するメソッド
def get_imagelist(folder):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    return sorted([f for ext in exts for f in glob.glob(os.path.join(folder, ext))])

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

# ffmpegによるフレーム抽出メソッド
def extract_frames(video, parent_path, fps):
    # 動画名の取得
    video_name = os.path.splitext(os.path.basename(video))[0]
    # 出力ディレクトリの作成(datasets/<動画名>)
    output_path = os.path.join(parent_path, video_name)
    os.makedirs(output_path, exist_ok=True)

    # ffmpegで画像抽出を実行
    command = [
        "ffmpeg",
        "-i", video,
        "-vf", f"fps={fps}",
        os.path.join(output_path, "%04d.png")
    ]
    subprocess.run(command, check=True)

    return output_path

# フレーム抽出およびオプションで画像群の冗長性を無くすメソッド
def extract_frames_with_filter(video, parent_path, fps, remove_similar, ssim_threshold):
    output_path = extract_frames(video, parent_path, fps)
    if remove_similar:
        remove_similar_images.main(output_path, output_path, ssim_threshold)
    imagelist = get_imagelist(output_path)
    print(output_path)
    return output_path, gr.Column(visible=True), output_path, imagelist

# ---nerfstudio_COLMAP実行メソッド---
def run_nscolmap(dataset):
    if dataset == "":
        return "データセットがセットされていません", gr.Column(visible=False)
    
    # nsフォルダを作成
    ns_dir = os.path.join(dataset, "ns")
    os.makedirs(ns_dir, exist_ok=True)

    # COLMAP実行コマンド
    cmd = [
        "conda", "run", "-n", "nerfstudio", "ns-process-data", "images",
        "--data", dataset,
        "--output-dir", ns_dir   # 出力先を ns_dir に変更
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    # 標準出力とエラーを結合
    error_output = result.stderr.strip()

    # returncode が 0 以外ならエラーとして返す
    if result.returncode != 0:
        log = f"前処理に失敗しました\n\nエラー内容:\n{error_output}"
        return log, gr.Column(visible=False)
    log = "前処理が完了しました"

    return log, gr.Column(visible=True)

# Stateの値取得メソッド（データセットタブで得たパスを各タブに渡す）
def get_state_value(state):
    return state
def get_state_values(state):
    return state, state, state, state, state

# メディアUI切り替えメソッド
# 戻り値 {画像入力UI, 動画入力UI}
def display_media_ui(choice):
    if choice == "画像":
        return gr.Column(visible=True), gr.Column(visible=False)
    elif choice == "動画":
        return gr.Column(visible=False), gr.Column(visible=True)

# GradioUI
def main_demo(tmpdir, datasetsdir, outputsdir):


    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        tmpdir_state = gr.State(tmpdir)
        datasetsdir_state = gr.State(datasetsdir)
        outputsdir_state = gr.State(outputsdir)
        dataset = gr.State("")

        outimgs_dust3r = gr.State()

        # データセット作成UI
        with gr.Tab("データセットの作成"):
            gr.Markdown("# 1.ファイルの種類を選択")
            radio = gr.Radio(["画像","動画"], label = "ファイルの種類を選択してください")

            # 画像入力UI
            with gr.Column(visible=False) as image_col:
                gr.Markdown("# 2.データセットの作成")
                gr.Markdown("""
                            ## 2.1.画像を入力してください
                            画像以外も選択できるため注意．
                            """)
                images = gr.File(file_count="multiple")
                gr.Markdown("""
                            ## 2.2.任意のデータセットの名前を入力してください
                            入力が無い場合，ランダムな名前がデータセットに付与されます．
                            """)
                name = gr.Textbox(label="データセットの名前")
                run_copy_btn = gr.Button("データセット作成")
                with gr.Column(visible=False) as iresult_col:
                    gr.Markdown("# 3.データセットが作成されました")
                    output_image = gr.Textbox(label="データセット保存先")
                    gallery_image = gr.Gallery(label="入力された画像", columns=4, height="auto")

            # 動画入力UI
            with gr. Column(visible=False) as video_col:
                gr.Markdown("# 2.動画を入力してください")
                video = gr.Video(label="推論に使用する動画を選択してください.")
                with gr.Row(equal_height=True):
                    fps = gr.Slider(value=3, minimum=1, maximum=5, step=1, label="1秒間に切り出すフレーム数")
                    rsi = gr.Checkbox(value=True, label="重複画像を削除する")
                    ssim = gr.Slider(value=0.8, minimum=0, maximum=1, label="SSIMの閾値（値が小さいほどデータセットは圧縮されます）")
                run_ffmpeg_btn = gr.Button("データセット作成")
                with gr.Column(visible=False) as vresult_col:
                    gr.Markdown("# 3.データセットが作成されました")
                    output_video = gr.Textbox(label="データセット保存先")
                    gallery_vide = gr.Gallery(label="抽出された画像", columns=4, height="auto")
        # NeRF系
        with gr.Tab("NeRF"):
            current_dataset_nerf = gr.Textbox(label="現在セットされているデータセット")
            # NeRF
            with gr.Tab("NeRF"):
                gr.Markdown("# 1. 前処理")
                gr.Markdown("データセットをNeRFで扱えるように前処理を行う必要があります．")
                run_nscolmap_btn1 = gr.Button("前処理実行")
                result_nscolmap1 = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as train_nerf_col:
                    gr.Markdown("# 2.学習")
                    with gr.Accordion("オプション", open=False):
                        gr.Markdown()
                    recon_nerf_btn = gr.Button("学習実行")
                    run_time_nerf = gr.Textbox(label="実行時間")
                    result_recon_nerf = gr.Textbox(label="実行結果")
                    output_recon_nerf = gr.Textbox(label="学習結果保存先")
                    outmodel_nerf = gr.Model3D(label="三次元再構築結果")
                    gallery_nerfacto = gr.Gallery(label="入力画像・深度マップ・信頼度マップ", columns=3, height="auto")
            
            # Nerfacto
            with gr.Tab("Nerfacto"):
                gr.Markdown("# 1. 前処理")
                gr.Markdown("データセットをNerfactoで扱えるように前処理を行う必要があります．")
                run_nscolmap_btn2 = gr.Button("前処理実行")
                result_nscolmap2 = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as train_nerfacto_col:
                    gr.Markdown("# 2.学習")
                    with gr.Accordion("オプション", open=False):
                        gr.Markdown()
                    recon_nerfacto_btn = gr.Button("学習実行")
                    run_time_nerfacto = gr.Textbox(label="実行時間")
                    result_recon_nerfacto = gr.Textbox(label="実行結果")
                    output_recon_nerfacto = gr.Textbox(label="学習結果保存先")
                    outmodel_nerfacto = gr.Model3D(label="三次元再構築結果")
                    gallery_nerfacto = gr.Gallery(label="入力画像・深度マップ・信頼度マップ", columns=3, height="auto")
            
            # mip-NeRF
            with gr.Tab("mip-NeRF"):
                gr.Markdown("# 1. 前処理")
                gr.Markdown("データセットをmip-NeRFで扱えるように前処理を行う必要があります．")
                run_nscolmap_btn3 = gr.Button("前処理実行")
                result_nscolmap3 = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as train_mipnerf_col:
                    gr.Markdown("# 2.学習")
                    with gr.Accordion("オプション", open=False):
                        gr.Markdown()
                    recon_mipnerf_btn = gr.Button("学習実行")
                    run_time_mipnerf = gr.Textbox(label="実行時間")
                    result_recon_mipnerf = gr.Textbox(label="実行結果")
                    output_recon_mipnerf = gr.Textbox(label="学習結果保存先")
                    outmodel_mipnerf = gr.Model3D(label="三次元再構築結果")
                    gallery_mipnerf = gr.Gallery(label="入力画像・深度マップ・信頼度マップ", columns=3, height="auto")
            
            # SeaThru-NeRF
            with gr.Tab("SeaThru-NeRF"):
                gr.Markdown("# 1. 前処理")
                gr.Markdown("データセットをSeaThru-NeRFで扱えるように前処理を行う必要があります．")
                run_nscolmap_btn4 = gr.Button("前処理実行")
                result_nscolmap4 = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as train_stnerf_col:
                    gr.Markdown("# 2.学習")
                    with gr.Accordion("オプション", open=False):
                        gr.Markdown()
                    recon_stnerf_btn = gr.Button("学習実行")
                    run_time_stnerf = gr.Textbox(label="実行時間")
                    result_recon_stnef = gr.Textbox(label="実行結果")
                    output_recon_stnerf = gr.Textbox(label="学習結果保存先")
                    outmodel_stnerf = gr.Model3D(label="三次元再構築結果")
                    gallery_stnerf = gr.Gallery(label="入力画像・深度マップ・信頼度マップ", columns=3, height="auto")

        # GaussianSplatting系            
        with gr.Tab("GS"):
            current_dataset_gs = gr.Textbox(label="現在セットされているデータセット")
            # 3DGS
            with gr.Tab("3DGS"):
                gr.Markdown("# 1. 前処理")
                gr.Markdown("データセットを3DGSで扱えるように前処理を行う必要があります．")
                run_gscolmap_btn = gr.Button("前処理実行")
                result_gscolmap = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as train_3dgs_col:
                    gr.Markdown("# 2. 学習")
                    with gr.Accordion("オプション", open=False):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("モデル・実行環境の設定")
                                sh_degree = gr.Slider(value=3, minimum=1, maximum=3, step=1, label="球面調和関数の次数")
                                data_device = gr.Radio(choices=["cuda", "cpu"], value="cuda", label="画像データの配置場所", info="cpuにすることでVRAMを節約できるが，学習時間が僅かに遅くなる．")
                                gr.Markdown("損失関数の設定")
                                lambda_dssim = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.01, label="DSSIM損失の重み．0ならL1損失，1ならSSIM損失のみ．")
                            with gr.Column():
                                gr.Markdown("学習スケジュールの設定")
                                iter_3dgs = gr.Slider(label="総イテレーション数")
                                with gr.Column():
                                    test_iter1_3dgs = gr.Slider(value=7000, minimum=0, maximum=50000, step=100, label="テストを実行するイテレーション数（1回目）")
                                    test_iter2_3dgs = gr.Slider(value=30000, minimum=0, maximum=50000, step=100, label="テストを実行するイテレーション数（2回目）")
                                with gr.Column():    
                                    save_iter1_3dgs = gr.Slider(value=7000, minimum=0, maximum=50000, step=100, label="モデルを保存するイテレーション数（1回目）")
                                    save_iter2_3dgs = gr.Slider(value=30000, minimum=0, maximum=50000, step=100, label="モデルを保存するイテレーション数（2回目）")
                            with gr.Column():
                                gr.Markdown("学習率の設定")
                                feature_lr = gr.Slider(value=0.0025, minimum=0, maximum=1, step=0.0001, label="球面調和関数の学習率")
                                opacity_lr = gr.Slider(value=0.05, minimum=0, maximum=1, step=0.001, label="不透明度の学習率")
                                scaling_lr = gr.Slider(value=0.005, minimum=0, maximum=1, step=0.001, label="scalingの学習率")
                                rotation_lr = gr.Slider(value=0.001, minimum=0, maximum=1, step=0.0001, label="rotationの学習率")
                                position_lr_init = gr.Slider(value=0.00016, minimum=0, maximum=1, step=0.00001, label="positionの初期学習率")
                                position_lr_final = gr.Slider(value=0.0000016, minimum=0, maximum=1, step=0.0000001, label="positionの最終学習率")
                                position_lr_delay_mult = gr.Slider(value=0.01, minimum=0, maximum=1, step=0.01, label="position学習率乗数")
                            with gr.Column():
                                gr.Markdown("3D gaussianのDensificationの設定")
                                densify_from_iter = gr.Slider(value=500, minimum=0, maximum=50000, step=100, label="Densificationを開始するイテレーション数")
                                densify_until_iter = gr.Slider(value=15000, minimum=0, maximum=50000, step=100, label="Densificationを終了するイテレーション数")
                                densify_grad_threshold = gr.Slider(value=0.0002, minimum=0, maximum=1, step=0.00001, label="Densificationの対象とする2D位置勾配の値の閾値（この値以上の時を対象）")
                                densification_interval = gr.Slider(value=100, minimum=0, maximum=10000, step=100, label="Densificationを行う間隔")
                                opacity_rest_interval = gr.Slider(value=3000, minimum=0, maximum=10000, step=100, label="不透明度リセットの間隔")
                                percent_dense = gr.Slider(value=0.01, minimum=0, maximum=1, step=0.001, label="シーンの大きさに対する比率．この値を超える3D gaussianは強制的にDensificationを行う．")
                    recon_3dgs_btn = gr.Button("学習実行")
                    run_time_3dgs = gr.Textbox(label="実行時間")
                    result_recon_3dgs = gr.Textbox(label="実行結果")
                    output_recon_3dgs = gr.Textbox(interactive=False, label="学習結果保存先")
                    outmodel1_3dgs = gr.Model3D("1回目のセーブポイント")
                    outmodel2_3dgs = gr.Model3D("2回目のセーブポイント")
                with gr.Column(visible=False) as render_3dgs_col:            
                    gr.Markdown("# 3.レンダリング・評価")
                    with gr.Row():
                        skip_train = gr.Checkbox(value=True, label="学習画像のレンダリングを行わない")
                        skip_test = gr.Checkbox(value=False, label="テスト画像のレンダリングを行わない")
                    eval_3dgs_btn = gr.Button("レンダリング実行")
                    result_render_3dgs = gr.Textbox(label="実行結果")
                    eval_3dgs = gr.DataFrame(headers=["PSNR", "SSIM", "LPIPS"], label="評価指標")
                    gallery_3dgs = gr.Gallery(label="Ground Truth・レンダリングされた画像", columns=2, height="auto")

            with gr.Tab("Mip-Splatting"):
                gr.Markdown
            with gr.Tab("Splatfacto"):
                gr.Markdown()
            with gr.Tab("4D-Gaussians"):
                gr.Markdown()

        with gr.Tab("3sters"):
            current_dataset_3sters = gr.Textbox(label="現在セットされているデータセット")
            # DUSt3R
            with gr.Tab("DUSt3R"):
                gr.Markdown("# 1.推論")
                with gr.Accordion("オプション", open=False):
                    with gr.Row():
                        schedule = gr.Dropdown(["linear", "cosine"], value='linear', label="schedule")
                        niter = gr.Number(value=300, label="num_iterations", precision=0)
                        scenegraph_type = gr.Dropdown(["complete", "swin", "oneref"], value='complete', label="Scenegraph")
                        winsize = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Window size")
                        refid = gr.Slider(minimum=0, maximum=5, value=0, step=1, label="Ref ID")

                    with gr.Row():
                        min_conf_thr = gr.Slider(minimum=1.0, maximum=20, step=0.1, value=3.0, label="min_conf_thr")
                        cam_size = gr.Slider(minimum=0.001, maximum=0.1, step=0.001, value=0.05, label="cam_size")

                    with gr.Row():
                        as_pointcloud = gr.Checkbox(label="As pointcloud", value=False)
                        mask_sky = gr.Checkbox(label="Mask sky", value=False)
                        clean_depth = gr.Checkbox(label="Clean depth", value=True)
                        transparent_cams = gr.Checkbox(label="Transparent cameras", value=False)


                recon_dust3r_btn = gr.Button("推論実行")
                run_time_dust3r = gr.Textbox(label="実行時間")
                result_recon_dust3r = gr.Textbox(label="実行結果")
                output_recon_dust3r = gr.Textbox(label="推論結果保存先")
                outmodel_dust3r = gr.Model3D(label="三次元再構築結果")
                gallery_dust3r = gr.Gallery(label="入力画像・深度マップ・信頼度マップ", columns=3, height="auto")

            with gr.Tab("MASt3R"):
                gr.Markdown()
            with gr.Tab("MonST3R"):
                gr.Markdown()
            with gr.Tab("Easi3R"):
                gr.Markdown()
            with gr.Tab("MUSt3R"):
                gr.Markdown()
            with gr.Tab("Fast3R"):
                gr.Markdown()
            with gr.Tab("Splatt3R"):
                gr.Markdown()

        with gr.Tab("mds"):
            current_dataset_mds = gr.Textbox(label="現在セットされているデータセット")
            with gr.Tab("MoGe"):
                gr.Markdown()
            with gr.Tab("UniK3D"):
                gr.Markdown()
        
        with gr.Tab("VGG"):
            current_dataset_vgg = gr.Textbox(label="現在セットされているデータセット")
            with gr.Tab("VGGT"):
                gr.Markdown()
            with gr.Tab("VGGSfM"):
                gr.Markdown()

        #イベントリスナ
        radio.change(fn=display_media_ui, 
                     inputs=radio, 
                     outputs=[image_col, video_col])
        run_copy_btn.click(fn=copy_images,
                       inputs=[images, datasetsdir_state, name],
                       outputs=[dataset, iresult_col, output_image, gallery_image]
                       ).success(
                           fn=get_state_values, 
                           inputs=dataset, 
                           outputs=[current_dataset_nerf, 
                                    current_dataset_gs,
                                    current_dataset_3sters,
                                    current_dataset_mds,
                                    current_dataset_vgg])
        run_ffmpeg_btn.click(fn=extract_frames_with_filter, 
                         inputs=[video, datasetsdir_state, fps, rsi, ssim], 
                         outputs=[dataset, vresult_col, output_video, gallery_vide]
                        ).success(
                            fn=get_state_values, 
                            inputs=dataset, 
                            outputs=[current_dataset_nerf, 
                                     current_dataset_gs,
                                     current_dataset_3sters,
                                     current_dataset_mds,
                                     current_dataset_vgg])
        run_nscolmap_btn1.click(fn=run_nscolmap,
                                inputs=dataset,
                                outputs=[result_nscolmap1, train_nerf_col])
        run_nscolmap_btn2.click(fn=run_nscolmap,
                                inputs=dataset,
                                outputs=[result_nscolmap2, train_nerfacto_col])
        run_nscolmap_btn3.click(fn=run_nscolmap,
                                inputs=dataset,
                                outputs=[result_nscolmap3, train_mipnerf_col])
        run_nscolmap_btn4.click(fn=run_nscolmap,
                                inputs=dataset,
                                outputs=[result_nscolmap4, train_stnerf_col])
        run_gscolmap_btn.click(fn=run_gscolmap,
                               inputs=dataset,
                               outputs=[result_gscolmap, train_3dgs_col])
        recon_nerf_btn.click(fn=methods.recon_nerf,
                             inputs=[dataset, outputsdir_state],
                             outputs=[run_time_nerf, result_recon_nerf, output_recon_nerf, outmodel_nerf])
        recon_nerfacto_btn.click(fn=methods.recon_nerfacto,
                                 inputs=[dataset, outputsdir_state],
                                 outputs=[run_time_nerfacto, result_recon_nerfacto, output_recon_nerfacto, outmodel_nerfacto])
        recon_mipnerf_btn.click(fn=methods.recon_mipNeRF,
                             inputs=[dataset, outputsdir_state],
                             outputs=[run_time_mipnerf, result_recon_mipnerf, output_recon_mipnerf, outmodel_mipnerf])
        recon_stnerf_btn.click(fn=methods.recon_seathruNerf,
                             inputs=[dataset, outputsdir_state],
                             outputs=[run_time_stnerf, result_recon_stnef, output_recon_stnerf, outmodel_stnerf])
        recon_3dgs_btn.click(fn=methods.recon_3dgs, 
                             inputs=[dataset, outputsdir_state, sh_degree, data_device, lambda_dssim, iter_3dgs,
                                     test_iter1_3dgs, test_iter2_3dgs, save_iter1_3dgs, save_iter2_3dgs, feature_lr,
                                     opacity_lr, scaling_lr, rotation_lr, position_lr_init, position_lr_final,
                                     position_lr_delay_mult, densify_from_iter, densify_until_iter, densify_grad_threshold,
                                     densification_interval, opacity_rest_interval, percent_dense], 
                                     outputs=[run_time_3dgs, result_recon_3dgs, output_recon_3dgs, outmodel1_3dgs, outmodel2_3dgs, render_3dgs_col ])
        eval_3dgs_btn.click(fn=methods.eval_3dgs,
                            inputs=[output_recon_3dgs, skip_train, skip_test],
                            outputs=[result_render_3dgs, eval_3dgs, gallery_3dgs])
        recon_dust3r_btn.click(fn=methods.recon_dust3r,
                               inputs=[dataset, outputsdir_state, schedule, niter, min_conf_thr, as_pointcloud,mask_sky, clean_depth, transparent_cams, cam_size,scenegraph_type, winsize, refid], 
                               outputs=[run_time_dust3r, result_recon_dust3r, output_recon_dust3r, outmodel_dust3r, outimgs_dust3r]).success(
                                           fn=get_imagelist,
                                           inputs=outimgs_dust3r,
                                           outputs=gallery_dust3r)
            
    demo.launch()
