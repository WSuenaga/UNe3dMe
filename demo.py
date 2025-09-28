import preprocess
import methods
import test
import gradio as gr

# Stateの値取得メソッド（データセットタブで得たパスを各タブに渡す）
def get_state_value(state):
    return state
def get_state_values(state):
    return state, state, state, state, state

# メディアUI切り替えメソッド
def display_media_ui(choice):
    if choice == "画像":
        return gr.Column(visible=True), gr.Column(visible=False)
    elif choice == "動画":
        return gr.Column(visible=False), gr.Column(visible=True)
def display_dataset_ui(choice):
    if choice == "作成したデータセット":
        return gr.Column(visible=True), gr.Column(visible=False)
    elif choice == "処理済データセット":
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
            media_radio = gr.Radio(["画像","動画"], label = "ファイルの種類を選択してください")

            # 画像入力UI
            with gr.Column(visible=False) as image_col:
                gr.Markdown("# 2.データセットの作成")
                gr.Markdown("""
                            ## 2.1.画像を入力してください
                            - 画像以外も選択できてしまうため注意してください．
                            """)
                images = gr.File(file_count="multiple")
                gr.Markdown("""
                            ## 2.2.任意のデータセットの名前を入力してください
                            - 入力が無い場合，ランダムな名前がデータセットに付与されます．
                            """)
                name = gr.Textbox(label="データセットの名前")
                run_copy_btn = gr.Button("データセット作成")
                with gr.Column(visible=False) as iresult_col:
                    gr.Markdown("# 3.データセットが作成されました")
                    output_image = gr.Textbox(label="データセット保存先")
                    gallery_image = gr.Gallery(label="入力された画像", columns=4, height="auto")

            # 動画入力UI
            with gr.Column(visible=False) as video_col:
                gr.Markdown("# 2.動画を入力してください")
                video = gr.Video(label="推論に使用する動画を選択してください.")
                fps = gr.Slider(value=3, minimum=1, maximum=5, step=1, label="1秒間に切り出すフレーム数")
                with gr.Accordion("オプション", open=False):
                    gr.Markdown("## データセットの圧縮")
                    gr.Markdown("重複している画像を削除し，計算リソースの節約を行います．")
                    gr.Markdown("- データセットの圧縮を行いますか？")
                    rsi = gr.Checkbox(value=True, label="圧縮する")
                    gr.Markdown("- SSIMの閾値を設定してください．値が小さいほどデータセットは圧縮されます．")
                    ssim = gr.Slider(value=0.8, minimum=0, maximum=1, label="SSIMの閾値")
                run_ffmpeg_btn = gr.Button("データセット作成")
                with gr.Column(visible=False) as vresult_col:
                    gr.Markdown("# 3.データセットが作成されました")
                    output_video = gr.Textbox(label="データセット保存先")
                    with gr.Row(equal_height=True):
                        comp_rate = gr.Textbox(label="圧縮率")
                        del_images_num = gr.Textbox(label="削除画像枚数")
                    gallery_vide = gr.Gallery(label="抽出された画像", columns=4, height="auto")
        # NeRF系
        with gr.Tab("NeRF"):
            current_dataset_nerf = gr.Textbox(label="現在セットされているデータセット")
            # NeRF
            with gr.Tab("NeRF"):
                gr.Markdown("# 1. データセットの種類を選択")
                nerf_radio = gr.Radio(["作成したデータセット","処理済データセット"], label = "3次元再構築に用いるデータセットを選択してください．")
                with gr.Column(visible=False) as pre_nerf_col:
                    gr.Markdown("""
                                # 2.前処理
                                - データセットをNeRFで扱えるように前処理を行う必要があります．
                                - 作成したデータセット以外が現在セットされていないか注意してください．
                                """)
                    run_nscolmap_btn1 = gr.Button("前処理実行")
                    result_nscolmap1 = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as ex_nerf_col:
                    gr.Markdown("""
                                # 2.データセットのアップロード
                                - ZIP圧縮を行ってからアップロードしてください．
                                - ZIPファイル以外も選択できてしまうため注意してください．
                                """)
                    ex_dataset_nerf = gr.File(label="データセットを選択してください")
                with gr.Column(visible=False) as train_nerf_col:
                    gr.Markdown("# 3.学習")
                    with gr.Accordion("オプション", open=False):
                        iter_nerf = gr.Slider(value=1000000, minimum=0, maximum=2000000, step=20000, label="総イテレーション数")
                    recon_nerf_btn = gr.Button("学習実行")
                    run_time_nerf = gr.Textbox(label="実行時間")
                    result_recon_nerf = gr.Textbox(label="実行結果")
                    output_recon_nerf = gr.Textbox(label="学習結果保存先")
                    outmodel_nerf = gr.Model3D(label="三次元再構築結果")
                    gallery_nerfacto = gr.Gallery(label="入力画像・深度マップ・信頼度マップ", columns=3, height="auto")
            
            # Nerfacto
            with gr.Tab("Nerfacto"):
                gr.Markdown("# 1. データセットの種類を選択")
                nerfacto_radio = gr.Radio(["作成したデータセット","処理済データセット"], label = "3次元再構築に用いるデータセットを選択してください．")
                with gr.Column(visible=False) as pre_nerfacto_col:
                    gr.Markdown("""
                                # 2.前処理
                                - データセットをNefactoで扱えるように前処理を行う必要があります．
                                - 作成したデータセット以外が現在セットされていないか注意してください．
                                """)
                    run_nscolmap_btn2 = gr.Button("前処理実行")
                    result_nscolmap2 = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as ex_nerfacto_col:
                    gr.Markdown("""
                                # 2.データセットのアップロード
                                - ZIP圧縮を行ってからアップロードしてください．
                                - ZIPファイル以外も選択できてしまうため注意してください．
                                """)
                    ex_dataset_nerfacto = gr.File(label="データセットを選択してください")
                with gr.Column(visible=False) as train_nerfacto_col:
                    gr.Markdown("# 3.学習")
                    with gr.Accordion("オプション", open=False):
                        iter_nerfacto = gr.Slider(value=100000, minimum=0, maximum=200000, step=2000, label="総イテレーション数")
                    recon_nerfacto_btn = gr.Button("学習実行")
                    run_time_nerfacto = gr.Textbox(label="実行時間")
                    result_recon_nerfacto = gr.Textbox(label="実行結果")
                    output_recon_nerfacto = gr.Textbox(label="学習結果保存先")
                    outmodel_nerfacto = gr.Model3D(label="三次元再構築結果")
                    gallery_nerfacto = gr.Gallery(label="入力画像・深度マップ・信頼度マップ", columns=3, height="auto")
            
            # mip-NeRF
            with gr.Tab("mip-NeRF"):
                gr.Markdown("# 1. データセットの種類を選択")
                mipnerf_radio = gr.Radio(["作成したデータセット","処理済データセット"], label = "3次元再構築に用いるデータセットを選択してください．")
                with gr.Column(visible=False) as pre_mipnerf_col:
                    gr.Markdown("""
                                # 2.前処理
                                - データセットをmip-NeRFで扱えるように前処理を行う必要があります．
                                - 作成したデータセット以外が現在セットされていないか注意してください．
                                """)
                    run_nscolmap_btn3 = gr.Button("前処理実行")
                    result_nscolmap3 = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as ex_mipnerf_col:
                    gr.Markdown("""
                                # 2.データセットのアップロード
                                - ZIP圧縮を行ってからアップロードしてください．
                                - ZIPファイル以外も選択できてしまうため注意してください．
                                """)
                    ex_dataset_mipnerf = gr.File(label="データセットを選択してください")
                with gr.Column(visible=False) as train_mipnerf_col:
                    gr.Markdown("# 3.学習")
                    with gr.Accordion("オプション", open=False):
                        iter_mipnerf = gr.Slider(value=1000000, minimum=0, maximum=2000000, step=20000, label="総イテレーション数")
                    recon_mipnerf_btn = gr.Button("学習実行")
                    run_time_mipnerf = gr.Textbox(label="実行時間")
                    result_recon_mipnerf = gr.Textbox(label="実行結果")
                    output_recon_mipnerf = gr.Textbox(label="学習結果保存先")
                    outmodel_mipnerf = gr.Model3D(label="三次元再構築結果")
                    gallery_mipnerf = gr.Gallery(label="入力画像・深度マップ・信頼度マップ", columns=3, height="auto")
            
            # SeaThru-NeRF
            with gr.Tab("SeaThru-NeRF"):
                gr.Markdown("# 1. データセットの種類を選択")
                stnerf_radio = gr.Radio(["作成したデータセット","処理済データセット"], label = "3次元再構築に用いるデータセットを選択してください．")
                with gr.Column(visible=False) as pre_stnerf_col:
                    gr.Markdown("""
                                # 2.前処理
                                - データセットをSeaThru-NeRFで扱えるように前処理を行う必要があります．
                                - 作成したデータセット以外が現在セットされていないか注意してください．
                                """)
                    run_nscolmap_btn4 = gr.Button("前処理実行")
                    result_nscolmap4 = gr.Textbox(label="実行結果")
                with gr.Column(visible=False) as ex_stnerf_col:
                    gr.Markdown("""
                                # 2.データセットのアップロード
                                - ZIP圧縮を行ってからアップロードしてください．
                                - ZIPファイル以外も選択できてしまうため注意してください．
                                """)
                    ex_dataset_stnerf = gr.File(label="データセットを選択してください")
                with gr.Column(visible=False) as train_stnerf_col:
                    gr.Markdown("# 3.学習")
                    with gr.Accordion("オプション", open=False):
                        iter_stnerf = gr.Slider(value=100000, minimum=0, maximum=200000, step=2000, label="総イテレーション数")
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
                gr.Model3D()

        """
        イベントリスナ
        """
        # UI切り替え
        media_radio.change(fn=display_media_ui, 
                           inputs=media_radio, 
                           outputs=[image_col, video_col])
        nerf_radio.change(fn=display_dataset_ui,
                          inputs=nerf_radio,
                          outputs=[pre_nerf_col, ex_nerf_col])
        nerfacto_radio.change(fn=display_dataset_ui,
                          inputs=nerfacto_radio,
                          outputs=[pre_nerfacto_col, ex_nerfacto_col])
        mipnerf_radio.change(fn=display_dataset_ui,
                          inputs=mipnerf_radio,
                          outputs=[pre_mipnerf_col, ex_mipnerf_col])
        stnerf_radio.change(fn=display_dataset_ui,
                          inputs=stnerf_radio,
                          outputs=[pre_stnerf_col, ex_stnerf_col])
        
        # データセット作成
        run_copy_btn.click(fn=preprocess.copy_images,
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
        run_ffmpeg_btn.click(fn=preprocess.extract_frames_with_filter, 
                         inputs=[video, datasetsdir_state, fps, rsi, ssim], 
                         outputs=[dataset, vresult_col, output_video, comp_rate, del_images_num, gallery_vide]
                        ).success(
                            fn=get_state_values, 
                            inputs=dataset, 
                            outputs=[current_dataset_nerf, 
                                     current_dataset_gs,
                                     current_dataset_3sters,
                                     current_dataset_mds,
                                     current_dataset_vgg])
        ex_dataset_nerf.upload(fn=preprocess.unzip2dataset,
                               inputs=[ex_dataset_nerf, datasetsdir_state],
                               outputs=[dataset, train_nerf_col]).success(
                                   fn=get_state_value,
                                   inputs=dataset,
                                   outputs=current_dataset_nerf)
        ex_dataset_nerfacto.upload(fn=preprocess.unzip2dataset,
                               inputs=[ex_dataset_nerfacto, datasetsdir_state],
                               outputs=[dataset, train_nerfacto_col]).success(
                                   fn=get_state_value,
                                   inputs=dataset,
                                   outputs=current_dataset_nerf)
        ex_dataset_mipnerf.upload(fn=preprocess.unzip2dataset,
                               inputs=[ex_dataset_mipnerf, datasetsdir_state],
                               outputs=[dataset, train_mipnerf_col]).success(
                                   fn=get_state_value,
                                   inputs=dataset,
                                   outputs=current_dataset_nerf)
        ex_dataset_stnerf.upload(fn=preprocess.unzip2dataset,
                               inputs=[ex_dataset_stnerf, datasetsdir_state],
                               outputs=[dataset, train_stnerf_col]).success(
                                   fn=get_state_value,
                                   inputs=dataset,
                                   outputs=current_dataset_nerf)   
        
        # 前処理
        run_nscolmap_btn1.click(fn=preprocess.run_nscolmap,
                                inputs=dataset,
                                outputs=[result_nscolmap1, train_nerf_col])
        run_nscolmap_btn2.click(fn=preprocess.run_nscolmap,
                                inputs=dataset,
                                outputs=[result_nscolmap2, train_nerfacto_col])
        run_nscolmap_btn3.click(fn=preprocess.run_nscolmap,
                                inputs=dataset,
                                outputs=[result_nscolmap3, train_mipnerf_col])
        run_nscolmap_btn4.click(fn=preprocess.run_nscolmap,
                                inputs=dataset,
                                outputs=[result_nscolmap4, train_stnerf_col])
        run_gscolmap_btn.click(fn=preprocess.run_gscolmap,
                               inputs=dataset,
                               outputs=[result_gscolmap, train_3dgs_col])
        
        # 三次元再構築，レンダリング・評価
        recon_nerf_btn.click(fn=methods.recon_nerf,
                             inputs=[dataset, outputsdir_state, iter_nerf],
                             outputs=[run_time_nerf, result_recon_nerf, output_recon_nerf, outmodel_nerf])
        recon_nerfacto_btn.click(fn=methods.recon_nerfacto,
                                 inputs=[dataset, outputsdir_state, iter_nerfacto],
                                 outputs=[run_time_nerfacto, result_recon_nerfacto, output_recon_nerfacto, outmodel_nerfacto])
        recon_mipnerf_btn.click(fn=methods.recon_mipNeRF,
                             inputs=[dataset, outputsdir_state, iter_mipnerf],
                             outputs=[run_time_mipnerf, result_recon_mipnerf, output_recon_mipnerf, outmodel_mipnerf])
        recon_stnerf_btn.click(fn=methods.recon_seathruNerf,
                             inputs=[dataset, outputsdir_state, iter_stnerf],
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
                                           fn=preprocess.get_imagelist,
                                           inputs=outimgs_dust3r,
                                           outputs=gallery_dust3r)
            
    demo.launch()
