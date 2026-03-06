import os
import json
import pandas as pd
import gradio as gr

import local_backend
import methods

# 言語データロードメソッド
def load_translations(lang_code):
    path = os.path.join("translations", f"{lang_code}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# UI言語切り替えメソッド
def update_ui(choice):
    if choice == "日本語":
        new_lang_code = "jp"
    elif choice =="ENGLISH":
        new_lang_code = "en"

    lang = load_translations(new_lang_code)

    return (
        new_lang_code, # lang_state
        gr.Textbox(label=lang["current_dataset_images"]), # current_dataset_images
        gr.Textbox(label=lang["current_dataset_colmap"]), # current_dataset_colmap
        # DatasetTab  
        gr.Tab(label=lang["dataset_tab"]["title"]), # dataset_tab
        gr.Markdown(lang["dataset_tab"]["subtitle1"]), # dataset_sub1
        gr.Markdown(lang["dataset_tab"]["info1"]), # dataset_info1
        gr.Radio(choices=[lang["dataset_tab"]["radio_new"], lang["dataset_tab"]["radio_load"]], 
                 label=lang["dataset_tab"]["dataset_radio"]), # dataset_radio 
        gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["subtitle2"]), # dataset_new_sub2
        gr.Radio(choices=[lang["dataset_tab"]["new_dataset_section"]["radio_image"], lang["dataset_tab"]["new_dataset_section"]["radio_video"]],
                 label=lang["dataset_tab"]["new_dataset_section"]["media_radio"]), # media_radio
        gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["image_section"]["subtitle3"]), # dataset_image_sub3
        gr.File(label=lang["dataset_tab"]["new_dataset_section"]["image_section"]["images"]), # images
        gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["image_section"]["dataset_name"], info=lang["dataset_tab"]["new_dataset_section"]["image_section"]["dataset_name_info"]), # dataset_name
        gr.Button(value=lang["dataset_tab"]["new_dataset_section"]["image_section"]["run_copy_btn"]), # run_copy_btn
        gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["image_section"]["subtitle4"]), # dataset_image_sub4
        gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["image_section"]["output_image"]), # output_image
        gr.Gallery(label=lang["dataset_tab"]["new_dataset_section"]["image_section"]["gallery_image"]), # gallery_image
        gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["video_section"]["subtitle3"]), # dataset_video_sub3
        gr.Video(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["video"]), # video
        gr.Slider(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["fps"]), # fps
        gr.Accordion(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["title"]), # dataset_video_option
        gr.Markdown(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["subtitle"]), # dataset_video_option_subtitle
        gr.Markdown(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["info1"]), # dataset_video_option_info1
        gr.Checkbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["rsi"]), # rsi
        gr.Markdown(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["info2"]), # dataset_video_option_info2
        gr.Slider(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["ssim"]), # ssim
        gr.Button(value=lang["dataset_tab"]["new_dataset_section"]["video_section"]["run_ffmpeg_btn"]), # run_ffmpeg_btn
        gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["output_video"]), # output_video
        gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["comp_rate"]), # comp_rate
        gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["sel_images_num"]), # sel_images_num
        gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["rej_images_num"]), # rej_images_num
        gr.Gallery(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["gallery_video"]), # gallery_video
        gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["video_section"]["subtitle4"]), # dataset_video_sub4
        gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["video_section"]["info1"]), # dataset_video_info1
        gr.DownloadButton(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["download_zipfile_btn"]), # zipfile_images
        gr.Markdown(lang["dataset_tab"]["load_dataset_section"]["subtitle2"]), # dataset_load_sub2
        gr.Markdown(lang["dataset_tab"]["load_dataset_section"]["info1"]), # load_dataset_info1
        gr.File(label=lang["dataset_tab"]["load_dataset_section"]["load_dataset"]), # load_dataset
        gr.Textbox(label=lang["dataset_tab"]["load_dataset_section"]["log_unzip"]), # log_unzip
        # COLMAPTab
        gr.Tab(label=lang["colmap_tab"]["title"]), # dataset_tab
        gr.Markdown(lang["colmap_tab"]["subtitle1"]), # colmap_sub1
        gr.Markdown(lang["colmap_tab"]["info1"]), # colmap_info1
        gr.Accordion(label=lang["colmap_tab"]["option"]["title"]), # colmap_option
        gr.Markdown(lang["colmap_tab"]["option"]["info1"]), # colmap_option_info1
        gr.Checkbox(label=lang["colmap_tab"]["option"]["rebuild"]), # rebuild
        gr.Button(value=lang["colmap_tab"]["run_colmap_btn"]), # run_colmap_btn
        gr.Textbox(label=lang["colmap_tab"]["result_colmap"]), # result_colmap
        gr.Markdown(lang["colmap_tab"]["subtitle2"]), # colmap_sub2
        gr.Markdown(lang["colmap_tab"]["info2"]), # colmap_info2
        gr.DownloadButton(label=lang["colmap_tab"]["download_zipfile_btn"]), # zipfile_colmap
        # NeRFTab
        gr.Tab(label=lang["nerf_tab"]["title"]), # nerf_tab
        # Vanilla NeRF
        gr.Tab(label=lang["nerf_tab"]["vnerf"]["title"]), # vnerf_tab
        gr.Markdown(lang["nerf_tab"]["vnerf"]["subtitle1"]), # vnerf_sub1
        gr.Accordion(label=lang["nerf_tab"]["vnerf"]["option"]["title"]), # vnerf_option
        gr.Radio(label= lang["nerf_tab"]["vnerf"]["option"]["exe_mode"]), # exe_mode_vnerf
        gr.Slider(label=lang["nerf_tab"]["vnerf"]["option"]["iter"]), # iter_vnerf
        gr.Button(value=lang["nerf_tab"]["vnerf"]["recon_btn"]), # recon_vnerf_btn
        gr.Markdown(lang["nerf_tab"]["vnerf"]["viewer"]), # vnerf_viewer
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["outdir_recon"]), # outdir_recon_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["runtime_recon"]), # runtime_recon_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["result_recon"]), # result_recon_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["log_recon"]), # log_recon_vnerf
        gr.Markdown(lang["nerf_tab"]["vnerf"]["subtitle2"]), # vnerf_sub2
        gr.Button(value=lang["nerf_tab"]["vnerf"]["export_btn"]), # export_vnerf_btn
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["outdir_export"]), # outdir_export_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["runtime_export"]), # runtime_export_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["result_export"]), # result_export_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["log_export"]), # log_export_vnerf
        gr.Markdown(lang["nerf_tab"]["vnerf"]["subtitle3"]), # vnerf_sub3
        gr.Button(value=lang["nerf_tab"]["vnerf"]["eval_btn"]), # eval_vnerf_btn
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["outdir_eval"]), # outdir_eval_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["runtime_eval"]), # runtime_eval_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["result_eval"]), # result_eval_vnerf
        gr.Textbox(label=lang["nerf_tab"]["vnerf"]["log_eval"]), # log_eval_vnerf
        gr.DataFrame(label=lang["nerf_tab"]["vnerf"]["metrics"]), # metrics_vnerf
        gr.Gallery(label=lang["nerf_tab"]["vnerf"]["gallery"]), # gallery_vnerf
        # Nerfacto
        gr.Tab(label=lang["nerf_tab"]["nerfacto"]["title"]), # nerfacto_tab
        gr.Markdown(lang["nerf_tab"]["nerfacto"]["subtitle1"]), # nerfacto_sub1
        gr.Accordion(label=lang["nerf_tab"]["nerfacto"]["option"]["title"]), # nerfacto_option
        gr.Radio(label= lang["nerf_tab"]["nerfacto"]["option"]["exe_mode"]), # exe_mode_nerfacto
        gr.Slider(label=lang["nerf_tab"]["nerfacto"]["option"]["iter"]), # iter_nerfacto
        gr.Button(value=lang["nerf_tab"]["nerfacto"]["recon_btn"]), # recon_nerfacto_btn
        gr.Markdown(lang["nerf_tab"]["nerfacto"]["viewer"]), # nerfacto_viewer
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["outdir_recon"]), # outdir_recon_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["runtime_recon"]), # runtime_recon_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["result_recon"]), # result_recon_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["log_recon"]), # log_recon_nerfacto
        gr.Markdown(lang["nerf_tab"]["nerfacto"]["subtitle2"]), # nerfacto_sub2
        gr.Button(value=lang["nerf_tab"]["nerfacto"]["export_btn"]), # export_nerfacto_btn
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["outdir_export"]), # outdir_export_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["runtime_export"]), # runtime_export_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["result_export"]), # result_export_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["log_export"]), # log_export_nerfacto
        gr.Markdown(lang["nerf_tab"]["nerfacto"]["subtitle3"]), # nerfacto_sub3
        gr.Button(value=lang["nerf_tab"]["nerfacto"]["eval_btn"]), # eval_nerfacto_btn
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["outdir_eval"]), # outdir_eval_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["runtime_eval"]), # runtime_eval_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["result_eval"]), # result_eval_nerfacto
        gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["log_eval"]), # log_eval_nerfacto
        gr.DataFrame(label=lang["nerf_tab"]["nerfacto"]["metrics"]), # metrics_nerfacto
        gr.Gallery(label=lang["nerf_tab"]["nerfacto"]["gallery"]), # gallery_nerfacto
        # mip-NeRF
        gr.Tab(label=lang["nerf_tab"]["mip-nerf"]["title"]), # mipnerf_tab
        gr.Markdown(lang["nerf_tab"]["mip-nerf"]["subtitle1"]), # mipnerf_sub1
        gr.Accordion(label=lang["nerf_tab"]["mip-nerf"]["option"]["title"]), # mipnerf_option
        gr.Radio(label= lang["nerf_tab"]["mip-nerf"]["option"]["exe_mode"]), # exe_mode_mipnerf
        gr.Slider(label=lang["nerf_tab"]["mip-nerf"]["option"]["iter"]), # iter_mipnerf
        gr.Button(value=lang["nerf_tab"]["mip-nerf"]["recon_btn"]), # recon_mipnerf_btn
        gr.Markdown(lang["nerf_tab"]["mip-nerf"]["viewer"]), # mipnerf_viewer
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["outdir_recon"]), # outdir_recon_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["runtime_recon"]), # runtime_recon_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["result_recon"]), # result_recon_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["log_recon"]), # log_recon_mipnerf
        gr.Markdown(lang["nerf_tab"]["mip-nerf"]["subtitle2"]), # mipnerf_sub2
        gr.Button(value=lang["nerf_tab"]["mip-nerf"]["export_btn"]), # export_mipnerf_btn
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["outdir_export"]), # outdir_export_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["runtime_export"]), # runtime_export_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["result_export"]), # result_export_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["log_export"]), # log_export_mipnerf
        gr.Markdown(lang["nerf_tab"]["mip-nerf"]["subtitle3"]), # mipnerf_sub3
        gr.Button(value=lang["nerf_tab"]["mip-nerf"]["eval_btn"]), # eval_mipnerf_btn
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["outdir_eval"]), # outdir_eval_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["runtime_eval"]), # runtime_eval_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["result_eval"]), # result_eval_mipnerf
        gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["log_eval"]), # log_eval_mipnerf
        gr.DataFrame(label=lang["nerf_tab"]["mip-nerf"]["metrics"]), # metrics_mipnerf
        gr.Gallery(label=lang["nerf_tab"]["mip-nerf"]["gallery"]), # gallery_mipnerf
        # SeaThru-NeRF
        gr.Tab(label=lang["nerf_tab"]["seathru-nerf"]["title"]), # stnerf_tab
        gr.Markdown(lang["nerf_tab"]["seathru-nerf"]["subtitle1"]), # stnerf_sub1
        gr.Accordion(label=lang["nerf_tab"]["seathru-nerf"]["option"]["title"]), # stnerf_option
        gr.Radio(label= lang["nerf_tab"]["seathru-nerf"]["option"]["exe_mode"]), # exe_mode_stnerf
        gr.Slider(label=lang["nerf_tab"]["seathru-nerf"]["option"]["iter"]), # iter_stnerf
        gr.Button(value=lang["nerf_tab"]["seathru-nerf"]["recon_btn"]), # recon_stnerf_btn
        gr.Markdown(lang["nerf_tab"]["seathru-nerf"]["viewer"]), # stnerf_viewer
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["outdir_recon"]), # outdir_recon_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["runtime_recon"]), # runtime_recon_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["result_recon"]), # result_recon_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["log_recon"]), # log_recon_stnerf
        gr.Markdown(lang["nerf_tab"]["seathru-nerf"]["subtitle2"]), # stnerf_sub2
        gr.Button(value=lang["nerf_tab"]["seathru-nerf"]["export_btn"]), # export_stnerf_btn
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["outdir_export"]), # outdir_export_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["runtime_export"]), # runtime_export_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["result_export"]), # result_export_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["log_export"]), # log_export_stnerf
        gr.Markdown(lang["nerf_tab"]["seathru-nerf"]["subtitle3"]), # stnerf_sub3
        gr.Button(value=lang["nerf_tab"]["seathru-nerf"]["eval_btn"]), # eval_stnerf_btn
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["outdir_eval"]), # outdir_eval_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["runtime_eval"]), # runtime_eval_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["result_eval"]), # result_eval_stnerf
        gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["log_eval"]), # log_eval_stnerf
        gr.DataFrame(label=lang["nerf_tab"]["seathru-nerf"]["metrics"]), # metrics_stnerf
        gr.Gallery(label=lang["nerf_tab"]["seathru-nerf"]["gallery"]), # gallery_stnerf
        # GSTab
        gr.Tab(label=lang["gs_tab"]["title"]), # gs_tab
        # Vanilla GS
        gr.Tab(label=lang["gs_tab"]["vgs"]["title"]), # vgs_tab
        gr.Markdown(lang["gs_tab"]["vgs"]["subtitle1"]), # vgs_sub1
        gr.Accordion(label=lang["gs_tab"]["vgs"]["option"]["title"]), # vgs_option
        gr.Radio(label= lang["gs_tab"]["vgs"]["option"]["exe_mode"]), # exe_mode_vgs
        gr.Button(value=lang["gs_tab"]["vgs"]["recon_btn"]), # recon_vgs_btn
        gr.Textbox(label=lang["gs_tab"]["vgs"]["outdir_recon"]), # outdir_recon_vgs
        gr.Textbox(label=lang["gs_tab"]["vgs"]["runtime_recon"]), # runtime_vgs
        gr.Textbox(label=lang["gs_tab"]["vgs"]["result_recon"]), # result_recon_vgs
        gr.Textbox(label=lang["gs_tab"]["vgs"]["log_recon"]), # log_recon_vgs
        gr.Model3D(label=lang["gs_tab"]["vgs"]["outmodel"]), # outmodel_vgs
        gr.Markdown(lang["gs_tab"]["vgs"]["subtitle2"]), # vgs_sub2
        gr.Checkbox(label=lang["gs_tab"]["vgs"]["skip_train"]), # skip_train
        gr.Checkbox(label=lang["gs_tab"]["vgs"]["skip_test"]), # skip_test
        gr.Button(value=lang["gs_tab"]["vgs"]["eval_btn"]), # eval_vgs_btn
        gr.Textbox(label=lang["gs_tab"]["vgs"]["runtime_eval"]), # runtime_eval_vgs
        gr.Textbox(label=lang["gs_tab"]["vgs"]["result_eval"]), # result_render_vgs
        gr.Textbox(label=lang["gs_tab"]["vgs"]["log_eval"]), # log_eval_vgs
        gr.DataFrame(label=lang["gs_tab"]["vgs"]["metrics"]), # metrics_vgs
        gr.Gallery(label=lang["gs_tab"]["vgs"]["gallery"]), # gallery_vgs
        # Mip-Splatting
        gr.Tab(label=lang["gs_tab"]["mip-splatting"]["title"]), # mips_tab
        gr.Markdown(lang["gs_tab"]["mip-splatting"]["subtitle1"]), # mips_sub1
        gr.Accordion(label=lang["gs_tab"]["mip-splatting"]["option"]["title"]), # mips_option
        gr.Radio(label= lang["gs_tab"]["mip-splatting"]["option"]["exe_mode"]), # exe_mode_mips
        gr.Slider(label=lang["gs_tab"]["mip-splatting"]["option"]["save_iter"]), # save_iter_mips
        gr.Button(value=lang["gs_tab"]["mip-splatting"]["recon_btn"]), # recon_mips_btn
        gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["outdir_recon"]), # outdir_recon_mips
        gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["runtime_recon"]), # runtime_recon_mips
        gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["result_recon"]), # result_recon_mips
        gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["log_recon"]), # log_recon_mips
        gr.Model3D(label=lang["gs_tab"]["mip-splatting"]["outmodel"]), # outmodel_mips
        gr.Markdown(lang["gs_tab"]["mip-splatting"]["subtitle2"]), # mips_sub2
        gr.Button(value=lang["gs_tab"]["mip-splatting"]["eval_btn"]), # eval_mips_btn
        gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["outdir_eval"]), # outdir_eval_mips
        gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["runtime_eval"]), # runtime_eval_mips
        gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["result_eval"]), # result_eval_mips
        gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["log_eval"]), # log_eval_mips
        gr.DataFrame(label=lang["gs_tab"]["mip-splatting"]["metrics"]), # metrics_mips
        gr.Gallery(label=lang["gs_tab"]["mip-splatting"]["gallery"]), # gallery_mips
        # Splatfacto
        gr.Tab(label=lang["gs_tab"]["splatfacto"]["title"]), # sfacto_tab
        gr.Markdown(lang["gs_tab"]["splatfacto"]["subtitle1"]), # sfacto_sub1
        gr.Accordion(label=lang["gs_tab"]["splatfacto"]["option"]["title"]), # sfacto_option
        gr.Radio(label= lang["gs_tab"]["splatfacto"]["option"]["exe_mode"]), # exe_mode_sfacto
        gr.Slider(label=lang["gs_tab"]["splatfacto"]["option"]["iter"]), # iter_sfacto
        gr.Button(value=lang["gs_tab"]["splatfacto"]["recon_btn"]), # recon_sfacto_btn
        gr.Markdown(lang["gs_tab"]["splatfacto"]["viewer"]), # sfacto_viewer
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["outdir_recon"]), # outdir_recon_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["runtime_recon"]), # runtime_recon_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["result_recon"]), # result_recon_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["log_recon"]), # log_recon_sfacto
        gr.Markdown(lang["gs_tab"]["splatfacto"]["subtitle2"]), # sfacto_sub2
        gr.Button(value=lang["gs_tab"]["splatfacto"]["export_btn"]), # export_sfacto_btn
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["outdir_export"]), # outdir_export_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["runtime_export"]), # runtime_export_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["result_export"]), # result_export_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["log_export"]), # log_export_sfacto
        gr.Markdown(lang["gs_tab"]["splatfacto"]["subtitle3"]), # sfacto_sub3
        gr.Button(value=lang["gs_tab"]["splatfacto"]["eval_btn"]), # eval_sfacto_btn
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["outdir_eval"]), # outdir_eval_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["runtime_eval"]), # runtime_eval_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["result_eval"]), # result_eval_sfacto
        gr.Textbox(label=lang["gs_tab"]["splatfacto"]["log_eval"]), # log_eval_sfacto
        gr.DataFrame(label=lang["gs_tab"]["splatfacto"]["metrics"]), # metrics_sfacto
        gr.Gallery(label=lang["gs_tab"]["splatfacto"]["gallery"]), # gallery_sfacto
        # 4D-Gaussians
        gr.Tab(label=lang["gs_tab"]["4d-gaussians"]["title"]), # gs4d_tab
        gr.Markdown(lang["gs_tab"]["4d-gaussians"]["subtitle1"]), # gs4d_sub1
        gr.Accordion(label=lang["gs_tab"]["4d-gaussians"]["option"]["title"]), # gs4d_option
        gr.Radio( label= lang["gs_tab"]["4d-gaussians"]["option"]["exe_mode"]), # exe_mode_4dgs
        gr.Slider(label=lang["gs_tab"]["4d-gaussians"]["option"]["save_iter"]), # save_iter_gs4d
        gr.Button(value=lang["gs_tab"]["4d-gaussians"]["recon_btn"]), # recon_gs4d_btn
        gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["outdir_recon"]), # outdir_recon_gs4d
        gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["runtime_recon"]), # runtime_recon_gs4d
        gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["result_recon"]), # result_recon_gs4d
        gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["log_recon"]), # log_recon_gs4d
        gr.Model3D(label=lang["gs_tab"]["4d-gaussians"]["outmodel"]), # outmodel_gs4d
        gr.Markdown(lang["gs_tab"]["4d-gaussians"]["subtitle2"]), # gs4d_sub2
        gr.Button(value=lang["gs_tab"]["4d-gaussians"]["eval_btn"]), # eval_4dgs_btn
        gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["outdir_eval"]), # outdir_eval_4dgs
        gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["runtime_eval"]), # runtime_eval_4dgs
        gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["result_eval"]), # result_eval_4dgs
        gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["log_eval"]), # log_eval_4dgs
        gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["gs_tab"]["4d-gaussians"]["metrics"]), # metrics_4dgs
        gr.Gallery(label=lang["gs_tab"]["4d-gaussians"]["gallery"]), # gallery_4dgs
        # 3stersTab
        gr.Tab(label=lang["3sters_tab"]["title"]), # esters_tab
        # DUSt3R
        gr.Tab(label=lang["3sters_tab"]["dust3r"]["title"]), # dust3r_tab
        gr.Markdown(lang["3sters_tab"]["dust3r"]["subtitle1"]), # dust3r_sub1
        gr.Accordion(label=lang["3sters_tab"]["dust3r"]["option"]["title"]), # dust3r_option
        gr.Radio(label= lang["3sters_tab"]["dust3r"]["option"]["exe_mode"]), # exe_mode_dust3r
        gr.Button(value=lang["3sters_tab"]["dust3r"]["recon_btn"]), # recon_dust3r_btn
        gr.Textbox(label=lang["3sters_tab"]["dust3r"]["outdir_recon"]), # outdir_recon_dust3r
        gr.Textbox(label=lang["3sters_tab"]["dust3r"]["runtime_recon"]), # runtime_recon_dust3r
        gr.Textbox(label=lang["3sters_tab"]["dust3r"]["result_recon"]), # result_recon_dust3r
        gr.Textbox(label=lang["3sters_tab"]["dust3r"]["log_recon"]), # log_recon_dust3r
        gr.Model3D(label=lang["3sters_tab"]["dust3r"]["outmodel"]), # outmodel_dust3r
        gr.Gallery(label=lang["3sters_tab"]["dust3r"]["gallery"]), # gallery_dust3r
        # MASt3R
        gr.Tab(label=lang["3sters_tab"]["mast3r"]["title"]), # mast3r_tab
        gr.Markdown(lang["3sters_tab"]["mast3r"]["subtitle1"]), # mast3r_sub1
        gr.Accordion(label=lang["3sters_tab"]["mast3r"]["option"]["title"]), # mast3r_option
        gr.Radio(label= lang["3sters_tab"]["mast3r"]["option"]["exe_mode"]), # exe_mode_mast3r
        gr.Button(value=lang["3sters_tab"]["mast3r"]["recon_btn"]), # recon_mast3r_btn
        gr.Textbox(label=lang["3sters_tab"]["mast3r"]["outdir_recon"]), # outdir_recon_mast3r
        gr.Textbox(label=lang["3sters_tab"]["mast3r"]["runtime_recon"]), # runtime_recon_mast3r
        gr.Textbox(label=lang["3sters_tab"]["mast3r"]["result_recon"]), # result_recon_mast3r
        gr.Textbox(label=lang["3sters_tab"]["mast3r"]["log_recon"]), # log_recon_mast3r
        gr.Model3D(label=lang["3sters_tab"]["mast3r"]["outmodel"]), # outmodel_mast3r
        # MonST3R
        gr.Tab(label=lang["3sters_tab"]["monst3r"]["title"]), # monst3r_tab
        gr.Markdown(lang["3sters_tab"]["monst3r"]["subtitle1"]), # monst3r_sub1
        gr.Accordion(label=lang["3sters_tab"]["monst3r"]["option"]["title"]), # monst3r_option
        gr.Radio(label= lang["3sters_tab"]["monst3r"]["option"]["exe_mode"]), # exe_mode_monst3r
        gr.Button(value=lang["3sters_tab"]["monst3r"]["recon_btn"]), # recon_monst3r_btn
        gr.Textbox(label=lang["3sters_tab"]["monst3r"]["outdir_recon"]), # outdir_recon_monst3r
        gr.Textbox(label=lang["3sters_tab"]["monst3r"]["runtime_recon"]), # runtime_recon_monst3r
        gr.Textbox(label=lang["3sters_tab"]["monst3r"]["result_recon"]), # result_recon_monst3r
        gr.Textbox(label=lang["3sters_tab"]["monst3r"]["log_recon"]), # log_recon_monst3r
        gr.Model3D(label=lang["3sters_tab"]["monst3r"]["outmodel"]), # outmodel_monst3r
        # Easi3R
        gr.Tab(label=lang["3sters_tab"]["easi3r"]["title"]), # easi3r_tab
        gr.Markdown(lang["3sters_tab"]["easi3r"]["subtitle1"]), # easi3r_sub1
        gr.Markdown(lang["3sters_tab"]["easi3r"]["info1"]), # easi3r_info1
        gr.Accordion(label=lang["3sters_tab"]["easi3r"]["option"]["title"]), # easi3r_option
        gr.Radio(label= lang["3sters_tab"]["easi3r"]["option"]["exe_mode"]), # exe_mode_easi3r
        gr.Button(value=lang["3sters_tab"]["easi3r"]["recon_btn"]), # recon_easi3r_btn
        gr.Textbox(label=lang["3sters_tab"]["easi3r"]["outdir_recon"]), # outdir_recon_easi3r
        gr.Textbox(label=lang["3sters_tab"]["easi3r"]["runtime_recon"]), # runtime_recon_easi3r
        gr.Textbox(label=lang["3sters_tab"]["easi3r"]["result_recon"]), # result_recon_easi3r
        gr.Textbox(label=lang["3sters_tab"]["easi3r"]["log_recon"]), # log_recon_easi3r
        gr.Model3D(label=lang["3sters_tab"]["easi3r"]["outmodel"]), # outmodel_easi3r
        # MUSt3R
        gr.Tab(label=lang["3sters_tab"]["must3r"]["title"]), # must3r_tab
        gr.Markdown(lang["3sters_tab"]["must3r"]["subtitle1"]), # must3r_sub1
        gr.Accordion(label=lang["3sters_tab"]["must3r"]["option"]["title"]), # must3r_option
        gr.Radio(label= lang["3sters_tab"]["must3r"]["option"]["exe_mode"]), # exe_mode_must3r
        gr.Button(value=lang["3sters_tab"]["must3r"]["recon_btn"]), # recon_must3r_btn
        gr.Textbox(label=lang["3sters_tab"]["must3r"]["outdir_recon"]), # outdir_recon_must3r
        gr.Textbox(label=lang["3sters_tab"]["must3r"]["runtime_recon"]), # runtime_recon_must3r
        gr.Textbox(label=lang["3sters_tab"]["must3r"]["result_recon"]), # result_recon_must3r
        gr.Textbox(label=lang["3sters_tab"]["must3r"]["log_recon"]), # log_recon_must3r
        gr.Model3D(label=lang["3sters_tab"]["must3r"]["outmodel"]), # outmodel_must3r
        # Fast3R
        gr.Tab(label=lang["3sters_tab"]["fast3r"]["title"]), # fast3r_tab
        gr.Markdown(lang["3sters_tab"]["fast3r"]["subtitle1"]), # fast3r_sub1
        gr.Accordion(label=lang["3sters_tab"]["fast3r"]["option"]["title"]), # fast3r_option
        gr.Radio(label= lang["3sters_tab"]["fast3r"]["option"]["exe_mode"]), # exe_mode_fast3r
        gr.Button(value=lang["3sters_tab"]["fast3r"]["recon_btn"]), # recon_fast3r_btn
        gr.Textbox(label=lang["3sters_tab"]["fast3r"]["outdir_recon"]), # outdir_recon_fast3r
        gr.Textbox(label=lang["3sters_tab"]["fast3r"]["runtime_recon"]), # runtime_recon_fast3r
        gr.Textbox(label=lang["3sters_tab"]["fast3r"]["result_recon"]), # result_recon_fast3r
        gr.Textbox(label=lang["3sters_tab"]["fast3r"]["log_recon"]), # log_recon_fast3r
        gr.Model3D(label=lang["3sters_tab"]["fast3r"]["outmodel"]), # outmodel_fast3r
        # Splatt3R
        gr.Tab(label=lang["3sters_tab"]["splatt3r"]["title"]), # splatt3r_tab
        gr.Markdown(lang["3sters_tab"]["splatt3r"]["subtitle1"]), # splatt3r_sub1
        gr.Markdown(lang["3sters_tab"]["splatt3r"]["info1"]), # splatt3r_info1
        gr.Image(label=lang["3sters_tab"]["splatt3r"]["image"]), # img_splatt3r
        gr.Markdown(lang["3sters_tab"]["splatt3r"]["subtitle2"]), # splatt3r_sub2
        gr.Accordion(label=lang["3sters_tab"]["splatt3r"]["option"]["title"]), # splatt3r_option
        gr.Radio(label= lang["3sters_tab"]["splatt3r"]["option"]["exe_mode"]), # exe_mode_splatt3r
        gr.Button(value=lang["3sters_tab"]["splatt3r"]["recon_btn"]), # recon_splatt3r_btn
        gr.Textbox(label=lang["3sters_tab"]["splatt3r"]["outdir_recon"]), # outdir_recon_splatt3r
        gr.Textbox(label=lang["3sters_tab"]["splatt3r"]["runtime_recon"]), # runtime_recon_splatt3r
        gr.Textbox(label=lang["3sters_tab"]["splatt3r"]["result_recon"]), # result_recon_splatt3r
        gr.Textbox(label=lang["3sters_tab"]["splatt3r"]["log_recon"]), # log_recon_splatt3r
        gr.Model3D(label=lang["3sters_tab"]["splatt3r"]["outmodel"]), # outmodel_splatt3r
        # CUT3R
        gr.Tab(label=lang["3sters_tab"]["cut3r"]["title"]), # cut3r_tab
        gr.Markdown(lang["3sters_tab"]["cut3r"]["subtitle1"]), # cut3r_sub1
        gr.Accordion(label=lang["3sters_tab"]["cut3r"]["option"]["title"]), # cut3r_option
        gr.Radio(label= lang["3sters_tab"]["cut3r"]["option"]["exe_mode"]), # exe_mode_cut3r
        gr.Button(value=lang["3sters_tab"]["cut3r"]["recon_btn"]), # recon_cut3r_btn
        gr.Textbox(label=lang["3sters_tab"]["cut3r"]["outdir_recon"]), # outdir_recon_cut3r
        gr.Textbox(label=lang["3sters_tab"]["cut3r"]["runtime_recon"]), # runtime_recon_cut3r
        gr.Textbox(label=lang["3sters_tab"]["cut3r"]["result_recon"]), # result_recon_cut3r
        gr.Textbox(label=lang["3sters_tab"]["cut3r"]["log_recon"]), # log_recon_cut3r
        gr.Model3D(label=lang["3sters_tab"]["cut3r"]["outmodel"]), # outmodel_cut3r
        # WinT3R
        gr.Tab(label=lang["3sters_tab"]["wint3r"]["title"]), # wint3r_tab
        gr.Markdown(lang["3sters_tab"]["wint3r"]["subtitle1"]), # wint3r_sub1
        gr.Accordion(label=lang["3sters_tab"]["wint3r"]["option"]["title"]), # wint3r_option
        gr.Radio(label= lang["3sters_tab"]["wint3r"]["option"]["exe_mode"]), # exe_mode_wint3r
        gr.Button(value=lang["3sters_tab"]["wint3r"]["recon_btn"]), # recon_wint3r_btn
        gr.Textbox(label=lang["3sters_tab"]["wint3r"]["outdir_recon"]), # outdir_recon_wint3r
        gr.Textbox(label=lang["3sters_tab"]["wint3r"]["runtime_recon"]), # runtime_recon_wint3r
        gr.Textbox(label=lang["3sters_tab"]["wint3r"]["result_recon"]), # result_recon_wint3r
        gr.Textbox(label=lang["3sters_tab"]["wint3r"]["log_recon"]), # log_recon_wint3r
        gr.Model3D(label=lang["3sters_tab"]["wint3r"]["outmodel"]), # outmodel_wint3r
        # vggtTab
        gr.Tab(label=lang["vggt_tab"]["title"]), # vggt_tab
        # VGGT
        gr.Tab(label=lang["vggt_tab"]["vggt"]["title"]), # vggt_tab
        gr.Markdown(lang["vggt_tab"]["vggt"]["subtitle1"]), # vggt_sub1
        gr.Accordion(label=lang["vggt_tab"]["vggt"]["option"]["title"]), # vggt_option
        gr.Radio(label= lang["vggt_tab"]["vggt"]["option"]["exe_mode"]), # exe_mode_vggt
        gr.Radio(label=lang["vggt_tab"]["vggt"]["option"]["mode_vggt"]), # mode_vggt
        gr.Button(value=lang["vggt_tab"]["vggt"]["recon_btn"]), # recon_vggt_btn
        gr.Textbox(label=lang["vggt_tab"]["vggt"]["outdir_recon"]), # outdir_recon_vggt
        gr.Textbox(label=lang["vggt_tab"]["vggt"]["runtime_recon"]), # runtime_recon_vggt
        gr.Textbox(label=lang["vggt_tab"]["vggt"]["result_recon"]), # result_recon_vggt
        gr.Textbox(label=lang["vggt_tab"]["vggt"]["log_recon"]), # log_recon_vggt
        gr.Textbox(label=lang["vggt_tab"]["vggt"]["outmodel"]), # outmodel_vggt
        # VGGSfM
        gr.Tab(label=lang["vggt_tab"]["vggsfm"]["title"]), # vggsfm_tab
        gr.Markdown(lang["vggt_tab"]["vggsfm"]["subtitle1"]), # vggsfm_sub1
        gr.Accordion(label=lang["vggt_tab"]["vggsfm"]["option"]["title"]), # vggsfm_option
        gr.Radio(label= lang["vggt_tab"]["vggsfm"]["option"]["exe_mode"]), # exe_mode_vggsfm
        gr.Button(value=lang["vggt_tab"]["vggsfm"]["recon_btn"]), # recon_vggsfm_btn
        gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["outdir_recon"]), # outdir_recon_vggsfm
        gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["runtime_recon"]), # runtime_recon_vggsfm
        gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["result_recon"]), # result_recon_vggsfm
        gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["log_recon"]), # log_recon_vggsfm
        gr.Markdown(lang["vggt_tab"]["vggsfm"]["subtitle2"]), # vggsfm_sub2
        gr.Button(value=lang["vggt_tab"]["vggsfm"]["export_btn"]), # export_vggsfm_btn
        gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["outdir_export"]), # outdir_export_vggsfm
        gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["runtime_export"]), # runtime_export_vggsfm
        gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["result_export"]), # result_export_vggsfm
        gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["log_export"]), # log_export_vggsfm
        gr.Model3D(label=lang["vggt_tab"]["vggsfm"]["outmodel"]), # outmodel_vggsfm
        # VGGT-SLAM
        gr.Tab(label=lang["vggt_tab"]["vggt-slam"]["title"]), # vggtslam_tab
        gr.Markdown(lang["vggt_tab"]["vggt-slam"]["subtitle1"]), # vggtslam_sub1
        gr.Accordion(label=lang["vggt_tab"]["vggt-slam"]["option"]["title"]), # vggtslam_option
        gr.Radio(label= lang["vggt_tab"]["vggt-slam"]["option"]["exe_mode"]), # exe_mode_vggtslam
        gr.Button(value=lang["vggt_tab"]["vggt-slam"]["recon_btn"]), # recon_vggtslam_btn
        gr.Markdown(lang["vggt_tab"]["vggt-slam"]["viewer"]), # vggtslam_viewer
        gr.Textbox(label=lang["vggt_tab"]["vggt-slam"]["outdir_recon"]), # outdir_recon_vggtslam
        gr.Textbox(label=lang["vggt_tab"]["vggt-slam"]["runtime_recon"]), # runtime_recon_vggtslam
        gr.Textbox(label=lang["vggt_tab"]["vggt-slam"]["result_recon"]), # result_recon_vggtslam
        gr.Textbox(label=lang["vggt_tab"]["vggt-slam"]["log_recon"]), # log_recon_vggtslam
        gr.Model3D(label=lang["vggt_tab"]["vggt-slam"]["outmodel"]), # outmodel_vggtslam
        # StreamVGGT
        gr.Tab(label=lang["vggt_tab"]["streamvggt"]["title"]), # stmvggt_tab
        gr.Markdown(lang["vggt_tab"]["streamvggt"]["subtitle1"]), # stmvggt_sub1
        gr.Accordion(label=lang["vggt_tab"]["streamvggt"]["option"]["title"]), # stmvggt_option
        gr.Radio(label= lang["vggt_tab"]["streamvggt"]["option"]["exe_mode"]), # exe_mode_stmvggt
        gr.Button(value=lang["vggt_tab"]["streamvggt"]["recon_btn"]), # recon_stmvggt_btn
        gr.Textbox(label=lang["vggt_tab"]["streamvggt"]["outdir_recon"]), # outdir_recon_stmvggt
        gr.Textbox(label=lang["vggt_tab"]["streamvggt"]["runtime_recon"]), # runtime_recon_stmvggt
        gr.Textbox(label=lang["vggt_tab"]["streamvggt"]["result_recon"]), # result_recon_stmvggt
        gr.Textbox(label=lang["vggt_tab"]["streamvggt"]["log_recon"]), # log_recon_stmvggt
        gr.Model3D(label=lang["vggt_tab"]["streamvggt"]["outmodel"]), # outmodel_stmvggt
        # FastVGGT
        gr.Tab(label=lang["vggt_tab"]["fastvggt"]["title"]), # fastvggt_tab
        gr.Markdown(lang["vggt_tab"]["fastvggt"]["subtitle1"]), # fastvggt_sub1
        gr.Accordion(label=lang["vggt_tab"]["fastvggt"]["option"]["title"]), # fastvggt_option
        gr.Radio(label= lang["vggt_tab"]["fastvggt"]["option"]["exe_mode"]), # exe_mode_fastvggt
        gr.Button(value=lang["vggt_tab"]["fastvggt"]["recon_btn"]), # recon_fastvggt_btn
        gr.Textbox(label=lang["vggt_tab"]["fastvggt"]["outdir_recon"]), # outdir_recon_fastvggt
        gr.Textbox(label=lang["vggt_tab"]["fastvggt"]["runtime_recon"]), # runtime_recon_fastvggt
        gr.Textbox(label=lang["vggt_tab"]["fastvggt"]["result_recon"]), # result_recon_fastvggt
        gr.Textbox(label=lang["vggt_tab"]["fastvggt"]["log_recon"]), # log_recon_fastvggt
        gr.Model3D(label=lang["vggt_tab"]["fastvggt"]["outmodel"]), # outmodel_fastvggt
        # Pi3
        gr.Tab(label=lang["vggt_tab"]["pi3"]["title"]), # pi3_tab
        gr.Markdown(lang["vggt_tab"]["pi3"]["subtitle1"]), # pi3_sub1
        gr.Accordion(label=lang["vggt_tab"]["pi3"]["option"]["title"]), # pi3_option
        gr.Radio(label= lang["vggt_tab"]["pi3"]["option"]["exe_mode"]), # exe_mode_pi3
        gr.Button(value=lang["vggt_tab"]["pi3"]["recon_btn"]), # recon_pi3_btn
        gr.Textbox(label=lang["vggt_tab"]["pi3"]["outdir_recon"]), # outdir_recon_pi3
        gr.Textbox(label=lang["vggt_tab"]["pi3"]["runtime_recon"]), # runtime_recon_pi3
        gr.Textbox(label=lang["vggt_tab"]["pi3"]["result_recon"]), # result_recon_pi3
        gr.Textbox(label=lang["vggt_tab"]["pi3"]["log_recon"]), # log_recon_pi3
        gr.Model3D(label=lang["vggt_tab"]["pi3"]["outmodel"]), # outmodel_pi3
        # mdsTab
        gr.Tab(label=lang["mds_tab"]["title"]), # mds_tab
        # moge2
        gr.Tab(label=lang["mds_tab"]["moge2"]["title"]), # moge2_tab
        gr.Markdown(lang["mds_tab"]["moge2"]["subtitle1"]), # moge2_sub1
        gr.Markdown(lang["mds_tab"]["moge2"]["info1"]), # moge2_info1
        gr.Image(label=lang["mds_tab"]["moge2"]["image"]), # img_moge2
        gr.Markdown(lang["mds_tab"]["moge2"]["subtitle2"]), # moge2_sub2
        gr.Accordion(label=lang["mds_tab"]["moge2"]["option"]["title"]), # moge2_option
        gr.Radio(label= lang["mds_tab"]["moge2"]["option"]["exe_mode"]), # exe_mode_moge2
        gr.Radio(choices=[lang["mds_tab"]["moge2"]["option"]["radio_standard"], lang["mds_tab"]["moge2"]["option"]["radio_panorama"]], 
                 value=lang["mds_tab"]["moge2"]["option"]["radio_default"]), # img_type_moge2
        gr.Button(value=lang["mds_tab"]["moge2"]["recon_btn"]), # recon_moge2_btn
        gr.Textbox(label=lang["mds_tab"]["moge2"]["outdir_recon"]), # outdir_recon_moge2
        gr.Textbox(label=lang["mds_tab"]["moge2"]["runtime_recon"]), # runtime_recon_moge2
        gr.Textbox(label=lang["mds_tab"]["moge2"]["result_recon"]), # result_recon_moge2
        gr.Textbox(label=lang["mds_tab"]["moge2"]["log_recon"]), # log_recon_moge2
        gr.Model3D(label=lang["mds_tab"]["moge2"]["outmodel"]), # outmodel_moge2
        # UniK3D
        gr.Tab(label=lang["mds_tab"]["unik3d"]["title"]), # unik3d_tab
        gr.Markdown(lang["mds_tab"]["unik3d"]["subtitle1"]), # unik3d_sub1
        gr.Markdown(lang["mds_tab"]["unik3d"]["info1"]), # unik3d_info1
        gr.Image(label=lang["mds_tab"]["unik3d"]["image"]), # img_unik3d
        gr.Markdown(lang["mds_tab"]["unik3d"]["subtitle2"]), # unik3d_sub2
        gr.Accordion(label=lang["mds_tab"]["unik3d"]["option"]["title"]), # unik3d_option
        gr.Radio(label= lang["mds_tab"]["unik3d"]["option"]["exe_mode"]), # exe_mode_unik3d
        gr.Button(value=lang["mds_tab"]["unik3d"]["recon_btn"]), # recon_unik3d_btn
        gr.Textbox(label=lang["mds_tab"]["unik3d"]["outdir_recon"]), # outdir_recon_unik3d
        gr.Textbox(label=lang["mds_tab"]["unik3d"]["runtime_recon"]), # runtime_recon_unik3d
        gr.Textbox(label=lang["mds_tab"]["unik3d"]["result_recon"]), # result_recon_unik3d
        gr.Textbox(label=lang["mds_tab"]["unik3d"]["log_recon"]), # log_recon_unik3d
        gr.Model3D(label=lang["mds_tab"]["unik3d"]["outmodel"]), # outmodel_unik3d
        # DA2
        gr.Tab(label=lang["mds_tab"]["da2"]["title"]), #da2_tab
        gr.Markdown(lang["mds_tab"]["da2"]["subtitle1"]), # da2_sub1
        gr.Radio(choices=[lang["mds_tab"]["da2"]["radio_image"], lang["mds_tab"]["da2"]["radio_video"]], 
                 label=lang["mds_tab"]["da2"]["input_radio"]), # da2_input_radio
        gr.Markdown(lang["mds_tab"]["da2"]["image_section"]["subtitle2"]), # da2_iamge_sub2
        gr.Markdown(lang["mds_tab"]["da2"]["image_section"]["info1"]), # da2_iamge_info1
        gr.Image(type="filepath", label=lang["mds_tab"]["da2"]["image_section"]["image"]), # image_da2
        gr.Markdown(lang["mds_tab"]["da2"]["subtitle3"]), # da2_image_sub3
        gr.Accordion(label=lang["mds_tab"]["da2"]["option"]["title"], open=False), #da2_option
        gr.Radio(label=lang["mds_tab"]["da2"]["option"]["exe_mode"]), # exe_mode_image_da2
        gr.Radio(label=lang["mds_tab"]["da2"]["option"]["exe_model"]), # exe_model_image_da2
        gr.Button(value=lang["mds_tab"]["da2"]["run_btn"]), # run_image_da2_btn
        gr.Textbox(label=lang["mds_tab"]["da2"]["outdir_recon"]), # outdir_image_da2
        gr.Textbox(label=lang["mds_tab"]["da2"]["runtime_recon"]), # runtime_image_da2
        gr.Textbox(label=lang["mds_tab"]["da2"]["result_recon"]), # result_image_da2
        gr.Textbox(label=lang["mds_tab"]["da2"]["log_recon"]), # log_image_da2
        gr.Gallery(label=lang["mds_tab"]["da2"]["outimage"]), # outimage_da2
        gr.Markdown(lang["mds_tab"]["da2"]["video_section"]["subtitle2"]), # da2_video_sub2
        gr.Markdown(lang["mds_tab"]["da2"]["video_section"]["info1"]), # da2_video_info1
        gr.Video(label=lang["mds_tab"]["da2"]["video_section"]["video"]), # video_da2
        gr.Markdown(lang["mds_tab"]["da2"]["subtitle3"]), # da2_video_sub3
        gr.Accordion(label=lang["mds_tab"]["da2"]["option"]["title"]), # da2_option
        gr.Radio(label=lang["mds_tab"]["da2"]["option"]["exe_mode"]), # exe_mode_video_da2
        gr.Radio(label=lang["mds_tab"]["da2"]["option"]["exe_model"]), # exe_model_video_da2
        gr.Button(value=lang["mds_tab"]["da2"]["run_btn"]), # run_video_da2_btn
        gr.Textbox(label=lang["mds_tab"]["da2"]["outdir_recon"]), # outdir_video_da2
        gr.Textbox(label=lang["mds_tab"]["da2"]["runtime_recon"]), # runtime_video_da2
        gr.Textbox(label=lang["mds_tab"]["da2"]["result_recon"]), # result_video_da2
        gr.Textbox(label=lang["mds_tab"]["da2"]["log_recon"]), # log_video_da2
        gr.Video(label=lang["mds_tab"]["da2"]["outvideo"]), # outvideo_da2
        # DA3
        gr.Markdown(lang["mds_tab"]["da3"]["subtitle1"]), # da3_sub1
        gr.Accordion(label=lang["mds_tab"]["da3"]["option"]["title"]), # da3_option
        gr.Radio(label=lang["mds_tab"]["da3"]["option"]["exe_mode"]), # exe_mode_da3
        gr.Button(value=lang["mds_tab"]["da3"]["run_btn"]), # recon_da3_btn
        gr.Textbox(label=lang["mds_tab"]["da3"]["outdir_recon"]), # outdir_da3
        gr.Textbox(label=lang["mds_tab"]["da3"]["runtime_recon"]), # runtime_da3
        gr.Textbox(label=lang["mds_tab"]["da3"]["result_recon"]), # result_da3
        gr.Textbox(label=lang["mds_tab"]["da3"]["log_recon"]), # log_da3
        gr.Model3D(label=lang["mds_tab"]["da3"]["outmodel"]), # outmodel_da3
        gr.Gallery(label=lang["mds_tab"]["da3"]["outimage"]), # gallery_da3
        gr.Video(label=lang["mds_tab"]["da3"]["outvideo"]), # outvideo_da3
        gr.Video(label=lang["mds_tab"]["da3"]["outGSvideo"]), # outGSvideo_da3
        # Metrics Tab
        gr.Tab(label=lang["metrics_tab"]["title"]), # metrics_tab
        gr.DataFrame(label=lang["metrics_tab"]["table"]), # method_metrics
        gr.DownloadButton(label=lang["metrics_tab"]["download_btn"]), # download_csv
    )

# メディアUI切り替えメソッド
def switch_dataset_ui(choice, lang_code):
    lang = load_translations(lang_code)

    if choice == lang["dataset_tab"]["radio_new"] :
        return gr.Column(visible=True), gr.Column(visible=False)
    elif choice == lang["dataset_tab"]["radio_load"] : 
        return gr.Column(visible=False), gr.Column(visible=True)
    
def switch_media_ui(choice, lang_code):
    lang = load_translations(lang_code)

    if choice == lang["dataset_tab"]["new_dataset_section"]["radio_image"] :
        return gr.Column(visible=True), gr.Column(visible=False)
    elif choice == lang["dataset_tab"]["new_dataset_section"]["radio_video"] : 
        return gr.Column(visible=False), gr.Column(visible=True)

def switch_da2_ui(choice, lang_code):
    lang = load_translations(lang_code)

    if choice == lang["mds_tab"]["da2"]["radio_image"] :
        return gr.Column(visible=True), gr.Column(visible=False)
    elif choice == lang["mds_tab"]["da2"]["radio_video"] : 
        return gr.Column(visible=False), gr.Column(visible=True)


def col_change():
    return gr.Column(visible=True)

# State_value代入メソッド
def get_state_value(state):
    return state

def get_state_value2(state):
    return state, state

# 評価指標タブのテーブル更新メソッド
def update_method_metrics(table, values, save_dir):
    # CSV ファイル名
    csv_path = os.path.join(save_dir, "method_metrics.csv")

    # DataFrame を更新
    table = pd.concat([table, values], ignore_index=True)
    # CSV に保存
    table.to_csv(csv_path, index=False)

    return table, csv_path

# GradioUI
def main_demo(tmpdir, datasetsdir, outputsdir):

    # デフォルト言語
    lang = load_translations("jp")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        tmpdir_state = gr.State(tmpdir)
        datasetsdir_state = gr.State(datasetsdir)
        outputsdir_state = gr.State(outputsdir)
        lang_state = gr.State("jp")
        image_dataset_state = gr.State("")
        colmap_dataset_state = gr.State("")

        # 言語切り替えボタン
        language_radio = gr.Radio(choices=["日本語", "ENGLISH"], value="日本語", label="🌐言語 / Language")

        with gr.Row():
            # 現在の画像データセット
            current_dataset_images = gr.Textbox(label=lang["current_dataset_images"])
            # 現在のCOLMAPデータセット
            current_dataset_colmap = gr.Textbox(label=lang["current_dataset_colmap"])

        # DatasetTab
        with gr.Tab(label=lang["dataset_tab"]["title"]) as dataset_tab:
            dataset_sub1 = gr.Markdown(lang["dataset_tab"]["subtitle1"])
            dataset_info1 = gr.Markdown(lang["dataset_tab"]["info1"])
            dataset_radio = gr.Radio(choices=[lang["dataset_tab"]["radio_new"], lang["dataset_tab"]["radio_load"]], 
                                     label=lang["dataset_tab"]["dataset_radio"])
            
            with gr.Column(visible=False) as new_dataset_col:
                dataset_new_sub2 = gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["subtitle2"])
                media_radio = gr.Radio(choices=[lang["dataset_tab"]["new_dataset_section"]["radio_image"], lang["dataset_tab"]["new_dataset_section"]["radio_video"]], 
                                    label=lang["dataset_tab"]["new_dataset_section"]["media_radio"])

                # 画像入力UI
                with gr.Column(visible=False) as image_col:
                    dataset_image_sub3 = gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["image_section"]["subtitle3"])
                    images = gr.File(label=lang["dataset_tab"]["new_dataset_section"]["image_section"]["images"], file_types=["image"], file_count="multiple")
                    dataset_name = gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["image_section"]["dataset_name"], info=lang["dataset_tab"]["new_dataset_section"]["image_section"]["dataset_name_info"])
                    run_copy_btn = gr.Button(value=lang["dataset_tab"]["new_dataset_section"]["image_section"]["run_copy_btn"])
                    with gr.Column(visible=False) as iresult_col:
                        dataset_image_sub4 = gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["image_section"]["subtitle4"])
                        output_image = gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["image_section"]["output_image"])
                        gallery_image = gr.Gallery(label=lang["dataset_tab"]["new_dataset_section"]["image_section"]["gallery_image"], columns=4, height="auto")

                # 動画入力UI
                with gr.Column(visible=False) as video_col:
                    dataset_video_sub3 = gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["video_section"]["subtitle3"])
                    video = gr.Video(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["video"])
                    fps = gr.Slider(value=3, minimum=1, maximum=5, step=1, label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["fps"])
                    with gr.Accordion(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["title"], open=False) as dataset_video_option:
                        dataset_video_option_subtitle = gr.Markdown(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["subtitle"])
                        dataset_video_option_info1 = gr.Markdown(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["info1"])
                        rsi = gr.Checkbox(value=True, label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["rsi"])
                        dataset_video_option_info2 = gr.Markdown(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["info2"])
                        ssim = gr.Slider(value=0.8, minimum=0, maximum=1, label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["option"]["ssim"])
                    run_ffmpeg_btn = gr.Button(value=lang["dataset_tab"]["new_dataset_section"]["video_section"]["run_ffmpeg_btn"])
                    output_video = gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["output_video"])
                    with gr.Row(equal_height=True):
                        comp_rate = gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["comp_rate"])
                        sel_images_num = gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["sel_images_num"])
                        rej_images_num = gr.Textbox(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["rej_images_num"])
                    gallery_video = gr.Gallery(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["gallery_video"], columns=4, height="auto")
                    with gr.Column(visible=False) as dl_images_col:
                        dataset_video_sub4 = gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["video_section"]["subtitle4"])
                        dataset_video_info1 = gr.Markdown(lang["dataset_tab"]["new_dataset_section"]["video_section"]["info1"])
                        zipfile_images = gr.DownloadButton(label=lang["dataset_tab"]["new_dataset_section"]["video_section"]["download_zipfile_btn"])
            
            with gr.Column(visible=False) as load_dataset_col:
                dataset_load_sub2 = gr.Markdown(lang["dataset_tab"]["load_dataset_section"]["subtitle2"])
                with gr.Row(equal_height=True):
                    load_dataset_info1 = gr.Markdown(lang["dataset_tab"]["load_dataset_section"]["info1"])
                    gr.Image(value=os.path.join("src", "external_dataset.png"))
                load_dataset = gr.File(label=lang["dataset_tab"]["load_dataset_section"]["load_dataset"], file_types=[".zip"], type="filepath")
                log_unzip = gr.Textbox(label=lang["dataset_tab"]["load_dataset_section"]["log_unzip"])

        # COLMAPTab
        with gr.Tab(label=lang["colmap_tab"]["title"]) as colmap_tab:
            colmap_sub1 = gr.Markdown(lang["colmap_tab"]["subtitle1"])
            colmap_info1 = gr.Markdown(lang["colmap_tab"]["info1"])
            with gr.Accordion(label=lang["colmap_tab"]["option"]["title"], open=False) as colmap_option:
                exe_mode_colmap = gr.Radio(choices=["local", "slurm"], value="local", label= lang["colmap_tab"]["option"]["exe_mode"])
                colmap_option_info1 = gr.Markdown(lang["colmap_tab"]["option"]["info1"])
                rebuild = gr.Checkbox(label=lang["colmap_tab"]["option"]["rebuild"], value=False)
            run_colmap_btn = gr.Button(value=lang["colmap_tab"]["run_colmap_btn"])
            result_colmap = gr.Textbox(label=lang["colmap_tab"]["result_colmap"])
            with gr.Column(visible=False) as dl_colmap_col:
                colmap_sub2 = gr.Markdown(lang["colmap_tab"]["subtitle2"])
                colmap_info2 = gr.Markdown(lang["colmap_tab"]["info2"])
                zipfile_colmap = gr.DownloadButton(label=lang["colmap_tab"]["download_zipfile_btn"])
      
        # NeRFTab
        with gr.Tab(label=lang["nerf_tab"]["title"]) as nerf_tab:

            # Vanilla-NeRF
            with gr.Tab(label=lang["nerf_tab"]["vnerf"]["title"]) as vnerf_tab:
                vnerf_sub1 = gr.Markdown(lang["nerf_tab"]["vnerf"]["subtitle1"])
                with gr.Accordion(label=lang["nerf_tab"]["vnerf"]["option"]["title"], open=False) as vnerf_option:
                    exe_mode_vnerf = gr.Radio(choices=["local", "slurm"], value="local", label= lang["nerf_tab"]["vnerf"]["option"]["exe_mode"])
                    iter_vnerf = gr.Slider(value=1000000, minimum=25000, maximum=2000000, step=25000, label=lang["nerf_tab"]["vnerf"]["option"]["iter"])
                recon_vnerf_btn = gr.Button(value=lang["nerf_tab"]["vnerf"]["recon_btn"])
                vnerf_viewer = gr.Markdown(lang["nerf_tab"]["vnerf"]["viewer"])
                outdir_recon_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["outdir_recon"])
                runtime_recon_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["runtime_recon"])
                result_recon_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["result_recon"])
                log_recon_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["log_recon"])
                with gr.Column(visible=False) as export_vnerf_col:
                    vnerf_sub2 = gr.Markdown(lang["nerf_tab"]["vnerf"]["subtitle2"])
                    export_vnerf_btn = gr.Button(value=lang["nerf_tab"]["vnerf"]["export_btn"])
                    outdir_export_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["outdir_export"])
                    runtime_export_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["runtime_export"])
                    result_export_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["result_export"])
                    log_export_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["log_export"])
                with gr.Column(visible=False) as eval_vnerf_col:
                    vnerf_sub3 = gr.Markdown(lang["nerf_tab"]["vnerf"]["subtitle3"])
                    eval_vnerf_btn = gr.Button(value=lang["nerf_tab"]["vnerf"]["eval_btn"])
                    outdir_eval_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["outdir_eval"])
                    runtime_eval_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["runtime_eval"])
                    result_eval_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["result_eval"])
                    log_eval_vnerf = gr.Textbox(label=lang["nerf_tab"]["vnerf"]["log_eval"])
                    metrics_vnerf = gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["nerf_tab"]["vnerf"]["metrics"])
                    gallery_vnerf = gr.Gallery(label=lang["nerf_tab"]["vnerf"]["gallery"], columns=2, height="auto")
            
            # Nerfacto
            with gr.Tab(label=lang["nerf_tab"]["nerfacto"]["title"]) as nerfacto_tab:
                nerfacto_sub1 = gr.Markdown(lang["nerf_tab"]["nerfacto"]["subtitle1"])
                with gr.Accordion(label=lang["nerf_tab"]["nerfacto"]["option"]["title"], open=False) as nerfacto_option:
                    exe_mode_nerfacto = gr.Radio(choices=["local", "slurm"], value="local", label= lang["nerf_tab"]["nerfacto"]["option"]["exe_mode"])
                    iter_nerfacto = gr.Slider(value=100000, minimum=25000, maximum=200000, step=25000, label=lang["nerf_tab"]["nerfacto"]["option"]["iter"])
                recon_nerfacto_btn = gr.Button(value=lang["nerf_tab"]["nerfacto"]["recon_btn"])
                nerfacto_viewer = gr.Markdown(lang["nerf_tab"]["nerfacto"]["viewer"])
                outdir_recon_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["outdir_recon"])
                runtime_recon_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["runtime_recon"])
                result_recon_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["result_recon"])
                log_recon_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["log_recon"])
                with gr.Column(visible=False) as export_nerfacto_col:
                    nerfacto_sub2 = gr.Markdown(lang["nerf_tab"]["nerfacto"]["subtitle2"])
                    export_nerfacto_btn = gr.Button(value=lang["nerf_tab"]["nerfacto"]["export_btn"])
                    outdir_export_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["outdir_export"])
                    runtime_export_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["runtime_export"])
                    result_export_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["result_export"])
                    log_export_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["log_export"])
                with gr.Column(visible=False) as eval_nerfacto_col:
                    nerfacto_sub3 = gr.Markdown(lang["nerf_tab"]["nerfacto"]["subtitle3"])
                    eval_nerfacto_btn = gr.Button(value=lang["nerf_tab"]["nerfacto"]["eval_btn"])
                    outdir_eval_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["outdir_eval"])
                    runtime_eval_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["runtime_eval"])
                    result_eval_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["result_eval"])
                    log_eval_nerfacto = gr.Textbox(label=lang["nerf_tab"]["nerfacto"]["log_eval"])
                    metrics_nerfacto = gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["nerf_tab"]["nerfacto"]["metrics"])
                    gallery_nerfacto = gr.Gallery(label=lang["nerf_tab"]["nerfacto"]["gallery"], columns=2, height="auto")
            
            # mip-NeRF
            with gr.Tab(label=lang["nerf_tab"]["mip-nerf"]["title"]) as mipnerf_tab:
                mipnerf_sub1 = gr.Markdown(lang["nerf_tab"]["mip-nerf"]["subtitle1"])
                with gr.Accordion(label=lang["nerf_tab"]["mip-nerf"]["option"]["title"], open=False) as mipnerf_option:
                    exe_mode_mipnerf = gr.Radio(choices=["local", "slurm"], value="local", label= lang["nerf_tab"]["mip-nerf"]["option"]["exe_mode"])
                    iter_mipnerf = gr.Slider(value=1000000, minimum=25000, maximum=2000000, step=25000, label=lang["nerf_tab"]["mip-nerf"]["option"]["iter"])
                recon_mipnerf_btn = gr.Button(value=lang["nerf_tab"]["mip-nerf"]["recon_btn"])
                mipnerf_viewer = gr.Markdown(lang["nerf_tab"]["mip-nerf"]["viewer"])
                outdir_recon_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["outdir_recon"])
                runtime_recon_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["runtime_recon"])
                result_recon_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["result_recon"])
                log_recon_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["log_recon"])
                with gr.Column(visible=False) as export_mipnerf_col:
                    mipnerf_sub2 = gr.Markdown(lang["nerf_tab"]["mip-nerf"]["subtitle2"])
                    export_mipnerf_btn = gr.Button(value=lang["nerf_tab"]["mip-nerf"]["export_btn"])
                    outdir_export_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["outdir_export"])
                    runtime_export_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["runtime_export"])
                    result_export_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["result_export"])
                    log_export_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["log_export"])
                with gr.Column(visible=False) as eval_mipnerf_col:
                    mipnerf_sub3 = gr.Markdown(lang["nerf_tab"]["mip-nerf"]["subtitle3"])
                    eval_mipnerf_btn = gr.Button(value=lang["nerf_tab"]["mip-nerf"]["eval_btn"])
                    outdir_eval_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["outdir_eval"])
                    runtime_eval_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["runtime_eval"])
                    result_eval_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["result_eval"])
                    log_eval_mipnerf = gr.Textbox(label=lang["nerf_tab"]["mip-nerf"]["log_eval"])
                    metrics_mipnerf = gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["nerf_tab"]["mip-nerf"]["metrics"])
                    gallery_mipnerf = gr.Gallery(label=lang["nerf_tab"]["mip-nerf"]["gallery"], columns=2, height="auto")
            
            # SeaThru-NeRF
            with gr.Tab(label=lang["nerf_tab"]["seathru-nerf"]["title"]) as stnerf_tab:
                stnerf_sub1 = gr.Markdown(lang["nerf_tab"]["seathru-nerf"]["subtitle1"])
                with gr.Accordion(label=lang["nerf_tab"]["seathru-nerf"]["option"]["title"], open=False) as stnerf_option:
                    exe_mode_stnerf = gr.Radio(choices=["local", "slurm"], value="local", label= lang["nerf_tab"]["seathru-nerf"]["option"]["exe_mode"])
                    iter_stnerf = gr.Slider(value=100000, minimum=25000, maximum=200000, step=25000, label=lang["nerf_tab"]["seathru-nerf"]["option"]["iter"])
                recon_stnerf_btn = gr.Button(value=lang["nerf_tab"]["seathru-nerf"]["recon_btn"])
                stnerf_viewer = gr.Markdown(lang["nerf_tab"]["seathru-nerf"]["viewer"])
                outdir_recon_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["outdir_recon"])
                runtime_recon_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["runtime_recon"])
                result_recon_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["result_recon"])
                log_recon_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["log_recon"])
                with gr.Column(visible=False) as export_stnerf_col:
                    stnerf_sub2 = gr.Markdown(lang["nerf_tab"]["seathru-nerf"]["subtitle2"])
                    gr.Markdown("※Nerfstudioでサポートされていません")
                    export_stnerf_btn = gr.Button(value=lang["nerf_tab"]["seathru-nerf"]["export_btn"])
                    outdir_export_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["outdir_export"])
                    runtime_export_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["runtime_export"])
                    result_export_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["result_export"])
                    log_export_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["log_export"])
                    gallery_stnerf = gr.Gallery(label=lang["nerf_tab"]["seathru-nerf"]["gallery"], columns=2, height="auto")
                with gr.Column(visible=False) as eval_stnerf_col:
                    stnerf_sub3 = gr.Markdown(lang["nerf_tab"]["seathru-nerf"]["subtitle3"])
                    eval_stnerf_btn = gr.Button(value=lang["nerf_tab"]["seathru-nerf"]["eval_btn"])
                    outdir_eval_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["outdir_eval"])
                    runtime_eval_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["runtime_eval"])
                    result_eval_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["result_eval"])
                    log_eval_stnerf = gr.Textbox(label=lang["nerf_tab"]["seathru-nerf"]["log_eval"])
                    metrics_stnerf = gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["nerf_tab"]["seathru-nerf"]["metrics"])
                    gallery_stnerf = gr.Gallery(label=lang["nerf_tab"]["seathru-nerf"]["gallery"], columns=2, height="auto")

        # GSTab         
        with gr.Tab(label=lang["gs_tab"]["title"]) as gs_tab:

            # Vanilla GS
            with gr.Tab(label=lang["gs_tab"]["vgs"]["title"]) as vgs_tab:
                vgs_sub1 = gr.Markdown(lang["gs_tab"]["vgs"]["subtitle1"])
                with gr.Accordion(label=lang["gs_tab"]["vgs"]["option"]["title"], open=False) as vgs_option:
                    exe_mode_vgs = gr.Radio(choices=["local", "slurm"], value="local", label= lang["gs_tab"]["vgs"]["option"]["exe_mode"])
                    gr.Markdown("※未実装")
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
                                test_iter_3dgs = gr.Slider(value=30000, minimum=0, maximum=50000, step=100, label="テストを実行するイテレーション数")
                            with gr.Column():    
                                save_iter_3dgs = gr.Slider(value=30000, minimum=0, maximum=50000, step=100, label="モデルを保存するイテレーション数")
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
                recon_vgs_btn = gr.Button(value=lang["gs_tab"]["vgs"]["recon_btn"])
                outdir_recon_vgs = gr.Textbox(interactive=False, label=lang["gs_tab"]["vgs"]["outdir_recon"])
                runtime_recon_vgs = gr.Textbox(label=lang["gs_tab"]["vgs"]["runtime_recon"])
                result_recon_vgs = gr.Textbox(label=lang["gs_tab"]["vgs"]["result_recon"])
                log_recon_vgs = gr.Textbox(label=lang["gs_tab"]["vgs"]["log_recon"])
                outmodel_vgs = gr.Model3D(label=lang["gs_tab"]["vgs"]["outmodel"])
                with gr.Column(visible=False) as eval_vgs_col:            
                    vgs_sub2 = gr.Markdown(lang["gs_tab"]["vgs"]["subtitle2"])
                    with gr.Row():
                        skip_train = gr.Checkbox(value=True, label=lang["gs_tab"]["vgs"]["skip_train"])
                        skip_test = gr.Checkbox(value=False, label=lang["gs_tab"]["vgs"]["skip_test"])
                    eval_vgs_btn = gr.Button(value=lang["gs_tab"]["vgs"]["eval_btn"])
                    outdir_eval_vgs = gr.Textbox(label=lang["gs_tab"]["vgs"]["outdir_eval"])
                    runtime_eval_vgs = gr.Textbox(label=lang["gs_tab"]["vgs"]["runtime_eval"])
                    result_eval_vgs = gr.Textbox(label=lang["gs_tab"]["vgs"]["result_eval"])
                    log_eval_vgs = gr.Textbox(label=lang["gs_tab"]["vgs"]["log_eval"])
                    metrics_vgs = gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["gs_tab"]["vgs"]["metrics"])
                    gallery_vgs = gr.Gallery(label=lang["gs_tab"]["vgs"]["gallery"], columns=2, height="auto")

            # Mip-Splatting
            with gr.Tab(label=lang["gs_tab"]["mip-splatting"]["title"]) as mips_tab:
                mips_sub1 = gr.Markdown(lang["gs_tab"]["mip-splatting"]["subtitle1"])
                with gr.Accordion(label=lang["gs_tab"]["mip-splatting"]["option"]["title"], open=False) as mips_option:
                    exe_mode_mips = gr.Radio(choices=["local", "slurm"], value="local", label= lang["gs_tab"]["mip-splatting"]["option"]["exe_mode"])
                    save_iter_mips = gr.Slider(value=30000, minimum=0, maximum=50000, step=100, label=lang["gs_tab"]["mip-splatting"]["option"]["save_iter"])     
                recon_mips_btn = gr.Button(value=lang["gs_tab"]["mip-splatting"]["recon_btn"])
                outdir_recon_mips = gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["outdir_recon"])
                runtime_recon_mips = gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["runtime_recon"])
                result_recon_mips = gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["result_recon"])
                log_recon_mips = gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["log_recon"])
                outmodel_mips = gr.Model3D(label=lang["gs_tab"]["mip-splatting"]["outmodel"])
                with gr.Column(visible=False) as eval_mips_col:
                    mips_sub2 = gr.Markdown(lang["gs_tab"]["mip-splatting"]["subtitle2"])
                    eval_mips_btn = gr.Button(value=lang["gs_tab"]["mip-splatting"]["eval_btn"])
                    outdir_eval_mips = gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["outdir_eval"])
                    runtime_eval_mips = gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["runtime_eval"])
                    result_eval_mips = gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["result_eval"])
                    log_eval_mips = gr.Textbox(label=lang["gs_tab"]["mip-splatting"]["log_eval"])
                    metrics_mips = gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["gs_tab"]["mip-splatting"]["metrics"])
                    gallery_mips = gr.Gallery(label=lang["gs_tab"]["mip-splatting"]["gallery"], columns=2, height="auto")

            # Splatfacto
            with gr.Tab(label=lang["gs_tab"]["splatfacto"]["title"]) as sfacto_tab:
                sfacto_sub1 = gr.Markdown(lang["gs_tab"]["splatfacto"]["subtitle1"])
                with gr.Accordion(label=lang["gs_tab"]["splatfacto"]["option"]["title"], open=False) as sfacto_option:
                    exe_mode_sfacto = gr.Radio(choices=["local", "slurm"], value="local", label= lang["gs_tab"]["splatfacto"]["option"]["exe_mode"])
                    iter_sfacto = gr.Slider(value=30000, minimum=0, maximum=50000, step=2000, label=lang["gs_tab"]["splatfacto"]["option"]["iter"])
                recon_sfacto_btn = gr.Button(value=lang["gs_tab"]["splatfacto"]["recon_btn"])
                sfacto_viewer = gr.Markdown(lang["gs_tab"]["splatfacto"]["viewer"])
                outdir_recon_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["outdir_recon"])
                runtime_recon_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["runtime_recon"])
                result_recon_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["result_recon"])
                log_recon_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["log_recon"])
                with gr.Column(visible=False) as export_sfacto_col:
                    sfacto_sub2 = gr.Markdown(lang["gs_tab"]["splatfacto"]["subtitle2"])
                    export_sfacto_btn = gr.Button(value=lang["gs_tab"]["splatfacto"]["export_btn"])
                    outdir_export_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["outdir_export"])
                    runtime_export_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["runtime_export"])
                    result_export_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["result_export"])
                    log_export_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["log_export"])
                with gr.Column(visible=False) as eval_sfacto_col:
                    sfacto_sub3 = gr.Markdown(lang["gs_tab"]["splatfacto"]["subtitle3"])
                    eval_sfacto_btn = gr.Button(value=lang["gs_tab"]["splatfacto"]["eval_btn"])
                    outdir_eval_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["outdir_eval"])
                    runtime_eval_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["runtime_eval"])
                    result_eval_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["result_eval"])
                    log_eval_sfacto = gr.Textbox(label=lang["gs_tab"]["splatfacto"]["log_eval"])
                    metrics_sfacto = gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["gs_tab"]["splatfacto"]["metrics"])
                    gallery_sfacto = gr.Gallery(label=lang["gs_tab"]["splatfacto"]["gallery"], columns=2, height="auto")

            # 4D-Gaussians
            with gr.Tab(label=lang["gs_tab"]["4d-gaussians"]["title"]) as gs4d_tab:
                gs4d_sub1 = gr.Markdown(lang["gs_tab"]["4d-gaussians"]["subtitle1"])
                with gr.Accordion(lang["gs_tab"]["4d-gaussians"]["option"]["title"], open=False) as gs4d_option:
                    exe_mode_4dgs = gr.Radio(choices=["local", "slurm"], value="local", label= lang["gs_tab"]["4d-gaussians"]["option"]["exe_mode"])
                    save_iter_4dgs = gr.Slider(value=30000, minimum=0, maximum=50000, step=100, label=lang["gs_tab"]["4d-gaussians"]["option"]["save_iter"])     
                recon_4dgs_btn = gr.Button(value=lang["gs_tab"]["4d-gaussians"]["recon_btn"])
                outdir_recon_4dgs = gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["outdir_recon"])
                runtime_recon_4dgs = gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["runtime_recon"])
                result_recon_4dgs = gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["result_recon"])
                log_recon_4dgs = gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["log_recon"])
                outmodel_4dgs = gr.Model3D(label=lang["gs_tab"]["4d-gaussians"]["outmodel"])
                with gr.Column(visible=False) as eval_4dgs_col:
                    gs4d_sub2 = gr.Markdown(lang["gs_tab"]["4d-gaussians"]["subtitle2"])
                    eval_4dgs_btn = gr.Button(value=lang["gs_tab"]["4d-gaussians"]["eval_btn"])
                    outdir_eval_4dgs = gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["outdir_eval"])
                    runtime_eval_4dgs = gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["runtime_eval"])
                    result_eval_4dgs = gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["result_eval"])
                    log_eval_4dgs = gr.Textbox(label=lang["gs_tab"]["4d-gaussians"]["log_eval"])
                    metrics_4dgs = gr.DataFrame(headers=lang["metrics_tab"]["headers"], label=lang["gs_tab"]["4d-gaussians"]["metrics"])
                    gallery_4dgs = gr.Gallery(label=lang["gs_tab"]["4d-gaussians"]["gallery"], columns=2, height="auto")

        # 3stersTab
        with gr.Tab(label=lang["3sters_tab"]["title"]) as esters_tab:

            # DUSt3R
            with gr.Tab(label=lang["3sters_tab"]["dust3r"]["title"]) as dust3r_tab:
                dust3r_sub1 = gr.Markdown(lang["3sters_tab"]["dust3r"]["subtitle1"])
                with gr.Accordion(label=lang["3sters_tab"]["dust3r"]["option"]["title"], open=False) as dust3r_option:
                    exe_mode_dust3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["dust3r"]["option"]["exe_mode"])
                    gr.Markdown("※未実装")
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
                recon_dust3r_btn = gr.Button(value=lang["3sters_tab"]["dust3r"]["recon_btn"])
                outdir_recon_dust3r = gr.Textbox(label=lang["3sters_tab"]["dust3r"]["outdir_recon"])
                runtime_recon_dust3r = gr.Textbox(label=lang["3sters_tab"]["dust3r"]["runtime_recon"])
                result_recon_dust3r = gr.Textbox(label=lang["3sters_tab"]["dust3r"]["result_recon"])
                log_recon_dust3r = gr.Textbox(label=lang["3sters_tab"]["dust3r"]["log_recon"])
                outmodel_dust3r = gr.Model3D(label=lang["3sters_tab"]["dust3r"]["outmodel"])
                outimages_dust3r = gr.State()
                gallery_dust3r = gr.Gallery(label=lang["3sters_tab"]["dust3r"]["gallery"], columns=3, height="auto")

            # MASt3R
            with gr.Tab(label=lang["3sters_tab"]["mast3r"]["title"]) as mast3r_tab:
                mast3r_sub1 = gr.Markdown(lang["3sters_tab"]["mast3r"]["subtitle1"])
                with gr.Accordion(lang["3sters_tab"]["mast3r"]["option"]["title"], open=False) as mast3r_option:
                    exe_mode_mast3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["mast3r"]["option"]["exe_mode"])
                recon_mast3r_btn = gr.Button(value=lang["3sters_tab"]["mast3r"]["recon_btn"])
                outdir_recon_mast3r = gr.Textbox(label=lang["3sters_tab"]["mast3r"]["outdir_recon"])
                runtime_recon_mast3r = gr.Textbox(label=lang["3sters_tab"]["mast3r"]["runtime_recon"])
                result_recon_mast3r = gr.Textbox(label=lang["3sters_tab"]["mast3r"]["result_recon"])
                log_recon_mast3r = gr.Textbox(label=lang["3sters_tab"]["mast3r"]["log_recon"])
                outmodel_mast3r = gr.Model3D(label=lang["3sters_tab"]["mast3r"]["outmodel"])

            # MonST3R
            with gr.Tab(label=lang["3sters_tab"]["monst3r"]["title"]) as monst3r_tab:
                monst3r_sub1 = gr.Markdown(lang["3sters_tab"]["monst3r"]["subtitle1"])
                with gr.Accordion(label=lang["3sters_tab"]["monst3r"]["option"]["title"], open=False) as monst3r_option:
                    exe_mode_monst3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["monst3r"]["option"]["exe_mode"])
                recon_monst3r_btn = gr.Button(value=lang["3sters_tab"]["monst3r"]["recon_btn"])
                outdir_recon_monst3r = gr.Textbox(label=lang["3sters_tab"]["monst3r"]["outdir_recon"])
                runtime_recon_monst3r = gr.Textbox(label=lang["3sters_tab"]["monst3r"]["runtime_recon"])
                result_recon_monst3r = gr.Textbox(label=lang["3sters_tab"]["monst3r"]["result_recon"])
                log_recon_monst3r = gr.Textbox(label=lang["3sters_tab"]["monst3r"]["log_recon"])
                outmodel_monst3r = gr.Model3D(label=lang["3sters_tab"]["monst3r"]["outmodel"])
            
            # Easi3R
            with gr.Tab(label=lang["3sters_tab"]["easi3r"]["title"]) as easi3r_tab:
                easi3r_sub1 = gr.Markdown(lang["3sters_tab"]["easi3r"]["subtitle1"])
                easi3r_info1 = gr.Markdown(lang["3sters_tab"]["easi3r"]["info1"])
                with gr.Accordion(label=lang["3sters_tab"]["easi3r"]["option"]["title"], open=False) as easi3r_option:
                    exe_mode_easi3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["easi3r"]["option"]["exe_mode"])
                recon_easi3r_btn = gr.Button(value=lang["3sters_tab"]["easi3r"]["recon_btn"])
                outdir_recon_easi3r = gr.Textbox(label=lang["3sters_tab"]["easi3r"]["outdir_recon"])
                runtime_recon_easi3r = gr.Textbox(label=lang["3sters_tab"]["easi3r"]["runtime_recon"])
                result_recon_easi3r = gr.Textbox(label=lang["3sters_tab"]["easi3r"]["result_recon"])
                log_recon_easi3r = gr.Textbox(label=lang["3sters_tab"]["easi3r"]["log_recon"])
                outmodel_easi3r = gr.Model3D(label=lang["3sters_tab"]["easi3r"]["outmodel"])

            # MUSt3R
            with gr.Tab(label=lang["3sters_tab"]["must3r"]["title"]) as must3r_tab:
                must3r_sub1 = gr.Markdown(lang["3sters_tab"]["must3r"]["subtitle1"])
                with gr.Accordion(label=lang["3sters_tab"]["must3r"]["option"]["title"], open=False) as must3r_option:
                    exe_mode_must3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["must3r"]["option"]["exe_mode"])
                recon_must3r_btn = gr.Button(value=lang["3sters_tab"]["must3r"]["recon_btn"])
                outdir_recon_must3r = gr.Textbox(label=lang["3sters_tab"]["must3r"]["outdir_recon"])
                runtime_recon_must3r = gr.Textbox(label=lang["3sters_tab"]["must3r"]["runtime_recon"])
                result_recon_must3r = gr.Textbox(label=lang["3sters_tab"]["must3r"]["result_recon"])
                log_recon_must3r = gr.Textbox(label=lang["3sters_tab"]["must3r"]["log_recon"])
                outmodel_must3r = gr.Model3D(label=lang["3sters_tab"]["must3r"]["outmodel"])

            # Fast3R
            with gr.Tab(label=lang["3sters_tab"]["fast3r"]["title"]) as fast3r_tab:
                fast3r_sub1 = gr.Markdown(lang["3sters_tab"]["fast3r"]["subtitle1"])
                with gr.Accordion(label=lang["3sters_tab"]["fast3r"]["option"]["title"], open=False) as fast3r_option:
                    exe_mode_fast3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["fast3r"]["option"]["exe_mode"])
                recon_fast3r_btn = gr.Button(value=lang["3sters_tab"]["fast3r"]["recon_btn"])
                outdir_recon_fast3r = gr.Textbox(label=lang["3sters_tab"]["fast3r"]["outdir_recon"])
                runtime_recon_fast3r = gr.Textbox(label=lang["3sters_tab"]["fast3r"]["runtime_recon"])
                result_recon_fast3r = gr.Textbox(label=lang["3sters_tab"]["fast3r"]["result_recon"])
                log_recon_fast3r = gr.Textbox(label=lang["3sters_tab"]["fast3r"]["log_recon"])
                outmodel_fast3r = gr.Model3D(label=lang["3sters_tab"]["fast3r"]["outmodel"])

            # Splatt3R
            with gr.Tab(label=lang["3sters_tab"]["splatt3r"]["title"]) as splatt3r_tab:
                splatt3r_sub1 = gr.Markdown(lang["3sters_tab"]["splatt3r"]["subtitle1"])
                splatt3r_info1 = gr.Markdown(lang["3sters_tab"]["splatt3r"]["info1"])
                img_splatt3r = gr.Image(type="filepath", label=lang["3sters_tab"]["splatt3r"]["image"])
                # 推論UI
                with gr.Column(visible=False) as infer_splatt3r_col:
                    splatt3r_sub2 = gr.Markdown(lang["3sters_tab"]["splatt3r"]["subtitle2"])
                    with gr.Accordion(label=lang["3sters_tab"]["splatt3r"]["option"]["title"], open=False) as splatt3r_option:
                        exe_mode_splatt3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["splatt3r"]["option"]["exe_mode"])
                    recon_splatt3r_btn = gr.Button(value=lang["3sters_tab"]["splatt3r"]["recon_btn"])
                    outdir_recon_splatt3r = gr.Textbox(label=lang["3sters_tab"]["splatt3r"]["outdir_recon"])
                    runtime_recon_splatt3r = gr.Textbox(label=lang["3sters_tab"]["splatt3r"]["runtime_recon"])
                    result_recon_splatt3r = gr.Textbox(label=lang["3sters_tab"]["splatt3r"]["result_recon"])
                    log_recon_splatt3r = gr.Textbox(label=lang["3sters_tab"]["splatt3r"]["log_recon"])
                    outmodel_splatt3r = gr.Model3D(label=lang["3sters_tab"]["splatt3r"]["outmodel"], clear_color=[1.0, 1.0, 1.0, 0.0])

            # CUT3R
            with gr.Tab(label=lang["3sters_tab"]["cut3r"]["title"]) as cut3r_tab:
                cut3r_sub1 = gr.Markdown(lang["3sters_tab"]["cut3r"]["subtitle1"])
                with gr.Accordion(label=lang["3sters_tab"]["cut3r"]["option"]["title"], open=False) as cut3r_option:
                    exe_mode_cut3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["cut3r"]["option"]["exe_mode"])
                recon_cut3r_btn = gr.Button(value=lang["3sters_tab"]["cut3r"]["recon_btn"])
                outdir_recon_cut3r = gr.Textbox(label=lang["3sters_tab"]["cut3r"]["outdir_recon"])
                runtime_recon_cut3r = gr.Textbox(label=lang["3sters_tab"]["cut3r"]["runtime_recon"])
                result_recon_cut3r = gr.Textbox(label=lang["3sters_tab"]["cut3r"]["result_recon"])
                log_recon_cut3r = gr.Textbox(label=lang["3sters_tab"]["cut3r"]["log_recon"])
                outmodel_cut3r = gr.Model3D(label=lang["3sters_tab"]["cut3r"]["outmodel"])

            # WinT3R
            with gr.Tab(label=lang["3sters_tab"]["wint3r"]["title"]) as wint3r_tab:
                wint3r_sub1 = gr.Markdown(lang["3sters_tab"]["wint3r"]["subtitle1"])
                with gr.Accordion(label=lang["3sters_tab"]["wint3r"]["option"]["title"], open=False) as wint3r_option:
                    exe_mode_wint3r = gr.Radio(choices=["local", "slurm"], value="local", label= lang["3sters_tab"]["wint3r"]["option"]["exe_mode"])
                recon_wint3r_btn = gr.Button(value=lang["3sters_tab"]["wint3r"]["recon_btn"])
                outdir_recon_wint3r = gr.Textbox(label=lang["3sters_tab"]["wint3r"]["outdir_recon"])
                runtime_recon_wint3r = gr.Textbox(label=lang["3sters_tab"]["wint3r"]["runtime_recon"])
                result_recon_wint3r = gr.Textbox(label=lang["3sters_tab"]["wint3r"]["result_recon"])
                log_recon_wint3r = gr.Textbox(label=lang["3sters_tab"]["wint3r"]["log_recon"])
                outmodel_wint3r = gr.Model3D(label=lang["3sters_tab"]["wint3r"]["outmodel"])
        
        # vggtTab
        with gr.Tab(label=lang["vggt_tab"]["title"]) as vggt_tab:
            # VGGT
            with gr.Tab(label=lang["vggt_tab"]["vggt"]["title"]) as vggt_tab:
                vggt_sub1 = gr.Markdown(lang["vggt_tab"]["vggt"]["subtitle1"])
                with gr.Accordion(label=lang["vggt_tab"]["vggt"]["option"]["title"], open=False) as vggt_option:
                    exe_mode_vggt = gr.Radio(choices=["local", "slurm"], value="local", label= lang["vggt_tab"]["vggt"]["option"]["exe_mode"])
                    mode_vggt = gr.Radio(choices=["crop","pad"], value="crop", label=lang["vggt_tab"]["vggt"]["option"]["mode_vggt"])
                recon_vggt_btn = gr.Button(value=lang["vggt_tab"]["vggt"]["recon_btn"])
                outdir_recon_vggt = gr.Textbox(label=lang["vggt_tab"]["vggt"]["outdir_recon"])
                runtime_recon_vggt = gr.Textbox(label=lang["vggt_tab"]["vggt"]["runtime_recon"])
                result_recon_vggt = gr.Textbox(label=lang["vggt_tab"]["vggt"]["result_recon"])
                log_recon_vggt = gr.Textbox(label=lang["vggt_tab"]["vggt"]["log_recon"])
                outmodel_vggt = gr.Model3D(label=lang["vggt_tab"]["vggt"]["outmodel"])
            
            # VGGSfM
            with gr.Tab(label=lang["vggt_tab"]["vggsfm"]["title"]) as vggsfm_tab:
                vggsfm_sub1 = gr.Markdown(lang["vggt_tab"]["vggsfm"]["subtitle1"])
                with gr.Accordion(label=lang["vggt_tab"]["vggsfm"]["option"]["title"], open=False) as vggsfm_option:
                    exe_mode_vggsfm = gr.Radio(choices=["local", "slurm"], value="local", label= lang["vggt_tab"]["vggsfm"]["option"]["exe_mode"])
                recon_vggsfm_btn = gr.Button(value=lang["vggt_tab"]["vggsfm"]["recon_btn"])
                outdir_recon_vggsfm = gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["outdir_recon"])
                runtime_recon_vggsfm = gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["runtime_recon"])
                result_recon_vggsfm = gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["result_recon"])
                log_recon_vggsfm = gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["log_recon"])
                with gr.Column(visible=False) as export_vggsfm_col:
                    vggsfm_sub2 = gr.Markdown(lang["vggt_tab"]["vggsfm"]["subtitle2"])
                    export_vggsfm_btn = gr.Button(value=lang["vggt_tab"]["vggsfm"]["export_btn"])
                    outdir_export_vggsfm = gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["outdir_export"])
                    runtime_export_vggsfm = gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["runtime_export"])
                    result_export_vggsfm = gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["result_export"])
                    log_export_vggsfm = gr.Textbox(label=lang["vggt_tab"]["vggsfm"]["log_export"])
                    outmodel_vggsfm = gr.Model3D(label=lang["vggt_tab"]["vggsfm"]["outmodel"])

            # VGGT-SLAM
            with gr.Tab(label=lang["vggt_tab"]["vggt-slam"]["title"]) as vggtslam_tab:
                vggtslam_sub1 = gr.Markdown(lang["vggt_tab"]["vggt-slam"]["subtitle1"])
                with gr.Accordion(label=lang["vggt_tab"]["vggt-slam"]["option"]["title"], open=False) as vggtslam_option:
                    exe_mode_vggtslam = gr.Radio(choices=["local", "slurm"], value="local", label= lang["vggt_tab"]["vggt-slam"]["option"]["exe_mode"])
                recon_vggtslam_btn = gr.Button(value=lang["vggt_tab"]["vggt-slam"]["recon_btn"])
                vggtslam_viewer = gr.Markdown(lang["vggt_tab"]["vggt-slam"]["viewer"])
                outdir_recon_vggtslam = gr.Textbox(label=lang["vggt_tab"]["vggt-slam"]["outdir_recon"])
                runtime_recon_vggtslam = gr.Textbox(label=lang["vggt_tab"]["vggt-slam"]["runtime_recon"])
                result_recon_vggtslam = gr.Textbox(label=lang["vggt_tab"]["vggt-slam"]["result_recon"])
                log_recon_vggtslam = gr.Textbox(label=lang["vggt_tab"]["vggt-slam"]["log_recon"])
                outmodel_vggtslam = gr.Model3D(label=lang["vggt_tab"]["vggt-slam"]["outmodel"])

            # StreamVGGT
            with gr.Tab(label=lang["vggt_tab"]["streamvggt"]["title"]) as stmvggt_tab:
                stmvggt_sub1 = gr.Markdown(lang["vggt_tab"]["streamvggt"]["subtitle1"])
                with gr.Accordion(label=lang["vggt_tab"]["streamvggt"]["option"]["title"], open=False) as stmvggt_option:
                    exe_mode_stmvggt = gr.Radio(choices=["local", "slurm"], value="local", label= lang["vggt_tab"]["streamvggt"]["option"]["exe_mode"])
                recon_stmvggt_btn = gr.Button(value=lang["vggt_tab"]["streamvggt"]["recon_btn"])
                outdir_recon_stmvggt = gr.Textbox(label=lang["vggt_tab"]["streamvggt"]["outdir_recon"])
                runtime_recon_stmvggt = gr.Textbox(label=lang["vggt_tab"]["streamvggt"]["runtime_recon"])
                result_recon_stmvggt = gr.Textbox(label=lang["vggt_tab"]["streamvggt"]["result_recon"])
                log_recon_stmvggt = gr.Textbox(label=lang["vggt_tab"]["streamvggt"]["log_recon"])
                outmodel_stmvggt = gr.Model3D(label=lang["vggt_tab"]["streamvggt"]["outmodel"])

            # FastVGGT
            with gr.Tab(label=lang["vggt_tab"]["fastvggt"]["title"]) as fastvggt_tab:
                fastvggt_sub1 = gr.Markdown(lang["vggt_tab"]["fastvggt"]["subtitle1"])
                with gr.Accordion(label=lang["vggt_tab"]["fastvggt"]["option"]["title"], open=False) as fastvggt_option:
                    exe_mode_fastvggt = gr.Radio(choices=["local", "slurm"], value="local", label= lang["vggt_tab"]["fastvggt"]["option"]["exe_mode"])
                recon_fastvggt_btn = gr.Button(value=lang["vggt_tab"]["fastvggt"]["recon_btn"])
                outdir_recon_fastvggt = gr.Textbox(label=lang["vggt_tab"]["fastvggt"]["outdir_recon"])
                runtime_recon_fastvggt = gr.Textbox(label=lang["vggt_tab"]["fastvggt"]["runtime_recon"])
                result_recon_fastvggt = gr.Textbox(label=lang["vggt_tab"]["fastvggt"]["result_recon"])
                log_recon_fastvggt = gr.Textbox(label=lang["vggt_tab"]["fastvggt"]["log_recon"])
                outmodel_fastvggt = gr.Model3D(label=lang["vggt_tab"]["fastvggt"]["outmodel"])

            # Pi3
            with gr.Tab(label=lang["vggt_tab"]["pi3"]["title"]) as pi3_tab:
                pi3_sub1 = gr.Markdown(lang["vggt_tab"]["pi3"]["subtitle1"])
                with gr.Accordion(label=lang["vggt_tab"]["pi3"]["option"]["title"], open=False) as pi3_option:
                    exe_mode_pi3 = gr.Radio(choices=["local", "slurm"], value="local", label= lang["vggt_tab"]["pi3"]["option"]["exe_mode"])
                recon_pi3_btn = gr.Button(value=lang["vggt_tab"]["pi3"]["recon_btn"])
                outdir_recon_pi3 = gr.Textbox(label=lang["vggt_tab"]["pi3"]["outdir_recon"])
                runtime_recon_pi3 = gr.Textbox(label=lang["vggt_tab"]["pi3"]["runtime_recon"])
                result_recon_pi3 = gr.Textbox(label=lang["vggt_tab"]["pi3"]["result_recon"])
                log_recon_pi3 = gr.Textbox(label=lang["vggt_tab"]["pi3"]["log_recon"])
                outmodel_pi3 = gr.Model3D(label=lang["vggt_tab"]["pi3"]["outmodel"])

        # mdsTab
        with gr.Tab(label=lang["mds_tab"]["title"]) as mds_tab:

            # moge2
            with gr.Tab(label=lang["mds_tab"]["moge2"]["title"]) as moge2_tab:
                moge2_sub1 = gr.Markdown(lang["mds_tab"]["moge2"]["subtitle1"])
                moge2_info1 = gr.Markdown(lang["mds_tab"]["moge2"]["info1"])
                img_moge2 = gr.Image(type="filepath", label=lang["mds_tab"]["moge2"]["image"])
                # 推論UI
                with gr.Column(visible=False) as infer_moge2_col:
                    moge2_sub2 = gr.Markdown(lang["mds_tab"]["moge2"]["subtitle2"])
                    with gr.Accordion(label=lang["mds_tab"]["moge2"]["option"]["title"], open=False) as moge2_option:
                        exe_mode_moge2 = gr.Radio(choices=["local", "slurm"], value="local", label= lang["mds_tab"]["moge2"]["option"]["exe_mode"])
                        img_type_moge2 = gr.Radio(choices=[lang["mds_tab"]["moge2"]["option"]["radio_standard"], lang["mds_tab"]["moge2"]["option"]["radio_panorama"]], value=lang["mds_tab"]["moge2"]["option"]["radio_default"])
                    recon_moge2_btn = gr.Button(value=lang["mds_tab"]["moge2"]["recon_btn"])
                    outdir_recon_moge2 = gr.Textbox(label=lang["mds_tab"]["moge2"]["outdir_recon"])
                    runtime_recon_moge2 = gr.Textbox(label=lang["mds_tab"]["moge2"]["runtime_recon"])
                    result_recon_moge2 = gr.Textbox(label=lang["mds_tab"]["moge2"]["result_recon"])
                    log_recon_moge2 = gr.Textbox(label=lang["mds_tab"]["moge2"]["log_recon"])
                    outmodel_moge2 = gr.Model3D(label=lang["mds_tab"]["moge2"]["outmodel"])

            # UniK3D
            with gr.Tab(label=lang["mds_tab"]["unik3d"]["title"]) as unik3d_tab:
                unik3d_sub1 = gr.Markdown(lang["mds_tab"]["unik3d"]["subtitle1"])
                unik3d_info1 = gr.Markdown(lang["mds_tab"]["unik3d"]["info1"])
                img_unik3d = gr.Image(type="filepath", label=lang["mds_tab"]["unik3d"]["image"])
                # 推論UI
                with gr.Column(visible=False) as infer_unik3d_col:
                    unik3d_sub2 = gr.Markdown(lang["mds_tab"]["unik3d"]["subtitle2"])
                    with gr.Accordion(label=lang["mds_tab"]["unik3d"]["option"]["title"], open=False) as unik3d_option:
                        exe_mode_unik3d = gr.Radio(choices=["local", "slurm"], value="local", label= lang["mds_tab"]["unik3d"]["option"]["exe_mode"])
                    recon_unik3d_btn = gr.Button(value=lang["mds_tab"]["unik3d"]["recon_btn"])
                    outdir_recon_unik3d = gr.Textbox(label=lang["mds_tab"]["unik3d"]["outdir_recon"])
                    runtime_recon_unik3d = gr.Textbox(label=lang["mds_tab"]["unik3d"]["runtime_recon"])
                    result_recon_unik3d = gr.Textbox(label=lang["mds_tab"]["unik3d"]["result_recon"])
                    log_recon_unik3d = gr.Textbox(label=lang["mds_tab"]["unik3d"]["log_recon"])
                    outmodel_unik3d = gr.Model3D(label=lang["mds_tab"]["unik3d"]["outmodel"])
            # DA2
            with gr.Tab(label=lang["mds_tab"]["da2"]["title"]) as da2_tab:
                da2_sub1 = gr.Markdown(lang["mds_tab"]["da2"]["subtitle1"])
                da2_input_radio = gr.Radio(choices=[lang["mds_tab"]["da2"]["radio_image"], lang["mds_tab"]["da2"]["radio_video"]], 
                                           label=lang["mds_tab"]["da2"]["input_radio"])
                with gr.Column(visible=False) as da2_image_col:
                    da2_iamge_sub2 = gr.Markdown(lang["mds_tab"]["da2"]["image_section"]["subtitle2"])
                    da2_iamge_info1 = gr.Markdown(lang["mds_tab"]["da2"]["image_section"]["info1"])
                    image_da2 = gr.Image(type="filepath", label=lang["mds_tab"]["da2"]["image_section"]["image"])
                    da2_image_sub3 = gr.Markdown(lang["mds_tab"]["da2"]["subtitle3"])
                    with gr.Accordion(label=lang["mds_tab"]["da2"]["option"]["title"], open=False) as da2_image_option:
                        exe_mode_image_da2 = gr.Radio(choices=["local", "slurm"], value="local", label=lang["mds_tab"]["da2"]["option"]["exe_mode"])
                        exe_model_image_da2 = gr.Radio(choices=["vits", "vitb", "vitl", "vitg"], value="vitl", label=lang["mds_tab"]["da2"]["option"]["exe_model"])
                    run_image_da2_btn =  gr.Button(value=lang["mds_tab"]["da2"]["run_btn"])
                    outdir_image_da2 = gr.Textbox(label=lang["mds_tab"]["da2"]["outdir_recon"])
                    runtime_image_da2 = gr.Textbox(label=lang["mds_tab"]["da2"]["runtime_recon"])
                    result_image_da2 = gr.Textbox(label=lang["mds_tab"]["da2"]["result_recon"])
                    log_image_da2 = gr.Textbox(label=lang["mds_tab"]["da2"]["log_recon"])
                    outimage_da2 = gr.State()
                    gallery_da2 = gr.Gallery(label=lang["mds_tab"]["da2"]["outimage"], columns=1, height="auto")
                with gr.Column(visible=False) as da2_video_col:
                    da2_video_sub2 = gr.Markdown(lang["mds_tab"]["da2"]["video_section"]["subtitle2"])
                    da2_video_info1 = gr.Markdown(lang["mds_tab"]["da2"]["video_section"]["info1"])
                    video_da2 = gr.Video(label=lang["mds_tab"]["da2"]["video_section"]["video"], height="auto")
                    da2_video_sub3 = gr.Markdown(lang["mds_tab"]["da2"]["subtitle3"])
                    with gr.Accordion(label=lang["mds_tab"]["da2"]["option"]["title"], open=False) as da2_video_option:
                        exe_mode_video_da2 = gr.Radio(choices=["local", "slurm"], value="local", label=lang["mds_tab"]["da2"]["option"]["exe_mode"])
                        exe_model_video_da2 = gr.Radio(choices=["vits", "vitb", "vitl", "vitg"], value="vitl", label=lang["mds_tab"]["da2"]["option"]["exe_model"])
                    run_video_da2_btn =  gr.Button(value=lang["mds_tab"]["da2"]["run_btn"])
                    outdir_video_da2 = gr.Textbox(label=lang["mds_tab"]["da2"]["outdir_recon"])
                    runtime_video_da2 = gr.Textbox(label=lang["mds_tab"]["da2"]["runtime_recon"])
                    result_video_da2 = gr.Textbox(label=lang["mds_tab"]["da2"]["result_recon"])
                    log_video_da2 = gr.Textbox(label=lang["mds_tab"]["da2"]["log_recon"])
                    outvideo_da2 = gr.Video(label=lang["mds_tab"]["da2"]["outvideo"], height="auto")

            # DA3
            with gr.Tab(label=lang["mds_tab"]["da3"]["title"]) as da3_tab:
                da3_sub1 = gr.Markdown(lang["mds_tab"]["da3"]["subtitle1"])
                with gr.Accordion(label=lang["mds_tab"]["da3"]["option"]["title"], open=False) as da3_option:
                    exe_mode_da3 = gr.Radio(choices=["local", "slurm"], value="local", label=lang["mds_tab"]["da3"]["option"]["exe_mode"])
                recon_da3_btn =  gr.Button(value=lang["mds_tab"]["da3"]["run_btn"])
                outdir_da3 = gr.Textbox(label=lang["mds_tab"]["da3"]["outdir_recon"])
                runtime_da3 = gr.Textbox(label=lang["mds_tab"]["da3"]["runtime_recon"])
                result_da3 = gr.Textbox(label=lang["mds_tab"]["da3"]["result_recon"])
                log_da3 = gr.Textbox(label=lang["mds_tab"]["da3"]["log_recon"])
                outmodel_da3 = gr.Model3D(label=lang["mds_tab"]["da3"]["outmodel"])
                outimages_da3 = gr.State()
                gallery_da3 = gr.Gallery(label=lang["mds_tab"]["da3"]["outimage"], columns=1, height="auto")
                outvideo_da3 = gr.Video(label=lang["mds_tab"]["da3"]["outvideo"], height="auto")
                outGSvideo_da3 = gr.Video(label=lang["mds_tab"]["da3"]["outGSvideo"], height="auto")

        # 評価指標Tab
        with gr.Tab(label=lang["metrics_tab"]["title"]) as metrics_tab:
            method_metrics = gr.DataFrame(label=lang["metrics_tab"]["table"], headers=lang["metrics_tab"]["headers"])
            download_csv = gr.DownloadButton(label=lang["metrics_tab"]["download_btn"])

        """
        イベントリスナ
        """
        language_radio.change(
            fn = update_ui,
            inputs=language_radio,
            outputs=[lang_state, 
                     current_dataset_images,
                     current_dataset_colmap,
                     # --- DatasetTab ---
                     dataset_tab, dataset_sub1, dataset_info1, dataset_radio, dataset_new_sub2, media_radio, dataset_image_sub3, images, dataset_name, run_copy_btn,
                     dataset_image_sub4, output_image, gallery_image, dataset_video_sub3, video, fps, dataset_video_option, dataset_video_option_subtitle,
                     dataset_video_option_info1, rsi, dataset_video_option_info2, ssim, run_ffmpeg_btn, output_video, comp_rate, sel_images_num, rej_images_num,
                     gallery_video, dataset_video_sub4, dataset_video_info1, zipfile_images, dataset_load_sub2, load_dataset_info1, load_dataset, log_unzip,
                     # --- COLMAPTab ---
                     colmap_tab, colmap_sub1, colmap_info1, colmap_option, colmap_option_info1, rebuild, run_colmap_btn, result_colmap, colmap_sub2, colmap_info2, zipfile_colmap,
                     # --- NeRFTab ---
                     nerf_tab,
                     # Vanilla NeRF
                     vnerf_tab, vnerf_sub1, vnerf_option, exe_mode_vnerf, iter_vnerf, recon_vnerf_btn, vnerf_viewer, outdir_recon_vnerf, runtime_recon_vnerf,
                     result_recon_vnerf, log_recon_vnerf, vnerf_sub2, export_vnerf_btn, outdir_export_vnerf, runtime_export_vnerf, result_export_vnerf, log_export_vnerf,
                     vnerf_sub3, eval_vnerf_btn, outdir_eval_vnerf, runtime_eval_vnerf, result_eval_vnerf, log_eval_vnerf, metrics_vnerf, gallery_vnerf,
                     # Nerfacto
                     nerfacto_tab, nerfacto_sub1, nerfacto_option, exe_mode_nerfacto, iter_nerfacto, recon_nerfacto_btn, nerfacto_viewer, outdir_recon_nerfacto,
                     runtime_recon_nerfacto, result_recon_nerfacto, log_recon_nerfacto, nerfacto_sub2, export_nerfacto_btn, outdir_export_nerfacto, runtime_export_nerfacto, 
                     result_export_nerfacto, log_export_nerfacto, nerfacto_sub3, eval_nerfacto_btn, outdir_eval_nerfacto, runtime_eval_nerfacto, result_eval_nerfacto,
                     log_eval_nerfacto, metrics_nerfacto, gallery_nerfacto,
                     # mip-NeRF
                     mipnerf_tab, mipnerf_sub1, mipnerf_option, exe_mode_mipnerf, iter_mipnerf, recon_mipnerf_btn, mipnerf_viewer, outdir_recon_mipnerf, runtime_recon_mipnerf,
                     result_recon_mipnerf, log_recon_mipnerf, mipnerf_sub2, export_mipnerf_btn, outdir_export_mipnerf, runtime_export_mipnerf, result_export_mipnerf,
                     log_export_mipnerf, mipnerf_sub3, eval_mipnerf_btn, outdir_eval_mipnerf, runtime_eval_mipnerf, result_eval_mipnerf, log_eval_mipnerf, metrics_mipnerf, gallery_mipnerf,
                     # SeaThru-NeRF
                     stnerf_tab, stnerf_sub1, stnerf_option, exe_mode_stnerf, iter_stnerf, recon_stnerf_btn, stnerf_viewer, outdir_recon_stnerf, runtime_recon_stnerf,
                     result_recon_stnerf, log_recon_stnerf, stnerf_sub2, export_stnerf_btn, outdir_export_stnerf, runtime_export_stnerf, result_export_stnerf,log_export_stnerf, stnerf_sub3,
                     eval_stnerf_btn, outdir_eval_stnerf, runtime_eval_stnerf, result_eval_stnerf, log_eval_stnerf, metrics_stnerf, gallery_stnerf,
                     # --- GSTab ---
                     gs_tab,
                     # Vanilla GS
                     vgs_tab, vgs_sub1, vgs_option, exe_mode_vgs, recon_vgs_btn, outdir_recon_vgs, runtime_recon_vgs, result_recon_vgs, log_recon_vgs, outmodel_vgs, vgs_sub2,
                     skip_train, skip_test, eval_vgs_btn, runtime_eval_vgs, result_eval_vgs, log_eval_vgs, metrics_vgs, gallery_vgs,
                     # Mip-Splatting
                     mips_tab, mips_sub1, mips_option, exe_mode_mips, save_iter_mips, recon_mips_btn, outdir_recon_mips, runtime_recon_mips, result_recon_mips, log_recon_mips,
                     outmodel_mips, mips_sub2, eval_mips_btn, outdir_eval_mips, runtime_eval_mips, result_eval_mips, log_eval_mips, metrics_mips, gallery_mips,
                     # Splatfacto
                     sfacto_tab, sfacto_sub1, sfacto_option, exe_mode_sfacto, iter_sfacto, recon_sfacto_btn, sfacto_viewer, outdir_recon_sfacto, runtime_recon_sfacto,
                     result_recon_sfacto, log_recon_sfacto, sfacto_sub2, export_sfacto_btn, outdir_export_sfacto, runtime_export_sfacto, result_export_sfacto,
                     log_export_sfacto, sfacto_sub3, eval_sfacto_btn, outdir_eval_sfacto, runtime_eval_sfacto, result_eval_sfacto, log_eval_sfacto, metrics_sfacto, gallery_sfacto,
                     # 4D-Gaussians
                     gs4d_tab, gs4d_sub1, gs4d_option, exe_mode_4dgs, save_iter_4dgs, recon_4dgs_btn, outdir_recon_4dgs, runtime_recon_4dgs, result_recon_4dgs,
                     log_recon_4dgs, outmodel_4dgs, gs4d_sub2, eval_4dgs_btn, outdir_eval_4dgs, runtime_eval_4dgs, result_eval_4dgs, log_eval_4dgs, metrics_4dgs, gallery_4dgs,
                     # --- 3stersTab ---
                     esters_tab,
                     # DUSt3R
                     dust3r_tab, dust3r_sub1, dust3r_option, exe_mode_dust3r, recon_dust3r_btn, outdir_recon_dust3r, runtime_recon_dust3r, result_recon_dust3r,
                     log_recon_dust3r, outmodel_dust3r, gallery_dust3r,
                     # MASt3R
                     mast3r_tab, mast3r_sub1, mast3r_option, exe_mode_mast3r, recon_mast3r_btn, outdir_recon_mast3r, runtime_recon_mast3r, result_recon_mast3r,
                     log_recon_mast3r, outmodel_mast3r,
                     # MonST3R
                     monst3r_tab, monst3r_sub1, monst3r_option, exe_mode_monst3r, recon_monst3r_btn, outdir_recon_monst3r, runtime_recon_monst3r, result_recon_monst3r,
                     log_recon_monst3r, outmodel_monst3r,
                     # Easi3R
                     easi3r_tab, easi3r_sub1, easi3r_info1, easi3r_option, exe_mode_easi3r, recon_easi3r_btn, outdir_recon_easi3r,runtime_recon_easi3r,
                     result_recon_easi3r, log_recon_easi3r, outmodel_easi3r,
                     # MUSt3R
                     must3r_tab, must3r_sub1, must3r_option, exe_mode_must3r, recon_must3r_btn, outdir_recon_must3r, runtime_recon_must3r, result_recon_must3r,
                     log_recon_must3r, outmodel_must3r,
                     # Fast3R
                     fast3r_tab, fast3r_sub1, fast3r_option, exe_mode_fast3r, recon_fast3r_btn, outdir_recon_fast3r, runtime_recon_fast3r, result_recon_fast3r,
                     log_recon_fast3r, outmodel_fast3r,
                     # Splatt3R
                     splatt3r_tab, splatt3r_sub1, splatt3r_info1, img_splatt3r, splatt3r_sub2, splatt3r_option, exe_mode_splatt3r, recon_splatt3r_btn, outdir_recon_splatt3r,
                     runtime_recon_splatt3r, result_recon_splatt3r, log_recon_splatt3r, outmodel_splatt3r,
                     # CUT3R
                     cut3r_tab, cut3r_sub1, cut3r_option, exe_mode_cut3r, recon_cut3r_btn, outdir_recon_cut3r, runtime_recon_cut3r, result_recon_cut3r,
                     log_recon_cut3r, outmodel_cut3r,
                     # WinT3R
                     wint3r_tab, wint3r_sub1, wint3r_option, exe_mode_wint3r, recon_wint3r_btn, outdir_recon_wint3r, runtime_recon_wint3r, result_recon_wint3r,
                     log_recon_wint3r, outmodel_wint3r,
                     # ---vggTab ---
                     vggt_tab,
                     # VGGT
                     vggt_tab, vggt_sub1, vggt_option, exe_mode_vggt, mode_vggt, recon_vggt_btn, outdir_recon_vggt, runtime_recon_vggt, result_recon_vggt,
                     log_recon_vggt, outmodel_vggt,
                     # VGGSfM
                     vggsfm_tab, vggsfm_sub1, vggsfm_option, exe_mode_vggsfm, recon_vggsfm_btn, outdir_recon_vggsfm, runtime_recon_vggsfm, result_recon_vggsfm,
                     log_recon_vggsfm, vggsfm_sub2, export_vggsfm_btn, outdir_export_vggsfm, runtime_export_vggsfm, result_export_vggsfm, log_export_vggsfm, outmodel_vggsfm,
                     # VGGT-SLAM
                     vggtslam_tab, vggtslam_sub1, vggtslam_option, exe_mode_vggtslam, recon_vggtslam_btn, vggtslam_viewer, outdir_recon_vggtslam, runtime_recon_vggtslam,
                     result_recon_vggtslam, log_recon_vggtslam, outmodel_vggtslam,
                     # StreamVGGT
                     stmvggt_tab, stmvggt_sub1, stmvggt_option, exe_mode_stmvggt, recon_stmvggt_btn, outdir_recon_stmvggt, runtime_recon_stmvggt, result_recon_stmvggt,
                     log_recon_stmvggt, outmodel_stmvggt,
                     # FastVGGT
                     fastvggt_tab, fastvggt_sub1, fastvggt_option, exe_mode_fastvggt, recon_fastvggt_btn, outdir_recon_fastvggt, runtime_recon_fastvggt, result_recon_fastvggt,
                     log_recon_fastvggt, outmodel_fastvggt,
                     # Pi3
                     pi3_tab, pi3_sub1, pi3_option, exe_mode_pi3, recon_pi3_btn, outdir_recon_pi3, runtime_recon_pi3, result_recon_pi3, log_recon_pi3, outmodel_pi3,
                     # --- mdsTab ---
                     mds_tab,
                     # moge2
                     moge2_tab, moge2_sub1, moge2_info1, img_moge2, moge2_sub2, moge2_option, exe_mode_moge2, img_type_moge2, recon_moge2_btn, outdir_recon_moge2,
                     runtime_recon_moge2, result_recon_moge2, log_recon_moge2, outmodel_moge2,
                     # UniK3D
                     unik3d_tab, unik3d_sub1, unik3d_info1, img_unik3d, unik3d_sub2, unik3d_option, exe_mode_unik3d, recon_unik3d_btn, outdir_recon_unik3d,
                     runtime_recon_unik3d, result_recon_unik3d, log_recon_unik3d, outmodel_unik3d,
                     # Depth Anything 2
                     da2_tab, da2_sub1, da2_input_radio, da2_iamge_sub2, da2_iamge_info1, image_da2, da2_image_sub3, da2_image_option, exe_mode_image_da2,
                     exe_model_image_da2, run_image_da2_btn, outdir_image_da2, runtime_image_da2, result_image_da2, log_image_da2, gallery_da2, da2_video_sub2,
                     da2_video_info1, video_da2,da2_video_sub3, da2_video_option, exe_mode_video_da2, exe_model_video_da2, run_video_da2_btn, outdir_video_da2,
                     runtime_video_da2, result_video_da2, log_video_da2, outvideo_da2,
                     # Depth Anything 3
                     da3_sub1, da3_option, exe_mode_da3, recon_da3_btn, outdir_da3, runtime_da3, result_da3, log_da3, outmodel_da3, gallery_da3, outvideo_da3, outGSvideo_da3,
                     # --- Metrics Tab ---
                     metrics_tab, method_metrics, download_csv
                     ]
        )

        # --- UI切り替え ---
        dataset_radio.change(fn=switch_dataset_ui,
                             inputs=[dataset_radio, lang_state],
                             outputs=[new_dataset_col, load_dataset_col])
        media_radio.change(fn=switch_media_ui, 
                           inputs=[media_radio, lang_state],
                           outputs=[image_col, video_col])
        img_splatt3r.change(fn=col_change, outputs=infer_splatt3r_col)
        img_moge2.change(fn=col_change, outputs=infer_moge2_col)
        img_unik3d.change(fn=col_change, outputs=infer_unik3d_col)
        da2_input_radio.change(fn=switch_da2_ui,
                               inputs=[da2_input_radio, lang_state],
                               outputs=[da2_image_col,  da2_video_col])

        # --- データセット ---
        # 画像データセット作成
        run_copy_btn.click(fn=local_backend.copy_images,
                       inputs=[images, datasetsdir_state, dataset_name],
                       outputs=[image_dataset_state, output_image, gallery_image, iresult_col]).success(
                             fn=local_backend.zip_dataset,
                             inputs=image_dataset_state,
                             outputs=zipfile_images).success(
                                 fn=get_state_value, 
                                 inputs=image_dataset_state, 
                                 outputs=current_dataset_images)
        run_ffmpeg_btn.click(fn=local_backend.extract_frames_with_filter, 
                         inputs=[video, datasetsdir_state, fps, rsi, ssim], 
                         outputs=[image_dataset_state, output_video, comp_rate, sel_images_num, rej_images_num, gallery_video, dl_images_col]).success(
                             fn=local_backend.zip_dataset,
                             inputs=image_dataset_state,
                             outputs=zipfile_images).success(
                                 fn=get_state_value, 
                                 inputs=image_dataset_state, 
                                 outputs=current_dataset_images)
        # colmapデータセット作成
        run_colmap_btn.click(fn=local_backend.run_colmap,
                        inputs=[exe_mode_colmap, image_dataset_state, rebuild],
                        outputs=[colmap_dataset_state, result_colmap, dl_colmap_col]).success(
                            fn=local_backend.zip_dataset,
                            inputs=colmap_dataset_state,
                            outputs=zipfile_colmap).success(
                                 fn=get_state_value, 
                                 inputs=colmap_dataset_state, 
                                 outputs=current_dataset_colmap)
        # 既存データセット展開
        load_dataset.upload(fn=local_backend.unzip_dataset,
                               inputs=[load_dataset, datasetsdir_state],
                               outputs=[image_dataset_state, colmap_dataset_state, log_unzip]).success(
                                   fn=get_state_value, 
                                   inputs=image_dataset_state, 
                                   outputs=current_dataset_images).success(
                                       fn=get_state_value, 
                                       inputs=colmap_dataset_state, 
                                       outputs=current_dataset_colmap)
        
        # --- 三次元再構築 ---
        recon_vnerf_btn.click(fn=methods.recon_vnerf,
                             inputs=[exe_mode_vnerf, colmap_dataset_state, outputsdir_state, iter_vnerf],
                             outputs=[outdir_recon_vnerf, runtime_recon_vnerf, result_recon_vnerf, log_recon_vnerf, export_vnerf_col, eval_vnerf_col])
        recon_nerfacto_btn.click(fn=methods.recon_nerfacto,
                                 inputs=[exe_mode_nerfacto, colmap_dataset_state, outputsdir_state, iter_nerfacto],
                                 outputs=[outdir_recon_nerfacto, runtime_recon_nerfacto, result_recon_nerfacto, log_recon_nerfacto, export_nerfacto_col, eval_nerfacto_col])
        recon_mipnerf_btn.click(fn=methods.recon_mipnerf,
                             inputs=[exe_mode_mipnerf, colmap_dataset_state, outputsdir_state, iter_mipnerf],
                             outputs=[outdir_recon_mipnerf, runtime_recon_mipnerf, result_recon_mipnerf, log_recon_mipnerf, export_mipnerf_col, eval_mipnerf_col])
        recon_stnerf_btn.click(fn=methods.recon_stnerf,
                             inputs=[exe_mode_stnerf, colmap_dataset_state, outputsdir_state, iter_stnerf],
                             outputs=[outdir_recon_stnerf, runtime_recon_stnerf, result_recon_stnerf, log_recon_stnerf, export_stnerf_col, eval_stnerf_col])
        recon_vgs_btn.click(fn=methods.recon_vgs, 
                             inputs=[exe_mode_vgs, colmap_dataset_state, outputsdir_state, sh_degree, data_device, lambda_dssim, iter_3dgs,
                                     test_iter_3dgs, save_iter_3dgs, feature_lr,
                                     opacity_lr, scaling_lr, rotation_lr, position_lr_init, position_lr_final,
                                     position_lr_delay_mult, densify_from_iter, densify_until_iter, densify_grad_threshold,
                                     densification_interval, opacity_rest_interval, percent_dense], 
                                     outputs=[outdir_recon_vgs, runtime_recon_vgs, result_recon_vgs, log_recon_vgs, outmodel_vgs, eval_vgs_col ])
        recon_mips_btn.click(fn=methods.recon_mipSplatting, 
                             inputs=[exe_mode_mips, colmap_dataset_state, outputsdir_state, save_iter_mips], 
                             outputs=[outdir_recon_mips, runtime_recon_mips, result_recon_mips, log_recon_mips, outmodel_mips, eval_mips_col])
        recon_sfacto_btn.click(fn=methods.recon_sfacto,
                             inputs=[exe_mode_sfacto, colmap_dataset_state, outputsdir_state, iter_sfacto],
                             outputs=[outdir_recon_sfacto, runtime_recon_sfacto, result_recon_sfacto, log_recon_sfacto, export_sfacto_col, eval_sfacto_col])
        recon_4dgs_btn.click(fn=methods.recon_4dGaussians, 
                             inputs=[exe_mode_4dgs, colmap_dataset_state, outputsdir_state, save_iter_4dgs], 
                             outputs=[outdir_recon_4dgs, runtime_recon_4dgs, result_recon_4dgs, log_recon_4dgs, outmodel_4dgs, eval_4dgs_col])
        recon_dust3r_btn.click(fn=methods.recon_dust3r,
                               inputs=[exe_mode_dust3r, image_dataset_state, outputsdir_state, schedule, niter, min_conf_thr, as_pointcloud,mask_sky, clean_depth, transparent_cams, cam_size,scenegraph_type, winsize, refid], 
                               outputs=[outdir_recon_dust3r, runtime_recon_dust3r, result_recon_dust3r, log_recon_dust3r, outmodel_dust3r, outimages_dust3r]).success(
                                           fn=local_backend.get_imagelist,
                                           inputs=outimages_dust3r,
                                           outputs=gallery_dust3r)
        recon_mast3r_btn.click(fn=methods.recon_mast3r,
                        inputs=[exe_mode_mast3r, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_mast3r, runtime_recon_mast3r, result_recon_mast3r, log_recon_mast3r, outmodel_mast3r])
        recon_monst3r_btn.click(fn=methods.recon_monst3r,
                        inputs=[exe_mode_monst3r, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_monst3r, runtime_recon_monst3r, result_recon_monst3r, log_recon_monst3r, outmodel_monst3r])
        recon_easi3r_btn.click(fn=methods.recon_easi3r,
                        inputs=[exe_mode_easi3r, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_easi3r, runtime_recon_easi3r, result_recon_easi3r, log_recon_easi3r, outmodel_easi3r])
        recon_must3r_btn.click(fn=methods.recon_must3r,
                        inputs=[exe_mode_must3r, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_must3r, runtime_recon_must3r, result_recon_must3r, log_recon_must3r, outmodel_must3r])
        recon_fast3r_btn.click(fn=methods.recon_fast3r,
                        inputs=[exe_mode_fast3r, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_fast3r, runtime_recon_fast3r, result_recon_fast3r, log_recon_fast3r, outmodel_fast3r])
        recon_cut3r_btn.click(fn=methods.recon_cut3r,
                        inputs=[exe_mode_cut3r, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_cut3r, runtime_recon_cut3r, result_recon_cut3r, log_recon_cut3r, outmodel_cut3r])
        recon_wint3r_btn.click(fn=methods.recon_wint3r,
                        inputs=[exe_mode_wint3r, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_wint3r, runtime_recon_wint3r, result_recon_wint3r, log_recon_wint3r, outmodel_wint3r])
        recon_splatt3r_btn.click(fn=methods.recon_splatt3r,
                        inputs=[exe_mode_splatt3r, img_splatt3r, outputsdir_state], 
                        outputs=[outdir_recon_splatt3r, runtime_recon_splatt3r, result_recon_splatt3r, log_recon_splatt3r, outmodel_splatt3r])
        recon_vggt_btn.click(fn=methods.recon_vggt,
                        inputs=[exe_mode_vggt, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_vggt, runtime_recon_vggt, result_recon_vggt, log_recon_vggt, outmodel_vggt])
        recon_vggsfm_btn.click(fn=methods.recon_vggsfm,
                                inputs=[exe_mode_vggsfm, image_dataset_state],
                                outputs=[outdir_recon_vggsfm, runtime_recon_vggsfm, result_recon_vggsfm, log_recon_vggsfm, export_vggsfm_col])
        recon_vggtslam_btn.click(fn=methods.recon_vggtslam,
                                 inputs=[exe_mode_vggtslam, image_dataset_state, outputsdir_state],
                                 outputs=[outdir_recon_vggtslam, runtime_recon_vggtslam, result_recon_vggtslam, log_recon_vggtslam, outmodel_vggtslam])
        recon_stmvggt_btn.click(fn=methods.recon_stmvggt,
                        inputs=[exe_mode_stmvggt, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_stmvggt, runtime_recon_stmvggt, result_recon_stmvggt, log_recon_stmvggt, outmodel_stmvggt])
        recon_fastvggt_btn.click(fn=methods.recon_fastvggt,
                        inputs=[exe_mode_fastvggt, image_dataset_state, outputsdir_state], 
                        outputs=[outdir_recon_fastvggt, runtime_recon_fastvggt, result_recon_fastvggt, log_recon_fastvggt, outmodel_fastvggt])
        recon_pi3_btn.click(fn=methods.recon_pi3,
                            inputs=[exe_mode_pi3, image_dataset_state, outputsdir_state],
                            outputs=[outdir_recon_pi3, runtime_recon_pi3, result_recon_pi3, log_recon_pi3, outmodel_pi3])
        recon_moge2_btn.click(fn=methods.recon_moge2,
                        inputs=[exe_mode_moge2, img_moge2, outputsdir_state, img_type_moge2], 
                        outputs=[outdir_recon_moge2, runtime_recon_moge2, result_recon_moge2, log_recon_moge2, outmodel_moge2])
        recon_unik3d_btn.click(fn=methods.recon_unik3d,
                        inputs=[exe_mode_unik3d, img_unik3d, outputsdir_state], 
                        outputs=[outdir_recon_unik3d, runtime_recon_unik3d, result_recon_unik3d, log_recon_unik3d, outmodel_unik3d])
        # DA2
        run_image_da2_btn.click(fn=methods.run_image_da2,
                                inputs=[exe_mode_image_da2, image_da2, outputsdir_state, exe_model_image_da2],
                                outputs=[outdir_image_da2, runtime_image_da2, result_image_da2, log_image_da2, outimage_da2]).success(
                                    fn=local_backend.get_imagelist,
                                    inputs=outimage_da2,
                                    outputs=gallery_da2
                                )
        run_video_da2_btn.click(fn=methods.run_video_da2,
                                inputs=[exe_mode_video_da2, video_da2, outputsdir_state, exe_model_video_da2],
                                outputs=[outdir_video_da2, runtime_video_da2, result_video_da2, log_video_da2, outvideo_da2])
        # DA3
        recon_da3_btn.click(fn=methods.recon_da3,
                            inputs=[exe_mode_da3, image_dataset_state, outputsdir_state],
                            outputs=[outdir_da3, runtime_da3, result_da3, log_da3, outmodel_da3, outimages_da3, outvideo_da3, outGSvideo_da3]).success(
                                fn=local_backend.get_imagelist,
                                inputs=outimages_da3,
                                outputs=gallery_da3)
        
        # --- 点群出力（Nerfstudio）---
        export_vnerf_btn.click(fn=methods.export_vnerf,
                              inputs=[exe_mode_vnerf, colmap_dataset_state, outputsdir_state],
                              outputs=[outdir_export_vnerf, runtime_export_vnerf, result_export_vnerf, log_export_vnerf])
        export_nerfacto_btn.click(fn=methods.export_nerfacto,
                                  inputs=[exe_mode_nerfacto, colmap_dataset_state, outputsdir_state],
                                  outputs=[outdir_export_nerfacto, runtime_export_nerfacto, result_export_nerfacto, log_export_nerfacto])
        export_mipnerf_btn.click(fn=methods.export_mipnerf,
                                 inputs=[exe_mode_mipnerf, colmap_dataset_state, outputsdir_state],
                                 outputs=[outdir_export_mipnerf, runtime_export_mipnerf, result_export_mipnerf, log_export_mipnerf])
        export_stnerf_btn.click(fn=methods.export_stnerf,
                                inputs=[exe_mode_stnerf, colmap_dataset_state, outputsdir_state],
                                outputs=[outdir_export_stnerf, runtime_export_stnerf, result_export_stnerf, log_export_stnerf])
        export_sfacto_btn.click(fn=methods.export_sfacto,
                                inputs=[exe_mode_sfacto, colmap_dataset_state, outputsdir_state],
                                outputs=[outdir_export_sfacto, runtime_export_sfacto, result_export_sfacto, log_export_sfacto])
        export_vggsfm_btn.click(fn=methods.export_vggsfm,
                                inputs=[image_dataset_state, outputsdir_state],
                                outputs=[outdir_export_vggsfm, runtime_export_vggsfm, result_export_vggsfm, log_export_vggsfm, outmodel_vggsfm])

        # --- レンダリング・評価 ---
        # Nerf Tab
        eval_vnerf_btn.click(fn=methods.render_eval_vnerf,
                             inputs=[exe_mode_vnerf, colmap_dataset_state, outputsdir_state],
                             outputs=[outdir_eval_vnerf, runtime_eval_vnerf, result_eval_vnerf, log_eval_vnerf, metrics_vnerf, gallery_vnerf]).success(
                                 fn=update_method_metrics,
                                 inputs=[method_metrics, metrics_vnerf, tmpdir_state],
                                 outputs=[method_metrics, download_csv])
        eval_nerfacto_btn.click(fn=methods.render_eval_nerfacto,
                             inputs=[exe_mode_nerfacto, colmap_dataset_state, outputsdir_state],
                             outputs=[outdir_eval_nerfacto, runtime_eval_nerfacto, result_eval_nerfacto, log_eval_nerfacto, metrics_nerfacto, gallery_nerfacto]).success(
                                 fn=update_method_metrics,
                                 inputs=[method_metrics, metrics_nerfacto, tmpdir_state],
                                 outputs=[method_metrics, download_csv])
        eval_mipnerf_btn.click(fn=methods.render_eval_mipnerf,
                             inputs=[exe_mode_mipnerf, colmap_dataset_state, outputsdir_state],
                             outputs=[outdir_eval_mipnerf, runtime_eval_mipnerf, result_eval_mipnerf, log_eval_mipnerf, metrics_mipnerf, gallery_mipnerf]).success(
                                 fn=update_method_metrics,
                                 inputs=[method_metrics, metrics_mipnerf, tmpdir_state],
                                 outputs=[method_metrics, download_csv])
        eval_stnerf_btn.click(fn=methods.render_eval_stnerf,
                             inputs=[exe_mode_stnerf, colmap_dataset_state, outputsdir_state],
                             outputs=[outdir_eval_stnerf, runtime_eval_stnerf, result_eval_stnerf, log_eval_stnerf, metrics_stnerf, gallery_stnerf]).success(
                                 fn=update_method_metrics,
                                 inputs=[method_metrics, metrics_stnerf, tmpdir_state],
                                 outputs=[method_metrics, download_csv])
        # GS Tab
        eval_vgs_btn.click(fn=methods.render_eval_3dgs,
                           inputs=[outdir_recon_vgs, skip_train, skip_test, save_iter_3dgs],
                           outputs=[outdir_eval_vgs, runtime_eval_vgs, result_eval_vgs, log_eval_vgs, metrics_vgs, gallery_vgs]).success(
                                 fn=update_method_metrics,
                                 inputs=[method_metrics, metrics_vgs, tmpdir_state],
                                 outputs=[method_metrics, download_csv])
        eval_mips_btn.click(fn=methods.render_eval_mips,
                            inputs=[outdir_recon_mips, skip_train, skip_test, save_iter_mips],
                            outputs=[outdir_eval_mips, runtime_eval_mips, result_eval_mips, log_eval_mips, metrics_mips, gallery_mips]).success(
                                 fn=update_method_metrics,
                                 inputs=[method_metrics, metrics_mips, tmpdir_state],
                                 outputs=[method_metrics, download_csv])
        eval_sfacto_btn.click(fn=methods.render_eval_sfacto,
                             inputs=[exe_mode_sfacto, colmap_dataset_state, outputsdir_state],
                             outputs=[outdir_eval_sfacto, runtime_eval_sfacto, result_eval_sfacto, log_eval_sfacto, metrics_sfacto, gallery_sfacto]).success(
                                 fn=update_method_metrics,
                                 inputs=[method_metrics, metrics_sfacto, tmpdir_state],
                                 outputs=[method_metrics, download_csv])
        eval_4dgs_btn.click(fn=methods.render_eval_4dgs,
                            inputs=[outdir_recon_4dgs, skip_train, skip_test, save_iter_4dgs],
                            outputs=[outdir_eval_4dgs, runtime_eval_4dgs, result_eval_4dgs, log_eval_4dgs, metrics_4dgs, gallery_4dgs]).success(
                                 fn=update_method_metrics,
                                 inputs=[method_metrics, metrics_4dgs, tmpdir_state],
                                 outputs=[method_metrics, download_csv])
            
    demo.launch()
