import os
import ast
import torch
import argparse
import sys 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.append(PROJECT_ROOT)      
sys.path.append(os.path.join(PROJECT_ROOT, "models", "mast3r"))

from models.mast3r.mast3r.model import AsymmetricMASt3R
import models.mast3r.mast3r.demo as demo

torch.backends.cuda.matmul.allow_tf32 = True  # for GPU >= Ampere

# パーサー定義
def parser():
    parser = argparse.ArgumentParser(description="Run MASt3R scene reconstruction")

    parser.add_argument("--filelist", type=str, required=True, help="入力画像ファイルのリスト")
    parser.add_argument("--outdir", type=str, help="出力ディレクトリ")
    parser.add_argument("--gradio_delete_cache", type=int, default=None, help="キャッシュ削除間隔")
    parser.add_argument("--model_name", type=str, required=True, help="MASt3Rモデル名（例: MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric）")
    parser.add_argument("--retrieval_model", type=str, default=None, help="Retrievalモデルパス（オプション）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用デバイス")
    parser.add_argument("--silent", action="store_true", help="ログを非表示にする")
    parser.add_argument("--image_size", type=int, default=512, help="入力画像のサイズ")

    parser.add_argument("--optim_level", type=str, choices=["coarse", "refine", "refine+depth"], default="refine+depth",
                        help="最適化レベル")
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--niter1", type=int, default=300)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--niter2", type=int, default=300)

    parser.add_argument("--scenegraph_type", type=str,
                        choices=["complete", "retrieval", "swin", "logwin", "oneref"],
                        default="complete", help="Scene graph 構築法")
    parser.add_argument("--winsize", type=int, default=1)
    parser.add_argument("--win_cyclic", action="store_true")
    parser.add_argument("--refid", type=int, default=0)

    parser.add_argument("--matching_conf_thr", type=float, default=0.0)
    parser.add_argument("--min_conf_thr", type=float, default=1.5)
    parser.add_argument("--shared_intrinsics", action="store_true")

    parser.add_argument("--as_pointcloud", action="store_true")
    parser.add_argument("--mask_sky", action="store_true")
    parser.add_argument("--clean_depth", action="store_false", default=True)
    parser.add_argument("--transparent_cams", action="store_true")
    parser.add_argument("--cam_size", type=float, default=0.2)
    parser.add_argument("--TSDF_thresh", type=float, default=0.0)

    parser.add_argument("--tmp_dir", type=str, default=None, help="一時ディレクトリ")
    return parser

if __name__ == "__main__":
    args = parser().parse_args()

    if isinstance(args.filelist, str):
        args.filelist = ast.literal_eval(args.filelist)

    # モデルのロード
    weights_path = f"naver/{args.model_name}"
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)

    class DummySceneState:
        def __init__(self, outfile_name):
            self.should_delete = False
            self.cache_dir = None
            self.outfile_name = outfile_name

    # 例：出力ファイル名を指定
    custom_outfile = os.path.join(args.outdir, "scene.glb")
    scene_state = DummySceneState(outfile_name=custom_outfile)

    # 再構築実行
    print("Running reconstruction...")
    scene_state, outfile = demo.get_reconstructed_scene(
        outdir=args.outdir,
        gradio_delete_cache=args.gradio_delete_cache,
        model=model,
        retrieval_model=args.retrieval_model,
        device=args.device,
        silent=args.silent,
        image_size=args.image_size,
        current_scene_state=scene_state,
        filelist=args.filelist,
        optim_level=args.optim_level,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        min_conf_thr=args.min_conf_thr,
        matching_conf_thr=args.matching_conf_thr,
        as_pointcloud=args.as_pointcloud,
        mask_sky=args.mask_sky,
        clean_depth=args.clean_depth,
        transparent_cams=args.transparent_cams,
        cam_size=args.cam_size,
        scenegraph_type=args.scenegraph_type,
        winsize=args.winsize,
        win_cyclic=args.win_cyclic,
        refid=args.refid,
        TSDF_thresh=args.TSDF_thresh,
        shared_intrinsics=args.shared_intrinsics
    )
