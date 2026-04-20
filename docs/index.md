# UNe3dMe

## はじめに
UNe3dMe は様々な3次元再構築手法を Web UI で一元的に扱えるようにしたシステムです．  
3次元再構築はもちろん，データセットの作成からビューアによる再構築結果の可視化までも行え，1つの UI 上で一連の作業を，個別のコマンドを都度意識することなく実行できます．  

## 実装手法一覧
### Nerf 系
- [Vanilla NeRF（Nerfstudio）](https://docs.nerf.studio/nerfology/methods/nerf.html)
- [Nerfacto（Nerfstudio）](https://docs.nerf.studio/nerfology/methods/nerfacto.html)
- [mip-NeRF（Nerfstudio）](https://docs.nerf.studio/nerfology/methods/mipnerf.html)
- [SeaThru-NeRF（Nerfstudio）](https://docs.nerf.studio/nerfology/methods/seathru_nerf.html)

### Gaussian Splatting 系
- [Vanilla GS](https://github.com/graphdeco-inria/gaussian-splatting)
- [Mip-Splatting](https://github.com/autonomousvision/mip-splatting)
- [Splatfacto（Nerfstudio）](https://docs.nerf.studio/nerfology/methods/splat.html)
- [4D-Gaussians](https://github.com/hustvl/4DGaussians)

### 3sters 系
- [DUSt3R](https://github.com/naver/dust3r)
- [MASt3R](https://github.com/naver/mast3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
- [Easi3R](https://github.com/Inception3D/Easi3R)
- [MUSt3R](https://github.com/naver/must3r)
- [Fast3R](https://github.com/facebookresearch/fast3r)
- [Splatt3R](https://github.com/btsmart/splatt3r)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [WinT3R](https://github.com/LiZizun/WinT3R)

### VGGT 系
- [VGGT](https://github.com/facebookresearch/vggt)
- [VGGSfM](https://github.com/facebookresearch/vggsfm)
- [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM)
- [StreamVGGT](https://github.com/wzzheng/StreamVGGT)
- [FastVGGT](https://github.com/mystorm16/FastVGGT)
- [Pi3](https://github.com/yyfz/Pi3)

### mds 系
- [MoGe2](https://github.com/microsoft/MoGe)
- [UniK3D](https://github.com/lpiccinelli-eth/UniK3D)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3)

## インストール
このシステムは Ubuntu を対象としています．Windows では一部利用できない手法があります．

`torch`，`torchvision`は起動環境の CUDA に合わせたものをインストールしてください．下記の例の実行環境には CUDA 12.1が入っています．
```
git clone --recursive https://github.com/WSuenaga/UNe3dMe.git
cd UNe3dMe

conda create -n UNe3dMe python=3.11 -y
conda activate UNe3dMe

pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

前処理手法として **FFmpeg**，**COLMAP** を用いています．
- FFmpegのインストール
    ```
    sudo apt update
    sudo apt install ffmpeg
    ```
- COLMAPのインストール  
    https://colmap.github.io/install.html  

各再構築手法はそれぞれ依存ライブラリおよび実行環境が異なるため，**Installation** を参考に，個別で環境構築を行ってください． 

```{toctree}
:hidden:
:includehidden:
:maxdepth: 2

Installation/index
Developer Guides/index
```