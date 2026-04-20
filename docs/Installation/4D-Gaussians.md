# 4D-Gaussians

## インストール
- 下記の例は CUDA 12.1環境です．`torch`，`torchvision`，`torchaudio`は起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/4DGaussians

    conda create -n Gaussians4D python=3.10
    conda activate Gaussians4D

    python -m pip install --upgrade pip

    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

    pip install "numpy<2.0" mmcv==1.6.0 matplotlib argparse lpips plyfile pytorch_msssim open3d imageio[ffmpeg]

    pip install submodules/depth-diff-gaussian-rasterization
    pip install submodules/simple-knn 

    cd ../..
    ```