# Vanilla GS

## インストール
- 下記の例は CUDA 12.1環境です．`torch`，`torchvision`，`torchaudio`は起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/gaussian-splatting

    conda create -n gaussian_splatting python=3.10
    conda activate gaussian_splatting

    python -m pip install --upgrade pip

    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

    pip install "numpy<2.0" opencv-python plyfile tqdm joblib

    pip install submodules\diff-gaussian-rasterization
    pip install submodules\simple-knn
    pip install submodules\fused-ssim 

    cd ../..
    ```