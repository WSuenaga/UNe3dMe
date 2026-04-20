# Mip-Splatting

## インストール
- 下記の例は CUDA 12.1環境です．`torch`，`torchvision`は起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/mip-splatting

    conda create -y -n mip-splatting python=3.10
    conda activate mip-splatting

    python -m pip install --upgrade pip

    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

    pip install "numpy<2.0" open3d plyfile ninja GPUtil opencv-python lpips

    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn

    cd ../..
    ```