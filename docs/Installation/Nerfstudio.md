# Nerfstudio

## インストール
- 下記の例は CUDA 12.1環境です．`torch`，`torchvision`，`torchaudio`は起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/nerfstudio

    conda create --name nerfstudio -y python=3.10
    conda activate nerfstudio

    python -m pip install --upgrade pip

    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    # pip --no-build-isolation install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

    pip install -e .

    cd ../..
    ```
### SeaThru-NeRF のインストール
```
pip install git+https://github.com/AkerBP/seathru_nerf
```
### Splatfacto のインストール
```
pip install gsplat
```