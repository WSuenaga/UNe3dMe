# MoGe-2

## インストール
- 下記の例は CUDA 12.1環境です．`torch`，`torchvision`は起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/MoGe

    conda create -n MoGe python=3.10
    conda activate MoGe

    pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

    pip install "numpy<2.0" gradio==3.30.0 click==8.1.7 opencv-python==4.10.0.84 scipy==1.14.1 matplotlib==3.9.2 trimesh==4.5.1 pillow==10.4.0 huggingface_hub==0.25.2 git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183　git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6
    ```
