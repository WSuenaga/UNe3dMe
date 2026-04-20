# Depth-Anything-3

## インストール
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/Depth-Anything-3

    conda create -n DA3 python=3.11 -y
    conda activate DA3

    pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121

    pip install -e . 
    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 
    pip install -e ".[app]" 
    pip install -e
    ```