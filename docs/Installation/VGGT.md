# VGGT

## インストール
- requirements_demo.txt の最初に以下を記載．
    ```
    numpy<2.0
    ```
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/vggt

    conda create -n vggt python=3.11
    conda activate vggt

    pip install -r requirements.txt

    pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

    pip install -r requirements_demo.txt
    ```