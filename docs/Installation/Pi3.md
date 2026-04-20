# FastVGGT

## インストール
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/Pi3

    conda create -n Pi3 python=3.11 -y
    conda activate Pi3

    pip install -r requirements.txt
    ```
    ```
    pip uninstall torch torchvision torchaudio -y

    pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
