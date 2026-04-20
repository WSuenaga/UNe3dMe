# WinT3R

## インストール
- equirements.txtの最初に以下を記載．
    ```
    numpy<2.0
    ```
- requirements.txt内のtransformersを下記の様に書き換えてください．
    ```
    transformers==4.40.2
    ```
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/WinT3R

    conda create -n WinT3R python=3.10 -y
    conda activate WinT3R

    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

## Checkpoint のダウンロード
- 以下をダウンロード．
    - https://huggingface.co/lizizun/WinT3R/resolve/main/pytorch_model.bin
- `checkpoints` ディレクトリを作成し，その中に配置．
    ```
    mkdir checkpoints && cd checkpoints
    ```