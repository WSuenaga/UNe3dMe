# FastVGGT

## インストール
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/FastVGGT

    conda create -n fastvggt python=3.10 -y
    conda activate fastvggt

    pip install -r requirements.txt

    pip uninstall torch torchvision -y
    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    ```
    
## Checkpoint のダウンロード
- [ここ](https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt)からダウンロード．
- `ckpt` ディレクトリを作成し，その中に配置．
    ```
    mkdir ckpt && cd ckpt
    ```