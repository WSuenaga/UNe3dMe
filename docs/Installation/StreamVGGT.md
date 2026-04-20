# StreamVGGT

## インストール
- requirements.txt内のtransformersを下記の様に書き換えてください．
    ```
    transformers==4.40.2
    ```
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/VGGT-SLAM

    conda create -n StreamVGGT python=3.11 cmake=3.14.0 -y
    conda activate StreamVGGT 

    pip install -r requirements.txt

    pip uninstall torch torchvision -y
    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

    conda install 'llvm-openmp<16' -y
    ```
    
## Checkpoint のダウンロード
- [ここ](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt)からダウンロード．
- `ckpt` ディレクトリを作成し，その中に配置．
    ```
    mkdir ckpt && cd ckpt
    ```