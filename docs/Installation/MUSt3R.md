# MUSt3R

## インストール
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/must3r

    micromamba create -n must3r python=3.11 cmake=3.14.0
    micromamba activate must3r 

    pip3 install "numpy<2.0" torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip3 install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu12

    pip install --no-deps git+https://github.com/naver/must3r.git
    ```

## Checkpoint のダウンロード
- [ここ](https://github.com/naver/must3r?tab=readme-ov-file)から以下のファイルをダウンロード．
    - `MUSt3R_512.pth`
    - `MUSt3R_512_retrieval_codebook.pkl`
    - `MUSt3R_512_retrieval_trainingfree.pth`
- `ckpt` ディレクトリを作成し，ダウンロードしたものを配置．
    ```
    mkdir ckpt
    ```