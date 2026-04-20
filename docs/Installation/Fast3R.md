# Fast3R

## インストール
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/fast3r

    conda create -n fast3r python=3.11 cmake=3.14.0 -y
    conda activate fast3r

    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    pip install -r requirements.txt

    pip install -e .
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