# Easi3R

## インストール
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/Easi3R

    conda create -n easi3r python=3.10 cmake=3.31
    conda activate easi3r

    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  
    pip install -r requirements.txt

    pip install -e viser

    pip install -e third_party/sam2 --verbose
    ```

    ```
    cd croco/models/curope/
    python setup.py build_ext --inplace
    cd ../../../
    ```

## Checkpoint のダウンロード
```
cd models/Easi3R/data
bash download_ckpt.sh
cd ../../..
```

## 既知の不具合と対処法
- このドキュメントの作成時点で，既知の不具合が確認されています．環境構築，推論等に失敗した場合は，以下を試してみてください．
    - demo.pyの324行目を以下に修正．
        ```
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                                minimum=1, maximum=10, step=1, visible=False)
        ```
    - demo.pyの505行目を以下に修正．
        ```
        scene, outfile, *_ = recon_fun(
        ```