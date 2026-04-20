# MonST3R

## インストール
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/monst3r

    conda create -n monst3r python=3.11 cmake=3.14.0
    conda activate monst3r 

    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

    pip install -r requirements.txt

    pip install pillow-heif pyrender==0.1.44 kapture kapture-localization numpy-quaternion pycolmap poselib boto3 tensorflow wandb tensorboard prettytable scikit-image scikit-learn h5py gdown pypng

    pip install -e viser
    ```

    ```
    cd croco/models/curope/
    python setup.py build_ext --inplace
    cd ../../../../..
    ```

## Checkpoint のダウンロード
```
cd models/monst3r/data
bash download_ckpt.sh
cd ../../..
```

## 既知の不具合と対処法
- このドキュメントの作成時点で，既知の不具合が確認されています．環境構築，推論等に失敗した場合は，以下を試してみてください．
    - demo.pyの297行目を以下に修正．
        ```
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                        minimum=1, maximum=10, step=1, visible=False)
        ```
    - typo の修正．
        ```
        cd models/monst3r/third_party/RAFT/core/configs
        ren congif_spring_M.json config_spring_M.json
        ```