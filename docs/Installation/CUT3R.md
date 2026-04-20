# CUT3R

## インストール
- requirements.txt内のtransformersを下記の様に書き換えてください．
    ```
    transformers==4.40.2
    ```
- 下記の例は CUDA 12.1環境です．起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/CUT3R

    conda create -n cut3r python=3.11 cmake=3.14.0 -y
    conda activate cut3r

    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt

    conda install 'llvm-openmp<16'

    pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git

    pip install evo
    pip install open3d
    ```

    ```
    cd src/croco/models/curope/
    python setup.py build_ext --inplace
    cd ../../../../
    ```

## Checkpoint のダウンロード
```
mkdir src && cd src

gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link

cd ..
```