# Depth-Anything-V2

## インストール
```
cd models/Depth-Anything-V2

conda create -n DA2 python=3.11 -y
conda activate DA2

pip install -r requirements.txt
pip install "gradio>=5.0,<6.0"
pip install "huggingface_hub==0.24.0" --no-deps --force-reinstall
```

## Checkpoint のダウンロード
- [ここ](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)からダウンロード．
- `checkpoints` ディレクトリを作成し，その中に配置．
    ```
    mkdir checkpoints
    ```