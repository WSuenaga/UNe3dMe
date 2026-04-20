# UniK3D

## インストール
```
cd models/UniK3D

conda create -n UniK3D python=3.11
conda activate UniK3D

pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121

conda install -c conda-forge libjpeg-turbo
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

cd ./unik3d/ops/knn;bash compile.sh;cd ../../../
```