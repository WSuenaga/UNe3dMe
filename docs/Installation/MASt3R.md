# MASt3R

## インストール
- 下記の例は CUDA 12.1環境です．`torch`，`torchvision`は起動環境の CUDA に合わせたものをインストールしてください．
    ```
    cd models/mast3r

    conda create -n mast3r python=3.11 cmake=3.14.0
    conda activate mast3r 

    python -m pip install --upgrade pip

    pip install "numpy<2.0" torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit

    pip install ninja

    pip install "numpy<2.0" scikit-learn
    pip install "numpy<2.0" roma gradio matplotlib tqdm opencv-python scipy einops trimesh tensorboard "pyglet<2" "huggingface-hub[torch]>=0.22"
    pip install "numpy<2.0" pillow-heif pyrender==0.1.44 "pyglet<2" kapture kapture-localization numpy-quaternion pycolmap poselib
    ```

    ```
    pip install faiss-gpu # or faiss-cpu
    pip install cython

    git clone https://github.com/jenicek/asmk
    cd asmk/cython/
    cythonize *.pyx
    cd ..
    pip install .  # or python3 setup.py build_ext --inplace
    cd ..
    ```

    ```
    cd dust3r/croco/models/curope/
    python setup.py build_ext --inplace
    cd ../../../../../..
    ```