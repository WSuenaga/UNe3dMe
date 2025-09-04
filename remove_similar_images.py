import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# SSIMの計算
def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, data_range=img2.max() - img2.min(), channel_axis=-1)

# フォルダ内の画像を比較して、類似画像を削除
def process_folder(input_dir: str, output_dir: str, ssim_threshold: float):
    images = sorted(os.listdir(input_dir))
    if not images:
        print("入力フォルダに画像がありません")
        return

    reference_path = os.path.join(input_dir, images[0])
    reference_img = cv2.imread(reference_path)
    if reference_img is None:
        print(f"基準画像の読み込みに失敗: {images[0]}")
        return

    # 最初の画像は残す
    csv_data = []
    current_group = [images[0]]

    for img_name in tqdm(images[1:], desc="Processing images"):
        img_path = os.path.join(input_dir, img_name)
        current_img = cv2.imread(img_path)
        if current_img is None:
            print(f"{img_name}: 読み込み失敗")
            continue

        ssim_val = compute_ssim(reference_img, current_img)
        if ssim_val < ssim_threshold:
            # 変化が大きい画像（残す）
            current_group = [img_name]
            reference_img = current_img
        else:
            # 変化が小さい画像（削除）
            os.remove(img_path)

    return csv_data

def main(input_dir: str, output_dir: str, ssim_threshold: float):
    original_count = len([f for f in os.listdir(input_dir) if f.endswith(".png")])
    csv_data = process_folder(input_dir, output_dir, ssim_threshold)
    remaining_count = len([f for f in os.listdir(output_dir) if f.endswith(".png")])

    if original_count == 0:
        print("画像が存在しません。")
        return

    reduction_rate = (original_count - remaining_count) / original_count * 100
    print(f"冗長性の削除処理後の画像数: {remaining_count}")
    print(f"削減率: {reduction_rate:.2f}%")

