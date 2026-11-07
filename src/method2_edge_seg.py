"""
Method 2: Edge + Segmentation-Guided SR
Curriculum: Edge Detection (Sobel, Canny) + Segmentation + DIP

Idea: Bicubic blurs everything. We sharpen ONLY edges,
smooth ONLY flat areas.
Usage: python -m src.method2_edge_seg
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import BSD100SRDataset
from src.utils import set_seed, evaluate_method, tensor_to_numpy, numpy_to_tensor


def get_edge_mask(gray):
    edges_canny = cv2.Canny(gray, 50, 150)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, grad_mask = cv2.threshold(grad_norm, 50, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(edges_canny, grad_mask)
    kernel = np.ones((3, 3), np.uint8)
    combined_dilated = cv2.dilate(combined, kernel, iterations=1)
    edge_mask = (combined_dilated > 0).astype(np.float32)
    edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 0)
    return edge_mask


def get_segmentation_mask(gray, edge_mask):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, otsu_mask = cv2.threshold(grad_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    textured = (otsu_mask > 0).astype(np.float32)
    flat_mask = (1.0 - textured) * (1.0 - edge_mask)
    flat_mask = np.clip(flat_mask, 0, 1)
    flat_mask = cv2.GaussianBlur(flat_mask, (5, 5), 0)
    return flat_mask


def edge_seg_sr_single_image(lr_np, scale=4):
    h_lr, w_lr = lr_np.shape[:2]
    h_hr, w_hr = h_lr * scale, w_lr * scale
    bicubic = cv2.resize(lr_np, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bicubic, cv2.COLOR_RGB2GRAY)
    edge_mask = get_edge_mask(gray)
    flat_mask = get_segmentation_mask(gray, edge_mask)
    edge_mask_3c = np.stack([edge_mask] * 3, axis=-1)
    flat_mask_3c = np.stack([flat_mask] * 3, axis=-1)
    blurred = cv2.GaussianBlur(bicubic, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(bicubic, 1.5, blurred, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.float32)
    smoothed = cv2.bilateralFilter(bicubic, d=9, sigmaColor=75, sigmaSpace=75)
    smoothed = smoothed.astype(np.float32)
    bicubic_f = bicubic.astype(np.float32)
    sr = bicubic_f + edge_mask_3c * (sharpened - bicubic_f)
    sr = (1.0 - flat_mask_3c) * sr + flat_mask_3c * smoothed
    sr = np.clip(sr, 0, 255).astype(np.uint8)
    return sr


def edge_seg_method(lr_tensor):
    batch = []
    for i in range(lr_tensor.shape[0]):
        lr_np = tensor_to_numpy(lr_tensor[i])
        sr_np = edge_seg_sr_single_image(lr_np, scale=4)
        batch.append(numpy_to_tensor(sr_np))
    return torch.stack(batch, dim=0)


def main():
    set_seed(42)
    print("=" * 55)
    print("METHOD 2: EDGE + SEGMENTATION-GUIDED SR (x4)")
    print("=" * 55)
    test_ds = BSD100SRDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test images: {len(test_ds)}")
    results = evaluate_method(
        sr_function=edge_seg_method,
        dataloader=test_loader,
        save_dir=Path("results/method2_edge_seg"),
        max_save=3
    )
    out_path = Path("results/method2_edge_seg/results.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Method 2: Edge + Segmentation-Guided SR (x4) - BSD100\n")
        f.write("=" * 50 + "\n")
        f.write(f"Avg PSNR: {results['avg_psnr']:.2f} dB\n")
        f.write(f"Avg SSIM: {results['avg_ssim']:.4f}\n")
        f.write(f"Images: {results['num_images']}\n")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()