"""
utils.py - FIXED for odd sizes.
calculate_psnr / calculate_ssim now crop to smallest common size,
so (321,481) vs (320,480) will not crash.
"""

import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as sk_ssim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")


def _crop_to_common(hr, sr):
    # hr, sr: (H,W,3) numpy. Crop both to min H, min W.
    h = min(hr.shape[0], sr.shape[0])
    w = min(hr.shape[1], sr.shape[1])
    return hr[:h, :w], sr[:h, :w]


def calculate_psnr(hr, sr, max_val=255.0):
    hr, sr = _crop_to_common(hr.astype(np.float64), sr.astype(np.float64))
    mse = np.mean((hr - sr) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def calculate_ssim(hr, sr):
    hr, sr = _crop_to_common(hr.astype(np.uint8), sr.astype(np.uint8))
    try:
        return float(sk_ssim(hr, sr, channel_axis=-1, data_range=255))
    except TypeError:
        return float(sk_ssim(hr, sr, multichannel=True, data_range=255))


def tensor_to_numpy(tensor):
    img = tensor.detach().cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255.0).round().astype(np.uint8)


def numpy_to_tensor(img_np):
    img = img_np.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def save_comparison(lr, sr, hr, save_path, title=""):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr); axes[0].set_title("LR Input"); axes[0].axis("off")
    axes[1].imshow(sr); axes[1].set_title(f"SR Output\n{title}"); axes[1].axis("off")
    axes[2].imshow(hr); axes[2].set_title("HR Ground Truth"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def save_single_image(img_np, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_np.astype(np.uint8)).save(save_path)


def evaluate_method(sr_function, dataloader, save_dir=None, max_save=3):
    psnr_list = []
    ssim_list = []
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, (lr, hr, name) in enumerate(dataloader):
        with torch.no_grad():
            sr = sr_function(lr)
        hr_np = tensor_to_numpy(hr[0])
        sr_np = tensor_to_numpy(sr[0])
        lr_np = tensor_to_numpy(lr[0])
        # For figure only: resize LR to HR size
        lr_pil = Image.fromarray(lr_np).resize((hr_np.shape[1], hr_np.shape[0]), Image.BICUBIC)
        lr_np_big = np.array(lr_pil)
        psnr = calculate_psnr(hr_np, sr_np)
        ssim = calculate_ssim(hr_np, sr_np)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        if save_dir and saved < max_save:
            img_name = name[0] if isinstance(name, (list, tuple)) else str(name)
            # Crop all to common size for saving
            h = min(hr_np.shape[0], sr_np.shape[0], lr_np_big.shape[0])
            w = min(hr_np.shape[1], sr_np.shape[1], lr_np_big.shape[1])
            save_comparison(lr_np_big[:h, :w], sr_np[:h, :w], hr_np[:h, :w], save_dir / f"{img_name}_compare.png", title=f"PSNR {psnr:.2f} | SSIM {ssim:.4f}")
            saved += 1
        print(f"[{i+1}/{len(dataloader)}] {name} - PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")
    results = {"avg_psnr": float(np.mean(psnr_list)), "avg_ssim": float(np.mean(ssim_list)), "num_images": len(psnr_list)}
    print("\n===== FINAL RESULT =====")
    print(f"Avg PSNR: {results['avg_psnr']:.2f} dB")
    print(f"Avg SSIM: {results['avg_ssim']:.4f}")
    print(f"Images: {results['num_images']}")
    return results


if __name__ == "__main__":
    set_seed(42)
    dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    print("PSNR identical:", calculate_psnr(dummy, dummy))
    print("SSIM identical:", calculate_ssim(dummy, dummy))
    # Odd-size test that used to crash
    a = np.random.randint(0, 255, (321, 481, 3), dtype=np.uint8)
    b = np.random.randint(0, 255, (320, 480, 3), dtype=np.uint8)
    print("PSNR odd-size (should not crash):", calculate_psnr(a, b))