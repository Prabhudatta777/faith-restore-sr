"""
Method 1: Bicubic Interpolation Baseline
Curriculum Topic: Review of Digital Image Processing

This is the simplest SR method.
It does NOT learn anything - it just interpolates pixels.

We use this as baseline:
  Method 2, 3, 4 must beat this PSNR / SSIM,
  otherwise they are useless.

Usage:
  python -m src.method1_bicubic
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from src.dataset import BSD100SRDataset
from src.utils import set_seed, evaluate_method

def bicubic_upsample(lr_tensor: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """
    Bicubic upsampling.

    Input:
        lr_tensor: (B, C, H, W)  e.g. (1, 3, 64, 64)
    Output:
        sr_tensor: (B, C, H*scale, W*scale) e.g. (1, 3, 256, 256)

    Args:
        scale: 4 for our project (BSD100 x4)
    """
    # align_corners=False is standard for SR in PyTorch
    sr = F.interpolate(
        lr_tensor,
        scale_factor=scale,
        mode="bicubic",
        align_corners=False
    )
    # Clip to valid range 0-1
    sr = torch.clamp(sr, 0.0, 1.0)
    return sr

# Wrapper so evaluate_method() can call it
# evaluate_method expects: function(lr) -> sr
def bicubic_method(lr_tensor: torch.Tensor) -> torch.Tensor:
    return bicubic_upsample(lr_tensor, scale=4)

def main():
    set_seed(42)

    print("=" * 50)
    print("METHOD 1: BICUBIC BASELINE (x4)")
    print("=" * 50)

    # 1. Load TEST set only - Bicubic needs NO training
    # Test = 10 images (from your split)
    from torch.utils.data import DataLoader

    test_ds = BSD100SRDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"Test images: {len(test_ds)}")

    # 2. Evaluate on all 10 test images
    # This will:
    # - Run bicubic on each LR
    # - Compare SR vs HR Ground Truth
    # - Calculate PSNR + SSIM
    # - Save first 3 comparison images to results/
    results = evaluate_method(
        sr_function=bicubic_method,
        dataloader=test_loader,
        save_dir=Path("results/method1_bicubic"),
        max_save=3
    )

    # 3. Save results to a text file for your report
    out_path = Path("results/method1_bicubic/results.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Method 1: Bicubic Baseline (x4) - BSD100\n")
        f.write("=" * 40 + "\n")
        f.write(f"Avg PSNR: {results['avg_psnr']:.2f} dB\n")
        f.write(f"Avg SSIM: {results['avg_ssim']:.4f}\n")
        f.write(f"Images: {results['num_images']}\n")

    print(f"\nResults saved to: {out_path}")
    print("\nThis is your BASELINE. Save these numbers!")
    print("Method 2, 3, 4 must beat this.")

if __name__ == "__main__":
    main()