"""
Method 5 - RCAN-Lite: ResNet + Channel Attention (EDSR / RCAN idea)
Highest-PSNR method from curriculum.

Architecture:
  Head: Conv 3 -> 64
  Body: 5 Residual Groups, each with 5 Residual Blocks + Channel Attention
  Tail: Conv 64 -> 3
  Global Residual: Output = Bicubic Input + Network Output (learn residual only)

Why this beats SRCNN:
  - SRCNN: 3 layers, 57k params
  - RCAN-Lite: 5*5=25 residual blocks, ~1.2M params, skip connections let it train deep
  - Channel Attention: re-weights channels, edge channels get more weight
  - Global Residual: network only learns detail to ADD to bicubic, easier to learn

Usage:
  Train + Eval (30 epochs, quick test):
    python -m src.method5_rcan --train --epochs 30 --eval
  Train full (100 epochs, Colab GPU recommended):
    python -m src.method5_rcan --train --epochs 100 --eval
  Eval only (uses saved checkpoint):
    python -m src.method5_rcan --eval
  One-command runner will also pick it up if you add it to run_all.py

Output (SAME as other methods):
  results/method5_rcan/*_compare.png  (LR | SR | HR)
  results/method5_rcan/results.txt     (Avg PSNR, Avg SSIM)
  results/method5_rcan/rcan_best.pth   (best checkpoint by Val PSNR)

Expected:
  30 epochs, BSD100 70 train: ~27.0 dB, ~0.70 SSIM
  100 epochs + more data: ~27.5-28.0 dB
  World SOTA on BSD100 x4 is ~29 dB, PSNR 50 is impossible for x4 SR.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BSD100SRDataset
from src.utils import set_seed, evaluate_method, save_comparison

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/method5_rcan")
CKPT = RESULTS_DIR / "rcan_best.pth"

# ---------------- Model ----------------
class ChannelAttention(nn.Module):
    """RCAN Channel Attention: learn which channels matter."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w

class ResidualBlock(nn.Module):
    """EDSR-style Residual Block: Conv-ReLU-Conv + Channel Attention + Skip."""
    def __init__(self, channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.ca = ChannelAttention(channels)
    def forward(self, x):
        out = self.body(x)
        out = self.ca(out)
        return x + out  # skip connection - key to training deep nets

class ResidualGroup(nn.Module):
    def __init__(self, channels=64, n_blocks=5):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
    def forward(self, x):
        out = self.blocks(x)
        out = self.tail(out)
        return x + out  # group-level skip

class RCANLite(nn.Module):
    def __init__(self, channels=64, n_groups=5, n_blocks=5):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.body = nn.Sequential(*[ResidualGroup(channels, n_blocks) for _ in range(n_groups)])
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.tail = nn.Conv2d(channels, 3, 3, padding=1, bias=False)
    def forward(self, x):
        # x = bicubic-upsampled LR, shape [B,3,H*4,W*4]
        base = x
        feat = self.head(x)
        out = self.body(feat)
        out = self.body_tail(out)
        out = out + feat  # long skip: head -> tail
        out = self.tail(out)
        return base + out  # global residual: learn only detail to add to bicubic

# ---------------- Training ----------------
def train(epochs=30):
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | Training RCAN-Lite for {epochs} epochs")
    train_ds = BSD100SRDataset(split="train", patch_size=64, augment=True, repeat=20)
    val_ds = BSD100SRDataset(split="val", patch_size=64, augment=False, repeat=1)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = RCANLite().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    # LR scheduler: halve LR every 20 epochs so it settles - fixes SRCNN plateau
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()
    best_psnr = 0.0
    if CKPT.exists():
        try:
            ck = torch.load(CKPT, map_location=DEVICE)
            best_psnr = ck.get("best_psnr", 0.0)
            print(f"Existing best checkpoint Val PSNR: {best_psnr:.2f} dB (will overwrite only if better)")
        except Exception:
            pass
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for lr, hr in pbar:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            # Upsample LR x4 with bicubic first, same as SRCNN input
            lr_up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
            sr = model(lr_up)
            loss = loss_fn(sr, hr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        sched.step()
        # Val PSNR
        model.eval()
        val_psnr, _ = evaluate_method(model, val_loader, DEVICE, save_dir=None, method_name="val")
        print(f"Epoch {epoch}: Train Loss {total/len(train_loader):.5f} | Val PSNR {val_psnr:.2f} dB | LR {sched.get_last_lr()[0]:.2e}")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"model": model.state_dict(), "best_psnr": best_psnr, "epoch": epoch}, CKPT)
            print(f"  -> New best! Saved to {CKPT}")
    print(f"Training done. Best Val PSNR: {best_psnr:.2f} dB")

# ---------------- Eval ----------------
@torch.no_grad()
def evaluate():
    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT.exists():
        print(f"No checkpoint found at {CKPT}. Train first: python -m src.method5_rcan --train --epochs 30 --eval")
        return
    model = RCANLite().to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded checkpoint epoch {ck.get('epoch')} Val PSNR {ck.get('best_psnr', 0):.2f} dB")
    test_ds = BSD100SRDataset(split="test", patch_size=100000, augment=False, repeat=1)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    # evaluate_method expects model that takes LR and returns SR.
    # Our model takes bicubic-upsampled LR, so wrap it:
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, lr):
            lr_up = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
            return self.m(lr_up)
    wrapped = Wrapper(model).to(DEVICE)
    avg_psnr, avg_ssim = evaluate_method(wrapped, test_loader, DEVICE, save_dir=str(RESULTS_DIR), method_name="method5_rcan")
    # Save results.txt SAME format as other methods
    with open(RESULTS_DIR / "results.txt", "w") as f:
        f.write(f"Method 5: RCAN-Lite (ResNet + Channel Attention)\n")
        f.write(f"Avg PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
    print(f"\nMethod 5 RCAN-Lite - Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f}")
    print(f"Saved: {RESULTS_DIR}/results.txt and {RESULTS_DIR}/*_compare.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.train:
        train(epochs=args.epochs)
    if args.eval:
        evaluate()
    if not args.train and not args.eval:
        print("Use: python -m src.method5_rcan --train --epochs 30 --eval  OR  python -m src.method5_rcan --eval")
