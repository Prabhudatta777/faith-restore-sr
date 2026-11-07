"""
Method 4: CNN-Based SR (SRCNN)
Paper: Dong et al. ECCV 2014
Architecture:
  Conv 9x9, 64 + ReLU -> Patch Extraction
  Conv 1x1, 32 + ReLU -> Non-linear Mapping
  Conv 5x5, 3         -> Reconstruction
Usage:
  python -m src.method4_srcnn --train --epochs 30 --eval
  python -m src.method4_srcnn --train --epochs 5 --eval  (quick test)
  python -m src.method4_srcnn --eval (eval only)
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import BSD100SRDataset
from src.utils import set_seed, evaluate_method, calculate_psnr, tensor_to_numpy

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def bicubic_upsample(lr, scale=4):
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)

CHECKPOINT = Path("results/method4_srcnn/srcnn_best.pth")

def train(epochs=30, batch_size=16, lr=1e-3):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    train_ds = BSD100SRDataset(split="train", patch_size=64, augment=True)
    val_ds = BSD100SRDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    model = SRCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    best_psnr = 0.0
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(range(20), desc=f"Epoch {epoch}/{epochs}")
        for _ in pbar:
            for lr_b, hr_b, _ in train_loader:
                lr_b, hr_b = lr_b.to(device), hr_b.to(device)
                lr_up = bicubic_upsample(lr_b, scale=4)
                optimizer.zero_grad()
                sr = model(lr_up)
                sr = torch.clamp(sr, 0, 1)
                loss = criterion(sr, hr_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.5f}")
        model.eval()
        psnr_list = []
        with torch.no_grad():
            for lr_b, hr_b, _ in val_loader:
                lr_b, hr_b = lr_b.to(device), hr_b.to(device)
                lr_up = bicubic_upsample(lr_b, scale=4)
                sr = torch.clamp(model(lr_up), 0, 1)
                hr_np = tensor_to_numpy(hr_b[0])
                sr_np = tensor_to_numpy(sr[0])
                psnr_list.append(calculate_psnr(hr_np, sr_np))
        avg_val_psnr = sum(psnr_list) / len(psnr_list)
        scheduler.step(avg_val_psnr)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val PSNR {avg_val_psnr:.2f} dB")
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  -> Saved BEST model (PSNR {best_psnr:.2f}) to {CHECKPOINT}")
    print(f"\nTraining done! Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoint: {CHECKPOINT}")

_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def srcnn_method(lr_tensor):
    global _MODEL
    lr_up = bicubic_upsample(lr_tensor.to(_DEVICE), scale=4)
    with torch.no_grad():
        sr = _MODEL(lr_up)
    return torch.clamp(sr.cpu(), 0, 1)

def evaluate():
    global _MODEL
    set_seed(42)
    if not CHECKPOINT.exists():
        print(f"ERROR: No checkpoint found at {CHECKPOINT}")
        print("Run training first: python -m src.method4_srcnn --train --epochs 30")
        return
    print("=" * 55)
    print("METHOD 4: CNN (SRCNN) - EVALUATION (x4)")
    print("=" * 55)
    _MODEL = SRCNN().to(_DEVICE)
    _MODEL.load_state_dict(torch.load(CHECKPOINT, map_location=_DEVICE))
    _MODEL.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")
    n_params = sum(p.numel() for p in _MODEL.parameters())
    print(f"SRCNN Parameters: {n_params:,}")
    test_ds = BSD100SRDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test images: {len(test_ds)}")
    results = evaluate_method(sr_function=srcnn_method, dataloader=test_loader, save_dir=Path("results/method4_srcnn"), max_save=3)
    out_path = Path("results/method4_srcnn/results.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Method 4: CNN (SRCNN) (x4) - BSD100\n")
        f.write("=" * 40 + "\n")
        f.write(f"Parameters: {n_params}\n")
        f.write(f"Avg PSNR: {results['avg_psnr']:.2f} dB\n")
        f.write(f"Avg SSIM: {results['avg_ssim']:.4f}\n")
        f.write(f"Images: {results['num_images']}\n")
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train SRCNN")
    parser.add_argument("--eval", action="store_true", help="Evaluate SRCNN on test set")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    args = parser.parse_args()
    if not args.train and not args.eval:
        args.train = True
        args.eval = True
    if args.train:
        train(epochs=args.epochs)
    if args.eval:
        evaluate()