"""
Method 5 - RCAN-Lite: ResNet + Channel Attention
Faith Restore SR - Single Image Super Resolution (x4)

Architecture:
    Head: Conv 3 -> 64
    Body: 5 Residual Groups
          Each group contains 5 Residual Blocks + Channel Attention
    Tail: Conv 64 -> 3
    Global Residual: Bicubic Input + Network Output

Features:
    - Residual learning
    - Channel Attention
    - Global skip connections
    - MPS acceleration for Apple Silicon
    - Automatic best-checkpoint saving
    - PSNR and SSIM evaluation

Usage:

    Train + Evaluate:
        python -m src.method5_rcanlite --train --epochs 30 --eval

    Train for 100 epochs:
        python -m src.method5_rcanlite --train --epochs 100 --eval

    Evaluate existing checkpoint:
        python -m src.method5_rcanlite --eval

    Custom batch size:
        python -m src.method5_rcanlite --train --epochs 30 --batch-size 4 --eval

Output:
    results/method5_rcan/
        rcan_best.pth
        results.txt
        *_compare.png
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BSD100SRDataset
from src.utils import set_seed, evaluate_method


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

RESULTS_DIR = Path("results/method5_rcan")
CKPT = RESULTS_DIR / "rcan_best.pth"

SCALE = 4
PATCH_SIZE = 64
BATCH_SIZE = 4
LEARNING_RATE = 1e-4


# ============================================================
# CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    """
    RCAN Channel Attention.

    Learns the importance of individual feature channels.
    """

    def __init__(self, channels=64, reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // reduction,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        weights = self.avg_pool(x)

        weights = self.fc(weights)

        return x * weights


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    """
    EDSR-style residual block.

    Conv -> ReLU -> Conv -> Channel Attention -> Skip
    """

    def __init__(self, channels=64):

        super().__init__()

        self.body = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.ca = ChannelAttention(channels)

    def forward(self, x):

        out = self.body(x)

        out = self.ca(out)

        return x + out


# ============================================================
# RESIDUAL GROUP
# ============================================================

class ResidualGroup(nn.Module):

    def __init__(self, channels=64, n_blocks=5):

        super().__init__()

        self.blocks = nn.Sequential(

            *[
                ResidualBlock(channels)
                for _ in range(n_blocks)
            ]

        )

        self.tail = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):

        out = self.blocks(x)

        out = self.tail(out)

        return x + out


# ============================================================
# RCAN-LITE NETWORK
# ============================================================

class RCANLite(nn.Module):

    """
    RCAN-Lite Super Resolution Network.

    Input:
        Bicubic-upsampled LR image [B, 3, H, W]

    Output:
        Super-resolved image [B, 3, H, W]
    """

    def __init__(
        self,
        channels=64,
        n_groups=5,
        n_blocks=5
    ):

        super().__init__()

        # Initial feature extraction
        self.head = nn.Conv2d(
            3,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        # Residual groups
        self.body = nn.Sequential(

            *[
                ResidualGroup(channels, n_blocks)
                for _ in range(n_groups)
            ]

        )

        # Long skip connection
        self.body_tail = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        # Reconstruction
        self.tail = nn.Conv2d(
            channels,
            3,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):

        base = x

        feat = self.head(x)

        out = self.body(feat)

        out = self.body_tail(out)

        out = out + feat

        out = self.tail(out)

        # Global residual learning
        return base + out


# ============================================================
# SUPER RESOLUTION WRAPPER
# ============================================================

class RCANWrapper(nn.Module):

    """
    Adapts RCANLite to the shared evaluation interface.

    evaluate_method supplies LR images.
    RCANLite expects bicubic-upsampled LR images.

    This wrapper performs the upsampling and device transfer.
    """

    def __init__(self, model, device):

        super().__init__()

        self.model = model

        self.device = device

    def forward(self, lr):

        lr = lr.to(self.device)

        lr_up = F.interpolate(

            lr,

            scale_factor=SCALE,

            mode="bicubic",

            align_corners=False

        )

        return self.model(lr_up)


# ============================================================
# TRAINING
# ============================================================

def train(epochs=30, batch_size=BATCH_SIZE):

    set_seed(42)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)

    print("METHOD 5: RCAN-LITE TRAINING")

    print("=" * 60)

    print(f"Device: {DEVICE}")

    print(f"Epochs: {epochs}")

    print(f"Batch size: {batch_size}")

    print(f"Patch size: {PATCH_SIZE}")

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_ds = BSD100SRDataset(

        split="train",

        patch_size=PATCH_SIZE,

        augment=True

    )

    val_ds = BSD100SRDataset(

        split="val",

        patch_size=PATCH_SIZE,

        augment=False

    )

    print(f"\nTraining images: {len(train_ds)}")

    print(f"Validation images: {len(val_ds)}")

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_ds,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=False

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=1,

        shuffle=False,

        num_workers=0

    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RCANLite().to(DEVICE)

    params = sum(

        p.numel()
        for p in model.parameters()

    )

    print(f"\nModel parameters: {params:,}")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE

    )

    scheduler = torch.optim.lr_scheduler.StepLR(

        optimizer,

        step_size=20,

        gamma=0.5

    )

    loss_fn = nn.MSELoss()

    # Start a fresh training run.
    # Do not compare a randomly initialized model against
    # an old checkpoint's validation score.

    best_psnr = float("-inf")

    # Wrapper used during validation
    val_model = RCANWrapper(model, DEVICE)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0

        pbar = tqdm(

            train_loader,

            desc=f"Epoch {epoch}/{epochs}",

            leave=True

        )

        for lr, hr, _ in pbar:

            lr = lr.to(DEVICE)

            hr = hr.to(DEVICE)

            # Bicubic upsampling x4
            lr_up = F.interpolate(

                lr,

                scale_factor=SCALE,

                mode="bicubic",

                align_corners=False

            )

            # Forward pass
            sr = model(lr_up)

            # Reconstruction loss
            loss = loss_fn(sr, hr)

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(

                loss=f"{loss.item():.5f}"

            )

        # Learning rate scheduling
        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        metrics = evaluate_method(

            val_model,

            val_loader,

            save_dir=None

        )

        val_psnr = metrics["avg_psnr"]

        val_ssim = metrics["avg_ssim"]

        current_lr = scheduler.get_last_lr()[0]

        print("\n" + "-" * 55)

        print(f"Epoch: {epoch}/{epochs}")

        print(f"Train Loss: {avg_loss:.6f}")

        print(f"Val PSNR: {val_psnr:.2f} dB")

        print(f"Val SSIM: {val_ssim:.4f}")

        print(f"Learning Rate: {current_lr:.2e}")

        print("-" * 55)

        # ----------------------------------------------------
        # SAVE BEST CHECKPOINT
        # ----------------------------------------------------

        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(

                {

                    "model": model.state_dict(),

                    "best_psnr": best_psnr,

                    "best_ssim": val_ssim,

                    "epoch": epoch

                },

                CKPT

            )

            print(f"New best checkpoint saved: {CKPT}")

    print("\n" + "=" * 60)

    print("TRAINING COMPLETED")

    print(f"Best Validation PSNR: {best_psnr:.2f} dB")

    print(f"Checkpoint: {CKPT}")

    print("=" * 60)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()

def evaluate():

    set_seed(42)

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    print("=" * 60)

    print("METHOD 5: RCAN-LITE EVALUATION")

    print("=" * 60)

    if not CKPT.exists():

        print("\nNo checkpoint found!")

        print("Train the model first.")

        return

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = RCANLite().to(DEVICE)

    ck = torch.load(

        CKPT,

        map_location=DEVICE

    )

    model.load_state_dict(ck["model"])

    model.eval()

    print(f"\nLoaded checkpoint: {CKPT}")

    print(f"Best epoch: {ck.get('epoch')}")

    print(

        f"Best validation PSNR: "
        f"{ck.get('best_psnr', 0):.2f} dB"

    )

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    test_ds = BSD100SRDataset(

        split="test",

        patch_size=PATCH_SIZE,

        augment=False

    )

    test_loader = DataLoader(

        test_ds,

        batch_size=1,

        shuffle=False,

        num_workers=0

    )

    print(f"\nTest images: {len(test_ds)}")

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    wrapped_model = RCANWrapper(

        model,

        DEVICE

    )

    metrics = evaluate_method(

        wrapped_model,

        test_loader,

        save_dir=str(RESULTS_DIR),

        max_save=10

    )

    avg_psnr = metrics["avg_psnr"]

    avg_ssim = metrics["avg_ssim"]

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(

        RESULTS_DIR / "results.txt",

        "w"

    ) as f:

        f.write(

            "Method 5: RCAN-Lite "
            "(ResNet + Channel Attention)\n"

        )

        f.write(

            f"Avg PSNR: {avg_psnr:.2f} dB\n"

        )

        f.write(

            f"Avg SSIM: {avg_ssim:.4f}\n"

        )

        f.write(

            f"Images: {metrics['num_images']}\n"

        )

        f.write(

            f"Checkpoint Epoch: {ck.get('epoch')}\n"

        )

    print("\n" + "=" * 60)

    print("FINAL RCAN-LITE RESULTS")

    print("=" * 60)

    print(f"Average PSNR: {avg_psnr:.2f} dB")

    print(f"Average SSIM: {avg_ssim:.4f}")

    print(f"Images evaluated: {metrics['num_images']}")

    print(f"\nResults saved in: {RESULTS_DIR}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description="Faith Restore SR - RCAN-Lite"

    )

    parser.add_argument(

        "--train",

        action="store_true",

        help="Train RCAN-Lite"

    )

    parser.add_argument(

        "--eval",

        action="store_true",

        help="Evaluate RCAN-Lite"

    )

    parser.add_argument(

        "--epochs",

        type=int,

        default=30,

        help="Number of training epochs"

    )

    parser.add_argument(

        "--batch-size",

        type=int,

        default=BATCH_SIZE,

        help="Training batch size"

    )

    args = parser.parse_args()

    if args.train:

        train(

            epochs=args.epochs,

            batch_size=args.batch_size

        )

    if args.eval:

        evaluate()

    if not args.train and not args.eval:

        parser.print_help()