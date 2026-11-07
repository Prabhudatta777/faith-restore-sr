"""
Shared Dataset loader for Faith Restore SR - FIXED v2
Fix: TF.RandomCrop does not exist in torchvision.transforms.functional
We do random crop manually with random.randint.
HR is cropped to multiple of 4 so HR = LR * 4 exactly.
"""
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _crop_to_multiple_of_4(hr, lr, scale=4):
    w, h = hr.size
    w4 = (w // scale) * scale
    h4 = (h // scale) * scale
    if w4 != w or h4 != h:
        hr = hr.crop((0, 0, w4, h4))
        lr = lr.crop((0, 0, w4 // scale, h4 // scale))
    return hr, lr


class BSD100SRDataset(Dataset):
    def __init__(self, split="train", patch_size=64, augment=False, root="data/processed/BSD100_x4"):
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        self.root = Path(root) / split
        hr_dir = self.root / "HR"
        lr_dir = self.root / "LR"
        if hr_dir.exists() and lr_dir.exists():
            self.hr_files = sorted(hr_dir.glob("*.*"))
            self.lr_files = [lr_dir / f.name for f in self.hr_files]
        else:
            all_files = sorted(self.root.rglob("*.*"))
            hr_files = [p for p in all_files if "_HR" in p.name]
            if not hr_files:
                raise FileNotFoundError(f"Expected {hr_dir} and {lr_dir}. Check: {self.root}")
            self.hr_files = hr_files
            self.lr_files = []
            for f in self.hr_files:
                lr_name = f.name.replace("_HR", "_LR")
                cand = f.parent / lr_name
                if cand.exists():
                    self.lr_files.append(cand)
                else:
                    self.lr_files.append(self.root / "LR" / lr_name)
        assert len(self.hr_files) > 0

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        lr = Image.open(self.lr_files[idx]).convert("RGB")
        hr, lr = _crop_to_multiple_of_4(hr, lr, scale=4)
        if self.split == "train":
            w, h = hr.size
            if w >= self.patch_size and h >= self.patch_size:
                # FIXED: manual random crop, no TF.RandomCrop
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                hr = TF.crop(hr, top, left, self.patch_size, self.patch_size)
                lr = TF.crop(lr, top // 4, left // 4, self.patch_size // 4, self.patch_size // 4)
            if self.augment:
                if torch.rand(1) < 0.5:
                    hr = TF.hflip(hr)
                    lr = TF.hflip(lr)
        return TF.to_tensor(lr), TF.to_tensor(hr), self.hr_files[idx].name


if __name__ == "__main__":
    ds = BSD100SRDataset(split="test")
    print(f"Test size: {len(ds)}")
    lr, hr, name = ds[0]
    print(f"Sample {name}: LR {tuple(lr.shape)} -> HR {tuple(hr.shape)}")
    assert hr.shape[1] == lr.shape[1] * 4 and hr.shape[2] == lr.shape[2] * 4
    print("OK: HR = LR x 4 exactly")
    ds2 = BSD100SRDataset(split="train")
    lr2, hr2, _ = ds2[0]
    print(f"Train sample: LR {tuple(lr2.shape)} -> HR {tuple(hr2.shape)}")
    print("OK: train random crop works")