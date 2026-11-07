"""
Prepare BSD100 dataset for Faith Restore SR - FIXED VERSION

Fix: Each image has 2 files (HR + LR), so:
  train = 80 images = 160 files
  val   = 20 images = 40 files
We now split by UNIQUE image name, not by file count.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed" / "BSD100_x4"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

def find_bsd100_root(extract_dir: Path) -> Path:
    candidates = list(extract_dir.rglob("bicubic_4x"))
    if not candidates:
        raise FileNotFoundError(f"bicubic_4x not found in {extract_dir}")
    return candidates[0].parent

def collect_images(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)

def get_image_id(file_path: Path) -> str:
    """
    Get unique image ID from filename.
    Examples:
      148026_HR.png -> 148026
      148026_LR.png -> 148026
      148026.png    -> 148026
    """
    name = file_path.stem  # without extension
    # Remove _HR, _LR, _hr, _lr, -HR, -LR suffixes
    for suffix in ["_HR", "_LR", "_hr", "_lr", "-HR", "-LR", "HR", "LR"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    # Also handle _x4_HR etc - keep only first part before _
    # e.g. 148026_x4_HR -> 148026
    if "_x4" in name:
        name = name.split("_x4")[0]
    return name

def main(zip_name: str = "BSD100.zip", seed: int = 42):
    zip_path = RAW_DIR / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f"Put your zip at: {zip_path}")

    extract_dir = RAW_DIR / "BSD100_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"1. Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    bsd_root = find_bsd100_root(extract_dir)
    scale_dir = bsd_root / "bicubic_4x"
    print(f"2. Found BSD100 at: {bsd_root}")
    print(f"   Using scale folder: {scale_dir}")

    train_src = scale_dir / "train"
    val_src = scale_dir / "val"

    train_files = collect_images(train_src)
    val_files = collect_images(val_src)

    # --- FIXED: Group by unique image ID ---
    train_ids = sorted(set(get_image_id(p) for p in train_files))
    val_ids = sorted(set(get_image_id(p) for p in val_files))

    print(f"3. Found: train={len(train_ids)} images ({len(train_files)} files), "
          f"val={len(val_ids)} images ({len(val_files)} files)")

    assert len(train_ids) == 80, f"Expected 80 train images, got {len(train_ids)}"
    assert len(val_ids) == 20, f"Expected 20 val images, got {len(val_ids)}"

    # Split 20 val -> 10 val + 10 test by sorted image ID
    new_val_ids = set(val_ids[:10])
    test_ids = set(val_ids[10:])

    print(f"   Split val: {len(new_val_ids)} val + {len(test_ids)} test")

    # Clean processed dir
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

    print("4. Copying files...")

    # Helper to copy files by ID set
    def copy_by_ids(files, src_root, dst_root, allowed_ids):
        count = 0
        for src in files:
            img_id = get_image_id(src)
            if img_id in allowed_ids:
                rel = src.relative_to(src_root)
                dst = dst_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                count += 1
        return count

    n_train = copy_by_ids(train_files, train_src, PROCESSED_DIR / "train", set(train_ids))
    n_val = copy_by_ids(val_files, val_src, PROCESSED_DIR / "val", new_val_ids)
    n_test = copy_by_ids(val_files, val_src, PROCESSED_DIR / "test", test_ids)

    splits = {
        "dataset": "BSD100",
        "scale": "bicubic_4x",
        "seed": seed,
        "note": "Each image has HR + LR files. Split by unique image ID.",
        "splits": {
            "train": train_ids,
            "val": sorted(new_val_ids),
            "test": sorted(test_ids),
        },
        "counts": {
            "train_images": len(train_ids),
            "val_images": len(new_val_ids),
            "test_images": len(test_ids),
            "train_files": n_train,
            "val_files": n_val,
            "test_files": n_test,
        },
    }
    with open(PROCESSED_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print("\nDone!")
    print(f"  Train: {len(train_ids)} images ({n_train} files)")
    print(f"  Val  : {len(new_val_ids)} images ({n_val} files)")
    print(f"  Test : {len(test_ids)} images ({n_test} files)")
    print(f"  Manifest: {PROCESSED_DIR / 'splits.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="BSD100.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(zip_name=args.zip, seed=args.seed)