"""
Method 3: Dictionary / Patch-Based SR
Curriculum: Feature Detection (HOG) + Bag of Words / VLAD (Codebook) + K-Means
Idea: Learn LR patch -> HR patch mapping from 70 train images.
Usage:
  python -m src.method3_dictionary
  python -m src.method3_dictionary --retrain
"""
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import BSD100SRDataset
from src.utils import set_seed, evaluate_method, tensor_to_numpy, numpy_to_tensor

LR_PATCH = 8
HR_PATCH = 32
SCALE = 4
STRIDE_TRAIN = 8  # FIX: 8 gives more patches
STRIDE_TEST = 8
N_CLUSTERS = 512
MAX_TRAIN_PATCHES = 20000
DICT_PATH = Path("results/method3_dictionary/dictionary.npz")

def extract_patches_pair(lr_img, hr_img, stride):
    if len(lr_img.shape) == 3:
        lr_gray = cv2.cvtColor(lr_img, cv2.COLOR_RGB2GRAY)
    else:
        lr_gray = lr_img
    if len(hr_img.shape) == 3:
        hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
    else:
        hr_gray = hr_img
    h_lr, w_lr = lr_gray.shape
    lr_patches, hr_patches = [], []
    for y in range(0, h_lr - LR_PATCH + 1, stride):
        for x in range(0, w_lr - LR_PATCH + 1, stride):
            lr_p = lr_gray[y:y+LR_PATCH, x:x+LR_PATCH]
            hr_p = hr_gray[y*SCALE:(y+LR_PATCH)*SCALE, x*SCALE:(x+LR_PATCH)*SCALE]
            lr_patches.append(lr_p)
            hr_patches.append(hr_p)
    return np.array(lr_patches), np.array(hr_patches)

def extract_hog_feature(lr_patch):
    patch = lr_patch.astype(np.float32)
    mean = patch.mean()
    std = patch.std() + 1e-6
    patch_norm = (patch - mean) / std
    gx = cv2.Sobel(patch_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch_norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    feat = np.concatenate([patch_norm.flatten(), mag.flatten()])
    return feat

def extract_features_batch(lr_patches):
    return np.array([extract_hog_feature(p) for p in lr_patches], dtype=np.float32)

def train_dictionary():
    print("=" * 55)
    print("TRAINING DICTIONARY - Method 3")
    print("=" * 55)
    train_ds = BSD100SRDataset(split="train", patch_size=100000)  # FIX: 100000 = no crop, full image
    print(f"Train images: {len(train_ds)}")
    all_lr, all_hr = [], []
    print(f"Extracting patches (stride={STRIDE_TRAIN})...")
    for idx in tqdm(range(len(train_ds))):
        lr_t, hr_t, name = train_ds[idx]
        lr_np = tensor_to_numpy(lr_t)
        hr_np = tensor_to_numpy(hr_t)
        lr_p, hr_p = extract_patches_pair(lr_np, hr_np, stride=STRIDE_TRAIN)
        all_lr.append(lr_p)
        all_hr.append(hr_p)
        total = sum(len(x) for x in all_lr)
        if total >= MAX_TRAIN_PATCHES:
            break
    all_lr = np.concatenate(all_lr, axis=0)
    all_hr = np.concatenate(all_hr, axis=0)
    if len(all_lr) > MAX_TRAIN_PATCHES:
        idx = np.random.choice(len(all_lr), MAX_TRAIN_PATCHES, replace=False)
        all_lr = all_lr[idx]
        all_hr = all_hr[idx]
    print(f"Total patches: {len(all_lr)}")
    print("Extracting HOG features...")
    feats = extract_features_batch(all_lr)
    print(f"Features shape: {feats.shape}")
    # FIX: K-Means needs n_samples >= n_clusters. Reduce K if needed.
    n_clusters_eff = min(N_CLUSTERS, len(all_lr))
    print(f"Running K-Means with K={n_clusters_eff} (asked {N_CLUSTERS}, have {len(all_lr)} patches)...")
    print("(This takes 1-3 minutes on laptop)")
    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feats)
    print("Computing average HR patch per cluster...")
    dict_hr = np.zeros((n_clusters_eff, HR_PATCH, HR_PATCH), dtype=np.float32)
    counts = np.zeros(n_clusters_eff, dtype=int)
    for i, label in enumerate(labels):
        dict_hr[label] += all_hr[i].astype(np.float32)
        counts[label] += 1
    for k in range(n_clusters_eff):
        if counts[k] > 0:
            dict_hr[k] /= counts[k]
        else:
            dict_hr[k] = all_hr.mean(axis=0)
    DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DICT_PATH, centers=kmeans.cluster_centers_.astype(np.float32), dict_hr=dict_hr, counts=counts)
    print(f"Dictionary saved to: {DICT_PATH}")
    return kmeans.cluster_centers_, dict_hr

def load_dictionary():
    if not DICT_PATH.exists():
        return None, None
    data = np.load(DICT_PATH)
    return data["centers"], data["dict_hr"]

def dictionary_sr_single_image(lr_np, centers, dict_hr):
    h_lr, w_lr = lr_np.shape[:2]
    h_hr, w_hr = h_lr * SCALE, w_lr * SCALE
    bicubic_rgb = cv2.resize(lr_np, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    lr_gray = cv2.cvtColor(lr_np, cv2.COLOR_RGB2GRAY)
    hr_acc = np.zeros((h_hr, w_hr), dtype=np.float32)
    weight = np.zeros((h_hr, w_hr), dtype=np.float32)
    for y in range(0, h_lr - LR_PATCH + 1, STRIDE_TEST):
        for x in range(0, w_lr - LR_PATCH + 1, STRIDE_TEST):
            lr_p = lr_gray[y:y+LR_PATCH, x:x+LR_PATCH]
            feat = extract_hog_feature(lr_p)
            dists = np.sum((centers - feat) ** 2, axis=1)
            best = int(np.argmin(dists))
            hr_p = dict_hr[best]
            y_hr, x_hr = y * SCALE, x * SCALE
            hr_acc[y_hr:y_hr+HR_PATCH, x_hr:x_hr+HR_PATCH] += hr_p
            weight[y_hr:y_hr+HR_PATCH, x_hr:x_hr+HR_PATCH] += 1.0
    weight[weight == 0] = 1.0
    hr_gray_sr = np.clip(hr_acc / weight, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(bicubic_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = hr_gray_sr
    sr_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return sr_rgb

_CENTERS = None
_DICT_HR = None

def dictionary_method(lr_tensor):
    global _CENTERS, _DICT_HR
    batch = []
    for i in range(lr_tensor.shape[0]):
        lr_np = tensor_to_numpy(lr_tensor[i])
        sr_np = dictionary_sr_single_image(lr_np, _CENTERS, _DICT_HR)
        batch.append(numpy_to_tensor(sr_np))
    return torch.stack(batch, dim=0)

def main(retrain=False):
    set_seed(42)
    centers, dict_hr = load_dictionary()
    if centers is None or retrain:
        centers, dict_hr = train_dictionary()
    else:
        print(f"Loaded existing dictionary from {DICT_PATH}")
        print(f"Centers: {centers.shape}, HR dict: {dict_hr.shape}")
    global _CENTERS, _DICT_HR
    _CENTERS, _DICT_HR = centers, dict_hr
    print("\n" + "=" * 55)
    print("METHOD 3: DICTIONARY / PATCH-BASED SR (x4)")
    print("=" * 55)
    test_ds = BSD100SRDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test images: {len(test_ds)}")
    results = evaluate_method(sr_function=dictionary_method, dataloader=test_loader, save_dir=Path("results/method3_dictionary"), max_save=3)
    out_path = Path("results/method3_dictionary/results.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Method 3: Dictionary / Patch-Based SR (x4) - BSD100\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dictionary size: {N_CLUSTERS}\n")
        f.write(f"Avg PSNR: {results['avg_psnr']:.2f} dB\n")
        f.write(f"Avg SSIM: {results['avg_ssim']:.4f}\n")
        f.write(f"Images: {results['num_images']}\n")
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force retrain dictionary")
    args = parser.parse_args()
    main(retrain=args.retrain)