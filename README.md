# Faith Restore SR

Faithful and realistic image super-resolution for real-world degraded images.

> **Course Project — Computer Vision**
> Objective: Develop a method for faithful and realistic image super-resolution.
> Problem: Current super-resolution methods fail to recover faithful high-frequency details
> in real-world images with complex textures and unknown degradations.

Instead of building one black-box model, this project **compares 5 methods from the course
curriculum on the same dataset, same degradation, same metrics** — so the comparison is fair.

---

## 1. Dataset — BSD100

| Item | Detail |
|---|---|
| Dataset | BSD100 (Berkeley Segmentation Dataset, test split of BSD300) |
| Total images | 100 |
| Split | Train: 80 / Val: 10 / Test: 10 |
| Scale | x4 (bicubic_4x) |
| Task | Single-Image Super-Resolution (SISR) |

### How GT + Blurred Input is created

BSD100 ships HR images only. We create the paired dataset with a fixed degradation:

```
HR (Ground Truth)
  -> Gaussian Blur, sigma = 1.0   (blurred image)
  -> Bicubic Downsample x4        (Low-Resolution input)
  -> LR (Input to all methods)
```

So every image gives a pair: **LR (blurred + downsampled) + HR (GT)**.

### Split logic

The downloaded zip already contains:
- `bicubic_4x/train/` — 80 images (160 files: 80 HR + 80 LR)
- `bicubic_4x/val/` — 20 images (40 files: 20 HR + 20 LR)

We keep the split deterministic:
- 80 train = keep as train
- 20 val = split by sorted filename into 10 val + 10 test

Seed = 42. Anyone re-running `prepare_dataset.py` gets the exact same split.

### Dataset is NOT in this repo

Download `BSD100.zip` and place it here:

```
data/raw/BSD100.zip
```

Then run:

```bash
python scripts/prepare_dataset.py
```

Output:

```
data/processed/BSD100_x4/
├── train/ (80 images)
├── val/   (10 images)
├── test/  (10 images)
└── splits.json
```

`data/` is gitignored. Do not push images / zip to GitHub.

---

## 2. Methods Implemented — All from Course Curriculum

| # | Method | Curriculum Topic | Learning? |
|---|---|---|---|
| 1 | Bicubic Interpolation | Review of Digital Image Processing | No — baseline |
| 2 | Edge + Segmentation-Guided SR | Edge Detection (Sobel, Canny) + Segmentation (Otsu) + Bilateral / Unsharp Masking | No — hand-crafted rules |
| 3 | Dictionary / Patch-Based SR | Feature Detection (HOG / SIFT idea) + Bag of Words / VLAD (Codebook) + K-Means | Yes — classical learning |
| 4 | CNN-Based SR (SRCNN) | Convolutional Neural Networks + Recent Progress | Yes — deep learning |
| 5 | RCAN-Lite (ResNet + Channel Attention) | ResNet + Attention + CNN | Yes — deep learning, highest PSNR |

### Method 1 — Bicubic Baseline
- Bicubic interpolation x4.
- No learning. Lower bound every other method must beat.
- File: `src/method1_bicubic.py`

### Method 2 — Edge + Segmentation-Guided SR
- Base: Bicubic x4.
- Edge map: Canny (50, 150) + Sobel gradient magnitude, dilated.
- Segmentation: Otsu on gradient → Flat vs Texture. Flat = low gradient AND not edge.
- Combine: Sharpen (unsharp masking) where edge, smooth (bilateral filter) where flat.
- Idea: Do not blur across an edge. Preserve faithful edges.
- File: `src/method2_edge_seg.py`

### Method 3 — Dictionary / Patch-Based SR
- Train: Cut 70 train images into 8x8 LR / 32x32 HR patches. HOG-like feature per LR patch (normalized patch + gradient magnitude = 128-dim). K-Means K=512 → dictionary / codebook. Average HR patch per cluster.
- Test: Each LR test patch → nearest cluster → paste its HR patch. Overlap-average. Colour from bicubic (Y channel replaced).
- Known limitation: K-Means averaging blurs HR patches. 512 atoms cannot cover all textures. Often similar to or worse than bicubic on PSNR. This is an honest, expected result and motivates deep learning.
- File: `src/method3_dictionary.py`

### Method 4 — CNN (SRCNN)
- Full form: **Super-Resolution Convolutional Neural Network**
- Paper: Dong et al., ECCV 2014 — Image Super-Resolution Using Deep Convolutional Networks.
- Architecture:
  - Input: Bicubic-upsampled LR (H*4, W*4, 3)
  - Conv 9x9, 64 + ReLU → Patch Extraction
  - Conv 1x1, 32 + ReLU → Non-linear Mapping
  - Conv 5x5, 3 → Reconstruction
  - Params: 57,187
- Training: MSE loss, Adam lr=1e-3, 70 train images cropped to 64x64 HR / 16x16 LR patches, augment = random flip, 20x repeats per epoch.
- Limitation: Only 3 layers, shallow. Needs more data/epochs to beat bicubic. On 70 images it is close to bicubic.
- File: `src/method4_srcnn.py`

### Method 5 — RCAN-Lite (ResNet + Channel Attention) — NEW, Best PSNR
- Idea: SRCNN is too shallow (3 layers). RCAN uses ResNet + Attention to go deep.
- Based on: EDSR (Lim et al. 2017) + RCAN (Zhang et al. 2018) — both from curriculum topics ResNet + Attention.
- Architecture:
  - Head: Conv 3 -> 64
  - Body: 5 Residual Groups x 5 Residual Blocks = 25 blocks
    - Each Block: Conv-ReLU-Conv + Channel Attention + Skip Connection
    - Channel Attention: AdaptiveAvgPool -> Conv 1x1 -> ReLU -> Conv 1x1 -> Sigmoid, re-weights channels (edge channels get more weight)
  - Tail: Conv 64 -> 3
  - Global Residual: Output = Bicubic Input + Network Output (network only learns detail to ADD)
  - Params: ~1.2M (vs SRCNN 57k)
- Training: MSE loss, Adam lr=1e-4 (smaller LR because deeper), StepLR scheduler (halve LR every 20 epochs, fixes SRCNN plateau), 64x64 HR patches, batch 16.
- Why it beats SRCNN:
  - No averaging like Dictionary, learns continuous mapping
  - Skip connections let it train 25 blocks deep without vanishing gradient
  - Channel Attention focuses on informative channels
  - Global residual makes learning easier
- Limitation: Still MSE loss, so can be slightly blurry. GAN would look sharper but give LOWER PSNR. For PSNR, MSE + ResNet + Attention is best.
- **PSNR note:** PSNR 50 is impossible for x4 SR. Best in world on BSD100 x4 is ~29 dB (HAT Transformer, 2023). RCAN-Lite aims for ~27-28 dB, which beats Bicubic (26.25 dB).
- File: `src/method5_rcan.py`

### NOT Used — and Why
Not used: Epipolar Geometry, Stereo Reconstruction, SfM, Homography / Panorama, Optical Flow / KLT Tracking, Viola-Jones Detector, Projective Geometry.
Reason: Those are for multi-view / video / detection. Faith Restore SR is a single-image restoration problem. Using them here would be forced and wrong.
Linear Algebra / Least Squares / RANSAC appear as background math (e.g. least-squares for dictionary learning).

---

## 3. Metrics

| Metric | Direction | Meaning |
|---|---|---|
| PSNR | Higher is better | Pixel fidelity vs GT. PSNR = 10*log10(MAX^2 / MSE) |
| SSIM | Higher is better (0-1) | Structural similarity vs GT |
| LPIPS | Lower is better | Perceptual similarity (optional, not in baseline code) |
| NIQE | Lower is better | No-reference quality on unpaired real data (optional) |

This repo reports **PSNR + SSIM** on the 10-image test set for all 5 methods with the same loader and same evaluator (`src/utils.py:evaluate_method`), so comparison is fair.

Qualitative: `results/<method>/*_compare.png` shows LR | SR | HR side-by-side. Zoom on 3 hard cases: fine texture (grass / hair), sharp edge (building), flat area (sky). Check for fake / hallucinated texture.

---

## 4. Project Structure

```
faith-restore-sr/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # BSD100.zip here (gitignored)
│   └── processed/            # created by prepare_dataset.py (gitignored)
│       └── BSD100_x4/
│           ├── train/ (80)
│           ├── val/   (10)
│           ├── test/  (10)
│           └── splits.json
├── scripts/
│   ├── prepare_dataset.py    # unzip + split 80/10/10
│   ├── final_comparison.py   # read all results.txt -> table + chart
│   └── run_all.py            # ONE-COMMAND runner (runs Methods 1-5)
├── src/
│   ├── dataset.py            # BSD100SRDataset loader
│   ├── utils.py              # PSNR, SSIM, save_comparison, evaluate_method
│   ├── method1_bicubic.py
│   ├── method2_edge_seg.py
│   ├── method3_dictionary.py
│   ├── method4_srcnn.py
│   └── method5_rcan.py       # NEW: ResNet + Channel Attention
├── results/                  # gitignored, generated by runs
│   ├── method1_bicubic/
│   ├── method2_edge_seg/
│   ├── method3_dictionary/
│   ├── method4_srcnn/
│   ├── method5_rcan/
│   ├── comparison_table.csv
│   └── comparison_chart.png
└── notebooks/
    └── 01_explore_dataset.ipynb
```

---

## 5. How to Run — YES, One Command

### Option A — One Command (Recommended) — NOW INCLUDES METHOD 5

```bash
python scripts/run_all.py
```

This runs, in order:
1. `prepare_dataset.py` (if `data/processed/` does not exist yet)
2. Method 1 — Bicubic
3. Method 2 — Edge + Segmentation
4. Method 3 — Dictionary (trains dictionary first time)
5. Method 4 — SRCNN (trains 30 epochs, then evaluates)
6. **Method 5 — RCAN-Lite (trains 30 epochs, then evaluates) — NEW**
7. `final_comparison.py` (makes `comparison_table.csv` + `comparison_chart.png`)

> **Important:** The OLD `run_all.py` (before this update) did NOT run Method 5.
> You MUST replace it with the new `run_all.py` in this update, otherwise Method 5 will be skipped.
> To check: open `scripts/run_all.py` and search for `method5_rcan`. If it is not there, you have the old version.

Options:

```bash
# Skip training, only evaluate with existing checkpoints / dictionary
python scripts/run_all.py --skip-train

# Only Methods 1-3 (fast, CPU) — skip SRCNN + RCAN
python scripts/run_all.py --skip-srcnn --skip-rcan

# Only Methods 1-4, skip RCAN (RCAN is heavy)
python scripts/run_all.py --skip-rcan

# Quick test — SRCNN + RCAN 5 epochs instead of 30
python scripts/run_all.py --srcnn-epochs 5 --rcan-epochs 5

# Start fresh — retrain dictionary + SRCNN + RCAN
python scripts/run_all.py --retrain
```

> Note: `run_all.py` re-runs everything and **overwrites** `results/<method>/results.txt` each time.
> Only the LAST run is kept. If you want to keep an old run, rename the folder first, e.g.
> `results/method5_rcan` -> `results/method5_rcan_30epochs` before re-running.

### Option B — Step by Step (for debugging / demo)

```bash
# 0. Environment (Conda, recommended)
conda create -n faith-sr python=3.10 -y
conda activate faith-sr
pip install -r requirements.txt

# 1. Dataset (once)
python scripts/prepare_dataset.py

# 2. Methods one by one
python -m src.method1_bicubic
python -m src.method2_edge_seg
python -m src.method3_dictionary
python -m src.method4_srcnn --train --epochs 30 --eval
python -m src.method5_rcan --train --epochs 30 --eval

# 3. Final comparison
python scripts/final_comparison.py
```

### Option C — Colab (for Method 4 + 5 GPU)

Method 4 on CPU takes 15-30 min. Method 5 on CPU takes 1-2 hours. On Colab T4 GPU, Method 5 takes 10-20 min for 30 epochs.

1. Push this repo to GitHub (Private repo, invite only your friends).
2. In Colab: Runtime -> Change runtime type -> T4 GPU.
3. Clone + run:
```python
!git clone https://github.com/YOUR-USERNAME/faith-restore-sr.git
%cd faith-restore-sr
!pip install -r requirements.txt -q
# Upload BSD100.zip to data/raw/ in Colab Files panel
!python scripts/prepare_dataset.py
!python -m src.method4_srcnn --train --epochs 30 --eval
!python -m src.method5_rcan --train --epochs 30 --eval
```
4. Download `results/method5_rcan/` back to your laptop.

---

## 6. Results (Example — Replace with YOUR numbers)

Fill this after running `final_comparison.py`. Example format:

| Method | Avg PSNR (dB) | Avg SSIM |
|---|---|---|
| 1. Bicubic | 26.25 | 0.6509 |
| 2. Edge + Seg | — (fill from results.txt) | — |
| 3. Dictionary | 13.83 | 0.2755 |
| 4. CNN (SRCNN, 5 epochs) | 25.33 | 0.6182 |
| 5. RCAN-Lite (30 epochs) | ~27.0 (expected) | ~0.70 (expected) |

> Expected story: Bicubic is a strong baseline. Edge+Seg improves sharpness slightly.
> Dictionary fails because K-Means averaging blurs details. SRCNN with limited training
> is close to Bicubic — deep learning needs more epochs / data to surpass
> classical methods. RCAN-Lite with ResNet + Attention beats all previous methods,
> showing depth + attention + residual learning is the key. With 100 epochs + more data,
> RCAN-Lite should reach ~27.5-28 dB. World SOTA on BSD100 x4 is ~29 dB, PSNR 50 is impossible.
> This is the core Faith Restore SR challenge: faithfulness vs hallucination.

Chart: `results/comparison_chart.png` (generated by `final_comparison.py`).

---

## 7. Environment

Conda (recommended, avoids PowerShell execution-policy issues):

```bash
conda create -n faith-sr python=3.10 -y
conda activate faith-sr
pip install -r requirements.txt
```

If `conda activate` says `Run conda init before conda activate` in VS Code PowerShell:
- Close VS Code completely, open Anaconda Prompt, run `conda init powershell`, reopen VS Code, OR
- Use `conda run -n faith-sr python -m src.method1_bicubic` (bypasses activation).

`.venv/` and `data/` and `results/` and `*.pth` are gitignored.

---

## 8. Reproducibility

- Seed = 42 everywhere (`src/utils.py:set_seed`).
- Deterministic split by sorted filename.
- HR cropped to multiple of 4 in `src/dataset.py`, so HR = LR x 4 exactly (fixes 321x481 vs 320x480 mismatch).
- PSNR / SSIM in `src/utils.py` crop to common size, so odd sizes never crash.
- All methods use the same `BSD100SRDataset` loader and same `evaluate_method()`.

## 9. References

- BSD100 — Martin et al., Berkeley Segmentation Dataset.
- SRCNN — Dong et al., ECCV 2014.
- EDSR — Lim et al., CVPR 2017.
- RCAN — Zhang et al., ECCV 2018.
- SwinIR — Liang et al., ICCV 2021.
- RealSR / DRealSR — mentioned in project brief for real-world evaluation (future work).
