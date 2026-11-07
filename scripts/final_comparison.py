"""
Final Comparison - Faith Restore SR
Reads all 4 results.txt files and makes:
  - comparison_table.csv
  - comparison_chart.png (bar chart PSNR + SSIM)
Usage: python scripts/final_comparison.py
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

METHODS = [
    ("1. Bicubic", "method1_bicubic"),
    ("2. Edge + Seg", "method2_edge_seg"),
    ("3. Dictionary", "method3_dictionary"),
    ("4. CNN (SRCNN)", "method4_srcnn"),
]

def parse_results(folder):
    p = RESULTS / folder / "results.txt"
    if not p.exists():
        return None, None
    text = p.read_text()
    psnr = re.search(r"PSNR:\s*([0-9.]+)", text)
    ssim = re.search(r"SSIM:\s*([0-9.]+)", text)
    return (float(psnr.group(1)) if psnr else None,
            float(ssim.group(1)) if ssim else None)

def main():
    rows = []
    print("=" * 55)
    print("FAITH RESTORE SR - FINAL COMPARISON (BSD100 x4, 10 test images)")
    print("=" * 55)
    print(f"{'Method':<20} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 55)
    for name, folder in METHODS:
        psnr, ssim = parse_results(folder)
        if psnr is None:
            print(f"{name:<20} {'MISSING':<12} {'MISSING':<10}")
            continue
        rows.append((name, psnr, ssim))
        print(f"{name:<20} {psnr:<12.2f} {ssim:<10.4f}")
    print("-" * 55)
    if rows:
        best_psnr = max(rows, key=lambda x: x[1])
        best_ssim = max(rows, key=lambda x: x[2])
        print(f"Best PSNR: {best_psnr[0]} ({best_psnr[1]:.2f} dB)")
        print(f"Best SSIM: {best_ssim[0]} ({best_ssim[1]:.4f})")
        # Save CSV
        out_csv = RESULTS / "comparison_table.csv"
        with open(out_csv, "w") as f:
            f.write("Method,PSNR_dB,SSIM\n")
            for n, p, s in rows:
                f.write(f"{n},{p:.2f},{s:.4f}\n")
        print(f"\nSaved: {out_csv}")
        # Chart
        names = [r[0] for r in rows]
        psnrs = [r[1] for r in rows]
        ssims = [r[2] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        colors = ["#7f7f7f", "#4e79a7", "#e15759", "#59a14f"]
        ax1.bar(names, psnrs, color=colors[:len(names)])
        ax1.set_title("PSNR (higher = better)")
        ax1.set_ylabel("dB")
        ax1.tick_params(axis="x", rotation=15)
        for i, v in enumerate(psnrs):
            ax1.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=9)
        ax2.bar(names, ssims, color=colors[:len(names)])
        ax2.set_title("SSIM (higher = better)")
        ax2.set_ylabel("SSIM")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="x", rotation=15)
        for i, v in enumerate(ssims):
            ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        out_png = RESULTS / "comparison_chart.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_png}")
        print("\nUse comparison_chart.png in your report!")

if __name__ == "__main__":
    main()
