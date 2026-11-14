"""
Final Comparison - Faith Restore SR

Compares all FIVE super-resolution methods.

Generates:
    results/comparison_table.csv
    results/comparison_chart.png

Usage:
    python scripts/final_comparison.py
"""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"


# ALL FIVE METHODS

METHODS = [
    ("1. Bicubic", "method1_bicubic"),
    ("2. Edge + Seg", "method2_edge_seg"),
    ("3. Dictionary", "method3_dictionary"),
    ("4. CNN (SRCNN)", "method4_srcnn"),
    ("5. RCAN-Lite", "method5_rcan"),
]


# Five distinct colours

COLORS = [
    "#7f7f7f",
    "#4e79a7",
    "#e15759",
    "#59a14f",
    "#f28e2b",
]


# ============================================================
# PARSE RESULTS
# ============================================================

def parse_results(folder):

    result_file = RESULTS / folder / "results.txt"

    if not result_file.exists():

        print(f"WARNING: Missing results file: {result_file}")

        return None, None

    text = result_file.read_text()

    psnr_match = re.search(
        r"PSNR:\s*([0-9.]+)",
        text
    )

    ssim_match = re.search(
        r"SSIM:\s*([0-9.]+)",
        text
    )

    psnr = (
        float(psnr_match.group(1))
        if psnr_match else None
    )

    ssim = (
        float(ssim_match.group(1))
        if ssim_match else None
    )

    return psnr, ssim


# ============================================================
# MAIN
# ============================================================

def main():

    rows = []

    print("\n" + "=" * 65)

    print("FAITH RESTORE SR - FINAL COMPARISON")

    print("BSD100 x4 | Five Super-Resolution Methods")

    print("=" * 65)

    print(
        f"\n{'Method':<22} "
        f"{'PSNR (dB)':<14} "
        f"{'SSIM':<12}"
    )

    print("-" * 65)


    # --------------------------------------------------------
    # READ ALL FIVE METHODS
    # --------------------------------------------------------

    for name, folder in METHODS:

        psnr, ssim = parse_results(folder)

        if psnr is None or ssim is None:

            print(
                f"{name:<22} "
                f"{'MISSING':<14} "
                f"{'MISSING':<12}"
            )

            continue

        rows.append((name, psnr, ssim))

        print(
            f"{name:<22} "
            f"{psnr:<14.2f} "
            f"{ssim:<12.4f}"
        )


    print("-" * 65)


    if not rows:

        print("No results found!")

        return


    # --------------------------------------------------------
    # BEST PERFORMING METHODS
    # --------------------------------------------------------

    best_psnr = max(rows, key=lambda x: x[1])

    best_ssim = max(rows, key=lambda x: x[2])

    print(
        f"\nBest PSNR: {best_psnr[0]} "
        f"({best_psnr[1]:.2f} dB)"
    )

    print(
        f"Best SSIM: {best_ssim[0]} "
        f"({best_ssim[2]:.4f})"
    )


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    out_csv = RESULTS / "comparison_table.csv"

    with open(out_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Method",
            "PSNR_dB",
            "SSIM"
        ])

        for name, psnr, ssim in rows:

            writer.writerow([
                name,
                f"{psnr:.2f}",
                f"{ssim:.4f}"
            ])


    print(f"\nCSV saved: {out_csv}")


    # --------------------------------------------------------
    # GENERATE COMPARISON CHART
    # --------------------------------------------------------

    names = [r[0] for r in rows]

    psnrs = [r[1] for r in rows]

    ssims = [r[2] for r in rows]

    # Use corresponding colours even if a method is missing

    method_colors = [
        COLORS[
            next(
                i for i, method in enumerate(METHODS)
                if method[0] == name
            )
        ]
        for name in names
    ]


    fig, (ax1, ax2) = plt.subplots(

        1,

        2,

        figsize=(15, 5)

    )


    # --------------------------------------------------------
    # PSNR CHART
    # --------------------------------------------------------

    bars1 = ax1.bar(

        names,

        psnrs,

        color=method_colors,

        width=0.65

    )

    ax1.set_title(

        "PSNR Comparison (Higher = Better)",

        fontsize=13,

        fontweight="bold"

    )

    ax1.set_ylabel("PSNR (dB)")

    ax1.set_xlabel("Super-Resolution Method")

    ax1.tick_params(

        axis="x",

        rotation=20

    )

    ax1.grid(

        axis="y",

        linestyle="--",

        alpha=0.3

    )

    for bar, value in zip(bars1, psnrs):

        ax1.text(

            bar.get_x() + bar.get_width() / 2,

            bar.get_height() + 0.3,

            f"{value:.2f}",

            ha="center",

            fontsize=9,

            fontweight="bold"

        )


    # --------------------------------------------------------
    # SSIM CHART
    # --------------------------------------------------------

    bars2 = ax2.bar(

        names,

        ssims,

        color=method_colors,

        width=0.65

    )

    ax2.set_title(

        "SSIM Comparison (Higher = Better)",

        fontsize=13,

        fontweight="bold"

    )

    ax2.set_ylabel("SSIM")

    ax2.set_xlabel("Super-Resolution Method")

    ax2.set_ylim(0, 1.1)

    ax2.tick_params(

        axis="x",

        rotation=20

    )

    ax2.grid(

        axis="y",

        linestyle="--",

        alpha=0.3

    )

    for bar, value in zip(bars2, ssims):

        ax2.text(

            bar.get_x() + bar.get_width() / 2,

            bar.get_height() + 0.025,

            f"{value:.4f}",

            ha="center",

            fontsize=9,

            fontweight="bold"

        )


    # --------------------------------------------------------
    # FINALIZE
    # --------------------------------------------------------

    fig.suptitle(

        "Faith Restore SR - BSD100 x4 Performance Comparison",

        fontsize=15,

        fontweight="bold"

    )

    plt.tight_layout()


    out_png = RESULTS / "comparison_chart.png"

    plt.savefig(

        out_png,

        dpi=200,

        bbox_inches="tight"

    )

    plt.close(fig)


    print(f"\nChart saved: {out_png}")


    print("\n" + "=" * 65)

    print("COMPARISON COMPLETED SUCCESSFULLY")

    print(f"Methods included: {len(rows)}/5")

    print("=" * 65)


if __name__ == "__main__":

    main()  