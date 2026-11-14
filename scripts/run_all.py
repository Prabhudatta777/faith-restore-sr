"""
ONE-COMMAND RUNNER - Faith Restore SR (Methods 1-5)

Usage:
    python scripts/run_all.py

    python scripts/run_all.py --skip-train
    python scripts/run_all.py --skip-srcnn --skip-rcan
    python scripts/run_all.py --skip-rcan
    python scripts/run_all.py --srcnn-epochs 5 --rcan-epochs 5
    python scripts/run_all.py --retrain

Pipeline:
    0. Prepare dataset (if missing)
    1. Method 1 - Bicubic
    2. Method 2 - Edge + Segmentation
    3. Method 3 - Dictionary / Patch-Based
    4. Method 4 - SRCNN
    5. Method 5 - RCAN-Lite
    6. Final comparison

IMPORTANT:
    Method 5 filename: method5_rcanlite.py
    All methods run using the current Python interpreter.
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

# Correct module names
SRCNN_MODULE = "src.method4_srcnn"
RCAN_MODULE = "src.method5_rcanlite"

METHOD1_MODULE = "src.method1_bicubic"
METHOD2_MODULE = "src.method2_edge_seg"
METHOD3_MODULE = "src.method3_dictionary"


# ============================================================
# COMMAND EXECUTION
# ============================================================

def run_cmd(cmd, desc):

    print("\n" + "=" * 60)
    print(desc)
    print("Running:", " ".join(cmd))
    print("=" * 60, flush=True)

    result = subprocess.run(
        [sys.executable, *cmd],
        cwd=ROOT
    )

    if result.returncode != 0:

        print("\n" + "!" * 60)
        print(f"ERROR: {desc} failed!")
        print(f"Exit code: {result.returncode}")
        print("Stopping pipeline.")
        print("Fix the error and run again.")
        print("!" * 60)

        sys.exit(result.returncode)

    print(f"\nDone: {desc}", flush=True)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Faith Restore SR - Complete Pipeline (Methods 1-5)"
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip SRCNN and RCAN training; evaluate existing checkpoints"
    )

    parser.add_argument(
        "--skip-srcnn",
        action="store_true",
        help="Skip Method 4 (SRCNN)"
    )

    parser.add_argument(
        "--skip-rcan",
        action="store_true",
        help="Skip Method 5 (RCAN-Lite)"
    )

    parser.add_argument(
        "--srcnn-epochs",
        type=int,
        default=30,
        help="Number of SRCNN training epochs (default: 30)"
    )

    parser.add_argument(
        "--rcan-epochs",
        type=int,
        default=30,
        help="Number of RCAN-Lite training epochs (default: 30)"
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of Method 3 dictionary"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FAITH RESTORE SR - ONE-COMMAND RUNNER")
    print("METHODS 1-5")
    print("=" * 60)

    print(f"\nPython interpreter: {sys.executable}")

    print(f"\nSRCNN epochs: {args.srcnn_epochs}")
    print(f"RCAN epochs: {args.rcan_epochs}")

    print(f"Skip SRCNN: {args.skip_srcnn}")
    print(f"Skip RCAN: {args.skip_rcan}")
    print(f"Skip training: {args.skip_train}")
    print(f"Retrain dictionary: {args.retrain}")

    # --------------------------------------------------------
    # STEP 0: DATASET PREPARATION
    # --------------------------------------------------------

    processed = ROOT / "data" / "processed" / "BSD100_x4"

    if not processed.exists():

        run_cmd(
            ["scripts/prepare_dataset.py"],
            "STEP 0: Prepare Dataset (BSD100)"
        )

    else:

        print(
            "\nSTEP 0: Dataset already prepared. Skipping."
        )

    # --------------------------------------------------------
    # STEP 1: BICUBIC
    # --------------------------------------------------------

    run_cmd(
        ["-m", METHOD1_MODULE],
        "STEP 1: Method 1 - Bicubic Baseline"
    )

    # --------------------------------------------------------
    # STEP 2: EDGE + SEGMENTATION
    # --------------------------------------------------------

    run_cmd(
        ["-m", METHOD2_MODULE],
        "STEP 2: Method 2 - Edge + Segmentation"
    )

    # --------------------------------------------------------
    # STEP 3: DICTIONARY / PATCH-BASED
    # --------------------------------------------------------

    m3_command = ["-m", METHOD3_MODULE]

    if args.retrain:
        m3_command.append("--retrain")

    run_cmd(
        m3_command,
        "STEP 3: Method 3 - Dictionary / Patch-Based"
    )

    # --------------------------------------------------------
    # STEP 4: SRCNN
    # --------------------------------------------------------

    if not args.skip_srcnn:

        if args.skip_train:

            run_cmd(
                ["-m", SRCNN_MODULE, "--eval"],
                "STEP 4: Method 4 - SRCNN (Evaluation Only)"
            )

        else:

            run_cmd(
                [
                    "-m",
                    SRCNN_MODULE,
                    "--train",
                    "--epochs",
                    str(args.srcnn_epochs),
                    "--eval"
                ],
                f"STEP 4: Method 4 - SRCNN "
                f"(Training: {args.srcnn_epochs} epochs)"
            )

    else:

        print("\nSTEP 4: SRCNN skipped.")

    # --------------------------------------------------------
    # STEP 5: RCAN-LITE
    # --------------------------------------------------------

    if not args.skip_rcan:

        if args.skip_train:

            run_cmd(
                ["-m", RCAN_MODULE, "--eval"],
                "STEP 5: Method 5 - RCAN-Lite (Evaluation Only)"
            )

        else:

            run_cmd(
                [
                    "-m",
                    RCAN_MODULE,
                    "--train",
                    "--epochs",
                    str(args.rcan_epochs),
                    "--eval"
                ],
                f"STEP 5: Method 5 - RCAN-Lite "
                f"(Training: {args.rcan_epochs} epochs)"
            )

    else:

        print("\nSTEP 5: RCAN-Lite skipped.")

    # --------------------------------------------------------
    # STEP 6: FINAL COMPARISON
    # --------------------------------------------------------

    run_cmd(
        ["scripts/final_comparison.py"],
        "STEP 6: Final Comparison Table + Charts"
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL METHODS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nCheck the following outputs:")

    print("\n1. Comparison table:")
    print("   results/comparison_table.csv")

    print("\n2. Comparison chart:")
    print("   results/comparison_chart.png")

    print("\n3. Individual method results:")

    print("   results/method1_bicubic/results.txt")
    print("   results/method2_edge_seg/results.txt")
    print("   results/method3_dictionary/results.txt")
    print("   results/method4_srcnn/results.txt")
    print("   results/method5_rcanlite/results.txt")

    print("\n4. Model checkpoints:")
    print("   results/method4_srcnn/srcnn_best.pth")
    print("   results/method5_rcanlite/")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()