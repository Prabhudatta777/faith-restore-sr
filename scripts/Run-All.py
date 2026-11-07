"""
ONE-COMMAND RUNNER - Faith Restore SR
Usage:
  python scripts/run_all.py
  python scripts/run_all.py --skip-train
  python scripts/run_all.py --skip-srcnn
  python scripts/run_all.py --srcnn-epochs 5
  python scripts/run_all.py --retrain

What it does, in order:
  1. prepare_dataset.py (if processed data missing)
  2. Method 1 - Bicubic
  3. Method 2 - Edge + Segmentation
  4. Method 3 - Dictionary
  5. Method 4 - SRCNN (train + eval)
  6. final_comparison.py

WARNING: Each method OVERWRITES its results/<method>/results.txt.
Only the LAST run is kept. Rename old results folder first if you want to keep it.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_cmd(cmd, desc):
    print("\n" + "=" * 60)
    print(desc)
    print("Running:", " ".join(cmd))
    print("=" * 60)
    result = subprocess.run([sys.executable, *cmd], cwd=ROOT)
    if result.returncode != 0:
        print(f"\nERROR: {desc} failed with code {result.returncode}")
        print("Stopping run_all. Fix the error above and re-run.")
        sys.exit(result.returncode)
    print(f"Done: {desc}")

def main():
    parser = argparse.ArgumentParser(description="Run entire Faith Restore SR pipeline with one command")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only evaluate (uses existing dictionary / checkpoint)")
    parser.add_argument("--skip-srcnn", action="store_true", help="Skip Method 4 SRCNN (fast CPU run: Methods 1-3 only)")
    parser.add_argument("--srcnn-epochs", type=int, default=30, help="Epochs for SRCNN training (default 30, use 5 for quick test)")
    parser.add_argument("--retrain", action="store_true", help="Force retrain dictionary (Method 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("FAITH RESTORE SR - ONE-COMMAND RUNNER")
    print("=" * 60)
    print(f"SRCNN epochs: {args.srcnn_epochs} | Skip SRCNN: {args.skip_srcnn} | Skip train: {args.skip_train} | Retrain dict: {args.retrain}")

    # 1. Dataset
    processed = ROOT / "data" / "processed" / "BSD100_x4"
    if not processed.exists():
        run_cmd(["scripts/prepare_dataset.py"], "STEP 0: Prepare Dataset (BSD100 80/10/10)")
    else:
        print("\nSTEP 0: Dataset already prepared, skipping. Delete data/processed to re-prepare.")

    # 2. Method 1
    run_cmd(["-m", "src.method1_bicubic"], "STEP 1: Method 1 - Bicubic Baseline")

    # 3. Method 2
    run_cmd(["-m", "src.method2_edge_seg"], "STEP 2: Method 2 - Edge + Segmentation")

    # 4. Method 3
    m3 = ["-m", "src.method3_dictionary"]
    if args.retrain:
        m3.append("--retrain")
    run_cmd(m3, "STEP 3: Method 3 - Dictionary / Patch-Based")

    # 5. Method 4
    if not args.skip_srcnn:
        if args.skip_train:
            run_cmd(["-m", "src.method4_srcnn", "--eval"], "STEP 4: Method 4 - SRCNN (Eval only, no training)")
        else:
            run_cmd(["-m", "src.method4_srcnn", "--train", "--epochs", str(args.srcnn_epochs), "--eval"],
                    f"STEP 4: Method 4 - SRCNN (Train {args.srcnn_epochs} epochs + Eval)")
    else:
        print("\nSTEP 4: Skipped SRCNN (--skip-srcnn)")

    # 6. Final comparison
    run_cmd(["scripts/final_comparison.py"], "STEP 5: Final Comparison Table + Chart")

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("Check:")
    print("  results/comparison_table.csv")
    print("  results/comparison_chart.png")
    print("  results/<method>/results.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
