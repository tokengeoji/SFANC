
"""
Analyze SFANC label quality after NR-margin analysis.

Inputs:
  noise_dataset/My_Dataset_16k/training_labels.csv
  noise_dataset/My_Dataset_16k/dataset_metadata.csv

Outputs:
  1) Console summary
  2) label_absolute_nr_summary.csv
  3) label_frequency_summary.csv
  4) label_nr_frequency_merged.csv

This script checks:
- Absolute best NR by label
- Fraction of samples with best NR < 0 dB, < 1 dB, >= 3 dB
- low/high/bandwidth distribution by label
- Detailed W6 frequency distribution
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "noise_dataset" / "My_Dataset_16k"

LABEL_CSV = DATA_DIR / "training_labels.csv"
META_CSV = DATA_DIR / "dataset_metadata.csv"

OUT_NR = DATA_DIR / "label_absolute_nr_summary.csv"
OUT_FREQ = DATA_DIR / "label_frequency_summary.csv"
OUT_MERGED = DATA_DIR / "label_nr_frequency_merged.csv"

LABEL_ORDER = ["W1", "W2", "W3", "W4", "W5", "W6"]


def pct(mask):
    n = len(mask)
    return 100.0 * float(np.sum(mask)) / n if n else 0.0


def main():
    labels = pd.read_csv(LABEL_CSV)
    meta = pd.read_csv(META_CSV)

    required_label_cols = {"filename", "label", "best_nr_db"}
    required_meta_cols = {"filename", "low_hz", "high_hz"}

    if not required_label_cols.issubset(labels.columns):
        raise RuntimeError(
            f"training_labels.csv missing columns: "
            f"{sorted(required_label_cols - set(labels.columns))}"
        )

    if not required_meta_cols.issubset(meta.columns):
        raise RuntimeError(
            f"dataset_metadata.csv missing columns: "
            f"{sorted(required_meta_cols - set(meta.columns))}"
        )

    df = labels.merge(meta, on="filename", how="inner")

    if len(df) != len(labels):
        print(
            f"WARNING: merged {len(df)} rows from {len(labels)} labels. "
            "Check filename consistency."
        )

    df["bandwidth_hz"] = df["high_hz"] - df["low_hz"]
    df.to_csv(OUT_MERGED, index=False, encoding="utf-8-sig")

    print()
    print("================ Absolute NR by label ================")

    nr_rows = []

    for label in LABEL_ORDER:
        sub = df[df["label"] == label]
        if sub.empty:
            continue

        nr = sub["best_nr_db"].to_numpy(dtype=np.float64)

        row = {
            "label": label,
            "count": len(sub),
            "mean_best_nr_db": float(np.mean(nr)),
            "median_best_nr_db": float(np.median(nr)),
            "min_best_nr_db": float(np.min(nr)),
            "max_best_nr_db": float(np.max(nr)),
            "pct_nr_lt_0db": pct(nr < 0.0),
            "pct_nr_lt_1db": pct(nr < 1.0),
            "pct_nr_ge_3db": pct(nr >= 3.0),
        }
        nr_rows.append(row)

        print(
            f"{label}: n={len(sub):6d} | "
            f"mean={row['mean_best_nr_db']:7.3f} dB | "
            f"median={row['median_best_nr_db']:7.3f} dB | "
            f"<0dB={row['pct_nr_lt_0db']:6.2f}% | "
            f"<1dB={row['pct_nr_lt_1db']:6.2f}% | "
            f">=3dB={row['pct_nr_ge_3db']:6.2f}%"
        )

    pd.DataFrame(nr_rows).to_csv(OUT_NR, index=False, encoding="utf-8-sig")

    print()
    print("================ Frequency range by label ================")

    freq_rows = []

    for label in LABEL_ORDER:
        sub = df[df["label"] == label]
        if sub.empty:
            continue

        row = {
            "label": label,
            "count": len(sub),

            "low_mean_hz": float(sub["low_hz"].mean()),
            "low_median_hz": float(sub["low_hz"].median()),
            "low_min_hz": int(sub["low_hz"].min()),
            "low_max_hz": int(sub["low_hz"].max()),

            "high_mean_hz": float(sub["high_hz"].mean()),
            "high_median_hz": float(sub["high_hz"].median()),
            "high_min_hz": int(sub["high_hz"].min()),
            "high_max_hz": int(sub["high_hz"].max()),

            "bandwidth_mean_hz": float(sub["bandwidth_hz"].mean()),
            "bandwidth_median_hz": float(sub["bandwidth_hz"].median()),
        }
        freq_rows.append(row)

        print(
            f"{label}: n={len(sub):6d} | "
            f"low mean/med={row['low_mean_hz']:7.1f}/{row['low_median_hz']:7.1f} Hz | "
            f"high mean/med={row['high_mean_hz']:7.1f}/{row['high_median_hz']:7.1f} Hz | "
            f"BW mean/med={row['bandwidth_mean_hz']:7.1f}/{row['bandwidth_median_hz']:7.1f} Hz"
        )

    pd.DataFrame(freq_rows).to_csv(OUT_FREQ, index=False, encoding="utf-8-sig")

    # W6 detail
    w6 = df[df["label"] == "W6"].copy()

    print()
    print("================ W6 detailed check ================")

    if len(w6):
        print(f"W6 count: {len(w6)}")
        print(f"low_hz  mean   : {w6['low_hz'].mean():.2f}")
        print(f"low_hz  median : {w6['low_hz'].median():.2f}")
        print(f"high_hz mean   : {w6['high_hz'].mean():.2f}")
        print(f"high_hz median : {w6['high_hz'].median():.2f}")
        print(f"BW      mean   : {w6['bandwidth_hz'].mean():.2f}")
        print(f"BW      median : {w6['bandwidth_hz'].median():.2f}")
        print()
        print(f"W6 low >= 1200 Hz : {pct(w6['low_hz'].to_numpy() >= 1200):6.2f}%")
        print(f"W6 low >= 1000 Hz : {pct(w6['low_hz'].to_numpy() >= 1000):6.2f}%")
        print(f"W6 high >= 1600 Hz: {pct(w6['high_hz'].to_numpy() >= 1600):6.2f}%")
        print(f"W6 high >= 1800 Hz: {pct(w6['high_hz'].to_numpy() >= 1800):6.2f}%")
        print(f"W6 best NR < 0 dB : {pct(w6['best_nr_db'].to_numpy() < 0.0):6.2f}%")
        print(f"W6 best NR < 1 dB : {pct(w6['best_nr_db'].to_numpy() < 1.0):6.2f}%")
        print(f"W6 best NR >= 3 dB: {pct(w6['best_nr_db'].to_numpy() >= 3.0):6.2f}%")

    print()
    print(f"Saved: {OUT_NR}")
    print(f"Saved: {OUT_FREQ}")
    print(f"Saved: {OUT_MERGED}")
    print("===================================================")


if __name__ == "__main__":
    main()
