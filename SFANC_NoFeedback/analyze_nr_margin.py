
"""
Analyze NR margin from SFANC training_labels.csv.

NR margin = best NR - second-best NR

Outputs:
1) Console summary
2) nr_margin_analysis.csv      : all samples + best/second/margin
3) ambiguous_margin_lt_0p2.csv : samples with margin < 0.2 dB
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "noise_dataset" / "My_Dataset_16k" / "training_labels.csv"

OUT_ALL = CSV_PATH.parent / "nr_margin_analysis.csv"
OUT_AMBIG = CSV_PATH.parent / "ambiguous_margin_lt_0p2.csv"

NR_COLS = [
    "nr_W1_db",
    "nr_W2_db",
    "nr_W3_db",
    "nr_W4_db",
    "nr_W5_db",
    "nr_W6_db",
]

THRESHOLDS = [0.1, 0.2, 0.5, 1.0]


def main():
    df = pd.read_csv(CSV_PATH)

    missing = [c for c in NR_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing NR columns: {missing}")

    nr = df[NR_COLS].to_numpy(dtype=np.float64)

    # Sort each row descending: last = best, second-last = second best
    order = np.argsort(nr, axis=1)
    best_idx = order[:, -1]
    second_idx = order[:, -2]

    row_idx = np.arange(len(df))
    best_nr = nr[row_idx, best_idx]
    second_nr = nr[row_idx, second_idx]
    margin = best_nr - second_nr

    best_name = np.array([f"W{i+1}" for i in best_idx], dtype=object)
    second_name = np.array([f"W{i+1}" for i in second_idx], dtype=object)

    result = df.copy()
    result["best_filter_check"] = best_name
    result["second_filter"] = second_name
    result["second_nr_db"] = second_nr
    result["nr_margin_db"] = margin

    result.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

    ambig = result[result["nr_margin_db"] < 0.2].copy()
    ambig.to_csv(OUT_AMBIG, index=False, encoding="utf-8-sig")

    print()
    print("=============== NR Margin Summary ===============")
    print(f"Samples      : {len(result)}")
    print(f"Mean margin  : {margin.mean():.4f} dB")
    print(f"Median margin: {np.median(margin):.4f} dB")
    print(f"Min margin   : {margin.min():.4f} dB")
    print(f"Max margin   : {margin.max():.4f} dB")
    print()

    for th in THRESHOLDS:
        count = int(np.sum(margin < th))
        pct = 100.0 * count / len(margin)
        print(f"margin < {th:>3.1f} dB : {count:6d} ({pct:6.2f}%)")

    print()
    print("============= Per-label Margin =================")
    for label in ["W1", "W2", "W3", "W4", "W5", "W6"]:
        sub = result[result["label"] == label]["nr_margin_db"]
        if len(sub) == 0:
            continue

        ambig_count = int((sub < 0.2).sum())
        ambig_pct = 100.0 * ambig_count / len(sub)

        print(
            f"{label}: n={len(sub):6d} | "
            f"mean={sub.mean():7.4f} dB | "
            f"median={sub.median():7.4f} dB | "
            f"<0.2 dB={ambig_count:6d} ({ambig_pct:6.2f}%)"
        )

    print()
    print("============= Common 1st vs 2nd =================")
    pair_counts = (
        result.groupby(["label", "second_filter"])
        .size()
        .sort_values(ascending=False)
        .head(15)
    )
    for (best, second), count in pair_counts.items():
        print(f"{best} vs {second}: {count}")

    print()
    print(f"Saved all analysis : {OUT_ALL}")
    print(f"Saved ambiguous    : {OUT_AMBIG}")
    print("=================================================")


if __name__ == "__main__":
    main()
