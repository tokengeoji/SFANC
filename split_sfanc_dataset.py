
"""
Create stratified Train / Validation / Test splits for the augmented 16 kHz SFANC dataset.

Input:
    noise_dataset/My_Dataset_16k/training_labels_augmented.csv

Output:
    train_labels.csv
    val_labels.csv
    test_labels.csv

Split:
    Train 80%
    Validation 10%
    Test 10%

The split is stratified by label so W1~W6 proportions are preserved.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "noise_dataset" / "My_Dataset_16k"

INPUT_CSV = DATA_DIR / "training_labels_augmented.csv"

TRAIN_CSV = DATA_DIR / "train_labels.csv"
VAL_CSV = DATA_DIR / "val_labels.csv"
TEST_CSV = DATA_DIR / "test_labels.csv"

RANDOM_STATE = 42


def print_distribution(df, title):
    print()
    print(title)
    print("=" * len(title))

    total = len(df)
    counts = df["label"].value_counts().sort_index()

    for label in ["W1", "W2", "W3", "W4", "W5", "W6"]:
        c = int(counts.get(label, 0))
        pct = 100.0 * c / total if total else 0.0
        print(f"{label}: {c:6d} ({pct:6.2f}%)")

    print(f"TOTAL: {total:6d}")


def main():
    df = pd.read_csv(INPUT_CSV)

    required = {"filename", "label"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    # 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Split temp equally -> 10% validation, 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print_distribution(train_df, "Train distribution")
    print_distribution(val_df, "Validation distribution")
    print_distribution(test_df, "Test distribution")

    print()
    print(f"Saved: {TRAIN_CSV}")
    print(f"Saved: {VAL_CSV}")
    print(f"Saved: {TEST_CSV}")


if __name__ == "__main__":
    main()
