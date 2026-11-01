import argparse
import csv
from pathlib import Path

import numpy as np
from numba import njit
from scipy.io import loadmat, savemat
from scipy.signal import firwin, lfilter
from scipy.optimize import linear_sum_assignment


# ============================================================
# Configuration
# ============================================================

FS = 16000
W_LEN = 1024

# 저자 논문: 15개의 broadband frequency ranges.
#
# 네 단계의 계층적 bandwidth:
# 1 + 2 + 4 + 8 = 15 filters
#
# 7980 Hz는 Nyquist(8000 Hz)를 직접 사용하지 않기 위한 상한입니다.
BANDS = [
    (20, 7980),

    (20, 4000),
    (4000, 7980),

    (20, 2000),
    (2000, 4000),
    (4000, 6000),
    (6000, 7980),

    (20, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000),
    (5000, 6000),
    (6000, 7000),
    (7000, 7980),
]

# 저자 pre-training의 정확한 값은 공개되지 않음.
# 사용자 측정 P/S에서 보수적으로 시작하는 값.
MU = 0.0005
TRAIN_SECONDS = 5
VALID_SECONDS = 5

# 기존 저자 필터와 frequency-band 대응 관계 확인용
MAP_SECONDS = 5
MAP_REPEATS = 2


ROOT = Path(__file__).resolve().parent

PATH_DIR = ROOT / "Primary and Secondary Path"
MODEL_DIR = ROOT / "Trained models"

NEW_P_FILE = PATH_DIR / "Primary_path.mat"
NEW_S_FILE = PATH_DIR / "Secondary_path.mat"

OLD_P_FILE = PATH_DIR / "Primary_path_original.mat"
OLD_S_FILE = PATH_DIR / "Secondary_path_original.mat"

OLD_W_FILE = MODEL_DIR / "Pretrained_Control_filters_original.mat"

OUTPUT_W_FILE = MODEL_DIR / "Pretrained_Control_filters_myduct.mat"
PARTIAL_W_FILE = MODEL_DIR / "Pretrained_Control_filters_myduct_partial.mat"
REPORT_FILE = MODEL_DIR / "Pretrained_Control_filters_myduct_report.csv"
MAPPING_FILE = MODEL_DIR / "Pretrained_Control_filters_myduct_mapping.txt"


# ============================================================
# Loading
# ============================================================

def load_path(path, key):
    data = loadmat(path)

    if key not in data:
        raise RuntimeError(
            f"{path} 안에 '{key}' 변수가 없습니다."
        )

    return np.asarray(data[key], dtype=np.float64).squeeze()


def load_bank(path):
    data = loadmat(path)

    if "Wc_v" not in data:
        raise RuntimeError(
            f"{path} 안에 Wc_v 변수가 없습니다."
        )

    W = np.asarray(data["Wc_v"], dtype=np.float64)

    if W.shape != (15, 1024):
        raise RuntimeError(
            f"원본 Wc_v shape이 {W.shape}입니다. "
            f"예상값은 (15, 1024)입니다."
        )

    return W


# ============================================================
# Broadband training noise
# ============================================================

def make_band_noise(low, high, seconds, seed):
    rng = np.random.default_rng(seed)

    noise_filter = firwin(
        1024,
        [low, high],
        pass_zero="bandpass",
        window="hamming",
        fs=FS,
    )

    N = int(seconds * FS)

    white = rng.standard_normal(N + 1024)

    x = lfilter(
        noise_filter,
        [1.0],
        white,
    )[1024:]

    std = np.std(x)

    if std < 1e-12:
        raise RuntimeError(
            f"Noise generation failed: {low}-{high} Hz"
        )

    x = x / std

    return np.asarray(x, dtype=np.float64)


# ============================================================
# Fixed-filter validation
# ============================================================

def evaluate_filter(w, x, P, S):
    d = lfilter(P, [1.0], x)

    control = lfilter(w, [1.0], x)

    anti = lfilter(S, [1.0], control)

    e = d - anti

    skip = min(
        4096,
        len(x) // 4
    )

    d2 = d[skip:]
    e2 = e[skip:]

    pd = np.mean(d2 * d2) + 1e-30
    pe = np.mean(e2 * e2) + 1e-30

    nr_db = 10.0 * np.log10(pd / pe)

    return nr_db


# ============================================================
# FxLMS
# ============================================================

@njit
def train_fxlms(
    x,
    d,
    xf,
    S,
    w,
    mu,
):
    M = len(w)
    LS = len(S)
    N = len(x)

    x_buffer = np.zeros(M)
    xf_buffer = np.zeros(M)
    y_buffer = np.zeros(LS)

    x_index = 0
    y_index = 0

    error = np.zeros(N)

    for n in range(N):

        # ---------------------------------------------
        # Reference delay lines
        # ---------------------------------------------
        x_index -= 1

        if x_index < 0:
            x_index = M - 1

        x_buffer[x_index] = x[n]
        xf_buffer[x_index] = xf[n]

        # ---------------------------------------------
        # y(n) = W^T x(n)
        # ---------------------------------------------
        y = 0.0

        for k in range(M):
            j = x_index + k

            if j >= M:
                j -= M

            y += w[k] * x_buffer[j]

        # ---------------------------------------------
        # Secondary path
        # ---------------------------------------------
        y_index -= 1

        if y_index < 0:
            y_index = LS - 1

        y_buffer[y_index] = y

        anti = 0.0

        for k in range(LS):
            j = y_index + k

            if j >= LS:
                j -= LS

            anti += S[k] * y_buffer[j]

        # ---------------------------------------------
        # Error
        #
        # e(n) = d(n) - anti(n)
        # ---------------------------------------------
        e = d[n] - anti

        error[n] = e

        # ---------------------------------------------
        # FxLMS coefficient update
        #
        # W(n+1) = W(n) + mu * e(n) * x_f(n)
        # ---------------------------------------------
        for k in range(M):
            j = x_index + k

            if j >= M:
                j -= M

            w[k] += mu * e * xf_buffer[j]

    return w, error


# ============================================================
# Original author bank -> frequency-band mapping
# ============================================================

def determine_original_mapping():
    required = [
        OLD_P_FILE,
        OLD_S_FILE,
        OLD_W_FILE,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        print()
        print("원본 파일이 부족합니다.")

        for p in missing:
            print("  missing:", p)

        raise RuntimeError(
            "원본 P/S/W가 있어야 model.pth의 "
            "filter index 순서를 보존할 수 있습니다."
        )

    P_old = load_path(
        OLD_P_FILE,
        "Pz1"
    )

    S_old = load_path(
        OLD_S_FILE,
        "S"
    )

    W_old = load_bank(
        OLD_W_FILE
    )

    print()
    print("============================================")
    print("Original author data")
    print("============================================")
    print("P original:", P_old.shape)
    print("S original:", S_old.shape)
    print("W original:", W_old.shape)
    print()

    scores = np.zeros((15, 15))

    for band_index, (low, high) in enumerate(BANDS):

        score_sum = np.zeros(15)

        for repeat in range(MAP_REPEATS):

            x = make_band_noise(
                low,
                high,
                MAP_SECONDS,
                seed=10000 + band_index * 100 + repeat,
            )

            for filter_index in range(15):

                score_sum[filter_index] += evaluate_filter(
                    W_old[filter_index],
                    x,
                    P_old,
                    S_old,
                )

        scores[band_index] = (
            score_sum / MAP_REPEATS
        )

    # 한 filter가 한 band에 대응하도록 최적 one-to-one assignment
    band_rows, filter_cols = linear_sum_assignment(
        -scores
    )

    filter_to_band = np.full(
        15,
        -1,
        dtype=np.int64,
    )

    for band_index, filter_index in zip(
        band_rows,
        filter_cols,
    ):
        filter_to_band[filter_index] = band_index

    print("============================================")
    print("Author W index -> training frequency band")
    print("============================================")

    lines = []

    for filter_index in range(15):

        band_index = int(
            filter_to_band[filter_index]
        )

        low, high = BANDS[band_index]

        nr = scores[
            band_index,
            filter_index
        ]

        line = (
            f"W{filter_index + 1:02d} "
            f"-> {low:4d}-{high:4d} Hz "
            f"(original NR={nr:7.2f} dB)"
        )

        lines.append(line)
        print(line)

    MAPPING_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(
        "Mapping saved:",
        MAPPING_FILE
    )

    return filter_to_band


# ============================================================
# My duct training
# ============================================================

def train_myduct(filter_to_band):
    P = load_path(
        NEW_P_FILE,
        "Pz1"
    )

    S = load_path(
        NEW_S_FILE,
        "S"
    )

    print()
    print("============================================")
    print("My duct")
    print("============================================")
    print("P:", P.shape)
    print("S:", S.shape)
    print("P max abs:", np.max(np.abs(P)))
    print("S max abs:", np.max(np.abs(S)))
    print()
    print("Control filters : 15")
    print("W length        :", W_LEN)
    print("Sampling rate   :", FS)
    print("FxLMS mu        :", MU)
    print("Training seconds:", TRAIN_SECONDS)
    print()

    Wc_v = np.zeros(
        (15, W_LEN),
        dtype=np.float64,
    )

    report = []

    for filter_index in range(15):

        band_index = int(
            filter_to_band[filter_index]
        )

        low, high = BANDS[band_index]

        print()
        print("--------------------------------------------")
        print(
            f"Training W{filter_index + 1:02d}/15 "
            f": {low}-{high} Hz"
        )
        print("--------------------------------------------")

        # -----------------------------------------
        # Training noise
        # -----------------------------------------

        x = make_band_noise(
            low,
            high,
            TRAIN_SECONDS,
            seed=20000 + filter_index,
        )

        # Primary disturbance
        d = lfilter(
            P,
            [1.0],
            x,
        )

        # Filtered reference x_f = S_hat * x
        #
        # 현재는 measured S를 perfect secondary estimate로 사용
        xf = lfilter(
            S,
            [1.0],
            x,
        )

        w0 = np.zeros(
            W_LEN,
            dtype=np.float64,
        )

        w, error = train_fxlms(
            x,
            d,
            xf,
            S,
            w0,
            MU,
        )

        if not np.all(np.isfinite(w)):
            raise RuntimeError(
                f"W{filter_index + 1} diverged."
            )

        w_max = float(
            np.max(np.abs(w))
        )

        if w_max > 1e4:
            raise RuntimeError(
                f"W{filter_index + 1} appears unstable. "
                f"max abs={w_max}"
            )

        # -----------------------------------------
        # Training error change
        # -----------------------------------------

        window = min(
            5 * FS,
            len(error) // 4,
        )

        mse_start = np.mean(
            error[:window] ** 2
        ) + 1e-30

        mse_end = np.mean(
            error[-window:] ** 2
        ) + 1e-30

        train_change_db = (
            10.0
            * np.log10(
                mse_end / mse_start
            )
        )

        # -----------------------------------------
        # Independent validation noise
        # -----------------------------------------

        x_val = make_band_noise(
            low,
            high,
            VALID_SECONDS,
            seed=30000 + filter_index,
        )

        val_nr_db = evaluate_filter(
            w,
            x_val,
            P,
            S,
        )

        print(
            f"training error change : "
            f"{train_change_db:8.3f} dB"
        )

        print(
            f"validation NR         : "
            f"{val_nr_db:8.3f} dB"
        )

        print(
            f"max |W|              : "
            f"{w_max:8.6f}"
        )

        Wc_v[filter_index] = w

        report.append(
            [
                filter_index + 1,
                low,
                high,
                train_change_db,
                val_nr_db,
                w_max,
            ]
        )

        # 매 filter 학습 후 임시 저장
        savemat(
            PARTIAL_W_FILE,
            {
                "Wc_v": Wc_v
            },
        )

    # ========================================================
    # Final save
    # ========================================================

    savemat(
        OUTPUT_W_FILE,
        {
            "Wc_v": Wc_v
        },
    )

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "filter",
                "low_hz",
                "high_hz",
                "train_error_change_db",
                "validation_nr_db",
                "max_abs_w",
            ]
        )

        writer.writerows(report)

    print()
    print("============================================")
    print("TRAINING COMPLETE")
    print("============================================")
    print("Wc_v shape:", Wc_v.shape)
    print()
    print("Saved:")
    print(OUTPUT_W_FILE)
    print(REPORT_FILE)

    # 다시 읽어서 마지막 검증
    check = loadmat(
        OUTPUT_W_FILE
    )["Wc_v"]

    print()
    print(
        "Reloaded Wc_v shape:",
        check.shape
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check original filter-to-band mapping only.",
    )

    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Map original filters and train all 15 filters.",
    )

    args = parser.parse_args()

    mapping = determine_original_mapping()

    if args.check_only:
        print()
        print("CHECK COMPLETE")
        return

    if args.train_all:
        train_myduct(mapping)
        return

    print()
    print(
        "Use either --check-only or --train-all"
    )


if __name__ == "__main__":
    main()
