
"""
Generate a fresh 16 kHz synthetic SFANC dataset.

- 30,000 mono WAV files
- 1.0 s each
- fs = 16 kHz
- random band-limited noise entirely within 50-2000 Hz
- output: noise_dataset/My_Dataset_16k

The generator uses FFT-domain shaping, so it is much faster than designing/applying
a new FIR band-pass filter for every WAV.
"""

from pathlib import Path
import csv
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "noise_dataset" / "My_Dataset_16k"

FS = 16000
DURATION_SEC = 1.0
N = int(FS * DURATION_SEC)
NUM_FILES = 30000

LOW_MIN = 50
LOW_MAX = 1900
HIGH_MAX = 2000
MIN_BW = 120

RMS_TARGET_MIN = 0.08
RMS_TARGET_MAX = 0.22

SEED = 20261005


def make_band_limited_noise(rng, low_hz, high_hz):
    # Complex random spectrum with conjugate symmetry handled by irfft.
    n_freq = N // 2 + 1
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)

    spec = np.zeros(n_freq, dtype=np.complex128)
    mask = (freqs >= low_hz) & (freqs <= high_hz)

    # Gaussian random spectrum within the desired band.
    re = rng.standard_normal(mask.sum())
    im = rng.standard_normal(mask.sum())
    spec[mask] = re + 1j * im

    # Remove DC/Nyquist imaginary components if they happen to be included.
    spec[0] = spec[0].real + 0j
    if N % 2 == 0:
        spec[-1] = spec[-1].real + 0j

    x = np.fft.irfft(spec, n=N)

    # Random RMS level.
    rms = np.sqrt(np.mean(x * x) + 1e-20)
    target_rms = rng.uniform(RMS_TARGET_MIN, RMS_TARGET_MAX)
    x = x * (target_rms / rms)

    # Keep safe headroom.
    peak = np.max(np.abs(x)) + 1e-20
    if peak > 0.95:
        x *= 0.95 / peak

    return x.astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    metadata = []

    for i in tqdm(range(NUM_FILES), desc="Generating 16 kHz dataset"):
        low = int(rng.integers(LOW_MIN, LOW_MAX + 1))
        high_min = min(HIGH_MAX, low + MIN_BW)

        if high_min >= HIGH_MAX:
            low = int(rng.integers(LOW_MIN, HIGH_MAX - MIN_BW + 1))
            high_min = low + MIN_BW

        high = int(rng.integers(high_min, HIGH_MAX + 1))

        x = make_band_limited_noise(rng, low, high)

        name = f"noise_{i:05d}.wav"
        wavfile.write(OUT_DIR / name, FS, np.round(x * 32767.0).astype(np.int16))
        metadata.append((name, low, high))

    meta_csv = OUT_DIR / "dataset_metadata.csv"
    with meta_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "low_hz", "high_hz"])
        w.writerows(metadata)

    print()
    print(f"Saved {NUM_FILES} WAV files to: {OUT_DIR}")
    print(f"Saved metadata to: {meta_csv}")


if __name__ == "__main__":
    main()
