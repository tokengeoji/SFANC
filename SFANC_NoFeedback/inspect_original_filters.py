import numpy as np
from scipy.io import loadmat
from scipy.signal import freqz

FS = 16000

data = loadmat(r"Trained models\Pretrained_Control_filters_original.mat")
W = data["Wc_v"]

print("Wc_v shape:", W.shape)
print()

for i in range(W.shape[0]):
    f, H = freqz(W[i], worN=8192, fs=FS)

    mag = np.abs(H)
    mag_db = 20 * np.log10(mag / (np.max(mag) + 1e-12) + 1e-12)

    # 최대응답보다 15 dB 이내인 주파수 영역
    mask = mag_db >= -15

    if np.any(mask):
        f_low = f[mask][0]
        f_high = f[mask][-1]

        print(
            f"W{i+1:02d}: "
            f"{f_low:7.1f} Hz ~ {f_high:7.1f} Hz"
        )
    else:
        print(f"W{i+1:02d}: band not detected")
