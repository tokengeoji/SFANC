import re
import numpy as np
from scipy.io import savemat, loadmat

C_FILE = "measured_paths.c"

PRIMARY_OUTPUT = r"Primary and Secondary Path\Primary_path.mat"
SECONDARY_OUTPUT = r"Primary and Secondary Path\Secondary_path.mat"

EXPECTED_PRIMARY_TAPS = 500
EXPECTED_SECONDARY_TAPS = 500


def extract_c_array(text, name):
    pattern = rf"\b{name}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}}\s*;"
    match = re.search(pattern, text, flags=re.S)

    if match is None:
        raise RuntimeError(f"{name} array not found.")

    body = match.group(1)

    # C comments are removed before extracting numbers.
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//.*?$", "", body, flags=re.M)

    numbers = re.findall(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?",
        body
    )

    values = [float(x.rstrip("fF")) for x in numbers]
    return np.asarray(values, dtype=np.float64)


with open(C_FILE, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

S_hat = extract_c_array(text, "S_hat")
P_hat = extract_c_array(text, "P_hat")

print("P_hat tap count:", len(P_hat))
print("S_hat tap count:", len(S_hat))

if len(P_hat) != EXPECTED_PRIMARY_TAPS:
    raise RuntimeError(
        f"P_hat length = {len(P_hat)}, expected = {EXPECTED_PRIMARY_TAPS}"
    )

if len(S_hat) != EXPECTED_SECONDARY_TAPS:
    raise RuntimeError(
        f"S_hat length = {len(S_hat)}, expected = {EXPECTED_SECONDARY_TAPS}"
    )

Pz1 = P_hat.reshape(-1, 1)
S = S_hat.reshape(-1, 1)

savemat(PRIMARY_OUTPUT, {"Pz1": Pz1})
savemat(SECONDARY_OUTPUT, {"S": S})

P_check = loadmat(PRIMARY_OUTPUT)["Pz1"]
S_check = loadmat(SECONDARY_OUTPUT)["S"]

print()
print("SAVE COMPLETE")
print("Pz1 shape :", P_check.shape)
print("S shape   :", S_check.shape)
print("P max abs :", np.max(np.abs(P_check)))
print("S max abs :", np.max(np.abs(S_check)))
print("P first 5 :", P_check[:5, 0])
print("S first 5 :", S_check[:5, 0])
