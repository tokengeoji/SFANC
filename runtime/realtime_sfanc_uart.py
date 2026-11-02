# -*- coding: utf-8 -*-
"""
Real-Time SFANC Noise Classifier and UART Filter Dispatcher
Target: TMS320C6748 / OMAP-L138 (Project: FxNLMS0_16_sfanc)
Sampling Rate: 16,000 Hz (16 kHz mono)

Features:
- Loads trained 6-class 1D CNN (sfanc_cnn_best.pth)
- Real-time audio stream from microphone (e.g., Behringer UMC404HD) or WAV simulation
- Multi-channel support: Select Left (Ch0) or Right (Ch1) for Reference Microphone
- Automatic COM10 detection & 115200 baud serial transmission
- [NEW] Debounce & Hysteresis Stabilization: Prevents boundary chattering (e.g. 300~500Hz ping-pong)
- Preprocesses audio (DC offset removal, RMS normalization) matching training
- Runs CNN inference every 1 second (16,000 samples)
- Transmits 1-byte pre-trained filter index (1 ~ 6) over UART to DSP
- Real-time terminal dashboard with clear filter hot-swap alert banners
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 1D CNN runs in <1ms on CPU cleanly

import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

import sys
import time
import argparse
import queue
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


# =========================================================================
# Class & Filter Metadata (Matches pretrained_filters.h)
# =========================================================================
CLASS_INFO = [
    {"id": 1, "name": "W1", "band": "50 ~ 300 Hz",   "desc": "Low-frequency rumble"},
    {"id": 2, "name": "W2", "band": "50 ~ 1000 Hz",  "desc": "Broadband low-mid noise"},
    {"id": 3, "name": "W3", "band": "50 ~ 2000 Hz",  "desc": "Wideband full noise"},
    {"id": 4, "name": "W4", "band": "300 ~ 700 Hz",  "desc": "Mid-frequency duct noise"},
    {"id": 5, "name": "W5", "band": "700 ~ 1200 Hz", "desc": "Upper-mid noise"},
    {"id": 6, "name": "W6", "band": "1200 ~ 2000 Hz", "desc": "High-frequency tonal/broad"},
]
FS = 16000
INPUT_LENGTH = 16000


# =========================================================================
# 1D CNN Architecture (Exact match to train_sfanc_1dcnn.py)
# =========================================================================
class SFANC1DCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# =========================================================================
# Signal Preprocessing
# =========================================================================
def preprocess_audio(raw_samples):
    """
    Converts raw 1-second audio array (16,000 samples) to normalized tensor.
    Matches train_sfanc_1dcnn.py:
      1. Remove DC bias (x - mean(x))
      2. Normalize RMS scale mildly (x / sqrt(mean(x^2) + 1e-12))
    """
    x = raw_samples.astype(np.float32)

    if len(x) < INPUT_LENGTH:
        x = np.pad(x, (0, INPUT_LENGTH - len(x)))
    elif len(x) > INPUT_LENGTH:
        x = x[:INPUT_LENGTH]

    raw_rms = float(np.sqrt(np.mean(x * x)))

    # Remove DC bias
    x = x - np.mean(x)

    # RMS Normalization
    rms = np.sqrt(np.mean(x * x) + 1e-12)
    if rms > 0:
        x = x / rms

    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1, 1, 16000]
    return tensor, raw_rms


# =========================================================================
# Serial Port Helper (Priority: COM10)
# =========================================================================
def find_default_com_port():
    if not SERIAL_AVAILABLE:
        return None
    ports = list(serial.tools.list_ports.comports())

    # Priority 1: User specified COM10
    for p in ports:
        if p.device.upper() == "COM10":
            return p.device

    # Priority 2: Standard USB Serial or User CDC
    for p in ports:
        desc = p.description.lower()
        if "usb serial" in desc or "user cdc" in desc:
            return p.device

    for p in ports:
        desc = p.description.lower()
        if "cp210" in desc or "ch340" in desc or "ftdi" in desc:
            return p.device

    if len(ports) > 0:
        return ports[0].device
    return None


# =========================================================================
# Synthetic Noise Generator (for test mode)
# =========================================================================
def generate_synthetic_noise(class_idx, duration=1.0, fs=16000):
    band_ranges = [
        (50, 300),
        (50, 1000),
        (50, 2000),
        (300, 700),
        (700, 1200),
        (1200, 2000),
    ]
    low, high = band_ranges[class_idx]
    n_samples = int(duration * fs)
    white = np.random.randn(n_samples)
    sos = scipy.signal.butter(6, [low, high], btype='bandpass', fs=fs, output='sos')
    filtered = scipy.signal.sosfilt(sos, white)
    filtered = filtered / (np.max(np.abs(filtered)) + 1e-6)
    return filtered.astype(np.float32)


# =========================================================================
# Quick Channel Check Helper
# =========================================================================
def check_mic_channels(device_id=None, duration=3.0):
    if not SOUNDDEVICE_AVAILABLE:
        print("[ERROR] sounddevice library not installed.")
        return

    print("\n" + "=" * 65)
    print("   [MIC CHANNEL CHECK] Measuring Left vs Right Audio Levels...")
    print(f"   Please make noise near Reference Mic or Error Mic for {duration:.0f}s...")
    print("=" * 65)

    try:
        rec = sd.rec(int(duration * FS), samplerate=FS, channels=2, dtype='float32', device=device_id)
        sd.wait()

        ch0 = rec[:, 0]
        ch1 = rec[:, 1]

        rms0 = float(np.sqrt(np.mean(ch0 * ch0)))
        rms1 = float(np.sqrt(np.mean(ch1 * ch1)))

        print(f"\n  [Channel 0 (Left / Input 1)]  RMS: {rms0:.5f}")
        print(f"  [Channel 1 (Right / Input 2)] RMS: {rms1:.5f}")
        print("-" * 65)
        if rms0 > rms1 * 2:
            print("  --> Channel 0 (Left) has stronger signal.")
            print("      If your Reference mic is on Ch0, run with: --channel 0")
        elif rms1 > rms0 * 2:
            print("  --> Channel 1 (Right) has stronger signal.")
            print("      If your Reference mic is on Ch1, run with: --channel 1")
        else:
            print("  --> Both channels have comparable signal levels.")
            print("      Select --channel 0 (Left) or --channel 1 (Right) based on wiring.")
        print("=" * 65 + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to read audio: {e}")


# =========================================================================
# Main Execution Loop
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="SFANC Real-Time Noise Classifier & DSP UART Dispatcher")
    parser.add_argument("--model", type=str, default="sfanc_cnn_best.pth", help="Path to trained .pth model")
    parser.add_argument("--port", type=str, default=None, help="Serial port (default: auto-detects COM10)")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud rate (default: 115200)")
    parser.add_argument("--mode", type=str, choices=["mic", "wav", "test", "scan", "check"], default="mic",
                        help="Operating mode: mic (live stream), wav (file play), test (synthetic sweep), scan (ports), check (test mic L/R)")
    parser.add_argument("--channel", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Audio input channel to classify (0=Left/Input 1, 1=Right/Input 2). Default: 1")
    parser.add_argument("--debounce", type=int, default=2,
                        help="Number of consecutive frames required to confirm filter swap (default: 2, 1=instant)")
    parser.add_argument("--cooldown", type=int, default=2,
                        help="Minimum seconds/frames to hold active filter before next swap (default: 2)")
    parser.add_argument("--wav", type=str, default=None, help="WAV file path for --mode wav")
    parser.add_argument("--device", type=int, default=None, help="Audio input device ID (optional)")
    parser.add_argument("--gate", type=float, default=0.002, help="Silence RMS threshold gate (default: 0.002)")
    parser.add_argument("--always-send", action="store_true", help="Send byte every frame even if class didn't change")
    args = parser.parse_args()

    # Find audio device (prioritize UMC404HD if device not explicitly set)
    dev_id = args.device
    if dev_id is None and SOUNDDEVICE_AVAILABLE:
        for i, d in enumerate(sd.query_devices()):
            if "umc404hd" in d["name"].lower() and d["max_input_channels"] > 0:
                dev_id = i
                break

    # Mode: Scan devices and exit
    if args.mode == "scan":
        print("\n--- Available Serial (COM) Ports ---")
        if SERIAL_AVAILABLE:
            for p in serial.tools.list_ports.comports():
                star = " [DEFAULT/SELECTED]" if p.device.upper() == "COM10" else ""
                print(f"  {p.device:6s} : {p.description}{star}")
        else:
            print("  pyserial not installed.")

        print("\n--- Available Audio Input Devices ---")
        if SOUNDDEVICE_AVAILABLE:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    sel = " [AUTO-DETECTED]" if i == dev_id else ""
                    print(f"  [{i:2d}] {d['name']} (Chs: {d['max_input_channels']}, Default SR: {int(d['default_samplerate'])}){sel}")
        else:
            print("  sounddevice not installed.")
        return

    # Mode: Check mic Left/Right levels
    if args.mode == "check":
        check_mic_channels(device_id=dev_id)
        return

    # 1. Load PyTorch model. Keep the trained model in its canonical
    # SFANC_NoFeedback directory instead of duplicating it under runtime.
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        candidates = [
            base_dir / model_path,
            base_dir / "models" / model_path,
            repo_root / "SFANC_NoFeedback" / model_path,
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {args.model}")
        print("  Searched runtime/, runtime/models/, and SFANC_NoFeedback/.")
        sys.exit(1)

    print(f"\n[INFO] Loading SFANC 1D CNN model from: {model_path}")
    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device)

    model = SFANC1DCNN(num_classes=6).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Model loaded successfully on device: {device}")

    # 2. Setup Serial Port (Prioritizing COM10)
    ser = None
    target_port = args.port or find_default_com_port()
    if target_port and SERIAL_AVAILABLE:
        try:
            ser = serial.Serial(port=target_port, baudrate=args.baud, timeout=0.1)
            print(f"[INFO] Connected to DSP UART on >> {target_port} << @ {args.baud} bps 8-N-1")
        except Exception as e:
            print(f"[WARN] Could not open {target_port}: {e}. Running in DRY-RUN mode (Terminal Display Only).")
            ser = None
    else:
        print("[INFO] No serial port specified or found. Running in DRY-RUN mode.")

    ch_name = "Channel 0 (Left / Input 1)" if args.channel == 0 else f"Channel {args.channel} (Right / Input {args.channel + 1})"
    print(f"[INFO] Reference Mic Source  : {ch_name}")
    print(f"[INFO] Filter Stabilization  : Confirmed after {args.debounce} consecutive frames | Min Hold: {args.cooldown}s")
    print(f"[INFO] Tip: If chattering occurs, adjust with '--debounce {args.debounce + 1}'")

    # 3. Execution based on mode
    print("\n" + "=" * 80)
    print(f"   SFANC Real-Time Noise Classifier (16 kHz) Active [Mode: {args.mode.upper()}]")
    print("=" * 80)
    print(f"{'Frame':>5} | {'Time (s)':>8} | {'Pred':>4} | {'Frequency Band':^15} | {'Conf (%)':>8} | {'RMS':>7} | {'UART Tx':^9} | Status")
    print("-" * 80)

    # Stabilization / Debounce State Tracking
    active_class_id = None
    active_class_name = "None"
    pending_class_id = None
    pending_count = 0
    last_swap_frame = 0

    frame_idx = 0
    t0 = time.time()

    def process_frame(audio_chunk):
        nonlocal active_class_id, active_class_name, pending_class_id, pending_count, last_swap_frame, frame_idx
        frame_idx += 1
        elapsed = time.time() - t0

        tensor, raw_rms = preprocess_audio(audio_chunk)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = probs[pred_idx] * 100.0

        class_meta = CLASS_INFO[pred_idx]
        class_id = class_meta["id"]
        is_silence = (raw_rms < args.gate)

        tx_str = "--"
        status_str = ""

        if is_silence:
            status_str = "[SILENCE / GATE]"
            pending_class_id = None
            pending_count = 0

        elif active_class_id is None:
            # First sound detected: Initialize immediately
            active_class_id = class_id
            active_class_name = class_meta["name"]
            last_swap_frame = frame_idx

            if ser and ser.is_open:
                try:
                    ser.write(bytes([class_id]))
                    ser.flush()
                    tx_str = f"0x{class_id:02X} ({class_id})"
                except Exception:
                    tx_str = "ERR"
            else:
                tx_str = f"0x{class_id:02X} (DRY)"

            print("\n" + "#" * 80)
            print(f"  [INITIAL FILTER SET] Starting on {class_meta['name']} ({class_meta['band']})")
            print(f"  Target Noise : {class_meta['desc']}")
            print(f"  Confidence   : {conf:.1f}%  |  Audio RMS: {raw_rms:.5f}")
            print(f"  UART Command : Sent byte 0x{class_id:02X} (Index {class_id}) -> DSP Initial W(z)")
            print("#" * 80 + "\n")
            status_str = f"*** INITIALIZED TO {class_meta['name']} ***"

        elif class_id == active_class_id:
            # Prediction matches currently active filter: Stable!
            pending_class_id = None
            pending_count = 0
            if args.always_send and ser and ser.is_open:
                try:
                    ser.write(bytes([class_id]))
                    ser.flush()
                    tx_str = f"0x{class_id:02X} ({class_id})"
                except Exception:
                    pass
                status_str = "[HEARTBEAT]"
            else:
                status_str = f"[MAINTAIN: {active_class_name}]"

        else:
            # Prediction differs from currently active filter: Apply Debounce Confirmation!
            if class_id == pending_class_id:
                pending_count += 1
            else:
                pending_class_id = class_id
                pending_count = 1

            time_since_last_swap = frame_idx - last_swap_frame

            if pending_count >= args.debounce and time_since_last_swap >= args.cooldown:
                # CONFIRMED: Swap authorized!
                old_name = active_class_name
                active_class_id = class_id
                active_class_name = class_meta["name"]
                last_swap_frame = frame_idx
                pending_class_id = None
                pending_count = 0

                if ser and ser.is_open:
                    try:
                        ser.write(bytes([class_id]))
                        ser.flush()
                        tx_str = f"0x{class_id:02X} ({class_id})"
                    except Exception:
                        tx_str = "ERR"
                else:
                    tx_str = f"0x{class_id:02X} (DRY)"

                print("\n" + "#" * 80)
                print(f"  [FILTER SWAP CONFIRMED] {old_name}  ===>  {class_meta['name']} ({class_meta['band']})")
                print(f"  Stabilization: Confirmed over {args.debounce} consecutive frames (No chattering)")
                print(f"  Target Noise : {class_meta['desc']}")
                print(f"  Confidence   : {conf:.1f}%  |  Audio RMS: {raw_rms:.5f}")
                print(f"  UART Command : Sent byte 0x{class_id:02X} (Index {class_id}) -> DSP Hot-Swapping W(z)")
                print("#" * 80 + "\n")
                status_str = f"*** SWAPPED TO {class_meta['name']} ***"
            else:
                # UNCONFIRMED: Hold active filter on DSP to protect FxNLMS adaptation
                cand_meta = CLASS_INFO[pending_class_id - 1]
                status_str = f"[HOLD: {active_class_name} | Candid: {cand_meta['name']} ({pending_count}/{args.debounce})]"

        print(f"{frame_idx:5d} | {elapsed:8.1f} | {class_meta['name']:>4} | {class_meta['band']:^15} | {conf:7.1f}% | {raw_rms:7.4f} | {tx_str:^9} | {status_str}")

    # MODE: TEST (Synthetic Sweep W1 to W6)
    if args.mode == "test":
        print("[INFO] Running synthetic noise test sweep through all 6 frequency bands...")
        try:
            for rep in range(2):
                for c in range(6):
                    target_meta = CLASS_INFO[c]
                    print(f"\n>> Feeding synthetic noise: {target_meta['name']} ({target_meta['band']})...")
                    for sub in range(args.debounce + 1):
                        noise = generate_synthetic_noise(c, duration=1.0, fs=FS)
                        process_frame(noise)
                        time.sleep(1.0)
            print("\n[INFO] Test sweep completed successfully!")
        except KeyboardInterrupt:
            print("\n[INFO] Test interrupted by user.")

    # MODE: WAV FILE SIMULATION
    elif args.mode == "wav":
        if not args.wav:
            print("[ERROR] Please specify WAV file path with --wav <file.wav>")
            return
        wav_path = Path(args.wav)
        if not wav_path.exists():
            print(f"[ERROR] File not found: {wav_path}")
            return

        from scipy.io import wavfile
        file_fs, data = wavfile.read(wav_path)
        if data.ndim > 1:
            data = data[:, args.channel] if data.shape[1] > args.channel else data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        else:
            data = data.astype(np.float32)

        if file_fs != FS:
            print(f"[INFO] Resampling from {file_fs} Hz to {FS} Hz...")
            num_samples = int(len(data) * FS / file_fs)
            data = scipy.signal.resample(data, num_samples)

        print(f"[INFO] Streaming WAV file: {wav_path.name} ({len(data)/FS:.1f} seconds total)")
        total_frames = len(data) // INPUT_LENGTH
        try:
            for i in range(total_frames):
                chunk = data[i * INPUT_LENGTH : (i + 1) * INPUT_LENGTH]
                process_frame(chunk)
                time.sleep(1.0)
            print("\n[INFO] WAV streaming completed.")
        except KeyboardInterrupt:
            print("\n[INFO] Playback stopped by user.")

    # MODE: LIVE MICROPHONE STREAMING
    elif args.mode == "mic":
        if not SOUNDDEVICE_AVAILABLE:
            print("[ERROR] sounddevice library is required for live mic capture.")
            return

        audio_q = queue.Queue()
        sel_ch = args.channel

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[AUDIO STATUS] {status}", file=sys.stderr)
            audio_q.put(indata[:, sel_ch].copy())

        block_size = FS  # 1-second blocks (16,000 samples)
        num_channels = max(2, sel_ch + 1)
        print(f"[INFO] Starting audio stream (Sample rate: {FS} Hz, Input Device: [{dev_id}])...")
        print("[INFO] Press Ctrl+C to terminate.\n")

        try:
            with sd.InputStream(samplerate=FS, blocksize=block_size, device=dev_id,
                                channels=num_channels, dtype='float32', callback=audio_callback):
                while True:
                    try:
                        chunk = audio_q.get(timeout=2.0)
                        process_frame(chunk)
                    except queue.Empty:
                        pass
        except KeyboardInterrupt:
            print("\n[INFO] Live streaming stopped by user.")
        except Exception as e:
            print(f"\n[ERROR] Audio stream exception: {e}")

    if ser and ser.is_open:
        ser.close()
        print("[INFO] UART port closed.")

if __name__ == '__main__':
    main()
