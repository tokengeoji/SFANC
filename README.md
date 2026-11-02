# SFANC

Selective Fixed-filter Active Noise Control for an OMAP-L138/TMS320C6748 DSP.
The repository contains the offline training workflow, PC-side real-time noise
classifier/UART dispatcher, and Code Composer Studio firmware projects.

## Repository layout

```text
SFANC_NoFeedback/             Training, analysis, measured paths, and base model
runtime/                      Real-time microphone/WAV inference and UART control
  models/                     Runtime-only model variants
firmware/
  FxNLMS0/                    Baseline FxNLMS Code Composer Studio project
  FxNLMS0_16_sfanc/           SFANC firmware with UART filter switching
```

Generated datasets, Python caches, and Code Composer Studio build output are
excluded from Git. Each model and filter set has one canonical copy to avoid
drift between the training, runtime, and firmware folders.

## Hardware setup

The reference setup uses an ECM8000 microphone, UMC404HD audio interface,
OMAP-L138 DSP, TPA3116D2 amplifier, and a DIY 4-inch full-range speaker.
See [`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md) for the complete device
list, signal flow, and cable connection table.

## Real-time runtime

Install the runtime dependencies and inspect the available audio/serial devices:

```powershell
python -m pip install -r runtime/requirements.txt
python runtime/realtime_sfanc_uart.py --mode scan
```

Run the no-feedback model (loaded from `SFANC_NoFeedback/sfanc_cnn_best.pth`):

```powershell
python runtime/realtime_sfanc_uart.py --mode mic --channel 1
```

Run the feedback-aware model stored in `runtime/models/`:

```powershell
python runtime/realtime_sfanc_uart_feedback.py --mode mic --channel 1
```

See [`runtime/README.md`](runtime/README.md) for operating modes and hardware
arguments.

## Training without acoustic feedback-path modeling

The implementation, measured-path data, models, and upstream examples are in
[`SFANC_NoFeedback/`](SFANC_NoFeedback/).

- [Training workflow and environment](SFANC_NoFeedback/LOCAL_TRAINING.md)
- [Six-class CNN trainer](SFANC_NoFeedback/train_sfanc_1dcnn.py)
- [Pretrained six-class model](SFANC_NoFeedback/sfanc_cnn_best.pth)
- [Original paper and upstream attribution](SFANC_NoFeedback/UPSTREAM_README.md)

Run training scripts from the implementation folder so their relative paths
resolve correctly:

```powershell
cd SFANC_NoFeedback
```

## DSP firmware

Import either folder under `firmware/` as an existing Code Composer Studio
project. The projects target CCS 9.3, TI C6000 compiler 8.3.5, SYS/BIOS 6.76,
and the LCDK OMAP-L138. See [`firmware/README.md`](firmware/README.md) for the
project roles and generated-file policy.
