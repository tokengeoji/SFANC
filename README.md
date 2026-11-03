# SFANC

Selective Fixed-filter Active Noise Control for an OMAP-L138/TMS320C6748 DSP.
The repository contains the offline training workflow, PC-side real-time noise
classifier/UART dispatcher, and Code Composer Studio firmware projects.

## Repository layout

```text
sfanc_filter_training/        Filter-selection model training and analysis
pc_sfanc_runtime/             PC microphone/WAV inference and UART control
  models/                     Runtime-only model variants
dsp_anc_firmware/
  FxNLMS0/                    Baseline FxNLMS Code Composer Studio project
  FxNLMS0_16_sfanc/           SFANC firmware with UART filter switching
docs/                         Hardware documentation, diagrams, and photos
```

Generated datasets, Python caches, and Code Composer Studio build output are
excluded from Git. Each model and filter set has one canonical copy to avoid
drift between the training, PC runtime, and DSP firmware folders.

The root `README.md` is the repository's main guide. `.gitignore` tells Git
which generated, temporary, or machine-specific files must not be uploaded.

## Hardware setup

The reference setup uses an ECM8000 microphone, UMC404HD audio interface,
OMAP-L138 DSP, TPA3116D2 amplifier, and a DIY 4-inch full-range speaker.
See [`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md) for the complete device
list, signal flow, and cable connection table.

## PC real-time SFANC runtime

Install the runtime dependencies and inspect the available audio/serial devices:

```powershell
python -m pip install -r pc_sfanc_runtime/requirements.txt
python pc_sfanc_runtime/realtime_sfanc_uart.py --mode scan
```

Run the no-feedback model (loaded from
`sfanc_filter_training/sfanc_cnn_best.pth`):

```powershell
python pc_sfanc_runtime/realtime_sfanc_uart.py --mode mic --channel 1
```

Run the feedback-aware model stored in `pc_sfanc_runtime/models/`:

```powershell
python pc_sfanc_runtime/realtime_sfanc_uart_feedback.py --mode mic --channel 1
```

See [`pc_sfanc_runtime/README.md`](pc_sfanc_runtime/README.md) for operating
modes and hardware arguments.

## SFANC filter training (without acoustic feedback-path modeling)

The implementation, measured-path data, models, and upstream examples are in
[`sfanc_filter_training/`](sfanc_filter_training/).

- [Training workflow and environment](sfanc_filter_training/LOCAL_TRAINING.md)
- [Six-class CNN trainer](sfanc_filter_training/train_sfanc_1dcnn.py)
- [Pretrained six-class model](sfanc_filter_training/sfanc_cnn_best.pth)
- [Original paper and upstream attribution](sfanc_filter_training/UPSTREAM_README.md)

Run training scripts from the implementation folder so their relative paths
resolve correctly:

```powershell
cd sfanc_filter_training
```

## DSP ANC firmware

Import either folder under `dsp_anc_firmware/` as an existing Code Composer
Studio project. The projects target CCS 9.3, TI C6000 compiler 8.3.5,
SYS/BIOS 6.76, and the LCDK OMAP-L138. See
[`dsp_anc_firmware/README.md`](dsp_anc_firmware/README.md) for the project
roles and generated-file policy.
