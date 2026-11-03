# Hybrid_FxNLMS_SFANC

An OMAP-L138-based Hybrid_FxNLMS_SFANC project. It includes filter training,
PC-side real-time noise classification and UART communication, and DSP ANC
execution code for Code Composer Studio.

## Repository layout

```text
Hybrid_FxNLMS_SFANC/
├─ hardware_setup/                         Hardware, duct diagrams, and photos
├─ filter_training/                        SFANC filter-selection model training
└─ execution/
   ├─ anc_execution/
   │  └─ ccs_projects/
   │     ├─ FxNLMS0/                       Baseline FxNLMS CCS project
   │     └─ FxNLMS0_16_sfanc/              SFANC-enabled CCS project
   └─ sfanc_filter_selection/               PC inference and UART dispatcher
```

Generated datasets, Python caches, and Code Composer Studio build output are
excluded from Git. Each model and filter set has one canonical copy to avoid
drift between the training, PC runtime, and DSP firmware folders.

The root `README.md` is the repository's main guide. `.gitignore` tells Git
which generated, temporary, or machine-specific files must not be uploaded.

## Hardware setup

The reference setup uses an ECM8000 microphone, UMC404HD audio interface,
OMAP-L138 DSP, TPA3116D2 amplifier, and a DIY 4-inch full-range speaker.
See
[`Hybrid_FxNLMS_SFANC/hardware_setup/HARDWARE_SETUP.md`](Hybrid_FxNLMS_SFANC/hardware_setup/HARDWARE_SETUP.md)
for the complete device list, signal flow, and cable connection table.

## PC real-time SFANC runtime

Install the runtime dependencies and inspect the available audio/serial devices:

```powershell
python -m pip install -r Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/requirements.txt
python Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/realtime_sfanc_uart.py --mode scan
```

Run the no-feedback model (loaded from
`Hybrid_FxNLMS_SFANC/filter_training/sfanc_cnn_best.pth`):

```powershell
python Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/realtime_sfanc_uart.py --mode mic --channel 1
```

Run the feedback-aware model stored in
`Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/models/`:

```powershell
python Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/realtime_sfanc_uart_feedback.py --mode mic --channel 1
```

See the
[PC filter-selection guide](Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/README.md)
for operating modes and hardware arguments.

## SFANC filter training (without acoustic feedback-path modeling)

The implementation, measured-path data, models, and upstream examples are in
[`Hybrid_FxNLMS_SFANC/filter_training/`](Hybrid_FxNLMS_SFANC/filter_training/).

- [Training workflow and environment](Hybrid_FxNLMS_SFANC/filter_training/LOCAL_TRAINING.md)
- [Six-class CNN trainer](Hybrid_FxNLMS_SFANC/filter_training/train_sfanc_1dcnn.py)
- [Pretrained six-class model](Hybrid_FxNLMS_SFANC/filter_training/sfanc_cnn_best.pth)
- [Original paper and upstream attribution](Hybrid_FxNLMS_SFANC/filter_training/UPSTREAM_README.md)

Run training scripts from the implementation folder so their relative paths
resolve correctly:

```powershell
cd Hybrid_FxNLMS_SFANC/filter_training
```

## DSP ANC firmware

Import either folder under
`Hybrid_FxNLMS_SFANC/execution/anc_execution/ccs_projects/` as an existing
Code Composer Studio project. The projects target CCS 9.3, TI C6000 compiler
8.3.5, SYS/BIOS 6.76, and the LCDK OMAP-L138. See the
[DSP ANC execution guide](Hybrid_FxNLMS_SFANC/execution/anc_execution/README.md)
for the project roles and generated-file policy.

## 프로젝트 설명 (한글)

이 프로젝트는 OMAP-L138 DSP를 기반으로 구현한 Hybrid FxNLMS-SFANC
능동소음제어 시스템입니다.

OMAP-L138 DSP에서 16 kHz 샘플링 주파수로 FxNLMS 알고리즘을 구현하는
과정에서 컨트롤 스피커의 출력이 레퍼런스 마이크로 다시 유입되는 음향
피드백 현상이 발생했습니다. 이를 보완하기 위해 컨트롤 스피커에서
레퍼런스 마이크로 전달되는 피드백 경로를 고려했습니다. 또한 1D CNN이
레퍼런스 마이크의 소음을 실시간으로 분류하고, 현재 소음에 적합한 사전
학습 필터를 선택합니다. PC는 선택한 필터 번호를 UART로 OMAP-L138에
전송하며, DSP는 해당 필터를 적용해 ANC를 수행합니다.
