# PC real-time SFANC runtime

These scripts classify one-second, 16 kHz audio frames into one of six fixed
ANC filters and send the selected one-byte filter ID to the DSP over UART.

## Entry points

- `realtime_sfanc_uart.py`: no-feedback classifier. Its default model is the
  canonical `../../filter_training/sfanc_cnn_best.pth`.
- `realtime_sfanc_uart_feedback.py`: feedback-aware classifier. Its default
  model is `models/sfanc_cnn_feedback_best.pth`.
- `pretrained_filters.py`: inspect the six 256-tap control-filter arrays.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/requirements.txt
```

Useful commands from the repository root:

```powershell
python Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/realtime_sfanc_uart.py --mode scan
python Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/realtime_sfanc_uart.py --mode check --channel 1
python Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/realtime_sfanc_uart.py --mode test
python Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/realtime_sfanc_uart.py --mode wav --wav path\to\sample.wav
python Hybrid_FxNLMS_SFANC/execution/sfanc_filter_selection/realtime_sfanc_uart.py --mode mic --port COM10 --channel 1
```

If `--port` is omitted, the runtime prioritizes `COM10` and then common USB
serial adapters. If no port is available, inference continues in dry-run mode.
Use `--model` to override either default checkpoint path.
