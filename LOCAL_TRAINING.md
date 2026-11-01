# Local 16 kHz SFANC training (without feedback-path modeling)

This upload covers SFANC training without acoustic feedback-path modeling. The original paper implementation and attribution remain in place. Feedback-aware experiments are preserved locally and excluded from this upload.

## Main files

| Stage | File | Result |
| --- | --- | --- |
| Generate synthetic noise | `generate_my_dataset_16k.py` | 30,000 one-second mono recordings at 16 kHz, 50-2000 Hz |
| Label by filter performance | `fast_label_sfanc_16k_gpu_READY.py` | `training_labels.csv` |
| Augment minority classes | `augment_minor_classes_16k_gpu.py` | Additional WAVs and augmented label/metadata CSVs |
| Split dataset | `split_sfanc_dataset.py` | Stratified 80/10/10 train/validation/test CSVs |
| Train and evaluate | `train_sfanc_1dcnn.py` | `sfanc_cnn_best.pth` and evaluation outputs |
| Analyze label quality | `analyze_nr_margin.py`, `analyze_absolute_nr_and_frequency.py` | NR margin and frequency statistics |

The labeler and augmentation scripts embed P_hat, S_hat and six control filters W1-W6. They do not use F_hat or simulate the acoustic feedback path. Primary/secondary path modeling is still used to calculate control-filter performance.

## Training code and model

The training source was found at `C:/Users/dongy/Downloads/train_sfanc_1dcnn.py` and copied to this repository. Its only code change is replacing the absolute personal BASE path with the script's directory. The Downloads original remains unchanged.

The existing root `sfanc_cnn_best.pth` is approximately 1.2 MB. Its checkpoint fields match this trainer: `model_state_dict`, `class_names`, `fs`, `input_length`, and `best_val_macro_f1`. Stored validation macro-F1 is 0.9856439115781558. This is a validation score, not test accuracy. No new training or test evaluation was performed during cleanup, and exact training provenance has not been independently established.

The CNN takes 16,000 samples, removes DC and normalizes RMS. It predicts six classes, ordered W1-W6. Training uses class-weighted cross entropy, AdamW, validation macro-F1 selection, and early stopping. Test metrics are printed; history, confusion matrix and predictions are written to separate files.

The original upstream `Network.py` and `Trained models/model.pth` belong to the paper implementation. The custom six-class checkpoint uses `SFANC1DCNN` in `train_sfanc_1dcnn.py`; it is not a direct replacement for the original 15-class notebook model.

## Environment and execution

The existing local environment uses Python 3.12.10. `environment-training-snapshot.txt` records observed direct dependency versions; it is a reference snapshot, not a complete portable lockfile. Install torch/torchaudio appropriate for your CUDA driver, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run from the repository root, in this order:

```powershell
python generate_my_dataset_16k.py
python fast_label_sfanc_16k_gpu_READY.py
python augment_minor_classes_16k_gpu.py
python split_sfanc_dataset.py
python train_sfanc_1dcnn.py
```

These commands can overwrite generated data and the existing best checkpoint. Back up results before rerunning. They were not run during cleanup. Dataset files are stored under `noise_dataset/My_Dataset_16k/`. Seeds and settings are defined in each script. The original dataset/split CSVs must be backed up separately for exact experiment reproduction.

Optional label analysis:

```powershell
python analyze_nr_margin.py
python analyze_absolute_nr_and_frequency.py
```

## Supporting experiment files

`measured_paths.c`, `path_data`, `pretrained_filters.h`, `make_my_paths.py`, `pretrain_myduct.py` and `inspect_original_filters.py` are retained as measured inputs and earlier path/filter experiments. Raw measurement files may contain additional measured paths; this does not mean the labeler uses feedback. The labeler and augmentation scripts use their embedded coefficient arrays.

The modified primary/secondary MAT files and original backups are retained, along with `Trained models/Pretrained_Control_filters_myduct.mat` and mapping/report. `pretrain_myduct.py` is an earlier 15-filter experiment, not an automatic generator for the embedded six-filter bank. These supporting scripts are not mandatory steps in the sequence above.

## Excluded from this upload

- `.venv/`, Python caches and editor files.
- `noise_dataset/`, including generated WAVs, labels and split CSVs.
- Entire `SFANC_Feedback_Training/`, including feedback coefficients, trainers, models, metrics, UART code and runtime distributions.
- Root `train_sfanc_feedback_1dcnn.py`.
- Partial filter-pretraining checkpoint.

No excluded files were deleted or moved. The IDE's UART script in Downloads was not changed. The environment snapshot may mention packages absent from the local environment; it records observations and does not add UART dependencies to this training scope.

## Review and upload

The current origin points to the original author's `Luo-Zhengding/SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning` repository. Configure your own repository before pushing. Staging, commit and push remain on hold for user review.

```powershell
git status --short
git diff --check
git ls-files --others --exclude-standard
```

The original README and citation remain preserved. Generated data must be backed up outside Git. Empty folders are not stored by Git.
