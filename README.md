# SFANC

SFANC filter-selection training and pretrained model for the capstone project.

## Training without acoustic feedback-path modeling

All implementation files, measured-path data, models and upstream examples are in [`SFANC_NoFeedback/`](SFANC_NoFeedback/).

- [Training workflow and environment](SFANC_NoFeedback/LOCAL_TRAINING.md)
- [Six-class CNN trainer](SFANC_NoFeedback/train_sfanc_1dcnn.py)
- [Pretrained six-class model](SFANC_NoFeedback/sfanc_cnn_best.pth)
- [Original paper and upstream attribution](SFANC_NoFeedback/UPSTREAM_README.md)

Run scripts from the implementation folder so relative paths resolve correctly:

```powershell
cd SFANC_NoFeedback
```

Training commands and their output-overwrite behavior are documented in the workflow. Organizing these files does not require retraining. Generated datasets, virtual environments and feedback-aware experiments are excluded from Git.
