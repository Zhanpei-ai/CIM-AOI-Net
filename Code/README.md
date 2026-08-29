# AOI-Net

Code repository for the paper

**AOI-Net: Structural Face AOI-Guided Eye-Gaze Track Representation Learning
for Autism Spectrum Disorder Detection**

AOI-Net is a model for **eye-tracking / AOI scanpath classification**. It
exploits the **AOI information** of each fixation to build an **AOI graph** and
jointly models **temporal and structural features** for classification. Given a
participant's sequence of fixations over Areas of Interest (AOIs), it predicts
a per-participant label — e.g. distinguishing autism (ASD) / typical groups
from their gaze behavior. Training uses k-fold cross-validation at the
participant level.


## Structure

```
Code/
├── config.json             # data schema: raw columns -> features / labels
├── dataloader.py           # loads / encodes / standardizes .xlsx scanpaths
├── requirements.txt
├── README.md
└── AOI_Net/
    ├── __init__.py
    ├── model.py            # AOI_Net (AOI-graph temporal+structural fusion) + blocks
    └── train.py            # parser-driven CLI entry point with k-fold CV
```

## Usage


```bash
python -m AOI_Net.train --data-dir /path/to/scanpaths --folds 5 --epochs 50
```


## Output

Per-fold best validation metrics (acc, sen, spe, f1, auc) are saved to
`--output` along with mean/std over all folds in `summary`. Training stops
early when validation accuracy does not improve for `--patience` epochs
(default 8; `0` disables early stopping).

