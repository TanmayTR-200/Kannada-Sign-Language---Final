# Stacking Ensemble Approach for Dynamic Kannada Sign Language Recognition

Word-level Kannada Sign Language (KSL) recognition from skeletal landmark
sequences. This repository replaces the earlier recurrent backbone with two
new architectures, a **Graph Convolutional Network (GCN)** and a **Temporal
Convolutional Network (TCN)**, alongside a retrained **Transformer**, all
trained under one identical protocol, and a stacking ensemble built on top
of them.

## Results (shared 464-sample stratified test set, 33 classes)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **GCN** | **96.55%** | 96.84% | 96.56% | 96.55% |
| Transformer | 91.59% | 92.30% | 91.60% | 91.40% |
| TCN | 93.97% | 94.38% | 93.94% | 93.86% |
| Stacking ensemble (GCN+TCN+TF) | 95.69% | 96.01% | 95.71% | 95.63% |

## Repository structure

```
├── New Models/               <- the new work
│   ├── GCN/                  <- graph convolutional model + training script
│   ├── TCN/                  <- dilated causal convolutional model + training script
│   ├── Transformer/          <- encoder-only transformer + training script
│   ├── build_ensemble.py     <- reproduces all ensemble results
│   ├── common.py             <- shared preprocessing (StandardScaler, split)
│   ├── scaler_stats.npz      <- scaler parameters (required at inference)
│   ├── *.keras               <- trained models
│   ├── *.tflite              <- TensorFlow Lite conversions (on-device)
│   ├── make_figures.py       <- generates the training-curve figures
│   └── fig_*.png             <- paper figures
├── Deep Learning Models/     <- earlier architecture notebooks (reference)
├── Ensemble Learning/        <- original stacking-ensemble notebook
├── Feature Extraction/       <- MediaPipe landmark extraction notebook
└── Dataset Description/      <- dataset documentation
```

## Dataset

The landmark dataset consists of 2,319 sign videos across 33 classes
(fruits, months, weekdays, time/seasons), represented as 75 frames x 195
features per frame (x, y, z of 23 pose + 21 left-hand + 21 right-hand
MediaPipe landmarks). The two files, `X_195_75.npy` (~271 MB) and
`y_195_75.npy`, are hosted on Google Drive and are too large for GitHub;
request them from the authors or provide your own Drive copy.

Split: stratified 70:10:20 (1,623 / 232 / 464), fixed random seed, shared
by every model. Preprocessing: StandardScaler fitted on the full corpus;
the fitted parameters ship in `New Models/scaler_stats.npz`.

## Reproducing the results

Requires Python 3.10+ and TensorFlow 2.16+ (models are saved in Keras 3
format). No GPU is needed to verify; a GPU (e.g. free Colab T4) is
recommended for retraining.

```bash
pip install tensorflow numpy scikit-learn joblib pandas matplotlib

# 1. Evaluate all models and rebuild the stacking ensemble (~2 min on CPU)
python "New Models/build_ensemble.py" X_195_75.npy y_195_75.npy

# 2. Retrain any model from scratch (~10-60 min each on a Colab T4)
cd "New Models"
python GCN/train_gcn.py ../X_195_75.npy ../y_195_75.npy
python TCN/train_tcn.py ../X_195_75.npy ../y_195_75.npy
python Transformer/train_transformer.py ../X_195_75.npy ../y_195_75.npy

# 3. Regenerate the paper figures
python make_figures.py
```

## On-device deployment

All three models ship as TensorFlow Lite files (dynamic-range, float16,
and full-integer variants, 0.12 to 1.2 MB each; see
`New Models/test_tflite.py`). At inference time, raw landmarks must be
scaled with `scaler_stats.npz` before being fed to any model.

## Citation

If you use this work, please cite the corresponding paper, "A Stacking
Ensemble Approach for Dynamic Kannada Sign Language Recognition."

