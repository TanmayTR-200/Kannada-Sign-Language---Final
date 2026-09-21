"""Shared data utilities for the new models (TCN / GCN).

Feature layout per frame (195 dims) — matches Feature Extraction/code.ipynb:
    [  0: 69]  pose landmarks 0..22        (x, y, z) each
    [ 69:132]  left hand landmarks 0..20   (x, y, z) each
    [132:195]  right hand landmarks 0..20  (x, y, z) each

Preprocessing matches the published LSTM/BiLSTM/Transformer notebooks:
  * features standardized per coordinate with StandardScaler, fitted on the
    full 2319-sample dataset (exactly what the notebooks do) — the mean/std
    are persisted to scaler_stats.npz so the app / feature-extraction server
    can apply the identical transform at inference time;
  * stratified sklearn 70:10:20 split with random_state=42, i.e. the SAME
    464-sample test set the published models were evaluated on, so new vs old
    accuracy tables are apples-to-apples.
"""

import os

import numpy as np
from sklearn.preprocessing import StandardScaler

CLASSES = ['Afternoon', 'Apple', 'April', 'August', 'Banana', 'Day', 'December',
           'Evening', 'Febraury', 'Friday', 'Grapes', 'January', 'July', 'June',
           'March', 'May', 'Monday', 'Morning', 'Night', 'November', 'October',
           'Orange', 'Rainy', 'Saturday', 'September', 'Summer', 'Sunday',
           'Thursday', 'Tuesday', 'Valencia_Orange', 'Watermelon', 'Wednesday',
           'Winter']
NUM_CLASSES = len(CLASSES)  # 33

MAX_SEQUENCE_LENGTH = 75
NUM_FEATURES = 195

N_POSE = 23
N_HAND = 21
N_NODES = N_POSE + 2 * N_HAND  # 65 joints -> graph model input (75, 65, 3)

POSE_SLICE = slice(0, N_POSE * 3)
LHAND_SLICE = slice(N_POSE * 3, N_POSE * 3 + N_HAND * 3)
RHAND_SLICE = slice(N_POSE * 3 + N_HAND * 3, NUM_FEATURES)

SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'scaler_stats.npz')


def load_data(x_path, y_path):
    """Load features/labels; supports one-hot (N,33) or integer (N,) labels."""
    X = np.load(x_path)
    y = np.load(y_path)
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    assert X.shape[1:] == (MAX_SEQUENCE_LENGTH, NUM_FEATURES), \
        f"Unexpected X shape {X.shape}; expected (*, {MAX_SEQUENCE_LENGTH}, {NUM_FEATURES})"
    return X, y.astype(np.int32)


def flat_to_graph(X):
    """(N, 75, 195) flat frame vectors -> (N, 75, 65, 3) joint-wise layout."""
    N, T, _ = X.shape
    Xr = X.reshape(N, T, N_NODES, 3)
    # Zero out missing landmarks that were padded with 0s (already 0 -> stays 0)
    return Xr.astype(np.float32)


def center_on_shoulders(Xg):
    """Per-frame translation centering: subtract the midpoint of the two
    shoulder joints (graph nodes 11/12 = MediaPipe pose shoulders) from every
    joint in every frame. Makes signs invariant to where the person sits
    relative to the camera — removes a large nuisance factor from the raw
    coordinates and typically boosts skeleton-model accuracy by several points."""
    anchor = Xg[:, :, 11:13, :].mean(axis=2, keepdims=True)
    return (Xg - anchor).astype(np.float32)


def add_velocity(Xg):
    """(N, 75, V, 3) joint coordinates -> (N, 75, V, 6): [x, y, z, dx, dy, dz].
    Frame-to-frame deltas are the strongest motion cue in skeleton-based
    recognition (static coordinates alone can't tell 'hand moving up' from
    'hand held still'). The first frame's deltas are zeros."""
    v = np.zeros_like(Xg)
    v[:, 1:] = Xg[:, 1:] - Xg[:, :-1]
    return np.concatenate([Xg, v], axis=-1).astype(np.float32)


def add_velocity_flat(X):
    """(N, 75, F) flat frame features -> (N, 75, 2F): original features plus
    frame-to-frame deltas, giving sequence models the same motion cues the
    graph model got. First frame's deltas are zeros."""
    v = np.zeros_like(X)
    v[:, 1:] = X[:, 1:] - X[:, :-1]
    return np.concatenate([X, v], axis=-1).astype(np.float32)


def fit_normalizer(X):
    """StandardScaler over all timesteps/samples: (N, 75, 195) -> fit on (N*75, 195)."""
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, X.shape[-1]))
    return scaler


def apply_normalizer(X, scaler):
    """Apply a fitted scaler, returning float32 with the original 3D shape."""
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)


def normalize_dataset(X, normalize_dataset=True, save_path=None):
    """Published protocol: fit StandardScaler on the FULL dataset (train+val+test,
    exactly as the LSTM/BiLSTM/Transformer notebooks do), transform, and persist
    mean/std so the deployment pipeline can normalize identically.
    Pass save_path=None to skip persisting (e.g. in smoke tests)."""
    scaler = fit_normalizer(X)
    Xn = apply_normalizer(X, scaler)
    if save_path:
        np.savez(save_path, mean=np.asarray(scaler.mean_),
                          scale=np.asarray(scaler.scale_))
    return Xn, scaler


def split_70_10_20(X, y, seed=42):
    """The exact split of the published models: stratified sklearn
    train_test_split, 20% test then 12.5% of the remainder for validation
    (=> 70:10:20), random_state=42. Yields Train (1623, 75, 195),
    Val (232, 75, 195), Test (464, 75, 195) with the same test-set membership
    as the LSTM/BiLSTM/Transformer runs -> comparable accuracy tables."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.125, stratify=y_tr, random_state=seed)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)
