"""Build the stacking ensemble for the conference paper.

Mirrors the published 'Ensemble Learning/Stacking Ensemble.ipynb' protocol:
each base model (TCN, GCN, Transformer) predicts 33-class probabilities on
the VALIDATION split; the concatenated 99-dim (3 x 33) vectors train a
multinomial logistic-regression meta-learner; the ensemble is evaluated
ONCE on the held-out 464-sample test set (same stratified 70:10:20 split,
random_state=42, same StandardScaler as every model).

Usage (from the 'New Models' directory):
    python build_ensemble.py                                  # default paths
    python build_ensemble.py X.npy y.npy                      # explicit dataset
"""

import os
import sys

import numpy as np
import tensorflow as tf

# Always resolve model/dataset paths relative to THIS file, so the script can
# be launched from any working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from common import (load_data, normalize_dataset, split_70_10_20,
                    add_velocity_flat, add_velocity, center_on_shoulders,
                    flat_to_graph)
from GCN.gcn_model import GraphConv
from Transformer.transformer_model import PositionalEncodingLayer


def main():
    if len(sys.argv) == 3:
        x_path, y_path = sys.argv[1], sys.argv[2]
    else:
        candidates = [
            os.path.join(HERE, "X_195_75.npy"),
            os.path.join(HERE, "..", "..", "X_195_75.npy"),
        ]
        x_path = next((c for c in candidates if os.path.exists(c)), None)
        if x_path is None:
            raise SystemExit(
                "Dataset not found. Searched:\n  " + "\n  ".join(candidates) +
                "\nEither place X_195_75.npy / y_195_75.npy two levels up from this "
                "script (original layout), or run:\n"
                "  python build_ensemble.py <X.npy> <y.npy>")
        y_path = x_path.replace("X_195_75.npy", "y_195_75.npy")
        print(f"Dataset found at: {x_path}")

    missing = [p for p in ("tcn.keras", "gcn.keras", "transformer.keras")
               if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Missing trained model file(s): " + ", ".join(missing) +
            "\nCopy them from Google Drive (New_Models_v2/New Models) into:\n"
            "  " + HERE)

    X, y = load_data(x_path, y_path)
    # Same StandardScaler as every model. Don't re-save the stats — the
    # deployed scaler_stats.npz is identical and stays untouched.
    X, _ = normalize_dataset(X, save_path=None)
    (X_tr, _), (X_va, y_va), (X_te, y_te) = split_70_10_20(X, y)

    # Each member gets EXACTLY the transform it was trained with.
    members = [
        ("TCN", "tcn.keras", add_velocity_flat),
        ("GCN", "gcn.keras",
         lambda X: add_velocity(center_on_shoulders(flat_to_graph(X)))),
        ("Transformer", "transformer.keras", lambda X: X),
    ]

    names, val_probs, test_probs, accs = [], [], [], []
    for name, path, prep in members:
        # Each member needs its own custom layers declared for deserialization.
        custom = {"GraphConv": GraphConv} if name == "GCN" else \
                 {"PositionalEncodingLayer": PositionalEncodingLayer} \
                 if name == "Transformer" else None
        model = tf.keras.models.load_model(path, custom_objects=custom)
        assert model.input_shape[1:] == prep(X_va).shape[1:], \
            f"{name}: model expects {model.input_shape[1:]}, " \
            f"prep produced {prep(X_va).shape[1:]}"
        p_val = model.predict(prep(X_va), batch_size=64, verbose=0)
        p_test = model.predict(prep(X_te), batch_size=64, verbose=0)
        acc = float((p_test.argmax(1) == y_te).mean())
        print(f"{name:12s} individual test accuracy: {acc:.4f}")
        names.append(name)
        accs.append(acc)
        val_probs.append(p_val)
        test_probs.append(p_test)

    meta_val = np.concatenate(val_probs, axis=1)     # (232, 99)
    meta_test = np.concatenate(test_probs, axis=1)   # (464, 99)
    print(f"Meta-features: val {meta_val.shape}, test {meta_test.shape}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score,
                                 classification_report)
    meta = LogisticRegression(max_iter=2000, C=1.0)
    meta.fit(meta_val, y_va)                          # published protocol: fit on val

    ens_pred = meta.predict(meta_test)
    acc = accuracy_score(y_te, ens_pred)
    f1w = f1_score(y_te, ens_pred, average="weighted")
    f1m = f1_score(y_te, ens_pred, average="macro")

    # Diagnostic: plain soft-voting (mean probability) — no meta-learner, so it
    # cannot overfit the 232-sample validation split. Cheap second reference
    # point for the paper.
    soft_pred = np.mean(test_probs, axis=0).argmax(1)
    acc_soft = accuracy_score(y_te, soft_pred)
    f1w_soft = f1_score(y_te, soft_pred, average="weighted")

    print("\n=== BASE MODELS (shared 464-sample test set) ===")
    for n, a in sorted(zip(names, accs), key=lambda t: -t[1]):
        print(f"  {n:12s} accuracy {a:.4f}")
    best = max(accs)

    print("\n=== STACKING ENSEMBLE (TCN + GCN + Transformer) ===")
    print(f"accuracy    : {acc:.4f}")
    print(f"weighted F1 : {f1w:.4f}")
    print(f"macro F1    : {f1m:.4f}")
    print(f"(soft-voting diagnostic: accuracy {acc_soft:.4f}, "
          f"weighted F1 {f1w_soft:.4f})")
    print(f"gain over best single model ({best:.4f}): "
          f"{acc - best:+.4f}")
    print(classification_report(y_te, ens_pred, digits=4))

    import joblib
    joblib.dump(meta, "meta-learner.pkl")
    with open("ensemble_members.txt", "w") as f:
        f.write("\n".join(names))
    print("Saved meta-learner.pkl + ensemble_members.txt")


if __name__ == "__main__":
    main()