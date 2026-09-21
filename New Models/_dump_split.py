"""Dump the split / scaler related cells from the published notebooks."""
import io
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = [
    ("LSTM", os.path.join("Deep Learning Models", "LSTM", "LSTM.ipynb")),
    ("BiLSTM", os.path.join("Deep Learning Models", "BiLSTM", "code.ipynb")),
    ("Transformers", os.path.join("Deep Learning Models", "Transformers", "code.ipynb")),
    ("Stacking", os.path.join("Ensemble Learning", "Stacking Ensemble.ipynb")),
]

KEYS = ("train_test_split", "StandardScaler", "scaler.fit", "scaler.transform",
        "load_model", "accuracy_score", "LogisticRegression", "fit(", "test_size",
        "concatenate", "VotingClassifier", "voting")

for label, rel in TARGETS:
    path = os.path.join(REPO, rel)
    print("\n" + "=" * 78)
    print(f"### {label}: {rel}")
    print("=" * 78)
    if not os.path.exists(path):
        print("NOT FOUND")
        continue
    nb = json.load(io.open(path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if any(k in src for k in KEYS):
            print(f"\n----- {label} CELL {i} -----")
            print(src[:2200])
