"""
make_figures.py -- generate paper-ready training-curve figures from the
per-epoch history CSVs (gcn_history.csv, tcn_history.csv, transformer_history.csv).

Outputs (PNG, 300 dpi) into this folder:
  fig_curves_gcn.png          accuracy + loss curves, GCN
  fig_curves_tcn.png          accuracy + loss curves, TCN
  fig_curves_transformer.png  accuracy + loss curves, Transformer
  fig_val_accuracy_all.png    combined val-accuracy comparison (paper figure)
"""
import os

import matplotlib

matplotlib.use("Agg")                      # headless: write PNGs, no window
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    ("GCN", "gcn_history.csv", "tab:green"),
    ("TCN", "tcn_history.csv", "tab:blue"),
    ("Transformer", "transformer_history.csv", "tab:orange"),
]


def load(name):
    path = os.path.join(HERE, name)
    df = pd.read_csv(path)
    if "epoch" in df.columns:              # CSVLogger adds an epoch column
        df["epoch"] = df["epoch"].astype(int) + 1
    else:
        df["epoch"] = range(1, len(df) + 1)
    return df


def single_figure(label, name, color):
    df = load(name)
    vals = df["val_accuracy"].to_numpy()
    best_pos = int(vals.argmax())
    best_epoch = int(df["epoch"].to_numpy()[best_pos])
    best_val = float(vals[best_pos])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(df["epoch"], df["accuracy"], color=color, alpha=0.45,
             label="train acc")
    ax1.plot(df["epoch"], df["val_accuracy"], color=color, lw=2,
             label="val acc")
    ax1.scatter([best_epoch], [best_val], color="black", zorder=5, s=28)
    ax1.annotate(f"best {best_val:.3f}\n(epoch {best_epoch})",
                 (best_epoch, best_val),
                 textcoords="offset points", xytext=(-10, -30), fontsize=8)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"{label} - accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(df["epoch"], df["loss"], color=color, alpha=0.45, label="train loss")
    ax2.plot(df["epoch"], df["val_loss"], color=color, lw=2, label="val loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title(f"{label} - loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, f"fig_curves_{label.lower()}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}  (epochs={len(df)}, best val acc={best_val:.4f})")


def combined_figure():
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name, color in RUNS:
        df = load(name)
        ax.plot(df["epoch"], df["val_accuracy"], color=color, lw=2, label=label)
        vals = df["val_accuracy"].to_numpy()
        best_pos = int(vals.argmax())
        best_epoch = int(df["epoch"].to_numpy()[best_pos])
        best_val = float(vals[best_pos])
        ax.scatter([best_epoch], [best_val], color=color, s=30)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Validation accuracy during training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_val_accuracy_all.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for label, name, color in RUNS:
        single_figure(label, name, color)
    combined_figure()
    print("ALL FIGURES DONE")
