"""Train the lightweight GCN model on the KSL landmark sequences.

Same protocol as TCN / the published models for a fair comparison:
StandardScaler-normalized flat features, identical stratified 70:10:20 split
(same 464-sample test set). Input is reshaped from flat (75, 195) to graph
layout (75, 65, 3) via common.flat_to_graph AFTER normalization.

Usage (from the 'New Models' directory):
    python GCN/train_gcn.py                # demo random data
    python GCN/train_gcn.py X.npy y.npy    # real dataset in repo root
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (NUM_CLASSES, MAX_SEQUENCE_LENGTH, NUM_FEATURES,
                    load_data, normalize_dataset, split_70_10_20, flat_to_graph,
                    center_on_shoulders, add_velocity)
from gcn_model import build_gcn, build_adjacency, GraphConv


def main():
    if len(sys.argv) == 3:
        X, y = load_data(sys.argv[1], sys.argv[2])
    else:  # demo run on random data — smoke-test the pipeline
        print("No dataset given: demo run on random data.")
        rng = np.random.default_rng(0)
        n = 600
        X = rng.random((n, MAX_SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
        y = rng.integers(0, NUM_CLASSES, n)

    # Published preprocessing: StandardScaler on the flat (75, 195) features
    # (stats persisted to scaler_stats.npz), then the same stratified
    # 70:10:20 split as the published models, then reshape to graph layout.
    X, _ = normalize_dataset(X)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_70_10_20(X, y)
    print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # v2 tuning: body-center the coordinates, then append per-frame velocity.
    # Input becomes (75, 65, 6): [x, y, z, dx, dy, dz] per joint.
    Xg_train = add_velocity(center_on_shoulders(flat_to_graph(X_train)))
    Xg_val = add_velocity(center_on_shoulders(flat_to_graph(X_val)))
    Xg_test = add_velocity(center_on_shoulders(flat_to_graph(X_test)))
    print(f"Graph input: {Xg_train.shape}")
    print(f"Adjacency: {build_adjacency().shape}, "
          f"edges w/ self-loops: {int(build_adjacency(normalized=False).sum())}")

    # Auto-resume: if a checkpoint from a previous (interrupted) run exists,
    # continue from it instead of starting over. The .keras checkpoint stores
    # weights + optimizer state, so training continues seamlessly.
    ckpt = 'gcn_best.keras'
    if os.path.exists(ckpt):
        print(f"Resuming from checkpoint: {ckpt}")
        model = tf.keras.models.load_model(ckpt, custom_objects={'GraphConv': GraphConv})
        csv_append = True
    else:
        model = build_gcn(Xg_train.shape[1:], NUM_CLASSES)
        model.summary()
        csv_append = False

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=12, factor=0.5),
        # Crash-resilience + monitoring for the long CPU run
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor='val_accuracy',
                                           save_best_only=True, verbose=0),
        tf.keras.callbacks.CSVLogger('gcn_history.csv', append=csv_append),
    ]
    model.fit(Xg_train, tf.keras.utils.to_categorical(y_train, NUM_CLASSES),
              validation_data=(Xg_val, tf.keras.utils.to_categorical(y_val, NUM_CLASSES)),
              batch_size=32, epochs=300, callbacks=callbacks, verbose=2)

    y_pred = model.predict(Xg_test).argmax(axis=1)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, digits=4))

    model.save('gcn.keras')
    print("Saved gcn.keras")


if __name__ == '__main__':
    main()
