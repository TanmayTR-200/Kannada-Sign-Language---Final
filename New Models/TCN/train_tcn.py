"""Train the TCN model on the KSL landmark sequences.

Same protocol as the published LSTM/BiLSTM/Transformer notebooks:
StandardScaler-normalized inputs (fitted on the full dataset, exactly like
the notebooks), identical stratified 70:10:20 split — the SAME 464-sample
test set the published models were evaluated on — Adam lr=1e-4, categorical
cross-entropy, batch 32, up to 200 epochs with early stopping, evaluated
with accuracy + F1.

Usage (from the 'New Models' directory):
    python TCN/train_tcn.py                # demo random data
    python TCN/train_tcn.py X.npy y.npy    # real dataset in repo root
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (NUM_CLASSES, MAX_SEQUENCE_LENGTH, NUM_FEATURES,
                    load_data, normalize_dataset, split_70_10_20,
                    add_velocity_flat)
from tcn_model import build_tcn


def main():
    if len(sys.argv) == 3:
        X, y = load_data(sys.argv[1], sys.argv[2])
    else:  # demo run on random data — smoke-test the pipeline
        print("No dataset given: demo run on random data.")
        rng = np.random.default_rng(0)
        n = 600
        X = rng.random((n, MAX_SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
        y = rng.integers(0, NUM_CLASSES, n)

    # Published preprocessing: StandardScaler fitted on the full dataset
    # (stats persisted to scaler_stats.npz for the deployment pipeline),
    # then the same stratified 70:10:20 split as the published models.
    # v2: append per-frame velocity -> (75, 390) input.
    X, _ = normalize_dataset(X)
    X = add_velocity_flat(X)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_70_10_20(X, y)
    print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # Auto-resume: continue from a previous (interrupted) run's checkpoint
    # instead of starting over. The .keras checkpoint stores weights +
    # optimizer state, so training continues seamlessly.
    ckpt = 'tcn_best.keras'
    if os.path.exists(ckpt):
        print(f"Resuming from checkpoint: {ckpt}")
        model = tf.keras.models.load_model(ckpt)
        csv_append = True
    else:
        model = build_tcn(X_train.shape[1:], NUM_CLASSES)
        model.summary()
        csv_append = False

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=12, factor=0.5),
        # Crash-resilience + monitoring for the long CPU run
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor='val_accuracy',
                                           save_best_only=True, verbose=0),
        tf.keras.callbacks.CSVLogger('tcn_history.csv', append=csv_append),
    ]
    model.fit(X_train, tf.keras.utils.to_categorical(y_train, NUM_CLASSES),
              validation_data=(X_val, tf.keras.utils.to_categorical(y_val, NUM_CLASSES)),
              batch_size=32, epochs=300, callbacks=callbacks, verbose=2)

    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=1)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred, digits=4))

    model.save('tcn.keras')
    print("Saved tcn.keras")


if __name__ == '__main__':
    main()
