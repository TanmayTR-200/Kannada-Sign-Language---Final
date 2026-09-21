"""HIGHEST-RISK ITEM: verify both new models convert to TFLite cleanly
(incl. full-integer quantization, which the Flutter app needs) BEFORE any
full training. Trains each model for 1 epoch on random data, then converts.

Usage (from the 'New Models' directory):
    python test_tflite.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (NUM_CLASSES, MAX_SEQUENCE_LENGTH, NUM_FEATURES,
                    normalize_dataset, center_on_shoulders, add_velocity,
                    add_velocity_flat)
from TCN.tcn_model import build_tcn
from GCN.gcn_model import build_gcn


def smoke_test(name, model, X):
    print(f"\n=== {name}: 1-epoch train ===")
    y = tf.keras.utils.to_categorical(
        np.random.randint(0, NUM_CLASSES, len(X)), NUM_CLASSES)
    model.fit(X, y, batch_size=32, epochs=1, verbose=2)

    def rep():
        for i in range(0, min(len(X), 256), 32):
            yield [X[i:i + 32].astype(np.float32)]

    converters = {
        'dynamic-range': tf.lite.TFLiteConverter.from_keras_model(model),
        'full-int-int-io': tf.lite.TFLiteConverter.from_keras_model(model),
        'float16': tf.lite.TFLiteConverter.from_keras_model(model),
    }
    converters['full-int-int-io'].target_spec.supported_ops = \
        [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converters['full-int-int-io'].optimizations = [tf.lite.Optimize.DEFAULT]
    converters['full-int-int-io'].representative_dataset = rep
    converters['full-int-int-io'].inference_input_type = tf.int8
    converters['full-int-int-io'].inference_output_type = tf.int8
    converters['dynamic-range'].optimizations = [tf.lite.Optimize.DEFAULT]
    converters['float16'].optimizations = [tf.lite.Optimize.DEFAULT]
    converters['float16'].target_spec.supported_types = [tf.float16]

    for tag, conv in converters.items():
        try:
            tflite = conv.convert()
            path = os.path.join(os.path.dirname(__file__),
                                f'{name}_{tag}.tflite')
            with open(path, 'wb') as f:
                f.write(tflite)
            print(f"  OK  {tag}: {len(tflite) / 1024:.1f} KB -> {path}")
        except Exception as e:
            print(f"  FAIL {tag}: {type(e).__name__}: {e}")


def main():
    rng = np.random.default_rng(0)
    X = rng.random((256, MAX_SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
    # Standardize the synthetic data so the int8 representative dataset lives
    # in the same value range the deployed model will actually see.
    X, _ = normalize_dataset(X, save_path=None)
    X_tcn = add_velocity_flat(X)
    smoke_test('tcn', build_tcn(), X_tcn)
    Xg = add_velocity(center_on_shoulders(
        X.reshape(len(X), MAX_SEQUENCE_LENGTH, 65, 3)))
    smoke_test('gcn', build_gcn(), Xg)


if __name__ == '__main__':
    main()
