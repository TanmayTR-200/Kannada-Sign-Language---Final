"""TCN (Temporal Convolutional Network) for dynamic KSL recognition.

v2: input carries motion features — (75, 390) = 195 coordinates + 195
frame-to-frame deltas. Dilations 1->2->4->8->16 (receptive field ~62 frames)
and a dual global-average + global-max pooling head; label smoothing 0.1.
Same training protocol as the other models (Adam lr=1e-4, categorical CE,
batch 32, early stopping). Bai et al., 2018.
"""

import os
import random
import numpy as np
# Deterministic seeding for reproducibility (TF 2.21 vs 2.15 RNG drift fix)
os.environ["TF_DETERMINISTIC_OP"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Dropout,
                                     GlobalAveragePooling1D, Dense, Activation)
from tensorflow.keras.optimizers import Adam


def causal_conv_block(x, filters, kernel_size, dilation, dropout_rate=0.4):
    """Causal dilated conv + BN + ReLU + Dropout, with residual connection."""
    # Causal padding: pad only on the left so output length == input length
    pad = (kernel_size - 1) * dilation
    y = tf.keras.layers.ZeroPadding1D(padding=(pad, 0))(x)
    y = Conv1D(filters, kernel_size, dilation_rate=dilation, padding='valid')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(dropout_rate)(y)

    # Residual: 1x1 conv if channel counts differ
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    return Activation('relu')(tf.keras.layers.Add()([x, y]))


def build_tcn(input_shape=(75, 390), num_classes=33,
              filters=64, kernel_size=3, dropout_rate=0.4):
    inputs = Input(shape=input_shape)
    x = inputs
    # Dilated residual blocks, dilation 1 -> 2 -> 4 -> 8 -> 16 (RF ~62 frames)
    for d in (1, 2, 4, 8, 16):
        x = causal_conv_block(x, filters, kernel_size, d, dropout_rate)
    # Dual pooling: average captures the overall flow, max keeps transient
    # peak poses that averaging blurs.
    x = tf.keras.layers.Concatenate()([
        GlobalAveragePooling1D()(x),
        tf.keras.layers.GlobalMaxPooling1D()(x)])
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='TCN')
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = build_tcn()
    model.summary()
