"""Lightweight ST-GCN-style model for dynamic KSL recognition.

Treats the 65 MediaPipe landmarks (23 pose + 21 left hand + 21 right hand)
as a graph: joints = nodes, anatomical connections = edges. Input is
(75, 65, 6): per-joint [x, y, z] coordinates (body-centered) + per-frame
velocity [dx, dy, dz], derived from the flat (75, 195) sequences.

The custom GraphConv layer uses tf.einsum (A @ X @ W) — see ../test_tflite.py
for the early TFLite-conversion smoke test (highest-risk item).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Dropout,
                                     GlobalAveragePooling2D, Dense, ReLU)
from tensorflow.keras.optimizers import Adam

from common import N_NODES, N_POSE, N_HAND  # 65, 23, 21


# ---------------------------------------------------------------- adjacency
# MediaPipe landmark index groups (0-based, per MediaPipe Holistic docs)
# NOTE: the pose block keeps MediaPipe landmarks 0..22 only (23 nodes), so the
# hips (pose lm 23/24) are NOT in the graph — graph nodes 23/24 are the LEFT
# HAND's wrist/index landmarks. Shoulder->hip / hip->hip edges from an older
# revision therefore wired the right shoulder to the left index finger and
# have been removed (face nodes 0..10 intentionally carry self-loops only).
_POSE_EDGES = [
    (11, 12),                      # shoulders
    (11, 13), (13, 15),            # left arm
    (12, 14), (14, 16),            # right arm
]
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]
_LHAND_BASE = N_POSE                # 23
_RHAND_BASE = N_POSE + N_HAND       # 44


def build_adjacency(normalized=True):
    """65x65 adjacency matrix (self-loops included, symmetric normalization)."""
    A = np.eye(N_NODES, dtype=np.float32)

    def add(base, edges):
        for i, j in edges:
            A[base + i, base + j] = 1.0
            A[base + j, base + i] = 1.0

    add(_LHAND_BASE, _HAND_EDGES)
    add(_RHAND_BASE, _HAND_EDGES)
    for i, j in _POSE_EDGES:
        A[i, j] = A[j, i] = 1.0

    # Hand-to-arm links: hand wrist (hand lm 0) to the same side's pose wrist
    # and elbow (pose lm: 13=left elbow, 15=left wrist, 14=right elbow, 16=right wrist)
    A[_LHAND_BASE + 0, 15] = A[15, _LHAND_BASE + 0] = 1.0   # left hand wrist <-> left pose wrist
    A[_LHAND_BASE + 0, 13] = A[13, _LHAND_BASE + 0] = 1.0   # left hand wrist <-> left elbow
    A[_RHAND_BASE + 0, 16] = A[16, _RHAND_BASE + 0] = 1.0   # right hand wrist <-> right pose wrist
    A[_RHAND_BASE + 0, 14] = A[14, _RHAND_BASE + 0] = 1.0   # right hand wrist <-> right elbow

    if normalized:  # D^-1/2 A D^-1/2
        deg = A.sum(axis=1)
        dinv = 1.0 / np.sqrt(deg)
        A = A * dinv[:, None] * dinv[None, :]
    return A.astype(np.float32)


# ------------------------------------------------------------- custom layer
class GraphConv(tf.keras.layers.Layer):
    """Graph convolution on (batch, T, V, in_ch):
    out[..., u] = sum_v A_eff[v, u] * X[..., v] * W, where
    A_eff = A + M and M is a per-layer learnable 65x65 edge mask (ST-GCN
    style). M starts at zeros (pure anatomical adjacency) so training begins
    from the correct skeleton, but the model can then strengthen/weaken
    connections — e.g. learn to trust the hand regions more. M is an ordinary
    trainable weight, so TFLite conversion still needs no custom operators."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        self.W = self.add_weight(name='W', shape=(in_ch, self.units),
                                 initializer='glorot_uniform', trainable=True)
        self.edge_mask = self.add_weight(name='edge_mask',
                                         shape=(N_NODES, N_NODES),
                                         initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, T, V, in_ch). A is baked in as a constant so TFLite
        # conversion needs no custom-operator wiring.
        A = tf.constant(build_adjacency())                     # (V, V)
        A_eff = A + self.edge_mask
        y = tf.einsum('btki,kv->btvi', inputs, A_eff)          # aggregate joints
        return tf.tensordot(y, self.W, axes=[[-1], [0]])       # (b, t, v, u)

    def get_config(self):
        cfg = super().get_config()
        cfg['units'] = self.units
        return cfg


def gcn_st_block(x, channels, temporal_kernel=5, dropout_rate=0.3):
    """ST-GCN-style block: graph conv -> temporal conv ((kernel_t, 1) Conv2D)
    -> ReLU, with a residual connection. Layout stays (batch, T, V, C)."""
    y = GraphConv(channels, name=f'gc_{channels}')(x)
    y = Conv2D(channels, (temporal_kernel, 1), padding='same',
               name=f'tc_{channels}')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Dropout(dropout_rate)(y)
    if x.shape[-1] != channels:
        x = Conv2D(channels, (1, 1), padding='same')(x)
    return tf.keras.layers.Add()([x, y])


def build_gcn(input_shape=(75, 65, 6), num_classes=33, dropout_rate=0.4):
    inputs = Input(shape=input_shape)
    x = inputs
    for ch in (64, 128, 256):
        x = gcn_st_block(x, ch, dropout_rate=dropout_rate)
    x = GlobalAveragePooling2D()(x)   # pools over T and V -> (channels,)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='LightGCN')
    # lr=1e-3: graph convs aggregate neighbors (signal dilution), so they
    # converge much slower than RNNs/TCNs at 1e-4. Architecture-specific
    # tuning, not a protocol change (each block still gets the same
    # optimizer family, batch size and early stopping as the other models).
    # Label smoothing 0.1 fights the overconfident memorisation seen in v1
    # (93.8% train vs 80.6% val).
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = build_gcn()
    model.summary()
