"""Encoder-only Transformer for dynamic KSL recognition.

Architecture kept IDENTICAL to the published model: fixed sinusoidal
positional encoding + 4 pre-LN encoder blocks (head_size 64, 4 heads,
FFN 128, dropout 0.3, L2 1e-4) + global-average-pooling head.
Input: (75, 195) standardized landmark sequences. Adam lr=5e-4 and
categorical cross-entropy — same as the published training.

Only difference from the notebook: the positional-encoding layer is
registered as a Keras-3 serializable object, so the saved model loads
cleanly in any Keras 3 environment (no custom_objects wiring needed).
"""

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

try:  # Keras 3.x on some TF versions
    _register_serializable = tf.keras.saving.register_keras_serializable
except AttributeError:  # other Keras 3.x versions expose it via utils
    _register_serializable = tf.keras.utils.register_keras_serializable


@_register_serializable(package="ksl")
class PositionalEncodingLayer(layers.Layer):
    """Fixed sinusoidal positional encoding (published implementation)."""

    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(seq_len, d_model)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _positional_encoding(self, seq_len, d_model):
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle = pos * angle
        sines = tf.sin(angle[:, 0::2])
        coses = tf.cos(angle[:, 1::2])
        encoding = tf.concat([sines, coses], axis=-1)
        return tf.expand_dims(encoding, 0)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.3):
    """Pre-LN transformer encoder block (published implementation)."""
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    _mha_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size,
                                   dropout=dropout)(x, x)
    x = _mha_out[0] if isinstance(_mha_out, tuple) else _mha_out
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([inputs, x])                     # residual

    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(ff_dim, activation="gelu",
                     kernel_regularizer=regularizers.l2(1e-4))(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(inputs.shape[-1])(y)
    y = layers.Dropout(dropout)(y)
    return layers.Add()([x, y])                       # residual


def build_transformer(input_shape=(75, 195), num_classes=33):
    inp = layers.Input(shape=input_shape)
    x = PositionalEncodingLayer(input_shape[0], input_shape[1])(inp)

    for _ in range(4):
        x = transformer_encoder(x, head_size=64, num_heads=4,
                                ff_dim=128, dropout=0.3)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out, name="Transformer")
    model.compile(optimizer=Adam(learning_rate=5e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_transformer().summary()