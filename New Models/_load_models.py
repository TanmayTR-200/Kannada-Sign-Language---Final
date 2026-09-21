"""Load the three local model files and print their true input shapes/params."""
import os
import sys

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from GCN.gcn_model import GraphConv
from Transformer.transformer_model import PositionalEncodingLayer

MEMBERS = [
    ("TCN", "tcn.keras", None),
    ("GCN", "gcn.keras", {"GraphConv": GraphConv}),
    ("Transformer", "transformer.keras",
     {"PositionalEncodingLayer": PositionalEncodingLayer}),
]

for name, path, custom in MEMBERS:
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        print(f"{name:12s} MISSING {p}")
        continue
    try:
        m = tf.keras.models.load_model(p, custom_objects=custom)
        print(f"{name:12s} OK  input={m.input_shape}  params={m.count_params():,}  "
              f"layers={len(m.layers)}")
        # report the graph-conv variable names if present
        for layer in m.layers:
            vn = [w.name.split("/")[-1].split(":")[0] for w in layer.weights]
            if layer.__class__.__name__ == "GraphConv":
                print(f"             {layer.name}: vars={vn}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:12s} FAILED to load: {type(exc).__name__}: {str(exc)[:300]}")