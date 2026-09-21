"""Verify the three local model files + dump the published stacking notebook cells."""
import os
import io
import json
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)

print("=" * 70)
print("MODEL FILE CHECK")
print("=" * 70)
for name in ["gcn.keras", "tcn.keras", "transformer.keras"]:
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        print(f"{name}: MISSING")
        continue
    z = zipfile.ZipFile(p)
    names = z.namelist()
    cfg = z.read("config.json").decode("utf-8", "ignore")
    has_edge = "edge_mask" in cfg
    # variable names live in the weights manifest
    wname = [n for n in names if "weights" in n.lower()]
    wtxt = ""
    if wname:
        try:
            wtxt = z.read(wname[0]).decode("utf-8", "ignore")
        except Exception:
            wtxt = ""
    print(f"{name}: size={os.path.getsize(p):,}")
    print(f"   config-layer-names: gc_64 in cfg={'gc_64' in cfg}, "
          f"edge_mask in cfg={has_edge}, edge_mask in weights={('edge_mask' in wtxt)}")
    print(f"   zip entries: {names[:6]}{' ...' if len(names) > 6 else ''}")

print()
print("=" * 70)
print("PUBLISHED STACKING NOTEBOOK (code cells)")
print("=" * 70)
nb_path = os.path.join(REPO, "Ensemble Learning", "Stacking Ensemble.ipynb")
if not os.path.exists(nb_path):
    print("not found:", nb_path)
else:
    nb = json.load(io.open(nb_path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print("total code cells:", len(cells))
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if not src.strip():
            continue
        print(f"\n----- CELL {i} -----")
        print(src[:3000])
