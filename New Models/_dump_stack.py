"""Dump the published stacking-ensemble notebook (code + stdout outputs)."""
import io
import json
import os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                  "Ensemble Learning", "Stacking Ensemble.ipynb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_stack_dump.txt")

nb = json.load(io.open(NB, encoding="utf-8"))
lines = []
for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if not src.strip():
        continue
    lines.append("----- CELL %d -----" % i)
    lines.append(src)
    for o in cell.get("outputs", []):
        txt = "".join(o.get("text", []))
        if not txt and o.get("data", {}).get("text/plain"):
            txt = "".join(o["data"]["text/plain"])
        if txt.strip():
            lines.append("  >>> OUTPUT:")
            lines.append(txt)
    lines.append("")

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT, len(lines), "lines")