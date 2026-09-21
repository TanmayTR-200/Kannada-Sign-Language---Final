import json

p = (r"C:\Users\Dell\Downloads\Dynamic_Kannada_Sign_Language_Recognition_major_project-main"
     r"\Dynamic_Kannada_Sign_Language_Recognition_major_project-main"
     r"\Ensemble Learning\Stacking Ensemble.ipynb")

nb = json.load(open(p, encoding="utf-8"))
out = []
for i, c in enumerate(nb.get("cells", [])):
    if c.get("cell_type") != "code":
        continue
    src = "".join(c.get("source", []))
    if not src.strip():
        continue
    out.append(f"# ============ CODE CELL {i} ============")
    out.append(src)
    for o in c.get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain")
        if t:
            out.append("# ---- output ----")
            out.append("".join(t) if isinstance(t, list) else str(t))

open("_pub_stack.txt", "w", encoding="utf-8").write("\n".join(out))
print("cells:", len(out))
