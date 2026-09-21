"""Dump the code cells of the published Stacking Ensemble notebook."""
import json
import sys

p = sys.argv[1]
nb = json.load(open(p, encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if not src.strip():
        continue
    print(f"\n{'=' * 70}\nCELL {i}\n{'=' * 70}")
    print(src)
    outs = c.get('outputs') or []
    for o in outs:
        txt = ''.join(o.get('text', [])) if 'text' in o else ''
        if not txt and 'data' in o:
            d = o['data']
            txt = ''.join(d.get('text/plain', []))
        if txt.strip():
            print(f"--- output ---\n{txt[:1500]}")