"""Recompute the recogniser labels (CLIP and geometric) of saved evaluate.py results, e.g. after
changing the prompts or the geometric classifier.

Usage: python scripts/relabel_clip.py artifacts/eval/results_expert.json [...] [--no-clip]
"""
import json, sys
from pathlib import Path
import cv2
from oekaki.eval.recognize import ClipRecognizer, geometric_classify

files = [a for a in sys.argv[1:] if not a.startswith("--")]
clip = None if "--no-clip" in sys.argv else ClipRecognizer()
for f in files:
    f = Path(f)
    rows = json.load(open(f))
    canv = [cv2.imread(str(f.parent / r["agent"] / "canvas" / f"{r['cond']}_{r['shape']}_{r['ep']:02d}.png"))[..., ::-1] for r in rows]
    for r, c in zip(rows, canv):
        r["geo_pred"] = geometric_classify(c)["pred"]
    if clip is not None:
        for r, p in zip(rows, clip(canv)):
            r["clip_pred"], r["clip_probs"] = p["pred"], p["probs"]
    json.dump(rows, open(f, "w"), indent=1)
    acc = lambda k: sum(r.get(k) == r["shape"] for r in rows) / len(rows)
    print(f, "clip acc", acc("clip_pred"), "geo acc", acc("geo_pred"))
