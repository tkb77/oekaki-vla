"""Accuracy of the canvas recognisers on expert drawings (reference for the evaluation metric)."""
import json, sys, glob
import cv2, numpy as np
from collections import Counter
from oekaki.eval.recognize import ClipRecognizer, geometric_classify
root = sys.argv[1] if len(sys.argv) > 1 else "data/touch/meta"
model = sys.argv[2] if len(sys.argv) > 2 else "openai/clip-vit-large-patch14"
files = sorted(glob.glob(f"{root}/job_*.json"))[:200]
metas = [json.load(open(f)) for f in files]
canv = [cv2.imread(f.replace(".json", ".png"))[..., ::-1] for f in files]
clip = ClipRecognizer(model)
cp = clip(canv)
geo = [geometric_classify(c) for c in canv]
y = [m["shape"] for m in metas]
acc_c = np.mean([p["pred"] == t for p, t in zip(cp, y)]); acc_g = np.mean([g["pred"] == t for g, t in zip(geo, y)])
print(f"n={len(y)} CLIP acc {acc_c:.3f}  geometric acc {acc_g:.3f}")
print("CLIP confusions", Counter((t, p["pred"]) for p, t in zip(cp, y) if p["pred"] != t).most_common(8))
print("GEO confusions", Counter((t, g["pred"]) for g, t in zip(geo, y) if g["pred"] != t).most_common(8))
json.dump(dict(model=model, n=len(y), clip_acc=float(acc_c), geometric_acc=float(acc_g)), open("artifacts/touch/recognizer_check.json", "w"), indent=2)
