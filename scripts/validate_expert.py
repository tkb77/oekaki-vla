"""the scripted expert must draw every shape under random paper height / disturbance / noise.

Usage: python scripts/validate_expert.py [--n 10]   -> artifacts/touch/expert_validation.{json,png}
"""
import argparse, json, time
import numpy as np, cv2
from oekaki.config import load_config, ROOT
from oekaki.env.drawing_env import DrawingEnv
from oekaki.data.rollout import run_expert_episode

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=123)
args = ap.parse_args()
cfg, dcfg = load_config(), load_config("configs/datagen.yaml")
out = ROOT / "artifacts" / "touch"; out.mkdir(parents=True, exist_ok=True)
env = DrawingEnv(cfg, render=False)
rng = np.random.default_rng(args.seed)
rows, tiles = [], []
t0 = time.time()
for shape in dcfg["shapes_to_use"]:
    for i in range(args.n):
        variant = ["clean", "noise", "disturb"][i % 3]
        ep = run_expert_episode(env, dcfg, shape, rng, noise=(variant == "noise"),
                                disturbance=(variant == "disturb"), record=False)
        m = ep["meta"]; m["variant"] = variant
        rows.append({k: m[k] for k in ["shape", "variant", "success", "coverage", "precision", "air_ratio",
                                        "over_press_ratio", "n_breaks", "n_steps", "paper_height_offset"]})
        if i < 3:
            tiles.append(cv2.resize(ep["canvas"], (192, 192)))
        if m["coverage"] < 0.95 or not m["success"]:
            print("WEAK", rows[-1])
print(f"{len(rows)} episodes in {time.time()-t0:.1f}s")
summary = {}
for shape in dcfg["shapes_to_use"]:
    r = [x for x in rows if x["shape"] == shape]
    summary[shape] = {k: float(np.mean([x[k] for x in r])) for k in ["success", "coverage", "precision", "air_ratio", "over_press_ratio", "n_steps"]}
    print(shape, {k: round(v, 3) for k, v in summary[shape].items()})
json.dump(dict(summary=summary, episodes=rows), open(out / "expert_validation.json", "w"), indent=2)
grid = np.vstack([np.hstack(tiles[i*3:(i+1)*3]) for i in range(len(tiles)//3)])
cv2.imwrite(str(out / "expert_canvases.png"), grid[..., ::-1])
