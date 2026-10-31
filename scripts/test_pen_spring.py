"""Tests for the spring pen (static press, line tracing, stability + rendering).

Usage: python scripts/test_pen_spring.py  (writes figures to artifacts/pen_tests/)
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oekaki.config import load_config, ROOT
from oekaki.env.drawing_env import DrawingEnv
from oekaki.expert.ik import TipCommander

OUT = ROOT / "artifacts" / "pen_tests"


def move(env, cmd, target, steps):
    obs = None
    for _ in range(steps):
        obs = env.step(cmd(target))
    return obs


def m1_static(env, cmd, cfg):
    """Command the rest tip 3 mm below the paper surface; expect compression ~3 mm, force ~ k*3 mm."""
    env.reset()
    cmd.reset(env.joint_pos())
    cx, cy = env.data.mocap_pos[env.paper_mocap, :2]
    z = env.paper_z
    move(env, cmd, [cx, cy, z + 0.02], 30)
    for i in range(30):
        move(env, cmd, [cx, cy, z + 0.02 - (0.023 * (i + 1) / 30)], 1)
    comps, forces = [], []
    import mujoco
    for _ in range(60):
        env.step(cmd([cx, cy, z - 0.003]))
        comps.append(env.compression)
        f = 0.0
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            if {c.geom1, c.geom2} == {env.tip_geom, env.canvas_geom}:
                ft = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, i, ft)
                f += ft[0]
        forces.append(f)
    comp = float(np.mean(comps[-30:]))
    force = float(np.mean(forces[-30:]))
    k = cfg["pen"]["stiffness"]
    expected_f = k * (comp - cfg["pen"]["springref"])
    res = dict(compression_mm=comp * 1e3, contact_force_N=force, spring_force_N=expected_f,
               k_times_3mm=k * 0.003, compression_std_mm=float(np.std(comps[-30:]) * 1e3))
    res["pass"] = abs(comp - 0.003) < 0.0006 and abs(force - k * 0.003) < 0.25 * k * 0.003
    return res, comps


def m2_trace(env, cmd, cfg, err_amp=0.002):
    """Draw a 10 cm line with a +/-2 mm sinusoidal height error; contact switch must stay on."""
    env.reset()
    cmd.reset(env.joint_pos())
    cx, cy = env.data.mocap_pos[env.paper_mocap, :2]
    z = env.paper_z
    x0, x1 = cx - 0.05, cx + 0.05
    move(env, cmd, [x0, cy, z + 0.02], 40)
    for i in range(30):
        move(env, cmd, [x0, cy, z + 0.02 - 0.023 * (i + 1) / 30], 1)
    n = int(0.10 / 0.04 * env.fps)  # 4 cm/s
    contacts, comps = [], []
    for i in range(n + 1):
        s = i / n
        dz = err_amp * np.sin(2 * np.pi * 2.5 * s)
        env.step(cmd([x0 + (x1 - x0) * s, cy, z - 0.003 + dz]))
        contacts.append(env.contact)
        comps.append(env.compression)
    move(env, cmd, [x1, cy, z + 0.03], 20)
    mask = env.canvas.ink_mask()
    rows, cols = np.nonzero(mask)
    res = dict(contact_ratio=float(np.mean(contacts)), min_compression_mm=float(np.min(comps) * 1e3),
               max_compression_mm=float(np.max(comps) * 1e3),
               ink_cols=[int(cols.min()), int(cols.max())] if len(cols) else None,
               ink_rows=[int(rows.min()), int(rows.max())] if len(rows) else None)
    # line runs along +x -> must be horizontal in canvas_img
    exp_px = env.canvas.to_px(np.array([[-0.05, 0.0], [0.05, 0.0]]))
    res["expected_cols"] = exp_px[:, 0].round().astype(int).tolist()
    res["pass"] = res["contact_ratio"] == 1.0 and res["ink_cols"] is not None and \
        abs(res["ink_cols"][0] - exp_px[0, 0]) < 8 and abs(res["ink_cols"][1] - exp_px[1, 0]) < 8 and \
        (res["ink_rows"][1] - res["ink_rows"][0]) < 15
    return res, comps


def m3_stability_render(env, cmd, cfg):
    """No bouncing while in contact; ink visible in the offscreen camera images."""
    env.reset()
    cmd.reset(env.joint_pos())
    before = env.render("top").astype(np.int32)
    cx, cy = env.data.mocap_pos[env.paper_mocap, :2]
    z = env.paper_z
    move(env, cmd, [cx, cy + 0.03, z + 0.02], 40)
    for i in range(30):
        move(env, cmd, [cx, cy + 0.03, z + 0.02 - 0.023 * (i + 1) / 30], 1)
    comps, touch = [], []
    n = int(2 * np.pi * 0.03 / 0.04 * env.fps)
    for i in range(n + 1):  # a circle of r = 3 cm
        a = 2 * np.pi * i / n
        obs = env.step(cmd([cx + 0.03 * np.sin(a), cy + 0.03 * np.cos(a), z - 0.003]))
        comps.append(env.compression)
        touch.append(obs["touch_frac"])
    move(env, cmd, [cx, cy + 0.03, z + 0.03], 20)
    after = env.render("top").astype(np.int32)
    wrist = env.render("wrist")
    changed = int((np.abs(after - before).sum(-1) > 60).sum())
    c = np.array(comps[10:])
    # bounce detection: compression should stay within a band, without lift-off
    res = dict(compression_mean_mm=float(c.mean() * 1e3), compression_std_mm=float(c.std() * 1e3),
               liftoffs=int((c < cfg["pen"]["contact_threshold"]).sum()),
               substep_touch_frac=float(np.mean(touch[10:])),
               top_cam_changed_px=changed, canvas_ink_px=int(env.canvas.ink_mask().sum()))
    res["pass"] = res["liftoffs"] == 0 and res["compression_std_mm"] < 0.5 and changed > 200 and \
        res["substep_touch_frac"] > 0.99
    cv2.imwrite(str(OUT / "m3_top_after.png"), after.astype(np.uint8)[..., ::-1])
    cv2.imwrite(str(OUT / "m3_wrist_after.png"), wrist[..., ::-1])
    cv2.imwrite(str(OUT / "m3_canvas.png"), env.canvas.img[..., ::-1])
    return res, comps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    env = DrawingEnv(cfg, cam_size=256)
    cmd = TipCommander(env.model, cfg["wrist_roll_cmd"], cfg["gripper_cmd"])
    results = {}
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, (name, fn) in zip(axs, [("M1_static", m1_static), ("M2_trace", m2_trace),
                                    ("M3_stability", m3_stability_render)]):
        res, comps = fn(env, cmd, cfg)
        results[name] = res
        ax.plot(np.arange(len(comps)) / env.fps, np.array(comps) * 1e3)
        ax.axhline(cfg["pen"]["contact_threshold"] * 1e3, color="r", ls="--", lw=0.8, label="threshold")
        ax.set_title(f"{name} ({'PASS' if res['pass'] else 'FAIL'})")
        ax.set_xlabel("time [s]"); ax.set_ylabel("compression [mm]"); ax.legend()
        print(name, json.dumps(res))
    fig.tight_layout(); fig.savefig(OUT / "pen_spring_tests.png", dpi=120)
    json.dump(results, open(OUT / "pen_spring_tests.json", "w"), indent=2)
    env.close()
    ok = all(r["pass"] for r in results.values())
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
