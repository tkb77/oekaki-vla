"""Fast smoke tests: `uv run pytest -q` (or .venv/bin/python -m pytest -q)."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from oekaki.config import load_config


@pytest.fixture(scope="module")
def cfgs():
    return load_config(), load_config("configs/datagen.yaml")


def test_imports():
    import lerobot  # noqa: F401
    import mujoco  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401


def test_canvas_roundtrip():
    from oekaki.env.canvas import Canvas
    c = Canvas(0.15, 512)
    xy = np.array([[0.03, -0.02], [-0.07, 0.07]])
    assert np.allclose(c.to_m(c.to_px(xy)), xy)
    # +x -> right, +y -> up in the image
    assert c.to_px([0.05, 0])[0] > c.to_px([-0.05, 0])[0]
    assert c.to_px([0, 0.05])[1] < c.to_px([0, -0.05])[1]


def test_shapes_fit_canvas(cfgs):
    from oekaki.expert.shapes import SHAPES, make_shape
    cfg, dcfg = cfgs
    rng = np.random.default_rng(0)
    lim = cfg["canvas"]["size"] / 2
    for name in SHAPES:
        for _ in range(20):
            s = make_shape(name, rng, {**dcfg, "canvas": cfg["canvas"]})
            assert np.abs(np.concatenate(s["strokes"])).max() < lim


def test_ik_reaches_canvas_vertically(cfgs):
    from oekaki.env.scene import build_model
    from oekaki.expert.ik import PenIK
    cfg, _ = cfgs
    m = build_model(cfg)
    ik = PenIK(m)
    q = np.zeros(m.nq)
    q[:4] = [0, -0.5, 0.7, 0.7]
    cx, cy = cfg["canvas"]["center"]
    for x, y in [(cx - 0.07, -0.07), (cx + 0.07, 0.07), (cx, 0)]:
        qs, err = ik.solve([x, y, cfg["canvas"]["height"]], q, iters=200)
        _, axis = ik.fk(qs)
        assert err < 1e-3 and axis[2] > np.cos(np.radians(3))


def test_env_draws_and_renders(cfgs):
    from oekaki.env.drawing_env import DrawingEnv
    from oekaki.expert.ik import TipCommander
    cfg, _ = cfgs
    env = DrawingEnv(cfg, render=True, cam_size=64)
    obs = env.reset()
    assert obs["images"]["top"].shape == (64, 64, 3) and obs["contact"] == 0
    cmd = TipCommander(env.model)
    cmd.reset(env.joint_pos())
    cx, cy = env.data.mocap_pos[env.paper_mocap, :2]
    z = env.paper_z
    for i in range(60):
        obs = env.step(cmd([cx, cy, z + 0.02 - 0.023 * min(1.0, i / 40)]))
    assert obs["contact"] == 1 and 0.002 < obs["compression"] < 0.004
    assert env.canvas.ink_mask().sum() > 0
    env.close()


def test_expert_line_episode(cfgs):
    from oekaki.data.rollout import run_expert_episode
    from oekaki.env.drawing_env import DrawingEnv
    cfg, dcfg = cfgs
    env = DrawingEnv(cfg, render=False)
    ep = run_expert_episode(env, dcfg, "line", np.random.default_rng(1), noise=False, disturbance=True,
                            record=False)
    assert ep["meta"]["success"] and ep["meta"]["coverage"] > 0.95
