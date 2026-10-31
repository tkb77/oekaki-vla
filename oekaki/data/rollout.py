"""Running one expert episode (used by data generation and the expert validation)."""
from __future__ import annotations

import numpy as np

from oekaki.env.drawing_env import DrawingEnv, EpisodeParams
from oekaki.expert.expert import DrawingExpert
from oekaki.expert.shapes import instruction, make_shape
from oekaki.eval.metrics import contact_metrics, coverage_metrics


def state_vector(obs: dict, with_contact: bool) -> np.ndarray:
    s = list(obs["joint_pos"])
    if with_contact:
        s += [float(obs["contact"]), obs["compression"] * 1000.0]  # compression in mm
    return np.asarray(s, dtype=np.float32)


def run_expert_episode(env: DrawingEnv, dcfg: dict, shape_name: str, rng: np.random.Generator,
                       noise: bool | None = None, disturbance: bool | None = None,
                       record: bool = True, shape: dict | None = None, task: str | None = None) -> dict:
    """shape/task: given strokes (paper-local metres) and instruction (goal-image episodes); default: a parametric shape."""
    ecfg, ncfg = dcfg["expert"], dcfg["noise"]
    if shape is None:
        shape = make_shape(shape_name, rng, {**dcfg, "canvas": env.cfg["canvas"]})
    if task is None:
        task = instruction(shape_name, rng)
    params: EpisodeParams = env.sample_params(rng, disturbance=disturbance)
    if noise is None:
        noise = bool(rng.random() < ncfg["prob"])
    obs = env.reset(params)
    paper_xy = env.data.mocap_pos[env.paper_mocap, :2].copy()
    strokes_world = [s + paper_xy for s in shape["strokes"]]
    expert = DrawingExpert(env, ecfg)
    expert.reset(strokes_world, obs)

    frames = []
    contacts, comps, phys, states_log = [], [], [], []
    dz = 0.0
    max_steps = int(dcfg["max_episode_s"] * env.fps)
    dist_set = False
    for t in range(max_steps):
        action = expert.act(obs)
        if params.disturbance is not None and not dist_set and expert.state == "DRAW":
            w = ncfg["disturbance_window"]
            params.disturbance.t_start = env.t + rng.uniform(*w) * expert.expected_draw_time()
            dist_set = True
        exec_action = action
        if noise:
            dz = ncfg["z_rho"] * dz + np.sqrt(1 - ncfg["z_rho"] ** 2) * ncfg["z_sigma"] * rng.standard_normal()
            if expert.state in ("DESCEND", "PRELOAD", "DRAW"):
                exec_action = expert.action_with_offset(dz)
        if record:
            frames.append(dict(images=obs.get("images"), joint_pos=obs["joint_pos"].astype(np.float32),
                               contact=int(obs["contact"]), compression=float(obs["compression"]),
                               action=action.astype(np.float32)))
        contacts.append(obs["contact"]); comps.append(obs["compression"]); phys.append(obs["physical_contact"])
        states_log.append(expert.state)
        obs = env.step(exec_action)
        if expert.done:
            break

    drawing_mask = np.array([s == "DRAW" for s in states_log])
    cov = coverage_metrics(env.canvas.img, shape["strokes"], env.cfg["canvas"]["size"],
                           line_width_px=env.cfg["canvas"]["line_width_px"])
    con = contact_metrics(np.array(contacts), np.array(comps), env.travel, drawing_mask)
    meta = dict(
        shape=shape_name, task=task, shape_params=shape["params"],
        strokes_local=[s.tolist() for s in shape["strokes"]],
        paper_height_offset=params.height_offset, paper_xy_offset=list(params.xy_offset),
        light_scale=params.light_scale, paper_tint=list(params.paper_tint),
        disturbance=None if params.disturbance is None else dict(
            t_start=params.disturbance.t_start, delta=params.disturbance.delta, rate=params.disturbance.rate),
        noise=dict(enabled=bool(noise), z_sigma=ncfg["z_sigma"] if noise else 0.0, z_rho=ncfg["z_rho"]),
        n_steps=len(states_log), success=bool(expert.done), final_state=expert.state,
        **cov, **con,
    )
    return dict(frames=frames, meta=meta, canvas=env.canvas.img.copy())
