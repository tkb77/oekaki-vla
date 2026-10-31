"""MuJoCo drawing environment: SO-101 + spring-loaded pen + paper with a live texture."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass, field

import mujoco
import numpy as np

from oekaki.env.canvas import Canvas
from oekaki.env.scene import build_model

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class Disturbance:
    """Mid-episode paper height change: starting at `t_start` s, move by `delta` m at `rate` m/s."""
    t_start: float
    delta: float
    rate: float


@dataclass
class EpisodeParams:
    height_offset: float = 0.0
    xy_offset: tuple = (0.0, 0.0)
    light_scale: float = 1.0
    paper_tint: tuple = (255, 255, 255)
    disturbance: Disturbance | None = None
    extra: dict = field(default_factory=dict)


class DrawingEnv:
    def __init__(self, cfg: dict, render: bool = True, cameras=("top", "wrist"), cam_size=None):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.fps = cfg["sim"]["fps"]
        self.substeps = cfg["sim"]["substeps"]
        self.dt = 1.0 / self.fps

        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.pen_qadr = m.jnt_qposadr[m.joint("pen_slide").id]
        self.pen_dadr = m.jnt_dofadr[m.joint("pen_slide").id]
        self.paper_mocap = m.body_mocapid[m.body("paper").id]
        self.tip_geom = m.geom("pen_tip").id
        self.canvas_geom = m.geom("canvas_col").id
        self.tip_site = m.site("pen_tip").id
        self.tip_rest_site = m.site("pen_tip_rest").id
        self.tex_id = m.texture("canvas_tex").id
        self.tex_adr = m.tex_adr[self.tex_id]
        self.light_diffuse0 = m.light_diffuse.copy()

        cc = cfg["canvas"]
        self.paper_nominal = np.array([cc["center"][0], cc["center"][1], cc["height"]], dtype=np.float64)
        self.canvas = Canvas(cc["size"], cc["tex_size"], cc["line_width_px"], cc["line_color"])
        self.threshold = cfg["pen"]["contact_threshold"]
        self.travel = cfg["pen"]["travel"]

        self.cameras = list(cameras)
        self.cam_names = {"top": "top", "wrist": cfg["cameras"]["wrist"]["name"]}
        self.renderer = None
        if render:
            w = cam_size or cfg["cameras"]["width"]
            h = cam_size or cfg["cameras"]["height"]
            self.renderer = mujoco.Renderer(m, h, w)
        self.home_qpos = self._solve_home()
        self._tex_dirty = True
        self.params = EpisodeParams()
        self.t = 0.0
        self.step_count = 0
        self.paper_target_z = self.paper_nominal[2]

    def _solve_home(self) -> np.ndarray:
        from oekaki.expert.ik import PenIK
        ik = PenIK(self.model)
        q = np.zeros(self.model.nq)
        q[self.qadr] = [0.0, -0.5, 0.7, 0.7, self.cfg["wrist_roll_cmd"], self.cfg["gripper_cmd"]]
        q, err = ik.solve(np.asarray(self.cfg["home_tip"]), q, iters=300)
        assert err < 1e-3, f"home pose unreachable ({err})"
        return q[self.qadr].copy()

    # ------------------------------------------------------------------ episode control
    def sample_params(self, rng: np.random.Generator, disturbance: bool | None = None) -> EpisodeParams:
        r = self.cfg["randomization"]
        p = EpisodeParams(
            height_offset=float(rng.uniform(-r["height_range"], r["height_range"])),
            xy_offset=tuple(rng.uniform(-r["xy_range"], r["xy_range"], size=2).tolist()),
            light_scale=float(1.0 + rng.uniform(-r["light_jitter"], r["light_jitter"])),
            paper_tint=tuple(int(255 * (1 - rng.uniform(0, r["paper_tint"]))) for _ in range(3)),
        )
        dcfg = r["disturbance"]
        if disturbance is None:
            disturbance = rng.random() < dcfg["prob"]
        if disturbance:
            mag = rng.uniform(dcfg["delta_min"], dcfg["delta_max"])
            sign = rng.choice([-1.0, 1.0])
            # keep the paper inside the trained height band where possible
            if p.height_offset + sign * mag > r["height_range"] + 0.005 or \
               p.height_offset + sign * mag < -r["height_range"] - 0.005:
                sign = -sign
            # t_start is decided by the caller (e.g. mid-drawing); placeholder here
            p.disturbance = Disturbance(t_start=-1.0, delta=float(sign * mag), rate=dcfg["rate"])
        return p

    def reset(self, params: EpisodeParams | None = None, qpos=None) -> dict:
        m, d = self.model, self.data
        self.params = params or EpisodeParams()
        mujoco.mj_resetData(m, d)
        q0 = np.asarray(qpos if qpos is not None else self.home_qpos, dtype=np.float64)
        d.qpos[self.qadr] = q0
        d.ctrl[:6] = q0
        pos = self.paper_nominal.copy()
        pos[:2] += self.params.xy_offset
        pos[2] += self.params.height_offset
        d.mocap_pos[self.paper_mocap] = pos
        self.paper_target_z = pos[2]
        m.light_diffuse[:] = self.light_diffuse0 * self.params.light_scale
        self.canvas.clear(self.params.paper_tint)
        self._tex_dirty = True
        self.t = 0.0
        self.step_count = 0
        self.max_compression = 0.0
        self.touch_frac = 0.0
        mujoco.mj_forward(m, d)
        # settle (pen spring + gravity)
        for _ in range(self.substeps * 3):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        return self.observe()

    # ------------------------------------------------------------------ stepping
    def _paper_motion(self, h):
        dist = self.params.disturbance
        if dist is None or dist.t_start < 0 or self.t < dist.t_start:
            return
        base = self.paper_nominal[2] + self.params.height_offset
        target = base + dist.delta
        z = self.data.mocap_pos[self.paper_mocap, 2]
        step = np.clip(target - z, -dist.rate * h, dist.rate * h)
        self.data.mocap_pos[self.paper_mocap, 2] = z + step

    def tip_in_contact(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == self.tip_geom and c.geom2 == self.canvas_geom) or \
               (c.geom2 == self.tip_geom and c.geom1 == self.canvas_geom):
                return True
        return False

    def paper_local_xy(self, p_world) -> np.ndarray:
        return np.asarray(p_world[:2]) - self.data.mocap_pos[self.paper_mocap, :2]

    def step(self, action) -> dict:
        m, d = self.model, self.data
        a = np.asarray(action, dtype=np.float64)
        goal = np.clip(a[:6], m.actuator_ctrlrange[:6, 0], m.actuator_ctrlrange[:6, 1])
        prev = d.ctrl[:6].copy()
        interp = self.cfg["sim"].get("interpolate_ctrl", False)
        h = m.opt.timestep
        n_touch = 0
        for k in range(self.substeps):
            d.ctrl[:6] = prev + (goal - prev) * ((k + 1) / self.substeps) if interp else goal
            self._paper_motion(h)
            mujoco.mj_step(m, d)
            self.t += h
            if self.tip_in_contact():
                n_touch += 1
                self.canvas.pen_down(self.paper_local_xy(d.site_xpos[self.tip_site]))
                self._tex_dirty = True
            else:
                self.canvas.pen_up()
            self.max_compression = max(self.max_compression, d.qpos[self.pen_qadr])
        self.step_count += 1
        self.touch_frac = n_touch / self.substeps
        return self.observe()

    # ------------------------------------------------------------------ observation
    @property
    def compression(self) -> float:
        return float(self.data.qpos[self.pen_qadr])

    @property
    def contact(self) -> int:
        return int(self.compression > self.threshold)

    @property
    def paper_z(self) -> float:
        return float(self.data.mocap_pos[self.paper_mocap, 2])

    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def _upload_texture(self):
        if not self._tex_dirty or self.renderer is None:
            return
        m = self.model
        n = self.canvas.res * self.canvas.res * 3
        m.tex_data[self.tex_adr:self.tex_adr + n] = self.canvas.img.reshape(-1)
        self.renderer._gl_context.make_current()
        mujoco.mjr_uploadTexture(m, self.renderer._mjr_context, self.tex_id)
        self._tex_dirty = False

    def render(self, camera: str) -> np.ndarray:
        self._upload_texture()
        self.renderer.update_scene(self.data, camera=self.cam_names.get(camera, camera))
        return self.renderer.render().copy()

    def observe(self) -> dict:
        obs = {
            "joint_pos": self.joint_pos(),
            "contact": self.contact,
            "compression": self.compression,
            "tip_pos": self.data.site_xpos[self.tip_site].copy(),
            "tip_rest_pos": self.data.site_xpos[self.tip_rest_site].copy(),
            "paper_z": self.paper_z,
            "physical_contact": self.tip_in_contact(),
            "touch_frac": getattr(self, "touch_frac", 0.0),
            "t": self.t,
        }
        if self.renderer is not None:
            obs["images"] = {c: self.render(c) for c in self.cameras}
        return obs

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
