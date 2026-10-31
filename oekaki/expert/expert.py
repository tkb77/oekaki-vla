"""Scripted drawing expert.

State machine: START -> APPROACH -> DESCEND -> PRELOAD -> DRAW -> LIFT -> (APPROACH ...) -> HOME -> DONE

The expert only uses what the policy could observe (joint angles, contact switch, compression):
the true paper height is never read. The paper height estimate starts at the upper bound of the
randomisation range and is refined from contact events / compression while drawing.
"""
from __future__ import annotations

import numpy as np

from oekaki.expert.ik import TipCommander
from oekaki.expert.shapes import resample


def _step_towards(cur: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    d = goal - cur
    n = np.linalg.norm(d)
    return goal.copy() if n <= max_step else cur + d / n * max_step


class DrawingExpert:
    def __init__(self, env, ecfg: dict):
        self.env = env
        self.c = ecfg
        self.dt = env.dt
        self.cmd = TipCommander(env.model, env.cfg["wrist_roll_cmd"], env.cfg["gripper_cmd"])
        self.threshold = env.threshold
        r = env.cfg["randomization"]
        self.z_upper = env.paper_nominal[2] + r["height_range"] + 0.002
        self.home_tip = np.asarray(env.cfg["home_tip"], dtype=np.float64)

    # ------------------------------------------------------------------
    def reset(self, strokes_world: list[np.ndarray], obs: dict):
        """strokes_world: list of (N,2) xy polylines in the robot base frame."""
        sp = self.c["draw_speed"] * self.dt
        self.strokes = [resample(s, sp) for s in strokes_world]
        self.cmd.reset(obs["joint_pos"])
        self.target = self.cmd.tip_of(obs["joint_pos"])  # current commanded rest-tip position
        self.z_est = self.z_upper
        self.have_contact_est = False
        self.stroke_i = 0
        self.prog = 0
        self.state = "START"
        self.timer = 0
        self.draw_steps = 0
        self.lost_contact_steps = 0
        self.tap_steps = 0
        self.done = False

    def expected_draw_time(self) -> float:
        return sum(len(s) for s in self.strokes) * self.dt

    # ------------------------------------------------------------------
    def _paper_measure(self, obs) -> float:
        tip_rest = self.cmd.tip_of(obs["joint_pos"])
        return tip_rest[2] + obs["compression"]

    def act(self, obs: dict) -> np.ndarray:
        c, dt = self.c, self.dt
        contact = obs["contact"] == 1
        comp = obs["compression"]
        if contact:
            z_meas = self._paper_measure(obs)
            if not self.have_contact_est:
                self.z_est = z_meas
                self.have_contact_est = True
            else:
                self.z_est += c["est_gain"] * (z_meas - self.z_est)

        st = self.state
        if st == "START":
            self.timer += 1
            if self.timer >= int(c["start_hold_s"] / dt):
                self.state = "APPROACH"
        elif st == "APPROACH":
            p0 = self.strokes[self.stroke_i][0]
            goal = np.array([p0[0], p0[1], self.z_est + c["hover"]])
            # go up first if we are low, then across
            if self.target[2] < goal[2] - 1e-4:
                up = self.target.copy(); up[2] = goal[2]
                self.target = _step_towards(self.target, up, c["travel_speed"] * dt)
            else:
                self.target = _step_towards(self.target, goal, c["travel_speed"] * dt)
            if np.linalg.norm(self.target - goal) < 1e-6:
                self.state = "DESCEND"
        elif st == "DESCEND":
            if contact:
                self.state = "PRELOAD"
            else:
                self.target[2] -= c["descend_speed"] * dt
        elif st == "PRELOAD":
            z_goal = self.z_est - c["target_compression"]
            self.target[2] = max(z_goal, self.target[2] - c["preload_speed"] * dt)
            if comp >= 0.8 * c["target_compression"]:
                self.state = "DRAW"
                self.prog = 0
                self.tap_steps = 0
            elif not contact:
                self.target[2] -= c["preload_speed"] * dt
        elif st == "DRAW":
            stroke = self.strokes[self.stroke_i]
            if contact:
                self.lost_contact_steps = 0
                z_cmd = self.z_est - c["target_compression"]
                if comp > c["max_compression"]:
                    z_cmd += (comp - c["target_compression"])
                self.prog = min(self.prog + 1, len(stroke) - 1)
                self.target = np.array([stroke[self.prog][0], stroke[self.prog][1], z_cmd])
            else:
                # lost contact: hold xy, search downward (paper moved away / pen bounced)
                self.lost_contact_steps += 1
                self.z_est -= c["search_speed"] * dt
                self.target[2] = min(self.target[2], self.z_est - c["target_compression"])
            self.draw_steps += 1
            if self.prog >= len(stroke) - 1 and contact:
                # a dot (stroke of ~zero length) is a tap: stay pressed for tap_hold_s, then lift
                if len(stroke) <= 2 and self.tap_steps < int(c.get("tap_hold_s", 0.0) / dt):
                    self.tap_steps += 1
                else:
                    self.state = "LIFT"
        elif st == "LIFT":
            z_goal = self.z_est + c["lift_height"]
            self.target[2] = min(z_goal, self.target[2] + c["lift_speed"] * dt)
            if self.target[2] >= z_goal - 1e-6 and not contact:
                self.stroke_i += 1
                self.state = "APPROACH" if self.stroke_i < len(self.strokes) else "HOME"
        elif st == "HOME":
            self.target = _step_towards(self.target, self.home_tip, c["travel_speed"] * dt)
            if np.linalg.norm(self.target - self.home_tip) < 1e-6:
                self.state = "HOLD"
                self.timer = 0
        elif st == "HOLD":
            self.timer += 1
            if self.timer >= int(c["home_hold_s"] / dt):
                self.done = True
        return self.cmd(self.target)

    def action_with_offset(self, dz: float) -> np.ndarray:
        """Joint targets for the current target shifted by dz (used for DART execution noise)."""
        q_save = self.cmd.q.copy()
        a = self.cmd(self.target + np.array([0.0, 0.0, dz]))
        self.cmd.q = q_save
        return a
