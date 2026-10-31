"""Damped-least-squares IK for the pen tip (position) with a soft "pen vertical" constraint."""
from __future__ import annotations

import mujoco
import numpy as np

IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


class PenIK:
    def __init__(self, model: mujoco.MjModel, site: str = "pen_tip_rest", axis_body: str = "pen_holder",
                 damping: float = 1e-4, orient_weight: float = 0.05):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(site).id
        self.body = model.body(axis_body).id
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in IK_JOINTS])
        self.dadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in IK_JOINTS])
        self.lo = np.array([model.jnt_range[model.joint(j).id][0] for j in IK_JOINTS])
        self.hi = np.array([model.jnt_range[model.joint(j).id][1] for j in IK_JOINTS])
        self.damping = damping
        self.w_ori = orient_weight
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, qpos_full: np.ndarray):
        self.d.qpos[:] = qpos_full
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.xmat[self.body].reshape(3, 3)[:, 2].copy()

    def solve(self, target: np.ndarray, q_init_full: np.ndarray, iters: int = 30, tol: float = 2e-5):
        """Returns (full qpos with IK joints updated, position error norm)."""
        m, d = self.m, self.d
        d.qpos[:] = q_init_full
        target = np.asarray(target, dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0])
        err = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site_xpos[self.site]
            axis = d.xmat[self.body].reshape(3, 3)[:, 2]
            e_pos = target - pos
            e_ori = np.cross(axis, up)
            err = np.linalg.norm(e_pos)
            if err < tol and np.linalg.norm(e_ori) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, self._jp, None, self.site)
            mujoco.mj_jacBody(m, d, None, self._jr, self.body)
            J = np.vstack([self._jp[:, self.dadr], self.w_ori * self._jr[:, self.dadr]])
            e = np.concatenate([e_pos, self.w_ori * e_ori])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), e)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        err = float(np.linalg.norm(target - d.site_xpos[self.site]))
        return d.qpos.copy(), err


class TipCommander:
    """Turns pen-tip (rest position) targets into 6-D joint-target actions, warm-starting IK."""

    def __init__(self, model: mujoco.MjModel, wrist_roll: float = 0.0, gripper: float = 0.0):
        self.ik = PenIK(model)
        self.q = np.zeros(model.nq)
        self.wrist_roll = wrist_roll
        self.gripper = gripper
        self.roll_adr = model.jnt_qposadr[model.joint("wrist_roll").id]
        self.grip_adr = model.jnt_qposadr[model.joint("gripper").id]
        self.arm_adr = self.ik.qadr

    def reset(self, joint_pos6):
        self.q[:] = 0.0
        self.q[self.arm_adr] = joint_pos6[:4]
        self.q[self.roll_adr] = self.wrist_roll
        self.q[self.grip_adr] = self.gripper

    def tip_of(self, joint_pos6) -> np.ndarray:
        q = self.q.copy()
        q[self.arm_adr] = joint_pos6[:4]
        q[self.roll_adr] = joint_pos6[4]
        q[self.grip_adr] = joint_pos6[5]
        return self.ik.fk(q)[0]

    def __call__(self, target) -> np.ndarray:
        self.q, _ = self.ik.solve(target, self.q, iters=20)
        return np.concatenate([self.q[self.arm_adr], [self.wrist_roll, self.gripper]])
