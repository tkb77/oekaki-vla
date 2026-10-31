"""Builds the drawing scene (SO-101 + spring pen holder + canvas) with MjSpec.

The SO-101 model comes from MuJoCo Menagerie (robotstudio_so101, Apache-2.0).
"""
from __future__ import annotations

import numpy as np
import mujoco

from oekaki.config import resolve

CANVAS_GROUP = 2  # contype/conaffinity bit shared only by the pen tip and the paper


def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion for a MuJoCo camera at `pos` looking at `target` (camera looks along -z, y up)."""
    pos, target, up = map(np.asarray, (pos, target, up))
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    mat = np.stack([x, y, -fwd], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def build_spec(cfg: dict) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(resolve(cfg["robot_xml"])))
    sim = cfg["sim"]
    spec.option.timestep = 1.0 / (sim["fps"] * sim["substeps"])
    spec.option.noslip_iterations = 0

    # ---- visuals -------------------------------------------------------------------
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.quality.shadowsize = 2048

    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=300, height=300,
                     rgb1=[0.55, 0.50, 0.45], rgb2=[0.50, 0.45, 0.40])
    floor_mat = spec.add_material(name="floor_mat", texrepeat=[6, 6], texuniform=True, reflectance=0.05)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"

    ccfg = cfg["canvas"]
    n = ccfg["tex_size"]
    spec.add_texture(name="canvas_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, width=n, height=n, nchannel=3,
                     rgb1=[1, 1, 1], rgb2=[1, 1, 1])
    canvas_mat = spec.add_material(name="canvas_mat", texrepeat=[1, 1], texuniform=False)
    canvas_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "canvas_tex"
    spec.add_material(name="table_mat", rgba=[0.45, 0.32, 0.22, 1])
    spec.add_material(name="pen_mat", rgba=[0.15, 0.25, 0.75, 1])
    spec.add_material(name="holder_mat", rgba=[0.25, 0.25, 0.25, 1])

    wb = spec.worldbody
    wb.add_light(name="sun", pos=[0.3, -0.3, 1.2], dir=[-0.2, 0.25, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.6, 0.6, 0.6], specular=[0.1, 0.1, 0.1], castshadow=1)
    wb.add_light(name="fill", pos=[-0.3, 0.4, 0.8], dir=[0.4, -0.4, -0.8], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.25, 0.25, 0.25], specular=[0, 0, 0], castshadow=0)
    # visual floor. Legacy scenes had it at -2.4 mm, which hid the paper when it was lowered > 22 mm;
    # it now sits lower and the robot stands on a visual pedestal (kinematics unchanged).
    floor_z = cfg.get("visual_floor_z", -0.0024)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, floor_z],
                material="floor_mat", contype=0, conaffinity=0)
    if floor_z < -0.005:
        wb.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.055, 0.065, -floor_z / 2],
                    pos=[0.006, 0, floor_z / 2], material="holder_mat", contype=0, conaffinity=0)

    # ---- canvas on a mocap body (height can be changed during an episode) ------------
    pcfg_solref = list(cfg["pen"].get("contact_solref", [0.004, 1.0]))
    half = ccfg["size"] / 2
    th = ccfg["thickness"]
    paper = wb.add_body(name="paper", mocap=True,
                        pos=[ccfg["center"][0], ccfg["center"][1], ccfg["height"]])
    # paper frame origin = paper top surface centre
    paper.add_geom(name="canvas", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, th / 2],
                   pos=[0, 0, -th / 2], material="canvas_mat", contype=0, conaffinity=0)
    # thick invisible collision slab whose top face coincides with the paper surface (prevents tunnelling)
    col_h = 0.02
    paper.add_geom(name="canvas_col", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, col_h / 2],
                   pos=[0, 0, -col_h / 2], rgba=[0, 0, 0, 0], group=3,
                   contype=1 << CANVAS_GROUP, conaffinity=1 << CANVAS_GROUP, condim=3,
                   friction=[cfg["pen"]["tip_friction"], 0.005, 0.0001],
                   solref=pcfg_solref, solimp=[0.95, 0.99, 0.001, 0.5, 2])
    # visual-only table block under the paper, long enough to reach the floor at any height
    table_h = 0.25
    paper.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[half + 0.03, half + 0.03, table_h / 2], pos=[0, 0, -th - table_h / 2],
                   material="table_mat", contype=0, conaffinity=0)
    paper.add_site(name="paper_origin", pos=[0, 0, 0], size=[0.002, 0, 0], group=5)

    # ---- top camera --------------------------------------------------------------------
    tcfg = cfg["cameras"]["top"]
    wb.add_camera(name="top", pos=tcfg["pos"], quat=look_at_quat(tcfg["pos"], tcfg["lookat"]),
                  fovy=tcfg["fovy"])

    # ---- spring pen holder fixed to the gripper (outside of the fixed jaw) -------------
    pcfg = cfg["pen"]
    gripper = spec.body("gripper")
    pitch = np.radians(pcfg.get("mount_pitch_deg", 0.0))
    holder = gripper.add_body(name="pen_holder", pos=pcfg["holder_offset"],
                              quat=[np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    holder.add_geom(name="holder_tube", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[pcfg["holder_radius"], pcfg["holder_length"] / 2],
                    pos=[0, 0, -pcfg["holder_length"] / 2], material="holder_mat",
                    contype=0, conaffinity=0, mass=0.01, group=1)
    holder.add_geom(name="holder_clamp", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, 0.008, 0.012],
                    pos=[0.008, 0, -0.012], material="holder_mat", contype=0, conaffinity=0,
                    mass=0.005, group=1)
    tip_z = pcfg["tip_offset"]
    holder.add_site(name="pen_tip_rest", pos=[0, 0, tip_z], size=[0.0015, 0, 0], group=5)

    wcfg = cfg["cameras"]["wrist"]
    if "pos" in wcfg:  # dedicated wrist camera on the holder, looking at the pen tip
        holder.add_camera(name="wrist_pen", pos=wcfg["pos"],
                          quat=look_at_quat(wcfg["pos"], wcfg["lookat"], up=wcfg.get("up", (0, 0, 1))),
                          fovy=wcfg["fovy"])

    pen = holder.add_body(name="pen", pos=[0, 0, 0])
    pen.add_joint(name="pen_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                  range=[0.0, pcfg["travel"]], limited=True,
                  stiffness=pcfg["stiffness"], springref=pcfg["springref"],
                  damping=pcfg["damping"], armature=pcfg["armature"], frictionloss=0.0,
                  solref_limit=[0.005, 1.0], solimp_limit=[0.99, 0.999, 0.0005, 0.5, 2])
    body_len = pcfg["pen_length"] - 0.008
    pen.add_geom(name="pen_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[pcfg["pen_radius"], body_len / 2],
                 pos=[0, 0, tip_z + 0.008 + body_len / 2], material="pen_mat",
                 contype=0, conaffinity=0, mass=pcfg["mass"], group=1)
    pen.add_geom(name="pen_cone", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                 size=[pcfg["tip_radius"] * 1.6, 0.003], pos=[0, 0, tip_z + 0.0045],
                 rgba=[0.85, 0.85, 0.85, 1], contype=0, conaffinity=0, mass=0.0005, group=1)
    pen.add_geom(name="pen_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[pcfg["tip_radius"], 0, 0],
                 pos=[0, 0, tip_z + pcfg["tip_radius"]], rgba=[0.05, 0.05, 0.3, 1],
                 contype=1 << CANVAS_GROUP, conaffinity=1 << CANVAS_GROUP, condim=3,
                 friction=[pcfg["tip_friction"], 0.005, 0.0001], mass=0.0005,
                 solref=pcfg_solref, solimp=[0.95, 0.99, 0.001, 0.5, 2], group=1)
    pen.add_site(name="pen_tip", pos=[0, 0, tip_z], size=[0.001, 0, 0], group=5)

    # sensors (useful for analysis; the env reads qpos directly)
    spec.add_sensor(name="pen_compression", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                    objtype=mujoco.mjtObj.mjOBJ_JOINT, objname="pen_slide")
    return spec


def build_model(cfg: dict) -> mujoco.MjModel:
    return build_spec(cfg).compile()
