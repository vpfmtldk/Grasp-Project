"""Small MuJoCo helpers shared by the can-grasp env, the scripted check and eval."""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "so101_rlhand_can.xml")
ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
HAND = ["h_f0", "h_f1", "h_f2", "h_f3"]           # h_f3 = thumb
GRIP_OPEN, GRIP_CLOSED = -0.83, 1.05                 # proximal flexion (rad), build_can_scene.PROX_RANGE
READY = np.array([0.0, -1.4, 1.7, 1.2, 0.0])          # robot/mujoco_backend.py READY_RAD


class Ids:
    def __init__(self, m):
        self.arm_q = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM])
        self.arm_v = np.array([m.jnt_dofadr[m.joint(n).id] for n in ARM])
        self.arm_u = np.array([m.actuator(n).id for n in ARM])
        self.hand_u = np.array([m.actuator(n).id for n in HAND])
        self.hand_q = np.array([m.jnt_qposadr[m.joint(n + "_j1").id] for n in HAND])
        self.tool = m.site("h_tool").id
        self.can_body = m.body("can").id
        self.can_q = m.jnt_qposadr[m.joint("can_free").id]
        self.can_v = m.jnt_dofadr[m.joint("can_free").id]
        self.can_geom = m.geom("can").id
        self.table_geom = m.body("table").geomadr[0]
        self.palm_body = m.body("h_palm").id
        self.hand_geoms = {g for g in range(m.ngeom)
                           if m.body(m.geom_bodyid[g]).name.startswith("h_")}
        self.thumb_geoms = {g for g in self.hand_geoms if m.body(m.geom_bodyid[g]).name.startswith("h_f3")}
        self.finger_geoms = {g for g in self.hand_geoms
                             if m.body(m.geom_bodyid[g]).name.startswith(("h_f0", "h_f1", "h_f2"))}


def finger_axis(d, ids):
    """World direction the fingers point (hand-root +z)."""
    return d.site_xmat[ids.tool].reshape(3, 3)[:, 2].copy()


def ik(m, d, ids, target, down=1.0, iters=150, q0=None, axes=None):
    """Damped least squares on the 5 arm joints: tool site -> target position plus a
    soft orientation term, weighted by `down` (0 = position only). Orientation: `axes`
    = {hand axis index: world direction}, default {2: down} = fingers straight down.
    Works on a scratch MjData copy; returns q."""
    axes = {2: (0, 0, -1.0)} if axes is None else axes
    dd = mujoco.MjData(m)
    dd.qpos[:] = d.qpos
    if q0 is not None:
        dd.qpos[ids.arm_q] = q0
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    lo, hi = m.jnt_range[[m.joint(n).id for n in ARM]].T
    for _ in range(iters):
        mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
        e = np.asarray(target) - dd.site_xpos[ids.tool]
        mujoco.mj_jacSite(m, dd, jp, jr, ids.tool)
        J, err = jp[:, ids.arm_v], e
        if down:
            Rm = dd.site_xmat[ids.tool].reshape(3, 3)
            ea = sum(np.cross(Rm[:, k], np.asarray(v, float)) for k, v in axes.items())
            w = 0.3 * down
            J = np.vstack([J, w * jr[:, ids.arm_v]]); err = np.concatenate([e, w * ea])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(err)), err)
        dd.qpos[ids.arm_q] = np.clip(dd.qpos[ids.arm_q] + dq, lo, hi)
        if np.linalg.norm(e) < 1e-3 and (not down or np.linalg.norm(ea) < 0.05):
            break
    return dd.qpos[ids.arm_q].copy()


def side_axes(yaw):
    """Side grasp of an upright can: fingers horizontal toward heading `yaw` (world, rad),
    finger row (hand y) vertical -- curling then wraps around the can's side."""
    return {2: (np.cos(yaw), np.sin(yaw), 0.0), 1: (0.0, 0.0, 1.0)}


def contacts(m, d, ids):
    """(thumb touches can, #finger geoms touching can, hand touches table)"""
    th, fi, tab = False, set(), False
    for c in d.contact[:d.ncon]:
        g1, g2 = c.geom1, c.geom2
        for a, b in ((g1, g2), (g2, g1)):
            if b == ids.can_geom and a in ids.thumb_geoms:
                th = True
            elif b == ids.can_geom and a in ids.finger_geoms:
                fi.add(m.geom_bodyid[a])
            elif b == ids.table_geom and a in ids.hand_geoms:
                tab = True
    return th, len(fi), tab


def touch_set(m, d, ids):
    """Digits touching the can: subset of {"f0" index, "f1" middle, "f2" ring/pinky, "f3" thumb}."""
    t = set()
    for c in d.contact[:d.ncon]:
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if b == ids.can_geom and a in ids.hand_geoms:
                t.add(m.body(m.geom_bodyid[a]).name.split("_")[1])
    return t
