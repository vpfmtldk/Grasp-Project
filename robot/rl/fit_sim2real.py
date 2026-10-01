"""
fit_sim2real.py -- joint-angle map between the MuJoCo arm (robot/rl scene) and the real SO-101
in the teammate's frame (robot_control degrees), from forward kinematics only (no hardware).

    python -m robot.rl.fit_sim2real       -> robot/rl/sim2real_map.json

Per joint:  sim_deg = sign * real_deg + offset      (pan, lift, elbow, wrist_flex, wrist_roll)
Frames:     sim_xyz = Rz(yaw) @ real_xyz + t        (real = team_fk URDF base frame)

Stage 1  wrist-roll joint ORIGIN (independent of the hand): sign/offset of the 4 positioning
         joints + the rigid frame transform, by global search over the 2^4 sign combinations.
Stage 2  roll sign/offset from the hand's tool site (sim h_tool vs real tool = link5 + 215 mm).
Report   residuals (mm), the sim hand's equivalent TOOL_OFFSET in the real link5 frame (vs the
         real 215 mm), and whether the table height agrees (t_z ~ real table surface z).
"""
import itertools
import json
import math
import os
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import differential_evolution, least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot import team_fk as T                       # noqa: E402
from robot.robot_control import Config               # noqa: E402
from robot.rl.sim_util import SCENE, Ids             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim2real_map.json")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def real_fk(q_ui):
    """(wrist-roll joint origin, tool, R_end) in the real URDF base frame; q_ui = robot_control deg."""
    u = T.to_urdf(q_ui)
    R, p, p_roll = np.eye(3), np.zeros(3), None
    for name, off, rpy, axis in T.CHAIN:
        p = p + R @ np.asarray(off)
        if name == "wrist_roll":
            p_roll = p.copy()
        R = R @ T._rpy(*rpy) @ T._rot(axis, math.radians(u.get(name, 0.0)))
    return p_roll, p + R @ np.asarray(T.TOOL_OFFSET), R


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.ids = Ids(self.m)
        self.jr = self.m.joint("Wrist_Roll").id

    def fk(self, q_deg5):
        self.d.qpos[self.ids.arm_q] = np.radians(q_deg5)
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xanchor[self.jr].copy(), self.d.site_xpos[self.ids.tool].copy()


def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main():
    cfg = Config()
    lo = np.array([j.min_deg for j in cfg.arm_joints]); hi = np.array([j.max_deg for j in cfg.arm_joints])
    rng = np.random.default_rng(0)
    Q = rng.uniform(lo, hi, size=(60, 5))
    Pr, Tr, Rr = zip(*[real_fk(q) for q in Q])
    Pr, Tr = np.array(Pr), np.array(Tr)
    sim = Sim()

    def res1(x, signs, yaw, n=None):
        o, t = x[:4], x[4:7]
        out = []
        for q, pr in zip(Q[:n], Pr[:n]):
            qs = np.r_[signs * q[:4] + o, 0.0]
            ps, _ = sim.fk(qs)
            out.append(ps - (rz(yaw) @ pr + t))
        return np.concatenate(out)

    best = None
    t0 = time.time()
    for yaw in (0.0, 180.0):
        for signs in itertools.product((1.0, -1.0), repeat=4):
            signs = np.array(signs)
            f = lambda x: float(np.sum(res1(x, signs, yaw, 20) ** 2))
            r = differential_evolution(f, [(-180, 180)] * 4 + [(-0.4, 0.4)] * 3, popsize=18, maxiter=120,
                                       tol=1e-9, seed=1, polish=False)
            rms = math.sqrt(r.fun / 60)                 # 20 samples x 3 coords
            print(f"yaw {yaw:5.0f} signs {signs.astype(int)}  rms {rms*1000:7.2f} mm   ({time.time()-t0:.0f}s)", flush=True)
            if best is None or rms < best[0]:
                best = (rms, yaw, signs, r.x)
    rms, yaw, signs, x = best
    ls = least_squares(lambda x_: res1(x_, signs, yaw), x)
    o, t = ls.x[:4], ls.x[4:7]
    res = ls.fun.reshape(-1, 3)
    err = np.linalg.norm(res, axis=1) * 1000
    print(f"\nSTAGE 1 best: yaw {yaw:.0f}  signs {signs.astype(int)}  offsets {np.round(o, 2)} deg  "
          f"t {np.round(t, 4)} m")
    print(f"  wrist-origin residual over 60 poses: mean {err.mean():.2f} mm, max {err.max():.2f} mm")

    # stage 2: roll sign/offset from the tool position
    best2 = None
    for s_r in (1.0, -1.0):
        for o_r in np.arange(-180, 180, 2.0):
            e = 0.0
            for q, tr in zip(Q[:25], Tr[:25]):
                qs = np.r_[signs * q[:4] + o, s_r * q[4] + o_r]
                _, ts = sim.fk(qs)
                e += np.sum((ts - (rz(yaw) @ tr + t)) ** 2)
            if best2 is None or e < best2[0]:
                best2 = (e, s_r, o_r)
    _, s_r, o_r = best2
    ro = least_squares(lambda y: np.concatenate([
        sim.fk(np.r_[signs * q[:4] + o, s_r * q[4] + y[0]])[1] - (rz(yaw) @ tr + t) for q, tr in zip(Q, Tr)]),
        [o_r])
    o_r = float(ro.x[0])
    tool_err = []
    eq_off = []
    for q, tr, r_end in zip(Q, Tr, Rr):
        qs = np.r_[signs * q[:4] + o, s_r * q[4] + o_r]
        pr, _, _ = real_fk(q)
        ps_roll, ts = sim.fk(qs)
        tool_err.append(np.linalg.norm(ts - (rz(yaw) @ tr + t)) * 1000)
        # sim tool in the real wrist frame: (R_end^T) (Rz^-1 (ts - t) - p_roll_real)
        eq_off.append(r_end.T @ (rz(yaw).T @ (ts - t) - pr))
    eq_off = np.array(eq_off)
    print(f"\nSTAGE 2 roll: sign {s_r:+.0f}  offset {o_r:+.1f} deg")
    print(f"  tool-position residual: mean {np.mean(tool_err):.1f} mm, max {np.max(tool_err):.1f} mm")
    print(f"  sim hand's equivalent tool offset in the real link5 frame: mean {np.round(eq_off.mean(0)*1000, 1)} mm"
          f"  (std {np.round(eq_off.std(0)*1000, 1)}) vs real TOOL_OFFSET {np.round(np.array(T.TOOL_OFFSET)*1000, 1)} mm")

    out = {
        "note": "sim_deg = sign * real_deg + offset (real = robot_control / teammate frame); "
                "sim_xyz = Rz(yaw) @ real_xyz + t. From FK only (fit_sim2real.py).",
        "joints": JOINTS,
        "sign": [int(s) for s in signs] + [int(s_r)],
        "offset_deg": [float(v) for v in o] + [o_r],
        "yaw_deg": float(yaw), "t_m": [float(v) for v in t],
        "wrist_origin_residual_mm": {"mean": float(err.mean()), "max": float(err.max())},
        "tool_residual_mm": {"mean": float(np.mean(tool_err)), "max": float(np.max(tool_err))},
        "sim_tool_in_real_link5_mm": [float(v) for v in eq_off.mean(0) * 1000],
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
