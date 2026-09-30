"""
team_fk.py -- forward kinematics + "lift the tool straight up" for the REAL arm, ported
from the teammate's (A팀) amazinghand/workspace.py + ik.py + grasp_run.lift_tool.

Why theirs: our joint frame is now theirs (robot_control home_steps=2048 == LeRobot
DEGREES), and their FK was checked on the real arm -- SO-101 link lengths agree with the
URDF within 1.3 mm, tool offset 215 mm along link5 +Y chosen from a physical upright-pose
test. Joint angles map to URDF angles by robot/calib/team_frame.json (their measured
sign/offset per joint; elbow motor is mounted reversed).

    from robot.team_fk import tool_xyz, lift_tool
    q_up = lift_tool(q, 0.025)     # same (u, v) on the table, tool 25 mm higher

Why lift at all: their pixel->joint points were recorded with the fingers ON the table,
so the raw calibrated pose would press the arm into the desk.
"""
import json
import math
import os

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
FRAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib", "team_frame.json")

# SO-101 (arm.xacro joint origins, m): (offset in parent, rpy, axis)
CHAIN = [
    ("shoulder_pan",  (-0.040000, -0.012852,  0.062400), (math.pi, 0, 0), (0, 0, 1)),
    ("shoulder_lift", ( 0.018300, -0.030600, -0.052200), (math.pi, 0, 0), (1, 0, 0)),
    ("elbow_flex",    (-0.001500, -0.114582,  0.018082), (0, 0, 0),       (1, 0, 0)),
    ("wrist_flex",    (-0.001500,  0.132932,  0.028720), (0, 0, 0),       (1, 0, 0)),
    ("wrist_roll",    (-0.020100,  0.025822, -0.055375), (0, 0, 0),       (0.0, 0.422618, -0.906308)),
]
TOOL_OFFSET = (0.0, 0.215, 0.0)          # finger meeting point in link5 (their physical check)
MAX_STEP_M = 0.01
JAC_DELTA_DEG = 0.5
SINGULAR, DAMP_MAX = 0.04, 0.06

_MAP = None


def _mapping():
    global _MAP
    if _MAP is None:
        _MAP = json.load(open(FRAME, encoding="utf-8"))["mapping"]
    return _MAP


def to_urdf(q):
    """robot_control degrees (list of 5, == their UI degrees) -> URDF degrees dict."""
    m = _mapping()
    return {k: m[k]["sign"] * float(v) + m[k]["offset"] for k, v in zip(JOINTS, q)}


def from_urdf(u):
    m = _mapping()
    return [(u[k] - m[k]["offset"]) / m[k]["sign"] for k in JOINTS]


def _rot(axis, a):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, t = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[t*x*x + c, t*x*y - s*z, t*x*z + s*y],
                     [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
                     [t*x*z - s*y, t*y*z + s*x, t*z*z + c]])


def _rpy(r, p, y):
    return _rot((0, 0, 1), y) @ _rot((0, 1, 0), p) @ _rot((1, 0, 0), r)


def _tool_urdf(u):
    R, p = np.eye(3), np.zeros(3)
    for name, off, rpy, axis in CHAIN:
        p = p + R @ np.asarray(off)          # translate in the parent frame, then rotate
        R = R @ _rpy(*rpy) @ _rot(axis, math.radians(u.get(name, 0.0)))
    return p + R @ np.asarray(TOOL_OFFSET)


def tool_xyz(q):
    """Finger meeting point (m, robot base frame) for robot_control degrees q."""
    return _tool_urdf(to_urdf(q))


def lift_tool(q, lift_m, limits=None, joints=JOINTS[:4]):
    """Joint angles that put the tool lift_m higher, same x/y (damped least squares in
    <= 1 cm steps). wrist_roll is excluded -- it is frozen on this arm. Returns
    (q_new, achieved_m); if limits/singularities block it, achieved_m < lift_m."""
    q = list(map(float, q))
    start = tool_xyz(q)
    for _ in range(int(math.ceil(lift_m / MAX_STEP_M)) + 4):
        want = start + [0, 0, lift_m] - tool_xyz(q)
        if np.linalg.norm(want) < 5e-4:
            break
        dp = want * min(1.0, MAX_STEP_M / np.linalg.norm(want))
        u = to_urdf(q)
        p0 = _tool_urdf(u)
        cols = []
        for j in joints:
            v = dict(u); v[j] += JAC_DELTA_DEG
            cols.append((_tool_urdf(v) - p0) / math.radians(JAC_DELTA_DEG))
        J = np.array(cols).T
        smin = float(np.linalg.svd(J, compute_uv=False)[-1])
        lam = 0.0 if smin >= SINGULAR else DAMP_MAX * (1 - smin / SINGULAR)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(3), dp)
        for i, j in enumerate(joints):
            u[j] += math.degrees(dq[i])
        qn = from_urdf(u)
        if limits:                           # clamp only the joints being solved for
            qn = [min(max(v, lo), hi) if n in joints else v
                  for n, v, (lo, hi) in zip(JOINTS, qn, limits)]
        q = qn
    return q, float(tool_xyz(q)[2] - start[2])
