"""
find_grasp_poses.py -- is there ANY arm pose where the upright can sits inside the open
AmazingHand (pinch centre on the can axis, no hand/can/table penetration)? Then does
closing the fingers there actually lift it (mj_step, real contact)?

Random joint sampling instead of a hand-picked approach direction: the AH is long and
mounted across the wrist, so the usual "fingers straight down" / "side grasp" poses are
outside SO-101's wrist range at can height.

    python robot/rl/find_grasp_poses.py --n 200000   -> robot/rl/grasp_poses.npz
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot.rl.sim_util import SCENE, Ids, GRIP_OPEN, GRIP_CLOSED, contacts   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_poses.npz")


def penetrating(m, d, ids):
    for c in d.contact[:d.ncon]:
        g = {c.geom1, c.geom2}
        if c.dist < -0.002 and (g & ids.hand_geoms) and (ids.can_geom in g or ids.table_geom in g):
            return True
    return False


def set_hand(m, d, ids, g):
    d.qpos[ids.hand_q] = g
    for k in range(4):
        d.qpos[m.jnt_qposadr[m.joint(f"h_f{k}_j2").id]] = 0.9 * g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--can", type=float, nargs=2, default=[0.0, -0.07])
    ap.add_argument("--test", type=int, default=30, help="physics close+lift tests on the best candidates")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m); ids = Ids(m)
    half_h = m.geom_size[ids.can_geom][1]
    lo, hi = m.jnt_range[[m.joint(n).id for n in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")]].T
    rng = np.random.default_rng(0)
    d.qpos[ids.can_q:ids.can_q + 7] = [*a.can, half_h, 1, 0, 0, 0]
    cand = []
    for i in range(a.n):
        q = rng.uniform(lo, hi)
        d.qpos[ids.arm_q] = q; set_hand(m, d, ids, GRIP_OPEN)
        mujoco.mj_kinematics(m, d)
        t = d.site_xpos[ids.tool]
        r = np.hypot(t[0] - a.can[0], t[1] - a.can[1])
        if r > 0.012 or not (0.02 < t[2] < 2 * half_h - 0.01):
            continue
        mujoco.mj_fwdPosition(m, d)
        if penetrating(m, d, ids):
            continue
        cand.append((r, q.copy(), t.copy()))
    print(f"{len(cand)} collision-free poses with the pinch centre on the can axis (of {a.n} samples)")
    cand.sort(key=lambda c: c[0])

    ok = []
    for r, q, t in cand[:a.test]:
        mujoco.mj_resetData(m, d)
        d.qpos[ids.can_q:ids.can_q + 7] = [*a.can, half_h, 1, 0, 0, 0]
        d.qpos[ids.arm_q] = q; d.ctrl[ids.arm_u] = q
        set_hand(m, d, ids, GRIP_OPEN); d.ctrl[ids.hand_u] = GRIP_OPEN
        for _ in range(200):
            mujoco.mj_step(m, d)
        d.ctrl[ids.hand_u] = GRIP_CLOSED
        for _ in range(600):
            mujoco.mj_step(m, d)
        th, nf, _ = contacts(m, d, ids)
        up = q.copy(); up[1] -= 0.25                       # shoulder lift up
        for k in range(1, 401):
            d.ctrl[ids.arm_u] = q + (up - q) * k / 400
            mujoco.mj_step(m, d)
        for _ in range(300):
            mujoco.mj_step(m, d)
        rise = d.xpos[ids.can_body][2] - half_h
        print(f"  q={np.round(q, 2)} tool z {t[2]:.3f}: thumb={th} fingers={nf} -> rise {rise*100:+.1f} cm")
        if rise > 0.03:
            ok.append(q)
    np.savez(OUT, poses=np.array([c[1] for c in cand]), lifted=np.array(ok), can=np.array(a.can))
    print(f"{len(ok)}/{min(a.test, len(cand))} lifted -> {OUT}")


if __name__ == "__main__":
    main()
