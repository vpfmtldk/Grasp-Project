"""
scripted_grasp.py -- feasibility check BEFORE any RL: can this hand physically
grasp and lift the can under real contact dynamics (mj_step, no gluing)?

Top grasp: fingers down over the can's upper part (the only can-height pose SO-101's
wrist range allows with the long AmazingHand -- see find_grasp_poses.py), then the
shoulder lifts it. Every move is a smooth ctrl ramp, not a step.

    python robot/rl/scripted_grasp.py            -> output/rl/scripted_grasp.gif
"""
import argparse
import os
import sys

import mujoco
import numpy as np
from imageio.v2 import mimsave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot.rl.sim_util import SCENE, Ids, READY, GRIP_OPEN, GRIP_CLOSED, ik, contacts   # noqa: E402

LIFT = 0.30        # rad of shoulder Pitch toward "up" for the lift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--grasp-dz", type=float, default=-0.02, help="tool height relative to the can top")
    ap.add_argument("--out", default="output/rl/scripted_grasp.gif")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m)
    ids = Ids(m)
    half_h = m.geom_size[ids.can_geom][1]
    R = mujoco.Renderer(m, 400, 400)
    frames, ok = [], 0
    rng = np.random.default_rng(1)

    def ramp(q_to, grip, secs, snap=0.08):
        q_from = d.ctrl[ids.arm_u].copy(); g_from = d.ctrl[ids.hand_u].copy()
        n, k = int(secs / m.opt.timestep), int(snap / m.opt.timestep)
        for i in range(n):
            s = min(1.0, (i + 1) / (0.8 * n)); s = s * s * (3 - 2 * s)
            d.ctrl[ids.arm_u] = q_from + (np.asarray(q_to) - q_from) * s
            d.ctrl[ids.hand_u] = g_from + (grip - g_from) * s
            mujoco.mj_step(m, d)
            if i % k == 0:
                R.update_scene(d, "side_cam"); frames.append(R.render())

    for t in range(a.trials):
        mujoco.mj_resetData(m, d)
        cx, cy = rng.uniform(-0.03, 0.03), rng.uniform(-0.08, -0.05)
        d.qpos[ids.can_q:ids.can_q + 3] = [cx, cy, half_h]
        tgt = [cx, cy, 2 * half_h + a.grasp_dz]
        at = ik(m, d, ids, tgt, q0=READY, iters=300)
        above = at.copy(); above[1] -= LIFT                       # same pose, shoulder raised
        d.qpos[ids.arm_q] = above; d.ctrl[ids.arm_u] = above
        d.qpos[ids.hand_q] = GRIP_OPEN; d.ctrl[ids.hand_u] = GRIP_OPEN
        mujoco.mj_forward(m, d)
        ramp(above, GRIP_OPEN, 0.4)
        ramp(at, GRIP_OPEN, 1.2)
        err = np.linalg.norm(d.site_xpos[ids.tool] - tgt)
        ramp(at, GRIP_CLOSED, 0.8)
        th, nf, _ = contacts(m, d, ids)
        ramp(above, GRIP_CLOSED, 1.5)
        ramp(above, GRIP_CLOSED, 0.5)
        rise = d.xpos[ids.can_body][2] - half_h
        tilt = np.degrees(np.arccos(np.clip(d.xmat[ids.can_body].reshape(3, 3)[2, 2], -1, 1)))
        ok += rise > 0.03
        print(f"trial {t}: can@({cx:+.3f},{cy:+.3f}) tool err {err*1000:.0f}mm  closed: thumb={th} fingers={nf}"
              f"  -> lifted {rise*100:.1f} cm, tilt {tilt:.0f} deg")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    mimsave(a.out, frames, duration=0.08)
    print(f"{ok}/{a.trials} lifted  -> {a.out}")


if __name__ == "__main__":
    main()
