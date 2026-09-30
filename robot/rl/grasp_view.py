"""
grasp_view.py -- scripted top grasp of the can, shown live in the MuJoCo viewer (rotate
freely while it runs). Real contact dynamics: the can is held only by finger friction.

    python robot/rl/grasp_view.py              # viewer, loops forever
    python robot/rl/grasp_view.py --headless --trials 10   # success rate only

Grasp parameters from a sweep (scratch grasp_sweep): fingers slightly open (-0.1 rad),
pinch centre at the can top, shifted 1.5 cm toward the thumb, lift with the shoulder.
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot.rl.sim_util import SCENE, Ids, READY, GRIP_CLOSED, ik, contacts   # noqa: E402

OPEN = -0.10       # finger pre-shape (rad)
DZ = 0.0           # pinch centre height relative to the can top
OFF = 0.015        # shift toward the thumb (m)
LIFT = 0.30        # shoulder Pitch raise (rad)


class Demo:
    def __init__(self, viewer=None):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.ids = Ids(self.m)
        self.v = viewer
        self.half_h = self.m.geom_size[self.ids.can_geom][1]

    def ramp(self, q_to, grip, secs):
        m, d, ids = self.m, self.d, self.ids
        q_from = d.ctrl[ids.arm_u].copy(); g_from = d.ctrl[ids.hand_u].copy()
        n = int(secs / m.opt.timestep)
        t0 = time.perf_counter()
        for i in range(n):
            s = min(1.0, (i + 1) / (0.8 * n)); s = s * s * (3 - 2 * s)
            d.ctrl[ids.arm_u] = q_from + (np.asarray(q_to) - q_from) * s
            d.ctrl[ids.hand_u] = g_from + (grip - g_from) * s
            mujoco.mj_step(m, d)
            if self.v is not None and i % 8 == 0:
                if not self.v.is_running():
                    raise SystemExit
                self.v.sync()
                lag = d.time - (time.perf_counter() - t0) - self.t_start
                if lag > 0:
                    time.sleep(lag)

    def trial(self, cx, cy):
        m, d, ids = self.m, self.d, self.ids
        mujoco.mj_resetData(m, d)
        d.qpos[ids.can_q:ids.can_q + 3] = [cx, cy, self.half_h]
        top = np.array([cx, cy, 2 * self.half_h + DZ])
        at0 = ik(m, d, ids, top, q0=READY, iters=300)
        dd = mujoco.MjData(m); dd.qpos[:] = d.qpos; dd.qpos[ids.arm_q] = at0; mujoco.mj_kinematics(m, dd)
        at = ik(m, d, ids, top + OFF * dd.site_xmat[ids.tool].reshape(3, 3)[:, 0], q0=at0, iters=200)
        above = at.copy(); above[1] -= LIFT
        d.qpos[ids.arm_q] = above; d.ctrl[ids.arm_u] = above
        d.qpos[ids.hand_q] = OPEN; d.ctrl[ids.hand_u] = OPEN
        mujoco.mj_forward(m, d)
        self.t_start = d.time - 0.0
        self.ramp(above, OPEN, 0.6)          # show the start
        self.ramp(at, OPEN, 1.5)             # descend over the can
        self.ramp(at, GRIP_CLOSED, 1.0)      # close
        th, nf, _ = contacts(m, d, ids)
        self.ramp(above, GRIP_CLOSED, 2.5)   # lift
        self.ramp(above, GRIP_CLOSED, 1.0)   # hold
        rise = d.xpos[ids.can_body][2] - self.half_h
        return rise, th, nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--trials", type=int, default=0, help="0 = loop forever (viewer)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    def run(demo):
        k, ok = 0, 0
        while a.trials == 0 or k < a.trials:
            cx, cy = rng.uniform(-0.015, 0.015), rng.uniform(-0.07, -0.05)
            rise, th, nf = demo.trial(cx, cy)
            k += 1; ok += rise > 0.03
            print(f"trial {k}: can@({cx:+.3f},{cy:+.3f})  thumb={th} fingers={nf}  lifted {rise*100:+.1f} cm"
                  f"   [{ok}/{k}]", flush=True)

    if a.headless:
        run(Demo())
        return
    import mujoco.viewer
    demo = Demo()
    with mujoco.viewer.launch_passive(demo.m, demo.d) as v:
        v.cam.lookat[:] = [0, -0.05, 0.08]; v.cam.distance = 0.6; v.cam.azimuth = 135; v.cam.elevation = -25
        demo.v = v
        run(demo)


if __name__ == "__main__":
    main()
