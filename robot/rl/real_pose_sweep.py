"""
real_pose_sweep.py -- take the REAL grasp poses (teammate calibration + 25 mm lift, as
grasp_and_execute does) into the simulator through sim2real.py and see which hand roll /
can orientation / offset actually lifts the can under contact physics.

    python -m robot.rl.real_pose_sweep

Part A  geometry of the real pose: finger-axis tilt from horizontal, wrist height.
Part B  for sample pixels: place a lying 355 ml can near the simulated hand, close the
        'power' grip, lift to the 80 mm approach pose. Sweep the simulated wrist-roll value
        (the real->sim roll OFFSET is unverified), the can's axis (radial / tangential to
        the arm) and the can's offset along the finger axis.
"""
import itertools
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot import team_fk as T                                     # noqa: E402
from robot.calib.pixel_to_arm import HandEye                       # noqa: E402
from robot.robot_control import Config                             # noqa: E402
from robot.rl import sim2real as S2R                               # noqa: E402
from robot.rl.sim_util import SCENE, Ids, GRIP_CLOSED, touch_set  # noqa: E402

PRESHAPE = -0.3
LIFT_GRASP, LIFT_APPROACH = 0.025, 0.08


def real_axis_tilt(q_real):
    """angle (deg) of the finger axis (link5 +Y) below horizontal, and wrist height (mm)."""
    u = T.to_urdf(q_real)
    R, p = np.eye(3), np.zeros(3)
    pw = None
    for name, off, rpy, axis in T.CHAIN:
        p = p + R @ np.asarray(off)
        if name == "wrist_roll":
            pw = p.copy()
        R = R @ T._rpy(*rpy) @ T._rot(axis, math.radians(u.get(name, 0.0)))
    a = R @ np.array([0.0, 1.0, 0.0])
    return math.degrees(math.asin(-a[2])), pw[2] * 1000, a


def main():
    cfg = Config()
    lim = [(j.min_deg, j.max_deg) for j in cfg.arm_joints]
    he = HandEye(os.path.join(os.path.dirname(__file__), "..", "calib", "handeye_teammate.json"))
    (u0, u1), (v0, v1) = he.uv_range["u"], he.uv_range["v"]
    pix = [(u0 + f * (u1 - u0), v0 + g * (v1 - v0)) for f in (0.2, 0.5, 0.8) for g in (0.3, 0.7)]

    print("PART A: real grasp pose geometry (calibrated table contact, then lifted 25 mm)")
    for (u, v) in pix:
        q = he.joints_for_pixel(u, v)
        qg, got = T.lift_tool(q, LIFT_GRASP, lim)
        tilt, wz, a = real_axis_tilt(qg)
        tool = T.tool_xyz(qg) * 1000
        print(f"  px({u:4.0f},{v:4.0f})  tool xyz {np.round(tool, 0)} mm   fingers point {tilt:+5.1f} deg below horizontal   "
              f"wrist z {wz:+5.0f} mm   joints {np.round(qg[:4], 0)}")

    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    ids = Ids(m)
    R_can, H_can = float(m.geom_size[ids.can_geom][0]), float(m.geom_size[ids.can_geom][1])

    def place_can(center, axis_dir):
        z = np.array([0.0, 0.0, 1.0])
        a = np.asarray(axis_dir, float); a /= np.linalg.norm(a)
        q = np.zeros(4)
        ax = np.cross(z, a)
        ang = math.acos(np.clip(z @ a, -1, 1))
        if np.linalg.norm(ax) < 1e-9:
            q[:] = [1, 0, 0, 0]
        else:
            mujoco.mju_axisAngle2Quat(q, ax / np.linalg.norm(ax), ang)
        d.qpos[ids.can_q:ids.can_q + 7] = [*center, *q]
        d.qvel[ids.can_v:ids.can_v + 6] = 0

    def run(q_grasp, q_above, phi, axis_kind, dx, dy):
        mujoco.mj_resetData(m, d)
        qs_g = np.r_[S2R.real_to_sim_rad(q_grasp)[:4], np.radians(phi)]
        qs_a = np.r_[S2R.real_to_sim_rad(q_above)[:4], np.radians(phi)]
        d.qpos[ids.arm_q] = qs_a; d.ctrl[ids.arm_u] = qs_a
        d.qpos[ids.hand_q] = PRESHAPE; d.ctrl[ids.hand_u] = PRESHAPE
        mujoco.mj_forward(m, d)
        # can placed relative to the simulated hand at the GRASP pose
        d.qpos[ids.arm_q] = qs_g; mujoco.mj_kinematics(m, d)
        tool = d.site_xpos[ids.tool].copy()
        Rt = d.site_xmat[ids.tool].reshape(3, 3)
        fing = Rt[:, 2].copy(); row = Rt[:, 1].copy()
        fh = np.r_[fing[:2], 0.0]; fh = fh / (np.linalg.norm(fh) + 1e-9)
        th = np.array([-fh[1], fh[0], 0.0])
        center = np.array([tool[0], tool[1], R_can]) + dx * fh + dy * th
        axis = fh if axis_kind == "radial" else th
        mujoco.mj_resetData(m, d)
        place_can(center, axis)
        d.qpos[ids.arm_q] = qs_a; d.ctrl[ids.arm_u] = qs_a
        d.qpos[ids.hand_q] = PRESHAPE; d.ctrl[ids.hand_u] = PRESHAPE
        mujoco.mj_forward(m, d)

        def ramp(q_to, grip, secs):
            q0 = d.ctrl[ids.arm_u].copy(); g0 = d.ctrl[ids.hand_u].copy()
            n = int(secs / m.opt.timestep)
            for i in range(n):
                s = min(1.0, (i + 1) / (0.8 * n)); s = s * s * (3 - 2 * s)
                d.ctrl[ids.arm_u] = q0 + (q_to - q0) * s
                d.ctrl[ids.hand_u] = g0 + (grip - g0) * s
                mujoco.mj_step(m, d)
        ramp(qs_a, PRESHAPE, 0.3)
        ramp(qs_g, PRESHAPE, 1.2)
        moved = np.linalg.norm(d.xpos[ids.can_body][:2] - center[:2])
        ramp(qs_g, GRIP_CLOSED, 1.0)
        ts = touch_set(m, d, ids)
        ramp(qs_a, GRIP_CLOSED, 2.0)
        ramp(qs_a, GRIP_CLOSED, 0.4)
        rise = d.xpos[ids.can_body][2] - R_can
        return rise, ts, moved

    print("\nPART B: lying can under the real-mapped hand: lift in simulation (power grip)")
    phis = list(range(-180, 180, 30))
    rows = []
    for (u, v) in pix[:3]:
        q = he.joints_for_pixel(u, v)
        qg, _ = T.lift_tool(q, LIFT_GRASP, lim)
        qa, _ = T.lift_tool(q, LIFT_APPROACH, lim)
        for phi, axis_kind, dx in itertools.product(phis, ("radial", "tangential"), (-0.04, 0.0, 0.04, 0.08)):
            rise, ts, moved = run(qg, qa, phi, axis_kind, dx, 0.0)
            rows.append((rise, phi, axis_kind, dx, "+".join(sorted(ts)) or "-", moved, (u, v)))
    rows.sort(key=lambda r: -r[0])
    print("  top 15 by lift height (m):")
    for r in rows[:15]:
        print(f"   rise {r[0]*100:+5.1f} cm  roll {r[1]:+4d}  can axis {r[2]:10}  dx {r[3]:+.2f}  touch {r[4]:14}  "
              f"pushed {r[5]*100:3.1f} cm  px({r[6][0]:.0f},{r[6][1]:.0f})")
    print("\n  lifted >3 cm, by roll / axis:")
    for axis_kind in ("radial", "tangential"):
        line = []
        for phi in phis:
            n = sum(1 for r in rows if r[1] == phi and r[2] == axis_kind and r[0] > 0.03)
            tot = sum(1 for r in rows if r[1] == phi and r[2] == axis_kind)
            line.append(f"{phi:+4d}:{n}/{tot}")
        print(f"   {axis_kind:10} " + "  ".join(line))
    print(f"\n  (mapped roll for the real frozen roll ~19.6 deg = {S2R.real_to_sim_rad([0,0,0,0,19.6])[4]*180/math.pi:+.0f} deg in the simulator)")


if __name__ == "__main__":
    main()
