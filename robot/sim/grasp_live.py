"""
grasp_live.py -- watch one sim grasp trial in an interactive MuJoCo viewer
window (drag to rotate, scroll to zoom, right-drag to pan) instead of a saved
GIF. Same pipeline as grasp_and_execute.py --backend sim / grasp_demo.py:
perception -> pixel_to_world -> arm IK -> approach/descend are real physics;
the AmazingHand finger close is kinematic.

    python -m robot.sim.grasp_live --network output/models/final_grconvnet_rgb1_d0/weights.pt

Close the viewer window to exit.
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot.mujoco_backend import MujocoRobot, READY_RAD, LOOK_RAD, D2R, R2D  # noqa: E402
from predict_grasp import GraspPredictor                          # noqa: E402
import pixel_to_world as p2w                                      # noqa: E402
import grasp_and_execute as G                                     # noqa: E402

WRIST_PITCH_DOWN = 70.0
CUBE_HALF = 0.020


def move_and_watch(sim, viewer, target_deg, secs, settle=0.3, sync_every=10, speed=2.0):
    """Same interpolated move as _ArmView.move_joints_deg, but syncs the
    viewer + sleeps in real time as it steps, so you can watch it happen.

    sync_every: physics steps per viewer.sync() -- higher = fewer, cheaper
    render calls (each sync has real GL overhead, which is what made the
    first live runs feel much slower than the nominal `secs`).
    speed: playback speed multiplier (2.0 = twice as fast as real time).
    """
    m, d = sim.model, sim.data
    start = sim.arm.read_joints_deg()
    dt = m.opt.timestep
    n = max(1, int(secs / dt))
    for k in range(1, n + 1):
        a = k / n
        for aid, s, t in zip(sim.arm_aids, start, target_deg):
            d.ctrl[aid] = ((1 - a) * s + a * t) * D2R
        mujoco.mj_step(m, d)
        if k % sync_every == 0:
            viewer.sync()
            time.sleep(dt * sync_every / speed)
    for aid, t in zip(sim.arm_aids, target_deg):
        d.ctrl[aid] = t * D2R
    for k in range(int(settle / dt)):
        mujoco.mj_step(m, d)
        if k % sync_every == 0:
            viewer.sync()
            time.sleep(dt * sync_every / speed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="output/models/final_grconvnet_rgb1_d0/weights.pt")
    ap.add_argument("--mjcf", default="robot/sim/scene_ah.xml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed multiplier (higher = faster)")
    a = ap.parse_args()

    sim = MujocoRobot(a.mjcf, render_size=(480, 480), ee_site="tool").connect()
    m, d = sim.model, sim.data
    pred = GraspPredictor(a.network, use_rgb=1, use_depth=0)
    bid = m.body("cube").id
    qa = int(m.jnt_qposadr[m.body_jntadr[bid]])
    cdof = int(m.jnt_dofadr[m.body_jntadr[bid]])
    sid = sim.ee_site
    rng = np.random.default_rng(a.seed)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        def park():
            d.qpos[qa:qa + 3] = d.site_xpos[sid]
            d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            d.qvel[cdof:cdof + 6] = 0

        sim.reset(pose_rad=READY_RAD)
        cx, cy = rng.uniform(-0.03, 0.03), rng.uniform(-0.12, -0.08)
        d.qpos[qa:qa + 7] = [cx, cy, CUBE_HALF, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        viewer.sync(); time.sleep(0.5 / a.speed)

        print("moving to look pose...")
        move_and_watch(sim, viewer, [v * R2D for v in LOOK_RAD], 2.0, speed=a.speed)
        sim.arm.hold(0.2); viewer.sync()

        rgb = sim.render()
        preds = pred.predict(rgb, n_grasps=1)
        if not preds:
            print("no grasp detected -- close the window to exit")
            while viewer.is_running():
                viewer.sync(); time.sleep(0.05)
            return
        g = preds[0]
        bg = p2w.grasp_to_base(g, G.sim_cam_cfg(sim), object_height=CUBE_HALF)
        pos = np.array(bg["position"]); yaw = float(bg["yaw_deg"])
        wl = {3: WRIST_PITCH_DOWN * D2R, 4: yaw * D2R}
        print(f"grasp: cube@({cx:+.3f},{cy:+.3f})  predicted angle {g['angle_deg']:+.0f} deg  "
              f"3D pos {np.round(pos, 3)}")

        sim.hand.open(0.3); viewer.sync()
        print("approaching...")
        for tgt, sc in [([cx, cy, 0.18], 1.6), ([cx, cy, 0.11], 1.4), ([cx, cy, pos[2] + 0.012], 1.1)]:
            e = np.zeros(3)
            for it in range(2):                   # 2nd pass only trims the residual -- quick, not a full re-move
                T = np.eye(4); T[:3, 3] = np.array(tgt) - e
                q = sim.arm.ee_pose_to_joints(T, lock=wl)
                move_and_watch(sim, viewer, q, sc if it == 0 else min(sc, 0.4), speed=a.speed)
                e = d.site_xpos[sid] - np.array(tgt)

        print("closing the hand...")
        cube0 = d.qpos[qa:qa + 3].copy()      # ease the cube into the grip (not a hard snap)
        for k in range(6):
            sim.hand.set_opening(1.0 - (k + 1) / 6.0, 0.02)
            frac = (k + 1) / 6
            d.qpos[qa:qa + 3] = (1 - frac) * cube0 + frac * d.site_xpos[sid]
            d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]; d.qvel[cdof:cdof + 6] = 0
            mujoco.mj_forward(m, d)
            viewer.sync(); time.sleep(0.15 / a.speed)

        print("lifting...")
        for tz in np.linspace(pos[2] + 0.012, pos[2] + 0.15, 12):
            T = np.eye(4); T[:3, 3] = [cx, cy, tz]
            q = sim.arm.ee_pose_to_joints(T, lock=wl)
            move_and_watch(sim, viewer, q, 0.35, settle=0.08, speed=a.speed)
            park(); mujoco.mj_forward(m, d); viewer.sync()

        rise = float(d.xpos[bid][2]) - CUBE_HALF
        print(f"carried {rise*100:.0f} cm -- {'SUCCESS' if rise > 0.05 else 'missed'}  "
              "(close the window to exit)")
        while viewer.is_running():
            viewer.sync(); time.sleep(0.05)

    sim.disconnect()


if __name__ == "__main__":
    main()
