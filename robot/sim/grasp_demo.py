"""
grasp_demo.py -- kinematic grasp demo (Pollen-style) with the real AmazingHand.

The official AmazingHand MJCF is a kinematic model (its parallel linkage does not
transmit grip force under mj_step -- Pollen drive it with mink IK / mj_forward).
So this demo runs the full perception -> 3D -> arm-IK pipeline for real, then
closes the AmazingHand fingers kinematically around the object and carries it
(the object is rigidly attached once the fingers enclose it). It produces a GIF.

    python robot/sim/grasp_demo.py --network output/models/final_grconvnet_rgb1_d0/weights.pt
"""
import argparse
import os
import sys

import numpy as np
import mujoco
from imageio.v2 import mimsave, imwrite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot.mujoco_backend import MujocoRobot, READY_RAD, D2R          # noqa: E402
from predict_grasp import GraspPredictor                              # noqa: E402
import pixel_to_world as p2w                                          # noqa: E402
import grasp_and_execute as G                                         # noqa: E402

WRIST_PITCH_DOWN = 70.0
CUBE_HALF = 0.020


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="output/models/final_grconvnet_rgb1_d0/weights.pt")
    ap.add_argument("--mjcf", default="robot/sim/scene_ah.xml")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default="output/ah_kinematic_grasp.gif")
    a = ap.parse_args()

    sim = MujocoRobot(a.mjcf, render_size=(480, 480), ee_site="tool").connect()
    m, d = sim.model, sim.data
    pred = GraspPredictor(a.network, use_rgb=1, use_depth=0)
    bid = m.body("cube").id
    qa = int(m.jnt_qposadr[m.body_jntadr[bid]])
    sid = sim.ee_site
    rng = np.random.default_rng(0)

    free_cam = mujoco.MjvCamera()
    free_cam.lookat[:] = [0.0, -0.10, 0.10]
    free_cam.distance, free_cam.azimuth, free_cam.elevation = 0.55, 60, -22
    R = mujoco.Renderer(m, 480, 480)

    frames, ok = [], 0
    for t in range(a.trials):
        sim.reset(pose_rad=READY_RAD)
        cx, cy = rng.uniform(-0.03, 0.03), rng.uniform(-0.12, -0.08)
        d.qpos[qa:qa + 7] = [cx, cy, CUBE_HALF, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)

        sim.arm.goto_look_pose(); sim.arm.hold(0.2)
        rgb = sim.render()
        g = pred.predict(rgb, n_grasps=1)
        if not g:
            continue
        bg = p2w.grasp_to_base(g[0], G.sim_cam_cfg(sim), object_height=CUBE_HALF)
        pos = np.array(bg["position"])
        yaw = float(bg["yaw_deg"])
        wl = {3: WRIST_PITCH_DOWN * D2R, 4: yaw * D2R}

        def snap():
            R.update_scene(d, free_cam); frames.append(R.render())

        sim.hand.open(0.3)
        cdof = m.jnt_dofadr[m.body_jntadr[bid]]
        # neutral reach, pre-grasp above, descend so the pinch centre reaches the object
        for tgt, sc in [([cx, cy, 0.18], 1.6), ([cx, cy, 0.11], 1.4), ([cx, cy, pos[2] + 0.012], 1.1)]:
            e = np.zeros(3)
            for it in range(2):                   # 2nd pass only trims the residual -- quick, not a full re-move
                T = np.eye(4); T[:3, 3] = np.array(tgt) - e
                q = sim.arm.ee_pose_to_joints(T, lock=wl)
                sim.arm.move_joints_deg(q, sc if it == 0 else min(sc, 0.4), settle=0.2)
                e = d.site_xpos[sid] - np.array(tgt)
            snap()

        # object rides the pinch centre from here on (kinematic grasp)
        def park_cube():
            d.qpos[qa:qa + 3] = d.site_xpos[sid]
            d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            d.qvel[cdof:cdof + 6] = 0

        cube0 = d.qpos[qa:qa + 3].copy()      # ease the cube into the grip (not a hard snap)
        # close the AmazingHand fingers around it
        for k in range(6):
            sim.hand.set_opening(1.0 - (k + 1) / 6.0, 0.25)
            frac = (k + 1) / 6
            d.qpos[qa:qa + 3] = (1 - frac) * cube0 + frac * d.site_xpos[sid]
            d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]; d.qvel[cdof:cdof + 6] = 0
            mujoco.mj_forward(m, d)
            snap()

        # lift, object held at the pinch centre
        for tz in np.linspace(pos[2] + 0.012, pos[2] + 0.15, 12):
            T = np.eye(4); T[:3, 3] = [cx, cy, tz]
            q = sim.arm.ee_pose_to_joints(T, lock=wl)
            sim.arm.move_joints_deg(q, 0.45, settle=0.12)
            park_cube(); mujoco.mj_forward(m, d)
            snap()
        for _ in range(6):
            park_cube(); mujoco.mj_forward(m, d); snap()
        rise = float(d.xpos[bid][2]) - CUBE_HALF
        ok += rise > 0.05
        print(f"trial {t}: cube@({cx:+.3f},{cy:+.3f}) grasp {g[0]['angle_deg']:+.0f}deg -> carried {rise*100:.0f} cm")

    mimsave(a.out, frames, duration=0.05)
    imwrite(a.out.replace(".gif", "_final.png"), frames[-1])
    print(f"\n{ok}/{a.trials} kinematic grasps carried the object")
    print("wrote", a.out)
    sim.disconnect()


if __name__ == "__main__":
    main()
