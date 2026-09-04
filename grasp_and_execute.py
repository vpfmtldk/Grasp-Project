"""
grasp_and_execute.py -- the full loop: image -> grasp -> 3D -> move -> close -> lift.

    render/capture  ->  predict_grasp  ->  pixel_to_world  ->  IK  ->  arm move
                    ->  hand close     ->  lift            ->  success check

Runs in MuJoCo now (--backend sim). The real-robot path (--backend real) uses the
same core once you have a calibrated camera config (see camera_calib.py / hand-eye).

    python grasp_and_execute.py --backend sim --network output/models/260903_1112_grconvnet_rgb1_d0/epoch_13_iou_0.92 --trials 10
"""
import argparse
import os

import numpy as np

from predict_grasp import GraspPredictor, grasp_corners
DEG = np.pi / 180.0
import pixel_to_world as p2w

TABLE_Z = 0.0          # scene table top
CUBE_HALF = 0.015      # object half-height -> grasp contact is this far above the plane
PRE_LIFT = 0.10        # approach / retreat height above the grasp point (m)
WRIST_PITCH_DOWN = 61.0      # Wrist_Pitch (deg) -> AmazingHand fingers point straight down
WRIST_ROLL_OFFSET = 0.0      # added to the grasp yaw to get Wrist_Roll (deg) -- tune


# --------------------------------------------------------------- sim cam config
def sim_cam_cfg(sim):
    """Build the pixel_to_world config dict straight from the MuJoCo camera."""
    import mujoco
    m, d = sim.model, sim.data
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, sim.camera)
    H, W = sim._rh, sim._rw
    fovy = float(m.cam_fovy[cid])
    fy = (H / 2) / np.tan(np.radians(fovy) / 2)
    K = [[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]]
    # MuJoCo cam frame: x right, y up, z toward viewer  ->  OpenCV: x right, y down, z fwd
    R_mj = np.array(d.cam_xmat[cid]).reshape(3, 3)
    R_cv = R_mj @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = R_cv
    T[:3, 3] = np.array(d.cam_xpos[cid])
    return p2w._to_np({
        "K": K, "dist": [0, 0, 0, 0, 0], "model": "pinhole",
        "T_base_cam": T.tolist(),
        "table_plane": {"point": [0, 0, TABLE_Z], "normal": [0, 0, 1]},
    })


# ------------------------------------------------------------------- one attempt
def execute_grasp(rgb, predictor, cam_cfg, arm, hand, vis_path=None):
    preds = predictor.predict(rgb, n_grasps=1)
    if not preds:
        return False, "no grasp detected", None
    g = preds[0]
    bg = p2w.grasp_to_base(g, cam_cfg, object_height=CUBE_HALF)
    pos = np.array(bg["position"])
    if not np.all(np.isfinite(pos)):
        return False, "grasp ray missed the table plane", g

    n = np.array([0, 0, 1.0])                        # IK targets the 'tool' site = grasp centre
    T_pre = np.eye(4);   T_pre[:3, 3]   = pos + PRE_LIFT * n
    T_grasp = np.eye(4); T_grasp[:3, 3] = pos + 0.005 * n

    yaw = float(bg["yaw_deg"])
    wlock = {3: WRIST_PITCH_DOWN * DEG, 4: (yaw + WRIST_ROLL_OFFSET) * DEG}

    def go(T, secs):
        q = arm.ee_pose_to_joints(T, lock=wlock)   # IK joints 0..2; wrist held palm-down at grasp yaw
        arm.move_joints_deg(q, secs=secs)
        return q

    from robot.mujoco_backend import READY_RAD
    hand.open()
    arm.move_joints_deg([v / DEG for v in READY_RAD], secs=2.0)   # neutral reach -> good IK seed
    go(T_pre, 2.0)
    go(T_grasp, 1.5)
    hand.close()
    go(T_pre, 1.5)
    arm.hold(0.4)

    if vis_path is not None:
        _save_vis(rgb, g, vis_path)
    return True, "executed", g


def _save_vis(rgb, g, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = grasp_corners(g["x"], g["y"], g["angle_rad"], g["width_px"], g["width_px"] / 2)
    c = np.vstack([c, c[0]])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.plot(c[:, 0], c[:, 1], "r-", lw=2)
    ax.plot(g["x"], g["y"], "r+", ms=10)
    ax.set_title("q=%.2f  %.0f deg" % (g["quality"], g["angle_deg"]))
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------- sim
def run_sim(args):
    from robot.mujoco_backend import MujocoRobot
    sim = MujocoRobot(args.mjcf, render_size=(args.res, args.res), ee_site="tool").connect()
    predictor = GraspPredictor(args.network, use_rgb=1, use_depth=0)
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    import mujoco
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_qadr = int(sim.model.jnt_qposadr[sim.model.body_jntadr[bid]])   # free-joint qpos start

    ok = 0
    for t in range(args.trials):
        sim.reset()
        x, y = rng.uniform(-0.05, 0.05), rng.uniform(-0.14, -0.06)      # central region
        sim.data.qpos[cube_qadr:cube_qadr + 7] = [x, y, CUBE_HALF, 1, 0, 0, 0]
        mujoco.mj_forward(sim.model, sim.data)

        sim.arm.goto_look_pose()
        sim.arm.hold(0.3)
        sim.mark_rest("cube")
        rgb = sim.render()

        run_ok, msg, g = execute_grasp(rgb, predictor, sim_cam_cfg(sim), sim.arm, sim.hand,
                                       vis_path=os.path.join(args.outdir, f"trial{t:02d}_pred.png"))
        lifted = sim.object_lifted("cube")
        ok += lifted
        from imageio.v2 import imwrite
        imwrite(os.path.join(args.outdir, f"trial{t:02d}_result.png"), sim.render())
        print(f"trial {t:02d}: cube@({x:+.3f},{y:+.3f})  {msg:28s}  lifted={lifted}")

    print(f"\n=== {ok}/{args.trials} = {ok/args.trials:.0%} grasp success ===")
    sim.disconnect()


# ------------------------------------------------------------------------ real
def run_real(args):
    import cv2
    from robot.robot_control import Config, SO101, AmazingHand
    cam_cfg = p2w.load_config(args.config)
    cap = cv2.VideoCapture(args.camera)
    cap.set(3, 1280); cap.set(4, 720)
    arm, hand = SO101(Config()), AmazingHand(Config())
    arm.connect(); hand.connect()
    predictor = GraspPredictor(args.network, use_rgb=1, use_depth=0)
    try:
        arm.goto_look_pose()
        ok, frame = cap.read()
        if not ok:
            print("camera read failed"); return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        done, msg, g = execute_grasp(rgb, predictor, cam_cfg, arm, hand,
                                     vis_path=os.path.join(args.outdir, "real_pred.png"))
        print(msg, "-- check the object by eye / retry")
        arm.goto_home()
    finally:
        cap.release(); arm.disconnect(); hand.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["sim", "real"], default="sim")
    p.add_argument("--network", required=True)
    p.add_argument("--mjcf", default="robot/sim/scene_ah.xml")
    p.add_argument("--res", type=int, default=480)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="output/cam.json", help="real: calibrated camera config")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--outdir", default="output/grasp_runs")
    a = p.parse_args()
    (run_sim if a.backend == "sim" else run_real)(a)


if __name__ == "__main__":
    main()
