"""
grasp_and_execute.py -- the full loop: image -> grasp -> 3D -> move -> close -> lift.

    render/capture  ->  predict_grasp  ->  pixel_to_world  ->  IK  ->  arm move
                    ->  hand close     ->  lift            ->  success check

Runs in MuJoCo now (--backend sim). The real-robot path (--backend real) uses the
same core once you have a calibrated camera config (see camera_calib.py / hand-eye).

    python grasp_and_execute.py --backend sim --network output/models/260903_1112_grconvnet_rgb1_d0/epoch_13_iou_0.92 --trials 10
"""
import argparse
import math
import os

import numpy as np

from predict_grasp import GraspPredictor, grasp_corners
DEG = np.pi / 180.0
import pixel_to_world as p2w

TABLE_Z = 0.0          # scene table top
CUBE_HALF = 0.020      # object half-height -> grasp contact is this far above the plane
PRE_LIFT = 0.10        # approach / retreat height above the grasp point (m)
WRIST_PITCH_DOWN = 70.0      # Wrist_Pitch (deg) -> AmazingHand palm horizontal, fingers down
WRIST_ROLL_OFFSET = 0.0      # added to the grasp yaw to get Wrist_Roll (deg) -- tune
AUTO_STILL_S = 1.5            # real --auto: object must be still this long before a grasp starts
AUTO_MOVED_PX = 40            # ... and this far (~1.5 cm) from the last grasp spot
TILT_WARN_DEG = 15.0         # real: warn when the object's across-direction is this far off the hand's


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
def execute_grasp(rgb, predictor, he, arm, hand, vis_path=None, secs=2.5, px_per_mm=2.0,
                  grasp_lift_m=0.025, approach_m=0.08, confirm=True, grip="auto", result_fn=None,
                  center_object=True):
    """One real-robot attempt using the pixel->joint-angle direct interpolation
    calibration (robot/calib/pixel_to_arm.HandEye) -- NOT the camera-3D/IK path
    (that's what run_sim still uses; the two backends deliberately differ, see
    RESULTS.md). No 3D pose, no FK/IK: the pixel maps straight to 5 joint angles.
    """
    from robot.calib.pixel_to_arm import ALL_JOINTS
    preds = predictor.predict(rgb, n_grasps=1)
    if not preds:
        return False, "no grasp detected", None
    g = preds[0]
    if center_object:
        # GR-ConvNet (parallel-jaw labels) aims at object edges; the AmazingHand wraps the
        # object, so move the target to the centre of the object nearest the prediction
        from object_center import center_grasp
        (u0_, u1_), (v0_, v1_) = he.uv_range["u"], he.uv_range["v"]
        ng, _obs = center_grasp(rgb, g, (u0_, v0_, u1_, v1_))
        if ng is None:
            print("  object centre: no object segmented -- using the model's point")
        else:
            print(f"  object centre: model ({g['x']:.0f},{g['y']:.0f}) -> centre ({ng['x']:.0f},{ng['y']:.0f}), "
                  f"across-object angle {ng['angle_deg']:+.1f}deg, width {ng['width_px']:.0f}px")
            g = ng
    u, v = float(g["x"]), float(g["y"])
    inside = he.in_range(u, v)
    q = he.joints_for_pixel(u, v, theta_img_deg=g["angle_deg"])
    print(f"  pixel=({u:.0f},{v:.0f}) angle={g['angle_deg']:+.1f}deg width={g['width_px']:.0f}px  "
          f"in_range={inside}")
    print("  target joints: " + "  ".join(f"{n}={a:+.1f}" for n, a in zip(ALL_JOINTS, q)))
    if not inside:
        print("  WARNING: pixel outside the calibrated area -- extrapolation, unreliable")

    # The calibration (teammate's, robot/calib/handeye_teammate.json) was recorded with the
    # fingers ON the table: going there directly presses the arm into the desk. Lift the
    # tool straight up by FK/IK (robot/team_fk.py, the teammate's verified FK) -- 25 mm for
    # the grasp (their DEFAULT_LIFT_M), more for the approach.
    from robot.team_fk import lift_tool
    lim = [(j.min_deg, j.max_deg) for j in arm.joints]
    q_grasp, got = lift_tool(q, grasp_lift_m, lim)
    q_above, got_a = lift_tool(q, approach_m, lim)
    print(f"  tool lift: grasp {got*1000:.0f}/{grasp_lift_m*1000:.0f} mm, "
          f"approach {got_a*1000:.0f}/{approach_m*1000:.0f} mm")
    if got < 0.8 * grasp_lift_m:
        return False, "cannot lift the grasp pose off the table (limits) -- not moving", g
    if callable(confirm):
        r = confirm("plan", g)
        if r is None:                            # operator moved the object: capture again
            return False, "recapture", g
        if not r:
            return False, "aborted by operator before moving", g

    start = arm.read_joints_deg()
    cur = [s if s is not None else 0.0 for s in start]

    # pre-shape the hand before approaching; power_splay approaches with the fingers fanned
    hand.set_preset("open_splay" if grip == "power_splay" else "open", 1.0)
    # stage 1: pan only, still folded -- point at the object first
    t1 = list(cur); t1[0] = q_above[0]
    arm.move_joints_deg(t1, secs=secs, max_step_deg=180)
    # stage 2: unfold to the approach pose above the object, then descend to the grasp
    arm.move_joints_deg(q_above, secs=secs * 1.5, max_step_deg=180)
    if confirm:
        # first real runs: check by eye that the hand hovers over the object before it
        # descends (catches a moved camera or a frame/offset error before contact).
        # confirm may be a callable (camera-window prompt, run_real) or True (terminal).
        ok_ = confirm("hover", g) if callable(confirm) else input(
            f"  hovering {approach_m*1000:.0f} mm above the target -- is it over the object? "
            f"Enter = descend, q = abort > ").strip().lower() != "q"
        if not ok_:
            back = [s if s is not None else q_above[i] for i, s in enumerate(start)]
            arm.move_joints_deg(back, secs=secs * 1.5, max_step_deg=180)
            return False, "aborted by operator at the hover pose", g
    arm.move_joints_deg(q_grasp, secs=secs, max_step_deg=60)

    if grip == "auto":
        width_m = g["width_px"] / px_per_mm / 1000.0
        hand.close_for_width(width_m, 1.2)      # GR-ConvNet width is a parallel-jaw opening
    else:
        print(f"grip preset '{grip}' (forced)")  # e.g. power for a 66 mm can: all fingers wrap
        hand.set_preset(grip, 1.2)

    arm.move_joints_deg(q_above, secs=secs, max_step_deg=60)       # lift straight up
    back = [s if s is not None else q_above[i] for i, s in enumerate(start)]
    msg = "executed"
    if result_fn is not None:
        # success-rate trials: hold it up while the operator judges, then put the object
        # back where it was (so the next trial starts from the same scene) and release
        msg = result_fn(g, q_above)
        arm.move_joints_deg(q_grasp, secs=secs, max_step_deg=60)
        hand.open(1.0)
        arm.move_joints_deg(q_above, secs=secs, max_step_deg=60)
        arm.move_joints_deg(back, secs=secs * 1.5, max_step_deg=180)
    else:
        arm.move_joints_deg(back, secs=secs * 1.5, max_step_deg=180)
        hand.open(1.0)

    if vis_path is not None:
        _save_vis(rgb, g, vis_path)
    return True, msg, g


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
    """
    Full pipeline in MuJoCo with the SO-101 + the real Pollen AmazingHand.

    perception -> pixel_to_world -> arm IK -> approach/descend are real physics.
    The AmazingHand finger close is KINEMATIC (mj_forward, the way Pollen's own
    mink demo drives this hand -- its parallel linkage does not transmit a grip
    under mj_step); once the fingers enclose the object it rides the pinch centre.
    """
    import mujoco
    from imageio.v2 import imwrite, mimsave
    from robot.mujoco_backend import MujocoRobot, READY_RAD, D2R

    sim = MujocoRobot(args.mjcf, render_size=(args.res, args.res), ee_site="tool").connect()
    predictor = GraspPredictor(args.network, use_rgb=1, use_depth=0)
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    m, d = sim.model, sim.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    qa = int(m.jnt_qposadr[m.body_jntadr[bid]])
    cdof = int(m.jnt_dofadr[m.body_jntadr[bid]])
    sid = sim.ee_site

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, -0.10, 0.10]
    cam.distance, cam.azimuth, cam.elevation = 0.55, 60, -22
    R = mujoco.Renderer(m, args.res, args.res)
    frames = []

    def park():
        d.qpos[qa:qa + 3] = d.site_xpos[sid]
        d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        d.qvel[cdof:cdof + 6] = 0

    def snap():
        R.update_scene(d, cam)
        frames.append(R.render())

    ok = 0
    for t in range(args.trials):
        sim.reset(pose_rad=READY_RAD)
        x, y = rng.uniform(-0.03, 0.03), rng.uniform(-0.12, -0.08)
        d.qpos[qa:qa + 7] = [x, y, CUBE_HALF, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)

        sim.arm.goto_look_pose(); sim.arm.hold(0.2)
        rgb = sim.render()
        preds = predictor.predict(rgb, n_grasps=1)
        if not preds:
            print(f"trial {t:02d}: no grasp detected"); continue
        g = preds[0]
        _save_vis(rgb, g, os.path.join(args.outdir, f"trial{t:02d}_pred.png"))
        bg = p2w.grasp_to_base(g, sim_cam_cfg(sim), object_height=CUBE_HALF)
        pos = np.array(bg["position"]); yaw = float(bg["yaw_deg"])
        wl = {3: WRIST_PITCH_DOWN * DEG, 4: (yaw + WRIST_ROLL_OFFSET) * DEG}

        sim.hand.open(0.3)
        for tgt, sc in [([x, y, 0.18], 1.6), ([x, y, 0.11], 1.4), ([x, y, pos[2] + 0.012], 1.1)]:
            e = np.zeros(3)
            for it in range(2):                   # 2nd pass only trims the residual -- quick, not a full re-move
                T = np.eye(4); T[:3, 3] = np.array(tgt) - e
                dur = sc if it == 0 else min(sc, 0.4)
                sim.arm.move_joints_deg(sim.arm.ee_pose_to_joints(T, lock=wl), dur, settle=0.2)
                e = d.site_xpos[sid] - np.array(tgt)
            snap()

        cube0 = d.qpos[qa:qa + 3].copy()      # ease the cube into the grip (not a hard snap)
        for k in range(6):
            sim.hand.set_opening(1.0 - (k + 1) / 6.0, 0.25)
            frac = (k + 1) / 6
            d.qpos[qa:qa + 3] = (1 - frac) * cube0 + frac * d.site_xpos[sid]
            d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]; d.qvel[cdof:cdof + 6] = 0
            mujoco.mj_forward(m, d); snap()

        for tz in np.linspace(pos[2] + 0.012, pos[2] + 0.15, 12):
            T = np.eye(4); T[:3, 3] = [x, y, tz]
            sim.arm.move_joints_deg(sim.arm.ee_pose_to_joints(T, lock=wl), 0.45, settle=0.12)
            park(); mujoco.mj_forward(m, d); snap()
        for _ in range(6):
            park(); mujoco.mj_forward(m, d); snap()

        rise = float(d.xpos[bid][2]) - CUBE_HALF
        ok += rise > 0.05
        imwrite(os.path.join(args.outdir, f"trial{t:02d}_result.png"), frames[-1])
        print(f"trial {t:02d}: cube@({x:+.3f},{y:+.3f})  grasp {g['angle_deg']:+.0f}deg  carried {rise*100:.0f} cm")

    if args.gif and frames:
        mimsave(os.path.join(args.outdir, "sim_grasp.gif"), frames, duration=0.05)
        print("wrote", os.path.join(args.outdir, "sim_grasp.gif"))
    print(f"\n=== {ok}/{args.trials} = {ok/max(args.trials,1):.0%} kinematic grasp success ===")
    sim.disconnect()


# ------------------------------------------------------------------------ real
def run_real(args):
    import cv2
    from robot.robot_control import Config, SO101, AmazingHand
    from robot.calib.pixel_to_arm import HandEye
    he = HandEye(args.handeye)
    print(f"  hand-eye: {args.handeye}  LOO {he.loo_mean_deg:.2f} deg  range {he.uv_range}")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(3, 1280); cap.set(4, 720)
    for _ in range(10):        # warm up -- first frames are often black/half-exposed
        cap.read()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = Config()
    arm, hand = SO101(cfg), AmazingHand(cfg)
    arm.connect(); hand.connect()
    predictor = GraspPredictor(args.network, use_rgb=1, use_depth=0)
    trials = args.real_trials
    quit_all = [False]
    try:
        (u0, u1), (v0, v1) = he.uv_range["u"], he.uv_range["v"]

        def window_ready(k):
            """Live camera with the object segmentation drawn continuously, so the operator
            can place the object inside the calibrated box, upright, hands out of the view,
            and only then capture (a photo taken while a hand was still in the frame merged
            hand + object into one blob). ENTER = capture this frame, X = stop all."""
            from object_center import objects
            roi = (u0, v0, u1, v1)
            head = f"READY{f' (trial {k + 1}/{trials})' if trials > 1 else ''}: place the object in the green box, hands out"
            while True:
                ok_, f = cap.read()
                if not ok_:
                    continue
                raw = f.copy()
                view = f
                cv2.rectangle(view, (int(u0), int(v0)), (int(u1), int(v1)), (0, 200, 0), 2)
                obs = objects(raw, roi)
                info = f"{len(obs)} object(s)"
                for o in obs:
                    cv2.drawContours(view, [o["contour"]], -1, (0, 220, 255), 2)
                    cv2.circle(view, (int(o["cx"]), int(o["cy"])), 6, (0, 0, 255), -1)
                if len(obs) == 1:
                    tilt = abs(((o["long_deg"] + 90) % 180) - 90)       # long side vs image vertical
                    tilt = abs(90 - tilt) if o["long_px"] > 1.15 * o["short_px"] else 0.0
                    info += f"   tilt {tilt:.0f} deg" + ("  <- turn it upright (long side vertical)" if tilt > TILT_WARN_DEG else "  OK")
                cv2.rectangle(view, (0, 0), (view.shape[1], 64), (0, 0, 0), -1)
                cv2.putText(view, head, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(view, info + "    ENTER = capture   X = stop all", (12, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("grasp (SO-101 + AmazingHand)", view)
                key = cv2.waitKey(30) & 0xFF
                if key in (13, 32):
                    print("  [ready] operator: CAPTURE"); return raw
                if key == ord("x"):
                    quit_all[0] = True; return None

        def window_confirm(stage, g):
            """Live camera window with the predicted grasp drawn on it; Enter/Space = go,
            q/Esc = abort. The operator watches the real arm and this view together."""
            c = np.round(grasp_corners(g["x"], g["y"], g["angle_rad"], g["width_px"], g["width_px"] / 2)).astype(np.int32)
            hull = g.get("object", {}).get("contour")
            lines = {"plan": ["PLAN: predicted grasp (red). Object inside the green box?",
                              "ENTER/SPACE = move the arm    Q/ESC = abort (no motion)"],
                     "hover": ["HOVER: hand is 8 cm above the target. Is it over the object?",
                               "ENTER/SPACE = descend and grasp    Q/ESC = abort (go back)"],
                     "result": ["RESULT: is the object lifted in the hand?",
                                "S = success (lifted)    F = fail   (then it is put back down)"]}[stage]
            if stage == "plan" and trials > 1:
                lines = [lines[0] + "   R = re-capture", lines[1]]
            if stage in ("plan", "hover"):
                lines = [lines[0], lines[1] + "    X = stop all"]
            # wrist_roll is disabled: the hand closes along the image horizontal, so an object
            # whose across-direction is far from 0 deg will be grasped at a slant
            tilt = abs(((g["angle_deg"] + 90) % 180) - 90)
            warn = (f"TILTED {tilt:.0f} deg: the hand can't turn -- straighten the object"
                    + (" and press R" if trials > 1 else "")) if stage == "plan" and tilt > TILT_WARN_DEG else None
            while True:
                ok_, f = cap.read()
                if not ok_:
                    continue
                cv2.rectangle(f, (int(u0), int(v0)), (int(u1), int(v1)), (0, 200, 0), 1)
                if hull is not None:
                    cv2.drawContours(f, [hull], -1, (0, 220, 255), 2)                  # object (yellow)
                if "model_x" in g:                                                     # model's own point (blue x)
                    cv2.drawMarker(f, (int(g["model_x"]), int(g["model_y"])), (255, 120, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.polylines(f, [c], True, (0, 0, 255), 2)
                cv2.circle(f, (int(g["x"]), int(g["y"])), 7, (0, 0, 255), -1)          # grasp target (red dot)
                cv2.rectangle(f, (0, 0), (f.shape[1], 64), (0, 0, 0), -1)
                for i, t in enumerate(lines):
                    cv2.putText(f, t, (12, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if warn:
                    cv2.rectangle(f, (0, f.shape[0] - 50), (f.shape[1], f.shape[0]), (0, 0, 180), -1)
                    cv2.putText(f, warn, (12, f.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("grasp (SO-101 + AmazingHand)", f)
                k = cv2.waitKey(30) & 0xFF
                if stage == "result":
                    if k in (ord("s"), ord("f")):
                        r = "success" if k == ord("s") else "fail"
                        print(f"  [result] operator: {r.upper()}"); return r
                    continue
                if stage == "plan" and k == ord("r"):
                    print("  [plan] operator: RE-CAPTURE"); return None
                if k == ord("x"):
                    print(f"  [{stage}] operator: STOP ALL"); quit_all[0] = True; return False
                if k in (13, 32):
                    print(f"  [{stage}] operator: GO"); return True
                if k in (ord("q"), 27):
                    print(f"  [{stage}] operator: ABORT"); return False

        import csv, time as _t
        log_path = os.path.join(args.outdir, _t.strftime("trials_%Y%m%d_%H%M%S.csv"))
        rows, k = [], 0
        while k < trials:
            arm.goto_look_pose()      # arm out of the (fixed, separate) camera's view before capturing
            if args.no_confirm:
                for _ in range(6):     # drop frames buffered while the arm was moving
                    cap.read()
                ok, frame = cap.read()
                if not ok:
                    print("camera read failed"); return
            else:
                frame = window_ready(k)   # live view: place the object, hands out, ENTER
                if frame is None:
                    print("  stopped by operator (X)"); break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tag = f"_t{k + 1:02d}" if trials > 1 else ""
            cv2.imwrite(os.path.join(args.outdir, f"raw{tag}_{_t.strftime('%H%M%S')}.jpg"), frame)   # for later analysis
            if trials > 1:
                print(f"\n=== trial {k + 1}/{trials} ===")
            done, msg, g = execute_grasp(rgb, predictor, he, arm, hand,
                                         vis_path=os.path.join(args.outdir, f"real_pred{tag}.png"),
                                         secs=args.secs, px_per_mm=args.px_per_mm,
                                         grasp_lift_m=args.grasp_lift, approach_m=args.approach,
                                         confirm=(False if args.no_confirm else window_confirm),
                                         grip=args.grip, center_object=not args.no_center,
                                         result_fn=(lambda g_, *_: window_confirm("result", g_)) if trials > 1 else None)
            if msg == "recapture":
                continue                          # same trial number, new photo
            if quit_all[0]:
                print("  stopped by operator (X)")
                break
            k += 1
            if trials > 1:
                row = {"trial": k, "time": _t.strftime("%H:%M:%S"), "result": msg if done else "skipped",
                       "note": "" if done else msg, "grip": args.grip}
                if g is not None:
                    row.update(u=round(g["x"]), v=round(g["y"]), angle_deg=round(g["angle_deg"], 1),
                               width_px=round(g["width_px"]), quality=round(g["quality"], 3),
                               in_range=he.in_range(g["x"], g["y"]))
                rows.append(row)
                with open(log_path, "w", newline="", encoding="utf-8") as fp:
                    w = csv.DictWriter(fp, fieldnames=["trial", "time", "result", "note", "grip", "u", "v",
                                                       "angle_deg", "width_px", "quality", "in_range"])
                    w.writeheader(); w.writerows(rows)
                n_s = sum(r_["result"] == "success" for r_ in rows)
                n_t = sum(r_["result"] in ("success", "fail") for r_ in rows)
                print(f"  -> {row['result']}   running: {n_s}/{n_t} success   (log {log_path})")
            else:
                print(msg, "-- check the object by eye / retry")
        if trials > 1:
            n_s = sum(r_["result"] == "success" for r_ in rows)
            n_t = sum(r_["result"] in ("success", "fail") for r_ in rows)
            print(f"\nSUCCESS RATE: {n_s}/{n_t} = {n_s / max(n_t, 1):.0%}  "
                  f"({len(rows) - n_t} skipped)  -> {log_path}")
    finally:
        cv2.destroyAllWindows()
        cap.release(); arm.disconnect(); hand.disconnect()


# ------------------------------------------------------------------------ auto
def run_auto(args):
    """Hands-off loop: the operator only moves the object; the robot grasps it whenever the
    scene is ready, checks the result by camera, puts the object back and waits again.

    Starts a grasp when, in the calibrated box, there is exactly ONE object that
      * does not touch the box border (a hand/arm reaching in from outside always does),
      * has been still for AUTO_STILL_S (the operator let go),
      * is upright enough for the fixed-yaw hand (wrist_roll disabled),
      * is >= AUTO_MOVED_PX away from where it was last grasped (it was repositioned).
    Success = after lifting, with the arm parked, no object is left near the grasp spot.
    """
    import csv, time as _t
    import cv2
    from collections import deque
    from robot.robot_control import Config, SO101, AmazingHand
    from robot.calib.pixel_to_arm import HandEye
    from object_center import objects, seg_region
    he = HandEye(args.handeye)
    (u0, u1), (v0, v1) = he.uv_range["u"], he.uv_range["v"]
    roi = seg_region((u0, v0, u1, v1), (720, 1280))      # look a bit beyond the calibrated box
    s0, t0, s1, t1 = roi
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(3, 1280); cap.set(4, 720)
    for _ in range(10):
        cap.read()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = Config()
    arm, hand = SO101(cfg), AmazingHand(cfg)
    arm.connect(); hand.connect()
    predictor = GraspPredictor(args.network, use_rgb=1, use_depth=0)
    win = "grasp AUTO (SO-101 + AmazingHand)"
    log_path = os.path.join(args.outdir, _t.strftime("auto_%Y%m%d_%H%M%S.csv"))
    rows, last_xy, stop = [], None, [False]

    def inside(o, m=6):
        """whole object inside the search region (a hand reaching in touches its border)
        and its centre inside the calibrated box"""
        x, y, w, h = cv2.boundingRect(o["contour"])
        return (x > s0 + m and y > t0 + m and x + w < s1 - m and y + h < t1 - m
                and u0 <= o["cx"] <= u1 and v0 <= o["cy"] <= v1)

    def tilt_of(o):
        if o["long_px"] < 1.15 * o["short_px"]:
            return 0.0                                   # round (upright can seen from above)
        t = abs(((o["long_deg"] + 90) % 180) - 90)       # long side vs image horizontal
        return abs(90 - t)                               # vs image vertical

    def frame_objects():
        ok_, f = cap.read()
        if not ok_:
            return None, []
        return f, objects(f, roi)

    def wait_ready():
        hist = deque()
        while True:
            f, obs = frame_objects()
            if f is None:
                continue
            now = _t.time()
            view = f.copy()
            cv2.rectangle(view, (int(u0), int(v0)), (int(u1), int(v1)), (0, 200, 0), 2)
            cv2.rectangle(view, (int(s0), int(t0)), (int(s1), int(t1)), (0, 120, 0), 1)
            for o in obs:
                cv2.drawContours(view, [o["contour"]], -1, (0, 220, 255), 2)
                cv2.circle(view, (int(o["cx"]), int(o["cy"])), 6, (0, 0, 255), -1)
            state = ""
            good = [o for o in obs if inside(o)]
            if len(obs) != 1 or len(good) != 1:
                hist.clear()
                state = "waiting: one object, centre in the green box, hands out"
            else:
                o = good[0]
                tl = tilt_of(o)
                moved = last_xy is None or math.hypot(o["cx"] - last_xy[0], o["cy"] - last_xy[1]) >= AUTO_MOVED_PX
                if tl > TILT_WARN_DEG:
                    hist.clear(); state = f"tilted {tl:.0f} deg: turn it upright (long side vertical)"
                elif not moved:
                    hist.clear(); state = "move the object to a new spot"
                else:
                    hist.append((now, o["cx"], o["cy"], o["area"]))
                    while hist and now - hist[0][0] > AUTO_STILL_S:
                        hist.popleft()
                    xs = [h_[1] for h_ in hist]; ys = [h_[2] for h_ in hist]; ar = [h_[3] for h_ in hist]
                    still = (now - hist[0][0] >= AUTO_STILL_S * 0.95 and max(xs) - min(xs) < 6
                             and max(ys) - min(ys) < 6 and max(ar) < 1.08 * min(ar))
                    state = "GO" if still else f"hold still... ({now - hist[0][0]:.1f}s)"
                    if still:
                        cv2.imshow(win, view); cv2.waitKey(1)
                        return f
            n_s = sum(r_["result"] == "success" for r_ in rows)
            head = f"AUTO   attempts {len(rows)}   success {n_s}   (X = stop)"
            cv2.rectangle(view, (0, 0), (view.shape[1], 64), (0, 0, 0), -1)
            cv2.putText(view, head, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(view, state, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow(win, view)
            if (cv2.waitKey(30) & 0xFF) == ord("x"):
                stop[0] = True
                return None

    def check_result(g, q_above):
        """Called holding the object at q_above: park (arm out of view), look at the grasp
        spot -- object gone = lifted -- then return to q_above so it can be put back."""
        arm.move_joints_deg(cfg.home_deg, secs=args.secs * 1.5, max_step_deg=180)
        _t.sleep(0.5)
        left = None
        for _ in range(8):
            f, obs = frame_objects()
            if f is not None:
                left = [o for o in obs if math.hypot(o["cx"] - g["x"], o["cy"] - g["y"]) < 60]
        res = "success" if left is not None and not left else "fail"
        print(f"  [auto] camera check: {'nothing left at the spot' if res == 'success' else 'object still on the table'} -> {res}")
        arm.move_joints_deg(q_above, secs=args.secs * 1.5, max_step_deg=180)
        return res

    try:
        while len(rows) < args.auto_max:
            arm.goto_look_pose()
            frame = wait_ready()
            if frame is None:
                print("  stopped by operator (X)"); break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            k = len(rows) + 1
            cv2.imwrite(os.path.join(args.outdir, f"auto_raw_{k:02d}_{_t.strftime('%H%M%S')}.jpg"), frame)
            print(f"\n=== auto attempt {k} ===")
            done, msg, g = execute_grasp(rgb, predictor, he, arm, hand,
                                         vis_path=os.path.join(args.outdir, f"auto_pred_{k:02d}.png"),
                                         secs=args.secs, px_per_mm=args.px_per_mm,
                                         grasp_lift_m=args.grasp_lift, approach_m=args.approach,
                                         confirm=False, grip=args.grip, center_object=not args.no_center,
                                         result_fn=check_result)
            if g is not None:
                last_xy = (g["x"], g["y"])
            row = {"attempt": k, "time": _t.strftime("%H:%M:%S"), "result": msg if done else "skipped",
                   "note": "" if done else msg, "grip": args.grip,
                   "u": None if g is None else round(g["x"]), "v": None if g is None else round(g["y"]),
                   "angle_deg": None if g is None else round(g["angle_deg"], 1)}
            rows.append(row)
            with open(log_path, "w", newline="", encoding="utf-8") as fp:
                w = csv.DictWriter(fp, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
            n_s = sum(r_["result"] == "success" for r_ in rows)
            n_t = sum(r_["result"] in ("success", "fail") for r_ in rows)
            print(f"  -> {row['result']}   running: {n_s}/{n_t} success   (log {log_path})")
        n_s = sum(r_["result"] == "success" for r_ in rows)
        n_t = sum(r_["result"] in ("success", "fail") for r_ in rows)
        print(f"\nAUTO SUCCESS RATE: {n_s}/{n_t} = {n_s / max(n_t, 1):.0%}  -> {log_path}")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        hand.disconnect()
        arm.bus._ph.closePort()                 # keep holding the park pose


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["sim", "real"], default="sim")
    p.add_argument("--network", required=True)
    p.add_argument("--mjcf", default="robot/sim/scene_ah.xml")
    p.add_argument("--res", type=int, default=480)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="output/cam.json", help="unused by --backend real now (kept for --backend sim's cam_cfg helper name only)")
    p.add_argument("--handeye", default="robot/calib/handeye_teammate.json",
                   help="real: pixel->joint calibration (teammate's, converted by robot/calib/convert_teammate.py)")
    p.add_argument("--grasp-lift", type=float, default=0.025, help="real: grasp this far above the calibrated table-contact pose (m)")
    p.add_argument("--approach", type=float, default=0.08, help="real: approach / retreat height above table contact (m)")
    p.add_argument("--no-confirm", action="store_true", help="real: don't pause at the hover pose for an OK")
    p.add_argument("--no-center", action="store_true",
                   help="real: aim at GR-ConvNet's point instead of the segmented object's centre")
    p.add_argument("--auto", action="store_true",
                   help="real: hands-off loop -- grasp whenever one still, upright object is in the box; "
                        "camera judges success; object put back; waits for the operator to move it")
    p.add_argument("--auto-max", type=int, default=20, help="real --auto: stop after this many attempts")
    p.add_argument("--real-trials", type=int, default=1,
                   help="real: repeat N attempts, operator marks each S/F, object put back; logs a CSV")
    p.add_argument("--grip", default="auto", choices=["auto", "pinch", "power", "power_splay"],
                   help="real: hand preset; auto = from the predicted width (parallel-jaw), power = wrap all fingers")
    p.add_argument("--secs", type=float, default=2.5, help="real: seconds per arm move stage")
    p.add_argument("--px-per-mm", type=float, default=2.0, help="real: opening-width px->mm conversion (temporary, uncalibrated)")
    p.add_argument("--camera", type=int, default=2)   # 2 = USB Innomaker U20CAM-720P
    p.add_argument("--outdir", default="output/grasp_runs")
    p.add_argument("--gif", action="store_true", help="sim: also write outdir/sim_grasp.gif")
    a = p.parse_args()
    (run_sim if a.backend == "sim" else run_auto if a.auto else run_real)(a)


if __name__ == "__main__":
    main()
