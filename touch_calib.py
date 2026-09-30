"""
touch_calib.py -- tie the checkerboard (world) frame to the SO-101 base frame by
touching known checkerboard corners with the end-effector, so the calibrated
camera pose (from extrinsic_click.py, which is in the CHECKERBOARD frame) becomes
usable in the ROBOT BASE frame.

Idea
----
extrinsic_click.py gave  T_board_cam  (camera pose in the checkerboard frame),
stored as `T_base_cam` in cam.json. Here we measure  T_base_board  by matching
>= 3 checkerboard corners (known board coords) to where the end-effector tip is
when it touches them (base-frame coords via the SO-101 forward kinematics), then

    T_base_cam(real)  =  T_base_board @ T_board_cam

and the table plane is carried into the base frame too. cam.json is rewritten.

Workflow (offline -- decoupled from flaky bus reads)
---------------------------------------------------
1.  Keep the checkerboard exactly where it was for extrinsic_click.py.
2.  python touch_calib.py --template poses.json
3.  For each corner: pose / jog the arm so the tip touches that inner corner,
    read the 5 joint angles (robot_control.py --read-pose or --jog), and fill a
    row in poses.json:  {"corner": [col, row], "joints_deg": [j0,j1,j2,j3,j4]}
    Use >= 3 non-collinear corners (5-6 spread across the board is better).
4.  python touch_calib.py --solve poses.json --grid 6x5 --square 0.020

The "tip" is the `--tool-site` of `--mjcf` (default: the AmazingHand pinch centre
of robot/sim/so101_amazinghand.xml). Touch corners with that point -- e.g. close
the hand and touch with the closed fingertips, or tape a pointer there and set
--tool-offset.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

import pixel_to_world as p2w


# ------------------------------------------------------- live jog + capture
def jog_capture(out_path, grid, mjcf="robot/sim/so101_amazinghand.xml",
                tool_site="tool", base_body="Base", free=False):
    """Jog the arm until the closed fingertips touch a checkerboard inner
    corner, then capture (col,row) + pose.

    free=False (default): torque ON, move with typed commands.
        <j> <deg>      nudge arm joint j (1-5) by <deg>          e.g.  2 -8
        x|y|z <mm>     nudge the tool point in the base frame    e.g.  z -5
    free=True: torque OFF, move the arm BY HAND instead -- no nudge commands,
        the printed pose is a live read every prompt.

    Commands (either mode):
        grip | open    close / open the AmazingHand
        c <col> <row>  capture: fingertips are touching inner corner (col,row)
        u              undo the last capture
        p              reprint pose + captures
        w              write <out_path> and quit
        q              quit without writing
    """
    from robot.robot_control import Config, SO101, AmazingHand, ADDR_TORQUE_ENABLE
    gx, gy = grid
    cfg = Config()
    arm = SO101(cfg)
    arm.connect()
    names = [j.name for j in arm.joints][:5]

    def hand_preset(name):
        """Open/close/disconnect the hand for ONE command, never held open at the
        same time as the arm's reads -- arm+hand connected together desynced
        the arm's bus (confirmed: reads came back COMM_SUCCESS but with garbage
        values, repeatable, the moment both were open)."""
        try:
            h = AmazingHand(cfg)
            h.connect()
            h.set_preset(name, 1.0)
            h.disconnect()
            return True
        except Exception as e:
            print(f"  hand: not connected ({e}) -- arm only; touch with a taped pointer + --tool-offset")
            return False

    if hand_preset("open"):
        print("  hand: opened (fingers straight) -- touch with ONE fingertip's real site "
              "(e.g. --tool-site ah_tip1), NOT a clenched fist -- closed fingers hide that "
              "point inside the curl and it can never reach the table.")

    ik = FK(mjcf, tool_site, base_body)
    caps = []

    if free:
        for _ in range(5):                    # retry -- a single disable write can be lost
            for i in arm.bus.ids:
                arm.bus._pk.write1ByteTxRx(arm.bus._ph, i, ADDR_TORQUE_ENABLE, 0)
            time.sleep(0.05)
        print("  torque OFF -- move the arm BY HAND. No brake -- support it, don't let it drop.")
    else:
        pose = [float(v) for v in cfg.look_deg]          # start from the known "look" pose
        arm.move_joints_deg(pose, secs=2.0, max_step_deg=140)

    print("\n  " + ("(free -- torque off, move by hand)   "
                    if free else "jog: '<joint 1-5> <deg>' | 'x|y|z <mm>'   ")
          + "grip | open   capture:'c <col> <row>'  undo:'u'  print:'p'  write:'w'  quit:'q'\n"
          f"  grid is {gx} x {gy} inner corners; corner (0,0) is the extrinsic_click origin.\n")
    def read_firm(tries=8, pause=0.05):
        """A full 5-joint read with NO Nones, or None if it never lands
        (bus is flaky -- see read_one's own retries; this retries the WHOLE
        5-joint read, since one bad joint shouldn't spoil the others)."""
        last = None
        for _ in range(tries):
            last = arm.read_joints_deg()
            if all(v is not None for v in last):
                return last
            time.sleep(pause)
        return None

    try:
        while True:
            if free:
                live = arm.read_joints_deg()          # display-only -- capture re-reads firmly
                shown = " ".join(f"{n}={'   n ' if v is None else f'{v:6.1f}'}"
                                  for n, v in zip(names, live))
                print("  pose:", shown, f"  [{len(caps)} captured]")
            else:
                tip = ik.tip_in_base(pose)
                print("  pose:", "  ".join(f"{n}={p:6.1f}" for n, p in zip(names, pose)),
                      f"  tip(base) {np.round(tip * 1000, 1)} mm   [{len(caps)} captured]")
            s = input("  > ").strip().lower()
            if s == "q":
                print("quit, nothing written"); return
            if s == "w":
                break
            if s == "p":
                for cp in caps:
                    print("   ", cp)
                continue
            if s == "u":
                if caps:
                    print("   dropped", caps.pop())
                continue
            if s in ("grip", "close"):
                hand_preset("power")
                continue
            if s == "open":
                hand_preset("open")
                continue
            p = s.split()
            if len(p) == 3 and p[0] == "c":
                c, r = int(p[1]), int(p[2])
                if not (0 <= c < gx and 0 <= r < gy):
                    print(f"   corner out of range (0..{gx-1}, 0..{gy-1})"); continue
                if free:
                    cur = read_firm()
                    if cur is None:
                        print("   read failed (bus flaky) -- hold still and try 'c' again"); continue
                else:
                    cur = pose
                caps.append({"corner": [c, r], "joints_deg": [round(v, 2) for v in cur]})
                print(f"   captured corner ({c},{r}) at {caps[-1]['joints_deg']}")
                continue
            if free:
                print("   free mode: nothing to command -- move it by hand, then 'c <col> <row>'")
                continue
            if len(p) == 2 and p[0] in ("x", "y", "z"):
                try:
                    mm = float(p[1])
                except ValueError:
                    print("   numbers only"); continue
                vec = {"x": (mm, 0, 0), "y": (0, mm, 0), "z": (0, 0, mm)}[p[0]]
                tgt = ik.solve_delta(pose, np.array(vec) / 1000.0)
                if max(abs(a - b) for a, b in zip(tgt, pose)) > 45:
                    print("   IK wants a >45 deg jump (limit/singularity) -- ignored"); continue
                arm.move_joints_deg(tgt, secs=0.7, max_step_deg=45)
                pose = tgt
                continue
            if len(p) == 2:
                try:
                    k, amt = int(p[0]) - 1, float(p[1])
                except ValueError:
                    print("   numbers only"); continue
                if not 0 <= k < 5:
                    print("   joint 1-5"); continue
                tgt = list(pose); tgt[k] += amt
                arm.move_joints_deg(tgt, secs=max(0.4, abs(amt) / 20), max_step_deg=45)
                pose[k] = tgt[k]
                continue
            print("   ?  '<j> <deg>' | 'x|y|z <mm>' | grip | open | 'c <col> <row>' | u | p | w | q")
    finally:
        arm.disconnect()

    if len(caps) < 3:
        print(f"only {len(caps)} captures -- need >= 3, not writing"); return
    json.dump({"_note": "from touch_calib.py --jog", "poses": caps}, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path} with {len(caps)} corners.  now:\n"
          f"  python touch_calib.py --solve {out_path} --grid {gx}x{gy} --square <m>")


# --------------------------------------------------------------- FK (via MuJoCo)
class FK:
    def __init__(self, mjcf, tool_site, base_body, tool_offset=(0, 0, 0)):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        self.joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self.jadr = [self.m.jnt_qposadr[self._id("JOINT", j)] for j in self.joints]
        self.dofadr = [self.m.jnt_dofadr[self._id("JOINT", j)] for j in self.joints]
        self.sid = self._id("SITE", tool_site)
        self.bid = self._id("BODY", base_body)
        self.off = np.asarray(tool_offset, float)

    def _id(self, kind, name):
        i = self.mj.mj_name2id(self.m, getattr(self.mj.mjtObj, f"mjOBJ_{kind}"), name)
        if i < 0:
            raise KeyError(f"{kind} {name!r} not in {os.path.basename('mjcf')}")
        return i

    def tip_in_base(self, joints_deg):
        d = self.d
        for adr, v in zip(self.jadr, np.radians(joints_deg)):
            d.qpos[adr] = v
        self.mj.mj_forward(self.m, d)
        p_w = d.site_xpos[self.sid] + d.site_xmat[self.sid].reshape(3, 3) @ self.off
        R_b = d.xmat[self.bid].reshape(3, 3)
        return R_b.T @ (p_w - d.xpos[self.bid])          # tip in the base body frame

    def solve_delta(self, joints_deg, dxyz_base, iters=200, tol=2e-4):
        """Joint angles (deg) that move the tool site by dxyz_base (m, base frame)
        from the given pose. DLS position IK on the 5 arm joints."""
        d = self.d
        q = np.radians(np.asarray(joints_deg, float))
        R_b = None
        target = None
        for _ in range(iters):
            for adr, v in zip(self.jadr, q):
                d.qpos[adr] = v
            self.mj.mj_forward(self.m, d)
            if target is None:
                R_b = d.xmat[self.bid].reshape(3, 3)
                target = d.site_xpos[self.sid] + R_b @ np.asarray(dxyz_base, float)
            err = target - d.site_xpos[self.sid]
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, self.m.nv))
            self.mj.mj_jacSite(self.m, d, jacp, None, self.sid)
            J = jacp[:, self.dofadr]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            q = np.clip(q + 0.5 * dq, -np.pi, np.pi)
        return [float(v) for v in np.degrees(q)]


# ------------------------------------------------------------------- kabsch fit
def rigid_fit(A, B):
    """Least-squares R,t with  R@A_i + t ~= B_i  (A,B: N x 3). Returns 4x4."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    resid = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return T, resid


# ---------------------------------------------------------- solve + write cam.json
def solve_and_write(board, base, config_path):
    """board, base: N x 3 corresponding points (checkerboard frame, robot base
    frame). Fits T_base_board, composes it with cam.json's current (board-frame)
    T_base_cam, and rewrites cam.json in the base frame."""
    board, base = np.asarray(board, float), np.asarray(base, float)
    T_base_board, resid = rigid_fit(board, base)
    print(f"touch fit over {len(board)} points:  residual mean {resid.mean()*1000:.1f} mm  "
          f"max {resid.max()*1000:.1f} mm")
    if resid.max() > 0.01:
        print("  WARNING max residual > 10 mm -- a touch/point was off")

    cfg = json.load(open(config_path))
    T_board_cam = np.asarray(cfg["T_base_cam"], float).reshape(4, 4)   # camera in BOARD frame
    T_base_cam = T_base_board @ T_board_cam

    R, t = T_base_board[:3, :3], T_base_board[:3, 3]
    plane_pt = t.tolist()
    plane_n = (R @ np.array([0, 0, 1.0])).tolist()

    cfg["T_base_cam"] = T_base_cam.tolist()
    cfg["table_plane"] = {"point": plane_pt, "normal": plane_n}
    cfg["T_base_board"] = T_base_board.tolist()
    cfg["touch_residual_mm"] = float(resid.max() * 1000)
    json.dump(cfg, open(config_path, "w"), indent=2)

    cp = T_base_cam[:3, 3]
    print("camera position in SO-101 BASE frame (m):", np.round(cp, 4))
    print("table plane in base frame:  point", np.round(plane_pt, 4), " normal", np.round(plane_n, 3))
    print(f"wrote base-frame T_base_cam + table_plane -> {config_path}")
    print("cam.json is now robot-ready: predict_grasp -> pixel_to_world gives base-frame poses.")


# ----------------------------------------------- solve from freehand pixel touches
def solve_from_pixels(csv_path, config_path, mjcf, tool_site, base_body, tool_offset):
    """Alternative to --solve: reuse freehand pointer-touch points (pixel u,v +
    5 joint angles, NOT tied to checkerboard grid corners -- e.g. collect.py's
    points.csv from a different session) IF the camera and checkerboard have
    NOT moved since extrinsic_click.py. Each touched pixel's board-frame 3D
    point is recovered by ray-casting it through the current (board-frame)
    camera calibration instead of grid math, then fit exactly like --solve.
    """
    cfg = json.load(open(config_path))
    K = np.asarray(cfg["K"], float).reshape(3, 3)
    dist = np.asarray(cfg.get("dist", [0] * 5), float).ravel()
    T_board_cam = np.asarray(cfg["T_base_cam"], float).reshape(4, 4)   # still board-frame here
    model = cfg.get("model", "pinhole")

    rows = list(csv.DictReader(open(csv_path)))
    if len(rows) < 3:
        raise SystemExit("need >= 3 points")
    fk = FK(mjcf, tool_site, base_body, tool_offset)
    board, base = [], []
    for r in rows:
        o, d = p2w.pixel_to_ray([float(r["u"]), float(r["v"])], K, dist, T_board_cam, model)
        pt = p2w.ray_plane_intersect(o, d, np.zeros(3), np.array([0, 0, 1.0]))
        if pt is None:
            print(f"  skip ({r['u']},{r['v']}): ray parallel to / away from the table plane")
            continue
        board.append(pt)
        joints = [float(r[j]) for j in
                  ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
        base.append(fk.tip_in_base(joints))
    print(f"{len(board)}/{len(rows)} points ray-cast onto the table plane")
    solve_and_write(board, base, config_path)


# ------------------------------------------------------------------------ main
def parse_grid(s):
    a, b = s.lower().split("x"); return int(a), int(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", metavar="poses.json", help="write a blank poses file and exit")
    ap.add_argument("--jog", metavar="poses.json", help="live: jog the SO-101, touch corners, capture -> write this file")
    ap.add_argument("--free", action="store_true", help="--jog with torque OFF: move the arm by hand instead of typed nudges")
    ap.add_argument("--solve", metavar="poses.json", help="solve from a filled poses file")
    ap.add_argument("--solve-pixels", metavar="points.csv",
                    help="solve from freehand pointer-touch points (u,v + 5 joint angles, "
                         "e.g. collect.py's points.csv) -- ONLY valid if the camera/board "
                         "haven't moved since extrinsic_click.py")
    ap.add_argument("--grid", default="6x5", help="inner corners, e.g. 6x5")
    ap.add_argument("--square", type=float, default=0.020, help="checkerboard square size (m)")
    ap.add_argument("--config", default="output/cam.json")
    ap.add_argument("--mjcf", default="robot/sim/so101_amazinghand.xml")
    ap.add_argument("--tool-site", default="tool")
    ap.add_argument("--base-body", default="Base")
    ap.add_argument("--tool-offset", type=float, nargs=3, default=(0, 0, 0),
                    help="extra tip offset in the tool-site frame (m), e.g. a pointer")
    a = ap.parse_args()

    if a.template:
        json.dump({
            "_note": "corner = [col,row] inner-corner index (0-based); joints_deg = "
                     "[Rotation,Pitch,Elbow,Wrist_Pitch,Wrist_Roll] when the tip touches it. "
                     ">=3 non-collinear rows.",
            "poses": [{"corner": [0, 0], "joints_deg": [0, 0, 0, 0, 0]},
                      {"corner": [5, 0], "joints_deg": [0, 0, 0, 0, 0]},
                      {"corner": [0, 4], "joints_deg": [0, 0, 0, 0, 0]},
                      {"corner": [5, 4], "joints_deg": [0, 0, 0, 0, 0]}],
        }, open(a.template, "w"), indent=2)
        print("wrote", a.template)
        return

    if a.jog:
        jog_capture(a.jog, parse_grid(a.grid), a.mjcf, a.tool_site, a.base_body, free=a.free)
        return

    if a.solve_pixels:
        solve_from_pixels(a.solve_pixels, a.config, a.mjcf, a.tool_site, a.base_body, a.tool_offset)
        return

    if not a.solve:
        ap.error("give --template, --jog, --solve, or --solve-pixels")

    data = json.load(open(a.solve))["poses"]
    if len(data) < 3:
        ap.error("need >= 3 poses")
    gx, gy = parse_grid(a.grid)

    fk = FK(a.mjcf, a.tool_site, a.base_body, a.tool_offset)
    board, base = [], []
    for row in data:
        c, r = row["corner"]
        if not (0 <= c < gx and 0 <= r < gy):
            ap.error(f"corner {row['corner']} outside the {gx}x{gy} grid")
        board.append([c * a.square, r * a.square, 0.0])
        base.append(fk.tip_in_base(row["joints_deg"]))
    solve_and_write(board, base, a.config)


if __name__ == "__main__":
    main()
