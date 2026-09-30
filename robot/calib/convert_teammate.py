r"""
convert_teammate.py -- the teammate's (A팀, LeRobot) pixel->joint calibration re-expressed
in OUR joint convention (robot_control.JointCfg), in pixel_to_arm.HandEye's format.

    python -m robot.calib.convert_teammate [--team-dir <...\amazinghand>]
        -> robot/calib/handeye_teammate.json   (never touches handeye.json)

Angle bridge (exact at the servo-register level, no measurement needed):
  theirs: LeRobot DEGREES with calibration range 0..4095 (their calibrate_joints.py opens
          the range; our read of the servos' Min/Max_Position_Limit also shows 0..4095)
            deg_t = (tick - 2047.5) * 360 / 4095
  ours:   deg_o = sign * (tick - home_steps) * 360 / 4096
Both read the same Present_Position register (which already includes the servo's EEPROM
Homing_Offset), so deg_o is an exact affine function of deg_t -- VALID ONLY IF the servos'
Homing_Offset hasn't changed between their collection and now.

Their marker offset (sticker -> finger meeting point, px) and fitted wrist_flex are baked
in: the polynomial is re-fitted on a dense grid over their valid pixel range (a shifted
and affinely-mapped quadratic is still quadratic, so this is exact up to float error).

2026-09-30: robot_control switched to the teammate's frame (home_steps=2048), so this
mapping is ~identity (-0.04 deg). Before that, our 09-17 home_steps disagreed with it by
~100 deg on lift / wrist_flex -- the servos had been re-homed by the teammate's LeRobot
calibration, which the team treats as the accurate one. Our own teach/points data were
recorded in the old frame and are no longer valid.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from robot.robot_control import Config   # noqa: E402

TEAM_DIR = r"C:\Users\135\Documents\카카오톡 받은 파일\leader-follower-arm-grasp\leader-follower-arm-grasp\amazinghand"
OUT = os.path.join(HERE, "handeye_teammate.json")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def features(uv):
    uv = np.atleast_2d(np.asarray(uv, float)) / 1000.0
    u, v = uv[:, 0], uv[:, 1]
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=TEAM_DIR)
    a = ap.parse_args()
    he = json.load(open(os.path.join(a.team_dir, "handeye.json"), encoding="utf-8"))
    if he.get("type") != "pixel_to_joint_poly2":
        raise SystemExit(f"unexpected type {he.get('type')}")
    cfg = {j.name: j for j in Config().arm_joints}

    def to_ours(name, deg_t):
        tick = np.asarray(deg_t) * 4095 / 360 + 2047.5
        j = cfg[name]
        return j.sign * (tick - j.home_steps) * 360 / 4096

    C = np.array(he["coef"]); fj = he["fit_joints"]
    du, dv = he.get("marker_offset_px", [0.0, 0.0])
    (u0, u1), (v0, v1) = he["uv_range"]["u"], he["uv_range"]["v"]
    U, V = np.meshgrid(np.linspace(u0, u1, 40), np.linspace(v0, v1, 30))
    uv = np.c_[U.ravel(), V.ravel()]                               # target (finger meeting) pixel
    q_t = features(uv + [du, dv]) @ C                              # their fit is sticker-pixel based
    q_o = np.stack([to_ours(k, q_t[:, i]) for i, k in enumerate(fj)], axis=1)
    coef, *_ = np.linalg.lstsq(features(uv), q_o, rcond=None)
    resid = np.abs(features(uv) @ coef - q_o).max()

    roll_t = float(np.median([float(x) for x in
                              [ln.split(",")[6] for ln in open(os.path.join(a.team_dir, "handeye_points.csv"),
                                                              encoding="utf-8").read().split("\n")[1:] if ln]]))
    roll_o = float(to_ours("wrist_roll", roll_t))
    out = {
        "type": "pixel_to_joint_poly2",
        "fit_joints": fj,
        "coef": coef.tolist(),
        "wrist_model": "fitted" if "wrist_flex" in fj else "const",
        "wrist_flex_const_deg": float(to_ours("wrist_flex", he.get("wrist_flex_const_deg", 0.0))),
        "wrist_roll_ref_deg": roll_o,
        "theta_gain": 1.0,
        "theta_offset_deg": roll_o,
        "n_points": he.get("n_points"),
        "loo_per_joint_deg": he.get("loo_mean_abs_deg"),
        "loo_mean_deg": float(np.mean(list((he.get("loo_mean_abs_deg") or {"_": float("nan")}).values()))),
        "uv_range": {"u": [u0, u1], "v": [v0, v1]},
        "source": "teammate (A팀) LeRobot calibration, converted by convert_teammate.py",
        "bridge": {k: {"a": 4095 / 4096 * cfg[k].sign,
                       "b": cfg[k].sign * (2047.5 - cfg[k].home_steps) * 360 / 4096} for k in JOINTS},
        "verified": "teammate: 22 pts, LOO 1.23 deg, 12 mm mean tip error on unseen spots (their HANDOVER.md)",
        "note": "Pixel = where the fingers should meet (their marker offset baked in). Collected with the "
                "fingers ON the table: lift the tool (~25 mm, robot/team_fk.lift_tool) before moving.",
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {OUT}  (refit residual {resid:.2e} deg, wrist_roll ref {roll_o:+.1f} deg ours)")
    for k in JOINTS:
        b = out["bridge"][k]
        print(f"  {k:14} ours = {b['a']:.5f} * theirs {b['b']:+7.2f}")


if __name__ == "__main__":
    main()
