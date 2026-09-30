r"""
verify_teammate.py -- READ-ONLY live check of the teammate's calibration on this arm +
camera before anything moves. Torque is never enabled; move the arm by hand.

    python -m robot.calib.verify_teammate --cam 2

Each Enter: rest the pointer tip (red tape, at the grasp centre) on the table somewhere
inside the calibrated area, let go. The script reads the joints (robot_control degrees ==
the teammate's frame) and the marker pixel, then asks the calibration which pose it
predicts for that pixel and compares the two finger-meeting points by FK (robot/team_fk):

    xy error  < ~15 mm   calibration + frame agree -> safe to try grasp_and_execute --backend real
    xy error  >> 30 mm   stop: camera moved, marker/offset differs, or servos re-homed again

(The pointer tip sits ~1 cm below the finger meeting point, so compare x/y, not z.)
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import scservo_sdk as scs                                  # noqa: E402
from robot.robot_control import Config, SO101              # noqa: E402
from robot.calib import marker                              # noqa: E402
from robot.calib.pixel_to_arm import HandEye                # noqa: E402
from robot.team_fk import tool_xyz                          # noqa: E402

JOINTS = ["pan", "lift", "elbow", "wflex", "roll"]
DBG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "output", "verify_teammate")


def open_readonly(cfg):
    """Open the arm bus WITHOUT SO101.connect() (that parks goals and enables torque)."""
    arm = SO101(cfg)
    b = arm.bus
    b._ph = scs.PortHandler(b.port); b._pk = scs.PacketHandler(b.protocol_end); b._sync_end()
    if not (b._ph.openPort() and b._ph.setBaudRate(b.baud)):
        raise SystemExit(f"cannot open {b.port}")
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--handeye", default=os.path.join(HERE, "handeye_teammate.json"))
    a = ap.parse_args()

    he = HandEye(a.handeye)
    arm = open_readonly(Config())
    cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(3, 1280); cap.set(4, 720)
    for _ in range(15):
        cap.read()
    os.makedirs(DBG, exist_ok=True)
    print(f"calibration: {a.handeye}\n  valid pixels u {he.uv_range['u']}  v {he.uv_range['v']}")
    print("torque stays OFF. Put the pointer tip on the table, let go, Enter.  q = quit\n")
    errs, k = [], 0
    try:
        while input(f"[{k + 1}] Enter / q > ").strip().lower() != "q":
            f = None
            for _ in range(6):
                ok, fr = cap.read()
                f = fr if ok else f
            q = arm.read_joints_deg()
            if f is None or any(v is None for v in q[:4]):
                print("   read failed (camera or servo) -- again"); continue
            k += 1
            p = marker.detect(f)
            cv2.imwrite(os.path.join(DBG, f"pt{k:02d}.png"), marker.draw(f)[0])
            if p is None:
                print(f"   marker not found -> {DBG}\\pt{k:02d}.png"); continue
            u, v = p
            qp = he.joints_for_pixel(u, v)
            qm = [x if x is not None else qp[i] for i, x in enumerate(q)]       # roll: servo off
            xm, xp = tool_xyz(qm), tool_xyz(qp)
            e = float(np.linalg.norm(xm[:2] - xp[:2])) * 1000
            errs.append(e)
            flag = "" if he.in_range(u, v) else "   (outside the calibrated area)"
            print(f"   px ({u:.0f},{v:.0f})  xy error {e:5.1f} mm   measured z {xm[2]*1000:+.0f} mm{flag}")
            print("   joints measured  " + "  ".join(f"{n} {x:+6.1f}" for n, x in zip(JOINTS[:4], qm)))
            print("   joints predicted " + "  ".join(f"{n} {x:+6.1f}" for n, x in zip(JOINTS[:4], qp)))
    finally:
        cap.release()
        arm.bus._ph.closePort()
    if errs:
        print(f"\n{len(errs)} points: xy error mean {np.mean(errs):.1f} mm, max {np.max(errs):.1f} mm")


if __name__ == "__main__":
    main()
