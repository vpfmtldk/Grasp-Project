"""
demo_move.py -- hardware sanity check for SO-101 + AmazingHand.

No vision. Connect, print joint state, cycle home -> look -> home, and
open/close the hand. Run with --dry-run first (no hardware, prints intended
commands), then for real once CONFIG in robot_control.py is filled in.

    python robot/demo_move.py --dry-run
    python robot/demo_move.py                # real hardware
    python robot/demo_move.py --hand-only    # skip the arm
    python robot/demo_move.py --arm-only
"""
import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("robot", 1)[0])   # repo root on path
from robot.robot_control import Config, SO101, AmazingHand


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--arm-only", action="store_true")
    p.add_argument("--hand-only", action="store_true")
    p.add_argument("--secs", type=float, default=2.5, help="seconds per arm move")
    a = p.parse_args()
    cfg = Config()

    arm = hand = None
    try:
        if not a.hand_only:
            print("== SO-101 ==")
            arm = SO101(cfg, dry_run=a.dry_run)
            arm.connect()
            print("  joints (deg):", [round(v, 1) for v in arm.read_joints_deg()])
            print("  -> home");  arm.goto_home(a.secs)
            print("  joints (deg):", [round(v, 1) for v in arm.read_joints_deg()])
            print("  -> look pose");  arm.goto_look_pose(a.secs)
            print("  joints (deg):", [round(v, 1) for v in arm.read_joints_deg()])
            time.sleep(0.5)
            print("  -> home");  arm.goto_home(a.secs)

        if not a.arm_only:
            print("== AmazingHand ==")
            hand = AmazingHand(cfg, dry_run=a.dry_run)
            hand.connect()
            for name in ("open", "pinch", "power", "open"):
                print(f"  preset: {name}")
                hand.set_preset(name, secs=1.2)
            print("  width->preset test:")
            for w in (0.02, 0.03, 0.06, 0.10):
                hand.close_for_width(w, secs=1.0)
            hand.open()

        print("\nOK -- if the real arm/hand moved sensibly and joint readings look "
              "right, the low-level driver works. Next: calibrate() then IK.")
    finally:
        if arm:
            arm.disconnect()
        if hand:
            hand.disconnect()


if __name__ == "__main__":
    main()
