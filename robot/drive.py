"""
drive.py -- the arm + hand primitives you actually call, and a demo pick sequence.

    from robot.robot_control import Config, SO101, AmazingHand

    arm  = SO101(Config());       arm.connect()
    hand = AmazingHand(Config()); hand.connect()

    arm.goto_look_pose()                     # fixed pose to photograph the table
    # ... capture frame, predict_grasp -> pixel, pixel_to_world -> base xyz+yaw ...
    arm.move_joints_deg([j0, j1, j2, j3, j4])   # from IK on that xyz+yaw
    hand.close_for_width(0.04)               # grip (preset chosen from width)
    arm.move_joints_deg([...lift...])
    hand.open()                             # release
    arm.goto_home()

CLI:
    python robot/drive.py --dry-run            # print the sequence, no hardware
    python robot/drive.py                      # run it for real
    python robot/drive.py --pick J0 J1 J2 J3 J4 --width 0.04   # grasp at a joint pose
"""
import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("robot", 1)[0])
from robot.robot_control import Config, SO101, AmazingHand


def demo_pick(cfg, pick_deg=None, width_m=0.04, dry_run=False):
    arm = SO101(cfg, dry_run=dry_run)
    hand = AmazingHand(cfg, dry_run=dry_run)
    arm.connect()
    hand.connect()
    try:
        print("open hand");            hand.open()
        print("-> look pose");         arm.goto_look_pose()
        print("  joints:", [None if v is None else round(v, 1) for v in arm.read_joints_deg()])

        if pick_deg is None:
            # no target given: just wave between look and home so you can see it move
            print("(no --pick target; look <-> home)")
            arm.goto_home(); arm.goto_look_pose()
        else:
            print(f"-> pre-grasp {pick_deg}");  arm.move_joints_deg(pick_deg, secs=2.5)
            time.sleep(0.3)
            print(f"grip (width {width_m} m)");  hand.close_for_width(width_m)
            time.sleep(0.3)
            lift = list(pick_deg); lift[1] += 15          # shoulder_lift up a bit
            print(f"-> lift {lift}");            arm.move_joints_deg(lift, secs=1.5)
            time.sleep(0.5)
            print("-> back to pre-grasp");       arm.move_joints_deg(pick_deg, secs=1.5)
            print("release");                    hand.open()

        print("-> home");                        arm.goto_home()
    finally:
        hand.disconnect()
        arm.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pick", type=float, nargs=5, metavar=("J0", "J1", "J2", "J3", "J4"),
                   help="pre-grasp joint pose (deg) -- normally from IK on a predicted grasp")
    p.add_argument("--width", type=float, default=0.04, help="target grip opening (m)")
    a = p.parse_args()
    demo_pick(Config(), pick_deg=a.pick, width_m=a.width, dry_run=a.dry_run)
