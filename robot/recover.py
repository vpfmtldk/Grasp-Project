"""
recover.py -- bring the arm home safely after a run was killed mid-grasp.

    python -m robot.recover [--lift 0.06] [--torque-off]

1. open the hand (releases whatever it holds)
2. lift the tool straight up by --lift metres (team FK/IK, so it doesn't sweep sideways
   into the object)
3. go to the park pose (Config.home_deg)
Torque is left ON at the park pose unless --torque-off (the folded arm sags a little
when released).
"""
import argparse

from robot.robot_control import AmazingHand, Config, SO101
from robot.team_fk import lift_tool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", type=float, default=0.06)
    ap.add_argument("--torque-off", action="store_true")
    a = ap.parse_args()
    cfg = Config()
    arm, hand = SO101(cfg), AmazingHand(cfg)
    arm.connect()                                  # parks every goal at the present pose first
    hand.connect()
    try:
        print("1/3 open hand"); hand.open(1.2)
        q = arm.read_joints_deg()
        if any(v is None for v in q[:4]):
            raise SystemExit(f"joint read failed: {q}")
        roll_steps = arm.bus.read_one(cfg.arm_joints[4].servo_id)          # read-only; id5 stays unpowered
        roll = arm._steps_to_deg(cfg.arm_joints[4], roll_steps) if roll_steps is not None else 19.6
        q = [*q[:4], roll]
        lim = [(j.min_deg, j.max_deg) for j in arm.joints]
        q_up, got = lift_tool(q, a.lift, lim)
        print(f"2/3 lift {got * 1000:.0f} mm  {[round(x, 1) for x in q[:4]]} -> {[round(x, 1) for x in q_up[:4]]}")
        arm.move_joints_deg(q_up, secs=2.0, max_step_deg=60)
        print("3/3 park"); arm.move_joints_deg(cfg.home_deg, secs=3.0, max_step_deg=180)
    finally:
        hand.disconnect()
        if a.torque_off:
            arm.disconnect()
            print("arm torque OFF")
        else:
            arm.bus._ph.closePort()                 # keep holding the park pose
            print("arm holding the park pose (torque ON)")


if __name__ == "__main__":
    main()
