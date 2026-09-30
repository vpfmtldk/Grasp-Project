"""
hand_splay_test.py -- show the AmazingHand's splay presets on the real hand (hand only,
the arm is not touched). Watch whether index and ring fan OUTWARD (away from the middle
finger). If they fold inward instead, run with --flip to try the other sign.

    python -m robot.hand_splay_test [--splay 15] [--flip]
"""
import argparse
import time

from robot.robot_control import AmazingHand, Config


def splayed(base, s):
    """base preset + s deg of splay: index (1,2) +s, ring (5,6) -s, both servos same way."""
    p = dict(base)
    for i in (1, 2):
        p[i] = p[i] + s
    for i in (5, 6):
        p[i] = p[i] - s
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splay", type=float, default=15.0)
    ap.add_argument("--flip", action="store_true")
    a = ap.parse_args()
    s = -a.splay if a.flip else a.splay
    cfg = Config()
    cfg.hand_presets["open_splay"] = splayed(cfg.hand_presets["open"], s)
    cfg.hand_presets["power_splay"] = splayed(cfg.hand_presets["power"], s)
    hand = AmazingHand(cfg)
    hand.connect()
    try:
        for name, secs, what in (("open", 2.0, "open (reference)"),
                                 ("open_splay", 3.0, f"open + splay {s:+.0f} deg: index/ring should fan OUT"),
                                 ("open", 2.0, "open"),
                                 ("power", 3.0, "power (normal wrap)"),
                                 ("open", 2.0, "open"),
                                 ("power_splay", 3.0, "power while splayed"),
                                 ("open", 1.5, "open")):
            print(f"  {name:12} {what}", flush=True)
            hand.set_preset(name, secs)
    finally:
        hand.disconnect()
    print("done -- did index/ring spread outward (away from the middle finger)?")


if __name__ == "__main__":
    main()
