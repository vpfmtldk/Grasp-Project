"""
diag_read.py -- temporary diagnostic: does merely touching the hand's serial
port (even if disconnected afterward) break the arm's reads for the rest of
the process? Or does it only break while both are open at once?

    python -m robot.diag_read
"""
import time

from robot.robot_control import Config, SO101, AmazingHand

cfg = Config()
arm = SO101(cfg)
print("connecting arm...")
arm.connect()
print("connected. bus ids:", arm.bus.ids)

def read_all(label):
    print(f"\n--- {label} ---")
    for i in arm.bus.ids:
        pos = arm.bus.read_one(i, retries=1, debug=True)
        print(f"id {i}: {pos}")

read_all("BEFORE touching the hand at all")

print("\nconnecting hand, presetting, DISCONNECTING hand...")
try:
    hand = AmazingHand(cfg)
    hand.connect()
    hand.set_preset("open", 1.0)
    hand.disconnect()
    print("hand connected, opened, and disconnected cleanly")
except Exception as e:
    print(f"hand step failed: {e!r}")

read_all("AFTER hand connected+disconnected (arm never touched in between)")

arm.disconnect()
print("\ndone")
