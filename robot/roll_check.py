"""
roll_check.py -- staged, reversible test that the repaired wrist_roll (id5) no longer
resets the bus. Stops at the first sign of trouble.

    python -m robot.roll_check [--swing 10]

  1. read id5 (and the others) without touching torque
  2. park id5's goal at its present position, torque limit 200, enable
  3. confirm ids 1-5 all still torque-enabled (the old fault reset 2/4/5 right here)
  4. raise id5's torque limit 400 -> 600 -> 1000, checking after each
  5. swing id5 +swing / -swing deg and back, checking position and the bus each time
The other joints are not commanded.
"""
import argparse
import time

import scservo_sdk as scs

from robot.robot_control import Config, SO101

IDS = [1, 2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swing", type=float, default=10.0)
    a = ap.parse_args()
    arm = SO101(Config())
    b = arm.bus
    b._ph = scs.PortHandler(b.port); b._pk = scs.PacketHandler(b.protocol_end); b._sync_end()
    assert b._ph.openPort() and b._ph.setBaudRate(b.baud)

    def r(i, addr, n=2):
        b._sync_end()
        f = b._pk.read1ByteTxRx if n == 1 else b._pk.read2ByteTxRx
        v, res, _ = f(b._ph, i, addr)
        return v if res == scs.COMM_SUCCESS else None

    def w(i, addr, v, n=2):
        b._sync_end()
        f = b._pk.write1ByteTxRx if n == 1 else b._pk.write2ByteTxRx
        return f(b._ph, i, addr, v)[0]

    def status(tag):
        time.sleep(0.4)
        st = {i: (r(i, 40, 1), r(i, 56), r(i, 60), r(i, 65, 1), r(i, 62, 1)) for i in IDS}
        print(f"{tag:22} " + "  ".join(f"id{i}:en={e} pos={p} load={ld} st={s} V={v}"
                                        for i, (e, p, ld, s, v) in st.items()), flush=True)
        return st

    try:
        st0 = status("0 read only")
        if st0[5][1] is None:
            raise SystemExit("id5 does not answer -- check the cable before enabling anything")
        on_before = {i for i in IDS if st0[i][0] == 1}
        p5 = st0[5][1]
        w(5, 48, 200); w(5, 42, p5); w(5, 40, 1, 1)
        st = status("1 id5 on @200")
        dead = [i for i in on_before | {5} if st[i][0] != 1]
        if dead:
            raise SystemExit(f"FAULT STILL THERE: ids {dead} dropped torque when id5 was enabled")
        for lim in (400, 600, 1000):
            w(5, 48, lim)
            st = status(f"2 id5 limit {lim}")
            dead = [i for i in on_before | {5} if st[i][0] != 1]
            if dead:
                raise SystemExit(f"FAULT at limit {lim}: ids {dead} dropped torque")
        steps = round(a.swing / b.deg_per_step)
        for tgt, tag in ((p5 + steps, f"+{a.swing:.0f}"), (p5 - steps, f"-{a.swing:.0f}"), (p5, "back")):
            w(5, 42, tgt)
            time.sleep(1.0)
            st = status(f"3 swing {tag}")
            dead = [i for i in on_before | {5} if st[i][0] != 1]
            if dead:
                raise SystemExit(f"FAULT during swing: ids {dead} dropped torque")
            pos = st[5][1]
            print(f"   id5 goal {tgt} -> present {pos}  (err {abs(pos - tgt) * b.deg_per_step:.1f} deg)", flush=True)
        print("\nwrist_roll OK: enabled, full torque, moved both ways, no resets")
    finally:
        b._ph.closePort()                          # leave torque as it is (id5 now holding)


if __name__ == "__main__":
    main()
