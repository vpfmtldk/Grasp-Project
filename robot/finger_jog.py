"""
finger_jog.py -- keyboard control of the AmazingHand's two finger motions.

Each finger has 2 motors (A, B). Per the Pollen parallel-mechanism diagram:
  * both motors same rotation  -> flexion / extension   -> fingertip UP / DOWN (flex)
  * motors opposite rotation   -> abduction / adduction  -> fingertip LEFT / RIGHT (abduct)

State is one (flex, abd) angle applied to all four fingers (or one finger).
  flex(+)  = curl in / down    | motor A += f , motor B -= f   (grip closes)
  abd(+)   = swing to one side | motor A += a , motor B += a

Commands (lower-case, ENTER):
  step keys, move by the current step (default 8 deg):
     u  = UP   (flex up / extend)       j  = DOWN (flex down / curl -> grip closes)
     h  = LEFT (abduct one way)         k  = RIGHT(abduct other way)
  + / -            bigger / smaller step
  c <deg>          set flex angle absolutely (all fingers)
  s <deg>          set abduction angle absolutely (all fingers)
  f <n> u|j|h|k    move only finger n (1..4) by one step
  0               centre this finger set (flex=0, abd=0)
  r               all servos to 0
  p               print the map
  q               quit + print the map (paste into Config.hand_presets)

  python -m robot.finger_jog
"""
import time

from robot.robot_control import Config, FeetechBus

FINGERS = {1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (7, 8)}   # finger -> (motor A id, motor B id)
LIM = 90.0


def _clamp(v):
    return max(-LIM, min(LIM, v))


def main():
    cfg = Config()
    bus = FeetechBus(cfg.hand_port, cfg.hand_baud, cfg.hand_servo_ids, series=cfg.hand_series)
    bus.connect()
    ang = {i: 0.0 for i in cfg.hand_servo_ids}
    mid = bus.steps_per_rev // 2
    flex = {n: 0.0 for n in FINGERS}
    abd = {n: 0.0 for n in FINGERS}
    step = 8.0

    def apply(ns):
        for n in ns:
            a, b = FINGERS[n]
            ang[a] = _clamp(flex[n] + abd[n])
            ang[b] = _clamp(-flex[n] + abd[n])
        bus.write_steps({i: int(round(mid + v / bus.deg_per_step)) for i, v in ang.items()})

    def bump(ns, dfl=0.0, dab=0.0):
        for n in ns:
            flex[n] = _clamp(flex[n] + dfl)
            abd[n] = _clamp(abd[n] + dab)
        apply(ns)

    KEYS = {"u": (+1, 0), "j": (-1, 0), "h": (0, +1), "k": (0, -1)}

    try:
        apply(FINGERS)
        print(__doc__)
        while True:
            print("  " + "  ".join(f"F{n}(flex{flex[n]:+.0f} abd{abd[n]:+.0f})" for n in FINGERS)
                  + f"   step={step:.0f}")
            s = input("  > ").strip().lower()
            if s == "q":
                break
            if s == "r":
                for d in (ang,): d.update({i: 0.0 for i in d})
                for n in FINGERS: flex[n] = abd[n] = 0.0
                apply(FINGERS); continue
            if s == "p":
                continue
            if s == "0":
                for n in FINGERS: flex[n] = abd[n] = 0.0
                apply(FINGERS); continue
            if s in ("+", "-"):
                step = min(30.0, step + 4) if s == "+" else max(2.0, step - 4)
                continue
            if s in KEYS:
                df, da = KEYS[s]
                bump(FINGERS, df * step, da * step); continue
            t = s.split()
            try:
                if len(t) == 2 and t[0] == "c":
                    for n in FINGERS: flex[n] = _clamp(float(t[1]))
                    apply(FINGERS)
                elif len(t) == 2 and t[0] == "s":
                    for n in FINGERS: abd[n] = _clamp(float(t[1]))
                    apply(FINGERS)
                elif len(t) == 3 and t[0] == "f" and t[2] in KEYS:
                    n = int(t[1]); df, da = KEYS[t[2]]
                    bump([n], df * step, da * step)
                elif len(t) == 2:
                    sid, d = int(t[0]), float(t[1])
                    if sid not in ang:
                        print("  servo must be", cfg.hand_servo_ids); continue
                    ang[sid] = _clamp(d)
                    bus.write_steps({i: int(round(mid + v / bus.deg_per_step)) for i, v in ang.items()})
                else:
                    print("  ? see the header for commands"); continue
            except ValueError:
                print("  numbers only"); continue
            time.sleep(0.12)
    finally:
        bus.disconnect()
    print("\n  map: {" + ", ".join(f"{i}: {ang[i]:.0f}" for i in cfg.hand_servo_ids) + "}")


if __name__ == "__main__":
    main()
