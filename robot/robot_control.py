"""
robot_control.py -- minimal driver for SO-101 (5 arm joints) + AmazingHand,
both on Feetech STS/SCS serial-bus servos, from a PC over a USB-serial adapter.

This is a SKELETON. Everything setup-specific is in CONFIG at the top -- fill it
in, then run demo_move.py --dry-run first, then without --dry-run.

Deps:
    pip install feetech-servo-sdk        # provides the `scservo_sdk` module
    pip install ikpy                     # optional, only for ee_pose -> joints

Unknowns you must resolve (search "TODO"):
  * COM port(s) of the USB-serial adapter(s)   -> Device Manager
  * servo IDs for each joint / finger          -> Feetech tool, or scan
  * whether AmazingHand is on the SAME bus as the arm or a separate adapter
  * per-joint sign and the "home" step count    -> run calibrate()
  * AmazingHand finger preset angles            -> tune by hand
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# scservo_sdk is only needed for real hardware; --dry-run works without it.
try:
    import scservo_sdk as scs
    _HAVE_SCS = True
except ImportError:
    scs = None
    _HAVE_SCS = False


# ==================== Feetech register map (STS + SCS share these) ===========
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_ACC = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
LEN_POSITION = 2

# Per servo family. SO-101 uses STS3215 (4096, protocol_end 0); the AmazingHand
# uses SC090 / SCS-series (1024). If SCS position reads come back as garbage,
# flip its protocol_end to 1.
SERIES = {
    "sts": {"steps_per_rev": 4096, "protocol_end": 0},
    "scs": {"steps_per_rev": 1024, "protocol_end": 1},   # SC090 reads byte-swapped under 0
}


# ================================== CONFIG ==================================
@dataclass
class JointCfg:
    name: str
    servo_id: int
    sign: int = 1              # +1 or -1, set during calibration
    home_steps: int = 2048     # raw step reading when the joint is at 0 deg
    min_deg: float = -120.0
    max_deg: float = 120.0


@dataclass
class Config:
    # --- SO-101 arm ---------------------------------------------------------
    arm_port: str = "COM9"                       # confirmed: arm bus, ids 1-5
    arm_baud: int = 1_000_000
    arm_joints: list = field(default_factory=lambda: [
        # ids 1-5 confirmed on COM9 (no gripper; AmazingHand is separate). Joint
        # names assume the standard SO-101 build order -- verify with --wiggle.
        JointCfg("shoulder_pan",  1, min_deg=-110, max_deg=110),
        JointCfg("shoulder_lift", 2, min_deg=-100, max_deg=100),
        JointCfg("elbow_flex",    3, min_deg=-100, max_deg=100),
        JointCfg("wrist_flex",    4, min_deg=-100, max_deg=100),
        JointCfg("wrist_roll",    5, min_deg=-160, max_deg=160),
    ])
    # a safe stow pose and the fixed pose used to photograph the table
    home_deg: list = field(default_factory=lambda: [0, -90, 90, 0, 0])           # TODO tune
    look_deg: list = field(default_factory=lambda: [0, -60, 70, -55, 0])         # TODO tune

    arm_series: str = "sts"                      # SO-101 = STS3215

    # --- AmazingHand (Feetech SC090 / SCS-series, 1024 steps/rev) -----------
    hand_port: str = "COM8"                      # confirmed: hand bus
    hand_baud: int = 1_000_000                   # confirmed (wiggle worked at 1M)
    hand_series: str = "scs"
    # confirmed on COM8: index=1,2  middle=3,4  ring=5,6  thumb=7,8
    hand_servo_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    # preset name -> {servo_id: angle_deg from that servo's centre}. STARTING GUESSES --
    # tune signs/magnitudes by hand (demo_move.py --hand-only, then edit).
    hand_presets: dict = field(default_factory=lambda: {
        "open":  {i: 0.0 for i in [1, 2, 3, 4, 5, 6, 7, 8]},
        # thumb (7,8) opposes index (1,2); middle/ring lightly curled
        "pinch": {1: 40, 2: 40, 3: 15, 4: 15, 5: 15, 6: 15, 7: 45, 8: 45},
        # all four fingers curl in
        "power": {i: 55.0 for i in [1, 2, 3, 4, 5, 6, 7, 8]},
    })
    # which preset to use for a given target opening width (metres)
    width_to_preset: list = field(default_factory=lambda: [
        (0.00, 0.045, "pinch"),
        (0.045, 0.12, "power"),
    ])

    move_hz: float = 50.0        # software interpolation rate for timed moves


# =============================== Feetech bus ================================
class FeetechBus:
    """Sync read/write of Present/Goal position (in degrees) for a set of ids."""

    def __init__(self, port: str, baud: int, ids: list[int], series: str = "sts", dry_run: bool = False):
        self.port, self.baud, self.ids, self.dry = port, baud, list(ids), dry_run
        self.steps_per_rev = SERIES[series]["steps_per_rev"]
        self.protocol_end = SERIES[series]["protocol_end"]
        self.deg_per_step = 360.0 / self.steps_per_rev
        self._ph = self._pk = None

    def connect(self):
        if self.dry:
            print(f"[dry] open {self.port} @ {self.baud}, ids={self.ids}, "
                  f"{self.steps_per_rev} steps/rev")
            return
        if not _HAVE_SCS:
            raise RuntimeError("scservo_sdk not installed (pip install feetech-servo-sdk)")
        self._ph = scs.PortHandler(self.port)
        self._pk = scs.PacketHandler(self.protocol_end)
        if not self._ph.openPort():
            raise RuntimeError(f"cannot open {self.port}")
        if not self._ph.setBaudRate(self.baud):
            raise RuntimeError(f"cannot set baud {self.baud}")
        for i in self.ids:
            self._pk.write1ByteTxRx(self._ph, i, ADDR_TORQUE_ENABLE, 1)

    def disconnect(self):
        if self.dry or self._ph is None:
            return
        for i in self.ids:
            self._pk.write1ByteTxRx(self._ph, i, ADDR_TORQUE_ENABLE, 0)
        self._ph.closePort()

    # ---- raw steps <-> degrees (no per-joint sign/home here; caller applies) ----
    def read_one(self, i: int, retries: int = 6):
        """Robust single-servo Present_Position read; returns int or None.

        This scservo_sdk build's read helpers can raise / return partial data on a
        noisy bus, so retry and range-check.
        """
        if self.dry:
            return self.steps_per_rev // 2
        for _ in range(retries):
            try:
                pos, res, _ = self._pk.read2ByteTxRx(self._ph, i, ADDR_PRESENT_POSITION)
            except Exception:
                time.sleep(0.01); continue
            if res == scs.COMM_SUCCESS and 0 <= pos < self.steps_per_rev:
                return pos
            time.sleep(0.01)
        return None

    def read_steps(self) -> dict[int, int]:
        if self.dry:
            return {i: self.steps_per_rev // 2 for i in self.ids}
        return {i: self.read_one(i) for i in self.ids}

    def write_steps(self, targets: dict[int, int]):
        if self.dry:
            print("[dry] goal steps:", targets)
            return
        gsw = scs.GroupSyncWrite(self._ph, self._pk, ADDR_GOAL_POSITION, LEN_POSITION)
        for i, s in targets.items():
            s = int(max(0, min(self.steps_per_rev - 1, s)))
            gsw.addParam(i, [scs.SCS_LOBYTE(s), scs.SCS_HIBYTE(s)])
        gsw.txPacket()

    def set_speed_acc(self, speed: int = 800, acc: int = 30):
        if self.dry or self._pk is None:
            return
        for i in self.ids:
            self._pk.write2ByteTxRx(self._ph, i, ADDR_GOAL_SPEED, speed)
            self._pk.write1ByteTxRx(self._ph, i, ADDR_GOAL_ACC, acc)


# ================================ SO-101 ===================================
class SO101:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.joints = cfg.arm_joints
        self.bus = FeetechBus(cfg.arm_port, cfg.arm_baud, [j.servo_id for j in self.joints],
                              series=cfg.arm_series, dry_run=dry_run)

    def connect(self):
        self.bus.connect()
        self.bus.set_speed_acc()

    def disconnect(self):
        self.bus.disconnect()

    def _steps_to_deg(self, j: JointCfg, steps: int) -> float:
        return j.sign * (steps - j.home_steps) * self.bus.deg_per_step

    def _deg_to_steps(self, j: JointCfg, deg: float) -> int:
        deg = max(j.min_deg, min(j.max_deg, deg))
        return int(round(j.home_steps + j.sign * deg / self.bus.deg_per_step))

    def read_joints_deg(self) -> list:
        """Current joint angles (deg); an entry is None if that servo didn't answer."""
        raw = self.bus.read_steps()
        return [None if raw[j.servo_id] is None else self._steps_to_deg(j, raw[j.servo_id])
                for j in self.joints]

    def move_joints_deg(self, target_deg: list, secs: float = 2.0, max_step_deg: float = 60.0):
        """
        Software-interpolated coordinated move from the current pose.
        Joints whose start angle can't be read, or whose move would exceed
        max_step_deg, are SKIPPED (not commanded) and reported -- a safety guard
        against a bad read slamming the arm.
        """
        assert len(target_deg) == len(self.joints)
        start = self.read_joints_deg()
        active, skipped = [], []
        for idx, (j, s, t) in enumerate(zip(self.joints, start, target_deg)):
            if s is None:
                skipped.append(f"{j.name}(no read)")
            elif abs(t - s) > max_step_deg:
                skipped.append(f"{j.name}(d={abs(t-s):.0f}>{max_step_deg:.0f})")
            else:
                active.append(idx)
        if skipped:
            print("  SKIPPING:", ", ".join(skipped), "-- move them by hand / raise max_step_deg")
        if not active:
            return
        n = max(1, int(secs * self.cfg.move_hz))
        for k in range(1, n + 1):
            a = k / n
            targets = {}
            for idx in active:
                j = self.joints[idx]
                d = (1 - a) * start[idx] + a * target_deg[idx]
                targets[j.servo_id] = self._deg_to_steps(j, d)
            self.bus.write_steps(targets)
            time.sleep(1.0 / self.cfg.move_hz)

    def goto_home(self, secs=2.5):
        self.move_joints_deg(self.cfg.home_deg, secs)

    def goto_look_pose(self, secs=2.5):
        self.move_joints_deg(self.cfg.look_deg, secs)

    # --- optional: end-effector pose -> joint angles (needs a URDF + ikpy) ---
    def ee_pose_to_joints(self, T_base_ee):
        """
        T_base_ee: 4x4 target pose of the tool/hand frame in the arm base frame.
        Returns joint angles (deg). Requires the SO-101 URDF and ikpy.
        """
        try:
            import numpy as np
            from ikpy.chain import Chain
        except ImportError as e:
            raise RuntimeError("need `pip install ikpy` and the SO-101 URDF") from e
        if not hasattr(self, "_chain"):
            # TODO: path to the SO-101 URDF (e.g. from the SO-ARM100 repo)
            self._chain = Chain.from_urdf_file("so101.urdf",
                                               active_links_mask=[False] + [True] * 5 + [False])
        target_pos = np.asarray(T_base_ee)[:3, 3]
        target_orient = np.asarray(T_base_ee)[:3, :3]
        q = self._chain.inverse_kinematics(target_pos, target_orient, orientation_mode="all")
        return [float(np.degrees(v)) for v in q[1:6]]


# ============================== AmazingHand ================================
class AmazingHand:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.ids = cfg.hand_servo_ids
        self.bus = FeetechBus(cfg.hand_port, cfg.hand_baud, self.ids,
                              series=cfg.hand_series, dry_run=dry_run)

    def connect(self):
        self.bus.connect()
        self.bus.set_speed_acc(speed=600, acc=20)

    def disconnect(self):
        self.bus.disconnect()

    def _angles_to_steps(self, angles: dict[int, float]) -> dict[int, int]:
        # assumes each finger servo's 0 deg == mid (2048). Adjust if your hand differs.
        mid = self.bus.steps_per_rev // 2
        return {i: int(round(mid + a / self.bus.deg_per_step)) for i, a in angles.items()}

    def set_preset(self, name: str, secs: float = 1.0):
        if name not in self.cfg.hand_presets:
            raise KeyError(f"unknown preset {name}; have {list(self.cfg.hand_presets)}")
        angles = self.cfg.hand_presets[name]
        self.bus.write_steps(self._angles_to_steps(angles))
        time.sleep(secs)

    def open(self, secs=1.0):
        self.set_preset("open", secs)

    def close_for_width(self, width_m: float, secs=1.0):
        preset = self.cfg.width_to_preset[-1][2]
        for lo, hi, name in self.cfg.width_to_preset:
            if lo <= width_m < hi:
                preset = name
                break
        print(f"width {width_m:.3f} m -> preset '{preset}'")
        self.set_preset(preset, secs)


# ============================== calibration ================================
def calibrate(cfg: Config):
    """
    Interactive: torque OFF, you move each joint to its 0-deg reference by hand,
    press Enter, and it prints home_steps / suggested sign to paste into CONFIG.
    """
    bus = FeetechBus(cfg.arm_port, cfg.arm_baud, [j.servo_id for j in cfg.arm_joints], series=cfg.arm_series)
    bus.connect()
    for i in bus.ids:                       # torque off so you can move it
        bus._pk.write1ByteTxRx(bus._ph, i, ADDR_TORQUE_ENABLE, 0)
    for j in cfg.arm_joints:
        input(f"  move '{j.name}' (id {j.servo_id}) to its 0-deg pose, then Enter...")
        s = bus.read_steps()[j.servo_id]
        print(f"    {j.name}: home_steps={s}")
    bus.disconnect()


def list_ports():
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("pip install pyserial"); return
    ports = list(lp.comports())
    if not ports:
        print("no serial ports found"); return
    for p in ports:
        print(f"  {p.device:10s}  {p.description}")


def scan_ids(port, baud, series="sts", lo=1, hi=30, proto=None):
    """Read Present_Position for every id in [lo, hi]; report which servos answer.

    Avoids packet_handler.ping() -- some scservo_sdk builds crash on a no-reply.
    A sane position (0..4095 for STS, 0..1023 for SC090) means the series /
    protocol_end are right; wild values mean flip SERIES[...]['protocol_end'].
    """
    if not _HAVE_SCS:
        print("pip install feetech-servo-sdk"); return
    pe = SERIES[series]["protocol_end"] if proto is None else proto
    ph = scs.PortHandler(port)
    pk = scs.PacketHandler(pe)
    try:
        if not ph.openPort() or not ph.setBaudRate(baud):
            print(f"cannot open {port} @ {baud}"); return
    except Exception as e:
        print(f"{port} not available: {e}"); return
    rev = SERIES[series]["steps_per_rev"]
    print(f"scanning {port} @ {baud} ({series}, {rev} steps/rev, protocol_end={pe}) ids {lo}..{hi}")
    found = []
    for i in range(lo, hi + 1):
        try:
            pos, res, err = pk.read2ByteTxRx(ph, i, ADDR_PRESENT_POSITION)
        except Exception:
            continue
        if res == scs.COMM_SUCCESS:
            flag = "" if 0 <= pos < rev else "  <-- out of range: wrong series/protocol_end?"
            print(f"  id {i:3d}  present_position={pos}{flag}")
            found.append(i)
    ph.closePort()
    print("found:", found or "(none)")


def read_pose(cfg, hz=2.0):
    """Torque OFF; print arm joint angles (deg) continuously so you can pose the
    arm by hand and read off values for home_deg / look_deg. Ctrl+C to stop."""
    bus = FeetechBus(cfg.arm_port, cfg.arm_baud, [j.servo_id for j in cfg.arm_joints],
                     series=cfg.arm_series)
    bus.connect()
    for _ in range(5):                      # retry -- the disable write can be lost on a noisy bus
        for i in bus.ids:
            bus._pk.write1ByteTxRx(bus._ph, i, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.05)
    print("torque OFF (sent x5) -- the arm should now be loose. Move it by hand. Ctrl+C to stop.")
    print("if it is still stiff: cut servo power briefly, or a servo's disable didn't land.")
    try:
        while True:
            raw = bus.read_steps()
            degs = []
            for j in cfg.arm_joints:
                s = raw[j.servo_id]
                degs.append("  n " if s is None else
                            f"{j.sign * (s - j.home_steps) * bus.deg_per_step:6.1f}")
            print("  ".join(f"{j.name}={d}" for j, d in zip(cfg.arm_joints, degs)))
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        bus.disconnect()


def check_arm(cfg, step_deg=10.0):
    """Read the arm, then nudge each joint +step_deg/-step_deg (small, safe) so you
    can confirm the id->joint map and the sign. Edit CONFIG from what you see."""
    arm = SO101(cfg)
    arm.connect()
    try:
        cur = arm.read_joints_deg()
        print("current joints (deg):",
              [None if v is None else round(v, 1) for v in cur])
        for k, j in enumerate(arm.joints):
            if cur[k] is None:
                print(f"  {j.name} (id {j.servo_id}): no read, skipping"); continue
            input(f"  Enter to nudge {j.name} (id {j.servo_id}) +-{step_deg} deg ...")
            base = list(cur)
            for d in (step_deg, -step_deg, 0.0):
                tgt = list(base)
                tgt[k] = cur[k] + d
                arm.move_joints_deg(tgt, secs=1.0)
            note = input(f"    which joint moved, and did + go the expected way? ")
            if note:
                print(f"      {j.name} id {j.servo_id}: {note}")
    finally:
        arm.disconnect()


def wiggle(port, baud, ids, series="sts", proto=None, amp=80, reps=3):
    """
    Write-only: nudge each id back and forth so you can SEE which joint/finger it
    drives. Reads are unreliable on a contended bus; writes usually get through.
    `ids` is a list; each is wiggled in turn, waiting for Enter between them.
    """
    if not _HAVE_SCS:
        print("pip install feetech-servo-sdk"); return
    pe = SERIES[series]["protocol_end"] if proto is None else proto
    rev = SERIES[series]["steps_per_rev"]
    ph = scs.PortHandler(port)
    pk = scs.PacketHandler(pe)
    try:
        if not ph.openPort() or not ph.setBaudRate(baud):
            print(f"cannot open {port} @ {baud}"); return
    except Exception as e:
        print(f"{port} not available: {e}"); return
    print(f"wiggle {port} @ {baud} ({series}, {rev}/rev, proto={pe})  amp={amp} steps")
    mid = rev // 2
    for i in ids:
        input(f"\n  press Enter to wiggle id {i} ...")
        pk.write1ByteTxRx(ph, i, ADDR_TORQUE_ENABLE, 1)
        center = mid
        try:
            cur, res, _ = pk.read2ByteTxRx(ph, i, ADDR_PRESENT_POSITION)
            if res == scs.COMM_SUCCESS and 0 <= cur < rev:
                center = cur
        except Exception:
            pass
        print(f"    center~{center}; moving +-{amp} x{reps}  (Ctrl+C to stop)")
        for _ in range(reps):
            for tgt in (center + amp, center - amp, center):
                tgt = int(max(0, min(rev - 1, tgt)))
                pk.write2ByteTxRx(ph, i, ADDR_GOAL_POSITION, tgt)
                time.sleep(0.35)
        pk.write1ByteTxRx(ph, i, ADDR_TORQUE_ENABLE, 0)
        note = input(f"    which joint/finger moved for id {i}? (type a note, Enter to skip) ")
        if note:
            print(f"      id {i} -> {note}")
    ph.closePort()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SO-101 + AmazingHand serial driver utils")
    p.add_argument("--list-ports", action="store_true", help="list COM ports")
    p.add_argument("--scan", metavar="PORT", help="read Present_Position for ids on PORT")
    p.add_argument("--wiggle", metavar="PORT", help="write-only nudge ids on PORT (see what moves)")
    p.add_argument("--ids", default="1-12", help="ids for --wiggle, e.g. 1-12 or 1,4,7")
    p.add_argument("--amp", type=int, default=80, help="wiggle amplitude in steps")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--series", choices=["sts", "scs"], default="sts")
    p.add_argument("--proto", type=int, choices=[0, 1], default=None,
                   help="override protocol_end (0=STS byte order, 1=SC090)")
    p.add_argument("--calibrate", action="store_true", help="record arm home_steps")
    p.add_argument("--check-arm", action="store_true", help="small +-10 deg nudge per arm joint")
    p.add_argument("--read-pose", action="store_true", help="torque OFF, print joint angles to pose by hand")
    a = p.parse_args()
    cfg = Config()

    def parse_ids(s):
        out = []
        for part in s.split(","):
            if "-" in part:
                lo, hi = part.split("-"); out += list(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
        return out

    if a.list_ports:
        list_ports()
    elif a.scan:
        scan_ids(a.scan, a.baud, a.series, proto=a.proto)
    elif a.wiggle:
        wiggle(a.wiggle, a.baud, parse_ids(a.ids), a.series, proto=a.proto, amp=a.amp)
    elif a.read_pose:
        read_pose(cfg)
    elif a.check_arm:
        check_arm(cfg)
    elif a.calibrate:
        calibrate(cfg)
    else:
        print("use --list-ports, --scan PORT, --wiggle PORT [--ids 1-12 --series scs "
              "--baud N], --calibrate, or run demo_move.py")
