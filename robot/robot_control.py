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
ADDR_MAX_TORQUE = 16          # EEPROM ceiling (STS3215 ~600, SC090 ~300)
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_ACC = 41
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_TORQUE_LIMIT = 48        # RAM. Ships as 0 on these units -> torque enable
                              # alone does NOTHING: goals are accepted, "moving"
                              # blips, but no force is applied. Must be set.
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
        # ids 1-5 confirmed on COM9 (no gripper; AmazingHand is separate).
        #
        # Angle frame = the teammate's (A팀) LeRobot DEGREES frame, 2026-09-30.
        # Their LeRobot calibration rewrote the servos' EEPROM Homing_Offset and
        # opened the range to 0..4095, so their deg = (tick - 2047.5) * 360/4095;
        # home_steps=2048 reproduces it within 0.1 deg. Their calibration (pixel ->
        # joint, soft limits, park pose, FK frame) is physically verified; our
        # 2026-09-17 home_steps (1975/846/3074/885/2263) predate that re-homing and
        # no longer matched the arm (lift / wrist_flex off by ~100 deg).
        # Limits = their measured soft limits (arm_limits.json, 5 deg margin).
        JointCfg("shoulder_pan",  1, home_steps=2048, min_deg=-104.0, max_deg=116.5),
        JointCfg("shoulder_lift", 2, home_steps=2048, min_deg=-100.4, max_deg=100.1),
        JointCfg("elbow_flex",    3, home_steps=2048, min_deg=-92.8,  max_deg=90.3),
        JointCfg("wrist_flex",    4, home_steps=2048, min_deg=-98.6,  max_deg=97.0),
        JointCfg("wrist_roll",    5, home_steps=2048, min_deg=-26.9,  max_deg=16.7),
    ])
    # wrist_roll (id5): when my scripts enabled its torque, ids 2/4/5 stopped answering
    # and lost torque (2026-09-22/30, 3 times, also at limit 200, unloaded, new cable) --
    # the CAUSE IS NOT ESTABLISHED (servo / power branch / settings / my test procedure;
    # the user reports id5 works with other tools). The teammate also leaves it frozen
    # (~19 deg). Kept off by default because a repeat drops the shoulder (id2); remove 5
    # from this list once robot/roll_check.py passes.
    arm_disabled_ids: list = field(default_factory=lambda: [5])
    # stow pose = the teammate's PARK_POSE (arm_ui_poses.py): folded, 12 deg back
    # from the soft limits so no joint pushes a hard stop. Also used as the photo
    # pose (folded arm is out of the overhead camera's view) until a separate
    # look pose is measured.
    home_deg: list = field(default_factory=lambda: [0.0, -88.4, 78.1, -86.6, 0.0])
    look_deg: list = field(default_factory=lambda: [0.0, -88.4, 78.1, -86.6, 0.0])

    arm_series: str = "sts"                      # SO-101 = STS3215

    # --- AmazingHand (Feetech SC090 / SCS-series, 1024 steps/rev) -----------
    hand_port: str = "COM8"                      # confirmed: hand bus
    hand_baud: int = 1_000_000                   # confirmed (wiggle worked at 1M)
    hand_series: str = "scs"
    # confirmed on COM8: index=1,2  middle=3,4  ring=5,6  thumb=7,8
    hand_servo_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    # AmazingHand fingers are DIFFERENTIAL: the two servos of a finger moving
    # OPPOSITE directions -> flex (grip); SAME direction -> splay sideways.
    # So a curl is {a: +X, b: -X}. Signs below are a guess -- if a finger splays
    # instead of curling, swap the two signs for that finger. Tune with --hand-jog.
    # flex of one finger (servos a,b) confirmed as {a: +90, b: -60}. Same pattern
    # for all fingers; if the THUMB opens instead of closing, swap 7 <-> 8 signs.
    hand_presets: dict = field(default_factory=lambda: {
        "open":  {i: 0.0 for i in [1, 2, 3, 4, 5, 6, 7, 8]},
        # light pinch: index (1,2) + thumb (7,8) curl, middle/ring barely
        "pinch": {1: 55, 2: -37, 3: 12, 4: -8, 5: 12, 6: -8, 7: 55, 8: -37},
        # power: all four fingers fully curl
        "power": {1: 85, 2: -57, 3: 85, 4: -57, 5: 85, 6: -57, 7: 85, 8: -57},
        # splayed: index and ring fan out sideways (both servos of a finger turned the
        # SAME way), middle and thumb unchanged -- a wider envelope while wrist_roll is
        # disabled and the hand can't turn to the object. SPLAY_DEG sign/size set by
        # robot/hand_splay_test.py on the real hand.
        "open_splay":  {1: 15, 2: 15, 3: 0, 4: 0, 5: -15, 6: -15, 7: 0, 8: 0},
        "power_splay": {1: 100, 2: -42, 3: 85, 4: -57, 5: 70, 6: -72, 7: 85, 8: -57},
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

    def _sync_end(self):
        """scservo_sdk's byte-order flag (SCS_END) is a MODULE-LEVEL global, not
        per-PacketHandler -- creating ANY PacketHandler (e.g. the AmazingHand's,
        protocol_end=1) silently flips it for every other bus in the process
        too (confirmed: arm reads came back COMM_SUCCESS with consistent
        garbage the moment the hand bus was touched, even after disconnecting
        it). Reassert this bus's own protocol_end before every real operation."""
        if _HAVE_SCS and not self.dry:
            scs.SCS_SETEND(self.protocol_end)

    def connect(self):
        if self.dry:
            print(f"[dry] open {self.port} @ {self.baud}, ids={self.ids}, "
                  f"{self.steps_per_rev} steps/rev")
            return
        if not _HAVE_SCS:
            raise RuntimeError("scservo_sdk not installed (pip install feetech-servo-sdk)")
        self._ph = scs.PortHandler(self.port)
        self._pk = scs.PacketHandler(self.protocol_end)
        self._sync_end()
        if not self._ph.openPort():
            raise RuntimeError(f"cannot open {self.port}")
        if not self._ph.setBaudRate(self.baud):
            raise RuntimeError(f"cannot set baud {self.baud}")
        for i in self.ids:
            # Park the goal at the present position BEFORE enabling torque. Goal_Position
            # keeps its last commanded value while torque is off, so after the arm was
            # moved by hand, enabling torque snaps it straight back to that stale goal.
            # (This park read was once removed as a suspected cause of garbage reads --
            # the real cause was scservo_sdk's global SCS_END, fixed in _sync_end.)
            pos = self.read_one(i, retries=4)
            if pos is not None:
                self._pk.write2ByteTxRx(self._ph, i, ADDR_GOAL_POSITION, pos)
                time.sleep(0.01)
            self._pk.write1ByteTxRx(self._ph, i, ADDR_TORQUE_ENABLE, 1)
            time.sleep(0.01)
            # Torque_Limit ships as 0 on these servos -> nothing moves without this.
            # Fixed default (STS3215 ceiling ~600, SC090 ~300); firmware clamps to
            # the servo's own ceiling, so this is safe for the lower-torque hand too.
            self._pk.write2ByteTxRx(self._ph, i, ADDR_TORQUE_LIMIT, 500)
            time.sleep(0.01)

    def disconnect(self):
        if self.dry or self._ph is None:
            return
        self._sync_end()
        for i in self.ids:
            self._pk.write1ByteTxRx(self._ph, i, ADDR_TORQUE_ENABLE, 0)
        self._ph.closePort()

    # ---- raw steps <-> degrees (no per-joint sign/home here; caller applies) ----
    def read_one(self, i: int, retries: int = 6, debug=False):
        """Robust single-servo Present_Position read; returns int or None.

        This scservo_sdk build's read helpers can raise / return partial data on a
        noisy bus, so retry and range-check. debug=True prints what actually
        went wrong instead of silently swallowing it (temporary diagnostic).
        """
        if self.dry:
            return self.steps_per_rev // 2
        self._sync_end()
        for _ in range(retries):
            try:
                pos, res, err = self._pk.read2ByteTxRx(self._ph, i, ADDR_PRESENT_POSITION)
            except Exception as e:
                if debug: print(f"    [id {i}] exception: {e!r}")
                time.sleep(0.01); continue
            if res == scs.COMM_SUCCESS and 0 <= pos < self.steps_per_rev:
                return pos
            if debug:
                print(f"    [id {i}] res={res} err={err} pos={pos}")
            time.sleep(0.01)
        return None

    def read_steps(self) -> dict[int, int]:
        if self.dry:
            return {i: self.steps_per_rev // 2 for i in self.ids}
        out = {}
        for i in self.ids:
            out[i] = self.read_one(i)
            time.sleep(0.005)          # settle between servos -- back-to-back reads
        return out                     # with no gap were desyncing this bus

    def write_steps(self, targets: dict[int, int]):
        if self.dry:
            return          # quiet -- milestone prints in the callers show the flow
        self._sync_end()
        gsw = scs.GroupSyncWrite(self._ph, self._pk, ADDR_GOAL_POSITION, LEN_POSITION)
        for i, s in targets.items():
            s = int(max(0, min(self.steps_per_rev - 1, s)))
            gsw.addParam(i, [scs.SCS_LOBYTE(s), scs.SCS_HIBYTE(s)])
        gsw.txPacket()

    def set_speed_acc(self, speed: int = 800, acc: int = 30):
        if self.dry or self._pk is None:
            return
        self._sync_end()
        for i in self.ids:
            self._pk.write2ByteTxRx(self._ph, i, ADDR_GOAL_SPEED, speed)
            self._pk.write1ByteTxRx(self._ph, i, ADDR_GOAL_ACC, acc)


# ================================ SO-101 ===================================
class SO101:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.joints = cfg.arm_joints
        self.disabled = set(cfg.arm_disabled_ids)
        self.bus = FeetechBus(cfg.arm_port, cfg.arm_baud,
                              [j.servo_id for j in self.joints if j.servo_id not in self.disabled],
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
        """Current joint angles (deg); an entry is None if that servo didn't answer
        (or is in cfg.arm_disabled_ids -- never polled)."""
        raw = self.bus.read_steps()
        return [None if raw.get(j.servo_id) is None else self._steps_to_deg(j, raw[j.servo_id])
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
            if j.servo_id in self.disabled:
                continue                        # never commanded, by config
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

    def goto_home(self, secs=3.0):
        # vetted pose -> allow a large coordinated move (guard still skips unread joints)
        self.move_joints_deg(self.cfg.home_deg, secs, max_step_deg=140)

    def goto_look_pose(self, secs=3.0):
        self.move_joints_deg(self.cfg.look_deg, secs, max_step_deg=140)

    # --- end-effector pose -> joint angles, solved against the MuJoCo model ---
    def ee_pose_to_joints(self, T_base_ee, lock=None):
        """
        T_base_ee: 4x4 target pose (or xyz) of the tool site, SO-101 base frame.
        lock: {joint index 0-4: angle (rad)} held fixed -- e.g. the wrist:
              {3: wrist_pitch_rad, 4: wrist_roll_rad}.
        Returns joint angles (deg). No URDF/ikpy -- solved with robot.ik.ArmIK
        (DLS Jacobian IK against robot/sim/so101_amazinghand.xml, mj_forward
        only). Seeded from the arm's current read pose; unread joints (flaky
        bus) fall back to cfg.look_deg for the seed only.
        """
        from robot.ik import ArmIK
        if not hasattr(self, "_ik"):
            self._ik = ArmIK()
        cur = self.read_joints_deg()
        q0 = [c if c is not None else d for c, d in zip(cur, self.cfg.look_deg)]
        if any(c is None for c in cur):
            print("  ee_pose_to_joints: some joints didn't read -- seeding those from look_deg")
        return self._ik.solve(T_base_ee, q0, lock=lock)


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


def jog(cfg):
    """Torque ON. Nudge one joint at a time by typing e.g.  '2 -5'  (joint 2, -5 deg).
    Enter alone reprints the pose; 'q' quits and prints the pose to paste into
    home_deg / look_deg. Small moves only, so it stays inside the safety guard."""
    arm = SO101(cfg)
    arm.connect()
    names = [j.name for j in arm.joints]
    try:
        # hard-retry an initial read; joints that never answer are tracked from 0
        pose = []
        for j in arm.joints:
            s = None
            for _ in range(25):
                s = arm.bus.read_one(j.servo_id, retries=1)
                if s is not None:
                    break
            pose.append(0.0 if s is None else arm._steps_to_deg(j, s))
            if s is None:
                print(f"  (!) {j.name} id {j.servo_id} not reading -- tracked from 0, may drift")

        print("\n  TYPE e.g.  '2 -10'  then ENTER  (joint 2 by -10 deg). "
              "Enter alone = re-read. 'q' = quit.\n")
        while True:
            print("  pose:", "  ".join(f"{n}={p:6.1f}" for n, p in zip(names, pose)))
            s = input("  > ").strip()
            if s.lower() == "q":
                break
            if not s:
                r = arm.read_joints_deg()
                pose = [pose[i] if r[i] is None else r[i] for i in range(len(pose))]
                continue
            parts = s.split()
            if len(parts) != 2:
                print("  format: <joint 1-5> <degrees>,  e.g.  4 8"); continue
            try:
                k = int(parts[0]) - 1
                amt = float(parts[1])
            except ValueError:
                print("  numbers only, e.g.  3 -12"); continue
            if not 0 <= k < len(pose):
                print("  joint must be 1-5"); continue
            tgt = list(pose)
            tgt[k] = pose[k] + amt
            arm.move_joints_deg(tgt, secs=max(0.4, abs(amt) / 20), max_step_deg=45)
            pose[k] = tgt[k]                         # track commanded value
            r = arm.read_joints_deg()
            for i in range(len(pose)):
                if r[i] is not None:
                    pose[i] = r[i]
    finally:
        arm.disconnect()
    print("\nfinal pose (deg):", [round(p, 1) for p in pose])
    print("paste into Config:  [" + ", ".join(f"{p:.0f}" for p in pose) + "]")


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


def hand_jog(cfg):
    """Keyboard-nudge AmazingHand servos to find each finger's flex combination.
    Type e.g.  '1 30'  (servo 1 to +30 deg from centre) then ENTER. Two servos of
    one finger with OPPOSITE signs should curl it. 'r' resets all to 0, 'q' quits
    and prints the current angle map to paste into hand_presets."""
    bus = FeetechBus(cfg.hand_port, cfg.hand_baud, cfg.hand_servo_ids, series=cfg.hand_series)
    bus.connect()
    ang = {i: 0.0 for i in cfg.hand_servo_ids}
    mid = bus.steps_per_rev // 2
    def push():
        bus.write_steps({i: int(round(mid + a / bus.deg_per_step)) for i, a in ang.items()})
    try:
        push()
        print("\n  fingers: index=1,2  middle=3,4  ring=5,6  thumb=7,8")
        print("  type '<servo> <deg>' then ENTER (e.g. '1 30' then '2 -30' to curl index)")
        print("  'r' = all to 0,  'q' = quit + print map\n")
        while True:
            print("  angles:", " ".join(f"{i}:{ang[i]:+.0f}" for i in cfg.hand_servo_ids))
            s = input("  > ").strip().lower()
            if s == "q":
                break
            if s == "r":
                ang = {i: 0.0 for i in cfg.hand_servo_ids}; push(); continue
            parts = s.split()
            if len(parts) != 2:
                print("  format: <servo 1-8> <deg>"); continue
            try:
                sid, a = int(parts[0]), float(parts[1])
            except ValueError:
                print("  numbers only"); continue
            if sid not in ang:
                print("  servo must be one of", cfg.hand_servo_ids); continue
            ang[sid] = max(-90, min(90, a))
            push(); time.sleep(0.2)
    finally:
        bus.disconnect()
    print("\n  map:", "{" + ", ".join(f"{i}: {ang[i]:.0f}" for i in cfg.hand_servo_ids) + "}")


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
    p.add_argument("--jog", action="store_true", help="torque ON, nudge joints by keyboard to build a pose")
    p.add_argument("--hand-jog", action="store_true", help="nudge AmazingHand servos to find flex combos")
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
    elif a.jog:
        jog(cfg)
    elif a.hand_jog:
        hand_jog(cfg)
    elif a.read_pose:
        read_pose(cfg)
    elif a.check_arm:
        check_arm(cfg)
    elif a.calibrate:
        calibrate(cfg)
    else:
        print("use --list-ports, --scan PORT, --wiggle PORT [--ids 1-12 --series scs "
              "--baud N], --calibrate, or run demo_move.py")
