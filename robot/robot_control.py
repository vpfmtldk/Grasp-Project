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
    arm_port: str = "COM8"                       # TODO confirm which of COM8/COM9 is the arm
    arm_baud: int = 1_000_000
    arm_joints: list = field(default_factory=lambda: [
        # TODO: confirm ids (typical SO-101 order is 1..6; no gripper here)
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
    hand_port: str = "COM9"                      # TODO confirm (the other of COM8/COM9)
    hand_baud: int = 1_000_000                   # TODO SC090 default is often 1M; check
    hand_series: str = "scs"
    hand_servo_ids: list = field(default_factory=lambda: [10, 11, 12, 13, 14, 15, 16, 17])  # TODO 8 = 4 fingers x 2
    # preset name -> {servo_id: angle_deg}. Tune these by hand.
    hand_presets: dict = field(default_factory=lambda: {
        "open":  {i: 0.0 for i in [10, 11, 12, 13, 14, 15, 16, 17]},             # TODO
        "pinch": {10: 35, 11: 40, 12: 35, 13: 40, 14: 35, 15: 40, 16: 0, 17: 0}, # TODO
        "power": {i: 60.0 for i in [10, 11, 12, 13, 14, 15, 16, 17]},            # TODO
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
    def read_steps(self) -> dict[int, int]:
        if self.dry:
            return {i: 2048 for i in self.ids}
        gsr = scs.GroupSyncRead(self._ph, self._pk, ADDR_PRESENT_POSITION, LEN_POSITION)
        for i in self.ids:
            gsr.addParam(i)
        gsr.txRxPacket()
        out = {}
        for i in self.ids:
            lo = gsr.getData(i, ADDR_PRESENT_POSITION, LEN_POSITION)
            out[i] = lo
        return out

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

    def read_joints_deg(self) -> list[float]:
        raw = self.bus.read_steps()
        return [self._steps_to_deg(j, raw[j.servo_id]) for j in self.joints]

    def move_joints_deg(self, target_deg: list[float], secs: float = 2.0):
        """Software-interpolated coordinated move from the current pose."""
        assert len(target_deg) == len(self.joints)
        start = self.read_joints_deg()
        n = max(1, int(secs * self.cfg.move_hz))
        for k in range(1, n + 1):
            a = k / n
            mid = [(1 - a) * s + a * t for s, t in zip(start, target_deg)]
            self.bus.write_steps({j.servo_id: self._deg_to_steps(j, d)
                                  for j, d in zip(self.joints, mid)})
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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SO-101 + AmazingHand serial driver utils")
    p.add_argument("--list-ports", action="store_true", help="list COM ports")
    p.add_argument("--scan", metavar="PORT", help="ping servo ids on PORT")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--series", choices=["sts", "scs"], default="sts")
    p.add_argument("--proto", type=int, choices=[0, 1], default=None,
                   help="override protocol_end (0=STS byte order, 1=SC090)")
    p.add_argument("--calibrate", action="store_true", help="record arm home_steps")
    a = p.parse_args()
    cfg = Config()
    if a.list_ports:
        list_ports()
    elif a.scan:
        scan_ids(a.scan, a.baud, a.series, proto=a.proto)
    elif a.calibrate:
        calibrate(cfg)
    else:
        print("use --list-ports, --scan PORT [--baud N --series scs], --calibrate, "
              "or run demo_move.py")
