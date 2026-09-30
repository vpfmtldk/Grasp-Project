r"""Automatic hand-eye point collection -- the robot finds the table itself.

Replaces clicking in collect.py. A rigid pointer (chopstick) is taped to the palm at
the grasp centre, with a red tape band on its tip (robot/calib/marker.py finds it).

    python -m robot.calib.auto_collect --teach     # 1. hand-place the tip at the area's corners
    python -m robot.calib.auto_collect --probe     # 2. which way of shoulder_lift is "down"
    python -m robot.calib.auto_collect --run       # 3. sweep a grid, touch the table, record

--teach   pan/lift/elbow go limp, the wrist stays held by its motors. Put the tip on the
          table at the corners of the area to cover (4 corners + centre), let go, press
          Enter each time. Only the joint RANGE is learned from these -- not used as data.
--probe   Arm goes limp; rest the tip on the table NEAR the base, Enter. Nudges shoulder_lift 4 deg each way;
          the side the table blocks is "down".
--run     Rows from near to far, pan spread across each row. Hand pitch (lift+elbow+
          wrist_flex) is held at the taught value throughout. Above each point, step
          shoulder_lift down until the table stalls it, read the joints, detect the
          marker. Writes calib/points.csv (collect.py format -> solve.py).

Why this beats clicking: the wrist is motor-held the whole time (no hand pushing it),
and posture varies smoothly along the grid (no elbow-up/down flip between points) --
the two things that wrecked the hand-collected set (LOO 29.6 deg).
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from robot.robot_control import (Config, SO101, ADDR_TORQUE_ENABLE, ADDR_TORQUE_LIMIT,   # noqa: E402
                                  ADDR_MAX_TORQUE)
from robot.calib import marker                                       # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
PAN, LIFT, ELBOW, WFLEX, WROLL = range(5)
TEACH = os.path.join(HERE, "teach.json")
POINTS = os.path.join(HERE, "points.csv")
DBG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "output", "auto_collect")

START_ABOVE = 12.0    # deg of shoulder_lift above the predicted contact to start descending
RETRACT = 15.0        # deg to back off after each touch
STEP = 0.5            # deg per descent step
STALL_DEG = 2.0       # lag beyond the free-air sag (in the "down" sense) = touching
MAX_PAST = 20.0       # give up if no contact this far past the prediction
ROI_MARGIN = 120      # px around the taught marker positions


# ----------------------------------------------------------------------- helpers
class Cam:
    """Keeps the camera open; each grab() flushes stale buffered frames first."""
    def __init__(self, idx):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else 0)
        self.cap.set(3, 1280); self.cap.set(4, 720)
        for _ in range(15):
            self.cap.read()

    def grab(self):
        f = None
        for _ in range(6):
            ok, fr = self.cap.read()
            if ok:
                f = fr
        return f

    def close(self):
        self.cap.release()


ROLL_FIXED = -1.0     # wrist_roll angle reported while that servo is left off (--with-roll not given)


def make_arm(args):
    """wrist_roll (id5) is left OFF by default: enabling its torque resets id2/4/5
    (electrical fault on that branch, found 2026-09-22). Roll is constant during
    calibration anyway, so it's reported as ROLL_FIXED and never written."""
    cfg = Config()
    roll_id = cfg.arm_joints[WROLL].servo_id
    cfg.arm_disabled_ids = sorted(set(cfg.arm_disabled_ids) | {roll_id}) if not args.with_roll else         [i for i in cfg.arm_disabled_ids if i != roll_id]
    arm = SO101(cfg)
    arm.connect()
    return arm


def live(arm):
    return [k for k, j in enumerate(arm.joints) if j.servo_id not in arm.disabled]


def read_firm(arm, tries=8):
    for _ in range(tries):
        q = arm.read_joints_deg()
        q = [ROLL_FIXED if (v is None and arm.joints[k].servo_id in arm.disabled) else v
             for k, v in enumerate(q)]
        if all(v is not None for v in q):
            return q
        time.sleep(0.05)
    return None


def set_goal(arm, q):
    arm.bus.write_steps({j.servo_id: arm._deg_to_steps(j, d) for j, d in zip(arm.joints, q)
                         if j.servo_id not in arm.disabled})


def torque(arm, idxs, on):
    arm.bus._sync_end()
    for k in idxs:
        if k not in live(arm):
            continue
        arm.bus._pk.write1ByteTxRx(arm.bus._ph, arm.joints[k].servo_id, ADDR_TORQUE_ENABLE, int(on))
        time.sleep(0.01)


def strong(arm, idxs=(LIFT, ELBOW, WFLEX, WROLL)):
    """Torque_Limit = the servo's own Max_Torque (1000 on these). connect() sets 500,
    which can't raise the extended arm: at the far teach point shoulder_lift didn't
    move either way (+2 blocked by weakness, -2 by the table)."""
    b = arm.bus; b._sync_end()
    for k in idxs:
        if k not in live(arm):
            continue
        sid = arm.joints[k].servo_id
        mx, res, _ = b._pk.read2ByteTxRx(b._ph, sid, ADDR_MAX_TORQUE)
        b._pk.write2ByteTxRx(b._ph, sid, ADDR_TORQUE_LIMIT, mx if res == 0 and 0 < mx <= 1000 else 600)
        time.sleep(0.01)


def hold_here(arm):
    """Park every goal at the present pose, then torque on (no snap to a stale goal)."""
    q = read_firm(arm)
    if q is None:
        raise SystemExit("관절 읽기 실패 -- 토크 안 켬")
    set_goal(arm, q); time.sleep(0.05)
    torque(arm, range(5), True)
    return q


def release_port(arm):
    """Close the port WITHOUT touching torque -- disconnect() turns torque off, which
    drops an arm held in the air."""
    if arm.bus._ph is not None:
        arm.bus._ph.closePort()


def fmt(q):
    return "  ".join(f"{n.split('_')[-1]}={v:+7.2f}" for n, v in zip(JOINTS, q))


def load_teach():
    if not os.path.exists(TEACH):
        raise SystemExit("teach.json 없음 -- --teach 먼저")
    return json.load(open(TEACH, encoding="utf-8"))


def save_teach(d):
    json.dump(d, open(TEACH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


# ------------------------------------------------------------------------ modes
def teach(args):
    cam = Cam(args.cam)
    arm = make_arm(args)                          # parks every goal at present, torque on
    torque(arm, (PAN, LIFT, ELBOW), False)        # arm limp, wrist held
    q0 = read_firm(arm)
    print(f"\n손목 고정됨: wrist_flex={q0[WFLEX]:+.1f}  wrist_roll={q0[WROLL]:+.1f}")
    print("pan/lift/elbow 는 풀림 -- 손으로 옮겨서 막대 끝을 테이블에 대고, 손 떼고 Enter.")
    print("작업 영역 네 귀퉁이 + 가운데 (5점). 끝나면 q.\n")
    pts = []
    try:
        while True:
            s = input(f"  [{len(pts)}점] 끝이 테이블에 닿았으면 Enter / q=끝 > ").strip().lower()
            if s == "q":
                break
            q = read_firm(arm)
            f = cam.grab()
            if q is None or f is None:
                print("    읽기 실패 -- 다시"); continue
            p = marker.detect(f)
            if p is None:
                os.makedirs(DBG, exist_ok=True)
                cv2.imwrite(os.path.join(DBG, "teach_fail.png"), marker.draw(f)[0])
                print("    스티커 인식 실패 -- 손이 화면에 있거나 끝이 가려짐. output/auto_collect/teach_fail.png"); continue
            pts.append({"uv": [p[0], p[1]], "joints": q})
            print(f"    기록: px=({p[0]:.0f},{p[1]:.0f})  {fmt(q)}")
    finally:
        release_port(arm)          # wrist stays held, arm stays resting on the tip
        cam.close()
    if len(pts) < 3:
        raise SystemExit(f"{len(pts)}점 -- 최소 3점 필요, 저장 안 함")
    d = {"wrist": [q0[WFLEX], q0[WROLL]], "points": pts}
    if os.path.exists(TEACH):
        d["down"] = json.load(open(TEACH, encoding="utf-8")).get("down")
    save_teach(d)
    print(f"\n저장: {TEACH}  ({len(pts)}점)  -- 다음: --probe (끝이 테이블에 닿은 채로)")


def probe(args):
    d = load_teach()
    arm = make_arm(args)
    try:
        # Probe at the NEAR side (arm folded): the gravity torque on shoulder_lift is small
        # there, so "can't move up" can only mean the table, not a weak motor.
        torque(arm, (PAN, LIFT, ELBOW), False)
        input("\n팔이 풀렸다 (손목은 잡혀 있음). 팔을 접어서 몸 쪽 가까운 곳에 막대 끝을 테이블에"
              "\n대고, 손 떼고 Enter > ")
        q0 = hold_here(arm)
        strong(arm)
        print("현재:", fmt(q0))
        res = {}
        for sign in (+1, -1):
            t = list(q0); t[LIFT] = q0[LIFT] + 4.0 * sign
            set_goal(arm, t); time.sleep(1.0)
            q = read_firm(arm)
            if q is None:
                print("  읽기 실패 -- 중단"); set_goal(arm, q0); return
            moved = q[LIFT] - q0[LIFT]
            res[sign] = moved
            print(f"  lift {4.0*sign:+.0f}deg 명령 -> 실제 {moved:+.2f}deg 이동")
            set_goal(arm, q0); time.sleep(1.0)
        blocked = [s for s, mv in res.items() if abs(mv) < 1.0]
        free = [s for s, mv in res.items() if abs(mv) >= 2.5]
        if len(blocked) != 1 or len(free) != 1:
            print("\n판정 불가 -- 끝이 테이블에 닿아 있는지 확인 (양쪽 다 움직이면 떠 있는 것,"
                  " 양쪽 다 막히면 관절 한계/걸림).")
            return
        d["down"] = blocked[0]
        save_teach(d)
        print(f"\ndown = shoulder_lift {'+' if d['down'] > 0 else '-'} 방향.  저장: {TEACH}  -- 다음: --run")
    finally:
        release_port(arm)


def run(args):
    d = load_teach()
    if not d.get("down"):
        raise SystemExit("down 방향 없음 -- --probe 먼저")
    DOWN = d["down"]; UP = -DOWN
    pts = d["points"]
    J = np.array([p["joints"] for p in pts]); UV = np.array([p["uv"] for p in pts])
    # lift, elbow and wrist_flex have parallel axes, so lift+elbow+wrist_flex is the hand's
    # pitch. The teach points hold it at ~166 deg while wrist_flex alone spans 50 deg --
    # keep the PITCH constant (wrist = pitch - lift - elbow), not the wrist angle.
    pitch = float(np.median(J[:, LIFT] + J[:, ELBOW] + J[:, WFLEX]))
    wr = float(np.median(J[:, WROLL]))
    pose = lambda pn, lift, e: [pn, lift, e, pitch - lift - e, wr]

    # rows = target contact lift (near -> far); elbow follows the taught lift/elbow curve;
    # each row's pan span is interpolated between the near-side and far-side teach points
    lo, hi = J[:, LIFT].min(), J[:, LIFT].max()
    ec = np.polyfit(J[:, LIFT], J[:, ELBOW], 2 if len(J) >= 4 else 1)
    near = J[J[:, LIFT] <= lo + 15, PAN]; far = J[J[:, LIFT] >= hi - 15, PAN]
    rows_L = np.linspace(lo, hi, args.ne)                  # near first: --probe leaves it there
    order = []
    for i, L in enumerate(rows_L):
        t = (L - lo) / max(1e-6, hi - lo)
        p0 = (1 - t) * near.min() + t * far.min()
        p1 = (1 - t) * near.max() + t * far.max()
        pans = np.linspace(p0, p1, args.np)
        for pn in (pans if i % 2 == 0 else pans[::-1]):
            order.append((float(pn), float(L), float(np.polyval(ec, L))))
    roi = (int(UV[:, 0].min() - ROI_MARGIN), int(UV[:, 1].min() - ROI_MARGIN),
           int(UV[:, 0].max() + ROI_MARGIN), int(UV[:, 1].max() + ROI_MARGIN))
    print(f"격자 {args.ne}행 x {args.np}열 = {len(order)}점   lift {lo:+.1f}..{hi:+.1f}   "
          f"손 피치 {pitch:.1f}   wrist_roll {wr:+.1f}   down={DOWN:+d}")

    os.makedirs(DBG, exist_ok=True)
    cam = Cam(args.cam)
    arm = make_arm(args)
    strong(arm)                                   # full torque: the extended arm is heavy
    rows, done = [], False
    try:
        q = read_firm(arm)
        cur = pose(q[PAN], q[LIFT] + UP * RETRACT, q[ELBOW])
        arm.move_joints_deg(cur, secs=1.5, max_step_deg=90)          # lift off the table first
        for n, (pn, L, e) in enumerate(order):
            start = L + UP * START_ABOVE
            if n < args.confirm:
                s = input(f"\n[{n+1}/{len(order)}] pan={pn:+.1f} lift={L:+.1f} elbow={e:+.1f} 로 이동"
                          f" -- Enter / q=중단 > ")
                if s.strip().lower() == "q":
                    break
            # rise first (in place), then swing, then drop to the start height
            transit = cur[LIFT] if UP * (cur[LIFT] - start) > 0 else start
            arm.move_joints_deg(pose(cur[PAN], transit, cur[ELBOW]), secs=1.0, max_step_deg=90)
            arm.move_joints_deg(pose(pn, transit, e), secs=2.0, max_step_deg=90)
            arm.move_joints_deg(pose(pn, start, e), secs=0.8, max_step_deg=90)
            time.sleep(0.4)
            q0 = read_firm(arm)
            base = DOWN * (start - q0[LIFT]) if q0 else 0.0     # gravity sag, not contact
            # descend until the table stalls shoulder_lift (lag grows beyond the sag)
            goal, hit, streak = start, None, 0
            while UP * (goal - L) > -MAX_PAST:
                goal += DOWN * STEP
                set_goal(arm, pose(pn, goal, e)); time.sleep(0.08)
                qq = read_firm(arm)
                if qq is None:
                    continue
                streak = streak + 1 if DOWN * (goal - qq[LIFT]) - base > STALL_DEG else 0
                if streak >= 2:
                    hit = qq; break
            if hit is None:
                print(f"  [{n+1}] 접촉 없음 (예측 {L:+.1f}에서 {MAX_PAST:.0f}deg 더 내려도) -- 건너뜀")
                cur = pose(pn, goal + UP * RETRACT, e)
                arm.move_joints_deg(cur, secs=1.0, max_step_deg=90)
                continue
            set_goal(arm, pose(pn, hit[LIFT], e)); time.sleep(0.4)   # stop pushing, rest on the tip
            qc = read_firm(arm) or hit
            f = cam.grab()
            p = marker.detect(f, roi) if f is not None else None
            img, _ = marker.draw(f, roi) if f is not None else (None, None)
            if img is not None:
                cv2.imwrite(os.path.join(DBG, f"pt{n+1:02d}.png"), img)
            if p is None:
                print(f"  [{n+1}] 접촉 OK, 스티커 인식 실패 -- 건너뜀 (output/auto_collect/pt{n+1:02d}.png)")
            else:
                rows.append([round(p[0], 1), round(p[1], 1)] + [round(v, 2) for v in qc])
                write_points(rows)
                print(f"  [{n+1}] px=({p[0]:.0f},{p[1]:.0f})  {fmt(qc)}   "
                      f"(예측 lift {L:+.1f}, 피치 {qc[LIFT]+qc[ELBOW]+qc[WFLEX]:.1f})")
            if n == len(order) - 1:
                done = True                               # end resting on the table
                break
            cur = pose(pn, qc[LIFT] + UP * RETRACT, e)
            arm.move_joints_deg(cur, secs=1.0, max_step_deg=90)
    except KeyboardInterrupt:
        print("\n중단 -- 토크 유지한 채 포트만 닫음 (팔이 떨어지지 않게)")
    finally:
        cam.close()
        if done:
            arm.disconnect()       # tip is on the table -> safe to let go
        else:
            release_port(arm)
    print(f"\n{len(rows)}점 -> {POINTS}   다음: python robot\\calib\\solve.py")


def write_points(rows):
    with open(POINTS, "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["u", "v"] + JOINTS); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--teach", action="store_true")
    g.add_argument("--probe", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--with-roll", action="store_true", help="wrist_roll(id5) 도 켬 (배선 고친 뒤)")
    ap.add_argument("--np", type=int, default=5, help="행마다 pan 점 수")
    ap.add_argument("--ne", type=int, default=4, help="행 수 (가까운 쪽 -> 먼 쪽)")
    ap.add_argument("--confirm", type=int, default=2, help="처음 N점은 이동 전 Enter 확인")
    a = ap.parse_args()
    (teach if a.teach else probe if a.probe else run)(a)
