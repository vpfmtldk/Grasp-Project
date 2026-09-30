r"""Hand-eye 대응점 수집 (평면, FK/IK 불필요).

접근: 카메라는 고정, 파지는 위에서 내려잡기 한 가지 자세.
     -> 픽셀 (u,v) 를 5개 관절각으로 직접 보간 학습. FK도 URDF도 필요 없음.
     대신 캘리브 때 쓴 접근 자세(높이/손목피치)에서만 유효.

준비:
  * 손 중심(손가락이 닫히는 축)에 아래로 향한 얇은 포인터를 단다 -- 그 끝이
    "파지 지점"이 되게. 캘리브 내내 같은 포인터/같은 끝.
  * 카메라가 작업대를 보게 고정.

사용:
  python calib/collect.py --cam 1

  1) 팔 jog 버튼으로 포인터 끝을 작업대의 한 점에 살짝 닿게
  2) [이 지점 캡처] -> 프레임 정지 -> 포인터 끝을 클릭 -> 관절각 자동 기록
  3) 작업 영역 전체에 12~16점 (구석/중앙/원근 골고루)
  4) [저장] -> calib/points.csv

wrist_roll(각도) 캘리브는 collect_theta.py 로 따로.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from robot.robot_control import (Config, SO101, ADDR_TORQUE_ENABLE as rc_TORQUE_ENABLE,
                           ADDR_TORQUE_LIMIT as rc_TORQUE_LIMIT,
                           ADDR_GOAL_POSITION as rc_GOAL_POSITION,
                           ADDR_GOAL_SPEED as rc_GOAL_SPEED)

HERE = os.path.dirname(os.path.abspath(__file__))
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
arm_torque_limit = 600   # STS3215 Max_Torque
# 잠근 관절 허용 흔들림. 손목 서보는 손 무게 때문에 팔 자세에 따라 몇 도씩
# 처지는데(재현성 있는 처짐 -- 실행 때도 같은 자세면 똑같이 처짐), 2deg 로
# 잡았더니 이 정상 처짐까지 전부 거부됐다. 막아야 할 건 지난번 15deg 같은 큰 흔들림.
DRIFT_MAX_DEG = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1, help="카메라 인덱스")
    ap.add_argument("--out", default=os.path.join(HERE, "points.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == "nt" else 0)   # plain
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)                    # VideoCapture(idx) picks
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)                    # MSMF on this machine,
    if not cap.read()[0]:                                      # which failed to open it
        print(f"카메라 {args.cam} 못 엶. --cam 다른 번호 시도"); sys.exit(1)

    cfg = Config()
    arm = SO101(cfg, dry_run=args.dry_run)
    arm.connect()
    # 손목(잠글 관절)은 서보 최대 토크로 -- connect() 기본 500(약 50%)이면 손 무게에
    # 더 크게 처진다. 이 서보의 상한(Max_Torque)이 600 이라 그 이상은 의미 없음.
    if not args.dry_run:
        for j in arm.joints:
            if j.name in ("wrist_flex", "wrist_roll"):
                arm.bus._sync_end()
                arm.bus._pk.write2ByteTxRx(arm.bus._ph, j.servo_id, rc_TORQUE_LIMIT, arm_torque_limit)

    rows = []
    frozen = {"img": None}
    pose = {"q": None}
    locked_ref = {}     # 잠근 관절의 첫 캡처 시 값 (드리프트 감시 기준)

    root = tk.Tk()
    root.title(f"hand-eye 대응점 수집  (cam {args.cam})")

    canvas = tk.Label(root)
    canvas.grid(row=0, column=0, rowspan=12)
    info = tk.Label(root, text="", justify="left", font=("Consolas", 10))
    info.grid(row=12, column=0, sticky="w")

    # ---- 팔 jog ----
    step = tk.DoubleVar(value=3.0)
    ctl = tk.Frame(root)
    ctl.grid(row=0, column=1, rowspan=12, padx=10, sticky="n")
    tk.Label(ctl, text="팔 jog (deg)").pack()
    tk.Scale(ctl, from_=0.5, to=15, resolution=0.5, orient="horizontal",
             variable=step, label="step").pack()

    def read_pose():
        q = arm.read_joints_deg()
        pose["q"] = q
        info.config(text="관절: " + "  ".join(
            f"{n}={'--' if v is None else f'{v:+.1f}'}" for n, v in zip(JOINTS, q))
            + f"\n수집: {len(rows)}점")
        return q

    def jog(idx, sign):
        """단일 서보를 스텝 단위로 직접 이동. 다관절 보간/안전가드를 거치지 않아
        실패 지점이 적고, 왜 안 움직였는지 화면에 그대로 찍힌다."""
        b = arm.bus
        sid = arm.joints[idx].servo_id
        name = JOINTS[idx]
        if name in lock and lock[name].get():
            info.config(text=f"{name} 잠김 -- 수집 내내 고정해야 한다.\n"
                             "(이 관절이 흔들리면 픽셀->관절 함수가 다가가 되어 정확도가 무너진다)")
            return
        if b._pk is None:
            info.config(text="버스 미연결"); return
        cur = b.read_one(sid, retries=8)
        if cur is None:
            info.config(text=f"id{sid}({JOINTS[idx]}) 읽기 실패 -- 배선/전원 확인"); return
        # 매번 토크 + 토크한계 재확인 (torque off 눌렀다가 다시 움직일 때 대비)
        b._pk.write1ByteTxRx(b._ph, sid, rc_TORQUE_ENABLE, 1)
        b._pk.write2ByteTxRx(b._ph, sid, rc_TORQUE_LIMIT, arm_torque_limit)
        b._pk.write2ByteTxRx(b._ph, sid, rc_GOAL_SPEED, 500)
        d = int(round(sign * step.get() / b.deg_per_step))
        tgt = int(max(0, min(b.steps_per_rev - 1, cur + d)))
        b._pk.write2ByteTxRx(b._ph, sid, rc_GOAL_POSITION, tgt)

        def check():
            now = b.read_one(sid, retries=6)
            if now is None:
                info.config(text=f"id{sid} 이동 후 읽기 실패")
            elif abs(now - cur) < 2:
                lim, _, _ = b._pk.read2ByteTxRx(b._ph, sid, rc_TORQUE_LIMIT)
                tq, _, _ = b._pk.read1ByteTxRx(b._ph, sid, rc_TORQUE_ENABLE)
                info.config(text=f"id{sid} 안 움직임: {cur}->{now} (목표 {tgt})  "
                                 f"torque_en={tq} torque_limit={lim}\n"
                                 f"limit 이 0 이면 전원/배선, 아니면 기구 간섭 의심")
            else:
                read_pose()
        root.after(int(600 + abs(d) * 2), check)

    # wrist_flex / wrist_roll 은 수집 내내 고정해야 한다 -> 기본 잠금
    lock = {n: tk.BooleanVar(value=True) for n in ("wrist_flex", "wrist_roll")}

    for i, n in enumerate(JOINTS):
        r = tk.Frame(ctl); r.pack(pady=1)
        tk.Label(r, text=n, width=13, anchor="w").pack(side="left")
        tk.Button(r, text="-", width=3, command=lambda k=i: jog(k, -1)).pack(side="left")
        tk.Button(r, text="+", width=3, command=lambda k=i: jog(k, +1)).pack(side="left")
        if n in lock:
            tk.Checkbutton(r, text="잠금", variable=lock[n]).pack(side="left")

    def torque_off():
        """잠긴 관절은 토크를 유지해서 손으로 옮겨도 안 틀어지게 한다."""
        b = arm.bus
        if b._pk is not None:
            for k, j in enumerate(arm.joints):
                if JOINTS[k] in lock and lock[JOINTS[k]].get():
                    continue          # 잠긴 관절은 그대로 잡고 있기
                b._pk.write1ByteTxRx(b._ph, j.servo_id, rc_TORQUE_ENABLE, 0)
        held = [n for n in lock if lock[n].get()]
        info.config(text="팔 torque OFF -- 손으로 자세 잡고 [이 지점 캡처]\n"
                         + (f"({', '.join(held)} 는 잠겨서 토크 유지)" if held else ""))
    tk.Button(ctl, text="팔 torque off", command=torque_off).pack(pady=4)

    # ---- 캡처 흐름 ----
    def capture():
        ok, f = cap.read()
        if not ok:
            return
        q = read_pose()
        if any(v is None for v in q):
            info.config(text="관절 읽기 실패 -- 다시"); return
        # 잠근 관절이 실제로 안 흔들렸는지 먼저 확인 (중력 처짐/수동 이동 대비).
        # 벗어났으면 아예 캡처를 거부한다 -- 경고만 띄우고 기록을 허용했더니
        # wrist_flex 가 15deg 흔들린 점들이 섞여 LOO 29deg 로 캘리브가 무너졌다.
        bad = ""
        for n in lock:
            k = JOINTS.index(n)
            ref = locked_ref.setdefault(n, q[k])
            drift = abs(q[k] - ref)
            if drift > DRIFT_MAX_DEG:
                bad += f"\n!! {n} = {q[k]:+.1f}  (기준 {ref:+.1f}, {drift:.1f}deg 벗어남)"
        if bad:
            info.config(text="캡처 거부 -- 잠근 관절이 첫 점 기준에서 벗어났다." + bad
                             + "\n기준값으로 되돌린 뒤 다시 캡처. (계속 안 맞으면 도구를 새로 시작)")
            return
        frozen["img"] = f.copy()
        frozen["q"] = list(q)
        show_frozen()
        info.config(text="정지됨. 사진에서 검지 끝을 클릭하라. (Esc=취소)")

    def show_frozen():
        f = frozen["img"]
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        im = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.im = im; canvas.config(image=im)

    def on_click(e):
        if frozen["img"] is None:
            return
        u, v = e.x, e.y
        rows.append([u, v] + [round(x, 2) for x in frozen["q"]])
        frozen["img"] = None
        info.config(text=f"저장: px=({u},{v})  총 {len(rows)}점")

    def cancel(_=None):
        frozen["img"] = None
    canvas.bind("<Button-1>", on_click)
    root.bind("<Escape>", cancel)

    def undo():
        if rows:
            rows.pop()
        read_pose()

    def save():
        with open(args.out, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["u", "v"] + JOINTS)
            w.writerows(rows)
        info.config(text=f"저장 완료: {args.out}  ({len(rows)}점)")

    br = tk.Frame(root); br.grid(row=13, column=0, columnspan=2, pady=6)
    tk.Button(br, text="이 지점 캡처", width=14, command=capture).pack(side="left", padx=4)
    tk.Button(br, text="마지막 취소", width=12, command=undo).pack(side="left", padx=4)
    tk.Button(br, text="저장", width=10, command=save).pack(side="left", padx=4)

    # ---- 라이브 프리뷰 루프 ----
    def loop():
        if frozen["img"] is None:
            ok, f = cap.read()
            if ok:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                im = ImageTk.PhotoImage(Image.fromarray(rgb))
                canvas.im = im; canvas.config(image=im)
        root.after(40, loop)

    read_pose()
    loop()
    try:
        root.mainloop()
    finally:
        cap.release()
        arm.disconnect()
    if rows:
        save()


if __name__ == "__main__":
    main()
