r"""theta 캘리브레이션: 이미지 파지각 -> wrist_roll.

GG-CNN 은 이미지 평면상의 파지각 theta_img 를 낸다. 손가락이 그 방향으로
닫히게 하려면 wrist_roll 을 얼마로 줘야 하는지 (gain, offset) 를 구한다.

    wrist_roll = gain * theta_img + offset

절차: wrist_roll 을 몇 개 값으로 돌려가며, 각 자세에서 손가락이 닫히는 선의
양 끝을 클릭 -> (wrist_roll, theta_img) 쌍을 모아 직선 적합.

    python calib/collect_theta.py --cam 0

각도 정의는 ggcnn/grasp.py 의 GraspRectangle.angle 과 동일:
    theta = atan2(-dy, dx) 를 (-90, +90] 으로 정규화 (180도 주기)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from robot.robot_control import (Config, SO101, ADDR_TORQUE_ENABLE as A_TQ,
                           ADDR_TORQUE_LIMIT as A_LIM, ADDR_GOAL_POSITION as A_GOAL,
                           ADDR_GOAL_SPEED as A_SPD)

HERE = os.path.dirname(os.path.abspath(__file__))
ROLL_IDX = 4          # wrist_roll
ARM_TORQUE_LIMIT = 600


def norm_angle_deg(dx, dy):
    """ggcnn 과 같은 정규화: (-90, +90]. dy 는 화면 아래가 +."""
    a = np.arctan2(-dy, dx)
    return np.degrees((a + np.pi / 2) % np.pi - np.pi / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--motor", action="store_true",
                    help="wrist_roll 을 모터로 돌림 (id5 교체 후). 기본은 손으로 돌리는 수동 모드")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == "nt" else 0)   # plain
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)                    # VideoCapture(idx) picks
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)                    # MSMF on this machine,
    if not cap.read()[0]:                                      # which failed to open it
        print(f"카메라 {args.cam} 못 엶"); sys.exit(1)

    cfg = Config()
    arm = SO101(cfg)
    arm.connect()
    bus, jroll = arm.bus, arm.joints[ROLL_IDX]
    manual = not args.motor
    if args.motor and jroll.servo_id in cfg.arm_disabled_ids:
        print(f"!! id{jroll.servo_id} 는 arm_disabled_ids 에 있다 (goal 쓰기에 리셋되는 고장). "
              "버튼을 눌러도 안 돌 수 있다.")

    samples = []            # (wrist_roll_deg, theta_img_deg)
    frozen = {"img": None, "roll": None, "p1": None}

    root = tk.Tk()
    root.title(f"theta 캘리브 (cam {args.cam})")
    canvas = tk.Label(root); canvas.grid(row=0, column=0, rowspan=10)
    info = tk.Label(root, text="", justify="left", font=("Consolas", 10))
    info.grid(row=10, column=0, sticky="w")

    ctl = tk.Frame(root); ctl.grid(row=0, column=1, rowspan=10, padx=10, sticky="n")

    last_roll = {"v": None}

    def read_roll(retries=8):
        """wrist_roll 은 읽기만 한다. id5 는 goal 쓰기에 리셋되는 고장이라
        버스(ids)에서 빠져 있으므로 id 를 직접 지정해 위치만 읽는다."""
        s = bus.read_one(jroll.servo_id, retries=retries)
        if s is not None:
            last_roll["v"] = arm._steps_to_deg(jroll, s)
        return None if s is None else last_roll["v"]

    def status(msg="", quick=False):
        # quick: 주기 갱신용. 재시도 1회만 -- 실패하면 직전 값 표시 (UI/카메라 안 멈추게)
        r = read_roll(retries=1) if quick else read_roll()
        if r is None and quick:
            r = last_roll["v"]
        info.config(text=f"wrist_roll = {'--' if r is None else f'{r:+.1f}'} deg   "
                         f"수집 {len(samples)}쌍\n"
                    + "  ".join(f"({a:+.0f},{b:+.0f})" for a, b in samples)
                    + ("\n" + msg if msg else ""))

    if manual:
        tk.Label(ctl, text="수동 회전 모드\n\n손목(wrist_roll)을\n손으로 돌려놓고\n"
                           "받친 뒤 캡처.\n각도는 자동으로\n읽는다.\n\n"
                           "-60~+60 사이\n4~5개 각도 권장", justify="left").pack()
    else:
        def goto_roll(deg):
            cur = bus.read_one(jroll.servo_id, retries=8)
            if cur is None:
                status("wrist_roll 읽기 실패"); return
            bus._pk.write1ByteTxRx(bus._ph, jroll.servo_id, A_TQ, 1)
            bus._pk.write2ByteTxRx(bus._ph, jroll.servo_id, A_LIM, ARM_TORQUE_LIMIT)
            bus._pk.write2ByteTxRx(bus._ph, jroll.servo_id, A_SPD, 400)
            tgt = arm._deg_to_steps(jroll, deg)
            bus._pk.write2ByteTxRx(bus._ph, jroll.servo_id, A_GOAL, int(tgt))
            root.after(1400, lambda: status(f"wrist_roll -> {deg:+.0f} 명령"))

        tk.Label(ctl, text="wrist_roll 이동").pack()
        for d in (-60, -30, 0, 30, 60):
            tk.Button(ctl, text=f"{d:+d}", width=8,
                      command=lambda x=d: goto_roll(x)).pack(pady=1)

    def tick():                 # 현재 wrist_roll 각도를 1초마다 표시
        if frozen["img"] is None:
            status(quick=True)
        root.after(1000, tick)
    root.after(1000, tick)

    # 쥔 막대의 축을 클릭하는 편이 손가락 끝보다 훨씬 정확하다.
    # 손가락은 막대를 '가로질러' 닫히므로 축 각도 + 90 = 닫히는 선 각도.
    perp = tk.BooleanVar(value=True)
    tk.Label(ctl, text="\n두 점 클릭 방법", justify="left").pack()
    tk.Checkbutton(ctl, text="쥔 막대의 축을 클릭\n(+90 보정)", variable=perp,
                   justify="left").pack()
    tk.Label(ctl, text="끄면 = 엄지끝→반대\n손가락 중앙을 직접 클릭",
             justify="left", fg="gray").pack()

    def capture():
        ok, f = cap.read()
        if not ok:
            return
        r = read_roll()
        if r is None:
            status("wrist_roll 읽기 실패"); return
        frozen.update(img=f.copy(), roll=r, p1=None)
        show()
        status("정지됨. 손가락이 닫히는 선의 한쪽 끝 클릭 (Esc=취소)")

    def show(mark=None):
        f = frozen["img"].copy()
        if mark is not None:
            cv2.circle(f, mark, 6, (0, 0, 255), 2)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        im = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.im = im; canvas.config(image=im)

    def on_click(e):
        if frozen["img"] is None:
            return
        if frozen["p1"] is None:
            frozen["p1"] = (e.x, e.y)
            show(frozen["p1"])
            status("반대쪽 끝 클릭")
            return
        x1, y1 = frozen["p1"]
        th = norm_angle_deg(e.x - x1, e.y - y1)
        if perp.get():          # 막대 축 -> 닫히는 선은 그 수직
            th = (th + 90.0 + 90.0) % 180.0 - 90.0
        samples.append((round(frozen["roll"], 2), round(float(th), 2)))
        frozen.update(img=None, p1=None)
        status(f"기록: roll={frozen['roll'] if False else samples[-1][0]:+.1f} "
               f"theta_img={samples[-1][1]:+.1f}")

    canvas.bind("<Button-1>", on_click)
    root.bind("<Escape>", lambda _e: (frozen.update(img=None, p1=None), status("취소")))

    def undo():
        if samples:
            samples.pop()
        status()

    def wrap90(x):
        return (np.asarray(x, float) + 90.0) % 180.0 - 90.0

    def solve_save():
        """기울기를 ±1 로 고정하고 부호와 offset 만 추정한다.

        손이 테이블을 수직으로 내려다보면 손목 1도 = 이미지 1도 (기하학적으로 고정).
        SO-101 wrist_roll 은 약 -24~+20deg 로 좁아서, 기울기까지 데이터로 추정하면
        클릭 오차에 크게 흔들린다. 자유 적합 기울기는 자세 점검용으로만 보고한다.
        """
        if len(samples) < 3:
            status("최소 3쌍 필요"); return
        roll = np.array([s[0] for s in samples], float)
        th = np.array([s[1] for s in samples], float)
        if np.ptp(roll) < 15:
            status(f"손목 각도 범위가 {np.ptp(roll):.0f}deg 뿐 -- 끝에서 끝까지 넓게 찍어라"); return

        best = None
        for sgn in (+1.0, -1.0):                         # theta = sgn*roll + b  (180도 주기)
            d2 = np.radians(2.0 * (th - sgn * roll))
            b = 0.5 * np.degrees(np.arctan2(np.sin(d2).mean(), np.cos(d2).mean()))
            res = np.abs(wrap90(th - sgn * roll - b))
            if best is None or res.mean() < best[2].mean():
                best = (sgn, b, res)
        sgn, b, res = best
        gain, offset = sgn, float(wrap90(-sgn * b))      # roll = gain*theta + offset

        # 진단용 자유 적합 (자세가 수직인지)
        order = np.argsort(roll)
        tu = th[order].copy()
        for i in range(1, len(tu)):
            tu[i] = tu[i - 1] + wrap90(tu[i] - tu[i - 1])
        free_slope = float(np.polyfit(roll[order], tu, 1)[0])

        path = os.path.join(HERE, "handeye.json")
        d = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        d["theta_gain"] = float(gain)
        d["theta_offset_deg"] = offset
        d["theta_fit"] = {
            "method": "slope fixed to +/-1, sign+offset estimated",
            "n": len(samples), "roll_span_deg": float(np.ptp(roll)),
            "resid_mean_deg": float(res.mean()), "resid_max_deg": float(res.max()),
            "free_fit_slope": free_slope,
            "samples": [[float(x), float(y)] for x, y in zip(roll, th)],
        }
        json.dump(d, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        msg = (f"gain={gain:+.0f}  offset={offset:+.1f}deg   잔차 mean {res.mean():.1f} / "
               f"max {res.max():.1f}deg   (자유적합 기울기 {free_slope:+.2f})\nhandeye.json 저장 완료")
        warn = []
        if res.max() > 6.0:
            warn.append("잔차가 크다 -- 클릭 불일치 또는 막대/테이프 미끄러짐")
        if abs(abs(free_slope) - 1.0) > 0.35:
            warn.append("자유적합 기울기가 ±1 과 멀다 -- 손이 수직으로 내려다보는지 확인")
        if warn:
            msg += "\n!! " + "\n!! ".join(warn)
        print(msg)
        status(msg)

    br = tk.Frame(root); br.grid(row=11, column=0, columnspan=2, pady=6)
    tk.Button(br, text="이 자세 캡처", width=13, command=capture).pack(side="left", padx=4)
    tk.Button(br, text="마지막 취소", width=11, command=undo).pack(side="left", padx=4)
    tk.Button(br, text="계산 + 저장", width=12, command=solve_save).pack(side="left", padx=4)

    def loop():
        if frozen["img"] is None:
            ok, f = cap.read()
            if ok:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                im = ImageTk.PhotoImage(Image.fromarray(rgb))
                canvas.im = im; canvas.config(image=im)
        root.after(40, loop)

    status()
    loop()
    try:
        root.mainloop()
    finally:
        cap.release(); arm.disconnect()


if __name__ == "__main__":
    main()
