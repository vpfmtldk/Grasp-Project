r"""points.csv -> 픽셀(u,v) → 팔 관절각 보간 모델.

위에서 내려잡기 한 자세만 쓰므로:
  * shoulder_pan / shoulder_lift / elbow_flex 만 (u,v) 의 함수로 맞춘다
  * wrist_flex 는 상수, 또는 손 피치(lift+elbow+wrist_flex) 상수 -- 데이터로 자동 판별
  * wrist_roll 은 파지 각도 theta 로 별도 지정
팔이 5자유도인데 목표는 2자유도라 여분(redundancy)이 있다 -- 같은 픽셀을
여러 관절 조합으로 도달할 수 있어서, 수집 중 자세가 흔들리면 함수가
다가(multi-valued)가 되어 맞지 않는다. 그래서 3개만 맞춘다.

    python calib/solve.py                 # 이상점 자동 제외
    python calib/solve.py --keep-all      # 전부 사용
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
FIT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex"]


def features(uv):
    uv = np.atleast_2d(np.asarray(uv, float))
    u, v = uv[:, 0] / 1000.0, uv[:, 1] / 1000.0
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)


def fit(uv, q):
    coef, *_ = np.linalg.lstsq(features(uv), q, rcond=None)
    return coef


def predict(coef, uv):
    return features(uv) @ coef


def loo_err(uv, q):
    n = len(uv)
    e = np.zeros_like(q)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        e[i] = np.abs(predict(fit(uv[m], q[m]), uv[i])[0] - q[i])
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-all", action="store_true", help="이상점 제외 안 함")
    ap.add_argument("--max-err", type=float, default=6.0, help="이상점 판정 (관절 deg)")
    args = ap.parse_args()

    path = os.path.join(HERE, "points.csv")
    if not os.path.exists(path):
        print("points.csv 없음 -- collect.py 먼저"); sys.exit(1)
    rows = list(csv.DictReader(open(path)))
    uv = np.array([[float(r["u"]), float(r["v"])] for r in rows])
    qall = np.array([[float(r[j]) for j in JOINTS] for r in rows])
    cols = [JOINTS.index(j) for j in FIT_JOINTS]
    q = qall[:, cols]
    print(f"{len(rows)}점 로드")

    keep = np.ones(len(rows), bool)
    if not args.keep_all:
        e = loo_err(uv, q)
        bad = e.max(axis=1) > args.max_err
        if bad.any() and (~bad).sum() >= 8:
            keep = ~bad
            for i in np.where(bad)[0]:
                print(f"  이상점 제외: idx{i} px=({uv[i,0]:.0f},{uv[i,1]:.0f}) "
                      f"최대오차 {e[i].max():.1f}deg")
        elif bad.any():
            print(f"  이상점 {bad.sum()}개 있으나 남는 점이 8개 미만 -> 전부 사용")

    uvk, qk = uv[keep], q[keep]
    e = loo_err(uvk, qk)
    print(f"\nLeave-one-out 오차 ({keep.sum()}점, {len(FIT_JOINTS)}관절):")
    for k, j in enumerate(FIT_JOINTS):
        print(f"  {j:14} mean {e[:,k].mean():5.2f}  max {e[:,k].max():5.2f}")
    print(f"  전체 mean {e.mean():.2f} deg")

    # 손목 모델 자동 판별: wrist_flex 자체가 일정했나(const), 아니면 손 피치
    # lift+elbow+wrist_flex 가 일정했나(pitch, auto_collect.py). 더 일정한 쪽을 쓴다.
    wf = qall[keep, JOINTS.index("wrist_flex")]
    pitch = qk[:, 1] + qk[:, 2] + wf
    wf_const, wf_spread = float(np.median(wf)), float(wf.max() - wf.min())
    pitch_const, pitch_spread = float(np.median(pitch)), float(pitch.max() - pitch.min())
    model = "pitch" if pitch_spread < wf_spread else "const"
    wr = qall[keep, JOINTS.index("wrist_roll")]
    if model == "pitch":
        print(f"\n손목 모델: 손 피치 일정  wrist_flex = {pitch_const:.1f} - lift - elbow"
              f"  (피치 범위 {pitch_spread:.1f} deg, wrist_flex 범위 {wf_spread:.1f})")
        spread = pitch_spread
    else:
        print(f"\n손목 모델: wrist_flex 상수 = {wf_const:.1f} deg  (수집값 범위 {wf_spread:.1f} deg)")
        spread = wf_spread
    if spread > 5:
        print("  !! 수집 중 손 각도가 크게 변했다. pan/lift/elbow 정확도도 같이 떨어진다. 재수집 권장.")
    print(f"wrist_roll 수집값 = {wr.mean():.1f} deg (파지 각도 theta 기준점)")

    out = {
        "type": "pixel_to_joint_poly2",
        "fit_joints": FIT_JOINTS,
        "coef": fit(uvk, qk).tolist(),      # (6, 3)
        "wrist_model": model,               # "pitch": wrist = pitch - lift - elbow
        "wrist_pitch_deg": pitch_const,
        "wrist_flex_const_deg": wf_const,
        "wrist_roll_ref_deg": float(wr.mean()),
        "theta_gain": 1.0,                  # collect_theta.py 로 확정
        "theta_offset_deg": float(wr.mean()),
        "n_points": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "loo_mean_deg": float(e.mean()),
        "loo_per_joint_deg": {j: float(e[:, k].mean()) for k, j in enumerate(FIT_JOINTS)},
        "uv_range": {"u": [float(uvk[:, 0].min()), float(uvk[:, 0].max())],
                     "v": [float(uvk[:, 1].min()), float(uvk[:, 1].max())]},
        "note": "collect.py 접근 자세에서만 유효. uv_range 밖은 외삽이라 신뢰 불가.",
    }
    with open(os.path.join(HERE, "handeye.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(HERE, 'handeye.json')}")
    print(f"유효 픽셀 범위: u {out['uv_range']['u']}  v {out['uv_range']['v']}")
    if e.mean() > 3.0:
        print("!! 평균 오차 3deg 초과 -- 점 추가 / 자세 일관성 확인 필요")


if __name__ == "__main__":
    main()
