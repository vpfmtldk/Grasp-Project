r"""handeye.json 을 써서 픽셀 -> 팔 관절각 5개.

    from calib.pixel_to_arm import HandEye
    he = HandEye()
    q = he.joints_for_pixel(700, 350)                    # theta 없이 (기준 손목각)
    q = he.joints_for_pixel(700, 350, theta_img_deg=30)  # 파지 각도 반영
    arm.move_joints_deg(q, secs=2.0)

pan/lift/elbow 는 (u,v) 다항식, wrist_flex 는 캘리브 상수 또는 (피치 - lift - elbow),
wrist_roll 은 theta_gain * theta + theta_offset.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALL_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class HandEye:
    def __init__(self, path=None):
        path = path or os.path.join(HERE, "handeye.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.fit_joints = d["fit_joints"]
        self.coef = np.array(d["coef"])                       # (6, 3)
        self.wrist_flex = d["wrist_flex_const_deg"]
        self.wrist_model = d.get("wrist_model", "const")
        self.wrist_pitch = d.get("wrist_pitch_deg")
        self.theta_gain = d.get("theta_gain", 1.0)
        self.theta_offset = d.get("theta_offset_deg", d.get("wrist_roll_ref_deg", 0.0))
        self.uv_range = d.get("uv_range")
        self.loo_mean_deg = d.get("loo_mean_deg")

    @staticmethod
    def _feat(u, v):
        u, v = u / 1000.0, v / 1000.0
        return np.array([1.0, u, v, u * u, u * v, v * v])

    def in_range(self, u, v, margin=0.0):
        """캘리브한 픽셀 범위 안인지. 밖은 외삽이라 신뢰할 수 없다."""
        if not self.uv_range:
            return True
        (u0, u1), (v0, v1) = self.uv_range["u"], self.uv_range["v"]
        du, dv = (u1 - u0) * margin, (v1 - v0) * margin
        return (u0 + du) <= u <= (u1 - du) and (v0 + dv) <= v <= (v1 - dv)

    def joints_for_pixel(self, u, v, theta_img_deg=None):
        fitted = self._feat(u, v) @ self.coef                 # pan, lift, elbow
        q = dict(zip(self.fit_joints, (float(x) for x in fitted)))
        if "wrist_flex" not in q:                   # "fitted" model (teammate data) keeps its own
            q["wrist_flex"] = float(self.wrist_pitch - q["shoulder_lift"] - q["elbow_flex"]
                                    if self.wrist_model == "pitch" else self.wrist_flex)
        q["wrist_roll"] = float(
            self.theta_offset if theta_img_deg is None
            else self.theta_gain * theta_img_deg + self.theta_offset)
        return [q[j] for j in ALL_JOINTS]


if __name__ == "__main__":
    he = HandEye()
    print(f"LOO 평균 관절오차 {he.loo_mean_deg:.2f} deg   유효범위 {he.uv_range}")
    print((f"wrist_flex = {he.wrist_pitch:.1f} - lift - elbow" if he.wrist_model == "pitch"
           else f"wrist_flex 상수 {he.wrist_flex:.1f}") + "   wrist_roll = "
          f"{he.theta_gain}*theta + {he.theta_offset:.1f}")
    for uv in [(600, 280), (700, 350), (820, 430), (400, 200)]:
        ok = "" if he.in_range(*uv) else "   <-- 범위 밖(외삽, 신뢰불가)"
        print(f"  px{uv} -> " + "  ".join(
            f"{n.split('_')[-1]}={a:+7.1f}" for n, a in
            zip(ALL_JOINTS, he.joints_for_pixel(*uv))) + ok)
