"""
can_grasp_env.py -- Gymnasium env: SO-101 + contact AmazingHand grasps and lifts a can.

The episode starts with the hand hovering over the can (randomised), the way the
vision pipeline (GR-ConvNet -> hand-eye) would hand over a rough pre-grasp; the policy
learns the last approach, the finger closure and the lift under real contact dynamics.

residual=True (default): the executed action = scripted grasp (descend -> close -> lift,
scripted_policy.py) + RES_SCALE x policy action. Plain PPO from scratch (residual=False)
got stuck touching the can without ever closing the hand (output/rl/ppo_can_v1_plain).

  obs  (50)  arm q/qd, finger angles, tool pose, can pose/vel, can-tool, contacts, rise,
             last action, episode phase, scripted grasp pose - current target
  act  (6)   [-1,1]: 5 arm joint target deltas (x MAX_DQ rad/step) + absolute grip target
  20 Hz control (25 x 2 ms physics steps), 150 steps = 7.5 s.
"""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robot.rl.sim_util import SCENE, Ids, READY, GRIP_OPEN, GRIP_CLOSED, ik, contacts, touch_set

MAX_DQ = 0.04            # rad per control step
FRAME_SKIP = 25
EP_STEPS = 150
SUCCESS_RISE = 0.06      # m
SUCCESS_HOLD = 10        # control steps held above SUCCESS_RISE
# 355 ml can (66 x 122 mm). Upright = top grasp with the fingers leaning 15 deg away from
# the base: straight down put Wrist_Pitch on its limit (0 deg margin, 10/48 lifts in the
# tilt sweep); 15 deg leaves ~20 deg margin and lifted at all 3 test positions (29/48).
# Finger row -> thumb runs tangentially; pinch centre 3 cm below the rim; pre-shape -0.1 rad.
U_TILT = np.radians(15)
U_DZ = -0.03             # pinch centre relative to the can top
OFF = 0.0                # along hand x (finger row -> thumb)
U_UP = 0.10              # vertical lift by IK
LIFT = 0.45              # scripted lift: shoulder Pitch raise (rad)
RES_SCALE = 0.5          # residual authority
G_PRE = 2 * (-0.10 - GRIP_OPEN) / (GRIP_CLOSED - GRIP_OPEN) - 1     # grip action for -0.1 rad
# lying can: fingers point away from the base, tilted 45 deg down, palm over the can,
# all four digits wrap it; pre-shape -0.3 rad; pinch centre 3 cm above the can axis;
# lift straight up by IK (a shoulder raise moves this folded arm mostly sideways).
L_TILT, L_DZ, L_BACK, L_UP = np.radians(45), 0.03, 0.0, 0.10
L_YAW_JITTER = np.radians(30)     # can axis within +-30 deg of across-the-reach (other yaws unreachable)
G_PRE_L = 2 * (-0.30 - GRIP_OPEN) / (GRIP_CLOSED - GRIP_OPEN) - 1
SEEDS = [READY, np.array([0, -2.5, 2.6, 1.0, 1.5]), np.array([0, -1.9, 2.0, 0.3, 0])]


class CanGraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, can_range=((-0.03, 0.03), (-0.135, -0.085)), residual=True,
                 can_mode="mixed"):
        """can_mode: "upright" | "lying" | "mixed" (50/50 per episode)."""
        self.residual = residual
        self.can_mode = can_mode
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.ids = Ids(self.m)
        self.half_h = float(self.m.geom_size[self.ids.can_geom][1])
        self.lo, self.hi = self.m.jnt_range[[self.m.joint(n).id for n in
                                             ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")]].T
        self.can_range = can_range
        self.render_mode = render_mode
        self._renderer = None
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (50,), np.float32)
        self.can_r = float(self.m.geom_size[self.ids.can_geom][0])
        mujoco.mj_kinematics(self.m, self.d)
        self.pan_xy = self.d.xanchor[self.m.joint("Rotation").id][:2].copy()

    # ------------------------------------------------------------------ helpers
    def _grip(self, a):
        return GRIP_OPEN + (a + 1.0) * 0.5 * (GRIP_CLOSED - GRIP_OPEN)

    def _obs(self):
        m, d, ids = self.m, self.d, self.ids
        tool = d.site_xpos[ids.tool]; Rt = d.site_xmat[ids.tool].reshape(3, 3)
        can = d.xpos[ids.can_body]; up = d.xmat[ids.can_body].reshape(3, 3)[:, 2]
        th, nf, _ = contacts(m, d, ids)
        return np.concatenate([
            d.qpos[ids.arm_q], 0.1 * d.qvel[ids.arm_v], d.qpos[ids.hand_q],
            tool, Rt[:, 0], Rt[:, 2],
            can, up, 0.5 * d.qvel[ids.can_v:ids.can_v + 3],
            (can - tool) * 10.0,
            [float(th), nf / 3.0, (can[2] - self.z0) * 10.0],
            self.last_a,
            [self.t / EP_STEPS], self.plan_at - self.target,
        ]).astype(np.float32)

    def base_action(self):
        """The scripted grasp as an action: descend, close, lift (rate-limited)."""
        if self.t < 35:
            goal, g, rate = self.plan_at, self.g_pre, 0.5
        elif self.t < 50:
            goal, g, rate = self.plan_at, 1.0, 0.5
        else:
            goal, g, rate = self.plan_up, 1.0, 0.25
        return np.r_[np.clip((goal - self.target) / MAX_DQ, -rate, rate), g]

    def _ik_oriented(self, tgt, fz=None, px=None):
        """Best of several IK seeds (a single seed lands in bad branches for some can
        positions). fz/px None = fingers straight down, free wrist roll."""
        m, d, ids = self.m, self.d, self.ids
        axes = None if fz is None else {2: tuple(fz), 0: tuple(px)}
        best = None
        for s in SEEDS:
            q = ik(m, d, ids, tgt, q0=s, iters=200, axes=axes)
            dd = mujoco.MjData(m); dd.qpos[:] = d.qpos; dd.qpos[ids.arm_q] = q; mujoco.mj_kinematics(m, dd)
            e = np.linalg.norm(dd.site_xpos[ids.tool] - tgt)
            if best is None or e < best[0]:
                best = (e, q)
            if e < 0.002:
                break
        return best[1]

    def _ik_from(self, tgt, fz, px, q0):
        """IK seeded from a neighbouring pose (keeps the same arm branch -- an independent
        best-of-seeds solve for the high start pose landed up to 112 deg away in Elbow)."""
        return ik(self.m, self.d, self.ids, tgt, q0=q0, iters=200, axes={2: tuple(fz), 0: tuple(px)})

    def _reset_lying(self, cx, cy, rng):
        m, d, ids = self.m, self.d, self.ids
        rad = np.array([cx, cy]) - self.pan_xy; rad /= np.linalg.norm(rad)
        yaw = np.arctan2(rad[1], rad[0]) + np.pi / 2 + rng.uniform(-L_YAW_JITTER, L_YAW_JITTER)
        qy, qz, q = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(qy, [0, 1.0, 0], np.pi / 2)        # cylinder axis -> horizontal
        mujoco.mju_axisAngle2Quat(qz, [0, 0, 1.0], yaw)
        mujoco.mju_mulQuat(q, qz, qy)
        d.qpos[ids.can_q:ids.can_q + 7] = [cx, cy, self.can_r, *q]
        fz = np.r_[rad * np.cos(L_TILT), -np.sin(L_TILT)]
        px = np.cross(fz, np.cross([0, 0, -1.0], fz)); px /= np.linalg.norm(px)
        tgt = np.array([cx, cy, self.can_r + L_DZ]) - L_BACK * np.r_[rad, 0]
        self.plan_at = self._ik_oriented(tgt, fz, px)
        self.plan_up = self._ik_from(tgt + [0, 0, L_UP], fz, px, self.plan_at)
        self.g_pre = G_PRE_L
        start = self._ik_from(tgt + np.r_[rng.uniform(-0.01, 0.01, 2), rng.uniform(0.04, 0.08)], fz, px, self.plan_at)
        return start + rng.normal(0, 0.03, 5), self.can_r

    def _reset_upright(self, cx, cy, rng):
        m, d, ids = self.m, self.d, self.ids
        d.qpos[ids.can_q:ids.can_q + 7] = [cx, cy, self.half_h, 1, 0, 0, 0]
        rad = np.array([cx, cy]) - self.pan_xy; rad /= np.linalg.norm(rad)
        fz = np.r_[rad * np.sin(U_TILT), -np.cos(U_TILT)]
        px = np.array([-rad[1], rad[0], 0.0])
        tgt = np.array([cx, cy, 2 * self.half_h + U_DZ]) + OFF * px
        self.plan_at = self._ik_oriented(tgt, fz, px)                        # scripted grasp pose
        self.plan_up = self._ik_from(tgt + [0, 0, U_UP], fz, px, self.plan_at)
        self.g_pre = G_PRE
        # Start 5-7 cm above the grasp with the open fingers already straddling the rim (as in
        # the tilt sweep). Clearance to the can is only ~2 cm a side, so start noise is small:
        # +-1.5 cm / 0.03 rad noise put fingers into the rim at reset; a start high enough to
        # clear the rim is outside the wrist range at this tilt (IK jumped to another branch).
        start = self._ik_from(tgt + np.r_[rng.uniform(-0.005, 0.005, 2), rng.uniform(0.05, 0.07)], fz, px, self.plan_at)
        return start + rng.normal(0, 0.01, 5), self.half_h

    # --------------------------------------------------------------------- api
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d, ids, rng = self.m, self.d, self.ids, self.np_random
        mujoco.mj_resetData(m, d)
        (x0, x1), (y0, y1) = self.can_range
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        self.lying = self.can_mode == "lying" or (self.can_mode == "mixed" and rng.random() < 0.5)
        q, self.z0 = (self._reset_lying if self.lying else self._reset_upright)(cx, cy, rng)
        self.top_dz = self.can_r if self.lying else self.half_h
        q = np.clip(q, self.lo, self.hi)
        g0 = rng.uniform(-0.3, 0.0)
        d.qpos[ids.arm_q] = q; d.ctrl[ids.arm_u] = q
        d.qpos[ids.hand_q] = g0; d.ctrl[ids.hand_u] = g0
        mujoco.mj_forward(m, d)
        for _ in range(50):
            mujoco.mj_step(m, d)
        self.target = q.copy()
        self.last_a = np.zeros(6)
        self.t = 0
        self.hold = 0
        self.succeeded = False
        self.can_xy0 = np.array([cx, cy])
        self.axis0 = d.xmat[ids.can_body].reshape(3, 3)[:, 2].copy()
        return self._obs(), {}

    def step(self, action):
        m, d, ids = self.m, self.d, self.ids
        pa = np.clip(np.asarray(action, float), -1, 1)
        a = np.clip(self.base_action() + RES_SCALE * pa, -1, 1) if self.residual else pa
        self.target = np.clip(self.target + MAX_DQ * a[:5], self.lo, self.hi)
        d.ctrl[ids.arm_u] = self.target
        d.ctrl[ids.hand_u] = self._grip(a[5])
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        self.t += 1

        tool = d.site_xpos[ids.tool]; can = d.xpos[ids.can_body]
        top = can + np.array([0, 0, self.top_dz])
        rise = can[2] - self.z0
        tilt = np.arccos(np.clip(abs(d.xmat[ids.can_body].reshape(3, 3)[:, 2] @ self.axis0), -1, 1))
        _, _, tab = contacts(m, d, ids)
        ts = touch_set(m, d, ids)
        th, opp = "f3" in ts, ("f0" in ts) + ("f1" in ts)
        held = th and opp >= 1              # thumb must oppose index/middle, not just the pinky

        r_reach = 1.0 - np.tanh(10.0 * np.linalg.norm(top - tool))
        r_touch = 0.5 * th + 0.25 * opp + 0.1 * ("f2" in ts)
        r_lift = 5.0 * np.clip(rise / SUCCESS_RISE, 0, 1) if held else 0.0
        r_act = 0.02 * float(np.sum(np.square(pa - self.last_a)))
        reward = r_reach + r_touch + r_lift - r_act - 0.2 * tab

        # Success does NOT end the episode: with a terminal bonus, hovering just below the
        # success height while farming the per-step lift reward paid more than finishing
        # (v2: reward up, success down). Instead every step held above it pays extra.
        up_held = held and rise > SUCCESS_RISE
        self.hold = self.hold + 1 if up_held else 0
        self.succeeded |= self.hold >= SUCCESS_HOLD
        reward += 3.0 * up_held
        fell = tilt > np.radians(45) and not held
        pushed = np.linalg.norm(can[:2] - self.can_xy0) > 0.06 and not held
        terminated = bool(fell or pushed)
        if terminated:
            reward -= 10.0
        truncated = self.t >= EP_STEPS
        self.last_a = pa
        info = {"success": self.succeeded, "rise": rise, "held": held, "tilt_deg": np.degrees(tilt),
                "lying": float(self.lying)}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 400, 400)
        self._renderer.update_scene(self.d, "side_cam")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
