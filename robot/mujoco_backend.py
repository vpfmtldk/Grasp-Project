"""
mujoco_backend.py -- a MuJoCo drop-in for robot_control's SO101 + AmazingHand.

Goal: the SAME code (drive.py, grasp_and_execute.py) runs in sim and on the real
robot. In sim you also get render() -> RGB to feed predict_grasp.py, and a
lift-success check.

    from robot.mujoco_backend import MujocoRobot
    sim = MujocoRobot("robot/sim/scene.xml")
    sim.connect()
    sim.arm.goto_look_pose()
    rgb = sim.render()                       # HxWx3 uint8, from the table camera
    # ... predict_grasp(rgb) -> pixel -> pixel_to_world -> IK -> joints ...
    sim.arm.move_joints_deg([j0, j1, j2, j3, j4])
    sim.hand.close_for_width(0.04)
    sim.arm.move_joints_deg(lift_pose)
    print("grasped" if sim.object_lifted("cube") else "missed")

You must supply an MJCF (robot/sim/scene.xml). See that file for where to get an
SO-ARM100 model. This module is written against the mujoco>=3.0 Python API but is
untested here (no mujoco / no MJCF in this env); expect to fix joint/camera names.
"""
from __future__ import annotations

import numpy as np

try:
    import mujoco
    _HAVE_MJ = True
except ImportError:
    mujoco = None
    _HAVE_MJ = False

# Reuse the real robot's poses / presets so sim and real share numbers.
try:
    from robot.robot_control import Config
except ImportError:
    from robot_control import Config


D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class _ArmView:
    """SO101-compatible view over a MujocoRobot."""

    def __init__(self, parent):
        self.p = parent
        self.cfg = parent.cfg

    def read_joints_deg(self):
        m, d = self.p.model, self.p.data
        out = []
        for j in self.p.arm_joint_ids:
            adr = m.jnt_qposadr[j]
            out.append(float(d.qpos[adr] * R2D))
        return out

    def move_joints_deg(self, target_deg, secs=2.0, max_step_deg=None):
        assert len(target_deg) == len(self.p.arm_joint_ids)
        start = self.read_joints_deg()
        dt = self.p.model.opt.timestep
        n = max(1, int(secs / dt))
        for k in range(1, n + 1):
            a = k / n
            for act_id, s, t in zip(self.p.arm_act_ids, start, target_deg):
                self.p.data.ctrl[act_id] = ((1 - a) * s + a * t) * D2R
            mujoco.mj_step(self.p.model, self.p.data)

    def goto_home(self, secs=3.0):
        self.move_joints_deg(self.cfg.home_deg, secs)

    def goto_look_pose(self, secs=3.0):
        self.move_joints_deg(self.cfg.look_deg, secs)

    def ee_pose_to_joints(self, T_base_ee, iters=100, tol=1e-3):
        """Damped least-squares Jacobian IK to the tool site. Returns joint deg."""
        m, d = self.p.model, self.p.data
        site = self.p.ee_site_id
        target = np.asarray(T_base_ee, float)[:3, 3]
        q = np.array(self.read_joints_deg()) * D2R
        for _ in range(iters):
            for act_id, qi in zip(self.p.arm_act_ids, q):
                d.ctrl[act_id] = qi
            mujoco.mj_forward(m, d)
            err = target - d.site_xpos[site]
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, None, site)
            J = jacp[:, self.p.arm_dof_ids]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            q = q + dq
        return list(q * R2D)


class _HandView:
    """AmazingHand-compatible view over a MujocoRobot."""

    def __init__(self, parent):
        self.p = parent
        self.cfg = parent.cfg

    def _apply(self, angles_deg: dict, settle=0.4):
        for sid, ang in angles_deg.items():
            act_id = self.p.hand_act_ids.get(sid)
            if act_id is not None:
                self.p.data.ctrl[act_id] = ang * D2R
        n = max(1, int(settle / self.p.model.opt.timestep))
        for _ in range(n):
            mujoco.mj_step(self.p.model, self.p.data)

    def set_preset(self, name, secs=0.6):
        self._apply(self.cfg.hand_presets[name], settle=secs)

    def open(self, secs=0.6):
        self.set_preset("open", secs)

    def close_for_width(self, width_m, secs=0.6):
        preset = self.cfg.width_to_preset[-1][2]
        for lo, hi, nm in self.cfg.width_to_preset:
            if lo <= width_m < hi:
                preset = nm
                break
        self.set_preset(preset, secs)


class MujocoRobot:
    def __init__(self, mjcf_path, cfg: Config = None, camera="table_cam",
                 render_size=(720, 720), ee_site="tool",
                 arm_joint_names=None, hand_actuator_prefix="finger"):
        if not _HAVE_MJ:
            raise RuntimeError("pip install mujoco")
        self.cfg = cfg or Config()
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.camera = camera
        self._renderer = None
        self._rh, self._rw = render_size

        names = arm_joint_names or [j.name for j in self.cfg.arm_joints]
        self.arm_joint_ids = [self._jid(n) for n in names]
        self.arm_act_ids = [self._aid(n) for n in names]
        self.arm_dof_ids = [self.model.jnt_dofadr[j] for j in self.arm_joint_ids]

        # hand: map config servo id -> an actuator named e.g. "finger1".."finger8"
        self.hand_act_ids = {}
        for sid in self.cfg.hand_servo_ids:
            try:
                self.hand_act_ids[sid] = self._aid(f"{hand_actuator_prefix}{sid}")
            except Exception:
                pass

        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        except Exception:
            self.ee_site_id = -1

        self.arm = _ArmView(self)
        self.hand = _HandView(self)

    # -------- name -> id helpers ------------------------------------------
    def _jid(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise KeyError(f"joint '{name}' not in MJCF")
        return i

    def _aid(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise KeyError(f"actuator '{name}' not in MJCF")
        return i

    # -------- lifecycle ------------------------------------------------------
    def connect(self):
        mujoco.mj_forward(self.model, self.data)
        return self

    def disconnect(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(self, keyframe=0):
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey > keyframe:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # -------- sensing ------------------------------------------------------
    def render(self):
        """RGB (H,W,3) uint8 from the table camera -- feed straight to predict_grasp."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._rh, self._rw)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def body_pose(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        T = np.eye(4)
        T[:3, 3] = self.data.xpos[bid]
        T[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        return T

    def object_lifted(self, body, min_rise=0.03):
        """True if `body` is >= min_rise above its rest height (call after settling)."""
        z = float(self.body_pose(body)[2, 3])
        z0 = getattr(self, f"_z0_{body}", None)
        if z0 is None:
            return False
        return (z - z0) >= min_rise

    def mark_rest(self, body):
        setattr(self, f"_z0_{body}", float(self.body_pose(body)[2, 3]))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MuJoCo backend smoke test")
    p.add_argument("--mjcf", default="robot/sim/scene.xml")
    p.add_argument("--out", default="output/sim_view.png")
    a = p.parse_args()
    if not _HAVE_MJ:
        raise SystemExit("pip install mujoco")
    sim = MujocoRobot(a.mjcf).connect()
    print("joints (deg):", [round(v, 1) for v in sim.arm.read_joints_deg()])
    sim.arm.goto_look_pose()
    sim.hand.open()
    rgb = sim.render()
    try:
        from imageio.v2 import imwrite
        imwrite(a.out, rgb)
        print("wrote", a.out, rgb.shape)
    except Exception as e:
        print("render ok", rgb.shape, "(save failed:", e, ")")
