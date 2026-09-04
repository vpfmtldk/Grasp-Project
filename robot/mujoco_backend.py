"""
mujoco_backend.py -- a MuJoCo drop-in for robot_control's SO101 + AmazingHand.

Goal: the SAME pipeline code runs in sim and on the real robot. In sim you also
get render() -> RGB to feed predict_grasp.py, and a lift-success check.

    from robot.mujoco_backend import MujocoRobot
    sim = MujocoRobot("robot/sim/scene.xml").connect()
    sim.reset()
    sim.arm.goto_look_pose()
    rgb = sim.render()                          # HxWx3 uint8, table camera
    # predict_grasp(rgb) -> pixel -> pixel_to_world -> IK -> joints
    sim.arm.move_joints_deg([j0,j1,j2,j3,j4])
    sim.hand.close()                            # parallel jaw (menagerie model)
    sim.arm.move_joints_deg(lift_pose)
    print("grasped" if sim.object_lifted("cube") else "missed")

Scene: robot/sim/scene.xml (menagerie SO-ARM100 + table + table_cam + cube).
Arm joints are the menagerie names; the "hand" is the stock parallel jaw
("Jaw" actuator) unless the model has finger1..8 (AmazingHand).
"""
from __future__ import annotations

import numpy as np

try:
    import mujoco
    _HAVE_MJ = True
except ImportError:
    mujoco = None
    _HAVE_MJ = False

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

# menagerie SO-ARM100 joint order == our shoulder_pan..wrist_roll
ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
HOME_RAD = [0, -1.57, 1.57, 1.57, -1.57]
LOOK_RAD = [0, -2.9, 3.0, 0.9, -1.57]   # arm folded up, out of table_cam view


class _ArmView:
    def __init__(self, p):
        self.p = p

    def read_joints_deg(self):
        m, d = self.p.model, self.p.data
        return [float(d.qpos[m.jnt_qposadr[j]] * R2D) for j in self.p.arm_jids]

    def move_joints_deg(self, target_deg, secs=2.0, max_step_deg=None):
        assert len(target_deg) == len(self.p.arm_aids)
        start = self.read_joints_deg()
        n = max(1, int(secs / self.p.model.opt.timestep))
        for k in range(1, n + 1):
            a = k / n
            for aid, s, t in zip(self.p.arm_aids, start, target_deg):
                self.p.data.ctrl[aid] = ((1 - a) * s + a * t) * D2R
            mujoco.mj_step(self.p.model, self.p.data)

    def hold(self, secs=0.5):
        for _ in range(max(1, int(secs / self.p.model.opt.timestep))):
            mujoco.mj_step(self.p.model, self.p.data)

    def goto_home(self, secs=2.5):
        self.move_joints_deg([v * R2D for v in HOME_RAD], secs)

    def goto_look_pose(self, secs=2.5):
        self.move_joints_deg([v * R2D for v in LOOK_RAD], secs)

    def ee_pose_to_joints(self, T_base_ee, iters=200, tol=2e-3):
        """Damped least-squares Jacobian IK to the tool site/body. Returns joint deg."""
        m, d = self.p.model, self.p.data
        target = np.asarray(T_base_ee, float)[:3, 3]
        q = np.array(self.read_joints_deg()) * D2R
        dof = np.array(self.p.arm_dofs)
        for _ in range(iters):
            for aid, qi in zip(self.p.arm_aids, q):
                d.ctrl[aid] = qi
            mujoco.mj_forward(m, d)
            cur = d.site_xpos[self.p.ee_site] if self.p.ee_site >= 0 else d.xpos[self.p.ee_body]
            err = target - cur
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, m.nv))
            if self.p.ee_site >= 0:
                mujoco.mj_jacSite(m, d, jacp, None, self.p.ee_site)
            else:
                mujoco.mj_jacBody(m, d, jacp, None, self.p.ee_body)
            J = jacp[:, dof]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            q = np.clip(q + dq, -np.pi, np.pi)
        return list(q * R2D)


class _HandView:
    """Parallel jaw (menagerie 'Jaw') or AmazingHand (finger1..8)."""

    def __init__(self, p):
        self.p = p
        self.jaw = p.aid_or_none("Jaw")
        # AmazingHand (merged model): 4 fingers x 2 motors, prefixed "ah_"
        self.ah = [(p.aid_or_none(f"ah_finger{f}_motor1"), p.aid_or_none(f"ah_finger{f}_motor2"))
                   for f in range(1, 5)]
        if self.jaw is not None:
            self.mode = "jaw"
        elif self.ah[0][0] is not None:
            self.mode = "ah"
        else:
            self.mode = "none"

    def _settle(self, secs):
        for _ in range(max(1, int(secs / self.p.model.opt.timestep))):
            mujoco.mj_step(self.p.model, self.p.data)

    def set_opening(self, frac, secs=0.6):
        """frac 0 = closed/flexed .. 1 = open."""
        frac = float(np.clip(frac, 0.0, 1.0))
        if self.mode == "jaw":
            lo, hi = self.p.model.actuator_ctrlrange[self.jaw]
            self.p.data.ctrl[self.jaw] = lo + frac * (hi - lo)
        elif self.mode == "ah":
            # differential: the two motors of a finger go OPPOSITE ways to flex
            flex = (1.0 - frac) * 1.2                     # rad
            for m1, m2 in self.ah:
                if m1 is not None:
                    self.p.data.ctrl[m1] = +flex
                if m2 is not None:
                    self.p.data.ctrl[m2] = -flex
        self._settle(secs)

    def open(self, secs=0.6):
        self.set_opening(1.0, secs)

    def close(self, secs=0.8):
        self.set_opening(0.0, secs)

    def close_for_width(self, width_m, secs=0.8):
        # map 0..0.08 m opening to jaw fraction; tune to the real jaw travel
        self.set_opening(float(np.clip(width_m / 0.08, 0.0, 1.0)) * 0.6, secs)


class MujocoRobot:
    def __init__(self, mjcf_path="robot/sim/scene.xml", camera="table_cam",
                 render_size=(720, 720), ee_body="Fixed_Jaw", ee_site="tool"):
        if not _HAVE_MJ:
            raise RuntimeError("pip install mujoco")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.camera = camera
        self._rh, self._rw = render_size
        self._renderer = None

        self.arm_jids = [self._id("JOINT", n) for n in ARM_JOINTS]
        self.arm_aids = [self._id("ACTUATOR", n) for n in ARM_JOINTS]
        self.arm_dofs = [self.model.jnt_dofadr[j] for j in self.arm_jids]
        self.ee_site = self._id_or(-1, "SITE", ee_site)
        self.ee_body = self._id("BODY", ee_body)

        self.arm = _ArmView(self)
        self.hand = _HandView(self)
        self._rest_z = {}

    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.model, getattr(mujoco.mjtObj, f"mjOBJ_{kind}"), name)
        if i < 0:
            raise KeyError(f"{kind} '{name}' not in the model")
        return i

    def _id_or(self, default, kind, name):
        i = mujoco.mj_name2id(self.model, getattr(mujoco.mjtObj, f"mjOBJ_{kind}"), name)
        return i if i >= 0 else default

    def aid_or_none(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return i if i >= 0 else None

    # -- lifecycle --
    def connect(self):
        mujoco.mj_forward(self.model, self.data)
        return self

    def disconnect(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(self, pose_rad=None):
        mujoco.mj_resetData(self.model, self.data)
        pose = HOME_RAD if pose_rad is None else pose_rad
        for j, a, q in zip(self.arm_jids, self.arm_aids, pose):
            self.data.qpos[self.model.jnt_qposadr[j]] = q
            self.data.ctrl[a] = q
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # -- sensing --
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._rh, self._rw)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def body_xyz(self, name):
        return np.array(self.data.xpos[self._id("BODY", name)])

    def mark_rest(self, body):
        self._rest_z[body] = float(self.body_xyz(body)[2])

    def object_lifted(self, body, min_rise=0.03):
        z0 = self._rest_z.get(body)
        return z0 is not None and (float(self.body_xyz(body)[2]) - z0) >= min_rise


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MuJoCo backend smoke test")
    p.add_argument("--mjcf", default="robot/sim/scene.xml")
    p.add_argument("--outdir", default="output")
    a = p.parse_args()
    from imageio.v2 import imwrite

    sim = MujocoRobot(a.mjcf).connect()
    sim.reset()
    print("hand mode:", sim.hand.mode)
    print("home joints (deg):", [round(v, 1) for v in sim.arm.read_joints_deg()])
    imwrite(f"{a.outdir}/sim_00_home.png", sim.render())

    sim.arm.goto_look_pose()
    print("look joints (deg):", [round(v, 1) for v in sim.arm.read_joints_deg()])
    imwrite(f"{a.outdir}/sim_01_look.png", sim.render())

    sim.mark_rest("cube")
    sim.hand.open(); sim.hand.close()
    sim.arm.hold(0.5)
    print("cube lifted after blind close:", sim.object_lifted("cube"))
    imwrite(f"{a.outdir}/sim_02_close.png", sim.render())
    print("wrote", a.outdir, "/sim_*.png")
    sim.disconnect()
