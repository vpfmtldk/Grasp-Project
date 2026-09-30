"""
ik.py -- Cartesian IK for the real SO-101 (+ AmazingHand), solved against the
MuJoCo model (robot/sim/so101_amazinghand.xml) used purely as a kinematics
engine: mj_forward + the tool-site Jacobian (damped least squares), never
mj_step. Same technique already verified in mujoco_backend._ArmView and
touch_calib.FK/solve_delta -- this just packages it for robot_control.SO101.

No URDF, no ikpy, no calibration required to SOLVE: hand-eye (pixel_to_world)
builds the target point, and robot_control's home_steps/sign convert the
returned degrees to raw servo steps -- both are OTHER layers. This module only
answers "what joint angles put the tool site at this base-frame point".

    from robot.ik import ArmIK
    ik = ArmIK()                                    # loads the model once
    q_deg = ik.solve(target_xyz, q0_deg=arm.read_joints_deg(),
                     lock={3: 70 * DEG, 4: yaw * DEG})   # lock wrist pitch/roll
    arm.move_joints_deg(q_deg)

Sanity check (no hardware):
    python -m robot.ik
"""
import numpy as np

try:
    import mujoco
    _HAVE_MJ = True
except ImportError:
    mujoco = None
    _HAVE_MJ = False

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

# index order must match robot_control.Config.arm_joints (ids 1-5) AND the
# MJCF joint names -- both are the standard SO-101 build order.
ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]


class ArmIK:
    """DLS position IK for the 5 SO-101 arm joints, against a MuJoCo model
    used as a pure kinematics engine (no simulation, no dynamics)."""

    def __init__(self, mjcf="robot/sim/so101_amazinghand.xml", tool_site="tool"):
        if not _HAVE_MJ:
            raise RuntimeError("pip install mujoco")
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        jids = [self._id("JOINT", n) for n in ARM_JOINTS]
        self.qadr = [self.m.jnt_qposadr[j] for j in jids]
        self.dofadr = [self.m.jnt_dofadr[j] for j in jids]
        self.sid = self._id("SITE", tool_site)

    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.m, getattr(mujoco.mjtObj, f"mjOBJ_{kind}"), name)
        if i < 0:
            raise KeyError(f"{kind} {name!r} not in the model")
        return i

    def fk(self, q_deg):
        """Tool-site position (m) in the model's base frame for these 5 joint angles (deg)."""
        for adr, v in zip(self.qadr, np.radians(q_deg)):
            self.d.qpos[adr] = v
        mujoco.mj_forward(self.m, self.d)
        return self.d.site_xpos[self.sid].copy()

    def solve(self, target, q0_deg, lock=None, iters=300, tol=1.5e-3, step=0.7):
        """
        target: length-3 xyz, or a 4x4 pose (only the translation is used --
                the tool site has no meaningful orientation of its own).
        q0_deg: 5-vector seed (deg). ALWAYS pass the arm's current read angles
                (closer seed = fewer iterations, avoids a distant local min).
        lock:   {joint index 0-4: angle (rad)} held fixed, e.g. the wrist:
                {3: wrist_pitch_rad, 4: wrist_roll_rad}. The remaining joints
                solve for position.
        Returns: 5 joint angles (deg).
        """
        target = np.asarray(target, float)
        target = target[:3, 3] if target.shape == (4, 4) else target[:3]
        q = np.radians(np.asarray(q0_deg, float)).copy()
        lock = lock or {}
        for k, v in lock.items():
            q[k] = v
        free = [i for i in range(5) if i not in lock]
        free_dof = np.array([self.dofadr[i] for i in free])

        for _ in range(iters):
            for adr, qi in zip(self.qadr, q):
                self.d.qpos[adr] = qi
            mujoco.mj_forward(self.m, self.d)
            cur = self.d.site_xpos[self.sid]
            err = target - cur
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jacp, None, self.sid)
            J = jacp[:, free_dof]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            q[free] = np.clip(q[free] + step * dq, -np.pi, np.pi)
        return [float(v) for v in np.degrees(q)]


if __name__ == "__main__":
    # no hardware needed -- checks the solver against its own FK
    ik = ArmIK()
    q0 = [0.0, -88.0, 89.0, -93.0, 154.0]           # Config.home_deg
    p0 = ik.fk(q0)
    print("home tip (base frame, m):", np.round(p0, 4))
    for name, tgt in [("+20mm x", p0 + [0.02, 0, 0]),
                       ("-30mm z", p0 + [0, 0, -0.03]),
                       ("+15mm y", p0 + [0, 0.015, 0])]:
        q = ik.solve(tgt, q0, lock={3: q0[3] * D2R, 4: q0[4] * D2R})
        err = np.linalg.norm(ik.fk(q) - tgt) * 1000
        print(f"{name:9s} -> joints {np.round(q,1)}  reach err {err:.2f} mm")
