"""
view.py -- look at a MuJoCo scene without the interactive viewer.

    python robot/sim/view.py                       # scene_ah.xml, default angle
    python robot/sim/view.py --mjcf robot/sim/scene.xml --az 90 --el -30 --dist 1.2
    python robot/sim/view.py --cam table_cam       # use a named camera in the model
    python robot/sim/view.py --pose 0 -1.0 1.3 1.0 0   # set the 5 arm joints (rad)
    python robot/sim/view.py --try-viewer          # attempt the GL viewer, fall back to a PNG
"""
import argparse
import mujoco
from imageio.v2 import imwrite

ARM = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", default="robot/sim/scene_ah.xml")
    p.add_argument("--out", default="output/view.png")
    p.add_argument("--cam", default="", help="named camera; blank = free camera")
    p.add_argument("--az", type=float, default=135)
    p.add_argument("--el", type=float, default=-20)
    p.add_argument("--dist", type=float, default=0.9)
    p.add_argument("--lookat", type=float, nargs=3, default=[0, -0.05, 0.08])
    p.add_argument("--pose", type=float, nargs=5, default=None, help="arm joint targets (rad)")
    p.add_argument("--res", type=int, default=700)
    p.add_argument("--try-viewer", action="store_true")
    a = p.parse_args()

    m = mujoco.MjModel.from_xml_path(a.mjcf)
    d = mujoco.MjData(m)
    if a.pose is not None:
        for n, q in zip(ARM, a.pose):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            d.qpos[m.jnt_qposadr[jid]] = q
            if aid >= 0:
                d.ctrl[aid] = q
    mujoco.mj_forward(m, d)

    if a.try_viewer:
        try:
            import mujoco.viewer as _mjv
            print("opening GL viewer -- close the window to exit")
            _mjv.launch(m, d)
            return
        except Exception as e:
            print("viewer failed (%s); rendering a PNG instead" % e)

    r = mujoco.Renderer(m, a.res, a.res)
    if a.cam:
        r.update_scene(d, camera=a.cam)
    else:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = a.lookat
        cam.azimuth, cam.elevation, cam.distance = a.az, a.el, a.dist
        r.update_scene(d, cam)
    imwrite(a.out, r.render())
    r.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
