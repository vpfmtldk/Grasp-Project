"""
view_hand.py -- interactive MuJoCo viewer for the RL scene (SO-101 + contact AmazingHand + can).

    python robot/rl/view_hand.py

Mouse: left-drag rotate, right-drag pan, scroll zoom, double-click a body to select.
Right panel "Control": sliders drive the arm joints and each finger (h_f3 = thumb).
Left panel "Rendering" > Geom groups: 2 = visual meshes, 3 = collision shapes.
"""
import os
import sys

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robot.rl.sim_util import SCENE, Ids   # noqa: E402

POSE = [0.0, -1.9, 0.9, 0.6, 0.0]          # hand up in the air, can in front


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    ids = Ids(m)
    d.qpos[ids.arm_q] = POSE
    d.ctrl[ids.arm_u] = POSE
    mujoco.mj_forward(m, d)
    mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
