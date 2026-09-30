"""
build_so101_ah.py -- merge the menagerie SO-ARM100 with the OFFICIAL Pollen
AmazingHand MJCF (robot/sim/ah_official/robot.xml) onto the SO-101 wrist.

The AmazingHand keeps its real CAD meshes and real 2-motor parallel linkage
(closed by 20 <connect> equalities). It is driven KINEMATICALLY the way Pollen's
own demo does it (mink IK / mj_forward, no contact dynamics): the 8 finger
motors are position-servoed to a flex angle and the linkage resolves through the
equality constraints. A `tool` site marks the finger/thumb pinch centre for arm
IK.

    python robot/sim/build_so101_ah.py   -> robot/sim/so101_amazinghand.xml
"""
import os, re
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ARM  = os.path.join(HERE, "so_arm100", "so_arm100.xml")
HAND = os.path.join(HERE, "ah_official", "robot.xml")
OUT  = os.path.join(HERE, "so101_amazinghand.xml")

MOUNT_POS   = (0.0, -0.02, 0.0)
MOUNT_EULER = (1.5708, 0.0, 0.0)          # fingers point down at the table
TOOL_HR     = (0.045, 0.0, 0.082)         # pinch centre = centroid of the 4 fully-closed fingertips
                                          # (FK-measured on the built model), hand-root frame


def _body(spec, name):
    return next(b for b in spec.bodies if b.name == name)


def _post(xml):
    tx, ty, tz = TOOL_HR
    site = f'    <site name="tool" pos="{tx} {ty} {tz}" size="0.004" rgba="1 0 0 0.7"/>\n'
    xml = re.sub(r'(<body name="ah_r_wrist_interface"[^>]*>\n)', r'\1' + site, xml, count=1)
    return xml


def main():
    arm  = mujoco.MjSpec.from_file(ARM)
    hand = mujoco.MjSpec.from_file(HAND)
    for spec, sub in ((arm, "so_arm100"), (hand, "ah_official")):
        for mesh in spec.meshes:
            mesh.file = f"{sub}/assets/" + os.path.basename(mesh.file)
        spec.meshdir = HERE

    arm.delete(_body(arm, "Moving_Jaw"))
    fj = _body(arm, "Fixed_Jaw")
    for g in list(fj.geoms):
        arm.delete(g)

    f = fj.add_frame(pos=list(MOUNT_POS))
    q = np.zeros(4); mujoco.mju_euler2Quat(q, list(MOUNT_EULER), "xyz")
    f.quat = q.tolist()
    arm.attach(hand, prefix="ah_", frame=f)

    arm.compile()
    xml = arm.to_xml()
    xml = xml.replace(f'meshdir="{HERE}"', 'meshdir=""').replace(f'meshdir="{HERE}/"', 'meshdir=""')
    xml = _post(xml)
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write(xml)
    print("wrote", OUT)
    m = mujoco.MjModel.from_xml_path(OUT)
    print("merged: nq=%d nu=%d neq=%d nbody=%d" % (m.nq, m.nu, m.neq, m.nbody))


if __name__ == "__main__":
    main()
