"""
build_so101_ah.py -- merge the menagerie SO-ARM100 with the Pollen AmazingHand
into one MJCF, replacing the stock parallel jaw with the hand.

    python robot/sim/build_so101_ah.py
    -> writes robot/sim/so101_amazinghand.xml

Tune MOUNT_POS / MOUNT_EULER until the hand sits on the wrist correctly
(check with:  python -m robot.mujoco_backend --mjcf robot/sim/scene_ah.xml).
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.path.join(HERE, "so_arm100", "so_arm100.xml")
HAND = os.path.join(HERE, "amazing_hand", "robot.xml")
OUT = os.path.join(HERE, "so101_amazinghand.xml")

# where the hand's r_wrist_interface frame sits, relative to the Fixed_Jaw link
# (which carries the Wrist_Roll joint -- keep it). Tune these.
MOUNT_POS = (0.0, 0.0, -0.10)        # bring the hand body back onto the wrist face
MOUNT_EULER = (1.5708, 0.0, 3.14159) # palm faces the forearm, fingers forward/down


def _body(spec, name):
    return next(b for b in spec.bodies if b.name == name)


def main():
    arm = mujoco.MjSpec.from_file(ARM)
    hand = mujoco.MjSpec.from_file(HAND)

    # bake each model's mesh files to absolute paths, then clear meshdir, so the
    # merged file (one meshdir) still finds meshes from both source trees.
    # meshdir = robot/sim (absolute, so compile() finds the STLs), and each mesh
    # path becomes "<sub>/assets/<name>". The absolute meshdir is swapped for ""
    # in the written XML so the merged file stays portable (it lives in robot/sim).
    for spec, sub in ((arm, "so_arm100"), (hand, "amazing_hand")):
        for mesh in spec.meshes:
            mesh.file = f"{sub}/assets/" + os.path.basename(mesh.file)
        spec.meshdir = HERE

    # remove the stock jaw but KEEP Fixed_Jaw (it carries the Wrist_Roll joint).
    arm.delete(_body(arm, "Moving_Jaw"))          # drops Jaw joint + Jaw actuator too
    fj = _body(arm, "Fixed_Jaw")
    for g in list(fj.geoms):                      # strip the stock gripper hardware
        arm.delete(g)

    # attach the hand on the wrist-roll link
    f = fj.add_frame(pos=list(MOUNT_POS))
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, list(MOUNT_EULER), "xyz")
    f.quat = q.tolist()
    arm.attach(hand, prefix="ah_", frame=f)

    arm.compile()                       # validates the merged model
    xml = arm.to_xml()
    xml = xml.replace(f'meshdir="{HERE}"', 'meshdir=""')      # make it portable
    xml = xml.replace(f'meshdir="{HERE}/"', 'meshdir=""')
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write(xml)
    print("wrote", OUT)

    m = mujoco.MjModel.from_xml_path(OUT)
    print("merged: nq=%d nu=%d nbody=%d" % (m.nq, m.nu, m.nbody))
    print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])


if __name__ == "__main__":
    main()
