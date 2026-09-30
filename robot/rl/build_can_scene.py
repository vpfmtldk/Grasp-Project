"""
build_can_scene.py -- SO-101 + a CONTACT-CAPABLE AmazingHand + a can, for RL.

Why not the official AmazingHand MJCF as-is: its 2-motor parallel linkage is closed by
<connect> equalities and does not transmit grip force under mj_step (grasp_demo.py
works around that by gluing the object to the hand). RL needs real contact.

So this rebuilds the hand from the OFFICIAL model: same meshes, same place on the
wrist (build_so101_ah.py's mount), and each digit's motion measured from the real
linkage -- driven at c=0 and c=1.5 it is exactly a proximal hinge plus a distal hinge
turning ~0.9x as far (axes/pivots recovered below). Those two hinges replace the
linkage; the proximal/distal meshes are the collision shapes (convex hulls).

    python robot/rl/build_can_scene.py      -> robot/rl/so101_rlhand_can.xml
"""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(os.path.dirname(HERE), "sim")
ARM = os.path.join(SIM, "so_arm100", "so_arm100.xml")
AH = os.path.join(SIM, "ah_official", "robot.xml")
OUT = os.path.join(HERE, "so101_rlhand_can.xml")

MOUNT_POS = (0.0, -0.0025, 0.0)      # palm flush on the wrist-roll servo face (build_so101_ah.py's -0.02 left a 17.5 mm gap)
MOUNT_EULER = (1.5708, 0.0, 0.0)

# 355 ml (12 oz) beverage can: 66 mm x 122 mm, ~15 g empty (dimensions.com/element/beverage-can-12-oz);
# full ~370 g (355 ml of drink + can). --can-mass picks which.
CAN_R, CAN_HALF_H, CAN_MASS = 0.033, 0.061, 0.015
C_OPEN, C_CLOSED = 0.0, 1.5                          # official motor command range (mujoco_backend AH_CLOSE)
DIST_RATIO = 0.9                                     # measured distal/proximal angle ratio (0.85..0.96)
PROX_RANGE = (-0.83, 1.05)                           # rad, -48..60 deg (official linkage opens to -48 cleanly at c=-1)
ROOT, PROX, DIST = ("r_wrist_interface", "parallel_pin_2_x_16__da4b7ddbe9d803fe3fbc70f2e822b99b",
                    "parallel_pin_2_x_10__fee063fca0c8b40e46bbc4ffff61d999")
SUFFIX = ["", "_2", "_3", "_4"]                      # digit 4 = thumb


# ------------------------------------------------------- measure the real linkage
def _settle(m, d, acts, c):
    mujoco.mj_resetData(m, d)
    for f in range(1, 5):
        d.ctrl[acts[f"finger{f}_motor1"]] = +c
        d.ctrl[acts[f"finger{f}_motor2"]] = -c
    for _ in range(4000):
        mujoco.mj_step(m, d)


def _rel(m, d, body, root):
    R0, p0 = d.xmat[root].reshape(3, 3), d.xpos[root]
    i = m.body(body).id
    return R0.T @ d.xmat[i].reshape(3, 3), R0.T @ (d.xpos[i] - p0)


def _screw(Ra, pa, Rb, pb):
    R = Rb @ Ra.T; t = pb - R @ pa
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    pt = np.linalg.lstsq(np.eye(3) - R, t, rcond=None)[0]
    return ax, pt - ax * (ax @ pt)


def measure():
    m = mujoco.MjModel.from_xml_path(AH); d = mujoco.MjData(m)
    m.opt.gravity[:] = 0
    acts = {m.actuator(i).name: i for i in range(m.nu)}
    root = m.body(ROOT).id
    _settle(m, d, acts, C_CLOSED)
    closed = [(_rel(m, d, PROX + s, root), _rel(m, d, DIST + s, root)) for s in SUFFIX]
    tips = np.array([(d.site_xmat[m.site(f"tip{f}").id], ) and
                     d.xmat[root].reshape(3, 3).T @ (d.site_xpos[m.site(f"tip{f}").id] - d.xpos[root])
                     for f in range(1, 5)])
    _settle(m, d, acts, C_OPEN)
    digits = []
    for k, s in enumerate(SUFFIX):
        (Rp, pp), (Rd, pd) = _rel(m, d, PROX + s, root), _rel(m, d, DIST + s, root)
        (Rpc, ppc), (Rdc, pdc) = closed[k]
        axp, ptp = _screw(Rp, pp, Rpc, ppc)
        Rdl, pdl = Rp.T @ Rd, Rp.T @ (pd - pp)                       # distal in proximal frame
        axd, ptd = _screw(Rdl, pdl, Rpc.T @ Rdc, Rpc.T @ (pdc - ppc))
        digits.append(dict(
            prox=(pp, Rp, Rp.T @ axp, Rp.T @ (ptp - pp), m.body(PROX + s).id),
            dist=(pdl, Rdl, Rdl.T @ axd, Rdl.T @ (ptd - pdl), m.body(DIST + s).id)))
    return m, root, digits, tips


# ------------------------------------------------------------ build the new hand
def _quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q


def _copy_geoms(src, used, body_name, dst, collide):
    """Copy every mesh geom of official body `body_name` onto spec body `dst`. Uses the
    SPEC (uncompiled) geom poses: compiled geom_pos/quat already include MuJoCo's
    mesh re-centring, which the new model would apply a second time."""
    rgba = {mt.name: list(mt.rgba) for mt in src.materials}
    for g in _body(src, body_name).geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        kw = dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.meshname, pos=list(g.pos), quat=list(g.quat))
        dst.add_geom(**kw, rgba=rgba.get(g.material, [0.8, 0.8, 0.8, 1]), group=2,
                     contype=0, conaffinity=0, density=0)
        if collide:
            dst.add_geom(**kw, group=3, contype=2, conaffinity=0, density=1100,
                         friction=[1.5, 0.02, 0.001], rgba=[0, 0, 0, 0])
        used.add(g.meshname)


def build_hand():
    m, root, digits, tips = measure()
    src = mujoco.MjSpec.from_file(AH)
    files = {(me.name or os.path.splitext(os.path.basename(me.file))[0]):     # unnamed -> file stem
             (os.path.basename(me.file), list(me.scale)) for me in src.meshes}

    h = mujoco.MjSpec()
    h.meshdir = SIM
    h.compiler.degree = False          # ranges below are radians (a fresh MjSpec defaults to degrees)
    used = set()
    palm = h.worldbody.add_body(name="palm")
    _copy_geoms(src, used, ROOT, palm, collide=True)
    tool = tips.mean(axis=0)                       # closed-fingertip centroid = pinch centre
    palm.add_site(name="tool", pos=tool.tolist(), size=[0.005, 0, 0], rgba=[1, 0, 0, 0.8])
    for k, dg in enumerate(digits):
        pp, Rp, axp, jp, bid = dg["prox"]
        pb = palm.add_body(name=f"f{k}_prox", pos=pp.tolist(), quat=_quat(Rp).tolist())
        pb.add_joint(name=f"f{k}_j1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axp.tolist(), pos=jp.tolist(),
                     range=list(PROX_RANGE), armature=0.002, damping=0.05)
        _copy_geoms(src, used, PROX + SUFFIX[k], pb, collide=True)
        pdl, Rdl, axd, jd, bid = dg["dist"]
        db = pb.add_body(name=f"f{k}_dist", pos=pdl.tolist(), quat=_quat(Rdl).tolist())
        db.add_joint(name=f"f{k}_j2", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axd.tolist(), pos=jd.tolist(),
                     range=[PROX_RANGE[0] * DIST_RATIO, PROX_RANGE[1] * DIST_RATIO], armature=0.001, damping=0.03)
        _copy_geoms(src, used, DIST + SUFFIX[k], db, collide=True)
        db.add_site(name=f"f{k}_tip", pos=[0, 0, 0], size=[0.003, 0, 0])
        eq = h.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f"f{k}_j2", name2=f"f{k}_j1")
        eq.data[:5] = [0, DIST_RATIO, 0, 0, 0]
        act = h.add_actuator(name=f"f{k}", target=f"f{k}_j1", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                             ctrlrange=list(PROX_RANGE), ctrllimited=True,
                             forcerange=[-0.4, 0.4], forcelimited=True)
        act.set_to_position(kp=3.0)
    for name in sorted(used):
        f, sc = files[name]
        h.add_mesh(name=name, file="ah_official/assets/" + f, scale=sc)
    return h


SCENE_TMPL = """
<mujoco>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" diffuse="0.8 0.8 0.8"/>
    <body name="table" pos="0 -0.10 -0.02">
      <geom type="box" size="0.25 0.25 0.02" rgba="0.95 0.95 0.95 1" contype="1" conaffinity="3"
            friction="1 0.02 0.001"/>
    </body>
    <camera name="side_cam" pos="0.42 -0.42 0.30" xyaxes="0.707 0.707 0 -0.3 0.3 0.905" fovy="50"/>
    <body name="can" pos="0 -0.11 {h}">
      <freejoint name="can_free"/>
      <geom name="can" type="cylinder" size="{r} {h}" mass="{mass}"
            contype="1" conaffinity="3" friction="1.2 0.02 0.001" rgba="0.85 0.1 0.1 1"/>
      <site name="can_center" size="0.004"/>
    </body>
  </worldbody>
</mujoco>
"""


def _body(spec, name):
    return next(b for b in spec.bodies if b.name == name)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--can-r", type=float, default=CAN_R)
    ap.add_argument("--can-half-h", type=float, default=CAN_HALF_H)
    ap.add_argument("--can-mass", type=float, default=CAN_MASS)
    args = ap.parse_args()
    arm = mujoco.MjSpec.from_file(ARM)
    for mesh in arm.meshes:
        mesh.file = "so_arm100/assets/" + os.path.basename(mesh.file)
    arm.meshdir = SIM
    arm.delete(_body(arm, "Moving_Jaw"))
    fj = _body(arm, "Fixed_Jaw")
    for g in list(fj.geoms):
        arm.delete(g)
    for a in list(arm.actuators):          # jaw actuator went with Moving_Jaw's joint
        if a.name == "Jaw":
            arm.delete(a)

    f = fj.add_frame(pos=list(MOUNT_POS))
    q = np.zeros(4); mujoco.mju_euler2Quat(q, list(MOUNT_EULER), "xyz")
    f.quat = q.tolist()
    arm.attach(build_hand(), prefix="h_", frame=f)

    # The menagerie position servos (kp 50, +-3.5 Nm) sag ~17 deg under the hand and
    # saturate; the real STS3215s hold pose to ~1 deg. Model that holding as gravity
    # compensation on the robot bodies (the can is attached after, keeps its weight).
    for b in arm.bodies:
        if b.name != "world":
            b.gravcomp = 1.0

    scene = SCENE_TMPL.replace("{r}", str(args.can_r)).replace("{h}", str(args.can_half_h)).replace("{mass}", str(args.can_mass))
    arm.attach(mujoco.MjSpec.from_string(scene), prefix="", frame=arm.worldbody.add_frame())
    arm.visual.global_.offwidth = arm.visual.global_.offheight = 640

    arm.compile()
    xml = arm.to_xml().replace(f'meshdir="{SIM}"', 'meshdir="../sim"').replace(f'meshdir="{SIM}/"', 'meshdir="../sim/"')
    xml = xml.replace("    <default/>\n", "")       # the un-prefixed scene spec leaves an empty class
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write(xml)
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f"wrote {OUT}: nq={m.nq} nu={m.nu} neq={m.neq} ngeom={m.ngeom} "
          f"actuators={[m.actuator(i).name for i in range(m.nu)]}")


if __name__ == "__main__":
    main()
