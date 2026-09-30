# isaac_run.py -- run the MuJoCo "so_arm100 + Amazing Hand" model inside Isaac Sim.
#
# Usage (from an Isaac Sim python env, e.g. C:\isaacsim\python.bat):
#   python isaac_run.py --usd path\to\so101_amazinghand.usd
#
# The USD is produced beforehand by isaac_import.py (MJCF -> USD conversion).
#
# NOTE: the source MJCF drives the AmazingHand fingers through closed-loop
# four-bar linkages (<equality><connect .../></equality> between sites, plus
# several ball joints). Isaac Sim's MJCF importer builds a tree-structured USD
# articulation and does NOT recreate those closing constraints (PhysX
# articulations don't have an MJCF-equality equivalent that the importer wires
# up automatically). So here you'll see: the arm move correctly, and the
# finger drive joints (ah_fingerX_motorY) turn -- but the passive linkage
# members that used to be held together by the dropped equality constraints
# may drift/interpenetrate instead of tracing the real closed mechanism.

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True, help="Path to the imported robot USD/USDA file")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.timeline
import omni.usd
from pxr import UsdPhysics
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import Articulation

D2R = np.pi / 180.0

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
HOME_RAD = [0, -1.9, 2.2, 1.3, 0]
LOOK_RAD = [0, -2.9, 2.0, 1.6, 0]
AH_CLOSE = 0.6  # mujoco_backend.py uses 1.5, but a full-amplitude swing seems
                # to push this converted closed-loop mechanism into a physically
                # unreachable/singular pose that invalidates the PhysX tensor

ROBOT_PATH = "/World/robot"


def main():
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise SystemExit(f"USD not found: {usd_path}. Run isaac_import.py first.")

    stage_utils.create_new_stage(template="sunlight")
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=ROBOT_PATH)
    simulation_app.update()

    robot = Articulation(ROBOT_PATH)
    simulation_app.update()  # let the articulation initialize (physics tensor entity)

    print("DOF names:", robot.dof_names)

    arm_idx = robot.get_dof_indices(ARM_JOINTS)
    ah_names = [n for n in robot.dof_names if n.startswith("ah_finger")]
    ah_names.sort()
    ah_idx = robot.get_dof_indices(ah_names) if ah_names else None
    print("Hand drive joints found:", ah_names)

    # The MJCF importer logged "Gain and bias prm arrays are not in the expected
    # format" for every actuator in this model, so none of the imported joints
    # got a PD stiffness/damping -- position targets would otherwise be no-ops.
    # Articulation.set_dof_gains() is unusable here: on first call it caches
    # "default gains" by internally calling get_dof_gains() with NO dof filter,
    # which walks every DOF in the articulation -- including the hand's passive
    # ball joints (DofType.Invalid for a single scalar drive) -- and throws.
    # So set stiffness/damping directly via USD instead of that wrapper.
    # The arm has no closed loops -- 50/1 (~MJCF's kp=50) is fine there. The
    # hand's actuated joints fight the loop-closing "Equality" spherical
    # joints (added by the importer for the MJCF <equality><connect> four-bar
    # linkages); driving them hard/fast made PhysX's solver blow up ("Invalid
    # PhysX transform detected" -> hard crash) on the first attempt. Softer
    # gains + slow ramps (see goto_arm/set_hand step counts below) avoid that.
    stage = omni.usd.get_context().get_stage()
    dof_paths = robot.dof_paths[0]
    all_actuated = ARM_JOINTS + ah_names
    gains = {name: (50.0, 1.0) for name in ARM_JOINTS}
    gains.update({name: (6.0, 0.3) for name in ah_names})
    for name in all_actuated:
        path = dof_paths[robot.dof_names.index(name)]
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        if not prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive = UsdPhysics.DriveAPI(prim, "angular")
        stiffness, damping = gains[name]
        drive.CreateStiffnessAttr().Set(stiffness)
        drive.CreateDampingAttr().Set(damping)

    # Applying new API schemas onto the prims (above) invalidates this
    # wrapper's cached physics-tensor entity; rebuild it so get/set_dof_*
    # calls don't assert "physics tensor entity is not valid" afterwards.
    robot = Articulation(ROBOT_PATH)
    simulation_app.update()
    arm_idx = robot.get_dof_indices(ARM_JOINTS)
    ah_idx = robot.get_dof_indices(ah_names) if ah_names else None
    all_idx = robot.get_dof_indices(all_actuated)

    omni.timeline.get_timeline_interface().play()
    for _ in range(30):  # let the freshly-spawned mechanism settle before driving it
        simulation_app.update()

    def goto_arm(target_rad, steps=120):
        start = robot.get_dof_positions(dof_indices=arm_idx).numpy().reshape(-1)
        for k in range(1, steps + 1):
            a = k / steps
            pose = (1 - a) * start + a * np.asarray(target_rad)
            robot.set_dof_position_targets(pose.reshape(1, -1), dof_indices=arm_idx)
            simulation_app.update()

    def set_hand(open_frac, steps=60):
        # 1 = open .. 0 = flexed (fingers 1-3: motor1=+c motor2=-c ; thumb: motor1=+c motor2=-c too, see mujoco_backend)
        if ah_idx is None:
            return
        c = (1.0 - open_frac) * AH_CLOSE
        targets = []
        for n in ah_names:
            sign = +1.0 if n.endswith("motor1") else -1.0
            targets.append(sign * c)
        targets = np.asarray(targets)
        cur = robot.get_dof_positions(dof_indices=ah_idx).numpy().reshape(-1)
        for k in range(1, steps + 1):
            a = k / steps
            pose = (1 - a) * cur + a * targets
            robot.set_dof_position_targets(pose.reshape(1, -1), dof_indices=ah_idx)
            simulation_app.update()

    print("-> homing arm")
    goto_arm(HOME_RAD)
    print("-> opening hand")
    set_hand(1.0, steps=240)
    print("-> look pose")
    goto_arm(LOOK_RAD)

    print("-> looping open/close")
    try:
        while simulation_app.is_running():
            try:
                set_hand(0.0, steps=240)  # slow close
                for _ in range(60):
                    simulation_app.update()
                set_hand(1.0, steps=240)  # slow open
                for _ in range(60):
                    simulation_app.update()
            except AssertionError:
                # The closed-loop hand mechanism can hit a physically
                # unreachable/singular pose and invalidate PhysX's tensor
                # view; re-play to recover instead of taking the whole app down.
                print("-> physics tensor invalidated, re-playing to recover")
                omni.timeline.get_timeline_interface().play()
                for _ in range(30):
                    simulation_app.update()
    except KeyboardInterrupt:
        pass

    simulation_app.close()


if __name__ == "__main__":
    main()
