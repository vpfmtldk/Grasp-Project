"""
eval_policy.py -- run the trained PPO policy: success rate, live viewer, or GIF.

    python -m robot.rl.eval_policy --episodes 50              # headless success rate
    python -m robot.rl.eval_policy --view                     # MuJoCo viewer, loops
    python -m robot.rl.eval_policy --gif output/rl/ppo_can.gif --episodes 4
"""
import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot.rl.can_grasp_env import CanGraspEnv, FRAME_SKIP

OUT = os.path.join("output", "rl", "ppo_can")
if not os.path.exists(os.path.join(OUT, "model.zip")):          # fresh clone: use the shipped policy
    OUT = os.path.join("pretrained", "ppo_can")


def load(model_dir, mode="mixed"):
    venv = DummyVecEnv([lambda: CanGraspEnv(can_mode=mode)])
    venv = VecNormalize.load(os.path.join(model_dir, "vecnormalize.pkl"), venv)
    venv.training = False; venv.norm_reward = False
    return PPO.load(os.path.join(model_dir, "model"), device="cpu"), venv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=OUT)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--gif")
    ap.add_argument("--mode", default="mixed", choices=["upright", "lying", "mixed"])
    ap.add_argument("--scripted", action="store_true", help="no model: zero residual = scripted grasp")
    a = ap.parse_args()
    if a.scripted:
        venv = DummyVecEnv([lambda: CanGraspEnv(can_mode=a.mode)]); model = None
    else:
        model, venv = load(a.model, a.mode)
    env = venv.envs[0]

    viewer, frames = None, []
    if a.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.m, env.d)
        viewer.cam.lookat[:] = [0, -0.05, 0.08]; viewer.cam.distance = 0.6
        viewer.cam.azimuth = 135; viewer.cam.elevation = -25
    ok, n, per = 0, 0, {0.0: [0, 0], 1.0: [0, 0]}
    obs = venv.reset()
    while (a.view and viewer.is_running()) or n < a.episodes:
        act = np.zeros((1, 6)) if model is None else model.predict(obs, deterministic=True)[0]
        t0 = time.perf_counter()
        obs, r, done, info = venv.step(act)
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, FRAME_SKIP * env.m.opt.timestep - (time.perf_counter() - t0)))
        if a.gif:
            frames.append(env.render())
        if done[0]:
            n += 1; s_ = bool(info[0].get("success")); ok += s_
            per[info[0]["lying"]][0] += s_; per[info[0]["lying"]][1] += 1
            print(f"episode {n}: {'lying ' if info[0]['lying'] else 'upright'} success={s_} "
                  f"rise={info[0]['rise']*100:+.1f}cm   [{ok}/{n}]", flush=True)
            if viewer is not None:
                time.sleep(0.5)
    print(f"success {ok}/{n} = {ok / max(n, 1):.0%}   upright {per[0.0][0]}/{per[0.0][1]}"
          f"   lying {per[1.0][0]}/{per[1.0][1]}")
    if a.gif:
        from imageio.v2 import mimsave
        mimsave(a.gif, frames[::2], duration=0.1)
        print("->", a.gif)


if __name__ == "__main__":
    main()
