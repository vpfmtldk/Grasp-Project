"""
scripted_policy.py -- baseline: the scripted grasp alone (residual env, zero policy action).
The residual PPO policy starts from exactly this behaviour and learns corrections.

    python -m robot.rl.scripted_policy --episodes 50
"""
import argparse

import numpy as np

from robot.rl.can_grasp_env import CanGraspEnv


def run_episode(env, seed):
    env.reset(seed=seed)
    ret = 0.0
    while True:
        _, r, term, trunc, info = env.step(np.zeros(6))
        ret += r
        if term or trunc:
            return info.get("success", False), ret, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--mode", default="mixed", choices=["upright", "lying", "mixed"])
    a = ap.parse_args()
    env = CanGraspEnv(residual=True, can_mode=a.mode)
    res = [run_episode(env, 10_000 + s) for s in range(a.episodes)]
    for i, (ok, ret, info) in enumerate(res):
        print(f"ep {i}: {'lying ' if info['lying'] else 'upright'} success={ok} return={ret:7.1f} "
              f"rise={info['rise']*100:5.1f}cm held={info['held']}")
    for name, flag in (("upright", 0.0), ("lying", 1.0)):
        sub = [r for r in res if r[2]["lying"] == flag]
        if sub:
            print(f"scripted {name}: success {sum(r[0] for r in sub)}/{len(sub)}  mean return {np.mean([r[1] for r in sub]):.1f}")


if __name__ == "__main__":
    main()
