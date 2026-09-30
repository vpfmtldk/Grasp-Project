"""
train_ppo.py -- PPO (stable-baselines3) on CanGraspEnv, parallel envs.

    python -m robot.rl.train_ppo --steps 5000000          -> output/rl/ppo_can/
    python -m robot.rl.train_ppo --resume                  # continue from the last save

Writes model.zip + vecnormalize.pkl (+ checkpoints/, progress.csv). Success rate is the
fraction of recent episodes that held the can >= 6 cm for 0.5 s (column rollout/success).
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from robot.rl.can_grasp_env import CanGraspEnv

OUT = os.path.join("output", "rl", "ppo_can")


class SuccessLog(BaseCallback):
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf:
            self.logger.record("rollout/success", float(np.mean([e.get("success", 0) for e in buf])))
            for name, flag in (("upright", 0.0), ("lying", 1.0)):
                sub = [e.get("success", 0) for e in buf if e.get("lying", 0.0) == flag]
                if sub:
                    self.logger.record(f"rollout/success_{name}", float(np.mean(sub)))


class SaveNorm(BaseCallback):
    def __init__(self, every):
        super().__init__(); self.every = every

    def _on_step(self):
        if self.n_calls % self.every == 0:
            self.model.save(os.path.join(OUT, "model"))
            self.training_env.save(os.path.join(OUT, "vecnormalize.pkl"))
        return True


def make_env(rank, mode):
    def f():
        env = CanGraspEnv(can_mode=mode)
        env.reset(seed=1000 + rank)
        return env
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--envs", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mode", default="mixed", choices=["upright", "lying", "mixed"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    venv = VecMonitor(SubprocVecEnv([make_env(i, a.mode) for i in range(a.envs)]),
                      info_keywords=("success", "lying"))
    if a.resume:
        venv = VecNormalize.load(os.path.join(OUT, "vecnormalize.pkl"), venv)
        model = PPO.load(os.path.join(OUT, "model"), env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=1792, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0), device="cpu", seed=0)
    model.set_logger(configure(OUT, ["stdout", "csv"]))
    per = max(1, 200_000 // a.envs)
    model.learn(a.steps, callback=[SuccessLog(), SaveNorm(per),
                                   CheckpointCallback(per * 5, os.path.join(OUT, "checkpoints"), "ppo")],
                reset_num_timesteps=not a.resume)
    model.save(os.path.join(OUT, "model"))
    venv.save(os.path.join(OUT, "vecnormalize.pkl"))
    venv.close()
    print("saved", OUT)


if __name__ == "__main__":
    main()
