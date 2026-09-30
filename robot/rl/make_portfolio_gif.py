"""
make_portfolio_gif.py -- portfolio GIF of the trained residual-PPO policy: an upright
355 ml can, then a lying one, with a slowly orbiting camera and Korean captions.

    python -m robot.rl.make_portfolio_gif        -> docs/figures/rl_portfolio.gif

Only successful episodes are used (seeds are searched); the clip stops ~1 s after the
success condition (>= 6 cm for 0.5 s) is met.
"""
import argparse
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot.rl.eval_policy import load

OUT = os.path.join("docs", "figures", "rl_portfolio.gif")
FONT = r"C:\Windows\Fonts\malgunbd.ttf"
FONT_R = r"C:\Windows\Fonts\malgun.ttf"
W, H = 560, 420


def run(model, venv, mode, seed, cam, R, tail=20, orbit=(115, 150)):
    env = venv.envs[0]
    env.can_mode = mode
    venv.seed(seed)
    obs = venv.reset()
    frames, succ_at, t = [], None, 0
    while True:
        act = model.predict(obs, deterministic=True)[0]
        obs, _, done, info = venv.step(act)
        if done[0]:
            break
        can = env.d.xpos[env.ids.can_body]
        cam.lookat[:] = 0.7 * cam.lookat + 0.3 * (can + [0, 0, 0.03])     # follow the can, smoothed
        cam.azimuth = orbit[0] + (orbit[1] - orbit[0]) * min(1.0, t / 120)
        R.update_scene(env.d, cam)
        frames.append((R.render().copy(), info[0]))
        t += 1
        if succ_at is None and info[0]["success"]:
            succ_at = t
        if succ_at is not None and t >= succ_at + tail:
            break
    return frames, succ_at is not None


def caption(img, title, sub, badge, done):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    f1, f2, f3 = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT_R, 15), ImageFont.truetype(FONT, 17)
    dr.rectangle([0, 0, W, 64], fill=(0, 0, 0, 150))
    dr.text((14, 8), title, font=f1, fill=(255, 255, 255))
    dr.text((14, 38), sub, font=f2, fill=(210, 210, 210))
    if badge:
        tw = dr.textlength(badge, font=f3)
        dr.rounded_rectangle([W - tw - 30, H - 42, W - 12, H - 12], 8,
                             fill=(30, 160, 80, 220) if done else (70, 70, 70, 200))
        dr.text((W - tw - 21, H - 39), badge, font=f3, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--every", type=int, default=2, help="keep every N-th control step (20 Hz / N fps)")
    a = ap.parse_args()
    model, venv = load(os.path.join("output", "rl", "ppo_can"), "upright")
    env = venv.envs[0]
    R = mujoco.Renderer(env.m, H, W)
    cam = mujoco.MjvCamera(); cam.distance = 0.62; cam.elevation = -22; cam.lookat[:] = [0, -0.11, 0.08]

    clips = []
    for mode, title, rate in (("upright", "서 있는 355 ml 캔", "성공률 79% (스크립트만: 42%)"),
                              ("lying", "누운 355 ml 캔", "성공률 100%")):
        for seed in range(100, 140):
            cam.lookat[:] = [0, -0.11, 0.08]
            frames, ok = run(model, venv, mode, seed, cam, R)
            if ok:
                print(f"{mode}: seed {seed}, {len(frames)} steps")
                break
        sub = "SO-101 + AmazingHand · MuJoCo · 잔차 PPO · 손가락 마찰만으로 파지"
        for i, (img, info) in enumerate(frames[::a.every]):
            badge = (f"성공 · {info['rise']*100:.0f} cm 들어 올림 · {rate}" if info["success"]
                     else f"들어 올린 높이 {max(0, info['rise'])*100:.0f} cm")      # (malgun has no check-mark glyph)
            clips.append(caption(img, title, sub, badge, info["success"]))
        clips.extend([clips[-1]] * 8)                                        # short pause between clips

    fps = 20 / a.every
    pal = [c.convert("P", palette=Image.ADAPTIVE, colors=128) for c in clips]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"{len(clips)} frames @ {fps:.0f} fps -> {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
