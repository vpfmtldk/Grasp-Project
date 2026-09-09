# Project status / handoff

Monocular (no-depth) RGB grasp detection on SO-101 + AmazingHand, low-cost.
Everything below is in this repo on `master`.

> **Portfolio-facing summary of finished results: [`RESULTS.md`](RESULTS.md).**
> This file is the working handoff; a few sections below predate the final runs.

## 1. Models (pretraining on Cornell) — DONE

| model | input | folder | clean eval IoU (89-img val, no aug) |
|---|---|---|---|
| GG-CNN | depth | `output/models/260901_1924_training_example/` epoch 29 | 0.798 |
| GG-CNN | RGB | `output/models/260902_1343_ggcnn_rgb1_d0/` epoch 22 | 0.854 |
| **GR-ConvNet** | **RGB** | `output/models/260903_1112_grconvnet_rgb1_d0/` **epoch 13** | **0.955** |

**Deployed model = `output/models/final_grconvnet_rgb1_d0/weights.pt`** (GR-ConvNet RGB,
trained on 95 % of Cornell, epoch 22; see that folder's `MODEL.md`). `eval_ggcnn.load_network`
now auto-detects the architecture from the state-dict keys, so a bare `weights.pt` loads fine.

RGB-only is competitive with / beats depth on Cornell -> supports the "no depth sensor" thesis.

### Cross-validation (image-wise 5-fold, `run_cv.ps1 -Network grconvnet ... -Epochs 25`) — DONE
`output/cv_grconvnet_rgb1_d0_e25.txt` — folds 0.944 / 0.831 / 0.843 / 0.944 / 0.955,
**mean IoU 0.903, sd 0.054**. Single-split 0.955 was optimistic; quote the mean ± sd.
(Ran via a Windows Scheduled Task so it survived window-close / sleep.)

## 2. Vision pipeline scripts — DONE (need calibration data to be live)

```
camera frame -> predict_grasp.py -> (u,v,theta,width,quality)
             -> pixel_to_world.py -> base-frame xyz + yaw + width_m + T_base_grasp
```

| script | what |
|---|---|
| `predict_grasp.py` | single image -> grasp dict(s); `--json` writes them for the next stage |
| `pixel_to_world.py` | ray-plane intersection with the table plane; needs a JSON config (K, dist, T_base_cam, table_plane). `--make-template`, `--selftest` |
| `grasp_geometry.py` | non-learned angle baseline (PCA / minAreaRect) |
| `compare_grasp_angles.py` | experiment 2: learned vs geometric vs GT angle on Cornell val |
| `camera_calib.py` | capture checkerboard + `cv2.calibrateCamera` -> writes K/dist into the pixel_to_world config |

### Experiment 2 result (GR-ConvNet vs geometry, Cornell val, angle error deg)
learned vs GT: 83% within 30 deg; geometric (PCA) vs GT: 67%; minAreaRect baseline is buggy (ignore).
Learned beats geometry; gap is on non-elongated objects.

## 3. Robot driver — WORKING (tuning left)

`robot/robot_control.py` (`Config`, `SO101`, `AmazingHand`), `robot/demo_move.py`, `robot/drive.py`.

Confirmed hardware:
- **Arm**: COM9, Feetech STS3215, ids 1-5 bottom->top
  (1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex, 4 wrist_flex, 5 wrist_roll). 1M baud.
  home `[0,-88,89,-93,154]`, look `[-1,9,89,-93,154]` (measured).
- **Hand**: COM8, Feetech **SC090** (SCS series, 1024 steps/rev, **protocol_end=1**), ids 1-8
  (index 1,2 / middle 3,4 / ring 5,6 / thumb 7,8). Fingers are **differential**:
  two servos opposite sign = flex, same sign = splay. Flex confirmed `{a:+90, b:-60}`.

Run:
```
python robot\demo_move.py --dry-run    # sim
python robot\demo_move.py --hand-only
python robot\demo_move.py --arm-only
python robot\demo_move.py              # both
python robot\robot_control.py --jog / --hand-jog   # keyboard pose building
```

Left to tune: hand preset angles (open/pinch/power), thumb sign if it opens instead of closing,
arm joint `sign` if any + direction is inverted.

Known bus issue: this scservo_sdk build's reads are flaky (retry logic added); torque-disable
is unreliable -> use `--jog` (torque on) not `--read-pose` for posing.

## 4. Camera placement — DECISION PENDING

Current mount (post at table edge, looking across) is **not good**: oblique side view,
poor depth resolution on the key axis, arm/hand in frame.

Do instead, best first:
1. Overhead boom over the workspace centre, pointing straight down (matches Cornell).
2. Elevated behind/beside the robot (~50 cm, tilted down 45-60 deg), looking the same
   direction the arm reaches.
Never: low + horizontal, or across the table.
After mounting: rigid, then `camera_calib.py` for K, then hand-eye for `T_base_cam`.

## 5. Next steps

1. Finish CV -> report mean +/- sd.
2. Fix camera placement (section 4).
3. `camera_calib.py run ...` -> K into `output/cam.json`.
4. `hand_eye_calibrate.py` (not written yet) -> `T_base_cam` into `output/cam.json`.
   Eye-to-hand: ChArUco on the hand, ~20 arm poses, `cv2.calibrateHandEye`.
5. Add IK: `SO101.ee_pose_to_joints` is stubbed for ikpy + the SO-101 URDF.
6. Collect ~100-200 RGB images of the real objects, label grasps (Cornell 4-corner format),
   fine-tune GR-ConvNet (`train_ggcnn.py --resume <grconvnet ckpt> --dataset-path <your data>`).
7. Wire `drive.py --pick` target to: capture -> predict_grasp -> pixel_to_world -> IK.
   That is `grasp_and_execute.py`.
8. Run N grasp trials, report success rate + failure modes (this is the actual contribution).

## 6. Infra notes

- `run_train.ps1` / `run_cv.ps1`: auto-resume via `--save-folder` + `ckpt_last.pt`.
  Re-run the same command after a crash/close to continue.
- This machine is **Modern Standby** — background/detached jobs get killed after ~20-60 min
  when idle. Long jobs must run in a foreground terminal window, machine kept awake, AC power.
