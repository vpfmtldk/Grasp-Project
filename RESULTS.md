# Grip-angle auto-detection for a low-cost monocular grasping robot

Predict **where** and **at what rotation** to grasp a tabletop object from a single
RGB image, recover a 3D grasp pose through a calibrated fixed camera, and hand it to
an **SO-101 arm + AmazingHand** — no depth sensor, no motion capture, off-the-shelf
parts.

```
  RGB frame ──▶ GR-ConvNet ──▶ (u, v, θ, width, q)      grasp in image space
            └─▶ camera model (K, dist, T_base_cam) ──▶ ray ∩ table plane
            └─▶ 3D grasp pose (x, y, z, yaw, width) in the robot base frame
            └─▶ IK ──▶ arm move ──▶ close ──▶ lift
```

## Constraints (deliberate)

| choice | why |
|---|---|
| **Monocular RGB only** — no depth camera | matches recent RGB-only grasp literature; removes a costly sensor. Depth is recovered geometrically by intersecting the pixel's viewing ray with the known table plane. |
| **Fixed camera** (eye-to-hand) | one calibration, valid for the whole session |
| **SO-101 + AmazingHand**, no Raspberry Pi / LeKiwi base | low-cost tabletop rig; the mobile base was dropped |
| **Coupled 1-DOF grip** | the research contribution is the *perception*; the hand is an effector, driven open/close by one command (as the real AmazingHand is) |

---

## 1 · Grasp model — GR-ConvNet, RGB-only

- `models/grconvnet.py` — fully-convolutional; conv stem → 5 residual blocks → 2× bilinear
  upsample → `pos / cos / sin / width` heads at 300×300. ~1.83 M parameters.
- Trained on the **Cornell Grasping Dataset** (885 images), RGB only (`input_channels = 3`).
- Deployed weights: `output/models/final_grconvnet_rgb1_d0/weights.pt` (see `MODEL.md`).

### 5-fold image-wise cross-validation (`run_cv.ps1`, 25 epochs/fold)

| fold (`--ds-rotate`) | held-out IoU (Jaccard @ 25 % overlap, ≤ 30° angle) |
|---|---|
| 0.0 | 0.944 |
| 0.2 | 0.831 |
| 0.4 | 0.843 |
| 0.6 | 0.944 |
| 0.8 | 0.955 |
| **mean ± sd** | **0.903 ± 0.054** |

The fold spread (0.83–0.96) reflects how small Cornell is — a single split is optimistic,
so the mean ± sd is the number to quote. This is in range for **RGB-only** models
(the GR-ConvNet paper reports ~0.97 with RGB-**D**).

![Cornell predictions](docs/figures/grconvnet_montage.png)
*Predicted grasp (red box, blue jaws) vs human labels (green) on held-out Cornell images.*

---

## 2 · Experiment 2 — learned vs geometric grip angle

Does a learned predictor actually beat a classical geometric baseline for the **grasp
angle**? Baseline: PCA on the Otsu-segmented object mask (`grasp_geometry.py`).
Evaluated on 89 held-out Cornell images with a leakage-free model (CV fold-0 checkpoint,
trained only on the first 90 %).

| method | mean err | median | p90 | **≤ 30° agreement** |
|---|---|---|---|---|
| **Learned (GR-ConvNet)** | **15.4°** | 8.3° | 50.4° | **84 %** |
| Geometric (PCA) | 27.2° | 12.7° | 73.6° | 67 % |

![Angle-error comparison](docs/figures/exp2_angle_comparison.png)

The learned model dominates across the whole error range. PCA has a fat failure tail
(~15–20 % of objects) on **round / symmetric objects** where the mask has no clear major
axis — the learned model still gets those right:

![PCA failure cases](docs/figures/exp2_failure_cases.png)
*Red = learned, cyan = geometric, green = human label. PCA is ~90° off; the learned model tracks the label.*

Reproduce: `python compare_grasp_angles.py --network <fold-0 ckpt> --dataset-path <cornell> --use-rgb 1 --use-depth 0`

---

## 3 · Camera calibration

Fixed **Innomaker U20CAM-720P** USB camera looking down at the tabletop.

### Intrinsics — `camera_calib.py` (checkerboard, `cv2.calibrateCamera`)

| | |
|---|---|
| RMS reprojection error | **0.27 px** (26 views, 11 bad views auto-culled) |
| focal length | fx ≈ 994, fy ≈ 989 px (ratio 1.005 — square pixels) |
| principal point | (708, 406) |
| radial distortion | k1 ≈ −0.43, k2 ≈ 0.21 (barrel) |

![Undistortion](docs/figures/undistort_check.png)
*Left: raw. Right: undistorted (the black pincushion border is the barrel correction.)*

### Extrinsics — `extrinsic_click.py` (checkerboard = world frame, 2 clicks fix orientation)

A plain checkerboard has no orientation marker, so `findChessboardCorners` can label
its grid 180°/transposed and silently rotate the world frame (this cost a 100 mm error
on the first attempt). Fix: auto-detect the board, then click the origin corner and a
+X corner. The board-normal sign is resolved by the camera-above constraint.

| | |
|---|---|
| RMS reprojection error | **0.22 px** |
| camera height above table | 0.31 m (vertical); ≈ 0.36 m slant — matches the tape measurement |

**Validation:** re-projecting all 30 board corners through the full pipeline
(undistort → ray → table-plane intersection) reconstructs the 20 mm grid with
**mean 0.08 mm / max 0.20 mm** error, span 100.1 × 80.2 mm (ideal 100 × 80).
The pixel → 3D conversion is sub-millimetre on the table plane.

![Extrinsic axes](docs/figures/extrinsic_check.png)

---

## 4 · Real-object generalization (no fine-tuning)

The Cornell-trained model + calibrated pipeline, run straight on the USB camera over
10 arrangements of 3 household objects (box, can, cigarette pack):

![Real predictions](docs/figures/real_pred_montage.png)

- Boxes get short-axis antipodal grasps; the can gets an across-the-cylinder grasp
  that follows its orientation whether it is upright or lying down.
- 3D grasp poses are recovered in the checkerboard frame, e.g.
  `X −2.5  Y +1.3  Z +3.0 cm, yaw −9°`.

This is the domain-transfer evidence: a 2009-benchmark model works on a different
camera and different objects with no retraining.

---

## Repo map

| script | role |
|---|---|
| `models/grconvnet.py` | GR-ConvNet architecture (added to this GG-CNN fork) |
| `train_ggcnn.py` / `run_train.ps1` / `run_cv.ps1` | training, crash/sleep auto-resume, 5-fold CV |
| `predict_grasp.py` | image → grasp `{x, y, angle, width, quality}` (+ `--json`) |
| `pixel_to_world.py` | grasp pixel → base-frame 3D pose (ray ∩ table plane) |
| `grasp_geometry.py` | non-learned angle baseline (PCA / minAreaRect) |
| `compare_grasp_angles.py` | experiment 2 |
| `camera_calib.py` | checkerboard intrinsics → `output/cam.json` |
| `extrinsic_click.py` | fixed-camera extrinsics (2-click, un-ambiguous) → `output/cam.json` |
| `robot/robot_control.py` | SO-101 (STS3215) + AmazingHand (SCS) Feetech driver |
| `robot/mujoco_backend.py` | same pipeline API in MuJoCo (`sim` / `real` share `execute_grasp`) |
| `grasp_and_execute.py` | full loop: image → grasp → 3D → IK → move → close → lift |

`output/cam.json` — live camera calibration (K, dist, T_base_cam, table_plane).

---

## Status

**Done:** grasp model + 5-fold CV, experiment 2, camera calibration (intrinsic +
extrinsic, sub-mm validated), real-object prediction, all pipeline scripts, the
Feetech hardware driver.

**Open:**

- **Sim grasp loop** — the vendored AmazingHand MJCF has no collision geometry and a
  non-transmitting open-loop linkage, so it cannot grip in physics. A parallel-jaw
  collision proxy grips + lifts 4/4 standalone, but a faithful AmazingHand physics
  model needs the CAD assembly (only the individual parts are on hand). The sim
  currently uses the SO-ARM100 menagerie model (kinematically close to SO-101, not
  identical).
- **Real robot execution** — needs the checkerboard→SO-101-base transform (measure
  once, or touch reference points with the end-effector) and a hardware session.
