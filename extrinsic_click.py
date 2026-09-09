"""
extrinsic_click.py -- fixed-camera extrinsics: auto-detect the checkerboard,
click twice to fix the frame orientation (no counting, sub-pixel accurate).

Why: a plain checkerboard has no orientation marker, so findChessboardCorners
can label its grid 180-deg / transposed from what you intend, silently rotating
the world frame. Two clicks remove that: you point at the corner you want as the
origin, then at any corner along the direction you want to be +X.

    python extrinsic_click.py --camera 2 --pattern 6x5 --square 0.020

Writes T_base_cam + table_plane into cam.json (K/dist read from it). The camera
is known to sit ABOVE the table, so the board-normal sign is resolved by forcing
the camera height positive.
"""
import argparse
import json
import os

import numpy as np
import cv2


def parse_pat(s):
    a, b = s.lower().split("x")
    return int(a), int(b)


def grab(cam):
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(3, 1280)
    cap.set(4, 720)
    frame = None
    for _ in range(30):
        ok, f = cap.read()
        if ok and f is not None and f.size:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit(f"camera {cam}: no frame")
    return frame


def click2(img, corners):
    """Return the two clicked pixel points (snapped to nearest detected corner)."""
    picks = []

    def nearest(x, y):
        d = np.linalg.norm(corners - [x, y], axis=1)
        return tuple(corners[int(np.argmin(d))])

    def redraw():
        d = img.copy()
        for c in corners:
            cv2.circle(d, (int(c[0]), int(c[1])), 2, (180, 180, 180), -1)
        tips = ["click the ORIGIN corner (0,0)", "click a corner along +X"]
        for i, p in enumerate(picks):
            cv2.circle(d, (int(p[0]), int(p[1])), 7, (0, 0, 255), 2)
            cv2.putText(d, ["ORIGIN", "+X"][i], (int(p[0]) + 9, int(p[1]) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(picks) < 2:
            cv2.putText(d, tips[len(picks)], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            cv2.putText(d, "ENTER=ok   r=redo", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("extrinsic", d)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            p = nearest(x, y)
            if len(picks) < 2:
                picks.append(p)
            else:
                picks[-1] = p
            redraw()

    cv2.namedWindow("extrinsic", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("extrinsic", 1280, 720)
    cv2.setMouseCallback("extrinsic", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(picks) == 2:
            break
        if k == ord("r"):
            picks.clear(); redraw()
        if k == 27:
            cv2.destroyAllWindows(); raise SystemExit("cancelled")
    cv2.destroyAllWindows()
    return np.array(picks, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--image")
    ap.add_argument("--pattern", default="6x5", help="inner corners, e.g. 6x5")
    ap.add_argument("--square", type=float, required=True)
    ap.add_argument("--config", default="output/cam.json")
    ap.add_argument("--out", default="output/extrinsic_check.png")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    K = np.asarray(cfg["K"], float).reshape(3, 3)
    dist = np.asarray(cfg.get("dist", [0] * 5), float).ravel()

    img = cv2.imread(a.image) if a.image else grab(a.camera)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pw, ph = parse_pat(a.pattern)
    found, cor = cv2.findChessboardCorners(
        gray, (pw, ph), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        cv2.imwrite(a.out, img)
        raise SystemExit(f"no checkerboard -- board flat & unobstructed & lit (saw -> {a.out})")
    cor = cv2.cornerSubPix(
        gray, cor, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)).reshape(ph, pw, 2)

    picks = click2(img, cor.reshape(-1, 2))

    # locate the two clicks in the (ph, pw) grid
    def idx_of(p):
        d = np.linalg.norm(cor.reshape(-1, 2) - p, axis=1)
        k = int(np.argmin(d))
        return k // pw, k % pw          # (row, col)

    r0, c0 = idx_of(picks[0])
    rx, cx = idx_of(picks[1])
    dr, dc = rx - r0, cx - c0
    if (dr, dc) == (0, 0):
        raise SystemExit("the two clicks are the same corner -- redo")

    # +X axis = grid direction of (dr,dc), normalised to a unit grid step
    def unit(v):
        return 0 if v == 0 else (1 if v > 0 else -1)
    xstep = np.array([unit(dr), unit(dc)])          # (drow, dcol) per +X step
    ystep = np.array([-xstep[1], xstep[0]])         # perpendicular in grid -> +Y

    obj, imgp = [], []
    for rr in range(ph):
        for cc in range(pw):
            g = np.array([rr - r0, cc - c0])        # grid offset from origin
            xi = g @ xstep                          # steps along +X
            yi = g @ ystep                          # steps along +Y
            obj.append([xi * a.square, yi * a.square, 0.0])
            imgp.append(cor[rr, cc])
    obj = np.array(obj, np.float32)
    imgp = np.array(imgp, np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    def world_from(rvec, tvec):
        R_cb, _ = cv2.Rodrigues(rvec)
        return R_cb.T, (-R_cb.T @ tvec.reshape(3))

    R_wc, t_wc = world_from(rvec, tvec)
    if t_wc[2] < 0:                                  # camera must be above the table
        obj[:, 1] = -obj[:, 1]                       # flip +Y -> flips Z too
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        R_wc, t_wc = world_from(rvec, tvec)
        print("(flipped +Y so the camera comes out above the plane)")

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp) ** 2, axis=1))))

    T = np.eye(4)
    T[:3, :3] = R_wc
    T[:3, 3] = t_wc
    cfg["T_base_cam"] = T.tolist()
    cfg["table_plane"] = {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]}
    cfg["extrinsic_rms_px"] = rms
    cfg.pop("_todo", None)
    json.dump(cfg, open(a.config, "w"), indent=2)

    xspan = obj[:, 0].max() - obj[:, 0].min()
    yspan = obj[:, 1].max() - obj[:, 1].min()
    print(f"reproj RMS over {len(obj)} corners: {rms:.2f} px")
    print("camera position in board frame (m):", np.round(t_wc, 4))
    print(f"  camera height above table: {t_wc[2]:.3f} m")
    print("camera view dir (world):", np.round(R_wc @ [0, 0, 1], 3), " (Z<0 = looking down, good)")
    print(f"board grid span: X {xspan*100:.0f} cm, Y {yspan*100:.0f} cm")
    print(f"wrote T_base_cam + table_plane -> {a.config}")
    if rms > 1.0:
        print("  WARNING rms > 1 px")

    cv2.drawFrameAxes(img, K, dist, rvec, tvec, a.square * 4, 3)
    cv2.imwrite(a.out, img)
    print(f"axes overlay -> {a.out}  (R=X  G=Y  B=Z up)")


if __name__ == "__main__":
    main()
