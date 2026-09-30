r"""Red marker (pointer tip) detection for auto hand-eye collection.

    python -m robot.calib.marker --cam 2          # live check: saves output/marker_check.png

Red wraps around hue 0 in OpenCV HSV, so two hue bands are ORed. Candidates are
filtered by area and roundness -- the tape wrapped on the pointer tip is a compact
blob; red servo-cable wires are thin and long, and label text is fragmented.
"""
import argparse
import os

import cv2
import numpy as np

HUE_BANDS = [(0, 10), (170, 180)]
S_MIN, V_MIN = 110, 70
AREA_MIN, AREA_MAX = 110, 6000      # px at 1280x720, camera ~0.3 m above the table.
                                    # measured: tape band on the pointer tip ~196 px;
                                    # skin specks (a hand on the mouse) 32-82 px.
SOLIDITY_MIN = 0.75                 # blob area / convex-hull area -- rejects wires/text
ASPECT_MAX = 3.0                    # bounding-box long/short side


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in HUE_BANDS:
        m |= cv2.inRange(hsv, (lo, S_MIN, V_MIN), (hi, 255, 255))
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def candidates(bgr, roi=None):
    """All red blobs passing the shape filter, largest first.
    roi: (x0, y0, x1, y1) -- blobs whose centre is outside are dropped."""
    m = red_mask(bgr)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (AREA_MIN <= a <= AREA_MAX):
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        sol = a / hull if hull > 0 else 0
        x, y, w, h = cv2.boundingRect(c)
        asp = max(w, h) / max(1, min(w, h))
        if sol < SOLIDITY_MIN or asp > ASPECT_MAX:
            continue
        M = cv2.moments(c)
        u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
        if roi and not (roi[0] <= u <= roi[2] and roi[1] <= v <= roi[3]):
            continue
        out.append({"u": u, "v": v, "area": a, "solidity": sol, "aspect": asp, "contour": c})
    return sorted(out, key=lambda d: -d["area"])


def detect(bgr, roi=None):
    """(u, v) of the marker, or None if there isn't exactly one clear candidate."""
    c = candidates(bgr, roi)
    if len(c) == 1:
        return c[0]["u"], c[0]["v"]
    if len(c) > 1 and c[0]["area"] > 3 * c[1]["area"]:   # one dominant blob
        return c[0]["u"], c[0]["v"]
    return None


def draw(bgr, roi=None):
    img = bgr.copy()
    m = red_mask(bgr)
    img[m > 0] = (0.5 * img[m > 0] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)   # raw mask in magenta
    for i, c in enumerate(candidates(bgr, roi)):
        cv2.drawContours(img, [c["contour"]], -1, (0, 255, 0), 2)
        cv2.putText(img, f"{i}: a={c['area']:.0f}", (int(c["u"]) + 8, int(c["v"]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    p = detect(bgr, roi)
    if p:
        cv2.drawMarker(img, (int(p[0]), int(p[1])), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
    if roi:
        cv2.rectangle(img, roi[:2], roi[2:], (255, 200, 0), 1)
    return img, p


def grab(cam):
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(3, 1280); cap.set(4, 720)
    f = None
    for _ in range(15):              # auto-exposure settles over the first frames
        ok, fr = cap.read()
        if ok:
            f = fr
    cap.release()
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--image", help="use a saved image instead of the camera")
    ap.add_argument("--out", default="output/marker_check.png")
    a = ap.parse_args()
    f = cv2.imread(a.image) if a.image else grab(a.cam)
    if f is None:
        raise SystemExit("no frame")
    img, p = draw(f)
    cv2.imwrite(a.out, img)
    print(f"candidates: {len(candidates(f))}   detected: {p}   -> {a.out}")
