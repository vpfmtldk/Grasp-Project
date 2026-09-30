"""
object_center.py -- find the tabletop object and its centre for the AmazingHand.

GR-ConvNet (trained on Cornell with a parallel-jaw label) likes grasp points on object
EDGES; on the real rig that put the hand ~2 cm off the can's axis and the power grasp
failed. The AmazingHand wraps the object, so aim at the object's centre instead:

  1. segment the object against the plain table inside the calibrated area:
     colour distance from the table (Lab) OR edges (Canny), closed + hole-filled, so a
     white/pink can on a white table still comes out as one solid blob; soft shadows
     (small lightness drop, no colour) are rejected by the colour threshold
  2. pick the blob nearest GR-ConvNet's point (it still decides WHICH object)
  3. target = blob centroid; grasp direction = across the blob's short axis
     (minAreaRect), width = short side

    python object_center.py --image frame.png      # debug overlay -> output/object_center.png
"""
import argparse
import math

import cv2
import numpy as np

COLOR_DE = 22          # Lab distance from the table colour that counts as "object"
CANNY = (40, 110)
CLOSE_PX = 15          # joins edge fragments / text into one silhouette
MIN_AREA = 2000        # px at 1280x720 (~1 px = 0.37 mm -> ~17 mm square)


def segment(bgr, roi):
    """Binary object mask (uint8 0/255) restricted to roi = (u0, v0, u1, v1)."""
    u0, v0, u1, v1 = [int(round(x)) for x in roi]
    sub = bgr[v0:v1, u0:u1]
    lab = cv2.cvtColor(cv2.GaussianBlur(sub, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    table = np.median(lab.reshape(-1, 3), axis=0)             # object is a small part of the roi
    dL = table[0] - lab[..., 0]
    dab = np.linalg.norm(lab[..., 1:] - table[1:], axis=2)
    colour = (dab > 12) | (np.abs(dL) > 45)                   # a shadow is only a mild dL, no colour
    edges = cv2.Canny(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), *CANNY) > 0
    m = ((colour | edges) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_PX, CLOSE_PX))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, -1)               # fill holes (labels, white parts)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    full = np.zeros(bgr.shape[:2], np.uint8)
    full[v0:v1, u0:u1] = filled
    return full


def objects(bgr, roi):
    out = []
    cnts, _ = cv2.findContours(segment(bgr, roi), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_AREA:
            continue
        c = cv2.convexHull(c)            # white parts of a label leave dents; the hull fills them
        M = cv2.moments(c)
        (rx, ry), (w, h), ang = cv2.minAreaRect(c)
        long_deg = ang if w >= h else ang + 90                 # direction of the long side
        out.append({"contour": c, "area": a, "cx": M["m10"] / M["m00"], "cy": M["m01"] / M["m00"],
                    "long_deg": long_deg, "short_px": min(w, h), "long_px": max(w, h)})
    return out


SEG_MARGIN = 150      # px: segment beyond the calibrated box so a 122 mm upright can (longer
                      # than the 11.4 cm box) is seen whole; only its centre must be in the box


def seg_region(roi, shape, margin=SEG_MARGIN):
    u0, v0, u1, v1 = roi
    return (max(0, u0 - margin), max(0, v0 - margin), min(shape[1], u1 + margin), min(shape[0], v1 + margin))


def center_grasp(rgb, g, roi):
    """Return a copy of grasp g moved onto the centre of the object nearest to it, or None
    if no object is found. Image-angle convention follows predict_grasp: angle_rad is the
    closing direction; for a wrap grasp it is across the short axis."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    u0, v0, u1, v1 = roi
    obs = [o for o in objects(bgr, seg_region(roi, bgr.shape))
           if u0 <= o["cx"] <= u1 and v0 <= o["cy"] <= v1]            # centre must be calibrated
    if not obs:
        return None, []
    o = min(obs, key=lambda o: math.hypot(o["cx"] - g["x"], o["cy"] - g["y"]))
    close_deg = o["long_deg"] + 90                              # close across the object
    # predict_grasp angles: image y down, angle measured like atan2(-dy, dx)
    ang = -math.radians(close_deg)
    ang = (ang + math.pi / 2) % math.pi - math.pi / 2
    ng = dict(g, x=o["cx"], y=o["cy"], angle_rad=ang, angle_deg=math.degrees(ang),
              width_px=o["short_px"], model_x=g["x"], model_y=g["y"], object=o)
    return ng, obs


def draw(bgr, obs, g=None):
    from predict_grasp import grasp_corners
    img = bgr.copy()
    for o in obs:
        cv2.drawContours(img, [o["contour"]], -1, (0, 220, 255), 2)
    if g is not None:
        if "model_x" in g:
            cv2.drawMarker(img, (int(g["model_x"]), int(g["model_y"])), (255, 120, 0), cv2.MARKER_TILTED_CROSS, 16, 2)
        c = np.round(grasp_corners(g["x"], g["y"], g["angle_rad"], g["width_px"], g["width_px"] / 2)).astype(np.int32)
        cv2.polylines(img, [c], True, (0, 0, 255), 2)
        cv2.circle(img, (int(g["x"]), int(g["y"])), 7, (0, 0, 255), -1)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--roi", type=float, nargs=4, default=[351.1, 256.1, 877.1, 558.3])
    ap.add_argument("--out", default="output/object_center.png")
    a = ap.parse_args()
    bgr = cv2.imread(a.image)
    fake = {"x": (a.roi[0] + a.roi[2]) / 2, "y": (a.roi[1] + a.roi[3]) / 2, "angle_rad": 0.0,
            "angle_deg": 0.0, "width_px": 50.0, "quality": 1.0}
    g, obs = center_grasp(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fake, a.roi)
    img = draw(bgr, obs, g)
    u0, v0, u1, v1 = map(int, a.roi)
    cv2.rectangle(img, (u0, v0), (u1, v1), (0, 200, 0), 1)
    cv2.imwrite(a.out, img)
    print(f"{len(obs)} object(s); target {None if g is None else (round(g['x']), round(g['y']))}  -> {a.out}")
