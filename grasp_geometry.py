"""
Explicit geometric grasp-angle baseline.

For an object lying on a plain background, estimate the parallel-jaw grasp angle
directly from geometry -- no learning. Two estimators:

  * 'pca'          -- principal axis of the object mask; grasp closes across the
                      short axis (perpendicular to the elongation).
  * 'minarearect'  -- cv2.minAreaRect of the largest contour; grasp closes across
                      the shorter side.

Angles are returned in the SAME convention as utils.dataset_processing.grasp
(``Grasp.angle``): radians in (-pi/2, pi/2], positive = anti-clockwise from the
image horizontal, pi-periodic.

This is the geometric side of experiment 2: compare what a learned policy
(GG-CNN) implicitly picks against what this explicit model computes.
"""
import argparse

import cv2
import numpy as np
from imageio.v2 import imread


def wrap_angle(a):
    """Wrap to (-pi/2, pi/2] (grasp orientation is pi-periodic)."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def _line_phi_to_grasp_angle(phi):
    """
    phi: orientation (rad) of the grasp/closing axis in image coords (y down),
         i.e. a direction (cos phi, sin phi) in (x, y).
    Convert to the repo's Grasp.angle convention (see grasp.GraspRectangle.angle,
    which uses arctan2(-dy, dx) on the jaw-separation vector).
    """
    return wrap_angle(np.pi - phi)


def _fill_holes(mask):
    ff = mask.copy()
    h, w = mask.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(ff)


def segment_object(rgb, method='otsu', min_area_frac=0.003, pick='center'):
    """
    Return a uint8 mask (255 = object) for a single object on a plain background.
    Cornell-style scenes: bright/uniform background, darker object.

    pick='center' keeps the blob whose centroid is closest to the frame centre
    (objects sit roughly centred); pick='largest' keeps the biggest blob.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if method == 'adaptive':
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 5)
    else:  # otsu
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = _fill_holes(mask)

    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    h, w = mask.shape
    fc = np.array([w / 2.0, h / 2.0])
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * mask.size]
    if not cand:
        cand = list(range(1, n))
    if pick == 'largest':
        best = max(cand, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    else:
        best = min(cand, key=lambda i: np.hypot(*(cents[i] - fc)))
    return np.uint8((lbl == best) * 255)


def angle_from_pca(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        raise ValueError('empty / tiny object mask')
    pts = np.column_stack([xs, ys]).astype(np.float64)      # (x, y)
    mean = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
    long_axis = evecs[:, int(np.argmax(evals))]             # (vx, vy)
    phi_long = np.arctan2(long_axis[1], long_axis[0])
    grasp_angle = _line_phi_to_grasp_angle(phi_long + np.pi / 2)   # close across short axis
    elongation = float(np.sqrt(evals.max() / max(evals.min(), 1e-9)))
    center = (float(mean[1]), float(mean[0]))               # (row, col)
    return grasp_angle, center, elongation


def angle_from_minarearect(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError('no contour found')
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), deg = cv2.minAreaRect(c)
    if w < h:                                                # close across the shorter side
        deg += 90.0
    grasp_angle = _line_phi_to_grasp_angle(np.radians(deg))
    elongation = float(max(w, h) / max(min(w, h), 1e-6))
    return grasp_angle, (float(cy), float(cx)), elongation


def _center_crop(img, crop_size):
    """Match the model's field of view before segmenting; returns (crop, top, left)."""
    h, w = img.shape[:2]
    if crop_size and h >= crop_size and w >= crop_size:
        side = crop_size
    else:
        side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return img[top:top + side, left:left + side], top, left


def estimate_grasp_angle(rgb, method='pca', seg_method='otsu', mask=None, crop_size=224, pick='center'):
    """
    rgb: HxWx3 uint8. Optionally pass a precomputed `mask` (uint8, 255=object),
    which is assumed to already be in the cropped frame.
    By default a `crop_size` centre window is segmented (so the object dominates
    and the plain background is clean), matching predict_grasp's field of view.
    `center` is returned in the ORIGINAL image coordinates.
    Returns dict: {angle_rad, angle_deg, center (row,col), elongation, method, mask}
    """
    off_r = off_c = 0
    work = rgb
    if mask is None and crop_size is not None:
        work, off_r, off_c = _center_crop(rgb, crop_size)
    if mask is None:
        mask = segment_object(work, method=seg_method, pick=pick)
    if method == 'minarearect':
        a, center, elong = angle_from_minarearect(mask)
    else:
        a, center, elong = angle_from_pca(mask)
    return {
        'angle_rad': float(a),
        'angle_deg': float(np.degrees(a)),
        'center': (center[0] + off_r, center[1] + off_c),
        'elongation': elong,
        'method': method,
        'mask': mask,
    }


def visualise(rgb, est, out_path=None):
    import matplotlib.pyplot as plt
    cy, cx = est['center']
    L = 0.4 * min(rgb.shape[:2])
    xo, yo = np.cos(est['angle_rad']), np.sin(est['angle_rad'])
    # jaw-separation axis (matches grasp.Grasp.as_gr direction)
    p1 = (cx - L / 2 * xo, cy + L / 2 * yo)
    p2 = (cx + L / 2 * xo, cy - L / 2 * yo)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(est['mask'], cmap='gray')
    axes[0].set_title('object mask')
    axes[0].axis('off')
    axes[1].imshow(rgb)
    axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3)
    axes[1].plot(cx, cy, 'yo')
    axes[1].set_title('geometric grasp angle: %.1f deg (%s)' % (est['angle_deg'], est['method']))
    axes[1].axis('off')
    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=120)
        print('saved', out_path)
    else:
        plt.show()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description='Geometric (non-learned) grasp-angle estimate for an image.')
    p.add_argument('--image', required=True, help='Path to an RGB image')
    p.add_argument('--method', choices=['pca', 'minarearect'], default='pca')
    p.add_argument('--seg', choices=['otsu', 'adaptive'], default='otsu')
    p.add_argument('--pick', choices=['center', 'largest'], default='center')
    p.add_argument('--crop-size', type=int, default=224,
                   help='Centre window segmented (default 224). 0 = whole frame.')
    p.add_argument('--vis', action='store_true')
    p.add_argument('--out', type=str, default='')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rgb = np.asarray(imread(args.image))
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = rgb[..., :3].astype(np.uint8)

    est = estimate_grasp_angle(rgb, method=args.method, seg_method=args.seg,
                               crop_size=(args.crop_size or None), pick=args.pick)
    print('geometric grasp angle: %.2f deg  (center row=%.0f col=%.0f, elongation=%.2f)'
          % (est['angle_deg'], est['center'][0], est['center'][1], est['elongation']))

    if args.vis or args.out:
        visualise(rgb, est, out_path=(args.out or None))
