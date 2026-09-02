"""
Experiment 2 core: compare grasp ANGLES from
  * a learned policy (GG-CNN / GG-CNN2)
  * an explicit geometric model (grasp_geometry: PCA / minAreaRect)
  * the human-labelled ground truth (Cornell)

For every image in the Cornell validation split it prints, in degrees and with
pi-periodic wrapping:
  learned vs GT   |   geometric vs GT   |   learned vs geometric
plus aggregate mean / median / <=30 deg agreement rate.

Usage
-----
python compare_grasp_angles.py \
    --network output/models/<rgb_run>/epoch_XX_iou_0.XX \
    --dataset-path C:\\path\\to\\cornell_grasp \
    --use-rgb 1 --use-depth 0 [--split 0.9] [--limit 0] [--method pca] [--save-fig-dir figs]
"""
import argparse
import glob
import os

import numpy as np
from imageio.v2 import imread

from predict_grasp import GraspPredictor, grasp_corners
from grasp_geometry import estimate_grasp_angle, wrap_angle
from utils.dataset_processing.grasp import GraspRectangles


def ang_diff_deg(a, b):
    """pi-periodic absolute difference in degrees."""
    return abs(np.degrees(wrap_angle(a - b)))


def nearest_gt(gt_rects, center_rc):
    """Return (angle_rad, (row, col)) of the GT grasp whose centre is closest to center_rc."""
    best, best_d = None, 1e18
    cr, cc = center_rc
    for gr in gt_rects:
        r, c = gr.center
        d = (r - cr) ** 2 + (c - cc) ** 2
        if d < best_d:
            best_d, best = d, gr
    return float(best.angle), tuple(best.center)


def cornell_val_files(dataset_path, split, ds_rotate=0.0):
    graspf = sorted(glob.glob(os.path.join(dataset_path, '*', 'pcd*cpos.txt')))
    if not graspf:
        raise FileNotFoundError('No Cornell files under %s' % dataset_path)
    if ds_rotate:
        n = int(len(graspf) * ds_rotate)
        graspf = graspf[n:] + graspf[:n]
    return graspf[int(len(graspf) * split):]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--network', required=True)
    p.add_argument('--dataset-path', required=True)
    p.add_argument('--use-rgb', type=int, default=1)
    p.add_argument('--use-depth', type=int, default=0)
    p.add_argument('--split', type=float, default=0.9, help='Fraction used for training; the rest is compared.')
    p.add_argument('--ds-rotate', type=float, default=0.0)
    p.add_argument('--method', choices=['pca', 'minarearect'], default='pca')
    p.add_argument('--seg', choices=['otsu', 'adaptive'], default='otsu')
    p.add_argument('--limit', type=int, default=0, help='Only process the first N images (0 = all).')
    p.add_argument('--save-fig-dir', type=str, default='', help='If set, save per-image overlays here.')
    args = p.parse_args()

    predictor = GraspPredictor(args.network, use_rgb=args.use_rgb, use_depth=args.use_depth)
    grasp_files = cornell_val_files(args.dataset_path, args.split, args.ds_rotate)
    if args.limit:
        grasp_files = grasp_files[:args.limit]

    if args.save_fig_dir:
        os.makedirs(args.save_fig_dir, exist_ok=True)

    rows = []
    print('%-24s %10s %10s %10s' % ('image', 'L-vs-GT', 'G-vs-GT', 'L-vs-G'))
    for gf in grasp_files:
        rgb_path = gf.replace('cpos.txt', 'r.png')
        if not os.path.exists(rgb_path):
            continue
        rgb = np.asarray(imread(rgb_path))[..., :3].astype(np.uint8)

        try:
            geo = estimate_grasp_angle(rgb, method=args.method, seg_method=args.seg)
        except Exception as e:                       # segmentation can fail on odd scenes
            print('%-24s  skipped (%s)' % (os.path.basename(rgb_path), e))
            continue

        # The learned model sees RGB or depth depending on how it was trained;
        # geometry + GT always come from the RGB frame.
        if args.use_depth and not args.use_rgb:
            depth_path = gf.replace('cpos.txt', 'd.tiff')
            net_input = np.asarray(imread(depth_path), dtype=np.float32)
        else:
            net_input = rgb
        pred = predictor.predict(net_input, n_grasps=1)
        if not pred:
            print('%-24s  skipped (no grasp detected)' % os.path.basename(rgb_path))
            continue
        learned = pred[0]

        gt_rects = GraspRectangles.load_from_cornell_file(gf)
        gt_angle, gt_center = nearest_gt(gt_rects, geo['center'])

        d_lg = ang_diff_deg(learned['angle_rad'], gt_angle)
        d_gg = ang_diff_deg(geo['angle_rad'], gt_angle)
        d_lgeo = ang_diff_deg(learned['angle_rad'], geo['angle_rad'])
        rows.append((d_lg, d_gg, d_lgeo))
        print('%-24s %10.1f %10.1f %10.1f' % (os.path.basename(rgb_path), d_lg, d_gg, d_lgeo))

        if args.save_fig_dir:
            _save_overlay(rgb, learned, geo, gt_angle, gt_center,
                          os.path.join(args.save_fig_dir, os.path.basename(rgb_path)))

    if not rows:
        print('\nNo images processed.')
        return

    a = np.array(rows)
    names = ['learned vs GT', 'geometric vs GT', 'learned vs geometric']
    print('\n=== summary over %d images (degrees) ===' % len(a))
    print('%-24s %8s %8s %8s %10s' % ('', 'mean', 'median', 'p90', '<=30deg'))
    for i, nm in enumerate(names):
        col = a[:, i]
        print('%-24s %8.1f %8.1f %8.1f %9.0f%%'
              % (nm, col.mean(), np.median(col), np.percentile(col, 90), 100.0 * np.mean(col <= 30)))


def _save_overlay(rgb, learned, geo, gt_angle, gt_center, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(rgb)
    # learned (red), geometric (cyan), GT (green) -- draw as short oriented segments
    for (ang, cen, color, lbl) in [
        (learned['angle_rad'], (learned['y'], learned['x']), 'r', 'learned'),
        (geo['angle_rad'], geo['center'], 'c', 'geometric'),
        (gt_angle, gt_center, 'lime', 'GT'),
    ]:
        cy, cx = cen
        L = 0.35 * min(rgb.shape[:2])
        xo, yo = np.cos(ang), np.sin(ang)
        ax.plot([cx - L / 2 * xo, cx + L / 2 * xo], [cy + L / 2 * yo, cy - L / 2 * yo],
                color=color, linewidth=2.5, label='%s %.0f deg' % (lbl, np.degrees(ang)))
    ax.legend(loc='lower right', fontsize=8)
    ax.axis('off')
    fig.savefig(out_path, bbox_inches='tight', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
