"""
camera_calib.py -- capture checkerboard views from the USB camera and compute
intrinsics (K + distortion), then write them into a pixel_to_world.py config.

Subcommands
-----------
  capture    live preview, SPACE saves a frame, Q/ESC quits
  calibrate  run calibration on a folder of images
  run        capture then calibrate in one go

Examples
--------
  python camera_calib.py run --camera 0 --pattern 9x6 --square 0.025 --config output/cam.json
  python camera_calib.py capture --camera 0 --out calib_imgs
  python camera_calib.py calibrate --images calib_imgs --pattern 9x6 --square 0.025 \
      --config output/cam.json --fisheye

Notes
-----
* --pattern is the number of INNER corners (a 10x7-square board -> 9x6).
* --square is the printed square size in METRES (measure it -- printers lie).
* Aim for 15-25 views: board filling different parts of the frame, tilted in
  several directions, a few close and a few far. Keep it flat and rigid.
* mean reprojection error < ~0.5 px is great, < 1 px is fine, > 2 px means bad
  shots or the wrong distortion model (try/untry --fisheye).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def parse_pattern(s):
    a, b = s.lower().split('x')
    return (int(a), int(b))


def open_camera(index, width=1280, height=720):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError('could not open camera index %d (try 1, 2, ...)' % index)
    return cap


def cmd_capture(args):
    os.makedirs(args.out, exist_ok=True)
    cap = open_camera(args.camera)
    pattern = parse_pattern(args.pattern)
    n = len(glob.glob(os.path.join(args.out, 'cal_*.png')))
    print('SPACE = save, Q/ESC = quit. Saving to %s (starting at %d)' % (args.out, n))
    while True:
        ok, frame = cap.read()
        if not ok:
            print('frame grab failed'); break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, 'saved: %d   board: %s' % (n, 'YES' if found else 'no'),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow('camera_calib (SPACE=save, Q=quit)', view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord(' '):
            p = os.path.join(args.out, 'cal_%03d.png' % n)
            cv2.imwrite(p, frame)
            print('  saved', p, '' if found else '(no board detected -- may be unusable)')
            n += 1
    cap.release()
    cv2.destroyAllWindows()
    return n


def calibrate(images, pattern, square, fisheye):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= square

    objpoints, imgpoints, used = [], [], []
    size = None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for f in sorted(images):
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print('  no board:', os.path.basename(f))
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used.append(f)

    if len(used) < 5:
        raise RuntimeError('only %d usable views -- need >= 5 (aim for 15+)' % len(used))
    print('using %d / %d views, image size %s' % (len(used), len(images), size))

    if fisheye:
        K = np.zeros((3, 3)); D = np.zeros((4, 1))
        op = [o.reshape(-1, 1, 3) for o in objpoints]
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW)
        rms, K, D, _, _ = cv2.fisheye.calibrate(op, imgpoints, size, K, D, flags=flags, criteria=crit)
        dist = D.ravel()
        model = 'fisheye'
        per_view = None
    else:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
        dist = dist.ravel()
        model = 'pinhole'
        per_view = []
        for i in range(len(used)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
            per_view.append((os.path.basename(used[i]), float(err)))

    print('\nmodel: %s   RMS reprojection error: %.4f px' % (model, rms))
    print('K =\n', np.round(K, 3))
    print('dist =', np.round(dist, 5))
    if per_view:
        worst = sorted(per_view, key=lambda t: -t[1])[:3]
        print('worst views:', ', '.join('%s=%.2f' % w for w in worst),
              ' (re-shoot or drop if >> the rest)')
    return K, dist, size, model, float(rms)


def write_config(path, K, dist, size, model, rms):
    cfg = {}
    if os.path.isfile(path):
        with open(path) as f:
            cfg = json.load(f)
        print('updating existing config (T_base_cam / table_plane kept):', path)
    cfg['K'] = [[float(v) for v in row] for row in np.asarray(K).reshape(3, 3)]
    cfg['dist'] = [float(v) for v in np.asarray(dist).ravel()]
    cfg['model'] = model
    cfg['image_size'] = [int(size[0]), int(size[1])]
    cfg['rms_reproj_px'] = rms
    cfg.setdefault('T_base_cam', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    cfg.setdefault('table_plane', {'point': [0.0, 0.0, 0.0], 'normal': [0.0, 0.0, 1.0]})
    if cfg['T_base_cam'] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]:
        cfg['_todo'] = 'T_base_cam is still identity -- fill it from hand-eye calibration.'
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('wrote', path)


def cmd_calibrate(args):
    imgs = glob.glob(os.path.join(args.images, '*.png')) + glob.glob(os.path.join(args.images, '*.jpg'))
    if not imgs:
        raise RuntimeError('no images in ' + args.images)
    K, dist, size, model, rms = calibrate(imgs, parse_pattern(args.pattern), args.square, args.fisheye)
    if args.config:
        write_config(args.config, K, dist, size, model, rms)


def cmd_run(args):
    n = cmd_capture(args)
    if n < 5:
        print('not enough views captured (%d)' % n); return
    cmd_calibrate(args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--pattern', default='9x6', help='inner corners, e.g. 9x6')
    common.add_argument('--square', type=float, default=0.025, help='square size in metres')
    common.add_argument('--config', default='output/cam.json', help='pixel_to_world config to write/update')
    common.add_argument('--fisheye', action='store_true', help='use the fisheye distortion model')

    c = sub.add_parser('capture', parents=[common]); c.set_defaults(func=cmd_capture)
    c.add_argument('--camera', type=int, default=0)
    c.add_argument('--out', default='calib_imgs')

    cal = sub.add_parser('calibrate', parents=[common]); cal.set_defaults(func=cmd_calibrate)
    cal.add_argument('--images', default='calib_imgs')

    r = sub.add_parser('run', parents=[common]); r.set_defaults(func=cmd_run)
    r.add_argument('--camera', type=int, default=0)
    r.add_argument('--out', default='calib_imgs')
    r.add_argument('--images', default='calib_imgs')

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
