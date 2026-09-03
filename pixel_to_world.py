"""
pixel_to_world.py

Map an image-space grasp (from predict_grasp.py) to a 3D grasp pose in the robot
base frame, for a FIXED MONOCULAR camera (no depth sensor).

Needs, from one-time calibration (put them in a JSON config -- see --make-template):
  * K (3x3) and distortion coeffs   -- cv2.calibrateCamera on a checkerboard
  * T_base_cam (4x4)                 -- camera pose in the robot base frame,
                                        from hand-eye (eye-to-hand) calibration
  * table_plane {point, normal}      -- a point on the table and its normal,
                                        both in the base frame (or just z = const)

A single RGB pixel only defines a viewing ray, so the 3D grasp point is found by
intersecting that ray with the table plane. This is exact only if the grasp
contact lies on that plane; an object's height above the table is the main
systematic error -- pass --object-height to lift the plane by that amount.

Chain:
    predict_grasp.py ... --json grasp.json
    pixel_to_world.py --config cam.json --grasp-json grasp.json
"""
import argparse
import json

import numpy as np

try:
    import cv2
except ImportError:                       # undistortion is skipped without cv2
    cv2 = None


# --------------------------------------------------------------------------- io

def _to_np(cfg):
    cfg = dict(cfg)
    cfg['K'] = np.asarray(cfg['K'], dtype=float).reshape(3, 3)
    cfg['dist'] = np.asarray(cfg.get('dist', [0, 0, 0, 0, 0]), dtype=float).ravel()
    cfg['T_base_cam'] = np.asarray(cfg['T_base_cam'], dtype=float).reshape(4, 4)
    tp = cfg.get('table_plane', {'point': [0, 0, 0], 'normal': [0, 0, 1]})
    cfg['plane_point'] = np.asarray(tp['point'], dtype=float).ravel()
    n = np.asarray(tp['normal'], dtype=float).ravel()
    cfg['plane_normal'] = n / np.linalg.norm(n)
    return cfg


def load_config(path):
    with open(path) as f:
        return _to_np(json.load(f))


def write_template_config(path):
    template = {
        "_comment": "Fill in after calibration. K/dist from cv2.calibrateCamera; "
                    "T_base_cam (camera pose in base frame) from cv2.calibrateHandEye "
                    "(eye-to-hand); table_plane point+normal measured in the base frame.",
        "K": [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "T_base_cam": [[1, 0, 0, 0.30],
                       [0, -1, 0, 0.00],
                       [0, 0, -1, 0.55],
                       [0, 0, 0, 1.0]],
        "table_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]}
    }
    with open(path, 'w') as f:
        json.dump(template, f, indent=2)
    print('wrote template config ->', path)


# ---------------------------------------------------------------- geometry core

def pixel_to_ray(uv, K, dist, T_base_cam):
    """Return (origin, direction) of the viewing ray for pixel uv, in the base frame."""
    uv = np.asarray(uv, dtype=float).reshape(1, 1, 2)
    if cv2 is not None and np.any(dist):
        norm = cv2.undistortPoints(uv, K, dist).reshape(2)      # already K^-1 applied
        d_cam = np.array([norm[0], norm[1], 1.0])
    else:
        Kinv = np.linalg.inv(K)
        d_cam = Kinv @ np.array([uv[0, 0, 0], uv[0, 0, 1], 1.0])
    d_cam = d_cam / np.linalg.norm(d_cam)
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    return t, R @ d_cam


def ray_plane_intersect(origin, direction, plane_point, plane_normal):
    """3D intersection point, or None if the ray is parallel to / points away from the plane."""
    denom = float(plane_normal @ direction)
    if abs(denom) < 1e-9:
        return None
    s = float(plane_normal @ (plane_point - origin)) / denom
    if s <= 0:
        return None
    return origin + s * direction


def _project(uv, cfg, plane_point):
    o, d = pixel_to_ray(uv, cfg['K'], cfg['dist'], cfg['T_base_cam'])
    return ray_plane_intersect(o, d, plane_point, cfg['plane_normal'])


def grasp_to_base(grasp, cfg, object_height=0.0, delta_px=15.0):
    """
    grasp: dict with x, y (pixels), angle_rad, width_px  (predict_grasp.py output).
    Returns a dict with the grasp in the base frame:
        position (x,y,z), yaw_rad/deg about the plane normal, width_m,
        approach (unit, into the table), T_base_grasp (4x4).
    """
    n = cfg['plane_normal']
    plane_point = cfg['plane_point'] + object_height * n

    u, v = float(grasp['x']), float(grasp['y'])
    a = float(grasp['angle_rad'])
    du, dv = np.cos(a), np.sin(a)                    # grasp-line direction in image
    pu, pv = -np.sin(a), np.cos(a)                   # perpendicular (width) direction

    p_c = _project((u, v), cfg, plane_point)
    if p_c is None:
        raise ValueError('grasp centre ray does not hit the table plane -- check T_base_cam / plane')

    p_a = _project((u + delta_px * du, v + delta_px * dv), cfg, plane_point)
    p_b = _project((u - delta_px * du, v - delta_px * dv), cfg, plane_point)
    axis = (p_a - p_b)
    axis = axis / (np.linalg.norm(axis) + 1e-12)     # grasp closing axis in base frame

    w1 = _project((u + grasp['width_px'] / 2 * pu, v + grasp['width_px'] / 2 * pv), cfg, plane_point)
    w2 = _project((u - grasp['width_px'] / 2 * pu, v - grasp['width_px'] / 2 * pv), cfg, plane_point)
    width_m = float(np.linalg.norm(w1 - w2)) if (w1 is not None and w2 is not None) else float('nan')

    approach = -n / np.linalg.norm(n)               # gripper comes straight down onto the plane

    # build a right-handed grasp frame: z = approach, x = closing axis (orthogonalised), y = z x x
    z_ax = approach
    x_ax = axis - (axis @ z_ax) * z_ax
    x_ax = x_ax / (np.linalg.norm(x_ax) + 1e-12)
    y_ax = np.cross(z_ax, x_ax)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x_ax, y_ax, z_ax, p_c

    # yaw of the closing axis around the plane normal, measured from base +x
    yaw = float(np.arctan2(axis[1], axis[0]))

    return {
        'position': p_c.tolist(),
        'yaw_rad': yaw,
        'yaw_deg': float(np.degrees(yaw)),
        'width_m': width_m,
        'approach': approach.tolist(),
        'closing_axis': axis.tolist(),
        'T_base_grasp': T.tolist(),
        'quality': float(grasp.get('quality', float('nan'))),
        'object_height': object_height,
    }


# ---------------------------------------------------------------------- selftest

def _selftest():
    """Synthetic camera 0.55 m above the table looking straight down; centre pixel -> table origin."""
    cfg = _to_np({
        "K": [[900, 0, 640], [0, 900, 360], [0, 0, 1]],
        "dist": [0, 0, 0, 0, 0],
        "T_base_cam": [[1, 0, 0, 0.0], [0, -1, 0, 0.0], [0, 0, -1, 0.55], [0, 0, 0, 1]],
        "table_plane": {"point": [0, 0, 0], "normal": [0, 0, 1]},
    })
    # a horizontal grasp (angle 0) at the principal point, 90 px wide
    g = {'x': 640, 'y': 360, 'angle_rad': 0.0, 'width_px': 90.0, 'quality': 0.9}
    out = grasp_to_base(g, cfg)
    pos = np.array(out['position'])
    print('centre pixel  -> base position', np.round(pos, 4), '(expect ~[0,0,0])')
    print('width 90 px   -> width_m', round(out['width_m'], 4),
          '(expect ~90 * 0.55 / 900 = %.4f)' % (90 * 0.55 / 900))
    print('yaw_deg', round(out['yaw_deg'], 2))
    assert np.allclose(pos, [0, 0, 0], atol=1e-3), pos
    assert abs(out['width_m'] - 90 * 0.55 / 900) < 1e-3
    # offset pixel should move along -x in base (camera y is flipped)
    g2 = dict(g, x=740)
    p2 = np.array(grasp_to_base(g2, cfg)['position'])
    print('pixel +100 u  -> base position', np.round(p2, 4))
    assert p2[0] > 0.05 and abs(p2[1]) < 1e-3
    print('selftest OK')


# --------------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', help='camera + table-plane JSON')
    p.add_argument('--make-template', metavar='PATH', help='write a blank config to PATH and exit')
    p.add_argument('--selftest', action='store_true', help='run the synthetic sanity check and exit')
    p.add_argument('--grasp-json', help='JSON from predict_grasp.py --json (list or single dict)')
    p.add_argument('--u', type=float); p.add_argument('--v', type=float)
    p.add_argument('--angle-deg', type=float, default=0.0)
    p.add_argument('--width-px', type=float, default=60.0)
    p.add_argument('--quality', type=float, default=float('nan'))
    p.add_argument('--object-height', type=float, default=0.0,
                   help='height (m) of the grasp contact above the table plane')
    args = p.parse_args()

    if args.make_template:
        write_template_config(args.make_template)
        return
    if args.selftest:
        _selftest()
        return
    if not args.config:
        p.error('--config is required (or use --make-template / --selftest)')

    cfg = load_config(args.config)

    if args.grasp_json:
        with open(args.grasp_json) as f:
            data = json.load(f)
        grasps = data if isinstance(data, list) else [data]
    elif args.u is not None and args.v is not None:
        grasps = [{'x': args.u, 'y': args.v, 'angle_rad': np.radians(args.angle_deg),
                   'width_px': args.width_px, 'quality': args.quality}]
    else:
        p.error('give --grasp-json or --u/--v (+ --angle-deg --width-px)')

    for i, g in enumerate(grasps):
        if 'angle_rad' not in g and 'angle_deg' in g:
            g['angle_rad'] = np.radians(g['angle_deg'])
        out = grasp_to_base(g, cfg, object_height=args.object_height)
        x, y, z = out['position']
        print('grasp %d: base xyz = (%.4f, %.4f, %.4f) m  yaw = %.1f deg  width = %.3f m  q = %.3f'
              % (i, x, y, z, out['yaw_deg'], out['width_m'], out['quality']))
        print('         T_base_grasp =')
        for row in out['T_base_grasp']:
            print('           [%9.4f %9.4f %9.4f %9.4f]' % tuple(row))


if __name__ == '__main__':
    main()
