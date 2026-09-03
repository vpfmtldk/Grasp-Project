"""
Single-image grasp inference for GG-CNN / GG-CNN2.

This is the pipeline-integration entry point: give it one camera image (RGB *or*
depth) and it returns the predicted antipodal grasp(s) in the ORIGINAL image's
pixel coordinates -- centre (x, y), rotation angle, gripper width, quality.

The image-space grasp is later mapped to robot coordinates with a separate
hand-eye calibration step; that is deliberately not done here.

Examples
--------
# RGB-only model (matches a depth-sensor-free camera setup)
python predict_grasp.py --network output/models/<run>/epoch_XX_iou_0.XX \
    --use-rgb 1 --use-depth 0 --image path/to/frame.png --vis

# depth-only model
python predict_grasp.py --network output/models/<run>/epoch_XX_iou_0.XX \
    --use-rgb 0 --use-depth 1 --image path/to/depth.tiff --vis
"""
import argparse

import numpy as np
import torch
from imageio.v2 import imread
from skimage.transform import resize as sk_resize

from eval_ggcnn import load_network
from models.common import post_process_output
from utils.dataset_processing.grasp import detect_grasps

IMG_SIZE = 300  # GG-CNN input/output resolution


def center_crop(img, crop_size=IMG_SIZE):
    """
    Return (crop, top, left, scale) where the model's 300x300 view maps back to
    the original image as:  orig = (top, left) + pixel * scale.

    crop_size > 0 and both image dims >= crop_size:
        take a native crop_size x crop_size centre window (scale = crop_size/300),
        which matches how Cornell training crops (no down-scaling of the object).
    otherwise:
        take the largest centred square and resize it to 300 (scale = side/300).
    """
    h, w = img.shape[:2]
    if crop_size and h >= crop_size and w >= crop_size:
        side = crop_size
    else:
        side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    crop = img[top:top + side, left:left + side]
    return crop, top, left, side / IMG_SIZE


def _to_size(img):
    return sk_resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True).astype(np.float32)


def preprocess_rgb(rgb, crop_size=IMG_SIZE):
    """rgb: HxWx3 uint8 (RGB). Returns (1x3x300x300 tensor, (top, left, scale))."""
    crop, top, left, scale = center_crop(rgb, crop_size)
    crop = _to_size(crop) / 255.0
    crop = crop - crop.mean()                       # same as utils Image.normalise()
    x = torch.from_numpy(crop.transpose(2, 0, 1)[np.newaxis].copy())
    return x, (top, left, scale)


def preprocess_depth(depth, crop_size=IMG_SIZE):
    """depth: HxW float. Returns (1x1x300x300 tensor, (top, left, scale))."""
    crop, top, left, scale = center_crop(depth.astype(np.float32), crop_size)
    crop = _to_size(crop)
    crop = np.clip(crop - crop.mean(), -1, 1)       # same as DepthImage.normalise()
    x = torch.from_numpy(crop[np.newaxis, np.newaxis].copy())
    return x, (top, left, scale)


class GraspPredictor:
    """Loads a trained network once and predicts grasps for individual images."""

    def __init__(self, network_path, use_rgb=True, use_depth=False, device=None, crop_size=IMG_SIZE):
        if use_rgb == use_depth:
            raise ValueError('Choose exactly one of use_rgb / use_depth (RGB-only or depth-only).')
        self.use_rgb = bool(use_rgb)
        self.use_depth = bool(use_depth)
        self.crop_size = crop_size
        input_channels = 3 * self.use_rgb + 1 * self.use_depth
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.net = load_network(network_path, input_channels=input_channels).to(self.device)
        self.net.eval()

    def _forward(self, x):
        with torch.no_grad():
            out = self.net(x.to(self.device))
        if isinstance(out, dict):                    # some checkpoints wrap the outputs
            out = (out['pos'], out['cos'], out['sin'], out['width'])
        return post_process_output(*out)

    def predict(self, image, n_grasps=1):
        """
        image: HxWx3 uint8 RGB (if use_rgb) or HxW float depth (if use_depth).
        Returns a list of dicts, best first:
            {x, y, angle_rad, angle_deg, width_px, quality}
        in the coordinates of the *input* image.
        """
        if self.use_rgb:
            x, (off_y, off_x, scale) = preprocess_rgb(np.asarray(image), self.crop_size)
        else:
            x, (off_y, off_x, scale) = preprocess_depth(np.asarray(image), self.crop_size)

        q_img, ang_img, width_img = self._forward(x)
        grasps = detect_grasps(q_img, ang_img, width_img, no_grasps=n_grasps)

        results = []
        for g in grasps:
            row, col = int(g.center[0]), int(g.center[1])
            results.append({
                'x': off_x + col * scale,
                'y': off_y + row * scale,
                'angle_rad': float(g.angle),
                'angle_deg': float(np.degrees(g.angle)),
                'width_px': float(g.length * scale),
                'quality': float(q_img[row, col]),
            })
        return results


def grasp_corners(cx, cy, angle, length, width):
    """4 corners (as (x, y)) of an oriented grasp rectangle -- mirrors grasp.Grasp.as_gr."""
    xo, yo = np.cos(angle), np.sin(angle)
    y1, x1 = cy + length / 2 * yo, cx - length / 2 * xo
    y2, x2 = cy - length / 2 * yo, cx + length / 2 * xo
    return np.array([
        [x1 - width / 2 * yo, y1 - width / 2 * xo],
        [x2 - width / 2 * yo, y2 - width / 2 * xo],
        [x2 + width / 2 * yo, y2 + width / 2 * xo],
        [x1 + width / 2 * yo, y1 + width / 2 * xo],
    ])


def visualise(base_img, grasps, out_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(base_img, cmap=None if base_img.ndim == 3 else 'gray')
    for i, g in enumerate(grasps):
        c = grasp_corners(g['x'], g['y'], g['angle_rad'], g['width_px'], g['width_px'] / 2)
        c = np.vstack([c, c[0]])
        ax.plot(c[:, 0], c[:, 1], color='r' if i == 0 else 'y', linewidth=2)
        ax.plot(g['x'], g['y'], 'o', color='r' if i == 0 else 'y')
        ax.text(g['x'] + 5, g['y'] - 5, "%.0f deg / q=%.2f" % (g['angle_deg'], g['quality']),
                color='r' if i == 0 else 'y', fontsize=9)
    ax.set_title('GG-CNN grasp prediction')
    ax.axis('off')
    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=120)
        print('saved', out_path)
    else:
        plt.show()
    plt.close(fig)


def load_image(path, use_depth):
    img = imread(path)
    if use_depth:
        return np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return np.asarray(img[..., :3], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description='Predict grasp(s) for a single image with GG-CNN.')
    p.add_argument('--network', required=True, help='Path to trained network (full model, statedict, or ckpt_last.pt)')
    p.add_argument('--image', required=True, help='Path to an RGB image or a depth image')
    p.add_argument('--use-rgb', type=int, default=1)
    p.add_argument('--use-depth', type=int, default=0)
    p.add_argument('--n-grasps', type=int, default=1)
    p.add_argument('--crop-size', type=int, default=IMG_SIZE,
                   help='Native centre-crop size (default 300, matches Cornell). 0 = largest square then resize.')
    p.add_argument('--vis', action='store_true', help='Show / save an overlay of the predicted grasp')
    p.add_argument('--out', type=str, default='', help='Save the overlay here instead of showing it')
    p.add_argument('--json', type=str, default='', help='Write the grasp dict(s) to this JSON file (feeds pixel_to_world.py)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predictor = GraspPredictor(args.network, use_rgb=args.use_rgb, use_depth=args.use_depth,
                               crop_size=args.crop_size)
    img = load_image(args.image, use_depth=bool(args.use_depth))
    grasps = predictor.predict(img, n_grasps=args.n_grasps)

    for i, g in enumerate(grasps):
        print('grasp %d: x=%.1f y=%.1f  angle=%.1f deg  width=%.1f px  quality=%.3f'
              % (i, g['x'], g['y'], g['angle_deg'], g['width_px'], g['quality']))

    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump(grasps if len(grasps) != 1 else grasps[0], f, indent=2)
        print('wrote', args.json)

    if args.vis or args.out:
        base = img.astype(np.uint8) if not args.use_depth else img
        visualise(base, grasps, out_path=(args.out or None))
