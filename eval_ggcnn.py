import argparse
import logging

import torch.utils.data

from models import get_network
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset

logging.basicConfig(level=logging.INFO)


def _v(t):
    """Unwrap a 1-element tensor/array to a Python scalar (DataLoader collates idx/rot/zoom)."""
    return t.item() if hasattr(t, 'item') else t


def _infer_arch(state):
    """Guess the network name from state_dict keys (grconvnet has residual blocks + bn6)."""
    keys = set(state.keys())
    if any(k.startswith('res1.') for k in keys) or 'bn6.weight' in keys:
        return 'grconvnet'
    if 'convt1.weight' in keys:
        return 'ggcnn'
    return None


def load_network(path, input_channels=1, network=None):
    """Load a network saved either as a full pickled model, a plain state_dict, or a training checkpoint dict.
    network=None -> infer the architecture from the state_dict keys (falls back to 'ggcnn')."""
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        return loaded
    state = loaded['model_state_dict'] if isinstance(loaded, dict) and 'model_state_dict' in loaded else loaded
    name = network or _infer_arch(state) or 'ggcnn'
    net = get_network(name)(input_channels=input_channels)
    net.load_state_dict(state)
    return net


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Load Network
    device = torch.device("cuda:0")
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    net = load_network(args.network, input_channels=input_channels)
    net = net.to(device)
    net.eval()

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                           random_rotate=args.augment, random_zoom=args.augment,
                           include_depth=args.use_depth, include_rgb=args.use_rgb)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img,
                                                   test_data.dataset.get_gtbb(_v(didx), _v(rot), _v(zoom)),
                                                   no_grasps=args.n_grasps,
                                                   grasp_width=width_img,
                                                   )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if args.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if args.vis:
                evaluation.plot_output(test_data.dataset.get_rgb(_v(didx), _v(rot), _v(zoom), normalise=False),
                                       test_data.dataset.get_depth(_v(didx), _v(rot), _v(zoom)), q_img,
                                       ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))