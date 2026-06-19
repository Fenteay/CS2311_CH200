"""
Evaluate a saved source model checkpoint with optional TTA.
Usage:
    python eval_tta.py --model models/leafandmask_full_resunet/model.pth \
                       --arch ResUNet --dataset leafandmask_full --tta \
                       --threshold-search --scale-tta \
                       --export-dir results/leafandmask_full_resunet_tta_eval
"""
import os, argparse
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import albumentations as A
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from train_source import (
    WeightedBCEDiceLoss, filter_ids_with_masks,
    binarize_with_collapse_fallback, postprocess_mask,
    tta_predict,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to model.pth checkpoint')
    p.add_argument('--arch', default='ResUNet', choices=sorted(archs.__all__))
    p.add_argument('--dataset', default='leafandmask_full')
    p.add_argument('--img-ext', default='.jpg', dest='img_ext')
    p.add_argument('--img-size', default=512, type=int, dest='img_size')
    p.add_argument('--batch-size', default=4, type=int, dest='batch_size')
    p.add_argument('--pred-threshold', default=0.45, type=float)
    p.add_argument('--pos-weight', default=2.0, type=float)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--scale-tta', action='store_true', dest='scale_tta',
                   help='Add multi-scale TTA (0.875x, 1.0x, 1.125x) on top of flip TTA')
    p.add_argument('--threshold-search', action='store_true', dest='threshold_search',
                   help='Grid-search threshold from 0.30 to 0.55 and report best Dice')
    p.add_argument('--export-dir', default=None)
    p.add_argument('--morph-close', default=5, type=int)
    p.add_argument('--morph-open', default=3, type=int)
    p.add_argument('--min-area', default=100, type=int)
    return p.parse_args()


def tta_predict_multiscale(model, input, use_amp, use_channels_last, scales=(0.875, 1.0, 1.125)):
    """Flip TTA at multiple scales, bilinear-resize back to original size."""
    orig_h, orig_w = input.shape[2], input.shape[3]
    all_preds = []
    for scale in scales:
        new_h = int(round(orig_h * scale / 32) * 32)
        new_w = int(round(orig_w * scale / 32) * 32)
        inp_scaled = F.interpolate(input, size=(new_h, new_w), mode='bilinear', align_corners=False)
        pred = tta_predict(model, inp_scaled, use_amp, use_channels_last)
        pred_orig = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        all_preds.append(pred_orig)
    return torch.stack(all_preds, dim=0).mean(dim=0)


def collect_all_probs(loader, model, use_amp, use_channels_last, tta, scale_tta):
    """Collect all predicted probabilities and targets for threshold search."""
    all_probs = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for inp, target, _ in tqdm(loader, desc='Collecting probs'):
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if scale_tta:
                prob = tta_predict_multiscale(model, inp, use_amp, use_channels_last)
            elif tta:
                prob = tta_predict(model, inp, use_amp, use_channels_last)
            else:
                if use_channels_last:
                    inp = inp.contiguous(memory_format=torch.channels_last)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    prob = torch.sigmoid(model(inp))
            all_probs.append(prob.cpu())
            all_targets.append(target.cpu())
    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def dice_at_threshold(probs, targets, threshold):
    smooth = 1e-5
    pred = (probs > threshold).float()
    num = pred.shape[0]
    pred_flat = pred.view(num, -1)
    tgt_flat = targets.view(num, -1)
    inter = (pred_flat * tgt_flat).sum(1)
    dice = ((2 * inter + smooth) / (pred_flat.sum(1) + tgt_flat.sum(1) + smooth)).mean().item()
    iou = ((inter + smooth) / (pred_flat.sum(1) + tgt_flat.sum(1) - inter + smooth)).mean().item()
    return dice, iou


def main():
    args = parse_args()
    cudnn.benchmark = True
    use_amp = torch.cuda.is_available()
    use_channels_last = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build test dataset
    img_dir = os.path.join('inputs', 'inputs', args.dataset, 'test', 'images')
    mask_dir = os.path.join('inputs', 'inputs', args.dataset, 'test', 'masks')
    img_paths = glob(os.path.join(img_dir, '*' + args.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    img_ids, _ = filter_ids_with_masks(img_ids, mask_dir, '.png', 1)
    test_samples = [
        {'img_id': i, 'save_id': i, 'img_dir': img_dir, 'mask_dir': mask_dir,
         'img_ext': args.img_ext, 'mask_ext': '.png'}
        for i in img_ids
    ]
    print(f"Loaded {len(test_samples)} test samples from {args.dataset}")

    transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(),
    ])
    dataset = Dataset(
        img_ids=test_samples, img_dir=None, mask_dir=None,
        img_ext=args.img_ext, mask_ext='.png', num_classes=1,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    # Load model
    print(f"Loading model {args.arch} from {args.model}...")
    model = archs.__dict__[args.arch](1, 3, False)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    criterion = WeightedBCEDiceLoss(pos_weight=args.pos_weight).to(device)

    export_dir = args.export_dir or f'results/{args.dataset}_{args.arch.lower()}_{"tta" if args.tta else "notta"}_eval'
    os.makedirs(export_dir, exist_ok=True)

    # Collect all probabilities first (for threshold search or direct eval)
    mode_label = 'scale-TTA' if args.scale_tta else ('TTA' if args.tta else 'no-TTA')
    print(f"Running inference with {mode_label}...")
    all_probs, all_targets = collect_all_probs(loader, model, use_amp, use_channels_last, args.tta, args.scale_tta)

    # Threshold search
    best_threshold = args.pred_threshold
    best_dice = 0.0
    if args.threshold_search:
        print("\nSearching best threshold...")
        thresholds = [round(t, 2) for t in np.arange(0.25, 0.60, 0.01)]
        results = []
        for t in thresholds:
            d, iou = dice_at_threshold(all_probs, all_targets, t)
            results.append((t, d, iou))
            if d > best_dice:
                best_dice = d
                best_threshold = t
        # Print top 5
        results.sort(key=lambda x: -x[1])
        print(f"\n{'Threshold':>10} {'Dice':>8} {'IoU':>8}")
        for t, d, iou in results[:5]:
            marker = ' <-- best' if t == best_threshold else ''
            print(f"  {t:>8.2f}   {d:.4f}   {iou:.4f}{marker}")
        print()

    # Final metrics at chosen threshold
    final_dice, final_iou = dice_at_threshold(all_probs, all_targets, best_threshold)
    print(f"Final results ({mode_label}, threshold={best_threshold:.2f}):")
    print(f"  iou={final_iou:.4f}, dice={final_dice:.4f}")

    metrics_path = os.path.join(export_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"dataset: {args.dataset}\narch: {args.arch}\n")
        f.write(f"mode: {mode_label}\n")
        f.write(f"threshold: {best_threshold:.4f}\n")
        f.write(f"iou: {final_iou:.6f}\n")
        f.write(f"dice: {final_dice:.6f}\n")
    print(f"Saved to {metrics_path}")


if __name__ == '__main__':
    main()

