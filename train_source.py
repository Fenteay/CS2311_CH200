import os
import argparse
import yaml
from glob import glob
from tqdm import tqdm
import albumentations as A
import cv2
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset import Dataset

import archs
import losses
from metrics import iou_score
from utils import AverageMeter
from collections import OrderedDict


class WeightedBCEDiceLoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = float(max(1.0, pos_weight))
        self.register_buffer('pos_weight_tensor', torch.tensor([self.pos_weight], dtype=torch.float32))

    def forward(self, input, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=self.pos_weight_tensor,
        )
        smooth = 1e-5
        prob = torch.sigmoid(input)
        num = target.size(0)
        prob = prob.view(num, -1)
        target = target.view(num, -1)
        intersection = prob * target
        dice = (2.0 * intersection.sum(1) + smooth) / (prob.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.mean()
        return 0.5 * bce + dice


def filter_ids_with_masks(img_ids, mask_dir, mask_ext, num_classes):
    valid_ids = []
    missing_examples = []

    for img_id in img_ids:
        ok = True
        for cls_idx in range(num_classes):
            mask_path = os.path.join(mask_dir, str(cls_idx), img_id + mask_ext)
            if not os.path.exists(mask_path):
                ok = False
                if len(missing_examples) < 5:
                    missing_examples.append(mask_path)
                break
        if ok:
            valid_ids.append(img_id)

    return valid_ids, missing_examples


def compute_binary_mask_ratio(img_ids, mask_dir, mask_ext):
    fg_pixels = 0
    total_pixels = 0

    for sample in img_ids:
        if isinstance(sample, dict):
            img_id = sample['img_id']
            sample_mask_dir = sample.get('mask_dir', mask_dir)
            sample_mask_ext = sample.get('mask_ext', mask_ext)
        else:
            img_id = sample
            sample_mask_dir = mask_dir
            sample_mask_ext = mask_ext

        mask_path = os.path.join(sample_mask_dir, '0', img_id + sample_mask_ext)
        mask_i = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_i is None:
            continue
        fg_pixels += int((mask_i > 0).sum())
        total_pixels += int(mask_i.size)

    if total_pixels == 0:
        return 0.0
    return fg_pixels / total_pixels


def postprocess_mask(mask_uint8, morph_close=5, morph_open=3, min_area=100):
    """Apply morphological cleanup and remove tiny components."""
    result = mask_uint8.copy()
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k)
    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k)
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_area:
                result[labels == lbl] = 0
    return result


def binarize_with_collapse_fallback(pred_prob, pred_threshold, expected_fg_ratio=None):
    pred_mask = (pred_prob > pred_threshold).float()
    fg_ratio = pred_mask.mean().item()

    # Prevent degenerate all-black/all-white exports by switching to ratio-based top-k thresholding.
    if expected_fg_ratio is not None and 0.0 < expected_fg_ratio < 1.0 and (fg_ratio <= 1e-4 or fg_ratio >= 1 - 1e-4):
        ratio = float(np.clip(expected_fg_ratio, 1e-4, 1 - 1e-4))
        flat_prob = pred_prob.reshape(-1)
        k = int(round(ratio * flat_prob.numel()))
        k = max(1, min(flat_prob.numel() - 1, k))
        topk_vals, _ = torch.topk(flat_prob, k)
        ratio_threshold = topk_vals[-1]
        pred_mask = (pred_prob >= ratio_threshold).float()

    return pred_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='hrf', help='dataset name (e.g., hrf, rite, chase)')
    parser.add_argument('--arch', default='UNet', choices=sorted(archs.__all__),
                        help='Segmentation architecture to train.')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--cosine-lr', action='store_true',
                        help='Use CosineAnnealingLR scheduler (recommended).')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--img-ext', default='.png', dest='img_ext',
                        help='Image file extension (default: .png). Use .jpg for greenhouse_clean.')
    parser.add_argument('--missing-mask-strategy', default='error', choices=['error', 'zeros'],
                        help='How to handle missing masks: error (strict) or zeros (fallback for smoke tests).')
    parser.add_argument('--export-dir', default=None,
                        help='Directory to save predicted test masks and metrics. Default: results/<dataset>_source_eval')
    parser.add_argument('--pred-threshold', default=0.5, type=float,
                        help='Threshold on sigmoid probability for binarizing predicted masks.')
    parser.add_argument('--auto-pos-weight', action='store_true',
                        help='Enable automatic positive class weighting from training mask ratio.')
    parser.add_argument('--pos-weight', default=1.0, type=float,
                        help='Manual positive class weight for BCE term (used when --auto-pos-weight is disabled).')
    parser.add_argument('--max-pos-weight', default=20.0, type=float,
                        help='Upper bound used when auto-computing positive class weight.')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of DataLoader workers. Use 0 on Windows to avoid worker spawn stalls.')
    parser.add_argument('--disable-amp', action='store_true',
                        help='Disable automatic mixed precision on CUDA (enabled by default).')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Force pin_memory=True for DataLoader (auto-enabled on CUDA by default).')
    parser.add_argument('--no-persistent-workers', action='store_true',
                        help='Disable persistent workers when num_workers > 0.')
    parser.add_argument('--morph-close', default=5, type=int,
                        help='Kernel size for morphological closing to fill holes (0 = disabled).')
    parser.add_argument('--morph-open', default=3, type=int,
                        help='Kernel size for morphological opening to remove noise (0 = disabled).')
    parser.add_argument('--min-area', default=100, type=int,
                        help='Remove connected components with area < this value (0 = disabled).')
    args = parser.parse_args()
    return args


def sanitize_name(name):
    return name.replace('+', '_plus_').replace(',', '_').replace('/', '_').replace(' ', '')


def parse_dataset_names(dataset_arg):
    dataset_names = [item.strip() for item in dataset_arg.replace(',', '+').split('+') if item.strip()]
    if not dataset_names:
        raise ValueError('At least one dataset name must be provided.')
    return dataset_names


def build_split_samples(dataset_names, split_name, img_ext, mask_ext, num_classes, missing_mask_strategy):
    samples = []

    for dataset_name in dataset_names:
        img_dir = os.path.join('inputs', 'inputs', dataset_name, split_name, 'images')
        mask_dir = os.path.join('inputs', 'inputs', dataset_name, split_name, 'masks')
        img_paths = glob(os.path.join(img_dir, '*' + img_ext))
        img_ids = [os.path.splitext(os.path.basename(path))[0] for path in img_paths]

        if missing_mask_strategy == 'error':
            img_ids, missing_examples = filter_ids_with_masks(
                img_ids,
                mask_dir,
                mask_ext,
                num_classes,
            )
            if len(img_ids) == 0:
                example_text = '\n'.join(missing_examples) if missing_examples else '(no examples available)'
                raise RuntimeError(
                    f"No valid {split_name} samples found for dataset '{dataset_name}'.\n"
                    f"Expected masks under: {mask_dir}\\<class_id>\\<image_id>{mask_ext}\n"
                    f"Examples of missing masks:\n{example_text}"
                )

        prefix = sanitize_name(dataset_name)
        dataset_samples = [
            {
                'img_id': img_id,
                'save_id': f'{prefix}__{img_id}',
                'img_dir': img_dir,
                'mask_dir': mask_dir,
                'img_ext': img_ext,
                'mask_ext': mask_ext,
            }
            for img_id in img_ids
        ]
        samples.extend(dataset_samples)
        print(f"Loaded {len(dataset_samples)} {split_name} samples from {dataset_name}.")

    if not samples:
        raise RuntimeError(f'No {split_name} samples found for datasets: {dataset_names}')

    return samples

def train(train_loader, model, criterion, optimizer, use_amp=False, scaler=None, use_channels_last=False):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target, _ in train_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if use_channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(input)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        iou, dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return avg_meters


def evaluate_and_export(test_loader, model, criterion, output_dir, pred_threshold=0.5, expected_fg_ratio=None, morph_close=5, morph_open=3, min_area=100, use_amp=False, use_channels_last=False):
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'pred_fg_ratio': AverageMeter(),
        'pred_prob_mean': AverageMeter(),
    }

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), desc='Evaluating')
        for input, target, meta in test_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if use_channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(input)
                loss = criterion(output, target)
            iou, dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            pred_prob = torch.sigmoid(output)
            pred_mask = binarize_with_collapse_fallback(
                pred_prob,
                pred_threshold,
                expected_fg_ratio=expected_fg_ratio,
            )
            avg_meters['pred_fg_ratio'].update(pred_mask.mean().item(), input.size(0))
            avg_meters['pred_prob_mean'].update(pred_prob.mean().item(), input.size(0))
            mask_np = pred_mask[0].squeeze().cpu().numpy()
            final_mask = (mask_np * 255).astype(np.uint8)
            if morph_close > 0 or morph_open > 0 or min_area > 0:
                final_mask = postprocess_mask(final_mask, morph_close=morph_close, morph_open=morph_open, min_area=min_area)

            img_id = meta['img_id'][0] if isinstance(meta['img_id'], list) else str(meta['img_id'])
            save_path = os.path.join(output_dir, f"{img_id}.png")
            cv2.imwrite(save_path, final_mask)

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('fg', avg_meters['pred_fg_ratio'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('pred_fg_ratio', avg_meters['pred_fg_ratio'].avg),
        ('pred_prob_mean', avg_meters['pred_prob_mean'].avg),
    ])

def main():
    args = parse_args()
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    # Common Configuration
    config = {
        'arch': args.arch,
        'num_classes': 1,
        'input_channels': 3,
        'deep_supervision': False,
        'name': f"{sanitize_name(args.dataset)}_{args.arch.lower()}",
        'img_ext': args.img_ext,
        'mask_ext': '.png',
        'input_h': 512,
        'input_w': 512,
        'num_workers': args.num_workers,
        'use_amp': torch.cuda.is_available() and not args.disable_amp,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'loss': 'BCEDiceLoss',
        'stage1': 15,
        'stage2': 15
    }

    dataset_names = parse_dataset_names(args.dataset)
    print(f"Loading dataset(s): {', '.join(dataset_names)}")

    train_samples = build_split_samples(
        dataset_names,
        'train',
        config['img_ext'],
        config['mask_ext'],
        config['num_classes'],
        args.missing_mask_strategy,
    )

    if args.missing_mask_strategy != 'error':
        print('Missing masks will be replaced with zero masks (fallback mode).')

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=60, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_samples,
        img_dir=None,
        mask_dir=None,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform,
        missing_mask_strategy=args.missing_mask_strategy,
    )

    pin_memory = args.pin_memory or torch.cuda.is_available()
    use_persistent_workers = (config['num_workers'] > 0) and (not args.no_persistent_workers)
    train_loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': config['num_workers'],
        'drop_last': True,
        'pin_memory': pin_memory,
    }
    if config['num_workers'] > 0:
        train_loader_kwargs['persistent_workers'] = use_persistent_workers

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )

    test_samples = build_split_samples(
        dataset_names,
        'test',
        config['img_ext'],
        config['mask_ext'],
        config['num_classes'],
        args.missing_mask_strategy,
    )

    test_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=test_samples,
        img_dir=None,
        mask_dir=None,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform,
        missing_mask_strategy=args.missing_mask_strategy,
    )

    test_loader_kwargs = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': config['num_workers'],
        'drop_last': False,
        'pin_memory': pin_memory,
    }
    if config['num_workers'] > 0:
        test_loader_kwargs['persistent_workers'] = use_persistent_workers

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **test_loader_kwargs,
    )

    print("Creating model %s..." % config['arch'])
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision']
    )
    model = model.to(device)
    use_channels_last = torch.cuda.is_available()
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.auto_pos_weight:
        train_fg_ratio = compute_binary_mask_ratio(
            train_samples,
            None,
            config['mask_ext'],
        )
        if train_fg_ratio <= 0.0:
            pos_weight = 1.0
        else:
            pos_weight = (1.0 - train_fg_ratio) / train_fg_ratio
        pos_weight = min(max(1.0, pos_weight), args.max_pos_weight)
        print(f"Auto pos_weight enabled. train_fg_ratio={train_fg_ratio:.6f}, pos_weight={pos_weight:.4f}")
    else:
        train_fg_ratio = compute_binary_mask_ratio(
            train_samples,
            None,
            config['mask_ext'],
        )
        pos_weight = max(1.0, args.pos_weight)
        print(f"Manual pos_weight={pos_weight:.4f}, observed train_fg_ratio={train_fg_ratio:.6f}")

    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp']) if torch.cuda.is_available() else None
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=config['lr'] * 0.01) if args.cosine_lr else None

    print(
        f"Speed settings: amp={config['use_amp']}, channels_last={use_channels_last}, "
        f"pin_memory={pin_memory}, persistent_workers={use_persistent_workers}, num_workers={config['num_workers']}"
    )

    if scheduler:
        print(f"Using CosineAnnealingLR: lr {config['lr']:.2e} -> {config['lr']*0.01:.2e} over {args.epochs} epochs")
    print(f"Training Source Model ({args.arch}) on {args.dataset} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}] lr={current_lr:.2e}")
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            use_amp=config['use_amp'],
            scaler=scaler,
            use_channels_last=use_channels_last,
        )
        if scheduler:
            scheduler.step()

    export_dir = args.export_dir or os.path.join('results', f'{sanitize_name(args.dataset)}_{args.arch.lower()}_source_eval')
    print(f"Evaluating source model on test set ({len(test_samples)} images)...")
    eval_log = evaluate_and_export(
        test_loader,
        model,
        criterion,
        export_dir,
        pred_threshold=args.pred_threshold,
        expected_fg_ratio=train_fg_ratio,
        morph_close=args.morph_close,
        morph_open=args.morph_open,
        min_area=args.min_area,
        use_amp=config['use_amp'],
        use_channels_last=use_channels_last,
    )

    metrics_path = os.path.join(export_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"arch: {args.arch}\n")
        f.write(f"test_samples: {len(test_samples)}\n")
        f.write(f"loss: {eval_log['loss']:.6f}\n")
        f.write(f"iou: {eval_log['iou']:.6f}\n")
        f.write(f"dice: {eval_log['dice']:.6f}\n")
        f.write(f"pred_fg_ratio_avg: {eval_log['pred_fg_ratio']:.6f}\n")
        f.write(f"pred_prob_mean_avg: {eval_log['pred_prob_mean']:.6f}\n")
        f.write(f"train_fg_ratio: {train_fg_ratio:.6f}\n")
        f.write(f"pred_threshold: {args.pred_threshold:.4f}\n")
        f.write(f"pos_weight: {pos_weight:.6f}\n")
        f.write(f"use_amp: {config['use_amp']}\n")
        f.write(f"pin_memory: {pin_memory}\n")
        f.write(f"persistent_workers: {use_persistent_workers}\n")
        f.write(f"num_workers: {config['num_workers']}\n")

    # Save logic
    model_dir = os.path.join('models', config['name'])
    os.makedirs(model_dir, exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    
    # Save a generic config file so the target script works 
    # (Since tt_sfuda reads config_rite.yml, config_hrf.yml, etc based on the target)
    for target_dataset in ['rite', 'hrf', 'chase', 'leafandmask_trial', 'leafandmask_full']:
        config_path = os.path.join(model_dir, f'config_{target_dataset}.yml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
    print(f"Finished training. Model and configurations saved to {model_dir}/")
    print(f"Evaluation complete. Predicted masks and metrics saved to {export_dir}/")
    print(f"Test metrics: loss={eval_log['loss']:.4f}, iou={eval_log['iou']:.4f}, dice={eval_log['dice']:.4f}")
    print(f"You can now run SFUDA adaptation using: python tt_sfuda_2d.py --source {config['name']} --target <target_dataset>")

if __name__ == '__main__':
    main()
