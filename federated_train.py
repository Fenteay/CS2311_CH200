#!/usr/bin/env python3
"""
Simplified Federated Learning (FedAvg) for leaf segmentation.
Trains a single model using FedAvg algorithm with multiple clients (local data partitions).
"""

import argparse
import copy
import csv
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from train_source import (
    WeightedBCEDiceLoss,
    build_split_samples,
    compute_binary_mask_ratio,
    sanitize_name,
)
from utils import AverageMeter

try:
    import albumentations as A
except ImportError:
    A = None


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning (FedAvg) for leaf segmentation')
    parser.add_argument('--dataset', default='leafandmask_full',
                        help='Dataset name (folder in inputs/inputs/)')
    parser.add_argument('--arch', default='UNet', choices=sorted(archs.__all__),
                        help='Model architecture')
    parser.add_argument('--img-ext', default='.png', dest='img_ext',
                        help='Image file extension')
    parser.add_argument('--clients', default=3, type=int,
                        help='Number of federated clients')
    parser.add_argument('--rounds', default=20, type=int,
                        help='Number of federated rounds')
    parser.add_argument('--local-epochs', default=1, type=int,
                        help='Local epochs per round')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--auto-pos-weight', action='store_true',
                        help='Auto-compute pos_weight from train set')
    parser.add_argument('--pos-weight', default=1.0, type=float)
    parser.add_argument('--max-pos-weight', default=20.0, type=float)
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_iid(samples, num_clients, seed):
    """IID split: shuffle then round-robin assign."""
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    
    partitions = [[] for _ in range(num_clients)]
    for i, sample in enumerate(shuffled):
        partitions[i % num_clients].append(sample)
    return partitions


def make_loader(img_ids, img_ext, mask_ext, num_classes, batch_size, num_workers, shuffle=False):
    """Create DataLoader for a set of image IDs."""
    if A is not None:
        transform = A.Compose([
            A.RandomRotate90(p=0.5) if shuffle else A.NoOp(),
            A.HorizontalFlip(p=0.5) if shuffle else A.NoOp(),
            A.Resize(512, 512),
            A.Normalize(),
        ])
    else:
        transform = None
    
    ds = Dataset(
        img_ids=img_ids,
        img_dir=None,
        mask_dir=None,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=num_classes,
        transform=transform,
    )
    
    pin_memory = torch.cuda.is_available()
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle and len(img_ids) >= batch_size,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp):
    """Train model for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    with tqdm(total=len(loader), leave=False, desc='Training') as pbar:
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            iou, _ = iou_score(logits, y)
            loss_meter.update(loss.item(), x.size(0))
            iou_meter.update(iou, x.size(0))
            pbar.update(1)
    
    return loss_meter.avg, iou_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    """Evaluate model."""
    model.eval()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        
        iou, dice = iou_score(logits, y)
        loss_meter.update(loss.item(), x.size(0))
        iou_meter.update(iou, x.size(0))
        dice_meter.update(dice, x.size(0))
    
    return loss_meter.avg, iou_meter.avg, dice_meter.avg


def fedavg(client_params, client_weights):
    """Federated averaging of model parameters."""
    total_weight = sum(client_weights)
    if total_weight == 0:
        raise ValueError("Total weight is zero")
    
    avg_params = OrderedDict()
    for key in client_params[0].keys():
        ref_tensor = client_params[0][key]

        # Buffers such as BatchNorm's num_batches_tracked are integer tensors.
        # Keep them from the reference client to avoid float-to-long cast errors.
        if not torch.is_floating_point(ref_tensor):
            avg_params[key] = ref_tensor.clone()
            continue

        param_sum = torch.zeros_like(ref_tensor)
        for params, weight in zip(client_params, client_weights):
            param_sum += params[key] * (weight / total_weight)
        avg_params[key] = param_sum
    
    return avg_params


def main():
    args = parse_args()
    print(f"[Config] dataset={args.dataset}, arch={args.arch}, clients={args.clients}, rounds={args.rounds}")
    
    set_seed(args.seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    use_amp = torch.cuda.is_available() and not args.no_amp
    num_classes = 1
    mask_ext = '.png'
    
    # Load samples
    train_samples = build_split_samples(
        [args.dataset], 'train', args.img_ext, mask_ext, num_classes, 'error'
    )
    test_samples = build_split_samples(
        [args.dataset], 'test', args.img_ext, mask_ext, num_classes, 'error'
    )
    print(f"[Data] train: {len(train_samples)}, test: {len(test_samples)}")
    
    # Create client partitions
    partitions = split_iid(train_samples, args.clients, args.seed)
    for cid, part in enumerate(partitions):
        print(f"  Client {cid}: {len(part)} samples")
    
    # Compute pos_weight
    if args.auto_pos_weight:
        fg_ratio = compute_binary_mask_ratio(train_samples, None, mask_ext)
        pos_weight = min(max(1.0, (1.0 - fg_ratio) / max(fg_ratio, 1e-6)), args.max_pos_weight)
    else:
        fg_ratio = compute_binary_mask_ratio(train_samples, None, mask_ext)
        pos_weight = args.pos_weight
    print(f"[Training] pos_weight={pos_weight:.4f}, fg_ratio={fg_ratio:.4f}")
    
    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight).to(device)
    test_loader = make_loader(test_samples, args.img_ext, mask_ext, num_classes, 1, args.num_workers)
    
    # Initialize global model
    model = archs.__dict__[args.arch](num_classes, 3, False).to(device)
    history = []
    best_dice = -1
    best_state = copy.deepcopy(model.state_dict())
    
    # Federated training loop
    for rd in range(1, args.rounds + 1):
        print(f"\n[Round {rd}/{args.rounds}]")
        client_states = []
        client_weights = []
        round_loss = AverageMeter()
        round_iou = AverageMeter()
        
        for cid, client_samples in enumerate(partitions):
            if not client_samples:
                continue
            
            # Local training
            local_model = archs.__dict__[args.arch](num_classes, 3, False).to(device)
            local_model.load_state_dict(copy.deepcopy(model.state_dict()))
            
            train_loader = make_loader(
                client_samples, args.img_ext, mask_ext, num_classes,
                args.batch_size, args.num_workers, shuffle=True
            )
            
            optimizer = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            for epoch in range(args.local_epochs):
                loss, iou = train_one_epoch(local_model, train_loader, criterion, optimizer, device, use_amp)
            
            print(f"  Client {cid}: loss={loss:.4f}, iou={iou:.4f}")
            client_states.append(copy.deepcopy(local_model.state_dict()))
            client_weights.append(len(client_samples))
            round_loss.update(loss, len(client_samples))
            round_iou.update(iou, len(client_samples))
        
        # Global aggregation
        model.load_state_dict(fedavg(client_states, client_weights))
        
        # Evaluate
        val_loss, val_iou, val_dice = evaluate(model, test_loader, criterion, device, use_amp)
        print(f"  Global: loss={val_loss:.4f}, iou={val_iou:.4f}, dice={val_dice:.4f}")
        
        history.append({
            'round': rd,
            'train_loss': round_loss.avg,
            'train_iou': round_iou.avg,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
        })
        
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = copy.deepcopy(model.state_dict())
    
    # Save outputs
    model_name = f"{sanitize_name(args.dataset)}_{args.arch.lower()}_fedavg"
    model_dir = Path('models') / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(best_state, model_dir / 'model.pth')
    torch.save(model.state_dict(), model_dir / 'model_final.pth')
    
    # Save history
    with open(model_dir / 'federated_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'train_loss', 'train_iou', 'val_loss', 'val_iou', 'val_dice'])
        writer.writeheader()
        writer.writerows(history)
    
    # Save config
    config = {
        'arch': args.arch,
        'num_classes': 1,
        'input_channels': 3,
        'deep_supervision': False,
        'input_h': 512,
        'input_w': 512,
        'img_ext': args.img_ext,
        'mask_ext': mask_ext,
        'federated': {
            'algorithm': 'FedAvg',
            'clients': args.clients,
            'rounds': args.rounds,
            'local_epochs': args.local_epochs,
            'best_dice': float(best_dice),
        }
    }
    
    for target in ['rite', 'hrf', 'chase', 'leafandmask_trial', 'leafandmask_full']:
        with open(model_dir / f'config_{target}.yml', 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
    
    print(f"\n✅ Training complete!")
    print(f"   Best Dice: {best_dice:.4f}")
    print(f"   Model dir: {model_dir}")


if __name__ == '__main__':
    main()
