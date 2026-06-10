"""Shared utilities for Flower federated learning."""

import copy
import random

import albumentations as A
import numpy as np
import torch

import archs
from dataset import Dataset
from metrics import iou_score
from train_source import WeightedBCEDiceLoss
from utils import AverageMeter


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_samples_among_clients(img_ids, n_clients, num_classes=1):
    """IID data split via shuffle + round-robin."""
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    
    img_ids = list(img_ids)
    random.shuffle(img_ids)
    
    partitions = [[] for _ in range(n_clients)]
    for i, img_id in enumerate(img_ids):
        partitions[i % n_clients].append(img_id)
    
    return partitions


def make_loader(img_ids, img_dir, mask_dir, batch_size, num_workers, config, mode='train'):
    """Create a DataLoader for federated clients."""
    if mode == 'train':
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(config['input_h'], config['input_w']),
            A.Normalize(),
        ])
    else:
        transform = A.Compose([
            A.Resize(config['input_h'], config['input_w']),
            A.Normalize(),
        ])
    
    ds = Dataset(
        img_ids=img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=config.get('img_ext', '.png'),
        mask_ext=config.get('mask_ext', '.png'),
        num_classes=config.get('num_classes', 1),
        transform=transform,
    )
    
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(mode == 'train' and len(img_ids) >= batch_size),
    )


def create_model(config, device):
    """Create model instance."""
    model = archs.__dict__[config['arch']](
        config.get('num_classes', 1),
        config.get('input_channels', 3),
        config.get('deep_supervision', False)
    )
    return model.to(device)


def get_parameters(model):
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model, parameters):
    """Load parameters from numpy arrays."""
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state_dict, strict=False)


def train_local(train_loader, model, criterion, optimizer, config, device):
    """Local training on client."""
    model.train()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for x, y, _ in train_loader:
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
    
    return loss_meter.avg, iou_meter.avg


@torch.no_grad()
def evaluate_local(val_loader, model, criterion, device):
    """Local evaluation on client."""
    model.eval()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    use_amp = torch.cuda.is_available()
    for x, y, _ in val_loader:
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


def build_client_splits(client_id, num_clients, all_img_ids, config):
    """Assign train/val data for client."""
    partitions = split_samples_among_clients(all_img_ids, num_clients)
    return partitions[client_id]


def infer_pos_weight(train_dataset, num_classes):
    """Infer pos_weight from training set."""
    fg_count = 0
    total_pixels = 0
    
    for img_id in train_dataset:
        mask_path = f"inputs/inputs/{img_id[0]}/train/masks/{img_id[1]}.png"
        try:
            from PIL import Image
            mask = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask) / 255.0
            fg_count += (mask_arr > 0.5).sum()
            total_pixels += mask_arr.size
        except:
            pass
    
    if total_pixels > 0:
        fg_ratio = fg_count / total_pixels
        if fg_ratio > 0:
            return (1.0 - fg_ratio) / fg_ratio
    
    return 1.0
