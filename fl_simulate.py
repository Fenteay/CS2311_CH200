"""
FL Simulation (no Docker) — continue FL training from an existing checkpoint.
Uses flwr.simulation.start_simulation for in-process federated learning.

Usage:
    python fl_simulate.py --init-weights models/multi_dataset_fedavg_warm_r5_e3_20260530/model.pth \
                          --rounds 5 --local-epochs 3 --output-dir models/fl_sim_r10_e3

This continues FL training from the warm-start round-5 model for an additional 5 rounds.
"""

import argparse
import copy
import os
import random
from pathlib import Path

import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

import archs
from federated_flwr_common import (
    create_model, evaluate_local, get_parameters, make_loader,
    set_parameters, split_samples_among_clients, train_local,
)
from train_source import WeightedBCEDiceLoss, build_split_samples, compute_binary_mask_ratio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--init-weights', required=True, help='Path to starting model weights (.pth)')
    p.add_argument('--arch', default='ResUNet', choices=sorted(archs.__all__),
                   help='Model architecture (must match init-weights checkpoint).')
    p.add_argument('--rounds', default=5, type=int, help='Number of additional FL rounds')
    p.add_argument('--local-epochs', default=3, type=int)
    p.add_argument('--num-clients', default=3, type=int)
    p.add_argument('--datasets', default='leafandmask_full,leaf_hrf_style,leaf_rite_style',
                   help='Comma-separated dataset names, one per client')
    p.add_argument('--img-ext', default='.jpg', dest='img_ext')
    p.add_argument('--lr', default=3e-5, type=float,
                   help='Local learning rate (lower than original to preserve warm-start)')
    p.add_argument('--batch-size', default=8, type=int)
    p.add_argument('--max-samples', default=100, type=int,
                   help='Max train samples per client (0=all). Default 100 for fast CPU run.')
    p.add_argument('--output-dir', default='models/fl_sim_continued')
    p.add_argument('--auto-pos-weight', action='store_true')
    return p.parse_args()


def fedavg(global_params, client_params_list, client_sizes):
    """FedAvg aggregation."""
    total = sum(client_sizes)
    aggregated = []
    for layer_idx in range(len(global_params)):
        weighted = sum(
            (client_sizes[i] / total) * client_params_list[i][layer_idx]
            for i in range(len(client_params_list))
        )
        aggregated.append(weighted)
    return aggregated


def main():
    args = parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(',')]
    n_clients = min(args.num_clients, len(dataset_names))
    print(f"FL Simulation: {args.rounds} rounds x {args.local_epochs} local epochs x {n_clients} clients")
    print(f"Init weights: {args.init_weights}")

    config = {
        'arch': args.arch,
        'num_classes': 1,
        'input_channels': 3,
        'deep_supervision': False,
        'input_h': 512,
        'input_w': 512,
        'img_ext': args.img_ext,
        'mask_ext': '.png',
    }

    # ── Build per-client data loaders ────────────────────────────────────────
    client_train_loaders = []
    client_test_loaders = []
    client_pos_weights = []

    for i in range(n_clients):
        ds_name = dataset_names[i]
        try:
            train_samples = build_split_samples([ds_name], 'train', args.img_ext, '.png', 1, 'error')
            test_samples  = build_split_samples([ds_name], 'test',  args.img_ext, '.png', 1, 'error')
        except RuntimeError as e:
            print(f"[Client {i}] WARNING: {e} - skipping dataset {ds_name}")
            continue

        # Limit samples for faster CPU training
        if args.max_samples > 0 and len(train_samples) > args.max_samples:
            random.shuffle(train_samples)
            train_samples = train_samples[:args.max_samples]
        if args.max_samples > 0 and len(test_samples) > args.max_samples // 4:
            test_samples = test_samples[:args.max_samples // 4]

        # Each client uses their own full dataset (no IID split needed — different datasets)
        train_loader = make_loader(train_samples, None, None, args.batch_size, 0, config, 'train')
        test_loader  = make_loader(test_samples,  None, None, 1,               0, config, 'test')

        if args.auto_pos_weight:
            fg = compute_binary_mask_ratio(train_samples, None, '.png')
            pw = min(max(1.0, (1 - fg) / max(fg, 1e-6)), 20.0)
        else:
            pw = 1.0

        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)
        client_pos_weights.append(pw)
        print(f"[Client {i}] {ds_name}: {len(train_samples)} train, {len(test_samples)} test, pos_w={pw:.2f}")

    n_clients = len(client_train_loaders)
    if n_clients == 0:
        raise RuntimeError("No valid client datasets found.")

    # ── Load global model ─────────────────────────────────────────────────────
    global_model = create_model(config, device)
    state = torch.load(args.init_weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    global_model.load_state_dict(state)
    print(f"Loaded global model from {args.init_weights}")

    history = []

    # ── FL rounds ─────────────────────────────────────────────────────────────
    for rnd in range(1, args.rounds + 1):
        print(f"\n{'='*60}\n Round {rnd}/{args.rounds}\n{'='*60}")

        global_params = get_parameters(global_model)
        client_params_list = []
        client_sizes = []
        round_metrics = []

        for i in range(n_clients):
            n_batches = len(client_train_loaders[i])
            print(f"  [Client {i}] training {args.local_epochs} epoch(s) x {n_batches} batches...")
            client_model = create_model(config, device)
            set_parameters(client_model, copy.deepcopy(global_params))

            criterion = WeightedBCEDiceLoss(pos_weight=client_pos_weights[i]).to(device)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=args.lr, weight_decay=1e-4)

            class _Args:
                local_epochs = args.local_epochs

            t0 = time.time()
            for ep in range(args.local_epochs):
                loss, iou = train_local(
                    client_train_loaders[i], client_model, criterion, optimizer, _Args(), device
                )
                elapsed = time.time() - t0
                print(f"    ep {ep+1}/{args.local_epochs}: loss={loss:.4f} iou={iou:.4f} ({elapsed:.0f}s)", flush=True)
            print(f"  [Client {i}] done: train_loss={loss:.4f}, train_iou={iou:.4f}")

            val_loss, val_iou, val_dice = evaluate_local(
                client_test_loaders[i], client_model, criterion, device
            )
            print(f"  [Client {i}] val_loss={val_loss:.4f}, val_iou={val_iou:.4f}, val_dice={val_dice:.4f}")

            client_params_list.append(get_parameters(client_model))
            client_sizes.append(len(client_train_loaders[i].dataset))
            round_metrics.append({'val_dice': val_dice, 'val_iou': val_iou, 'train_loss': loss})

        # FedAvg
        aggregated_params = fedavg(global_params, client_params_list, client_sizes)
        set_parameters(global_model, aggregated_params)

        avg_dice = np.mean([m['val_dice'] for m in round_metrics])
        avg_iou  = np.mean([m['val_iou']  for m in round_metrics])
        print(f"\n  [Server] Round {rnd} — avg val_dice={avg_dice:.4f}, avg_val_iou={avg_iou:.4f}")
        history.append({'round': rnd, 'avg_val_dice': float(avg_dice), 'avg_val_iou': float(avg_iou)})

        # Save per-round checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(global_model.state_dict(), os.path.join(args.output_dir, f'model_round_{rnd}.pth'))

    # ── Save final model ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(global_model.state_dict(), os.path.join(args.output_dir, 'model.pth'))

    # Save config files for tt_sfuda
    for target in ['leafandmask_trial', 'leafandmask_full', 'hrf', 'rite', 'chase']:
        cfg = {
            'arch': 'UNet', 'num_classes': 1, 'input_channels': 3, 'deep_supervision': False,
            'input_h': 512, 'input_w': 512, 'img_ext': args.img_ext, 'mask_ext': '.png',
            'lr': 7e-5, 'weight_decay': 1e-4, 'loss': 'BCEDiceLoss', 'stage1': 1, 'stage2': 1,
            'name': Path(args.output_dir).name,
        }
        with open(os.path.join(args.output_dir, f'config_{target}.yml'), 'w') as f:
            yaml.dump(cfg, f)

    print(f"\nFinal model saved to {args.output_dir}/model.pth")
    print("History:")
    for h in history:
        print(f"  Round {h['round']}: val_dice={h['avg_val_dice']:.4f}, val_iou={h['avg_val_iou']:.4f}")


if __name__ == '__main__':
    main()
