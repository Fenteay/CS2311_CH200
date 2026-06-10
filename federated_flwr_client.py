"""Flower Federated Learning Client for leaf segmentation."""

import argparse
import os
from pathlib import Path

import flwr as fl
import torch

from federated_flwr_common import (
    create_model,
    evaluate_local,
    make_loader,
    split_samples_among_clients,
    train_local,
)
from federated_flwr_common import get_parameters, set_parameters
from train_source import (
    WeightedBCEDiceLoss,
    build_split_samples,
    compute_binary_mask_ratio,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Flower FL Client')
    parser.add_argument('--server-address', default='127.0.0.1:8080',
                        dest='server_address')
    parser.add_argument('--dataset', default='leafandmask_full',
                        help='Dataset name')
    parser.add_argument('--arch', default='UNet')
    parser.add_argument('--img-ext', default='.png', dest='img_ext')
    parser.add_argument('--client-id', default=0, type=int, dest='client_id')
    parser.add_argument('--num-clients', default=3, type=int, dest='num_clients')
    parser.add_argument('--batch-size', default=4, type=int, dest='batch_size')
    parser.add_argument('--local-epochs', default=1, type=int, dest='local_epochs')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--num-workers', default=0, type=int, dest='num_workers')
    parser.add_argument('--auto-pos-weight', action='store_true', dest='auto_pos_weight')
    return parser.parse_args()


class SegClient(fl.client.NumPyClient):
    """Federated learning client for segmentation."""
    
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.local_epochs = args.local_epochs
    
    def get_parameters(self, config):
        """Send model parameters to server."""
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        """Perform local training."""
        # Load parameters from server
        set_parameters(self.model, parameters)
        
        # Train
        local_epochs = config.get('local_epochs', self.local_epochs)
        for epoch in range(local_epochs):
            train_loss, train_iou = train_local(
                self.train_loader,
                self.model,
                self.criterion,
                self.optimizer,
                self.args,
                device
            )
        
        # Get updated parameters
        new_parameters = get_parameters(self.model)
        
        # Metrics
        metrics = {
            'train_loss': float(train_loss),
            'train_iou': float(train_iou),
        }
        
        return new_parameters, len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters, config):
        """Perform local evaluation."""
        # Load parameters from server
        set_parameters(self.model, parameters)
        
        # Evaluate
        val_loss, val_iou, val_dice = evaluate_local(
            self.test_loader,
            self.model,
            self.criterion,
            device
        )
        
        # Metrics
        metrics = {
            'val_loss': float(val_loss),
            'val_iou': float(val_iou),
            'val_dice': float(val_dice),
        }
        
        return float(val_loss), len(self.test_loader.dataset), metrics


def main():
    args = parse_args()
    
    print(f"[Client {args.client_id}] Starting")
    print(f"  Dataset: {args.dataset}")
    print(f"  Server: {args.server_address}")
    
    # Load data
    mask_ext = '.png'
    num_classes = 1
    
    train_samples = build_split_samples(
        [args.dataset], 'train', args.img_ext, mask_ext, num_classes, 'error'
    )
    test_samples = build_split_samples(
        [args.dataset], 'test', args.img_ext, mask_ext, num_classes, 'error'
    )
    
    # Partition training data IID by client_id
    all_train_samples = train_samples
    partitions = split_samples_among_clients(all_train_samples, args.num_clients)
    train_samples = partitions[args.client_id]
    
    print(f"  Train samples: {len(train_samples)}/{len(all_train_samples)} (client {args.client_id}/{args.num_clients}), Test samples: {len(test_samples)}")
    
    # Create model
    config = {
        'arch': args.arch,
        'num_classes': 1,
        'input_channels': 3,
        'deep_supervision': False,
        'input_h': 512,
        'input_w': 512,
        'img_ext': args.img_ext,
        'mask_ext': mask_ext,
    }
    
    model = create_model(config, device)
    
    # Create loaders
    train_loader = make_loader(
        train_samples, None, None, args.batch_size, args.num_workers, config, 'train'
    )
    test_loader = make_loader(
        test_samples, None, None, 1, args.num_workers, config, 'test'
    )
    
    # Compute pos_weight
    if args.auto_pos_weight:
        fg_ratio = compute_binary_mask_ratio(train_samples, None, mask_ext)
        pos_weight = (1.0 - fg_ratio) / max(fg_ratio, 1e-6) if fg_ratio > 0 else 1.0
        pos_weight = min(max(1.0, pos_weight), 20.0)
    else:
        pos_weight = 1.0
    
    print(f"  pos_weight: {pos_weight:.4f}")
    
    # Create loss and optimizer
    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create client
    client = SegClient(model, train_loader, test_loader, criterion, optimizer, args)
    
    # Connect to server
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == '__main__':
    main()
