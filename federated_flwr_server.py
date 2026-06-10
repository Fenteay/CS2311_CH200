"""Flower Federated Learning Server for leaf segmentation."""

import argparse
import copy
import os
from pathlib import Path

import torch
import yaml
from flwr.server import ServerConfig, start_server
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

from federated_flwr_common import create_model
from train_source import sanitize_name


def parse_args():
    parser = argparse.ArgumentParser(description='Flower FL Server')
    parser.add_argument('--host', default='127.0.0.1:8080',
                        help='Server bind address')
    parser.add_argument('--dataset', default='leafandmask_full')
    parser.add_argument('--arch', default='UNet')
    parser.add_argument('--img-ext', default='.png', dest='img_ext')
    parser.add_argument('--rounds', default=20, type=int)
    parser.add_argument('--num-clients', default=3, type=int)
    parser.add_argument('--local-epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--save-name', default=None)
    parser.add_argument('--init-weights', default=None, dest='init_weights',
                        help='Path to pretrained model.pth for warm-start initialization')
    return parser.parse_args()


def fit_config(server_round):
    """Create per-round config for clients."""
    config = {
        'round': server_round,
        'local_epochs': int(os.environ.get('LOCAL_EPOCHS', '1')),
        'batch_size': int(os.environ.get('BATCH_SIZE', '4')),
        'lr': float(os.environ.get('LR', '1e-4')),
    }
    print(f"[Server Round {server_round}] Config: {config}")
    return config


def weighted_average(metrics):
    """Aggregate client metrics weighted by sample count."""
    if not metrics:
        return {}
    
    total_samples = sum(num_examples for num_examples, _ in metrics)
    weighted_metrics = {}
    
    for num_examples, m in metrics:
        for key, val in m.items():
            if key not in weighted_metrics:
                weighted_metrics[key] = 0.0
            weighted_metrics[key] += val * (num_examples / total_samples)
    
    return weighted_metrics


class SaveModelFedAvg(FedAvg):
    """FedAvg with model saving after each round."""
    
    def __init__(self, model_dir, model_config, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.round = 0
        self.model_config = model_config
        self.device = device
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate and save model after fit round."""
        aggregated_weights, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights is not None:
            self.round = server_round
            # Save checkpoint
            checkpoint_path = self.model_dir / f'model_round_{server_round}.pth'
            # Convert Flower Parameters → numpy → PyTorch state_dict
            ndarrays = parameters_to_ndarrays(aggregated_weights)
            tmp_model = create_model(self.model_config, self.device)
            params_dict = zip(tmp_model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            tmp_model.load_state_dict(state_dict, strict=False)
            torch.save(tmp_model.state_dict(), str(checkpoint_path))
            print(f"[Server] Saved checkpoint: {checkpoint_path}")
        
        return aggregated_weights, aggregated_metrics


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[Server] Starting on {args.host}")
    print(f"[Server] Config: dataset={args.dataset}, arch={args.arch}, rounds={args.rounds}")
    
    # Create model
    config = {
        'arch': args.arch,
        'num_classes': 1,
        'input_channels': 3,
        'deep_supervision': False,
        'input_h': 512,
        'input_w': 512,
    }
    
    model = create_model(config, device)
    if args.init_weights and Path(args.init_weights).exists():
        state = torch.load(args.init_weights, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=True)
        print(f"[Server] Warm-start from: {args.init_weights}")
    else:
        print("[Server] Starting from random weights")
    initial_weights = [val.cpu().numpy() for val in model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_weights)
    
    # Setup model directory
    model_name = args.save_name or f"{sanitize_name(args.dataset)}_{args.arch.lower()}_fedavg_3containers"
    model_dir = Path('models') / model_name
    
    # Create strategy
    strategy = SaveModelFedAvg(
        model_dir,
        config,
        device,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
    )
    
    # Start server
    config_dict = ServerConfig(num_rounds=args.rounds)
    
    start_server(
        server_address=args.host,
        config=config_dict,
        strategy=strategy,
    )
    
    # Save final model — reload from last round checkpoint
    final_model_path = model_dir / 'model.pth'
    last_ckpt = model_dir / f'model_round_{args.rounds}.pth'
    if last_ckpt.exists():
        final_state = torch.load(str(last_ckpt), map_location=device, weights_only=False)
        torch.save(final_state, str(final_model_path))
    else:
        torch.save(model.state_dict(), str(final_model_path))
    print(f"[Server] Saved final model: {final_model_path}")
    
    # Save config files for targets
    fl_config = {
        'arch': args.arch,
        'num_classes': 1,
        'input_channels': 3,
        'deep_supervision': False,
        'input_h': 512,
        'input_w': 512,
        'img_ext': args.img_ext,
        'mask_ext': '.png',
        'federated': {
            'algorithm': 'FedAvg',
            'clients': args.num_clients,
            'rounds': args.rounds,
            'local_epochs': args.local_epochs,
        }
    }
    
    for target in ['rite', 'hrf', 'chase', 'leafandmask_trial', 'leafandmask_full']:
        config_path = model_dir / f'config_{target}.yml'
        with open(config_path, 'w') as f:
            yaml.safe_dump(fl_config, f, sort_keys=False)
    
    print(f"\n✅ Federated training complete!")
    print(f"   Model directory: {model_dir}")


if __name__ == '__main__':
    main()
