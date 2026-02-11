"""
Hyperparameter tuning script for downstream binding energy prediction.

This script performs grid search or random search over hyperparameters
to find the best configuration for minimizing validation MAE.

Usage:
    python tune_hyperparams.py --mode grid
    python tune_hyperparams.py --mode random --n_trials 20
"""
import os
import csv
import json
import random
import itertools
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from config import Config
from models.gin_e import GINEEncoder
from models.downstream_model import DownstreamModel
from train_downstream import (
    load_adsorption_data, split_data, create_data_loaders,
    BindingEnergyDataset, collate_binding_batch, set_seed
)


@dataclass
class TuningConfig:
    """Configuration for a single hyperparameter trial (downstream head only)."""
    # Downstream MLP architecture
    mlp_hidden_dim: int = 512
    task_hidden_dim: int = 256
    
    # Regularization (downstream only)
    mlp_dropout: float = 0.2
    task_dropout: float = 0.2
    weight_decay: float = 1e-4
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.0005
    num_epochs: int = 150
    scheduler: str = "cosine"  # "cosine", "plateau", "onecycle"
    optimizer: str = "adamw"  # "adam", "adamw"
    
    # Early stopping
    patience: int = 20
    min_delta: float = 0.001


# Hyperparameter search space (downstream head only - GIN-E encoder uses config.py settings)
SEARCH_SPACE = {
    # Downstream MLP architecture
    "mlp_hidden_dim": [256, 512, 768],
    "task_hidden_dim": [64, 128, 256],
    
    # Regularization (downstream only)
    "mlp_dropout": [0.1, 0.2],
    "task_dropout": [0.1, 0.2],
    "weight_decay": [1e-5],
    
    # Training
    "batch_size": [32],
    "learning_rate": [0.001],
    "scheduler": ["cosine"],
    "optimizer": ["adam"],
}

# Recommended configurations to try first (downstream head only)
RECOMMENDED_CONFIGS = [
    # Config 1: Higher regularization
    {
        "mlp_hidden_dim": 512, "task_hidden_dim": 256,
        "mlp_dropout": 0.3, "task_dropout": 0.3, "weight_decay": 1e-4,
        "batch_size": 64, "learning_rate": 0.0005, "scheduler": "cosine", "optimizer": "adamw",
    },
    # Config 2: Larger downstream head
    {
        "mlp_hidden_dim": 768, "task_hidden_dim": 384,
        "mlp_dropout": 0.3, "task_dropout": 0.3, "weight_decay": 5e-4,
        "batch_size": 64, "learning_rate": 0.0003, "scheduler": "cosine", "optimizer": "adamw",
    },
    # Config 3: Smaller downstream head
    {
        "mlp_hidden_dim": 256, "task_hidden_dim": 128,
        "mlp_dropout": 0.2, "task_dropout": 0.2, "weight_decay": 1e-4,
        "batch_size": 64, "learning_rate": 0.001, "scheduler": "cosine", "optimizer": "adamw",
    },
    # Config 4: OneCycleLR scheduler
    {
        "mlp_hidden_dim": 512, "task_hidden_dim": 256,
        "mlp_dropout": 0.2, "task_dropout": 0.2, "weight_decay": 1e-4,
        "batch_size": 64, "learning_rate": 0.001, "scheduler": "onecycle", "optimizer": "adamw",
    },
    # Config 5: Smaller batch, lower LR
    {
        "mlp_hidden_dim": 512, "task_hidden_dim": 256,
        "mlp_dropout": 0.2, "task_dropout": 0.2, "weight_decay": 5e-5,
        "batch_size": 32, "learning_rate": 0.0003, "scheduler": "plateau", "optimizer": "adamw",
    },
]


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def create_model(config: TuningConfig, base_config: Config, device: torch.device) -> DownstreamModel:
    """Create model with given configuration (encoder uses base_config, downstream uses TuningConfig)."""
    # GIN-E encoder uses fixed configuration from config.py
    gin_e_encoder = GINEEncoder(
        node_feature_dim=base_config.node_feature_dim,
        edge_feature_dim=base_config.edge_feature_dim,
        node_embedding_dim=base_config.node_embedding_dim,
        edge_embedding_dim=base_config.edge_embedding_dim,
        hidden_dim=base_config.hidden_dim,
        num_layers=base_config.num_gin_layers,
        dropout=base_config.dropout
    )
    
    gin_e_checkpoint_path = os.path.join(base_config.checkpoint_dir, "best_model.pt")
    if not os.path.exists(gin_e_checkpoint_path):
        gin_e_checkpoint_path = None
    
    # Downstream head uses tuning configuration
    model = DownstreamModel(
        gin_e_encoder=gin_e_encoder,
        gin_e_checkpoint_path=gin_e_checkpoint_path,
        freeze_gin_e=base_config.freeze_pretrained_encoder,
        mlp_hidden_dim=config.mlp_hidden_dim,
        mlp_dropout=config.mlp_dropout,
        num_tasks=1,
        task_hidden_dim=config.task_hidden_dim,
        task_dropout=config.task_dropout
    ).to(device)
    
    return model


def create_optimizer(model: nn.Module, config: TuningConfig) -> torch.optim.Optimizer:
    """Create optimizer for trainable parameters."""
    # Get all trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    if config.optimizer == "adamw":
        return AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        return Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, config: TuningConfig, 
                    train_loader: DataLoader) -> Any:
    """Create learning rate scheduler."""
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-7)
    elif config.scheduler == "plateau":
        # Note: some PyTorch versions of ReduceLROnPlateau do not support 'verbose' kwarg
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                 min_lr=1e-7)
    elif config.scheduler == "onecycle":
        return OneCycleLR(optimizer, max_lr=config.learning_rate, 
                         epochs=config.num_epochs, steps_per_epoch=len(train_loader))
    else:
        return None


def train_single_config(
    config: TuningConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_config: Config,
    device: torch.device,
    verbose: bool = False
) -> Tuple[float, float, int]:
    """
    Train model with a single configuration.
    
    Returns:
        Tuple of (best_val_mae, final_train_mae, best_epoch)
    """
    model = create_model(config, base_config, device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, train_loader)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    best_val_mae = float('inf')
    final_train_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(1, config.num_epochs + 1):
        # Training
        model.train()
        train_maes = []
        
        for batch_graph, batch_energies in train_loader:
            batch_graph = batch_graph.to(device)
            batch_energies = batch_energies.to(device)
            
            predictions = model(
                x=batch_graph.x,
                edge_index=batch_graph.edge_index,
                edge_attr=batch_graph.edge_attr,
                batch=batch_graph.batch
            )
            
            loss = criterion(predictions, batch_energies)
            mae = torch.mean(torch.abs(predictions - batch_energies)).item()
            train_maes.append(mae)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if config.scheduler == "onecycle":
                scheduler.step()
        
        # Validation
        model.eval()
        val_maes = []
        
        with torch.no_grad():
            for batch_graph, batch_energies in val_loader:
                batch_graph = batch_graph.to(device)
                batch_energies = batch_energies.to(device)
                
                predictions = model(
                    x=batch_graph.x,
                    edge_index=batch_graph.edge_index,
                    edge_attr=batch_graph.edge_attr,
                    batch=batch_graph.batch
                )
                
                mae = torch.mean(torch.abs(predictions - batch_energies)).item()
                val_maes.append(mae)
        
        train_mae = np.mean(train_maes)
        val_mae = np.mean(val_maes)
        
        # Update scheduler
        if config.scheduler == "cosine":
            scheduler.step()
        elif config.scheduler == "plateau":
            scheduler.step(val_mae)
        
        # Track best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            final_train_mae = train_mae
            best_epoch = epoch
        
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")
        
        # Early stopping
        if early_stopping(val_mae, epoch):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break
    
    return best_val_mae, final_train_mae, best_epoch


def run_grid_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_config: Config,
    device: torch.device,
    output_dir: str
) -> List[Dict]:
    """Run grid search over recommended configurations."""
    results = []
    
    print(f"\nRunning grid search over {len(RECOMMENDED_CONFIGS)} configurations...")
    
    for i, config_dict in enumerate(RECOMMENDED_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(RECOMMENDED_CONFIGS)}")
        print(f"{'='*60}")
        
        config = TuningConfig(**config_dict)
        print(f"Config: {config_dict}")
        
        val_mae, train_mae, best_epoch = train_single_config(
            config, train_loader, val_loader, base_config, device, verbose=True
        )
        
        result = {
            "config_id": i + 1,
            **config_dict,
            "best_val_mae": val_mae,
            "best_train_mae": train_mae,
            "best_epoch": best_epoch,
            "overfitting_gap": train_mae - val_mae
        }
        results.append(result)
        
        print(f"\nResult: Val MAE={val_mae:.4f}, Train MAE={train_mae:.4f}, Best Epoch={best_epoch}")
    
    return results


def run_random_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_config: Config,
    device: torch.device,
    output_dir: str,
    n_trials: int = 20
) -> List[Dict]:
    """Run random search over hyperparameter space."""
    results = []
    
    print(f"\nRunning random search with {n_trials} trials...")
    
    for trial in range(n_trials):
        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{n_trials}")
        print(f"{'='*60}")
        
        # Sample random configuration
        config_dict = {
            key: random.choice(values) for key, values in SEARCH_SPACE.items()
        }
        
        config = TuningConfig(**config_dict)
        print(f"Config: {config_dict}")
        
        try:
            val_mae, train_mae, best_epoch = train_single_config(
                config, train_loader, val_loader, base_config, device, verbose=True
            )
            
            result = {
                "trial": trial + 1,
                **config_dict,
                "best_val_mae": val_mae,
                "best_train_mae": train_mae,
                "best_epoch": best_epoch,
                "overfitting_gap": train_mae - val_mae
            }
            results.append(result)
            
            print(f"\nResult: Val MAE={val_mae:.4f}, Train MAE={train_mae:.4f}, Best Epoch={best_epoch}")
        
        except Exception as e:
            print(f"Trial failed: {e}")
            continue
    
    return results


def save_results(results: List[Dict], output_dir: str, mode: str):
    """Save tuning results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sort by validation MAE
    results = sorted(results, key=lambda x: x["best_val_mae"])
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"tuning_results_{mode}_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Save JSON
    json_path = os.path.join(output_dir, f"tuning_results_{mode}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    
    return results


def print_summary(results: List[Dict]):
    """Print summary of tuning results."""
    print("\n" + "="*60)
    print("TUNING RESULTS SUMMARY (Downstream Head Only)")
    print("="*60)
    
    print("\nTop 5 Configurations by Validation MAE:")
    print("-" * 80)
    
    for i, result in enumerate(results[:5]):
        print(f"\n#{i+1}: Val MAE = {result['best_val_mae']:.4f} eV")
        print(f"    Train MAE = {result['best_train_mae']:.4f} eV")
        print(f"    Overfitting Gap = {result['overfitting_gap']:.4f} eV")
        print(f"    Best Epoch = {result['best_epoch']}")
        print(f"    Key params: mlp_hidden_dim={result['mlp_hidden_dim']}, "
              f"task_hidden_dim={result['task_hidden_dim']}, "
              f"mlp_dropout={result['mlp_dropout']}, "
              f"lr={result['learning_rate']}, "
              f"weight_decay={result['weight_decay']}")
    
    # Best config
    best = results[0]
    print("\n" + "="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    print(f"\nValidation MAE: {best['best_val_mae']:.4f} eV")
    print(f"\nRecommended config.py settings (downstream head):")
    print(f"  downstream_mlp_hidden_dim: int = {best['mlp_hidden_dim']}")
    print(f"  downstream_task_hidden_dim: int = {best['task_hidden_dim']}")
    print(f"  downstream_mlp_dropout: float = {best['mlp_dropout']}")
    print(f"  downstream_task_dropout: float = {best['task_dropout']}")
    print(f"  downstream_batch_size: int = {best['batch_size']}")
    print(f"  downstream_learning_rate: float = {best['learning_rate']}")
    print(f"  downstream_weight_decay: float = {best['weight_decay']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for binding energy prediction")
    parser.add_argument("--mode", type=str, choices=["grid", "random"], default="grid",
                        help="Search mode: 'grid' for recommended configs, 'random' for random search")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials for random search")
    parser.add_argument("--output_dir", type=str, default="./tuning_results",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load base config
    base_config = Config()
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    csv_path = "./combined_data.csv"
    graphs, energies, cids = load_adsorption_data(csv_path, use_pubchem=False)
    
    if len(graphs) == 0:
        raise RuntimeError("No molecules loaded!")
    
    # Split data
    print("\nSplitting data...")
    train_graphs, train_energies, val_graphs, val_energies, _, _ = split_data(
        graphs, energies,
        train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
        seed=args.seed
    )
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Create data loaders (fixed batch size for fair comparison)
    train_dataset = BindingEnergyDataset(train_graphs, train_energies)
    val_dataset = BindingEnergyDataset(val_graphs, val_energies)
    
    # Run tuning
    if args.mode == "grid":
        # Full grid search over SEARCH_SPACE
        keys = list(SEARCH_SPACE.keys())
        value_lists = [SEARCH_SPACE[k] for k in keys]
        all_combinations = list(itertools.product(*value_lists))
        total_configs = len(all_combinations)

        print(f"\nRunning full grid search over {total_configs} configurations...")

        results = []
        for i, values in enumerate(all_combinations):
            config_dict = dict(zip(keys, values))
            batch_size = config_dict.get("batch_size", 64)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_binding_batch,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_binding_batch,
                pin_memory=True,
            )

            print(f"\n{'='*60}")
            print(f"Configuration {i+1}/{total_configs}")
            print(f"{'='*60}")
            print(f"Config: {config_dict}")

            config = TuningConfig(**config_dict)
            val_mae, train_mae, best_epoch = train_single_config(
                config, train_loader, val_loader, base_config, device, verbose=True
            )

            result = {
                "config_id": i + 1,
                **config_dict,
                "best_val_mae": val_mae,
                "best_train_mae": train_mae,
                "best_epoch": best_epoch,
                "overfitting_gap": train_mae - val_mae,
            }
            results.append(result)
            print(
                f"\nResult: Val MAE={val_mae:.4f}, "
                f"Train MAE={train_mae:.4f}, Best Epoch={best_epoch}"
            )
    else:
        results = []
        for trial in range(args.n_trials):
            config_dict = {key: random.choice(values) for key, values in SEARCH_SPACE.items()}
            batch_size = config_dict.get("batch_size", 64)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=4, collate_fn=collate_binding_batch, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=4, collate_fn=collate_binding_batch, pin_memory=True)
            
            print(f"\n{'='*60}")
            print(f"Trial {trial+1}/{args.n_trials}")
            print(f"{'='*60}")
            print(f"Config: {config_dict}")
            
            try:
                config = TuningConfig(**config_dict)
                val_mae, train_mae, best_epoch = train_single_config(
                    config, train_loader, val_loader, base_config, device, verbose=True
                )
                
                result = {
                    "trial": trial + 1,
                    **config_dict,
                    "best_val_mae": val_mae,
                    "best_train_mae": train_mae,
                    "best_epoch": best_epoch,
                    "overfitting_gap": train_mae - val_mae
                }
                results.append(result)
                print(f"\nResult: Val MAE={val_mae:.4f}, Train MAE={train_mae:.4f}, Best Epoch={best_epoch}")
            except Exception as e:
                print(f"Trial failed: {e}")
    
    # Save and summarize results
    results = save_results(results, args.output_dir, args.mode)
    print_summary(results)


if __name__ == "__main__":
    main()

