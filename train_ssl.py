"""
Training script for contrastive self-supervised learning of charge-aware molecular representation.
Uses pre-augmented graph pairs from cache (built by build_graph_cache.py).
"""
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import random

from config import Config
from dataset.ssl.data_loader import create_val_loader, create_train_loader
from models.gin_e import GINEEncoder
from utils.loss import NTXentLoss

NUM_TRAIN_SHARDS = 6


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_shard(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    shard_idx: int,
    pbar_position: int = 0
) -> tuple:
    """
    Train on a single shard.
    
    Returns:
        Tuple of (total_loss, num_batches, skipped_batches).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    
    pbar = tqdm(train_loader, desc=f"  Shard {shard_idx}", position=pbar_position, leave=False)
    for batch1, batch2 in pbar:
        # Move batches to device
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        
        # Check for empty batches
        if batch1.num_graphs == 0 or batch2.num_graphs == 0:
            skipped_batches += 1
            continue
        
        # Forward pass
        z1 = model(
            x=batch1.x,
            edge_index=batch1.edge_index,
            edge_attr=batch1.edge_attr,
            batch=batch1.batch
        )  # [batch_size, hidden_dim]
        
        z2 = model(
            x=batch2.x,
            edge_index=batch2.edge_index,
            edge_attr=batch2.edge_attr,
            batch=batch2.batch
        )  # [batch_size, hidden_dim]
        
        # Check for NaN in embeddings
        if torch.isnan(z1).any() or torch.isnan(z2).any():
            skipped_batches += 1
            continue
        
        # Check batch size consistency
        if z1.size(0) != z2.size(0):
            skipped_batches += 1
            continue
        
        # Compute loss
        loss = criterion(z1, z2)
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            skipped_batches += 1
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'skip': skipped_batches})
    
    pbar.close()
    return total_loss, num_batches, skipped_batches


def train_epoch(
    model: nn.Module,
    train_shard_paths: list,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config
) -> float:
    """
    Train for one epoch, cycling through all shards.
    
    Returns:
        Average training loss across all shards.
    """
    total_loss = 0.0
    total_batches = 0
    total_skipped = 0
    
    # Progress bar for shards
    shard_pbar = tqdm(range(NUM_TRAIN_SHARDS), desc="Training shards", position=0)
    
    for shard_idx in shard_pbar:
        shard_path = train_shard_paths[shard_idx]
        
        # Load shard
        train_pairs = torch.load(shard_path, weights_only=False)
        train_loader = create_train_loader(
            train_pairs=train_pairs,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Train on this shard
        shard_loss, shard_batches, shard_skipped = train_shard(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            shard_idx=shard_idx,
            pbar_position=1
        )
        
        total_loss += shard_loss
        total_batches += shard_batches
        total_skipped += shard_skipped
        
        # Update shard progress bar with running average
        if total_batches > 0:
            avg_loss = total_loss / total_batches
            shard_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}', 'skipped': total_skipped})
        
        # Free memory
        del train_pairs, train_loader
    
    shard_pbar.close()
    
    if total_batches == 0:
        print(f"Warning: No valid batches in training set! Skipped {total_skipped} batches.")
        return float('nan')
    
    avg_loss = total_loss / total_batches
    if total_skipped > 0:
        print(f"Warning: Skipped {total_skipped} invalid batches during training")
    return avg_loss


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    config: Config
) -> float:
    """
    Validate the model.
    
    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    skip_reasons = {'empty_batch': 0, 'nan_embedding': 0, 'size_mismatch': 0, 'nan_loss': 0}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch1, batch2 in pbar:
            # Move batches to device
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            
            # Check for empty batches
            if batch1.num_graphs == 0 or batch2.num_graphs == 0:
                skipped_batches += 1
                skip_reasons['empty_batch'] += 1
                continue
            
            # Forward pass
            z1 = model(
                x=batch1.x,
                edge_index=batch1.edge_index,
                edge_attr=batch1.edge_attr,
                batch=batch1.batch
            )
            
            z2 = model(
                x=batch2.x,
                edge_index=batch2.edge_index,
                edge_attr=batch2.edge_attr,
                batch=batch2.batch
            )
            
            # Check for NaN in embeddings
            if torch.isnan(z1).any() or torch.isnan(z2).any():
                skipped_batches += 1
                skip_reasons['nan_embedding'] += 1
                continue
            
            # Check batch size consistency
            if z1.size(0) != z2.size(0):
                skipped_batches += 1
                skip_reasons['size_mismatch'] += 1
                continue
            
            # Compute loss
            loss = criterion(z1, z2)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                skip_reasons['nan_loss'] += 1
                continue
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    if num_batches == 0:
        print(f"Warning: No valid batches in validation set! Skipped {skipped_batches}/{len(val_loader)} batches.")
        print(f"  Skip reasons: {skip_reasons}")
        return float('nan')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str
):
    """Save periodic epoch checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def save_best_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str
):
    """Save best model checkpoint immediately."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    torch.save(checkpoint, best_path)
    print(f"Saved best model (epoch {epoch}, loss {loss:.4f}) to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device
) -> tuple:
    """
    Load checkpoint and restore model, optimizer, and scheduler states.
    
    Returns:
        Tuple of (start_epoch, best_val_loss) to resume training.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_val_loss = checkpoint['loss']
    
    # Advance scheduler to the correct state
    for _ in range(checkpoint['epoch']):
        scheduler.step()
    
    print(f"Resumed from epoch {checkpoint['epoch']} (best val loss: {best_val_loss:.4f})")
    return start_epoch, best_val_loss


def main():
    """Main training function."""
    # Load configuration
    config = Config()
    
    # Set random seed
    set_seed(config.seed)
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Check pre-augmented graph cache exists (val.pt + 6 training shards)
    val_pt = os.path.join(config.cache_dir, "val.pt")
    train_shard_paths = [os.path.join(config.cache_dir, f"train_shard_{i}.pt") for i in range(NUM_TRAIN_SHARDS)]
    
    # Check val.pt exists
    if not os.path.isfile(val_pt):
        raise FileNotFoundError(
            f"Validation cache not found: {val_pt}. Run build_graph_cache first, e.g.:\n"
            f"  python dataset/ssl/build_graph_cache.py --sdf_file {config.sdf_file} --cache_dir {config.cache_dir}"
        )
    
    # Check all training shards exist
    for p in train_shard_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Training shard not found: {p}. Run build_graph_cache first, e.g.:\n"
                f"  python dataset/ssl/build_graph_cache.py --sdf_file {config.sdf_file} --cache_dir {config.cache_dir}"
            )
    
    print(f"Using pre-augmented graph cache: {config.cache_dir}")
    print(f"  Validation: val.pt")
    print(f"  Training: {NUM_TRAIN_SHARDS} shards (train_shard_0.pt to train_shard_{NUM_TRAIN_SHARDS-1}.pt)")

    # Load validation pairs and create val_loader once (kept in memory for whole run)
    val_pairs = torch.load(val_pt, weights_only=False)
    val_loader = create_val_loader(
        val_pairs=val_pairs,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    print(f"Validation: {len(val_pairs):,} pre-augmented pairs, {len(val_loader)} batches")
    
    # Create model
    print("Initializing model...")
    model = GINEEncoder(
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=config.edge_feature_dim,
        node_embedding_dim=config.node_embedding_dim,
        edge_embedding_dim=config.edge_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gin_layers,
        dropout=config.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = NTXentLoss(temperature=config.temperature)
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    if config.resume_checkpoint is not None:
        if os.path.isfile(config.resume_checkpoint):
            start_epoch, best_val_loss = load_checkpoint(
                checkpoint_path=config.resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device
            )
        else:
            print(f"Warning: Checkpoint not found at {config.resume_checkpoint}, starting from scratch.")
    
    # Training loop
    print("Starting training...")
    print(f"Each epoch trains on all {NUM_TRAIN_SHARDS} shards")
    
    # Epoch progress bar
    epoch_pbar = tqdm(range(start_epoch, config.num_epochs + 1), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{config.num_epochs}")
        
        # Train on all shards
        train_loss = train_epoch(
            model=model,
            train_shard_paths=train_shard_paths,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config
        )
        
        # Validate every epoch
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            config=config
        )
        
        # Save best model immediately when found (only if val_loss is valid)
        if not torch.isnan(torch.tensor(val_loss)) and not torch.isinf(torch.tensor(val_loss)):
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_best_model(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    checkpoint_dir=config.checkpoint_dir
                )
        
        # Update learning rate
        scheduler.step()
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({'train': f'{train_loss:.4f}', 'val': f'{val_loss:.4f}', 'best': f'{best_val_loss:.4f}'})
        tqdm.write(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save periodic checkpoint every N epochs
        if epoch % config.checkpoint_frequency == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                checkpoint_dir=config.checkpoint_dir
            )

    epoch_pbar.close()
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
