"""
Training script for contrastive self-supervised learning of charge-aware molecular representation.

Supports two data loading options (set via config.use_cache):
  Option 1 (use_cache=True): Load pre-augmented graph pairs from cache (built by build_graph_cache.py)
  Option 2 (use_cache=False): Load CSV, convert SMILES to graphs, augment in-memory
    - fixed_augmentation=True: Pre-augment once at start, same pairs each epoch
    - fixed_augmentation=False: Augment on-the-fly, fresh random pairs each epoch
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
from dataset.ssl.data_loader import (
    create_val_loader, create_train_loader,  # Option 1
    prepare_inmemory_data, create_inmemory_train_loader, create_inmemory_val_loader,  # Option 2 (on-the-fly)
    prepare_inmemory_data_fixed, create_fixed_train_loader, create_fixed_val_loader  # Option 2 (fixed)
)
from models.gin_e import GINEEncoder
from utils.loss import NTXentLoss

NUM_TRAIN_SHARDS = 4


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
    epoch: int,
    num_epochs: int
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
    
    pbar = tqdm(
        train_loader, 
        desc=f"  Epoch {epoch}/{num_epochs} Shard {shard_idx}/{NUM_TRAIN_SHARDS-1}",
        leave=False,
        dynamic_ncols=True
    )
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
    config: Config,
    epoch: int
) -> float:
    """
    Train for one epoch, cycling through all shards.
    
    Returns:
        Average training loss across all shards.
    """
    total_loss = 0.0
    total_batches = 0
    total_skipped = 0
    
    for shard_idx in range(NUM_TRAIN_SHARDS):
        shard_path = train_shard_paths[shard_idx]
        
        # Load shard with progress indication
        tqdm.write(f"  Loading shard {shard_idx}...")
        train_pairs = torch.load(shard_path, weights_only=False)
        train_loader = create_train_loader(
            train_pairs=train_pairs,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        tqdm.write(f"  Loaded {len(train_pairs):,} pairs ({len(train_loader)} batches)")
        
        # Train on this shard
        shard_loss, shard_batches, shard_skipped = train_shard(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            shard_idx=shard_idx,
            epoch=epoch,
            num_epochs=config.num_epochs
        )
        
        total_loss += shard_loss
        total_batches += shard_batches
        total_skipped += shard_skipped
        
        # Print shard summary
        if shard_batches > 0:
            shard_avg = shard_loss / shard_batches
            tqdm.write(f"  Shard {shard_idx} done: loss={shard_avg:.4f}, batches={shard_batches}, skipped={shard_skipped}")
        
        # Free memory
        del train_pairs, train_loader
    
    if total_batches == 0:
        print(f"Warning: No valid batches in training set! Skipped {total_skipped} batches.")
        return float('nan')
    
    avg_loss = total_loss / total_batches
    if total_skipped > 0:
        print(f"Warning: Skipped {total_skipped} invalid batches during training")
    return avg_loss


def train_epoch_inmemory(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> float:
    """
    Train for one epoch using in-memory data with on-the-fly augmentation (Option 2).
    
    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    
    pbar = tqdm(
        train_loader,
        desc=f"  Epoch {epoch}/{num_epochs} Training",
        leave=False,
        dynamic_ncols=True
    )
    
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'skip': skipped_batches})
    
    pbar.close()
    
    if num_batches == 0:
        print(f"Warning: No valid batches in training set! Skipped {skipped_batches} batches.")
        return float('nan')
    
    avg_loss = total_loss / num_batches
    if skipped_batches > 0:
        print(f"Warning: Skipped {skipped_batches} invalid batches during training")
    return avg_loss


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int
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
        pbar = tqdm(
            val_loader, 
            desc=f"  Epoch {epoch}/{num_epochs} Validation",
            leave=False,
            dynamic_ncols=True
        )
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
    val_loss: float,
    train_loss: float,
    checkpoint_dir: str
):
    """Save periodic epoch checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,  # Keep 'loss' for backward compatibility
        'val_loss': val_loss,
        'train_loss': train_loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def save_best_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    train_loss: float,
    checkpoint_dir: str
):
    """Save best model checkpoint immediately."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,  # Keep 'loss' for backward compatibility
        'val_loss': val_loss,
        'train_loss': train_loss,
    }
    
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    torch.save(checkpoint, best_path)
    print(f"Saved best model (epoch {epoch}, val_loss {val_loss:.4f}, train_loss {train_loss:.4f}) to {best_path}")


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

    # ==========================================================================
    # Data Loading: Branch based on use_cache option
    # ==========================================================================
    train_shard_paths = None  # Only used for Option 1
    train_loader = None       # Only used for Option 2
    train_graphs = None       # Only used for Option 2
    augmentation = None       # Only used for Option 2
    
    if config.use_cache:
        # ======================================================================
        # Option 1: Load pre-augmented pairs from cache
        # ======================================================================
        print("\n" + "=" * 70)
        print("OPTION 1: Loading pre-augmented graph pairs from cache")
        print("=" * 70)
        
        val_pt = os.path.join(config.cache_dir, "val.pt")
        train_shard_paths = [os.path.join(config.cache_dir, f"train_shard_{i}.pt") for i in range(NUM_TRAIN_SHARDS)]
        
        # Check val.pt exists
        if not os.path.isfile(val_pt):
            raise FileNotFoundError(
                f"Validation cache not found: {val_pt}. Run build_graph_cache first, e.g.:\n"
                f"  python dataset/ssl/build_graph_cache.py --csv_file {config.csv_file} --cache_dir {config.cache_dir}"
            )
        
        # Check all training shards exist
        for p in train_shard_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"Training shard not found: {p}. Run build_graph_cache first, e.g.:\n"
                    f"  python dataset/ssl/build_graph_cache.py --csv_file {config.csv_file} --cache_dir {config.cache_dir}"
                )
        
        print(f"Cache directory: {config.cache_dir}")
        print(f"  Validation: val.pt")
        print(f"  Training: {NUM_TRAIN_SHARDS} shards (train_shard_0.pt to train_shard_{NUM_TRAIN_SHARDS-1}.pt)")

        # Load validation pairs
        print("\nLoading validation data...")
        with tqdm(total=1, desc="Loading val.pt", leave=False) as pbar:
            val_pairs = torch.load(val_pt, weights_only=False)
            pbar.update(1)
        val_loader = create_val_loader(
            val_pairs=val_pairs,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        print(f"Validation: {len(val_pairs):,} pre-augmented pairs, {len(val_loader)} batches")
        
    else:
        # ======================================================================
        # Option 2: Load CSV, convert to graphs, augment in memory
        # ======================================================================
        if not os.path.isfile(config.csv_file):
            raise FileNotFoundError(f"CSV file not found: {config.csv_file}")
        
        if config.fixed_augmentation:
            # Option 2 (fixed): Pre-augment once, same pairs each epoch
            print("\n" + "=" * 70)
            print("OPTION 2 (FIXED): Pre-augment once, same pairs each epoch")
            print("=" * 70)
            
            print(f"CSV file: {config.csv_file}")
            print(f"Max molecules: {config.max_molecules if config.max_molecules else 'all'}")
            print(f"Train/Val split: {config.train_val_split*100:.0f}% / {(1-config.train_val_split)*100:.0f}%")
            print(f"Augmentation: subgraph removal (ratio={config.subgraph_removal_ratio})")
            print(f"Mode: Fixed (same augmented pairs each epoch)")
            
            # Load, convert, split, and pre-augment
            train_pairs, val_pairs = prepare_inmemory_data_fixed(
                csv_file=config.csv_file,
                train_ratio=config.train_val_split,
                max_molecules=config.max_molecules,
                removal_ratio=config.subgraph_removal_ratio,
                seed=config.seed
            )
            
            # Create data loaders (reuse PreAugmentedDataset like Option 1)
            train_loader = create_fixed_train_loader(
                train_pairs=train_pairs,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=True
            )
            val_loader = create_fixed_val_loader(
                val_pairs=val_pairs,
                batch_size=config.batch_size,
                num_workers=config.num_workers
            )
            
            print(f"\nData loaders created:")
            print(f"  Train: {len(train_pairs):,} pairs, {len(train_loader)} batches")
            print(f"  Val: {len(val_pairs):,} pairs, {len(val_loader)} batches")
            
        else:
            # Option 2 (on-the-fly): Fresh random augmentation each epoch
            print("\n" + "=" * 70)
            print("OPTION 2 (ON-THE-FLY): Fresh random augmentation each epoch")
            print("=" * 70)
            
            print(f"CSV file: {config.csv_file}")
            print(f"Max molecules: {config.max_molecules if config.max_molecules else 'all'}")
            print(f"Train/Val split: {config.train_val_split*100:.0f}% / {(1-config.train_val_split)*100:.0f}%")
            print(f"Augmentation: subgraph removal (ratio={config.subgraph_removal_ratio})")
            print(f"Mode: On-the-fly (fresh random pairs each epoch)")
            
            # Load and prepare data
            train_graphs, val_graphs, augmentation = prepare_inmemory_data(
                csv_file=config.csv_file,
                train_ratio=config.train_val_split,
                max_molecules=config.max_molecules,
                removal_ratio=config.subgraph_removal_ratio,
                seed=config.seed
            )
            
            # Create data loaders
            train_loader = create_inmemory_train_loader(
                train_graphs=train_graphs,
                augmentation=augmentation,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=True
            )
            val_loader = create_inmemory_val_loader(
                val_graphs=val_graphs,
                augmentation=augmentation,
                batch_size=config.batch_size,
                num_workers=config.num_workers
            )
            
            print(f"\nData loaders created:")
            print(f"  Train: {len(train_graphs):,} graphs, {len(train_loader)} batches")
            print(f"  Val: {len(val_graphs):,} graphs, {len(val_loader)} batches")
    
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
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    if config.use_cache:
        mode_str = "Option 1 (cached shards)"
    elif config.fixed_augmentation:
        mode_str = "Option 2 (in-memory, fixed augmentation)"
    else:
        mode_str = "Option 2 (in-memory, on-the-fly augmentation)"
    print(f"Mode: {mode_str}")
    print(f"Epochs: {start_epoch} to {config.num_epochs}")
    if config.use_cache:
        print(f"Each epoch trains on all {NUM_TRAIN_SHARDS} shards")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate} -> 1e-6 (cosine annealing)")
    print("=" * 70 + "\n")
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{config.num_epochs}")
        print(f"{'='*70}")
        
        # Train based on mode
        if config.use_cache:
            # Option 1: Train on shards from cache
            train_loss = train_epoch(
                model=model,
                train_shard_paths=train_shard_paths,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                config=config,
                epoch=epoch
            )
        else:
            # Option 2: Train on in-memory data with on-the-fly augmentation
            train_loss = train_epoch_inmemory(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=config.num_epochs
            )
        
        # Validate every epoch
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            num_epochs=config.num_epochs
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
                    val_loss=val_loss,
                    train_loss=train_loss,
                    checkpoint_dir=config.checkpoint_dir
                )
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\n>>> Epoch {epoch} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Best Val:   {best_val_loss:.4f}")
        print(f"    LR:         {scheduler.get_last_lr()[0]:.6f}")
        
        # Save periodic checkpoint every N epochs
        if epoch % config.checkpoint_frequency == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                train_loss=train_loss,
                checkpoint_dir=config.checkpoint_dir
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
