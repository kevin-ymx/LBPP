"""
Training script for contrastive self-supervised learning of charge-aware molecular representation.
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
from dataset.molecular_graph import MolecularGraphDataset
from dataset.augmentation import SubgraphRemovalAugmentation
from models.gin_e import GINEEncoder
from utils.loss import NTXentLoss
from dataset.data_loader import create_data_loaders, split_graphs


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
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
        pbar.set_postfix({'loss': loss.item(), 'skipped': skipped_batches})
    
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
    
    # Load dataset
    print("Loading molecules from SDF file...")
    dataset = MolecularGraphDataset(config.sdf_file, max_molecules=config.max_molecules)
    print(f"Loaded {len(dataset)} molecules")
    
    # Convert molecules to graphs
    print("Converting molecules to graphs...")
    graphs = dataset.get_all_graphs()
    print(f"Created {len(graphs)} graphs")
    
    # Filter out invalid graphs
    print("Filtering invalid graphs...")
    valid_graphs = []
    invalid_count = 0
    invalid_reasons = {'zero_nodes': 0, 'no_edges': 0, 'size_mismatch': 0, 'nan_features': 0, 'too_small': 0}
    
    for i, graph in enumerate(graphs):
        # Check if graph is valid
        if graph.num_nodes == 0:
            invalid_count += 1
            invalid_reasons['zero_nodes'] += 1
            continue
        if graph.num_nodes < 2:
            invalid_count += 1
            invalid_reasons['too_small'] += 1
            continue
        if graph.edge_index.size(1) == 0:
            invalid_count += 1
            invalid_reasons['no_edges'] += 1
            continue
        if graph.x.size(0) != graph.num_nodes:
            invalid_count += 1
            invalid_reasons['size_mismatch'] += 1
            continue
        # Check for NaN or Inf in features
        if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
            invalid_count += 1
            invalid_reasons['nan_features'] += 1
            continue
        if graph.edge_attr is not None:
            if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
                invalid_count += 1
                invalid_reasons['nan_features'] += 1
                continue
        valid_graphs.append(graph)
    
    graphs = valid_graphs
    print(f"Filtered out {invalid_count} invalid graphs. Remaining: {len(graphs)} valid graphs")
    if invalid_count > 0:
        print(f"  Invalid reasons: {invalid_reasons}")
    
    if len(graphs) == 0:
        raise ValueError("No valid graphs after filtering! Check your data.")
    
    # Split into train and validation sets
    print("Splitting into train and validation sets...")
    train_graphs, val_graphs = split_graphs(
        graphs,
        train_ratio=config.train_split,
        val_ratio=config.val_split,
        seed=config.seed
    )
    print(f"Training graphs: {len(train_graphs)}, Validation graphs: {len(val_graphs)}")
    
    # Create augmentation function
    augmentation = SubgraphRemovalAugmentation(
        removal_ratio=config.subgraph_removal_ratio,
        seed=config.seed
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        augmentation_fn=augmentation,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Check validation set
    if len(val_graphs) == 0:
        raise ValueError("Validation set is empty! Check your data split.")
    
    # Check data loaders
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a validation batch to see if there are issues
    if len(val_loader) > 0:
        try:
            test_batch1, test_batch2 = next(iter(val_loader))
            print(f"Test validation batch - batch1 graphs: {test_batch1.num_graphs}, batch2 graphs: {test_batch2.num_graphs}")
            if test_batch1.num_graphs == 0 or test_batch2.num_graphs == 0:
                print("WARNING: Validation batches contain zero graphs! Check augmentation function.")
        except Exception as e:
            print(f"WARNING: Error testing validation batch: {e}")
    
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
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config
        )
        
        # Validate
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            config=config
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
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
        
        # Save periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                checkpoint_dir=config.checkpoint_dir
            )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
