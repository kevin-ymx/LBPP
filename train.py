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
    
    pbar = tqdm(train_loader, desc="Training")
    for batch1, batch2 in pbar:
        # Move batches to device
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        
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
        
        # Compute loss
        loss = criterion(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
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
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch1, batch2 in pbar:
            # Move batches to device
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            
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
            
            # Compute loss
            loss = criterion(z1, z2)
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


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
    dataset = MolecularGraphDataset(config.sdf_file)
    print(f"Loaded {len(dataset)} molecules")
    
    # Convert molecules to graphs
    print("Converting molecules to graphs...")
    graphs = dataset.get_all_graphs()
    print(f"Created {len(graphs)} graphs")
    
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
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % 10 == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                checkpoint_dir=config.checkpoint_dir,
                is_best=is_best
            )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

