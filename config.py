"""
Configuration file for contrastive self-supervised learning of charge-aware molecular representation.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Data loading mode
    use_cache: bool =False  # True = Option 1 (load pre-augmented pairs from cache), False = Option 2 (load CSV, convert, augment in-memory)
    
    # Data paths
    csv_file: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/combine.csv"  # Full CSV with PUBCHEM_COMPOUND_CID, SMILES
    csv_file_sampled: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/sampled.csv"  # Sampled CSV (for train_ssl_sampled.py)
    cache_dir: str = "/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/cache"  # val.pt + train_shard_0.pt to train_shard_3.pt (pre-augmented pairs, built by build_graph_cache.py)
    max_molecules: Optional[int] = None  # Limit molecules to load (None = all). Used by visualize_tsne.py and Option 2
    
    # Augmentation (used by Option 2 and build_graph_cache.py)
    subgraph_removal_ratio: float = 0.25
    train_val_split: float = 0.8  # 80% train, 20% val (used by Option 2)
    fixed_augmentation: bool = True  # Option 2 only: True = augment once at start (same pairs each epoch), False = augment on-the-fly (fresh pairs each epoch)
    
    # Model parameters
    node_feature_dim: int = 8  # atomic_num, chirality, partial_charge, hybridization, coordination_num, valence_electrons, electronegativity, binding_tag
    edge_feature_dim: int = 2  # bond_type, bond_direction
    node_embedding_dim: int = 128
    edge_embedding_dim: int = 64
    hidden_dim: int = 256
    num_gin_layers: int = 6
    dropout: float = 0.1
    
    # GIN-E training parameters
    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    temperature: float = 0.07  # Temperature parameter for NT-Xent loss
    checkpoint_frequency: int = 3  # Save periodic checkpoint every N epochs (each epoch trains on all 4 shards)
    resume_checkpoint: str = "./checkpoints/best_model.pt"  # Path to checkpoint to resume from (e.g., "./checkpoints/best_model.pt")
    
    # Downstream model parameters
    num_property_tasks: int = 1  # Number of molecular properties to predict (1 = binding energy only)
    downstream_mlp_hidden_dim: int = 512
    downstream_mlp_dropout: float = 0.2  # Increased dropout for regularization
    downstream_task_hidden_dim: int = 256
    downstream_task_dropout: float = 0.2  # Increased dropout for regularization
    freeze_pretrained_encoder: bool = False  # Whether to freeze pretrained GIN-E encoder
    
    # Downstream data split (train/val/test)
    downstream_train_split: float = 0.7
    downstream_val_split: float = 0.2
    downstream_test_split: float = 0.1
    
    # Downstream training parameters
    downstream_batch_size: int = 64  # Larger batch for more stable gradients
    downstream_num_epochs: int = 100  # Number of training epochs
    downstream_learning_rate: float = 0.0005  # Lower learning rate for fine-tuning
    downstream_weight_decay: float = 1e-4  # Increased weight decay for regularization
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Output paths
    checkpoint_dir: str = "./checkpoints/"
    log_dir: str = "./logs"
    
    # Other
    seed: int = 42
    num_workers: int = 64  # DataLoader workers

