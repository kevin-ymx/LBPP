"""
Configuration file for contrastive self-supervised learning of charge-aware molecular representation.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Data paths
    sdf_file: str = "/pscratch/sd/y/yeming/AI4M/SSL/combined.sdf.gz"
    max_molecules: Optional[int] = 2000000  # Limit molecules to load (None = all). Set e.g. 1000000 for testing
    train_split: float = 0.8
    val_split: float = 0.2
    
    # Augmentation
    subgraph_removal_ratio: float = 0.25
    
    # Model parameters
    node_feature_dim: int = 8  # atomic_num, chirality, partial_charge, hybridization, coordination_num, valence_electrons, electronegativity, binding_tag
    edge_feature_dim: int = 3  # bond_type, bond_direction, coulombic_term
    node_embedding_dim: int = 128
    edge_embedding_dim: int = 64
    hidden_dim: int = 256
    num_gin_layers: int = 5
    dropout: float = 0.1
    
    # GIN-E training parameters
    batch_size: int = 512
    num_epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    temperature: float = 0.07  # Temperature parameter for NT-Xent loss
    
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
    
    # Distributed training (only for train_ssl_ddp.py)
    distributed: bool = False  # Set True for multi-GPU training
    num_gpus: int = 4  # Number of GPUs (for DDP)
    
    # Output paths
    checkpoint_dir: str = "./checkpoints/"
    log_dir: str = "./logs"
    
    # Other
    seed: int = 42
    num_workers: int = 16  # DataLoader workers
