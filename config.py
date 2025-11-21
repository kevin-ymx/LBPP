"""
Configuration file for contrastive self-supervised learning of charge-aware molecular representation.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Data paths
    sdf_file: str = "data/molecules.sdf"
    train_split: float = 0.8
    val_split: float = 0.2
    
    # Augmentation
    subgraph_removal_ratio: float = 0.25
    
    # Model parameters
    node_feature_dim: int = 7  # atomic_num, chirality, partial_charge, hybridization, coordination_num, valence_electrons, electronegativity
    edge_feature_dim: int = 3  # bond_type, bond_direction, coulombic_term
    node_embedding_dim: int = 128
    edge_embedding_dim: int = 64
    hidden_dim: int = 256
    num_gin_layers: int = 5
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    temperature: float = 0.07  # Temperature parameter for NT-Xent loss
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Output paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Other
    seed: int = 42
    num_workers: int = 4

