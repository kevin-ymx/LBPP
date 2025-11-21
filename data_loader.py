"""
Data loading utilities for creating training and validation sets.
"""
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random
import numpy as np


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning that generates graph pairs.
    """
    
    def __init__(
        self,
        graphs: List[Data],
        augmentation_fn,
        split: str = "train"
    ):
        """
        Initialize contrastive dataset.
        
        Args:
            graphs: List of molecular graphs.
            augmentation_fn: Augmentation function that takes a graph and returns a pair.
            split: Dataset split ("train" or "val").
        """
        self.graphs = graphs
        self.augmentation_fn = augmentation_fn
        self.split = split
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, Data]:
        """
        Get a pair of augmented graphs.
        
        Args:
            idx: Index of the graph.
            
        Returns:
            Tuple of two augmented graphs.
        """
        graph = self.graphs[idx]
        graph1, graph2 = self.augmentation_fn(graph)
        return graph1, graph2


def collate_contrastive_batch(batch: List[Tuple[Data, Data]]) -> Tuple[Batch, Batch]:
    """
    Collate function for contrastive learning batches.
    Creates two separate batches from graph pairs.
    
    Args:
        batch: List of (graph1, graph2) tuples.
        
    Returns:
        Tuple of two Batched graphs.
    """
    graph1_list = [pair[0] for pair in batch]
    graph2_list = [pair[1] for pair in batch]
    
    batch1 = Batch.from_data_list(graph1_list)
    batch2 = Batch.from_data_list(graph2_list)
    
    return batch1, batch2


def create_data_loaders(
    train_graphs: List[Data],
    val_graphs: List[Data],
    augmentation_fn,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_graphs: Training graphs.
        val_graphs: Validation graphs.
        augmentation_fn: Augmentation function.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        shuffle_train: Whether to shuffle training data.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = ContrastiveDataset(train_graphs, augmentation_fn, split="train")
    val_dataset = ContrastiveDataset(val_graphs, augmentation_fn, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True
    )
    
    return train_loader, val_loader


def split_graphs(
    graphs: List[Data],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[List[Data], List[Data]]:
    """
    Split graphs into training and validation sets.
    
    Args:
        graphs: List of graphs to split.
        train_ratio: Ratio of training data.
        val_ratio: Ratio of validation data.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_graphs, val_graphs).
    """
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio must equal 1.0, got {train_ratio + val_ratio}")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Shuffle graphs
    indices = list(range(len(graphs)))
    random.shuffle(indices)
    
    # Split
    train_size = int(len(graphs) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    
    return train_graphs, val_graphs

