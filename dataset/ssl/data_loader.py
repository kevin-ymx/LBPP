"""
Data loading utilities for creating training and validation sets.
Works with pre-augmented graph pairs from build_graph_cache.py.
"""
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data


class PreAugmentedDataset(Dataset):
    """
    Dataset for contrastive learning that loads pre-augmented graph pairs.
    Each item is already a (graph1, graph2) tuple from the cache.
    """
    
    def __init__(self, pairs: List[Tuple[Data, Data]], split: str = "train"):
        """
        Initialize dataset with pre-augmented pairs.
        
        Args:
            pairs: List of (graph1, graph2) tuples.
            split: Dataset split ("train" or "val").
        """
        self.pairs = pairs
        self.split = split
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, Data]:
        """
        Get a pre-augmented pair of graphs.
        
        Args:
            idx: Index of the pair.
            
        Returns:
            Tuple of two augmented graphs.
        """
        return self.pairs[idx]


def collate_contrastive_batch(batch: List[Tuple[Data, Data]]) -> Tuple[Batch, Batch]:
    """
    Collate function for contrastive learning batches.
    Creates two separate batches from graph pairs.
    
    Args:
        batch: List of (graph1, graph2) tuples.
        
    Returns:
        Tuple of two Batched graphs.
    """
    # Filter out invalid graphs
    valid_pairs = []
    for pair in batch:
        graph1, graph2 = pair
        # Check if both graphs are valid
        if (graph1.num_nodes > 0 and graph2.num_nodes > 0 and
            graph1.x.size(0) == graph1.num_nodes and graph2.x.size(0) == graph2.num_nodes):
            valid_pairs.append(pair)
    
    if len(valid_pairs) == 0:
        # Return empty batches if no valid pairs
        # Create dummy empty batch structure
        dummy_graph = Data(
            x=torch.zeros((1, 8), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 2), dtype=torch.float),
            num_nodes=1
        )
        batch1 = Batch.from_data_list([dummy_graph])
        batch2 = Batch.from_data_list([dummy_graph])
        return batch1, batch2
    
    graph1_list = [pair[0] for pair in valid_pairs]
    graph2_list = [pair[1] for pair in valid_pairs]
    
    try:
        batch1 = Batch.from_data_list(graph1_list)
        batch2 = Batch.from_data_list(graph2_list)
    except Exception as e:
        # Fallback: if batching fails, return empty batches
        print(f"Warning: Batch collation failed: {e}")
        dummy_graph = Data(
            x=torch.zeros((1, 8), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 2), dtype=torch.float),
            num_nodes=1
        )
        batch1 = Batch.from_data_list([dummy_graph])
        batch2 = Batch.from_data_list([dummy_graph])
    
    return batch1, batch2


def create_val_loader(
    val_pairs: List[Tuple[Data, Data]],
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Create validation DataLoader from pre-augmented pairs."""
    ds = PreAugmentedDataset(val_pairs, split="val")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True,
    )


def create_train_loader(
    train_pairs: List[Tuple[Data, Data]],
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Create training DataLoader from pre-augmented pairs. No shuffle: pairs were already randomly assigned in build_graph_cache."""
    ds = PreAugmentedDataset(train_pairs, split="train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True,
    )


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
