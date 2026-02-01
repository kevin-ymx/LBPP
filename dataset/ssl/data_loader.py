"""
Data loading utilities for creating training and validation sets.

Option 1 (use_cache=True): Load pre-augmented graph pairs from cache (built by build_graph_cache.py)
Option 2 (use_cache=False): Load CSV, convert SMILES to graphs, augment on-the-fly in memory
"""
import csv
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from rdkit import Chem
from tqdm import tqdm

from dataset.ssl.molecular_graph import MolToGraphConverter, is_valid_graph
from dataset.ssl.augmentation import SubgraphRemovalAugmentation


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


# =============================================================================
# Option 2: On-the-fly augmentation (load CSV, convert, augment in memory)
# =============================================================================

class OnTheFlyAugmentedDataset(Dataset):
    """
    Dataset for contrastive learning that applies augmentation on-the-fly.
    Stores graphs in memory and augments them when accessed.
    """
    
    def __init__(
        self, 
        graphs: List[Data], 
        augmentation: SubgraphRemovalAugmentation,
        split: str = "train"
    ):
        """
        Initialize dataset with graphs and augmentation.
        
        Args:
            graphs: List of molecular graphs.
            augmentation: Augmentation function to apply.
            split: Dataset split ("train" or "val").
        """
        self.graphs = graphs
        self.augmentation = augmentation
        self.split = split
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, Data]:
        """
        Get an augmented pair of graphs (augmented on-the-fly).
        
        Args:
            idx: Index of the graph.
            
        Returns:
            Tuple of two augmented graphs.
        """
        graph = self.graphs[idx]
        # Apply augmentation to create two views
        graph1, graph2 = self.augmentation(graph)
        return graph1, graph2


def load_graphs_from_csv(
    csv_file: str,
    max_molecules: Optional[int] = None,
    seed: int = 42
) -> List[Data]:
    """
    Load molecules from CSV file and convert to graphs.
    
    Args:
        csv_file: Path to CSV file with SMILES column.
        max_molecules: Maximum number of molecules to load (None = all).
        seed: Random seed for reproducibility.
        
    Returns:
        List of valid molecular graphs.
    """
    random.seed(seed)
    
    print(f"Loading molecules from CSV: {csv_file}")
    if max_molecules:
        print(f"  (Limited to {max_molecules:,} molecules)")
    
    # Count total rows for progress bar
    with open(csv_file, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    print(f"  Total rows in CSV: {total_rows:,}")
    
    converter = MolToGraphConverter()
    graphs = []
    invalid_smiles = 0
    invalid_graphs = 0
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_rows, desc="Loading & converting"):
            smiles = row.get('SMILES', '').strip()
            
            if not smiles:
                invalid_smiles += 1
                continue
            
            # Convert SMILES to Mol
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles += 1
                continue
            
            # Convert Mol to graph
            try:
                graph = converter.convert(mol)
                if is_valid_graph(graph):
                    graphs.append(graph)
                else:
                    invalid_graphs += 1
            except Exception:
                invalid_graphs += 1
            
            # Check limit
            if max_molecules and len(graphs) >= max_molecules:
                print(f"  Reached limit of {max_molecules:,} molecules")
                break
    
    print(f"  Loaded {len(graphs):,} valid graphs")
    print(f"  Skipped: {invalid_smiles:,} invalid SMILES, {invalid_graphs:,} invalid graphs")
    
    return graphs


def prepare_inmemory_data(
    csv_file: str,
    train_ratio: float = 0.8,
    max_molecules: Optional[int] = None,
    removal_ratio: float = 0.25,
    seed: int = 42
) -> Tuple[List[Data], List[Data], SubgraphRemovalAugmentation]:
    """
    Load CSV, convert to graphs, and split into train/val sets (Option 2).
    
    Args:
        csv_file: Path to CSV file with SMILES column.
        train_ratio: Ratio of training data (default: 0.8).
        max_molecules: Maximum number of molecules to load.
        removal_ratio: Subgraph removal ratio for augmentation.
        seed: Random seed.
        
    Returns:
        Tuple of (train_graphs, val_graphs, augmentation).
    """
    # Load all graphs
    graphs = load_graphs_from_csv(csv_file, max_molecules=max_molecules, seed=seed)
    
    if len(graphs) == 0:
        raise ValueError(f"No valid graphs loaded from {csv_file}")
    
    # Split into train/val
    print(f"\nSplitting into train ({train_ratio*100:.0f}%) and val ({(1-train_ratio)*100:.0f}%)...")
    train_graphs, val_graphs = split_graphs(
        graphs, 
        train_ratio=train_ratio, 
        val_ratio=1-train_ratio, 
        seed=seed
    )
    print(f"  Train: {len(train_graphs):,} graphs")
    print(f"  Val: {len(val_graphs):,} graphs")
    
    # Create augmentation
    augmentation = SubgraphRemovalAugmentation(removal_ratio=removal_ratio, seed=seed)
    
    return train_graphs, val_graphs, augmentation


def create_inmemory_train_loader(
    train_graphs: List[Data],
    augmentation: SubgraphRemovalAugmentation,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create training DataLoader with on-the-fly augmentation (Option 2)."""
    ds = OnTheFlyAugmentedDataset(train_graphs, augmentation, split="train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True,
    )


def create_inmemory_val_loader(
    val_graphs: List[Data],
    augmentation: SubgraphRemovalAugmentation,
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    """Create validation DataLoader with on-the-fly augmentation (Option 2)."""
    ds = OnTheFlyAugmentedDataset(val_graphs, augmentation, split="val")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True,
    )


# =============================================================================
# Option 2 with fixed augmentation: Pre-augment once, reuse same pairs each epoch
# =============================================================================

def preaugment_graphs(
    graphs: List[Data],
    augmentation: SubgraphRemovalAugmentation,
    desc: str = "Pre-augmenting"
) -> List[Tuple[Data, Data]]:
    """
    Pre-augment all graphs once and return list of (graph1, graph2) pairs.
    
    Args:
        graphs: List of molecular graphs.
        augmentation: Augmentation function to apply.
        desc: Description for progress bar.
        
    Returns:
        List of (graph1, graph2) tuples.
    """
    pairs = []
    skipped = 0
    
    for graph in tqdm(graphs, desc=desc):
        try:
            graph1, graph2 = augmentation(graph)
            if is_valid_graph(graph1) and is_valid_graph(graph2):
                pairs.append((graph1, graph2))
            else:
                skipped += 1
        except Exception:
            skipped += 1
    
    if skipped > 0:
        print(f"  Skipped {skipped:,} invalid augmentations")
    
    return pairs


def prepare_inmemory_data_fixed(
    csv_file: str,
    train_ratio: float = 0.8,
    max_molecules: Optional[int] = None,
    removal_ratio: float = 0.25,
    seed: int = 42
) -> Tuple[List[Tuple[Data, Data]], List[Tuple[Data, Data]]]:
    """
    Load CSV, convert to graphs, split, and pre-augment (Option 2 fixed).
    
    Args:
        csv_file: Path to CSV file with SMILES column.
        train_ratio: Ratio of training data (default: 0.8).
        max_molecules: Maximum number of molecules to load.
        removal_ratio: Subgraph removal ratio for augmentation.
        seed: Random seed.
        
    Returns:
        Tuple of (train_pairs, val_pairs) - pre-augmented graph pairs.
    """
    # Load all graphs
    graphs = load_graphs_from_csv(csv_file, max_molecules=max_molecules, seed=seed)
    
    if len(graphs) == 0:
        raise ValueError(f"No valid graphs loaded from {csv_file}")
    
    # Split into train/val
    print(f"\nSplitting into train ({train_ratio*100:.0f}%) and val ({(1-train_ratio)*100:.0f}%)...")
    train_graphs, val_graphs = split_graphs(
        graphs, 
        train_ratio=train_ratio, 
        val_ratio=1-train_ratio, 
        seed=seed
    )
    print(f"  Train: {len(train_graphs):,} graphs")
    print(f"  Val: {len(val_graphs):,} graphs")
    
    # Create augmentation
    augmentation = SubgraphRemovalAugmentation(removal_ratio=removal_ratio, seed=seed)
    
    # Pre-augment all graphs
    print(f"\nPre-augmenting graphs (fixed pairs for all epochs)...")
    train_pairs = preaugment_graphs(train_graphs, augmentation, desc="Pre-augmenting train")
    val_pairs = preaugment_graphs(val_graphs, augmentation, desc="Pre-augmenting val")
    
    print(f"\nPre-augmented pairs:")
    print(f"  Train: {len(train_pairs):,} pairs")
    print(f"  Val: {len(val_pairs):,} pairs")
    
    # Free original graphs
    del graphs, train_graphs, val_graphs
    
    return train_pairs, val_pairs


def create_fixed_train_loader(
    train_pairs: List[Tuple[Data, Data]],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create training DataLoader from pre-augmented pairs (Option 2 fixed)."""
    ds = PreAugmentedDataset(train_pairs, split="train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True,
    )


def create_fixed_val_loader(
    val_pairs: List[Tuple[Data, Data]],
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    """Create validation DataLoader from pre-augmented pairs (Option 2 fixed)."""
    ds = PreAugmentedDataset(val_pairs, split="val")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=True,
    )
