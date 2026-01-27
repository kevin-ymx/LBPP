"""
Preprocessing script: stream SDF, assign 20% val / 80% train (split into 6 shards),
then process and save each split separately to avoid memory issues.

Two-pass approach for memory efficiency:
  Pass 1: Stream SDF, assign each molecule to a split (val or train shard 0-5)
  Pass 2: For each split, re-stream SDF, convert to graphs, apply augmentation, save

Output files:
  - val.pt: List of (graph1, graph2) tuples for validation
  - train_shard_0.pt to train_shard_5.pt: Lists of (graph1, graph2) tuples for training

All molecules go through augmentation before being stored in cache, which can be
directly loaded for training and validation without on-the-fly augmentation.

Run from project root:
  python dataset/ssl/build_graph_cache.py --sdf_file /path/to/file.sdf.gz --cache_dir /path/to/cache
  python -m dataset.ssl.build_graph_cache --sdf_file /path/to/file.sdf.gz --cache_dir /path/to/cache
"""
import argparse
import gzip
import os
import random
import sys

import torch
from rdkit import Chem
from tqdm import tqdm

# Ensure dataset.ssl is importable when run as script
_TOP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOP not in sys.path:
    sys.path.insert(0, _TOP)

from dataset.ssl.molecular_graph import MolToGraphConverter, is_valid_graph
from dataset.ssl.augmentation import SubgraphRemovalAugmentation

NUM_TRAIN_SHARDS = 6
SPLIT_VAL = -1  # Special marker for validation split


def assign_splits(sdf_file: str, seed: int) -> list:
    """
    First pass: Stream SDF and assign each molecule to a split.
    
    Returns:
        List of split assignments for each molecule index.
        -1 = validation, 0-5 = train shard index, None = invalid/skipped molecule
    """
    random.seed(seed)
    
    gz = sdf_file.endswith(".gz")
    opener = gzip.open if gz else open
    
    assignments = []
    
    print(f"Pass 1: Assigning molecules to splits...")
    with opener(sdf_file, "rb") as f:
        supp = Chem.ForwardSDMolSupplier(f, removeHs=False)
        for mol in tqdm(supp, desc="Assigning splits"):
            if mol is None:
                assignments.append(None)  # Invalid molecule
                continue
            
            # Assign to split based on random number
            r = random.random()
            if r < 0.2:
                # Validation: 20%
                assignments.append(SPLIT_VAL)
            else:
                # Training: 80% split into 6 shards
                # Map r in [0.2, 1.0) to shard index 0-5
                shard_idx = int((r - 0.2) / 0.8 * NUM_TRAIN_SHARDS)
                shard_idx = min(shard_idx, NUM_TRAIN_SHARDS - 1)  # Safety clamp
                assignments.append(shard_idx)
    
    # Count assignments
    val_count = sum(1 for a in assignments if a == SPLIT_VAL)
    shard_counts = [sum(1 for a in assignments if a == i) for i in range(NUM_TRAIN_SHARDS)]
    invalid_count = sum(1 for a in assignments if a is None)
    
    print(f"  Total molecules: {len(assignments):,}")
    print(f"  Invalid/skipped: {invalid_count:,}")
    print(f"  Validation: {val_count:,}")
    for i, count in enumerate(shard_counts):
        print(f"  Train shard {i}: {count:,}")
    
    return assignments


def process_split(
    sdf_file: str,
    assignments: list,
    target_split: int,
    output_path: str,
    converter: MolToGraphConverter,
    augmentation: SubgraphRemovalAugmentation,
    split_name: str
) -> int:
    """
    Process a single split: re-stream SDF, convert molecules assigned to this split
    to graphs, apply augmentation, and save to output_path.
    
    Args:
        sdf_file: Path to SDF file
        assignments: List of split assignments from first pass
        target_split: The split to process (-1 for val, 0-5 for train shards)
        output_path: Path to save the .pt file
        converter: MolToGraphConverter instance
        augmentation: SubgraphRemovalAugmentation instance
        split_name: Name for progress bar display
    
    Returns:
        Number of valid pairs saved
    """
    gz = sdf_file.endswith(".gz")
    opener = gzip.open if gz else open
    
    buffer = []
    mol_idx = 0
    skipped = 0
    
    with opener(sdf_file, "rb") as f:
        supp = Chem.ForwardSDMolSupplier(f, removeHs=False)
        for mol in tqdm(supp, desc=f"Processing {split_name}"):
            # Check if this molecule belongs to target split
            if mol_idx < len(assignments) and assignments[mol_idx] == target_split:
                if mol is not None:
                    # Convert to graph
                    try:
                        g = converter.convert(mol)
                        if is_valid_graph(g):
                            # Apply augmentation
                            g1, g2 = augmentation(g)
                            if is_valid_graph(g1) and is_valid_graph(g2):
                                buffer.append((g1, g2))
                            else:
                                skipped += 1
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
                else:
                    skipped += 1
            
            mol_idx += 1
    
    # Save buffer
    torch.save(buffer, output_path)
    print(f"  {split_name}: {len(buffer):,} pairs saved (skipped {skipped:,}) -> {output_path}")
    
    # Clear buffer to free memory
    del buffer
    
    return len(buffer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pre-augmented graph cache from SDF: val.pt + 6 training shards (20% val / 80% train)."
    )
    parser.add_argument("--sdf_file", required=True, help="Path to .sdf or .sdf.gz")
    parser.add_argument("--cache_dir", required=True, help="Output directory for cache files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--removal_ratio", type=float, default=0.25, help="Subgraph removal ratio for augmentation")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"SDF file: {args.sdf_file}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Seed: {args.seed}")
    print(f"Augmentation: subgraph removal with ratio={args.removal_ratio}")
    print(f"Splitting into: 20% val, 80% train (6 shards)")
    print()

    # Pass 1: Assign molecules to splits
    assignments = assign_splits(args.sdf_file, args.seed)
    print()

    # Create converter and augmentation
    converter = MolToGraphConverter()
    augmentation = SubgraphRemovalAugmentation(removal_ratio=args.removal_ratio, seed=args.seed)

    # Pass 2: Process each split separately
    print("Pass 2: Processing and saving each split...")
    
    total_val = 0
    total_train = 0

    # Process validation set
    val_path = os.path.join(args.cache_dir, "val.pt")
    val_count = process_split(
        sdf_file=args.sdf_file,
        assignments=assignments,
        target_split=SPLIT_VAL,
        output_path=val_path,
        converter=converter,
        augmentation=augmentation,
        split_name="val"
    )
    total_val = val_count

    # Process training shards one by one
    for shard_idx in range(NUM_TRAIN_SHARDS):
        shard_path = os.path.join(args.cache_dir, f"train_shard_{shard_idx}.pt")
        shard_count = process_split(
            sdf_file=args.sdf_file,
            assignments=assignments,
            target_split=shard_idx,
            output_path=shard_path,
            converter=converter,
            augmentation=augmentation,
            split_name=f"train_shard_{shard_idx}"
        )
        total_train += shard_count

    print()
    print(f"Total: {total_val:,} val pairs, {total_train:,} train pairs ({NUM_TRAIN_SHARDS} shards)")
    print(f"Cache written to {args.cache_dir}")


if __name__ == "__main__":
    main()
