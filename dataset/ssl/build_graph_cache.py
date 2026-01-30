"""
Preprocessing script: read CSV (CID, SMILES), assign to splits, then convert → augment → save per split.

Resumable approach with low memory usage:
  1. Check if SMILES splits already exist in cache; if so, load them instead of reading CSV
  2. If not, read CSV file, extract SMILES, assign to splits, save SMILES splits to cache
  3. For each split: skip if .pt already exists; otherwise convert SMILES → Mol → graph → augment → save

Input: CSV file with columns PUBCHEM_COMPOUND_CID, SMILES (output from sample_molecules.py sample_csv)

Output files:
  - smiles_splits.pt: Dictionary of {split_name: [smiles_list]} for resumability
  - val.pt: List of (graph1, graph2) tuples for validation
  - train_shard_0.pt to train_shard_3.pt: Lists of (graph1, graph2) tuples for training

All molecules go through augmentation before being stored in cache, which can be
directly loaded for training and validation without on-the-fly augmentation.

NOTE: This approach keeps only SMILES strings in memory (~1-1.5 GB for 7.3M molecules),
      much lighter than keeping Mol objects (~20-50 GB).
      SMILES preserves chirality info if present in the original mol.

Run from project root:
  python dataset/ssl/build_graph_cache.py --csv_file /path/to/sampled.csv --cache_dir /path/to/cache
  python -m dataset.ssl.build_graph_cache --csv_file /path/to/sampled.csv --cache_dir /path/to/cache
"""
import argparse
import csv
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

NUM_TRAIN_SHARDS = 4
SPLIT_VAL = "val"  # Use string keys for cleaner dict serialization
SMILES_CACHE_FILE = "smiles_splits.pt"


def count_csv_rows(csv_path: str) -> int:
    """Count total rows in CSV file (excluding header) for progress bar."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pre-augmented graph cache from CSV: val.pt + 4 training shards (20% val / 80% train)."
    )
    parser.add_argument("--csv_file", required=True, help="Path to CSV file with columns: PUBCHEM_COMPOUND_CID, SMILES")
    parser.add_argument("--cache_dir", required=True, help="Output directory for cache files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--removal_ratio", type=float, default=0.25, help="Subgraph removal ratio for augmentation")
    parser.add_argument("--force_resplit", action="store_true", help="Force re-reading CSV even if SMILES splits exist")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"CSV file: {args.csv_file}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Seed: {args.seed}")
    print(f"Augmentation: subgraph removal with ratio={args.removal_ratio}")
    print(f"Splitting into: 20% val, 80% train ({NUM_TRAIN_SHARDS} shards)")
    print()

    smiles_cache_path = os.path.join(args.cache_dir, SMILES_CACHE_FILE)

    # =========================================================================
    # Step 1: Load or create SMILES splits
    # =========================================================================
    if os.path.exists(smiles_cache_path) and not args.force_resplit:
        # Load existing SMILES splits
        print(f"Step 1: Loading existing SMILES splits from {smiles_cache_path}...")
        split_smiles = torch.load(smiles_cache_path, weights_only=False)
        print(f"  Validation: {len(split_smiles[SPLIT_VAL]):,} SMILES")
        for i in range(NUM_TRAIN_SHARDS):
            print(f"  Train shard {i}: {len(split_smiles[f'train_{i}']):,} SMILES")
        print()
    else:
        # Read CSV and create SMILES splits
        print("Step 1: Reading CSV, extracting SMILES, and assigning splits...")
        
        # Count total rows for progress bar
        total_rows = count_csv_rows(args.csv_file)
        print(f"  Total rows in CSV: {total_rows:,}")

        # Create split containers for SMILES
        split_smiles = {SPLIT_VAL: []}  # validation
        for i in range(NUM_TRAIN_SHARDS):
            split_smiles[f"train_{i}"] = []
        
        empty_smiles = 0
        
        with open(args.csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=total_rows, desc="Reading CSV"):
                smiles = row.get('SMILES', '').strip()
                
                if not smiles:
                    empty_smiles += 1
                    continue
                
                # Assign to split
                r = random.random()
                if r < 0.2:
                    # Validation: 20%
                    split_smiles[SPLIT_VAL].append(smiles)
                else:
                    # Training: 80% split into NUM_TRAIN_SHARDS shards
                    shard_idx = int((r - 0.2) / 0.8 * NUM_TRAIN_SHARDS)
                    shard_idx = min(shard_idx, NUM_TRAIN_SHARDS - 1)  # Safety clamp
                    split_smiles[f"train_{shard_idx}"].append(smiles)
        
        total_smiles = len(split_smiles[SPLIT_VAL]) + sum(len(split_smiles[f"train_{i}"]) for i in range(NUM_TRAIN_SHARDS))
        print(f"  Total SMILES loaded: {total_smiles:,}")
        print(f"  Empty/missing SMILES: {empty_smiles:,}")
        print(f"  Validation: {len(split_smiles[SPLIT_VAL]):,} SMILES")
        for i in range(NUM_TRAIN_SHARDS):
            print(f"  Train shard {i}: {len(split_smiles[f'train_{i}']):,} SMILES")
        
        # Save SMILES splits for resumability
        print(f"\n  Saving SMILES splits to {smiles_cache_path}...")
        torch.save(split_smiles, smiles_cache_path)
        print(f"  SMILES splits saved.")
        print()

    # =========================================================================
    # Step 2: For each split: check existence, then SMILES → Mol → graph → augment → save
    # =========================================================================
    print("Step 2: Converting SMILES to graphs, augmenting, and saving each split...")
    
    converter = MolToGraphConverter()
    augmentation = SubgraphRemovalAugmentation(removal_ratio=args.removal_ratio, seed=args.seed)
    
    total_val = 0
    total_train = 0
    skipped_splits = 0

    # Process validation set
    val_path = os.path.join(args.cache_dir, "val.pt")
    if os.path.exists(val_path):
        print(f"\n  val.pt already exists, skipping... ({val_path})")
        skipped_splits += 1
        # Try to get count from existing file for summary
        try:
            existing_val = torch.load(val_path, weights_only=False)
            total_val = len(existing_val)
            del existing_val
        except Exception:
            pass
    else:
        print(f"\nProcessing val ({len(split_smiles[SPLIT_VAL]):,} SMILES)...")
        buffer = []
        skipped = 0
        for smiles in tqdm(split_smiles[SPLIT_VAL], desc="val"):
            try:
                # SMILES → Mol
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    skipped += 1
                    continue
                
                # Mol → graph
                g = converter.convert(mol)
                if not is_valid_graph(g):
                    skipped += 1
                    continue
                
                # Augment
                g1, g2 = augmentation(g)
                if is_valid_graph(g1) and is_valid_graph(g2):
                    buffer.append((g1, g2))
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        
        torch.save(buffer, val_path)
        print(f"  val: {len(buffer):,} pairs saved (skipped {skipped:,}) -> {val_path}")
        total_val = len(buffer)
        del buffer
    
    # Clear val SMILES from memory
    if SPLIT_VAL in split_smiles:
        del split_smiles[SPLIT_VAL]

    # Process training shards one by one
    for shard_idx in range(NUM_TRAIN_SHARDS):
        shard_key = f"train_{shard_idx}"
        shard_path = os.path.join(args.cache_dir, f"train_shard_{shard_idx}.pt")
        
        if os.path.exists(shard_path):
            print(f"\n  train_shard_{shard_idx}.pt already exists, skipping... ({shard_path})")
            skipped_splits += 1
            # Try to get count from existing file for summary
            try:
                existing_shard = torch.load(shard_path, weights_only=False)
                total_train += len(existing_shard)
                del existing_shard
            except Exception:
                pass
        else:
            print(f"\nProcessing train_shard_{shard_idx} ({len(split_smiles[shard_key]):,} SMILES)...")
            buffer = []
            skipped = 0
            for smiles in tqdm(split_smiles[shard_key], desc=f"train_shard_{shard_idx}"):
                try:
                    # SMILES → Mol
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        skipped += 1
                        continue
                    
                    # Mol → graph
                    g = converter.convert(mol)
                    if not is_valid_graph(g):
                        skipped += 1
                        continue
                    
                    # Augment
                    g1, g2 = augmentation(g)
                    if is_valid_graph(g1) and is_valid_graph(g2):
                        buffer.append((g1, g2))
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
            
            torch.save(buffer, shard_path)
            print(f"  train_shard_{shard_idx}: {len(buffer):,} pairs saved (skipped {skipped:,}) -> {shard_path}")
            total_train += len(buffer)
            del buffer
        
        # Clear this shard's SMILES from memory
        if shard_key in split_smiles:
            del split_smiles[shard_key]

    print()
    print(f"Total: {total_val:,} val pairs, {total_train:,} train pairs ({NUM_TRAIN_SHARDS} shards)")
    if skipped_splits > 0:
        print(f"Skipped {skipped_splits} split(s) that already existed.")
    print(f"Cache written to {args.cache_dir}")


if __name__ == "__main__":
    main()
