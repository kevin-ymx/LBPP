"""
Preprocessing script: stream SDF, assign 20% val / 40% train1 / 40% train2,
convert to graphs, and write one file per split: val.pt, train1.pt, train2.pt.
All molecules in each split are stored together (no chunks).

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build graph cache from SDF: val.pt, train1.pt, train2.pt (20/40/40)."
    )
    parser.add_argument("--sdf_file", required=True, help="Path to .sdf or .sdf.gz")
    parser.add_argument("--cache_dir", required=True, help="Output directory for val.pt, train1.pt, train2.pt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)

    converter = MolToGraphConverter()
    buf_val: list = []
    buf_t1: list = []
    buf_t2: list = []

    gz = args.sdf_file.endswith(".gz")
    opener = gzip.open if gz else open

    print(f"Reading {args.sdf_file} (gzip={gz}), seed={args.seed}")
    with opener(args.sdf_file, "rb") as f:
        supp = Chem.ForwardSDMolSupplier(f, removeHs=False)
        for mol in tqdm(supp, desc="Streaming SDF"):
            if mol is None:
                continue
            r = random.random()
            if r < 0.2:
                buf = buf_val
            elif r < 0.6:
                buf = buf_t1
            else:
                buf = buf_t2

            try:
                g = converter.convert(mol)
            except Exception:
                continue
            if not is_valid_graph(g):
                continue

            buf.append(g)

    # Save one file per split
    val_path = os.path.join(args.cache_dir, "val.pt")
    train1_path = os.path.join(args.cache_dir, "train1.pt")
    train2_path = os.path.join(args.cache_dir, "train2.pt")

    torch.save(buf_val, val_path)
    print(f"  val:   {len(buf_val):,} graphs -> {val_path}")

    torch.save(buf_t1, train1_path)
    print(f"  train1: {len(buf_t1):,} graphs -> {train1_path}")

    torch.save(buf_t2, train2_path)
    print(f"  train2: {len(buf_t2):,} graphs -> {train2_path}")

    print(f"Cache written to {args.cache_dir}")


if __name__ == "__main__":
    main()
