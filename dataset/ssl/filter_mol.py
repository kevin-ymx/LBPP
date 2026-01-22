"""
Filter molecules from PubChem for contrastive SSL pretraining.

Criteria applied:
- Single connected component
- Allowed atom types: H,C,N,O,S,P,F,Cl,Br,I
- Heavy atoms <= 40
- No valence errors
- No radicals
- Max ring size <= 6
- Molecular weight < 500

Reads multiple .sdf.gz files with pattern:
stage0_5_ha50_neutral_elem_HA01_05__Compound_000000001_000500000.sdf.gz
through
stage0_5_ha50_neutral_elem_HA36_40__Compound_000000001_000500000.sdf.gz

Combines all filtered molecules, randomly selects 20%, and saves to a single .sdf.gz file.

Usage:
python ./filter_mol.py --input_dir /global/cfs/cdirs/m3342/jhxie/database/pubchem/outputs/stage0_5_parent_ha50_neutral_elem_bins_sdf/shard__Compound_000000001_000500000 --output /pscratch/sd/y/yeming/AI4M/SSL/SDFs --sample_ratio 0.2 --workers 1
"""

import argparse
import gzip
import os
import random
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

ALLOWED_ATOMS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}

# File name pattern components
FILE_PREFIX = "stage0_5_ha50_neutral_elem_HA"
FILE_SUFFIX = "__Compound_000000001_000500000.sdf.gz"

# HA ranges: 01_05, 06_10, 11_15, 16_20, 21_25, 26_30, 31_35, 36_40
HA_RANGES = [
    "01_05", "06_10", "11_15", "16_20",
    "21_25", "26_30", "31_35", "36_40"
]


# -------------------------
# Filtering functions
# -------------------------

def passes_filters(mol, require_3d=False):
    """Apply all filtering criteria to a single RDKit molecule."""

    # Reject if sanitization fails
    try:
        Chem.SanitizeMol(mol)
    except:
        return False

    # Reject disconnected molecules
    if len(Chem.GetMolFrags(mol, asMols=True)) != 1:
        return False

    # Atom checks
    atoms = list(mol.GetAtoms())

    # Allowed elements only
    for a in atoms:
        if a.GetSymbol() not in ALLOWED_ATOMS:
            return False

    # Must contain at least one heteroatom O/N/S/P
    if not any(a.GetSymbol() in {"O", "N", "S", "P"} for a in atoms):
        return False

    # Heavy atom count (excluding H)
    heavy_atoms = sum(1 for a in atoms if a.GetAtomicNum() > 1)
    if heavy_atoms > 40:
        return False

    # Radicals
    if any(a.GetNumRadicalElectrons() != 0 for a in atoms):
        return False

    # Max ring size
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) > 6:
            return False

    # Molecular weight
    if Descriptors.MolWt(mol) > 500:
        return False

    return True


# -------------------------
# Worker wrapper
# -------------------------

def process_molecule(mol, require_3d=False):
    """Apply filters to RDKit mol object."""
    if mol is None:
        return None
    if passes_filters(mol, require_3d=require_3d):
        return mol
    return None


# -------------------------
# File handling
# -------------------------

def generate_file_paths(input_dir):
    """Generate all input file paths based on the naming pattern."""
    file_paths = []
    for ha_range in HA_RANGES:
        filename = f"{FILE_PREFIX}{ha_range}{FILE_SUFFIX}"
        filepath = os.path.join(input_dir, filename)
        file_paths.append((ha_range, filepath))
    return file_paths


def load_molecules_from_gzip_sdf(filepath):
    """Load molecules from a gzipped SDF file."""
    molecules = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return molecules
    
    print(f"Loading molecules from {os.path.basename(filepath)}...")
    
    with gzip.open(filepath, 'rb') as gz_file:
        # Use ForwardSDMolSupplier for reading from file-like objects
        suppl = Chem.ForwardSDMolSupplier(gz_file)
        for mol in suppl:
            if mol is not None:
                molecules.append(mol)
    
    print(f"  Loaded {len(molecules)} molecules")
    return molecules


def save_molecules_to_gzip_sdf(molecules, output_path):
    """Save molecules to a gzipped SDF file."""
    print(f"Saving {len(molecules)} molecules to {output_path}...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with gzip.open(output_path, 'wt') as gz_file:
        writer = Chem.SDWriter(gz_file)
        for mol in molecules:
            writer.write(mol)
        writer.close()
    
    print(f"Saved to {output_path}")


def filter_molecules_from_file(input_path, ha_range, args, pool):
    """Load and filter molecules from a single .sdf.gz file."""
    print(f"\n{'='*60}")
    print(f"Processing HA range: {ha_range}")
    print(f"{'='*60}")
    
    # Load molecules
    mol_list = load_molecules_from_gzip_sdf(input_path)
    
    if len(mol_list) == 0:
        print(f"No molecules loaded from {ha_range}, skipping...")
        return []
    
    # Filter molecules
    func = partial(process_molecule, require_3d=args.require_3d)
    valid_mols = []
    
    print(f"Filtering {len(mol_list)} molecules...")
    
    if args.workers > 1:
        for result in tqdm(pool.imap(func, mol_list, chunksize=500),
                           total=len(mol_list), desc=f"Filtering {ha_range}"):
            if result is not None:
                valid_mols.append(result)
    else:
        # Single-threaded processing
        for mol in tqdm(mol_list, desc=f"Filtering {ha_range}"):
            result = process_molecule(mol, require_3d=args.require_3d)
            if result is not None:
                valid_mols.append(result)
    
    print(f"Valid molecules: {len(valid_mols)} / {len(mol_list)} ({len(valid_mols)/len(mol_list)*100:.2f}%)")
    
    return valid_mols


# -------------------------
# Main script
# -------------------------

def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Generate all input file paths
    file_paths = generate_file_paths(args.input_dir)
    
    print(f"Found {len(file_paths)} files to process:")
    for ha_range, fp in file_paths:
        exists = "+" if os.path.exists(fp) else "-"
        print(f"  [{exists}] {os.path.basename(fp)}")
    
    # Set up multiprocessing pool
    pool = None
    if args.workers > 1:
        pool = mp.Pool(args.workers)
    
    # Process each file and collect all filtered molecules
    all_filtered_molecules = []
    files_processed = 0
    
    for ha_range, input_path in file_paths:
        if not os.path.exists(input_path):
            print(f"\nSkipping {ha_range}: file not found")
            continue
        
        # Filter molecules from this file
        valid_mols = filter_molecules_from_file(input_path, ha_range, args, pool)
        all_filtered_molecules.extend(valid_mols)
        files_processed += 1
        
        print(f"Running total: {len(all_filtered_molecules)} filtered molecules")
    
    if pool is not None:
        pool.close()
        pool.join()
    
    # Print statistics before sampling
    print(f"\n{'='*60}")
    print("FILTERING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {files_processed} / {len(file_paths)}")
    print(f"Total filtered molecules: {len(all_filtered_molecules)}")
    
    if len(all_filtered_molecules) == 0:
        print("No molecules passed the filters. Exiting.")
        return
    
    # Randomly sample molecules
    sample_size = int(len(all_filtered_molecules) * args.sample_ratio)
    print(f"\nRandomly selecting {args.sample_ratio*100:.0f}% of molecules...")
    print(f"Sample size: {sample_size} molecules")
    
    if sample_size >= len(all_filtered_molecules):
        selected_molecules = all_filtered_molecules
        print("Sample size >= total, keeping all molecules")
    else:
        selected_molecules = random.sample(all_filtered_molecules, sample_size)
        print(f"Selected {len(selected_molecules)} molecules")
    
    # Save to single output file
    save_molecules_to_gzip_sdf(selected_molecules, args.output)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {files_processed} / {len(file_paths)}")
    print(f"Total filtered molecules: {len(all_filtered_molecules)}")
    print(f"Sample ratio: {args.sample_ratio*100:.0f}%")
    print(f"Selected molecules: {len(selected_molecules)}")
    print(f"Output file: {args.output}")
    print("\nDone!")


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter molecules from multiple PubChem SDF.gz files, combine, sample, and save to single file"
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the input SDF.gz files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path for the combined filtered molecules (.sdf.gz)")
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="Ratio of filtered molecules to randomly select (default: 0.2 = 20%%)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of CPU workers for parallel filtering")
    parser.add_argument("--require_3d", action="store_true",
                        help="Filter only molecules with successful 3D conformers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")

    args = parser.parse_args()
    main(args)
