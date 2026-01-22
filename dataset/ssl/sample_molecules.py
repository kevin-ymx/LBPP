"""
Script to combine .sdf.gz files and sample/visualize molecular structures.

Usage:
    # Combine all .sdf.gz files into one
    python sample_molecules.py combine --input_dir /path/to/sdfs --output combined.sdf.gz
    
    # Sample and visualize from a file
    python sample_molecules.py sample --input combined.sdf.gz --num_samples 10 --output samples.png
"""

import argparse
import gzip
import os
import random
from glob import glob

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors


def load_molecules_from_gzip_sdf(filepath, max_mols=None):
    """Load molecules from a gzipped SDF file."""
    molecules = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return molecules
    
    print(f"Loading from {os.path.basename(filepath)}...")
    
    with gzip.open(filepath, 'rb') as gz_file:
        suppl = Chem.ForwardSDMolSupplier(gz_file)
        for mol in suppl:
            if mol is not None:
                molecules.append(mol)
                if max_mols and len(molecules) >= max_mols:
                    break
    
    print(f"  Loaded {len(molecules)} molecules")
    return molecules


def fast_sample_from_gzip_sdf(filepath, num_samples, seed=42):
    """
    Fast reservoir sampling from a gzipped SDF file.
    Streams through the file without loading all molecules into memory.
    
    Uses reservoir sampling algorithm - O(n) time, O(k) space where k = num_samples.
    """
    random.seed(seed)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return []
    
    print(f"Fast sampling {num_samples} molecules from {os.path.basename(filepath)}...")
    print("(Streaming through file with reservoir sampling...)")
    
    reservoir = []  # Will hold our sampled molecules
    count = 0
    
    with gzip.open(filepath, 'rb') as gz_file:
        suppl = Chem.ForwardSDMolSupplier(gz_file)
        
        for mol in suppl:
            if mol is None:
                continue
            
            count += 1
            
            # Reservoir sampling algorithm
            if len(reservoir) < num_samples:
                # Fill reservoir first
                reservoir.append(mol)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, count - 1)
                if j < num_samples:
                    reservoir[j] = mol
            
            # Progress indicator every 100k molecules
            if count % 100000 == 0:
                print(f"  Processed {count:,} molecules...")
    
    print(f"  Total molecules in file: {count:,}")
    print(f"  Sampled: {len(reservoir)} molecules")
    
    return reservoir


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


def combine_sdf_files(input_dir, output_path):
    """Combine all .sdf.gz files in a directory into one file."""
    # Find all .sdf.gz files
    pattern = os.path.join(input_dir, "*.sdf.gz")
    files = sorted(glob(pattern))
    
    if len(files) == 0:
        print(f"No .sdf.gz files found in {input_dir}")
        return
    
    print(f"Found {len(files)} .sdf.gz files to combine:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    
    # Load and combine all molecules
    all_molecules = []
    for filepath in files:
        mols = load_molecules_from_gzip_sdf(filepath)
        all_molecules.extend(mols)
        print(f"  Running total: {len(all_molecules)} molecules")
    
    print(f"\nTotal molecules: {len(all_molecules)}")
    
    # Save combined file
    save_molecules_to_gzip_sdf(all_molecules, output_path)
    
    print(f"\nCombined {len(files)} files into {output_path}")
    print(f"Total molecules: {len(all_molecules)}")


def get_mol_info(mol):
    """Get basic molecular information."""
    info = {
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'mw': round(Descriptors.MolWt(mol), 2),
        'heavy_atoms': rdMolDescriptors.CalcNumHeavyAtoms(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': rdMolDescriptors.CalcNumRings(mol),
        'smiles': Chem.MolToSmiles(mol),
    }
    return info


def sample_and_visualize(input_file, num_samples, output_path, mols_per_row=5, seed=42, max_load=None, fast_mode=True):
    """Sample molecules and create a visualization grid."""
    random.seed(seed)
    
    if fast_mode:
        # Use fast reservoir sampling (doesn't load all molecules)
        sampled = fast_sample_from_gzip_sdf(input_file, num_samples, seed)
    else:
        # Load molecules (slow for large files)
        molecules = load_molecules_from_gzip_sdf(input_file, max_load)
        
        if len(molecules) == 0:
            print("No molecules to visualize!")
            return
        
        print(f"Total molecules loaded: {len(molecules)}")
        
        # Sample molecules
        if num_samples >= len(molecules):
            sampled = molecules
        else:
            sampled = random.sample(molecules, num_samples)
    
    print(f"\nSampled {len(sampled)} molecules:")
    print("=" * 80)
    
    # Print molecule info
    legends = []
    for i, mol in enumerate(sampled):
        info = get_mol_info(mol)
        print(f"\nMolecule {i+1}:")
        print(f"  Formula: {info['formula']}")
        print(f"  MW: {info['mw']}")
        print(f"  Heavy atoms: {info['heavy_atoms']}")
        print(f"  Total atoms: {info['num_atoms']}")
        print(f"  Bonds: {info['num_bonds']}")
        print(f"  Rings: {info['num_rings']}")
        print(f"  SMILES: {info['smiles'][:80]}{'...' if len(info['smiles']) > 80 else ''}")
        
        # Create legend for image
        legends.append(f"{info['formula']}\nMW={info['mw']}")
    
    # Create visualization
    print(f"\nGenerating visualization...")
    
    img = Draw.MolsToGridImage(
        sampled,
        molsPerRow=mols_per_row,
        subImgSize=(300, 300),
        legends=legends,
        returnPNG=False
    )
    
    # Save image
    img.save(output_path)
    print(f"Saved visualization to: {output_path}")
    
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Combine .sdf.gz files and sample/visualize molecular structures"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine all .sdf.gz files into one')
    combine_parser.add_argument("--input_dir", type=str, required=True,
                                help="Directory containing .sdf.gz files")
    combine_parser.add_argument("--output", type=str, required=True,
                                help="Output combined .sdf.gz file")
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Sample and visualize molecules')
    sample_parser.add_argument("--input", type=str, required=True,
                               help="Input .sdf.gz file")
    sample_parser.add_argument("--num_samples", type=int, default=10,
                               help="Number of molecules to sample (default: 10)")
    sample_parser.add_argument("--output", type=str, default="sampled_molecules.png",
                               help="Output image file (default: sampled_molecules.png)")
    sample_parser.add_argument("--mols_per_row", type=int, default=5,
                               help="Molecules per row in the grid (default: 5)")
    sample_parser.add_argument("--seed", type=int, default=42,
                               help="Random seed (default: 42)")
    sample_parser.add_argument("--max_load", type=int, default=None,
                               help="Max molecules to load - only used with --no-fast (default: all)")
    sample_parser.add_argument("--no-fast", action="store_true",
                               help="Disable fast mode (loads all molecules, slower but allows other operations)")
    
    args = parser.parse_args()
    
    if args.command == 'combine':
        combine_sdf_files(args.input_dir, args.output)
    
    elif args.command == 'sample':
        fast_mode = not getattr(args, 'no_fast', False)
        sample_and_visualize(
            args.input,
            args.num_samples,
            args.output,
            args.mols_per_row,
            args.seed,
            args.max_load,
            fast_mode
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
