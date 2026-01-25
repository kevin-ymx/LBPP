"""
Script to combine .sdf.gz files and sample/visualize molecular structures.

Usage:
    # Combine all .sdf.gz files into one (no filtering)
    python sample_molecules.py combine --input_dir /path/to/sdfs --output combined.sdf.gz
    
    # Randomly sample from each .sdf.gz file and combine (default: 10%); supports resume if interrupted
    python sample_molecules.py combine_sample --input_dir /path/to/sdfs --output combined.sdf.gz --sample_ratio 0.1
    # Re-run with same --output to resume. Use --no-resume to start over.
    
    # Sample and visualize from a file
    python sample_molecules.py sample --input combined.sdf.gz --num_samples 10 --output samples.png
"""

import argparse
import gzip
import os
import random
import shutil
from glob import glob

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from tqdm import tqdm


def _load_processed_files(checkpoint_path):
    """Load set of already-processed input file paths from checkpoint."""
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def _save_checkpoint(checkpoint_path, filepath):
    """Append one processed file path to checkpoint."""
    with open(checkpoint_path, "a") as f:
        f.write(filepath + "\n")


def _finalize_output(partial_path, output_path, checkpoint_path):
    """Compress partial SDF to final gzip and remove temporary files."""
    with open(partial_path, "rb") as f_in:
        with gzip.open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(partial_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


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


def combine_and_sample_sdf_files(input_dir, output_path, sample_ratio=0.1, seed=42, resume=True):
    """
    Combine all .sdf.gz files in a directory into one file,
    randomly sampling a percentage of molecules from each file.

    Uses streaming and a partial file so the run can be resumed if killed
    (e.g. by a job time limit). Checkpoint and .partial files are removed
    after successful completion. Re-run with the same --output to resume.

    Args:
        input_dir: Directory containing .sdf.gz files
        output_path: Output path for combined sampled .sdf.gz file
        sample_ratio: Fraction of molecules to sample from each file (default: 0.1 = 10%)
        seed: Random seed for reproducibility (default: 42)
        resume: If True, skip files already in checkpoint and append to partial (default: True)
    """
    random.seed(seed)

    checkpoint_path = output_path + ".checkpoint"
    partial_path = output_path + ".partial"

    if not resume:
        for p in (checkpoint_path, partial_path):
            if os.path.exists(p):
                os.remove(p)
                print(f"Starting fresh (--no-resume): removed {p}")

    processed = _load_processed_files(checkpoint_path) if resume else set()
    if processed and not os.path.exists(partial_path):
        print("Checkpoint exists but partial file missing (inconsistent). Starting fresh.")
        processed = set()
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    pattern = os.path.join(input_dir, "*.sdf.gz")
    all_files = sorted(glob(pattern))
    files = [f for f in all_files if f not in processed]

    if len(all_files) == 0:
        print(f"No .sdf.gz files found in {input_dir}")
        return

    if len(files) == 0:
        if processed and os.path.exists(partial_path):
            print("All files already processed. Finalizing from previous run...")
            _finalize_output(partial_path, output_path, checkpoint_path)
            print(f"Output saved to: {output_path}")
        elif processed:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
        return

    if processed:
        print(f"Resuming: {len(processed)} files already done, {len(files)} remaining")

    print(f"Found {len(all_files)} .sdf.gz files ({len(files)} to process)")
    print(f"Sample ratio: {sample_ratio*100:.1f}%")
    print(f"Random seed: {seed}")
    print("=" * 60)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_processed = 0
    total_sampled = 0
    partial_mode = "a" if processed else "w"

    with open(partial_path, partial_mode) as partial_file:
        writer = Chem.SDWriter(partial_file)
        for filepath in tqdm(files, desc="Processing files"):
            file_processed = 0
            file_sampled = 0

            with gzip.open(filepath, "rb") as gz_file:
                suppl = Chem.ForwardSDMolSupplier(gz_file)
                for mol in suppl:
                    if mol is None:
                        continue
                    file_processed += 1
                    if random.random() < sample_ratio:
                        writer.write(mol)
                        file_sampled += 1
                        total_sampled += 1

            total_processed += file_processed
            partial_file.flush()
            _save_checkpoint(checkpoint_path, filepath)
            tqdm.write(f"  {os.path.basename(filepath)}: {file_sampled:,}/{file_processed:,} sampled")
        writer.close()

    _finalize_output(partial_path, output_path, checkpoint_path)

    print("\n" + "=" * 60)
    print("SAMPLING SUMMARY")
    print("=" * 60)
    print(f"Molecules processed (this run): {total_processed:,}")
    if total_processed > 0:
        actual_ratio = total_sampled / total_processed
        print(f"Molecules sampled: {total_sampled:,} ({100*actual_ratio:.2f}%)")
        print(f"Target sample ratio: {sample_ratio*100:.1f}%")
    print("=" * 60)
    print(f"\nOutput saved to: {output_path}")


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
    
    # Combine command (no filtering)
    combine_parser = subparsers.add_parser('combine', help='Combine all .sdf.gz files into one (no filtering)')
    combine_parser.add_argument("--input_dir", type=str, required=True,
                                help="Directory containing .sdf.gz files")
    combine_parser.add_argument("--output", type=str, required=True,
                                help="Output combined .sdf.gz file")
    
    # Combine with random sampling command
    combine_sample_parser = subparsers.add_parser('combine_sample', 
                                                   help='Randomly sample from each .sdf.gz file and combine')
    combine_sample_parser.add_argument("--input_dir", type=str, required=True,
                                        help="Directory containing .sdf.gz files")
    combine_sample_parser.add_argument("--output", type=str, required=True,
                                        help="Output combined sampled .sdf.gz file")
    combine_sample_parser.add_argument("--sample_ratio", type=float, default=0.1,
                                        help="Fraction of molecules to sample from each file (default: 0.1 = 10%%)")
    combine_sample_parser.add_argument("--seed", type=int, default=42,
                                        help="Random seed for reproducibility (default: 42)")
    combine_sample_parser.add_argument("--no-resume", action="store_true",
                                        help="Ignore any checkpoint and start from scratch (overwrites partial)")
    
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
    
    elif args.command == 'combine_sample':
        resume = not getattr(args, "no_resume", False)
        combine_and_sample_sdf_files(args.input_dir, args.output, args.sample_ratio, args.seed, resume=resume)
    
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
