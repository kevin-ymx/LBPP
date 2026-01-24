"""
Functional Group-Based Molecule Sampling Script

This script:
1. Reads molecules from a root SDF file
2. Classifies molecules by functional groups using SMARTS patterns from funct_groups.csv
3. Creates separate SDF files for each functional group
4. Samples 50,000 molecules with distribution proportional to functional group counts
5. Stores the final samples in a combined SDF file

Usage:
    python sampling_Eb.py --input root_molecules.sdf --output_dir ./fg_samples --final_output sampled_50k.sdf
    python sampling_Eb.py --input root_molecules.sdf --output_dir ./fg_samples --n_samples 50000
"""

import os
import sys
import argparse
import csv
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


def load_functional_groups(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load functional group names, structure, SMILES, and SMARTS patterns from CSV file.
    
    Args:
        csv_path: Path to funct_groups.csv
        
    Returns:
        Dictionary mapping functional group names to {'structure': ..., 'smiles': ..., 'smarts': ...}
    """
    functional_groups = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            fg_name = row.get('functional group', '').strip()
            structure = row.get('chemical structure', '').strip()
            smiles = row.get('SMILES', '').strip()
            smarts = row.get('SMARTS', '').strip()
            
            # Skip empty rows or rows without SMARTS
            if not fg_name or not smarts:
                continue
            
            # Validate SMARTS pattern
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                print(f"Warning: Invalid SMARTS pattern for '{fg_name}': {smarts}")
                continue
            
            functional_groups[fg_name] = {
                'structure': structure,
                'smiles': smiles,
                'smarts': smarts
            }
    
    print(f"Loaded {len(functional_groups)} functional groups")
    return functional_groups


def sanitize_filename(name: str) -> str:
    """Convert functional group name to a valid filename."""
    # Replace problematic characters
    replacements = {
        '/': '_',
        '\\': '_',
        ' ': '_',
        '(': '',
        ')': '',
        ',': '',
        '–': '-',
        '₂': '2',
        '₃': '3',
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def count_molecules_in_sdf(sdf_path: str) -> int:
    """Count total molecules in SDF file (for progress bar)."""
    count = 0
    if sdf_path.endswith('.gz'):
        import gzip
        with gzip.open(sdf_path, 'rt') as f:
            for line in f:
                if line.strip() == '$$$$':
                    count += 1
    else:
        with open(sdf_path, 'r') as f:
            for line in f:
                if line.strip() == '$$$$':
                    count += 1
    return count


def classify_molecules_by_functional_group(
    input_sdf: str,
    functional_groups: Dict[str, Dict[str, str]]
) -> Tuple[Dict[str, List[Chem.Mol]], Dict[str, int]]:
    """
    Classify molecules from SDF by functional groups.
    
    Args:
        input_sdf: Path to input SDF file
        functional_groups: Dictionary of functional group names to {'smiles': ..., 'smarts': ...}
        
    Returns:
        Tuple of (molecules_by_fg, count_by_fg)
        - molecules_by_fg: Dict mapping FG name to list of molecules
        - count_by_fg: Dict mapping FG name to count
    """
    # Compile SMARTS patterns
    patterns = {}
    for fg_name, fg_info in functional_groups.items():
        smarts = fg_info['smarts']
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            patterns[fg_name] = pattern
    
    molecules_by_fg = defaultdict(list)
    count_by_fg = defaultdict(int)
    
    # Read molecules from SDF
    print(f"\nReading molecules from {input_sdf}...")
    
    # Count total molecules for progress bar
    print("Counting molecules...")
    total_in_file = count_molecules_in_sdf(input_sdf)
    print(f"Found {total_in_file} molecules in file")
    
    # Handle both .sdf and .sdf.gz files
    if input_sdf.endswith('.gz'):
        import gzip
        supplier = Chem.ForwardSDMolSupplier(gzip.open(input_sdf, 'rb'))
    else:
        supplier = Chem.SDMolSupplier(input_sdf)
    
    total_mols = 0
    classified_mols = 0
    
    pbar = tqdm(supplier, total=total_in_file, desc="Classifying molecules", unit="mol")
    for mol in pbar:
        if mol is None:
            continue
        
        total_mols += 1
        mol_classified = False
        
        # Check each functional group
        for fg_name, pattern in patterns.items():
            if mol.HasSubstructMatch(pattern):
                molecules_by_fg[fg_name].append(mol)
                count_by_fg[fg_name] += 1
                mol_classified = True
        
        if mol_classified:
            classified_mols += 1
        
        # Update progress bar postfix
        if total_mols % 1000 == 0:
            pbar.set_postfix({
                'valid': total_mols,
                'classified': classified_mols,
                'FGs': len(count_by_fg)
            })
    
    pbar.close()
    
    print(f"\nTotal molecules read: {total_mols}")
    print(f"Molecules with at least one functional group: {classified_mols}")
    
    return dict(molecules_by_fg), dict(count_by_fg)


def save_functional_group_sdfs(
    molecules_by_fg: Dict[str, List[Chem.Mol]],
    output_dir: str
) -> Dict[str, str]:
    """
    Save molecules to separate SDF files for each functional group.
    
    Args:
        molecules_by_fg: Dictionary mapping FG names to molecule lists
        output_dir: Output directory for SDF files
        
    Returns:
        Dictionary mapping FG names to output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    
    # Calculate total molecules to write
    total_mols_to_write = sum(len(mols) for mols in molecules_by_fg.values())
    
    print(f"\nSaving functional group SDF files to {output_dir}...")
    print(f"Total molecules to write: {total_mols_to_write}")
    
    # Progress bar for overall writing progress
    pbar = tqdm(total=total_mols_to_write, desc="Writing SDF files", unit="mol")
    
    for fg_name, mols in molecules_by_fg.items():
        if len(mols) == 0:
            continue
            
        filename = sanitize_filename(fg_name) + ".sdf"
        filepath = os.path.join(output_dir, filename)
        
        writer = Chem.SDWriter(filepath)
        for mol in mols:
            writer.write(mol)
            pbar.update(1)
        writer.close()
        
        pbar.set_postfix({'current': fg_name[:20], 'count': len(mols)})
        output_paths[fg_name] = filepath
    
    pbar.close()
    
    return output_paths


def print_distribution(count_by_fg: Dict[str, int]):
    """Print the distribution of functional groups."""
    total = sum(count_by_fg.values())
    
    print("\n" + "=" * 70)
    print("FUNCTIONAL GROUP DISTRIBUTION")
    print("=" * 70)
    print(f"{'Functional Group':<40} {'Count':>10} {'Percentage':>12}")
    print("-" * 70)
    
    # Sort by count (descending)
    sorted_fg = sorted(count_by_fg.items(), key=lambda x: x[1], reverse=True)
    
    for fg_name, count in sorted_fg:
        pct = (count / total * 100) if total > 0 else 0
        print(f"{fg_name:<40} {count:>10} {pct:>11.2f}%")
    
    print("-" * 70)
    print(f"{'TOTAL (molecule instances)':<40} {total:>10}")
    print("=" * 70)


def get_mol_identifier(mol: Chem.Mol) -> Tuple[str, str, str]:
    """
    Get unique identifier for a molecule (SMILES), its name, and molecular formula.
    
    Returns:
        Tuple of (smiles, name, molecular_formula)
    """
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except:
        smiles = ""
    
    # Try to get molecule name from properties
    name = mol.GetProp('_Name') if mol.HasProp('_Name') else ""
    if not name:
        name = mol.GetProp('PUBCHEM_COMPOUND_CID') if mol.HasProp('PUBCHEM_COMPOUND_CID') else ""
    if not name:
        name = smiles[:50] if smiles else "Unknown"
    
    # Get molecular formula
    try:
        molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
    except:
        molecular_formula = ""
    
    return smiles, name, molecular_formula


def stratified_sample(
    molecules_by_fg: Dict[str, List[Chem.Mol]],
    count_by_fg: Dict[str, int],
    functional_groups: Dict[str, Dict[str, str]],
    n_samples: int,
    seed: int = 42
) -> Tuple[List[Chem.Mol], List[Dict]]:
    """
    Sample unique molecules with distribution proportional to functional group counts.
    
    Args:
        molecules_by_fg: Dictionary mapping FG names to molecule lists
        count_by_fg: Dictionary mapping FG names to counts
        functional_groups: Dictionary with FG info (smiles, smarts)
        n_samples: Total number of samples to draw
        seed: Random seed
        
    Returns:
        Tuple of (sampled_molecules, sample_info)
        - sampled_molecules: List of unique sampled molecules
        - sample_info: List of dicts with molecule and FG info
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total_count = sum(count_by_fg.values())
    if total_count == 0:
        print("Error: No molecules found in any functional group!")
        return [], []
    
    # Calculate samples per functional group (proportional to count)
    samples_per_fg = {}
    remaining_samples = n_samples
    
    # Sort by count to handle rounding (larger groups first)
    sorted_fgs = sorted(count_by_fg.items(), key=lambda x: x[1], reverse=True)
    
    for i, (fg_name, count) in enumerate(sorted_fgs):
        if i == len(sorted_fgs) - 1:
            # Last group gets remaining samples
            n_fg_samples = remaining_samples
        else:
            # Proportional allocation
            n_fg_samples = int(round(n_samples * count / total_count))
        
        # Cap at available molecules
        n_fg_samples = min(n_fg_samples, len(molecules_by_fg.get(fg_name, [])))
        samples_per_fg[fg_name] = n_fg_samples
        remaining_samples -= n_fg_samples
    
    # Print sampling plan
    print("\n" + "=" * 70)
    print("SAMPLING PLAN")
    print("=" * 70)
    print(f"{'Functional Group':<40} {'Available':>10} {'To Sample':>12}")
    print("-" * 70)
    
    for fg_name, n_samp in sorted(samples_per_fg.items(), key=lambda x: x[1], reverse=True):
        avail = len(molecules_by_fg.get(fg_name, []))
        print(f"{fg_name:<40} {avail:>10} {n_samp:>12}")
    
    print("-" * 70)
    print(f"{'TOTAL':<40} {'':<10} {sum(samples_per_fg.values()):>12}")
    print("=" * 70)
    
    # Perform sampling with uniqueness tracking
    sampled_molecules = []
    sample_info = []
    seen_smiles = set()  # Track unique molecules by canonical SMILES
    total_to_sample = sum(samples_per_fg.values())
    
    print(f"\nSampling {total_to_sample} unique molecules from {len(samples_per_fg)} functional groups...")
    
    pbar = tqdm(total=total_to_sample, desc="Sampling molecules", unit="mol")
    
    # Sort functional groups by sample count (descending) for consistent ordering
    sorted_samples_per_fg = sorted(samples_per_fg.items(), key=lambda x: x[1], reverse=True)
    
    for fg_name, n_fg_samples in sorted_samples_per_fg:
        if n_fg_samples == 0:
            continue
        
        fg_mols = molecules_by_fg.get(fg_name, [])
        if len(fg_mols) == 0:
            continue
        
        # Get FG info
        fg_info = functional_groups.get(fg_name, {})
        fg_smiles = fg_info.get('smiles', '')
        fg_smarts = fg_info.get('smarts', '')
        
        # Shuffle molecules for random selection
        fg_mols_shuffled = fg_mols.copy()
        random.shuffle(fg_mols_shuffled)
        
        sampled_count = 0
        for mol in fg_mols_shuffled:
            if sampled_count >= n_fg_samples:
                break
            
            mol_smiles, mol_name, mol_formula = get_mol_identifier(mol)
            
            # Skip if already sampled (ensure uniqueness)
            if mol_smiles in seen_smiles:
                continue
            
            # Skip if no valid SMILES
            if not mol_smiles:
                continue
            
            seen_smiles.add(mol_smiles)
            sampled_molecules.append(mol)
            
            # Record sample info
            sample_info.append({
                'molecule_name': mol_name,
                'molecule_formula': mol_formula,
                'molecule_smiles': mol_smiles,
                'functional_group_name': fg_name,
                'functional_group_smiles': fg_smiles,
                'functional_group_smarts': fg_smarts
            })
            
            sampled_count += 1
            pbar.update(1)
        
        pbar.set_postfix({'FG': fg_name[:25], 'unique': len(sampled_molecules)})
        
        # If we couldn't get enough unique samples, log warning
        if sampled_count < n_fg_samples:
            print(f"\nWarning: Only sampled {sampled_count}/{n_fg_samples} unique molecules for '{fg_name}'")
    
    pbar.close()
    
    print(f"\nTotal unique molecules sampled: {len(sampled_molecules)}")
    
    return sampled_molecules, sample_info


def save_sampled_molecules(molecules: List[Chem.Mol], output_path: str):
    """Save sampled molecules to an SDF file."""
    print(f"\nSaving {len(molecules)} sampled molecules to {output_path}...")
    
    writer = Chem.SDWriter(output_path)
    for mol in tqdm(molecules, desc="Writing molecules"):
        writer.write(mol)
    writer.close()
    
    print(f"Successfully saved {len(molecules)} molecules!")


def save_distribution_csv(count_by_fg: Dict[str, int], output_path: str):
    """Save the distribution to a CSV file."""
    total = sum(count_by_fg.values())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['functional_group', 'count', 'percentage'])
        
        for fg_name, count in sorted(count_by_fg.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            writer.writerow([fg_name, count, f"{pct:.2f}"])
        
        writer.writerow(['TOTAL', total, '100.00'])
    
    print(f"Distribution saved to {output_path}")


def save_sample_details_csv(sample_info: List[Dict], output_path: str):
    """
    Save detailed sample information to CSV file, organized by functional group.
    
    Args:
        sample_info: List of dicts with molecule and FG info
        output_path: Output CSV file path
    """
    if not sample_info:
        print("Warning: No sample info to save!")
        return
    
    # Sort by functional group name for organized output
    sorted_info = sorted(sample_info, key=lambda x: (x['functional_group_name'], x['molecule_name']))
    
    print(f"\nSaving sample details to {output_path}...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            'molecule_name',
            'molecule_formula',
            'molecule_smiles',
            'functional_group_name',
            'functional_group_smiles',
            'functional_group_smarts'
        ])
        
        # Write data rows
        for info in tqdm(sorted_info, desc="Writing CSV", unit="row"):
            writer.writerow([
                info['molecule_name'],
                info['molecule_formula'],
                info['molecule_smiles'],
                info['functional_group_name'],
                info['functional_group_smiles'],
                info['functional_group_smarts']
            ])
    
    # Print summary by functional group
    fg_counts = defaultdict(int)
    for info in sample_info:
        fg_counts[info['functional_group_name']] += 1
    
    print(f"\nSample details saved: {len(sample_info)} molecules")
    print(f"Functional groups represented: {len(fg_counts)}")
    print(f"Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample molecules by functional group from SDF file"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input root SDF file (can be .sdf or .sdf.gz)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./fg_samples",
        help="Output directory for functional group SDF files"
    )
    parser.add_argument(
        "--final_output", "-f",
        type=str,
        default="sampled_molecules.sdf",
        help="Output path for final sampled molecules SDF"
    )
    parser.add_argument(
        "--fg_csv",
        type=str,
        default=None,
        help="Path to functional groups CSV (default: funct_groups.csv in same directory)"
    )
    parser.add_argument(
        "--n_samples", "-n",
        type=int,
        default=50000,
        help="Number of molecules to sample (default: 50000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--skip_fg_sdfs",
        action="store_true",
        help="Skip creating individual functional group SDF files"
    )
    
    args = parser.parse_args()
    
    # Determine functional groups CSV path
    if args.fg_csv is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.fg_csv = os.path.join(script_dir, "funct_groups.csv")
    
    if not os.path.exists(args.fg_csv):
        print(f"Error: Functional groups CSV not found: {args.fg_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input SDF file not found: {args.input}")
        sys.exit(1)
    
    # Load functional groups
    print("=" * 70)
    print("FUNCTIONAL GROUP MOLECULE SAMPLING")
    print("=" * 70)
    print(f"\nInput SDF: {args.input}")
    print(f"Functional groups CSV: {args.fg_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Final output: {args.final_output}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    
    # Step 1: Load functional groups
    print("\n[Step 1] Loading functional groups...")
    functional_groups = load_functional_groups(args.fg_csv)
    
    if len(functional_groups) == 0:
        print("Error: No valid functional groups loaded!")
        sys.exit(1)
    
    # Step 2: Classify molecules by functional group
    print("\n[Step 2] Classifying molecules by functional group...")
    molecules_by_fg, count_by_fg = classify_molecules_by_functional_group(
        args.input, functional_groups
    )
    
    if len(molecules_by_fg) == 0:
        print("Error: No molecules matched any functional group!")
        sys.exit(1)
    
    # Print distribution
    print_distribution(count_by_fg)
    
    # Step 3: Save functional group SDF files (optional)
    if not args.skip_fg_sdfs:
        print("\n[Step 3] Saving functional group SDF files...")
        os.makedirs(args.output_dir, exist_ok=True)
        fg_paths = save_functional_group_sdfs(molecules_by_fg, args.output_dir)
        
        # Save distribution CSV
        dist_csv_path = os.path.join(args.output_dir, "distribution.csv")
        save_distribution_csv(count_by_fg, dist_csv_path)
    else:
        print("\n[Step 3] Skipping functional group SDF files (--skip_fg_sdfs)")
    
    # Step 4: Stratified sampling (unique molecules only)
    print(f"\n[Step 4] Sampling {args.n_samples} unique molecules...")
    sampled_molecules, sample_info = stratified_sample(
        molecules_by_fg, count_by_fg, functional_groups, args.n_samples, args.seed
    )
    
    if len(sampled_molecules) == 0:
        print("Error: No molecules sampled!")
        sys.exit(1)
    
    # Step 5: Save final sampled molecules and details CSV
    print("\n[Step 5] Saving sampled molecules...")
    
    # Ensure output directory exists
    final_output_dir = os.path.dirname(args.final_output)
    if final_output_dir:
        os.makedirs(final_output_dir, exist_ok=True)
    
    save_sampled_molecules(sampled_molecules, args.final_output)
    
    # Save sample details CSV (organized by functional group)
    details_csv_path = args.final_output.replace('.sdf', '_details.csv')
    if details_csv_path == args.final_output:
        details_csv_path = args.final_output + '_details.csv'
    save_sample_details_csv(sample_info, details_csv_path)
    
    print("\n" + "=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)
    print(f"Total unique sampled molecules: {len(sampled_molecules)}")
    print(f"Output SDF file: {args.final_output}")
    print(f"Sample details CSV: {details_csv_path}")
    
    if not args.skip_fg_sdfs:
        print(f"Functional group SDFs: {args.output_dir}/")
        print(f"Distribution CSV: {dist_csv_path}")


if __name__ == "__main__":
    main()
