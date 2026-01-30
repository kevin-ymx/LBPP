"""
Functional Group-Based Molecule Sampling Script

This script:
1. Reads molecules from a root SDF file
2. Classifies molecules by functional groups using SMARTS patterns from funct_groups.csv
   - Stores (CID, SMILES) tuples for each FG
3. Creates separate CSV files (CID, SMILES) for each functional group
4. Samples a fixed ratio (default 0.3%) from each functional group
   - FGs with 0 samples after rounding are skipped
5. Stores the final sample details in a CSV file (sorted by FG)

Usage:
    python sampling_Eb.py --input root_molecules.sdf --output_dir ./fg_samples --final_output sampled_details.csv
    python sampling_Eb.py --input root_molecules.sdf --output_dir ./fg_samples --sample_ratio 0.003
"""

import os
import sys
import argparse
import csv
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

from rdkit import Chem


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


def get_mol_cid_smiles(mol: Chem.Mol, mol_index: int = 0) -> Tuple[str, str]:
    """
    Get CID and canonical SMILES for a molecule.
    
    Args:
        mol: RDKit molecule
        mol_index: Index of molecule in file (used as fallback CID)
    
    Returns:
        Tuple of (cid, smiles)
    """
    # Get SMILES
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except:
        smiles = ""
    
    # Get CID from properties (try multiple common property names)
    cid = ""
    cid_props = ['PUBCHEM_COMPOUND_CID', '_Name', 'CID', 'ID', 'Name', 'COMPOUND_CID']
    for prop in cid_props:
        if mol.HasProp(prop):
            val = mol.GetProp(prop).strip()
            if val:
                cid = val
                break
    
    # Fallback to molecule index if no CID found
    if not cid:
        cid = f"MOL_{mol_index}"
    
    return cid, smiles


def classify_molecules_by_functional_group(
    input_sdf: str,
    functional_groups: Dict[str, Dict[str, str]]
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, int]]:
    """
    Classify molecules from SDF by functional groups.
    Stores (CID, SMILES) tuples instead of molecule objects.
    
    Args:
        input_sdf: Path to input SDF file
        functional_groups: Dictionary of functional group names to {'smiles': ..., 'smarts': ...}
        
    Returns:
        Tuple of (molecules_by_fg, count_by_fg)
        - molecules_by_fg: Dict mapping FG name to list of (CID, SMILES) tuples
        - count_by_fg: Dict mapping FG name to count
    """
    # Compile SMARTS patterns
    patterns = {}
    for fg_name, fg_info in functional_groups.items():
        smarts = fg_info['smarts']
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            patterns[fg_name] = pattern
    
    molecules_by_fg = defaultdict(list)  # FG name -> list of (CID, SMILES) tuples
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
        
        # Get CID and SMILES once per molecule (pass index as fallback CID)
        cid, smiles = get_mol_cid_smiles(mol, mol_index=total_mols)
        
        # Skip if no valid SMILES
        if not smiles:
            continue
        
        # Check each functional group
        for fg_name, pattern in patterns.items():
            if mol.HasSubstructMatch(pattern):
                # Store (CID, SMILES) tuple instead of molecule object
                molecules_by_fg[fg_name].append((cid, smiles))
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


def save_functional_group_csvs(
    molecules_by_fg: Dict[str, List[Tuple[str, str]]],
    output_dir: str
) -> Dict[str, str]:
    """
    Save (CID, SMILES) to separate CSV files for each functional group.
    
    Args:
        molecules_by_fg: Dictionary mapping FG names to (CID, SMILES) lists
        output_dir: Output directory for CSV files
        
    Returns:
        Dictionary mapping FG names to output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    
    # Calculate total entries to write
    total_entries = sum(len(mols) for mols in molecules_by_fg.values())
    
    print(f"\nSaving functional group CSV files to {output_dir}...")
    print(f"Total entries to write: {total_entries}")
    
    pbar = tqdm(total=total_entries, desc="Writing CSV files", unit="entry")
    
    for fg_name, mol_list in molecules_by_fg.items():
        if len(mol_list) == 0:
            continue
            
        filename = sanitize_filename(fg_name) + ".csv"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['CID', 'SMILES'])
            for cid, smiles in mol_list:
                writer.writerow([cid, smiles])
                pbar.update(1)
        
        pbar.set_postfix({'current': fg_name[:20], 'count': len(mol_list)})
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


def stratified_sample(
    molecules_by_fg: Dict[str, List[Tuple[str, str]]],
    count_by_fg: Dict[str, int],
    functional_groups: Dict[str, Dict[str, str]],
    sample_ratio: float = 0.003,
    seed: int = 42
) -> List[Dict]:
    """
    Sample molecules from each functional group at a fixed ratio.
    Uses random.sample() for sampling.
    
    Args:
        molecules_by_fg: Dictionary mapping FG names to (CID, SMILES) lists
        count_by_fg: Dictionary mapping FG names to counts
        functional_groups: Dictionary with FG info (smiles, smarts)
        sample_ratio: Fraction of molecules to sample from each FG (default: 0.003 = 0.3%)
        seed: Random seed
        
    Returns:
        List of dicts with CID, SMILES, functional_group_name, functional_group_smarts
    """
    random.seed(seed)
    
    total_count = sum(count_by_fg.values())
    if total_count == 0:
        print("Error: No molecules found in any functional group!")
        return []
    
    # Calculate samples per functional group (fixed ratio of each FG's count)
    samples_per_fg = {}
    
    for fg_name, count in count_by_fg.items():
        # Sample ratio of each FG's count
        n_fg_samples = int(round(count * sample_ratio))
        
        # Cap at available molecules and ensure non-negative
        available = len(molecules_by_fg.get(fg_name, []))
        n_fg_samples = max(0, min(n_fg_samples, available))
        
        # Only include FGs with samples > 0
        if n_fg_samples > 0:
            samples_per_fg[fg_name] = n_fg_samples
    
    # Print sampling plan (only FGs with samples > 0)
    print("\n" + "=" * 70)
    print(f"SAMPLING PLAN (ratio: {sample_ratio*100:.2f}%)")
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
    sample_info = []
    seen_smiles = set()  # Track unique molecules by canonical SMILES
    total_to_sample = sum(samples_per_fg.values())
    
    print(f"\nSampling {total_to_sample} molecules from {len(samples_per_fg)} functional groups...")
    
    pbar = tqdm(total=total_to_sample, desc="Sampling molecules", unit="mol")
    
    # Sort functional groups by sample count (descending) for consistent ordering
    sorted_samples_per_fg = sorted(samples_per_fg.items(), key=lambda x: x[1], reverse=True)
    
    for fg_name, n_fg_samples in sorted_samples_per_fg:
        if n_fg_samples <= 0:
            continue
        
        fg_mol_list = molecules_by_fg.get(fg_name, [])
        if len(fg_mol_list) == 0:
            continue
        
        # Get FG info
        fg_info = functional_groups.get(fg_name, {})
        fg_smarts = fg_info.get('smarts', '')
        
        # Use random.sample() to randomly select candidates
        # Sample more than needed to handle duplicates, but cap at available
        sample_size = min(len(fg_mol_list), max(1, n_fg_samples * 2))
        candidates = random.sample(fg_mol_list, sample_size)
        
        sampled_count = 0
        for cid, smiles in candidates:
            if sampled_count >= n_fg_samples:
                break
            
            # Skip if already sampled (ensure uniqueness)
            if smiles in seen_smiles:
                continue
            
            # Skip if no valid SMILES
            if not smiles:
                continue
            
            seen_smiles.add(smiles)
            
            # Record sample info: CID, SMILES, FG name, FG SMARTS
            sample_info.append({
                'CID': cid,
                'SMILES': smiles,
                'functional_group_name': fg_name,
                'functional_group_smarts': fg_smarts
            })
            
            sampled_count += 1
            pbar.update(1)
        
        pbar.set_postfix({'FG': fg_name[:25], 'unique': len(sample_info)})
        
        # If we couldn't get enough unique samples, log warning
        if sampled_count < n_fg_samples:
            print(f"\nWarning: Only sampled {sampled_count}/{n_fg_samples} unique molecules for '{fg_name}'")
    
    pbar.close()
    
    print(f"\nTotal unique molecules sampled: {len(sample_info)}")
    
    return sample_info


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
    Save sample details to CSV file, sorted by functional group.
    
    Args:
        sample_info: List of dicts with CID, SMILES, functional_group_name, functional_group_smarts
        output_path: Output CSV file path
    """
    if not sample_info:
        print("Warning: No sample info to save!")
        return
    
    # Sort by functional group name
    sorted_info = sorted(sample_info, key=lambda x: x['functional_group_name'])
    
    print(f"\nSaving sample details to {output_path}...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            'CID',
            'SMILES',
            'functional_group_name',
            'functional_group_smarts'
        ])
        
        # Write data rows
        for info in tqdm(sorted_info, desc="Writing CSV", unit="row"):
            writer.writerow([
                info['CID'],
                info['SMILES'],
                info['functional_group_name'],
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
        help="Output directory for functional group CSV files"
    )
    parser.add_argument(
        "--final_output", "-f",
        type=str,
        default="sampled_details.csv",
        help="Output path for final sample details CSV"
    )
    parser.add_argument(
        "--fg_csv",
        type=str,
        default=None,
        help="Path to functional groups CSV (default: funct_groups.csv in same directory)"
    )
    parser.add_argument(
        "--sample_ratio", "-r",
        type=float,
        default=0.003,
        help="Fraction of molecules to sample from each FG (default: 0.003 = 0.3%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--skip_fg_csvs",
        action="store_true",
        help="Skip creating individual functional group CSV files"
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
    print(f"Sample ratio: {args.sample_ratio*100:.2f}%")
    print(f"Random seed: {args.seed}")
    
    # Step 1: Load functional groups
    print("\n[Step 1] Loading functional groups...")
    functional_groups = load_functional_groups(args.fg_csv)
    
    if len(functional_groups) == 0:
        print("Error: No valid functional groups loaded!")
        sys.exit(1)
    
    # Step 2: Classify molecules by functional group (stores CID, SMILES tuples)
    print("\n[Step 2] Classifying molecules by functional group...")
    molecules_by_fg, count_by_fg = classify_molecules_by_functional_group(
        args.input, functional_groups
    )
    
    if len(molecules_by_fg) == 0:
        print("Error: No molecules matched any functional group!")
        sys.exit(1)
    
    # Print distribution
    print_distribution(count_by_fg)
    
    # Step 3: Save functional group CSV files (optional)
    if not args.skip_fg_csvs:
        print("\n[Step 3] Saving functional group CSV files...")
        os.makedirs(args.output_dir, exist_ok=True)
        fg_paths = save_functional_group_csvs(molecules_by_fg, args.output_dir)
        
        # Save distribution CSV
        dist_csv_path = os.path.join(args.output_dir, "distribution.csv")
        save_distribution_csv(count_by_fg, dist_csv_path)
    else:
        print("\n[Step 3] Skipping functional group CSV files (--skip_fg_csvs)")
    
    # Step 4: Sampling (ratio-based per FG)
    print(f"\n[Step 4] Sampling {args.sample_ratio*100:.2f}% from each functional group...")
    sample_info = stratified_sample(
        molecules_by_fg, count_by_fg, functional_groups, args.sample_ratio, args.seed
    )
    
    if len(sample_info) == 0:
        print("Error: No molecules sampled!")
        sys.exit(1)
    
    # Step 5: Save sample details CSV (sorted by FG)
    print("\n[Step 5] Saving sample details CSV...")
    
    # Ensure output directory exists
    final_output_dir = os.path.dirname(args.final_output)
    if final_output_dir:
        os.makedirs(final_output_dir, exist_ok=True)
    
    save_sample_details_csv(sample_info, args.final_output)
    
    print("\n" + "=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)
    print(f"Total unique sampled molecules: {len(sample_info)}")
    print(f"Sample details CSV: {args.final_output}")
    
    if not args.skip_fg_csvs:
        print(f"Functional group CSVs: {args.output_dir}/")
        print(f"Distribution CSV: {dist_csv_path}")


if __name__ == "__main__":
    main()
