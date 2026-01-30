"""
Combined molecule filter for PubChem data.

Applies all filtering criteria:
1. Single connected component
2. Allowed atom types: H,C,N,O,S,P,F,Cl,Br,I
3. Must contain at least one heteroatom (O/N/S/P)
4. Heavy atoms <= 30
5. No valence errors (sanitization must pass)
6. No radicals
7. Max ring size <= 6
8. Molecular weight < 500
9. No zwitterions (molecules with both + and - charges)

Reads multiple .sdf.gz files with pattern:
stage0_5_ha50_neutral_elem_HA01_05__Compound_XXXXXXXXX_XXXXXXXXX.sdf.gz
through
stage0_5_ha50_neutral_elem_HA26_30__Compound_XXXXXXXXX_XXXXXXXXX.sdf.gz

Output: CSV file with PUBCHEM_COMPOUND_CID and SMILES for molecules that pass all filters.

Usage:
    python filter_molecules.py --input_dir /path/to/shard_dir --output filtered.csv --workers 128
"""

import argparse
import csv
import gzip
import os
import multiprocessing as mp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

ALLOWED_ATOMS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}

# File name pattern components (modified by run_filter_mol.sh via sed)
FILE_PREFIX = "stage0_5_ha50_neutral_elem_HA"
FILE_SUFFIX = "__Compound_000000001_000500000.sdf.gz"

# HA ranges: 01_05, 06_10, 11_15, 16_20, 21_25, 26_30 (only up to 30 heavy atoms)
HA_RANGES = ["01_05", "06_10", "11_15", "16_20", "21_25", "26_30"]


# -------------------------
# Filtering functions
# -------------------------

def is_zwitterion(mol):
    """Check if a molecule has both positive and negative formal charges."""
    has_positive = False
    has_negative = False
    
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            has_positive = True
        elif charge < 0:
            has_negative = True
        
        if has_positive and has_negative:
            return True
    
    return False


def passes_all_filters(mol):
    """
    Apply all filtering criteria to a molecule.
    
    Returns:
        True if molecule passes ALL filters, False otherwise.
    """
    if mol is None:
        return False
    
    # 1. Sanitization (no valence errors)
    try:
        Chem.SanitizeMol(mol)
    except:
        return False
    
    # 2. Single connected component
    if len(Chem.GetMolFrags(mol, asMols=True)) != 1:
        return False
    
    atoms = list(mol.GetAtoms())
    
    # 3. Allowed elements only
    for a in atoms:
        if a.GetSymbol() not in ALLOWED_ATOMS:
            return False
    
    # 4. Must contain at least one heteroatom O/N/S/P
    if not any(a.GetSymbol() in {"O", "N", "S", "P"} for a in atoms):
        return False
    
    # 5. Heavy atom count <= 30
    heavy_atoms = sum(1 for a in atoms if a.GetAtomicNum() > 1)
    if heavy_atoms > 30:
        return False
    
    # 6. No radicals
    if any(a.GetNumRadicalElectrons() != 0 for a in atoms):
        return False
    
    # 7. Max ring size <= 6
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) > 6:
            return False
    
    # 8. Molecular weight < 500
    if Descriptors.MolWt(mol) > 500:
        return False
    
    # 9. No zwitterions
    if is_zwitterion(mol):
        return False
    
    return True


def extract_cid_smiles(mol):
    """
    Extract CID and SMILES from a molecule's SDF properties.
    Must be called BEFORE multiprocessing (properties are lost during pickling).
    
    Returns:
        Tuple of (cid, smiles) or (None, None) if extraction fails.
    """
    if mol is None:
        return (None, None)
    
    # Get all properties as dict (handles both string and numeric properties)
    props = mol.GetPropsAsDict()
    
    # Extract PUBCHEM_COMPOUND_CID from SDF properties
    cid = ""
    # Try common property names for CID
    cid_props = ['PUBCHEM_COMPOUND_CID', '_Name', 'CID', 'ID', 'Name', 'COMPOUND_CID']
    for prop in cid_props:
        if prop in props:
            cid = str(props[prop]).strip()
            if cid:
                break
    
    # If still empty, search all properties for one containing 'CID'
    if not cid:
        for prop, value in props.items():
            if 'CID' in prop.upper():
                cid = str(value).strip()
                if cid:
                    break
    
    # Extract SMILES from SDF properties
    smiles = ""
    # Try common property names for SMILES
    smiles_props = ['PUBCHEM_SMILES', 'SMILES', 'PUBCHEM_OPENEYE_CAN_SMILES', 'PUBCHEM_OPENEYE_ISO_SMILES']
    for prop in smiles_props:
        if prop in props:
            smiles = str(props[prop]).strip()
            if smiles:
                break
    
    # If still empty, search all properties for one containing 'SMILES'
    if not smiles:
        for prop, value in props.items():
            if 'SMILES' in prop.upper():
                smiles = str(value).strip()
                if smiles:
                    break
    
    # Final fallback: generate canonical SMILES
    if not smiles:
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
    
    return (cid, smiles)


def process_molecule_with_metadata(mol_data):
    """
    Process a molecule with pre-extracted CID and SMILES.
    Called via multiprocessing - mol_data is (mol, cid, smiles).
    
    Returns:
        Tuple of (cid, smiles) if passes filters, None otherwise.
    """
    mol, cid, smiles = mol_data
    
    if mol is None:
        return None
    
    if not passes_all_filters(mol):
        return None
    
    if not smiles:
        return None
    
    return (cid, smiles)


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


def load_molecules_from_gzip_sdf(filepath, debug=False):
    """Load molecules from a gzipped SDF file."""
    molecules = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return molecules
    
    print(f"Loading molecules from {os.path.basename(filepath)}...")
    
    with gzip.open(filepath, 'rb') as gz_file:
        suppl = Chem.ForwardSDMolSupplier(gz_file)
        for i, mol in enumerate(suppl):
            if mol is not None:
                molecules.append(mol)
                
                # Debug: print properties of first molecule
                if debug and i == 0:
                    print("\n" + "="*60)
                    print("DEBUG: First molecule properties")
                    print("="*60)
                    props = mol.GetPropsAsDict()
                    print(f"Number of properties: {len(props)}")
                    print(f"Property names: {list(props.keys())}")
                    print("\nProperty values:")
                    for key, value in props.items():
                        print(f"  {key}: {repr(value)}")
                    print("="*60 + "\n")
    
    print(f"  Loaded {len(molecules):,} molecules")
    return molecules


def filter_molecules_from_file(input_path, ha_range, args, pool, debug=False):
    """Load, filter molecules, and extract (CID, SMILES) from a single .sdf.gz file."""
    print(f"\n{'='*60}")
    print(f"Processing HA range: {ha_range}")
    print(f"{'='*60}")
    
    # Load molecules
    mol_list = load_molecules_from_gzip_sdf(input_path, debug=debug)
    
    if len(mol_list) == 0:
        print(f"No molecules loaded from {ha_range}, skipping...")
        return []
    
    # Extract CID and SMILES BEFORE multiprocessing (properties are lost during pickling)
    print(f"Extracting CID and SMILES from {len(mol_list):,} molecules...")
    mol_data_list = []
    for mol in tqdm(mol_list, desc="Extracting properties"):
        cid, smiles = extract_cid_smiles(mol)
        mol_data_list.append((mol, cid, smiles))
    
    # Filter molecules
    valid_results = []
    
    print(f"Filtering {len(mol_data_list):,} molecules...")
    
    if args.workers > 1:
        for result in tqdm(pool.imap(process_molecule_with_metadata, mol_data_list, chunksize=500),
                           total=len(mol_data_list), desc=f"Filtering {ha_range}"):
            if result is not None:
                valid_results.append(result)
    else:
        for mol_data in tqdm(mol_data_list, desc=f"Filtering {ha_range}"):
            result = process_molecule_with_metadata(mol_data)
            if result is not None:
                valid_results.append(result)
    
    print(f"Valid molecules: {len(valid_results):,} / {len(mol_list):,} ({len(valid_results)/len(mol_list)*100:.2f}%)")
    
    return valid_results


def save_to_csv(data, output_path):
    """Save list of (cid, smiles) tuples to CSV."""
    print(f"\nSaving {len(data):,} molecules to {output_path}...")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['PUBCHEM_COMPOUND_CID', 'SMILES'])
        for cid, smiles in data:
            writer.writerow([cid, smiles])
    
    print(f"Saved to {output_path}")


# -------------------------
# Main script
# -------------------------

def main(args):
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
    
    # Process each file and collect all filtered results
    all_results = []
    files_processed = 0
    
    for ha_range, input_path in file_paths:
        if not os.path.exists(input_path):
            print(f"\nSkipping {ha_range}: file not found")
            continue
        
        # Filter molecules from this file (debug only on first file)
        debug_this_file = args.debug and files_processed == 0
        valid_results = filter_molecules_from_file(input_path, ha_range, args, pool, debug=debug_this_file)
        all_results.extend(valid_results)
        files_processed += 1
        
        print(f"Running total: {len(all_results):,} filtered molecules")
    
    if pool is not None:
        pool.close()
        pool.join()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("FILTERING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {files_processed} / {len(file_paths)}")
    print(f"Total filtered molecules: {len(all_results):,}")
    
    if len(all_results) == 0:
        print("No molecules passed the filters. Exiting.")
        return
    
    # Save all filtered molecules to CSV
    save_to_csv(all_results, args.output)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {files_processed} / {len(file_paths)}")
    print(f"Total filtered molecules: {len(all_results):,}")
    print(f"Output file: {args.output}")
    print("\nDone!")


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter molecules from multiple PubChem SDF.gz files and save CID + SMILES to CSV"
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the input SDF.gz files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV file path")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of CPU workers for parallel filtering")
    parser.add_argument("--debug", action="store_true",
                        help="Print properties of first molecule for debugging")

    args = parser.parse_args()
    main(args)
