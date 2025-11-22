"""
Filter molecules from PubChem for contrastive SSL pretraining.

Criteria applied:
- Single connected component
- Allowed atom types: H,C,N,O,S,P,F,Cl,Br,I
- Heavy atoms <= 50
- No valence errors
- No radicals
- Formal charge in {-1, 0, +1}
- Max ring size <= 8
- Rotatable bonds <= 20
- Molecular weight < 700
- No unassigned stereochemistry
- Optional: successful 3D embedding (off by default)

Usage:
python filter_mol.py --input pubchem.sdf --output filtered.sdf --max_mols 1000000 --workers 16
"""

import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

ALLOWED_ATOMS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


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

    # Heavy atom count (excluding H)
    heavy_atoms = sum(1 for a in atoms if a.GetAtomicNum() > 1)
    if heavy_atoms > 50:
        return False

    # Radicals
    if any(a.GetNumRadicalElectrons() != 0 for a in atoms):
        return False

    # Net charge
    net_charge = sum(a.GetFormalCharge() for a in atoms)
    if net_charge not in (-1, 0, 1):
        return False

    # Stereochemistry: no unassigned centers
    chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if any(center[1] == '?' for center in chiral):
        return False

    # Max ring size
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) > 8:
            return False

    # Rotatable bonds
    if Lipinski.NumRotatableBonds(mol) > 20:
        return False

    # Molecular weight
    if Descriptors.MolWt(mol) > 700:
        return False

    # Optional: test 3D embedding
    if require_3d:
        mol_3d = Chem.AddHs(mol)
        if Chem.rdDistGeom.EmbedMolecule(mol_3d, maxAttempts=10) != 0:
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
# Main script
# -------------------------

def main(args):
    print(f"Loading molecules from {args.input}")

    # Load molecules from SDF file
    suppl = Chem.SDMolSupplier(args.input)
    mol_list = [mol for mol in suppl if mol is not None]

    print(f"Total molecules loaded: {len(mol_list)}")

    # Prepare multiprocessing
    pool = mp.Pool(args.workers)
    func = partial(process_molecule, require_3d=args.require_3d)

    print("Filtering molecules...")
    valid_mols = []

    for result in tqdm(pool.imap(func, mol_list, chunksize=500),
                       total=len(mol_list)):
        if result is not None:
            valid_mols.append(result)
        if len(valid_mols) >= args.max_mols:
            break

    pool.close()
    pool.join()

    print(f"Valid molecules: {len(valid_mols)}")

    # Save results to SDF file
    writer = Chem.SDWriter(args.output)
    for mol in valid_mols:
        writer.write(mol)
    writer.close()

    print(f"Saved to {args.output}")


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True,
                        help="Input SDF file from PubChem")
    parser.add_argument("--output", type=str, required=True,
                        help="Output filtered SDF file")
    parser.add_argument("--max_mols", type=int, default=1000000,
                        help="Maximum number of valid molecules to save")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of CPU workers")
    parser.add_argument("--require_3d", action="store_true",
                        help="Filter only molecules with successful 3D conformers")

    args = parser.parse_args()

    main(args)
