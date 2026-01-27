"""
Molecular graph dataset for extracting molecules from SDF files and constructing molecular graphs.
Supports both .sdf and .sdf.gz (gzipped) files.
"""
import gzip
import os
from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm


def is_valid_graph(graph: Data) -> bool:
    """Check if a graph is valid for training (same criteria as train_ssl filter)."""
    if graph.num_nodes == 0 or graph.num_nodes < 2:
        return False
    if graph.edge_index.size(1) == 0:
        return False
    if graph.x.size(0) != graph.num_nodes:
        return False
    if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
        return False
    if graph.edge_attr is not None:
        if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
            return False
    return True


class MolToGraphConverter:
    """
    Converts RDKit molecules to PyTorch Geometric Data graphs.
    Used by MolecularGraphDataset and by build_graph_cache preprocessing.
    """

    ELECTRONEGATIVITY = {
        1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 11: 0.93, 12: 1.31,
        13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 19: 0.82, 20: 1.00,
        3: 0.98, 5: 2.04, 34: 2.55, 35: 2.96, 53: 2.66,
    }
    VALENCE_ELECTRONS = {
        1: 1, 3: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 11: 1, 12: 2, 13: 3, 14: 4,
        15: 5, 16: 6, 17: 7, 19: 1, 20: 2, 34: 6, 35: 7, 53: 7,
    }

    def _get_partial_charges(self, mol: Chem.Mol) -> List[float]:
        try:
            charges = []
            for atom in mol.GetAtoms():
                c = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else None
                if c is None:
                    c = atom.GetDoubleProp("PartialCharge") if atom.HasProp("PartialCharge") else 0.0
                charges.append(c)
            if all(c == 0.0 for c in charges):
                from rdkit.Chem import AllChem
                AllChem.ComputeGasteigerCharges(mol)
                charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
            charges = [0.0 if (c != c) else c for c in charges]
            return charges
        except Exception:
            return [0.0] * mol.GetNumAtoms()

    def _get_electronegativity(self, atomic_num: int) -> float:
        return self.ELECTRONEGATIVITY.get(atomic_num, 2.0)

    def _get_valence_electrons(self, atomic_num: int) -> int:
        return self.VALENCE_ELECTRONS.get(atomic_num, 4)

    def convert(self, mol: Chem.Mol) -> Data:
        partial_charges = self._get_partial_charges(mol)
        node_features = []
        for atom in mol.GetAtoms():
            an = atom.GetAtomicNum()
            node_features.append([
                float(an), float(atom.GetChiralTag()), float(partial_charges[atom.GetIdx()]),
                float(atom.GetHybridization()), float(len(atom.GetNeighbors())),
                float(self._get_valence_electrons(an)), float(self._get_electronegativity(an)), 0.0
            ])
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index, edge_features = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            fe = [float(bond.GetBondType()), float(bond.GetBondDir())]
            edge_features.extend([fe, fe])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, num_nodes=mol.GetNumAtoms())


class MolecularGraphDataset:
    """
    Dataset for constructing molecular graphs from SDF files.
    Extracts node features: atomic number, local atom chirality, partial charges,
    hybridization, coordination number, valence electrons, electronegativity and zero-padded binding tag (for downstream prediction tasks).
    Extracts edge features: bond type, bond direction.
    """

    def __init__(self, sdf_file: str, max_molecules: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            sdf_file: Path to the SDF file containing molecules.
            max_molecules: Maximum number of molecules to load (None = all).
        """
        self.sdf_file = sdf_file
        self.max_molecules = max_molecules
        self.molecules = []
        self._converter = MolToGraphConverter()
        self._load_molecules(max_molecules)

    def _load_molecules(self, max_molecules: Optional[int] = None) -> None:
        """Load molecules from SDF file. Supports both .sdf and .sdf.gz files."""
        if not os.path.exists(self.sdf_file):
            raise FileNotFoundError(f"SDF file not found: {self.sdf_file}")
        
        print(f"Loading molecules from {self.sdf_file}...")
        if max_molecules:
            print(f"  (Limited to {max_molecules:,} molecules)")
        
        count = 0
        
        # Check if file is gzipped
        if self.sdf_file.endswith('.gz'):
            # Use gzip + ForwardSDMolSupplier for .sdf.gz files
            with gzip.open(self.sdf_file, 'rb') as gz_file:
                supplier = Chem.ForwardSDMolSupplier(gz_file, removeHs=False)
                for mol in supplier:
                    if mol is not None:
                        self.molecules.append(mol)
                        count += 1
                        
                        # Progress update every 100k molecules
                        if count % 100000 == 0:
                            print(f"  Loaded {count:,} molecules...")
                        
                        # Check limit
                        if max_molecules and count >= max_molecules:
                            print(f"  Reached limit of {max_molecules:,} molecules")
                            break
        else:
            # Use SDMolSupplier for regular .sdf files
            supplier = Chem.SDMolSupplier(self.sdf_file, removeHs=False)
            for mol in tqdm(supplier, desc="Loading molecules"):
                if mol is not None:
                    self.molecules.append(mol)
                    count += 1
                    
                    if max_molecules and count >= max_molecules:
                        break
        
        print(f"Loaded {len(self.molecules):,} molecules")

    def mol_to_graph(self, mol: Chem.Mol) -> Data:
        """Convert a molecule to a PyTorch Geometric graph."""
        return self._converter.convert(mol)

    def get_all_graphs(self) -> List[Data]:
        """
        Convert all molecules to graphs.
        
        Returns:
            List of Data objects.
        """
        graphs = []
        for mol in tqdm(self.molecules, desc="Converting to graphs"):
            try:
                graph = self.mol_to_graph(mol)
                graphs.append(graph)
            except Exception as e:
                # Skip molecules that fail conversion
                continue
        return graphs
    
    def __len__(self) -> int:
        return len(self.molecules)
    
    def __getitem__(self, idx: int) -> Data:
        return self.mol_to_graph(self.molecules[idx])

