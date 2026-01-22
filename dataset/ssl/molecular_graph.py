"""
Molecular graph dataset for extracting molecules from SDF files and constructing molecular graphs.
Supports both .sdf and .sdf.gz (gzipped) files.
"""
import gzip
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typing import List, Tuple, Optional
import os
from tqdm import tqdm


class MolecularGraphDataset:
    """
    Dataset for constructing molecular graphs from SDF files.
    Extracts node features: atomic number, local atom chirality, partial charges,
    hybridization, coordination number, valence electrons, electronegativity and zero-padded binding tag (for downstream prediction tasks).
    Extracts edge features: bond type, bond direction, coulombic term.
    """
    
    # Electronegativity values (Pauling scale)
    ELECTRONEGATIVITY = {
        1: 2.20,    # H
        3: 0.98,    # Li
        5: 2.04,    # B
        6: 2.55,    # C
        7: 3.04,    # N
        8: 3.44,    # O
        9: 3.98,    # F
        11: 0.93,   # Na
        12: 1.31,   # Mg
        13: 1.61,   # Al
        14: 1.90,   # Si
        15: 2.19,   # P
        16: 2.58,   # S
        17: 3.16,   # Cl
        19: 0.82,   # K
        20: 1.00,   # Ca
        34: 2.55,   # Se
        35: 2.96,   # Br
        53: 2.66,   # I
    }
    
    def __init__(self, sdf_file: str, max_molecules: int = None):
        """
        Initialize the dataset.
        
        Args:
            sdf_file: Path to the SDF file containing molecules.
            max_molecules: Maximum number of molecules to load (None = all).
        """
        self.sdf_file = sdf_file
        self.max_molecules = max_molecules
        self.molecules = []
        self._load_molecules(max_molecules)
    
    def _load_molecules(self, max_molecules: int = None):
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
    
    def _get_partial_charges(self, mol: Chem.Mol) -> List[float]:
        """
        Extract partial charges from molecule.
        If not present in properties, compute Gasteiger charges.
        Handles NaN values by replacing with 0.0.
        """
        try:
            # Try to get charges from SDF properties
            charges = []
            for atom in mol.GetAtoms():
                charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else None
                if charge is None:
                    charge = atom.GetDoubleProp('PartialCharge') if atom.HasProp('PartialCharge') else 0.0
                charges.append(charge)
            
            # If no charges found, compute Gasteiger charges
            if all(c == 0.0 for c in charges):
                from rdkit.Chem import AllChem
                AllChem.ComputeGasteigerCharges(mol)
                charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
            
            # Replace NaN values with 0.0 (Gasteiger can produce NaN for some atoms)
            charges = [0.0 if (c != c) else c for c in charges]  # NaN != NaN is True
            
            return charges
        except:
            # Fallback to zero charges
            return [0.0] * mol.GetNumAtoms()
    
    def _get_electronegativity(self, atomic_num: int) -> float:
        """Get electronegativity for an atom."""
        return self.ELECTRONEGATIVITY.get(atomic_num, 2.0)  # Default: 2.0
    
    def _get_coordination_number(self, atom: Chem.Atom, mol: Chem.Mol) -> int:
        """Calculate coordination number (number of bonded atoms)."""
        return len(atom.GetNeighbors())
    
    def _get_valence_electrons(self, atomic_num: int) -> int:
        """Get number of valence electrons."""
        # Simplified valence electron count
        valence_electron_map = {
            1: 1,    # H
            3: 1,    # Li
            5: 3,    # B
            6: 4,    # C
            7: 5,    # N
            8: 6,    # O
            9: 7,    # F
            11: 1,   # Na
            12: 2,   # Mg
            13: 3,   # Al
            14: 4,   # Si
            15: 5,   # P
            16: 6,   # S
            17: 7,   # Cl
            19: 1,   # K
            20: 2,   # Ca
            34: 6,   # Se
            35: 7,   # Br
            53: 7,   # I
        }
        return valence_electron_map.get(atomic_num, 4)  # Default: 4
    
    def _compute_coulombic_term(self, atom1_idx: int, atom2_idx: int, mol: Chem.Mol, 
                                partial_charges: List[float], bond: Chem.Bond) -> float:
        """
        Compute coulombic term for bond: q1 * q2 / r^2
        where q1, q2 are partial charges and r is bond length.
        """
        q1 = partial_charges[atom1_idx]
        q2 = partial_charges[atom2_idx]
        
        # Get bond length (approximate from bond type if exact distance not available)
        bond_length = bond.GetLength() if hasattr(bond, 'GetLength') else None
        if bond_length is None:
            # Approximate bond lengths (in Angstroms)
            bond_type_map = {
                Chem.BondType.SINGLE: 1.5,
                Chem.BondType.DOUBLE: 1.3,
                Chem.BondType.TRIPLE: 1.2,
                Chem.BondType.AROMATIC: 1.4,
            }
            bond_length = bond_type_map.get(bond.GetBondType(), 1.5)
        
        # Coulombic term: q1 * q2 / r^2 (scaled for numerical stability)
        coulombic = (q1 * q2) / (bond_length ** 2 + 1e-6)
        return coulombic
    
    def mol_to_graph(self, mol: Chem.Mol) -> Data:
        """
        Convert a molecule to a PyTorch Geometric graph.
        
        Args:
            mol: RDKit molecule object.
            
        Returns:
            Data object with node and edge features.
        """
        # Get partial charges
        partial_charges = self._get_partial_charges(mol)
        
        # Node features: [atomic_num, chirality, partial_charge, hybridization, coordination_num, valence_electrons, electronegativity, binding_tag]
        # binding_tag: 1 if atom binds with Pb, 0 if spectator (zero-padded for SSL pretraining)
        node_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            chirality = int(atom.GetChiralTag())  # ChiralTag enum value
            partial_charge = partial_charges[atom.GetIdx()]
            hybridization = int(atom.GetHybridization())  # HybridizationType enum value
            coordination_num = self._get_coordination_number(atom, mol)
            valence_electrons = self._get_valence_electrons(atomic_num)
            electronegativity = self._get_electronegativity(atomic_num)
            binding_tag = 0.0  # Zero-padded for SSL; set to 1.0 for Pb-binding atoms in downstream tasks
            
            node_features.append([
                float(atomic_num),
                float(chirality),
                float(partial_charge),
                float(hybridization),
                float(coordination_num),
                float(valence_electrons),
                float(electronegativity),
                float(binding_tag),
            ])
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Edge features: [bond_type, bond_direction, coulombic_term]
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            # Bond type
            bond_type = int(bond.GetBondType())  # BondType enum value
            
            # Bond direction
            bond_direction = int(bond.GetBondDir())  # BondDir enum value
            
            # Coulombic term
            coulombic_term = self._compute_coulombic_term(i, j, mol, partial_charges, bond)
            
            edge_feat = [
                float(bond_type),
                float(bond_direction),
                float(coulombic_term),
            ]
            
            # Add edge features for both directions
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            num_nodes=mol.GetNumAtoms()
        )
        
        return data
    
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

