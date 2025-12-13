"""
Inference script for predicting binding energy from SMILES and donor type.

Usage:
    python inference.py --smiles "CCO" --donor_type "hydroxyl"
    python inference.py --smiles "CC(=O)C" --donor_type "carbonyl_O"
    python inference.py --csv input.csv --output predictions.csv

Supported donor types:
    - alkoxide_O: Negatively charged oxygen [O-;H0]
    - amide: Amide (binding on O) [O]=C[N]
    - amide_carbonyl_O: Amide carbonyl oxygen (binding on O) [C](=O)[N]
    - amine_primary: Primary amine nitrogen [N;X3;H2;!$(N=*)]
    - amine_secondary: Secondary amine nitrogen [N;X3;H1;!$(N=*)]
    - amine_tertiary: Tertiary amine nitrogen [N;X3;H0;!$(N=*)]
    - aromatic_N_pyridinic: Pyridinic nitrogen in aromatic ring [n;H0]
    - aromatic_N_pyrrolic: Pyrrolic nitrogen [nH]
    - carbonyl_O: Carbonyl oxygen [O]=C
    - cooh_like: Carboxylic acid-like (binding on first O) [O]=C[O;H1]
    - ether_O: Ether oxygen [O;X2;H0;!$(O=*);!$([O-])]
    - hydroxyl: Hydroxyl oxygen [O;X2;H1;!$(O=*)]
    - imine: Imine nitrogen [N;X2]=C
    - nitrile_CN: Nitrile nitrogen [N]#C
    - phenoxide_O: Phenoxide oxygen [O-]-[c]
    - phosphine: Phosphine phosphorus [P;X3;!$(P=*)]
    - p_oxide: Phosphine oxide (binding on O) [O]=P
    - sox_like: Sulfonate-like (binding on first O) [O]=S(=O)O
    - sulfoxide: Sulfoxide (binding on O) [O]=S
    - thiocarbonyl: Thiocarbonyl sulfur [S]=C
    - thioether_S: Thioether sulfur [S;X2;H0;!$(S=*);!$([S-])]
    - thiol: Thiol sulfur [S;X2;H1]
"""
import os
import sys
import argparse
import csv
import torch
from typing import Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch

from config import Config
from models.gin_e import GINEEncoder
from models.downstream_model import DownstreamModel


# SMARTS patterns for identifying binding heteroatoms in each functional group
# Format: 'donor_type': ('SMARTS_pattern', binding_atom_index_in_match)
# binding_atom_index specifies which atom in the SMARTS match should have binding_tag=1
FUNCTIONAL_GROUP_SMARTS = {
    'alkoxide_O': ('[O-;H0]', 0),  # Negatively charged oxygen
    'amide': ('[O]=C[N]', 0),  # Amide (binding on O)
    'amide_carbonyl_O': ('[C](=O)[N]', 1),  # Amide carbonyl (binding on O, not C or N)
    'amine_primary': ('[N;X3;H2;!$(N=*)]', 0),  # Primary amine nitrogen
    'amine_secondary': ('[N;X3;H1;!$(N=*)]', 0),  # Secondary amine nitrogen  
    'amine_tertiary': ('[N;X3;H0;!$(N=*)]', 0),  # Tertiary amine nitrogen
    'aromatic_N_pyridinic': ('[n;H0]', 0),  # Pyridinic nitrogen in aromatic ring
    'aromatic_N_pyrrolic': ('[nH]', 0),  # Pyrrolic nitrogen
    'carbonyl_O': ('[O]=C', 0),  # Carbonyl oxygen
    'cooh_like': ('[O]=C[O;H1]', 0),  # Carboxylic acid-like (binding on first O)
    'ether_O': ('[O;X2;H0;!$(O=*);!$([O-])]', 0),  # Ether oxygen
    'hydroxyl': ('[O;X2;H1;!$(O=*)]', 0),  # Hydroxyl oxygen
    'imine': ('[N;X2]=C', 0),  # Imine nitrogen
    'nitrile_CN': ('[N]#C', 0),  # Nitrile nitrogen
    'phenoxide_O': ('[O-]-[c]', 0),  # Phenoxide oxygen (oxygen attached to aromatic carbon)
    'phosphine': ('[P;X3;!$(P=*)]', 0),  # Phosphine phosphorus
    'p_oxide': ('[O]=P', 0),  # Phosphine oxide (binding on O)
    'sox_like': ('[O]=S(=O)O', 0),  # Sulfonate-like (binding on first O)
    'sulfoxide': ('[O]=S', 0),  # Sulfoxide (binding on O)
    'thiocarbonyl': ('[S]=C', 0),  # Thiocarbonyl sulfur
    'thioether_S': ('[S;X2;H0;!$(S=*);!$([S-])]', 0),  # Thioether sulfur
    'thiol': ('[S;X2;H1]', 0),  # Thiol sulfur
}


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


# Valence electrons map
VALENCE_ELECTRONS = {
    1: 1, 3: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7,
    19: 1, 20: 2, 34: 6, 35: 7, 53: 7,
}


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert SMILES string to RDKit molecule with 3D coordinates.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        RDKit molecule object or None if failed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            # Try to embed 3D coordinates
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                # If embedding fails, try with more random seeds
                AllChem.EmbedMolecule(mol, maxAttempts=100, randomSeed=42)
        return mol
    except Exception as e:
        print(f"Error creating molecule from SMILES '{smiles}': {e}")
        return None


def find_binding_atom_indices(mol: Chem.Mol, donor_type: str) -> List[int]:
    """
    Find the indices of binding heteroatoms using SMARTS matching.
    
    Args:
        mol: RDKit molecule object.
        donor_type: Type of functional group.
        
    Returns:
        List of atom indices that are binding heteroatoms.
    """
    smarts_entry = FUNCTIONAL_GROUP_SMARTS.get(donor_type)
    if smarts_entry is None:
        print(f"Warning: Unknown donor type: {donor_type}")
        print(f"Available donor types: {list(FUNCTIONAL_GROUP_SMARTS.keys())}")
        return []
    
    # Unpack SMARTS pattern and binding atom index
    smarts, binding_atom_idx = smarts_entry
    
    try:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            return []
        
        matches = mol.GetSubstructMatches(pattern)
        # Return the specified binding atom from each match
        binding_indices = [match[binding_atom_idx] for match in matches if len(match) > binding_atom_idx]
        return binding_indices
    except Exception as e:
        print(f"Warning: SMARTS matching failed for {donor_type}: {e}")
        return []


def get_partial_charges(mol: Chem.Mol) -> List[float]:
    """Extract or compute Gasteiger partial charges."""
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        # Replace NaN values with 0.0
        charges = [0.0 if (c != c) else c for c in charges]
        return charges
    except:
        return [0.0] * mol.GetNumAtoms()


def compute_coulombic_term(atom1_idx: int, atom2_idx: int, partial_charges: List[float], bond: Chem.Bond) -> float:
    """Compute coulombic term for a bond."""
    q1 = partial_charges[atom1_idx]
    q2 = partial_charges[atom2_idx]
    
    bond_type_map = {
        Chem.BondType.SINGLE: 1.5,
        Chem.BondType.DOUBLE: 1.3,
        Chem.BondType.TRIPLE: 1.2,
        Chem.BondType.AROMATIC: 1.4,
    }
    bond_length = bond_type_map.get(bond.GetBondType(), 1.5)
    
    coulombic = (q1 * q2) / (bond_length ** 2 + 1e-6)
    return coulombic


def mol_to_graph(mol: Chem.Mol, binding_atom_indices: List[int] = None) -> Data:
    """
    Convert a molecule to a PyTorch Geometric graph.
    
    Args:
        mol: RDKit molecule object.
        binding_atom_indices: List of atom indices to mark as binding (binding_tag=1).
        
    Returns:
        PyTorch Geometric Data object with node and edge features.
    """
    if binding_atom_indices is None:
        binding_atom_indices = []
    
    # Get partial charges
    partial_charges = get_partial_charges(mol)
    
    # Node features: [atomic_num, chirality, partial_charge, hybridization, 
    #                 coordination_num, valence_electrons, electronegativity, binding_tag]
    node_features = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        chirality = int(atom.GetChiralTag())
        partial_charge = partial_charges[atom_idx]
        hybridization = int(atom.GetHybridization())
        coordination_num = len(atom.GetNeighbors())
        valence_electrons = VALENCE_ELECTRONS.get(atomic_num, 4)
        electronegativity = ELECTRONEGATIVITY.get(atomic_num, 2.0)
        
        # Set binding_tag to 1 if this atom is in the binding_atom_indices list
        binding_tag = 1.0 if atom_idx in binding_atom_indices else 0.0
        
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
        
        edge_index.append([i, j])
        edge_index.append([j, i])
        
        bond_type = int(bond.GetBondType())
        bond_direction = int(bond.GetBondDir())
        coulombic_term = compute_coulombic_term(i, j, partial_charges, bond)
        
        edge_feat = [float(bond_type), float(bond_direction), float(coulombic_term)]
        edge_features.append(edge_feat)
        edge_features.append(edge_feat)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        num_nodes=mol.GetNumAtoms()
    )


class BindingEnergyPredictor:
    """
    Predictor class for binding energy inference.
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        gin_e_checkpoint_path: str = None,
        device: str = None,
        config: Config = None
    ):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to the downstream model checkpoint.
            gin_e_checkpoint_path: Path to the GIN-E encoder checkpoint (optional).
            device: Device to run inference on ('cuda' or 'cpu').
            config: Config object (if None, uses default Config).
        """
        self.config = config if config is not None else Config()
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Default checkpoint paths
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir, "downstream", "downstream_best_model.pt"
            )
        
        if gin_e_checkpoint_path is None:
            gin_e_checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        
        # Load model
        self.model = self._load_model(checkpoint_path, gin_e_checkpoint_path)
        self.model.eval()
    
    def _load_model(self, checkpoint_path: str, gin_e_checkpoint_path: str) -> DownstreamModel:
        """Load the downstream model from checkpoint."""
        # Create GIN-E encoder
        gin_e_encoder = GINEEncoder(
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            node_embedding_dim=self.config.node_embedding_dim,
            edge_embedding_dim=self.config.edge_embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_gin_layers,
            dropout=self.config.dropout
        )
        
        # Create downstream model (without loading GIN-E checkpoint here, 
        # since we'll load the full model weights)
        model = DownstreamModel(
            gin_e_encoder=gin_e_encoder,
            gin_e_checkpoint_path=None,  # Don't load separately
            freeze_gin_e=True,  # Freeze for inference
            mlp_hidden_dim=self.config.downstream_mlp_hidden_dim,
            mlp_dropout=self.config.downstream_mlp_dropout,
            num_tasks=1,  # Single task: binding energy
            task_hidden_dim=self.config.downstream_task_hidden_dim,
            task_dropout=self.config.downstream_task_dropout
        )
        
        # Load downstream checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Downstream model checkpoint not found: {checkpoint_path}\n"
                f"Please train the model first using train_downstream.py"
            )
        
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            print(f"  Loaded model from epoch {epoch}, validation loss: {loss:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("  Loaded model weights")
        
        model = model.to(self.device)
        return model
    
    def predict_single(self, smiles: str, donor_type: str) -> Tuple[Optional[float], str]:
        """
        Predict binding energy for a single molecule.
        
        Args:
            smiles: SMILES string of the molecule.
            donor_type: Type of functional group for binding.
            
        Returns:
            Tuple of (predicted_energy, status_message).
            predicted_energy is None if prediction failed.
        """
        # Convert SMILES to molecule
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None, f"Failed to parse SMILES: {smiles}"
        
        # Find binding atoms
        binding_indices = find_binding_atom_indices(mol, donor_type)
        if len(binding_indices) == 0:
            status = f"Warning: No binding atoms found for donor type '{donor_type}'"
        else:
            status = f"Found {len(binding_indices)} binding atom(s)"
        
        # Convert to graph
        try:
            graph = mol_to_graph(mol, binding_indices)
        except Exception as e:
            return None, f"Failed to convert molecule to graph: {e}"
        
        # Run inference
        graph = graph.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=None  # Single molecule
            )
        
        # Extract predicted energy (handle different tensor shapes)
        if prediction.dim() > 0:
            predicted_energy = prediction.squeeze().item()
        else:
            predicted_energy = prediction.item()
        
        return predicted_energy, status
    
    def predict_batch(
        self, 
        smiles_list: List[str], 
        donor_types: List[str]
    ) -> List[Tuple[Optional[float], str]]:
        """
        Predict binding energy for a batch of molecules.
        
        Args:
            smiles_list: List of SMILES strings.
            donor_types: List of donor types (same length as smiles_list).
            
        Returns:
            List of (predicted_energy, status_message) tuples.
        """
        if len(smiles_list) != len(donor_types):
            raise ValueError("smiles_list and donor_types must have the same length")
        
        results = []
        valid_graphs = []
        valid_indices = []
        
        # Process all molecules
        for i, (smiles, donor_type) in enumerate(zip(smiles_list, donor_types)):
            mol = smiles_to_mol(smiles)
            if mol is None:
                results.append((None, f"Failed to parse SMILES: {smiles}"))
                continue
            
            binding_indices = find_binding_atom_indices(mol, donor_type)
            if len(binding_indices) == 0:
                status = f"Warning: No binding atoms found for donor type '{donor_type}'"
            else:
                status = f"Found {len(binding_indices)} binding atom(s)"
            
            try:
                graph = mol_to_graph(mol, binding_indices)
                valid_graphs.append(graph)
                valid_indices.append((i, status))
            except Exception as e:
                results.append((None, f"Failed to convert molecule to graph: {e}"))
        
        # Batch inference for valid graphs
        if len(valid_graphs) > 0:
            batched_graph = Batch.from_data_list(valid_graphs).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(
                    x=batched_graph.x,
                    edge_index=batched_graph.edge_index,
                    edge_attr=batched_graph.edge_attr,
                    batch=batched_graph.batch
                )
            
            predictions = predictions.cpu().numpy().flatten()
            
            # Merge results
            pred_idx = 0
            final_results = [None] * len(smiles_list)
            
            for i, status in valid_indices:
                final_results[i] = (float(predictions[pred_idx]), status)
                pred_idx += 1
            
            # Fill in failed results
            result_idx = 0
            for i in range(len(smiles_list)):
                if final_results[i] is None:
                    final_results[i] = results[result_idx]
                    result_idx += 1
            
            return final_results
        
        return results


def predict_from_csv(
    input_csv: str,
    output_csv: str,
    predictor: BindingEnergyPredictor,
    smiles_col: str = 'SMILES',
    donor_col: str = 'DonorType'
):
    """
    Predict binding energies from a CSV file.
    
    Args:
        input_csv: Path to input CSV file.
        output_csv: Path to output CSV file.
        predictor: BindingEnergyPredictor instance.
        smiles_col: Column name for SMILES.
        donor_col: Column name for donor type.
    """
    print(f"Reading input from {input_csv}...")
    
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"Found {len(rows)} samples")
    
    # Collect SMILES and donor types
    smiles_list = [row[smiles_col] for row in rows]
    donor_types = [row[donor_col] for row in rows]
    
    # Predict
    print("Running predictions...")
    results = predictor.predict_batch(smiles_list, donor_types)
    
    # Write output
    print(f"Writing results to {output_csv}...")
    output_fieldnames = list(fieldnames) + ['predicted_binding_energy', 'prediction_status']
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for row, (energy, status) in zip(rows, results):
            row['predicted_binding_energy'] = energy if energy is not None else 'N/A'
            row['prediction_status'] = status
            writer.writerow(row)
    
    # Summary
    successful = sum(1 for energy, _ in results if energy is not None)
    print(f"\nPrediction complete!")
    print(f"  Successful: {successful}/{len(rows)}")
    print(f"  Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict binding energy from SMILES and donor type',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported donor types:
  alkoxide_O           - Negatively charged oxygen [O-;H0]
  amide                - Amide (binding on O) [O]=C[N]
  amide_carbonyl_O     - Amide carbonyl oxygen (binding on O) [C](=O)[N]
  amine_primary        - Primary amine nitrogen [N;X3;H2;!$(N=*)]
  amine_secondary      - Secondary amine nitrogen [N;X3;H1;!$(N=*)]
  amine_tertiary       - Tertiary amine nitrogen [N;X3;H0;!$(N=*)]
  aromatic_N_pyridinic - Pyridinic nitrogen in aromatic ring [n;H0]
  aromatic_N_pyrrolic  - Pyrrolic nitrogen [nH]
  carbonyl_O           - Carbonyl oxygen [O]=C
  cooh_like            - Carboxylic acid-like (binding on first O) [O]=C[O;H1]
  ether_O              - Ether oxygen [O;X2;H0;!$(O=*);!$([O-])]
  hydroxyl             - Hydroxyl oxygen [O;X2;H1;!$(O=*)]
  imine                - Imine nitrogen [N;X2]=C
  nitrile_CN           - Nitrile nitrogen [N]#C
  phenoxide_O          - Phenoxide oxygen [O-]-[c]
  phosphine            - Phosphine phosphorus [P;X3;!$(P=*)]
  p_oxide              - Phosphine oxide (binding on O) [O]=P
  sox_like             - Sulfonate-like (binding on first O) [O]=S(=O)O
  sulfoxide            - Sulfoxide (binding on O) [O]=S
  thiocarbonyl         - Thiocarbonyl sulfur [S]=C
  thioether_S          - Thioether sulfur [S;X2;H0;!$(S=*);!$([S-])]
  thiol                - Thiol sulfur [S;X2;H1]

Examples:
  # Single prediction
  python inference.py --smiles "CCO" --donor_type "hydroxyl"
  python inference.py --smiles "CC(=O)C" --donor_type "carbonyl_O"
  python inference.py --smiles "CS(=O)C" --donor_type "sulfoxide"
  python inference.py --smiles "CC(=O)NC" --donor_type "amide"
  
  # Batch prediction from CSV
  python inference.py --csv input.csv --output predictions.csv
        """
    )
    
    # Input options
    parser.add_argument('--smiles', type=str, help='SMILES string of the molecule')
    parser.add_argument('--donor_type', type=str, help='Donor type for binding')
    parser.add_argument('--csv', type=str, help='Path to input CSV file for batch prediction')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                        help='Path to output CSV file (default: predictions.csv)')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to downstream model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu, default: auto-detect)')
    
    # CSV column options
    parser.add_argument('--smiles_col', type=str, default='SMILES',
                        help='Column name for SMILES in CSV (default: SMILES)')
    parser.add_argument('--donor_col', type=str, default='DonorType',
                        help='Column name for donor type in CSV (default: DonorType)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.csv is None and (args.smiles is None or args.donor_type is None):
        parser.error("Either --csv or both --smiles and --donor_type are required")
    
    # Initialize predictor
    print("="*60)
    print("Binding Energy Prediction")
    print("="*60)
    
    predictor = BindingEnergyPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    if args.csv:
        # Batch prediction from CSV
        predict_from_csv(
            input_csv=args.csv,
            output_csv=args.output,
            predictor=predictor,
            smiles_col=args.smiles_col,
            donor_col=args.donor_col
        )
    else:
        # Single prediction
        print(f"\nInput:")
        print(f"  SMILES: {args.smiles}")
        print(f"  Donor Type: {args.donor_type}")
        
        energy, status = predictor.predict_single(args.smiles, args.donor_type)
        
        print(f"\nResult:")
        print(f"  Status: {status}")
        if energy is not None:
            print(f"  Predicted Binding Energy: {energy:.4f} eV")
        else:
            print(f"  Prediction failed")


if __name__ == "__main__":
    main()

