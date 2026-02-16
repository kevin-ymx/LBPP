"""
Training script for downstream molecular property prediction.
Uses a pretrained GIN-E encoder plus a single MLP head for binding energy prediction.
Loads molecules from merged CSV (min_ads_mult1p2_struct_cleaned_merged.csv): CIDs, DFT
adsorbate_structure, pb_bond_encoding. Binding sites are identified by geometry-based
mapping (DFT -> canonical mol by element + Hungarian assignment), then heavy-atom graphs
are built with binding tags.
"""
import ast
import csv
import json
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import random
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any

from scipy.optimize import linear_sum_assignment

from rdkit import Chem
from rdkit.Chem import AllChem

from config import Config
from models.gin_e import GINEEncoder
from models.downstream_model import DownstreamModel


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fetch_molecule_from_pubchem(cid: int) -> Optional[Chem.Mol]:
    """
    Fetch molecule from PubChem using CID.
    
    Args:
        cid: PubChem Compound ID.
        
    Returns:
        RDKit molecule object or None if failed.
    """
    try:
        import urllib.request
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
        
        with urllib.request.urlopen(url, timeout=30) as response:
            sdf_data = response.read().decode('utf-8')
        
        # Parse SDF data
        mol = Chem.MolFromMolBlock(sdf_data, removeHs=False)
        if mol is None:
            # Try adding hydrogens
            mol = Chem.MolFromMolBlock(sdf_data, removeHs=True)
            if mol is not None:
                mol = Chem.AddHs(mol)
        
        return mol
    except Exception as e:
        print(f"  Warning: Failed to fetch CID {cid}: {e}")
        return None


def fetch_molecule_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Create molecule from SMILES string (fallback if PubChem fetch fails).
    
    Args:
        smiles: SMILES string.
        
    Returns:
        RDKit molecule object or None if failed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol
    except Exception as e:
        print(f"  Warning: Failed to create mol from SMILES {smiles}: {e}")
        return None


# --------------- Merged CSV parsing and geometry-based binding mapping ---------------

def parse_pb_bond_encoding(s: str) -> Optional[List[int]]:
    """Parse pb_bond_encoding string to list of 0/1. Same index as adsorbate_structure atoms."""
    if not s or not str(s).strip():
        return None
    try:
        out = ast.literal_eval(str(s).strip())
        if not isinstance(out, list) or not all(x in (0, 1) for x in out):
            return None
        return out
    except (ValueError, SyntaxError):
        return None


def parse_adsorbate_structure(s: str) -> Optional[Dict[str, Any]]:
    """Parse adsorbate_structure JSON. Keys: coords['3d'], elements['number']."""
    if not s or not str(s).strip():
        return None
    try:
        return json.loads(str(s).strip())
    except json.JSONDecodeError:
        return None


def extract_dft_atomic_numbers_and_coords(struct: Optional[Dict]) -> Optional[Tuple[List[int], np.ndarray]]:
    """
    Extract atomic_numbers and coords from adsorbate_structure.
    coords['3d'] is flat [x0,y0,z0, x1,y1,z1, ...], elements['number'] is list of atomic numbers.
    Returns (atomic_numbers, coords) with coords shape (n_atoms, 3).
    """
    if not struct or not isinstance(struct, dict):
        return None
    coords = struct.get("coords") or {}
    elements = struct.get("elements") or {}
    numbers = elements.get("number")
    flat = coords.get("3d") if isinstance(coords, dict) else None
    if not numbers or not flat or len(flat) != 3 * len(numbers):
        return None
    atomic_numbers = [int(numbers[i]) for i in range(len(numbers))]
    coords_arr = np.array(
        [[float(flat[3 * i]), float(flat[3 * i + 1]), float(flat[3 * i + 2])] for i in range(len(numbers))],
        dtype=np.float64,
    )
    return (atomic_numbers, coords_arr)


def build_rdkit_mol_dft(atomic_numbers: List[int], coords: np.ndarray) -> Optional[Chem.Mol]:
    """
    Build RDKit Mol for DFT structure from atomic numbers and 3D coords (reference step 1).
    RWMol + AddAtom per Z, then AddConformer with positions.
    """
    mol_dft = Chem.RWMol()
    for Z in atomic_numbers:
        atom = Chem.Atom(int(Z))
        mol_dft.AddAtom(atom)
    mol_dft = mol_dft.GetMol()
    conf = Chem.Conformer(len(atomic_numbers))
    for i in range(len(atomic_numbers)):
        x, y, z = coords[i, 0], coords[i, 1], coords[i, 2]
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol_dft.AddConformer(conf)
    return mol_dft


def center_coords_to_com(coords: np.ndarray) -> np.ndarray:
    """Translate coords (n, 3) so center-of-mass is at origin."""
    com = coords.mean(axis=0)
    return coords - com


def get_canonical_mol_with_coords(cid: int) -> Optional[Chem.Mol]:
    """Fetch mol from PubChem by CID. Ensure it has 3D coords (embed if missing)."""
    mol = fetch_molecule_from_pubchem(cid)
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except Exception:
            pass
    if mol.GetNumConformers() == 0:
        return None
    return mol


def geometry_based_mapping(
    atomic_numbers: List[int],
    coords: np.ndarray,
    canon_mol: Chem.Mol,
) -> Optional[Dict[int, int]]:
    """
    Geometry-based mapping: DFT index -> canonical mol index (reference step 2).
    Center both to COM, then for each element Z use Hungarian assignment on distance matrix.
    """
    # Center DFT coords to COM
    dft_coords = center_coords_to_com(np.asarray(coords, dtype=np.float64))

    # Canonical mol coords (n_atoms, 3), then center to COM
    conf = canon_mol.GetConformer()
    can_coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(canon_mol.GetNumAtoms())],
        dtype=np.float64,
    )
    can_coords = center_coords_to_com(can_coords)

    mapping: Dict[int, int] = {}
    for Z in set(atomic_numbers):
        dft_indices = [i for i, z in enumerate(atomic_numbers) if z == Z]
        can_indices = [i for i, a in enumerate(canon_mol.GetAtoms()) if a.GetAtomicNum() == Z]

        if len(dft_indices) != len(can_indices):
            return None

        dft_pts = dft_coords[dft_indices]
        can_pts = can_coords[can_indices]

        dist_matrix = np.linalg.norm(dft_pts[:, None, :] - can_pts[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        for r, c in zip(row_ind, col_ind):
            mapping[dft_indices[r]] = can_indices[c]
    return mapping


def transfer_binding_and_remove_hydrogens(
    canon_mol: Chem.Mol,
    binding_indices_canonical: List[int],
) -> Tuple[Chem.Mol, List[int]]:
    """
    Transfer multiple binding sites to heavy-atom indices and remove hydrogens (reference steps 3 & 4).
    heavy_map: canonical atom idx -> heavy index; binding_indices_heavy for graph.
    """
    heavy_map: Dict[int, int] = {}
    heavy_counter = 0
    for atom in canon_mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            heavy_map[atom.GetIdx()] = heavy_counter
            heavy_counter += 1

    binding_indices_heavy = [heavy_map[i] for i in binding_indices_canonical if i in heavy_map]
    mol_heavy = Chem.RemoveHs(canon_mol)
    return mol_heavy, binding_indices_heavy


class MolecularGraphWithBinding:
    """
    Helper class for constructing molecular graphs with binding tags.
    Uses the same 8 node features as the GIN-E encoder.
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
    
    @staticmethod
    def get_partial_charges(mol: Chem.Mol) -> List[float]:
        """Extract or compute partial charges."""
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
            # Replace NaN values with 0.0
            charges = [0.0 if (c != c) else c for c in charges]
            return charges
        except:
            return [0.0] * mol.GetNumAtoms()
    
    @staticmethod
    def get_electronegativity(atomic_num: int) -> float:
        """Get electronegativity for an atom."""
        return MolecularGraphWithBinding.ELECTRONEGATIVITY.get(atomic_num, 2.0)
    
    @staticmethod
    def get_coordination_number(atom: Chem.Atom) -> int:
        """Calculate coordination number (number of bonded atoms)."""
        return len(atom.GetNeighbors())
    
    @staticmethod
    def get_valence_electrons(atomic_num: int) -> int:
        """Get number of valence electrons."""
        valence_map = {
            1: 1, 3: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
            11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7,
            19: 1, 20: 2, 34: 6, 35: 7, 53: 7,
        }
        return valence_map.get(atomic_num, 4)
    
    @classmethod
    def mol_to_graph(cls, mol: Chem.Mol, binding_atom_indices: List[int] = None) -> Data:
        """
        Convert a molecule to a PyTorch Geometric graph with binding tags.
        
        Args:
            mol: RDKit molecule object.
            binding_atom_indices: List of atom indices to mark as binding (binding_tag=1).
            
        Returns:
            Data object with node and edge features.
        """
        if binding_atom_indices is None:
            binding_atom_indices = []
        
        # Get partial charges
        partial_charges = cls.get_partial_charges(mol)
        
        # Node features: [atomic_num, chirality, partial_charge, hybridization, 
        #                 coordination_num, valence_electrons, electronegativity, binding_tag]
        node_features = []
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            chirality = int(atom.GetChiralTag())
            partial_charge = partial_charges[atom_idx]
            hybridization = int(atom.GetHybridization())
            coordination_num = cls.get_coordination_number(atom)
            valence_electrons = cls.get_valence_electrons(atomic_num)
            electronegativity = cls.get_electronegativity(atomic_num)
            
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
        
        # Edge features: [bond_type, bond_direction]
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            bond_type = int(bond.GetBondType())
            bond_direction = int(bond.GetBondDir())
            
            edge_feat = [float(bond_type), float(bond_direction)]
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


def load_adsorption_data(csv_path: str, use_pubchem: bool = True) -> Tuple[List[Data], List[float], List[str]]:
    """
    Load adsorption data from merged cleaned CSV (min_ads_mult1p2_struct_cleaned_merged.csv).
    Columns: cid, functional_group, formula, pb_bond_encoding, adsorption_energy, config_name, adsorbate_structure.
    Builds graphs from CIDs with binding tags determined by geometry-based mapping (DFT -> canonical -> heavy-atom).
    """
    print(f"Loading adsorption data from {csv_path}...")
    required = ["cid", "adsorption_energy", "pb_bond_encoding", "adsorbate_structure"]
    graphs = []
    energies = []
    cid_list = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for col in required:
        if col not in fieldnames:
            raise ValueError(f"CSV missing column: {col}. Required: {required}")

    print(f"Found {len(rows)} entries in CSV")

    for row in tqdm(rows, desc="Processing molecules"):
        try:
            cid = int(row["cid"])
        except (ValueError, KeyError):
            continue
        adsorption_energy = float(row["adsorption_energy"])

        binding_mask_dft = parse_pb_bond_encoding(row.get("pb_bond_encoding", ""))
        struct = parse_adsorbate_structure(row.get("adsorbate_structure", ""))
        extracted = extract_dft_atomic_numbers_and_coords(struct)
        if not extracted:
            continue
        atomic_numbers, coords = extracted
        if not binding_mask_dft or len(binding_mask_dft) != len(atomic_numbers):
            continue

        canon_mol = get_canonical_mol_with_coords(cid)
        if canon_mol is None:
            continue

        mapping = geometry_based_mapping(atomic_numbers, coords, canon_mol)
        if mapping is None:
            continue

        # Transfer multiple binding sites (reference step 3)
        binding_indices_dft = [i for i, v in enumerate(binding_mask_dft) if v == 1]
        binding_indices_canonical = [mapping[i] for i in binding_indices_dft]
        if not binding_indices_canonical:
            continue

        try:
            mol_heavy, binding_indices_heavy = transfer_binding_and_remove_hydrogens(
                canon_mol, binding_indices_canonical
            )
        except Exception:
            continue

        if mol_heavy.GetNumAtoms() < 2:
            continue

        try:
            graph = MolecularGraphWithBinding.mol_to_graph(mol_heavy, binding_indices_heavy)
            graphs.append(graph)
            energies.append(adsorption_energy)
            cid_list.append(str(cid))
        except Exception:
            continue

    print(f"Successfully processed {len(graphs)} molecules")
    return graphs, energies, cid_list


class BindingEnergyDataset(Dataset):
    """Dataset for binding energy prediction."""
    
    def __init__(self, graphs: List[Data], energies: List[float]):
        self.graphs = graphs
        self.energies = torch.tensor(energies, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        return self.graphs[idx], self.energies[idx]


def collate_binding_batch(batch: List[Tuple[Data, torch.Tensor]]) -> Tuple[Batch, torch.Tensor]:
    """Collate function for binding energy batches."""
    graphs = [item[0] for item in batch]
    energies = [item[1] for item in batch]
    
    batched_graph = Batch.from_data_list(graphs)
    batched_energies = torch.stack(energies, dim=0).unsqueeze(1)  # [batch_size, 1]
    
    return batched_graph, batched_energies


def split_data(
    graphs: List[Data], 
    energies: List[float],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Data], List[float], List[Data], List[float], List[Data], List[float]]:
    """
    Split data into train, validation, and test sets.
    
    Returns:
        Tuple of (train_graphs, train_energies, val_graphs, val_energies, test_graphs, test_energies).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    indices = list(range(len(graphs)))
    random.shuffle(indices)
    
    n_train = int(len(graphs) * train_ratio)
    n_val = int(len(graphs) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_graphs = [graphs[i] for i in train_indices]
    train_energies = [energies[i] for i in train_indices]
    
    val_graphs = [graphs[i] for i in val_indices]
    val_energies = [energies[i] for i in val_indices]
    
    test_graphs = [graphs[i] for i in test_indices]
    test_energies = [energies[i] for i in test_indices]
    
    return train_graphs, train_energies, val_graphs, val_energies, test_graphs, test_energies


def create_data_loaders(
    train_graphs: List[Data],
    train_energies: List[float],
    val_graphs: List[Data],
    val_energies: List[float],
    test_graphs: List[Data] = None,
    test_energies: List[float] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create data loaders for train, validation, and optionally test sets."""
    
    train_dataset = BindingEnergyDataset(train_graphs, train_energies)
    val_dataset = BindingEnergyDataset(val_graphs, val_energies)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_binding_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_binding_batch,
        pin_memory=True
    )
    
    test_loader = None
    if test_graphs is not None and test_energies is not None:
        test_dataset = BindingEnergyDataset(test_graphs, test_energies)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_binding_batch,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    collect_predictions: bool = False,
    max_grad_norm: float = 1.0
) -> Tuple[float, float, Optional[List[float]], Optional[List[float]]]:
    """
    Train for one epoch.
    
    Args:
        collect_predictions: If True, collect and return predictions and targets.
        max_grad_norm: Maximum gradient norm for gradient clipping.
    
    Returns:
        Tuple of (avg_loss, avg_mae, predictions, targets).
        predictions and targets are None if collect_predictions=False.
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    all_predictions = [] if collect_predictions else None
    all_targets = [] if collect_predictions else None
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_graph, batch_energies in pbar:
        batch_graph = batch_graph.to(device)
        batch_energies = batch_energies.to(device)
        
        predictions = model(
            x=batch_graph.x,
            edge_index=batch_graph.edge_index,
            edge_attr=batch_graph.edge_attr,
            batch=batch_graph.batch
        )
        
        loss = criterion(predictions, batch_energies)
        mae = torch.mean(torch.abs(predictions - batch_energies)).item()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_mae += mae
        num_batches += 1
        
        if collect_predictions:
            all_predictions.extend(predictions.detach().cpu().numpy().flatten().tolist())
            all_targets.extend(batch_energies.cpu().numpy().flatten().tolist())
        
        pbar.set_postfix({'loss': loss.item(), 'mae': mae})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_mae, all_predictions, all_targets


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_predictions: bool = False
) -> Tuple[float, float, Optional[List[float]], Optional[List[float]]]:
    """
    Validate the model.
    
    Args:
        collect_predictions: If True, collect and return predictions and targets.
    
    Returns:
        Tuple of (avg_loss, avg_mae, predictions, targets).
        predictions and targets are None if collect_predictions=False.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    all_predictions = [] if collect_predictions else None
    all_targets = [] if collect_predictions else None
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_graph, batch_energies in pbar:
            batch_graph = batch_graph.to(device)
            batch_energies = batch_energies.to(device)
            
            predictions = model(
                x=batch_graph.x,
                edge_index=batch_graph.edge_index,
                edge_attr=batch_graph.edge_attr,
                batch=batch_graph.batch
            )
            
            loss = criterion(predictions, batch_energies)
            mae = torch.mean(torch.abs(predictions - batch_energies)).item()
            
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            
            if collect_predictions:
                all_predictions.extend(predictions.cpu().numpy().flatten().tolist())
                all_targets.extend(batch_energies.cpu().numpy().flatten().tolist())
            
            pbar.set_postfix({'loss': loss.item(), 'mae': mae})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_mae, all_predictions, all_targets


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating"
) -> Tuple[float, float, List[float], List[float]]:
    """
    Evaluate the model on a dataset and return predictions.
    
    Args:
        model: Model to evaluate.
        data_loader: DataLoader for the dataset.
        criterion: Loss function.
        device: Device to run on.
        desc: Description for progress bar.
        
    Returns:
        Tuple of (avg_loss, avg_mae, predictions, targets).
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for batch_graph, batch_energies in pbar:
            batch_graph = batch_graph.to(device)
            batch_energies = batch_energies.to(device)
            
            predictions = model(
                x=batch_graph.x,
                edge_index=batch_graph.edge_index,
                edge_attr=batch_graph.edge_attr,
                batch=batch_graph.batch
            )
            
            loss = criterion(predictions, batch_energies)
            mae = torch.mean(torch.abs(predictions - batch_energies)).item()
            
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            
            all_predictions.extend(predictions.cpu().numpy().flatten().tolist())
            all_targets.extend(batch_energies.cpu().numpy().flatten().tolist())
            
            pbar.set_postfix({'loss': loss.item(), 'mae': mae})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_mae, all_predictions, all_targets


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[float], List[float]]:
    """Test the model and return predictions."""
    return evaluate_model(model, test_loader, criterion, device, desc="Testing")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str
):
    """Save periodic epoch checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'downstream_checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def save_best_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    save_finetuned_encoder: bool = False,
):
    """Save best model checkpoint immediately. If save_finetuned_encoder=True, also save GIN-E encoder state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    best_path = os.path.join(checkpoint_dir, 'downstream_best_model.pt')
    torch.save(checkpoint, best_path)
    print(f"Saved best downstream model (epoch {epoch}, loss {loss:.4f}) to {best_path}")

    if save_finetuned_encoder and hasattr(model, 'gin_e_encoder'):
        encoder_path = os.path.join(checkpoint_dir, 'gin_e_finetuned.pt')
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': model.gin_e_encoder.state_dict(),
        }, encoder_path)
        print(f"Saved finetuned GIN-E encoder to {encoder_path}")


def save_predictions(
    predictions: List[float],
    targets: List[float],
    output_path: str,
    dataset_name: str = "dataset"
):
    """
    Save predictions and targets to a CSV file.
    
    Args:
        predictions: List of predicted binding energies.
        targets: List of target binding energies.
        output_path: Path to save the CSV file.
        dataset_name: Name of the dataset (for logging).
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_binding_energy', 'predicted_binding_energy', 'error'])
        
        for target, pred in zip(targets, predictions):
            error = pred - target
            writer.writerow([target, pred, error])
    
    print(f"Saved {dataset_name} predictions to {output_path}")
    print(f"  Total samples: {len(predictions)}")


def main():
    """Main training function."""
    # Load configuration
    config = Config()
    
    # Set random seed
    set_seed(config.seed)
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    downstream_checkpoint_dir = os.path.join(config.checkpoint_dir, "downstream")
    os.makedirs(downstream_checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Load adsorption data from merged cleaned CSV (binding sites from DFT geometry mapping)
    default_csv = os.path.join(os.path.dirname(__file__), "dataset", "prediction", "min_ads_mult1p2_struct_cleaned_merged.csv")
    csv_path = (config.downstream_csv or os.environ.get("DOWNSTREAM_CSV") or default_csv).strip()
    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Merged CSV not found: {csv_path}. Set config.downstream_csv in config.py, or DOWNSTREAM_CSV, or place file at {default_csv}")
    print("\nLoading molecules from merged CSV (CID + PubChem, geometry-based binding tags)...")
    graphs, energies, cids = load_adsorption_data(csv_path, use_pubchem=True)
    
    if len(graphs) == 0:
        raise RuntimeError("No molecules loaded! Check the CSV file.")
    
    # Split data: 0.7 train, 0.2 val, 0.1 test
    print("\nSplitting data (70% train, 20% val, 10% test)...")
    train_graphs, train_energies, val_graphs, val_energies, test_graphs, test_energies = split_data(
        graphs, energies,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=config.seed
    )
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_graphs, train_energies,
        val_graphs, val_energies,
        test_graphs, test_energies,
        batch_size=config.downstream_batch_size,
        num_workers=config.num_workers
    )
    
    # Load pretrained encoder
    print("\n" + "="*60)
    print("Loading pretrained GIN-E encoder...")
    
    gin_e_encoder = GINEEncoder(
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=config.edge_feature_dim,
        node_embedding_dim=config.node_embedding_dim,
        edge_embedding_dim=config.edge_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gin_layers,
        dropout=config.dropout
    )
    
    # Check for checkpoint
    gin_e_checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    if not os.path.exists(gin_e_checkpoint_path):
        print(f"WARNING: GIN-E checkpoint not found at {gin_e_checkpoint_path}")
        print(f"         Will train downstream model with randomly initialized GIN-E encoder.")
        gin_e_checkpoint_path = None
    else:
        print(f"Found GIN-E checkpoint at: {gin_e_checkpoint_path}")
    print("="*60 + "\n")
    
    # Create downstream model with only 1 prediction task (binding energy)
    print("Initializing downstream model (single prediction head for binding energy)...")
    model = DownstreamModel(
        gin_e_encoder=gin_e_encoder,
        gin_e_checkpoint_path=gin_e_checkpoint_path,
        freeze_gin_e=config.freeze_pretrained_encoder,
        mlp_hidden_dim=config.downstream_mlp_hidden_dim,
        mlp_dropout=config.downstream_mlp_dropout,
        num_tasks=1,  # Single task: binding energy prediction
        task_hidden_dim=config.downstream_task_hidden_dim,
        task_dropout=config.downstream_task_dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(
        trainable_params,
        lr=config.downstream_learning_rate,
        weight_decay=config.downstream_weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.downstream_num_epochs,
        eta_min=1e-6
    )
    
    # Training loop
    print("\nStarting downstream training for binding energy prediction...")
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    
    for epoch in range(1, config.downstream_num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.downstream_num_epochs}")
        
        # Train
        train_loss, train_mae, _, _ = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            collect_predictions=False,
            max_grad_norm=1.0
        )
        
        # Validate
        val_loss, val_mae, _, _ = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            collect_predictions=False
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} eV")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f} eV, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model immediately
        if not torch.isnan(torch.tensor(val_loss)) and not torch.isinf(torch.tensor(val_loss)):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_mae
                save_best_model(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    checkpoint_dir=downstream_checkpoint_dir,
                    save_finetuned_encoder=not config.freeze_pretrained_encoder,
                )
        
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                checkpoint_dir=downstream_checkpoint_dir
            )
    
    # Final evaluation with best model
    print("\n" + "="*60)
    print("Final Evaluation with Best Model")
    print("="*60)
    
    # Load best model for evaluation
    best_model_path = os.path.join(downstream_checkpoint_dir, 'downstream_best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Collect predictions for all three sets
    print("\nCollecting predictions for train, validation, and test sets...")
    
    # Train set predictions
    print("  Evaluating on training set...")
    train_loss, train_mae, train_predictions, train_targets = evaluate_model(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        device=device,
        desc="Evaluating train set"
    )
    
    # Validation set predictions
    print("  Evaluating on validation set...")
    val_loss, val_mae, val_predictions, val_targets = evaluate_model(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=device,
        desc="Evaluating val set"
    )
    
    # Test set predictions
    print("  Evaluating on test set...")
    test_loss, test_mae, test_predictions, test_targets = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        desc="Evaluating test set"
    )
    
    # Print results
    print(f"\nFinal Results (Best Model):")
    print(f"  Train Loss (MSE): {train_loss:.4f}, Train MAE: {train_mae:.4f} eV")
    print(f"  Val Loss (MSE): {val_loss:.4f}, Val MAE: {val_mae:.4f} eV")
    print(f"  Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f} eV")
    
    # Compute R^2 scores
    def compute_r2(targets, predictions):
        targets_arr = np.array(targets)
        predictions_arr = np.array(predictions)
        ss_res = np.sum((targets_arr - predictions_arr) ** 2)
        ss_tot = np.sum((targets_arr - np.mean(targets_arr)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    train_r2 = compute_r2(train_targets, train_predictions)
    val_r2 = compute_r2(val_targets, val_predictions)
    test_r2 = compute_r2(test_targets, test_predictions)
    
    print(f"  Train R² Score: {train_r2:.4f}")
    print(f"  Val R² Score: {val_r2:.4f}")
    print(f"  Test R² Score: {test_r2:.4f}")
    
    # Save predictions to CSV files
    print("\nSaving predictions and targets...")
    predictions_dir = os.path.join(config.log_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    save_predictions(
        train_predictions, train_targets,
        os.path.join(predictions_dir, "train_predictions.csv"),
        "train"
    )
    
    save_predictions(
        val_predictions, val_targets,
        os.path.join(predictions_dir, "val_predictions.csv"),
        "validation"
    )
    
    save_predictions(
        test_predictions, test_targets,
        os.path.join(predictions_dir, "test_predictions.csv"),
        "test"
    )
    
    print("\nDownstream training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Predictions saved to: {predictions_dir}")


if __name__ == "__main__":
    main()
