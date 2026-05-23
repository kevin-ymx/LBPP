"""
Downstream binding-energy training (3-feature: atomic_num + tetrahedral_chirality + coordination_number).

Same training pipeline as the main LBPP train_downstream.py (weighted MSE, val-MAE
checkpointing, CLI, eval splits). Graph node features: 3-D (atomic_num, chirality, coordination_number).
Uses comparison_2feat SSL checkpoint at config.checkpoint_dir/best_model.pt.
"""
import argparse
import ast
import csv
import json
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import random
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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


def fetch_molecule_from_pubchem(
    cid: int,
    max_retries: int = 6,
    retry_base_delay: float = 3.0,
    timeout: int = 30,
) -> Optional[Chem.Mol]:
    """
    Fetch molecule from PubChem using CID, with retries and exponential backoff on 503/timeout.
    """
    import urllib.request
    import urllib.error
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
    last_error = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                sdf_data = response.read().decode('utf-8')
            mol = Chem.MolFromMolBlock(sdf_data, removeHs=False)
            if mol is None:
                mol = Chem.MolFromMolBlock(sdf_data, removeHs=True)
                if mol is not None:
                    mol = Chem.AddHs(mol)
            return mol
        except (urllib.error.HTTPError, OSError, TimeoutError) as e:
            last_error = e
            is_retryable = (
                getattr(e, "code", None) in (503, 502, 429) or
                "timeout" in str(e).lower() or "timed out" in str(e).lower()
            )
            if attempt < max_retries - 1 and is_retryable:
                delay = retry_base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            break
        except Exception as e:
            last_error = e
            break
    if last_error is not None:
        print(f"  Warning: Failed to fetch CID {cid}: {last_error}")
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


def get_canonical_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Build mol with 3D coords from SMILES (no API call). For geometry-based mapping."""
    mol = fetch_molecule_from_smiles(smiles)
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


def get_canonical_mol_with_coords(
    cid: int,
    max_retries: int = 6,
    retry_base_delay: float = 3.0,
) -> Optional[Chem.Mol]:
    """Fetch mol from PubChem by CID. Ensure it has 3D coords (embed if missing)."""
    mol = fetch_molecule_from_pubchem(cid, max_retries=max_retries, retry_base_delay=retry_base_delay)
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

        # Node features: [atomic_num, chirality, coordination_number]
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append([
                float(atom.GetAtomicNum()),
                float(int(atom.GetChiralTag())),
                float(len(atom.GetNeighbors())),
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


def load_adsorption_data(
    csv_path: str,
    use_pubchem: bool = True,
    pubchem_request_delay: float = 0.5,
    pubchem_max_retries: int = 6,
    pubchem_retry_base_delay: float = 3.0,
    use_graph_cache: bool = True,
    graph_cache_path: Optional[str] = None,
    prefer_smiles: bool = True,
    skip_pubchem: bool = False,
    zero_binding_tags: bool = False,
) -> Tuple[List[Data], List[float], List[str]]:
    """
    Load adsorption data from merged cleaned CSV (min_ads_mult1p2_struct_cleaned_merged.csv).
    Columns: cid, adsorption_energy, pb_bond_encoding, adsorbate_structure; optional: SMILES.
    (1) If prefer_smiles and row has SMILES: build mol from SMILES (no API). Else if not skip_pubchem: fetch from PubChem (consecutive same CID reused).
    (2) If skip_pubchem=True: never call PubChem; only process rows with valid SMILES.
    (3) If use_graph_cache and cache exists: load from cache; else build and save cache.
    """
    if graph_cache_path is None or graph_cache_path == "":
        base = os.path.splitext(os.path.basename(csv_path))[0]
        graph_cache_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), base + "_graph_cache.pt")

    if use_graph_cache and os.path.isfile(graph_cache_path):
        print(f"Loading adsorption data from cache: {graph_cache_path}")
        data = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
        graphs = data["graphs"]
        energies = data["energies"]
        cid_list = data["cids"]
        if zero_binding_tags:
            print("  zero_binding_tags=True: setting binding_tag feature to 0 for all cached graphs")
            for g in graphs:
                if hasattr(g, "x") and g.x is not None and g.x.size(-1) > 7:
                    g.x[:, 7] = 0.0
        print(f"Loaded {len(graphs)} molecules from cache")
        return graphs, energies, cid_list

    required = ["cid", "adsorption_energy", "pb_bond_encoding", "adsorbate_structure"]
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for col in required:
        if col not in fieldnames:
            raise ValueError(f"CSV missing column: {col}. Required: {required}")

    has_smiles_col = "SMILES" in fieldnames or "smiles" in fieldnames
    smiles_col = "SMILES" if "SMILES" in fieldnames else ("smiles" if "smiles" in fieldnames else None)
    if skip_pubchem and not has_smiles_col:
        raise ValueError("downstream_skip_pubchem=True but CSV has no SMILES column. Add SMILES or set skip_pubchem=False.")

    print(f"Loading adsorption data from {csv_path}...")
    print(f"  Prefer SMILES (no API): {prefer_smiles}, Skip PubChem: {skip_pubchem}, CSV has SMILES: {has_smiles_col}")
    if not skip_pubchem:
        print(f"  PubChem: delay={pubchem_request_delay}s, max_retries={pubchem_max_retries}; consecutive same CID reused.")
    print(f"  Graph cache: will save to {graph_cache_path}")
    total_rows = len(rows)
    print(f"Found {total_rows} entries in CSV")

    graphs: List[Data] = []
    energies: List[float] = []
    cid_list: List[str] = []

    # Track why rows are dropped (fatal) vs. where binding info is missing (soft)
    rejection_counts: Dict[str, int] = {}
    binding_info_issues: Dict[str, int] = {}

    def inc(d: Dict[str, int], key: str) -> None:
        d[key] = d.get(key, 0) + 1

    last_cid = None
    last_canon_mol = None

    for row in tqdm(rows, desc="Processing molecules"):
        # cid and adsorption_energy are mandatory for training
        try:
            cid = int(row["cid"])
        except (ValueError, KeyError, TypeError):
            inc(rejection_counts, "invalid_cid")
            continue
        try:
            adsorption_energy = float(row["adsorption_energy"])
        except (ValueError, KeyError, TypeError):
            inc(rejection_counts, "invalid_adsorption_energy")
            continue

        # Build canonical molecule (SMILES preferred, otherwise PubChem if allowed)
        use_smiles = (prefer_smiles or skip_pubchem) and smiles_col and row.get(smiles_col, "").strip()
        canon_mol = None
        if use_smiles:
            canon_mol = get_canonical_mol_from_smiles(row[smiles_col].strip())
        elif not skip_pubchem:
            if cid == last_cid and last_canon_mol is not None:
                canon_mol = last_canon_mol
            else:
                if use_pubchem and pubchem_request_delay > 0:
                    time.sleep(pubchem_request_delay)
                canon_mol = get_canonical_mol_with_coords(
                    cid,
                    max_retries=pubchem_max_retries,
                    retry_base_delay=pubchem_retry_base_delay,
                )
                last_cid = cid
                last_canon_mol = canon_mol
        else:
            inc(rejection_counts, "no_smiles_and_skip_pubchem")
            continue

        if canon_mol is None:
            inc(rejection_counts, "canonical_mol_failed")
            continue

        # Default: no binding indices (binding_tag will be all zeros) unless mapping succeeds
        binding_indices_canonical: List[int] = []

        # Try to recover binding information from DFT structure + pb_bond_encoding
        binding_mask_dft = parse_pb_bond_encoding(row.get("pb_bond_encoding", ""))
        struct = parse_adsorbate_structure(row.get("adsorbate_structure", ""))
        if binding_mask_dft is None or struct is None:
            inc(binding_info_issues, "missing_or_invalid_binding_or_structure")
        else:
            extracted = extract_dft_atomic_numbers_and_coords(struct)
            if not extracted:
                inc(binding_info_issues, "struct_extract_failed")
            else:
                atomic_numbers, coords = extracted
                if len(binding_mask_dft) != len(atomic_numbers):
                    inc(binding_info_issues, "binding_struct_length_mismatch")
                else:
                    mapping = geometry_based_mapping(atomic_numbers, coords, canon_mol)
                    if mapping is None:
                        inc(binding_info_issues, "geometry_mapping_failed")
                    else:
                        binding_indices_dft = [i for i, v in enumerate(binding_mask_dft) if v == 1]
                        binding_indices_canonical = [mapping[i] for i in binding_indices_dft if i in mapping]
                        if not binding_indices_canonical:
                            inc(binding_info_issues, "no_binding_indices_canonical")

        # Build heavy-atom mol and graph (even if no binding indices; then binding_tag is all zeros)
        try:
            mol_heavy, binding_indices_heavy = transfer_binding_and_remove_hydrogens(
                canon_mol, binding_indices_canonical
            )
        except Exception:
            inc(rejection_counts, "transfer_binding_and_remove_hs_failed")
            continue

        if mol_heavy.GetNumAtoms() < 2:
            inc(rejection_counts, "too_few_heavy_atoms")
            continue

        try:
            graph = MolecularGraphWithBinding.mol_to_graph(mol_heavy, binding_indices_heavy)
        except Exception:
            inc(rejection_counts, "graph_build_failed")
            continue

        graphs.append(graph)
        energies.append(adsorption_energy)
        cid_list.append(str(cid))

    kept = len(graphs)
    dropped = total_rows - kept
    print(f"Successfully processed {kept} molecules (from {total_rows} rows)")
    if dropped > 0:
        print(f"  Dropped rows: {dropped}")
        for reason, cnt in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt}")
    if binding_info_issues:
        print("  Binding info issues (molecules kept but binding_tag may be all zeros):")
        for reason, cnt in sorted(binding_info_issues.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt}")

    if zero_binding_tags:
        print("zero_binding_tags=True: setting binding_tag feature to 0 for all graphs before caching/training")
        for g in graphs:
            if hasattr(g, "x") and g.x is not None and g.x.size(-1) > 7:
                g.x[:, 7] = 0.0

    if use_graph_cache and graph_cache_path:
        cache_dir = os.path.dirname(os.path.abspath(graph_cache_path))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save({"graphs": graphs, "energies": energies, "cids": cid_list}, graph_cache_path)
        print(f"Saved graph cache to {graph_cache_path} ({len(graphs)} molecules)")

    # Print node features for the first 5 samples
    node_feature_names = ['atomic_num', 'chirality', 'coordination_number']
    n_show = min(5, len(graphs))
    for s in range(n_show):
        g = graphs[s]
        cid = cid_list[s] if s < len(cid_list) else "?"
        print(f"\n--- Sample {s + 1}/{n_show} (CID {cid}), num_nodes={g.num_nodes} ---")
        print("Node features (rows=atoms, cols=" + ", ".join(node_feature_names) + "):")
        x = g.x.detach().cpu().numpy() if g.x.is_cuda else g.x.numpy()
        print(x)

    return graphs, energies, cid_list


def load_extra_smiles_csv(
    csv_path: str,
    zero_binding_tags: bool = True,
) -> Tuple[List[Data], List[float]]:
    """
    Load additional (SMILES, binding energy) pairs from a CSV and convert to graphs.
    CSV must have columns: SMILES, best_adsorption_energy.
    Graphs use 3 node features: atomic_num, chirality, coordination_number (same as MolecularGraphWithBinding.mol_to_graph).

    Returns:
        graphs: list of Data objects (heavy-atom only)
        energies: list of float adsorption energies
    """
    graphs: List[Data] = []
    energies: List[float] = []
    skipped = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loading extra training data from {csv_path} ({len(rows)} rows)...")

    for row in tqdm(rows, desc="Extra CSV"):
        smiles = (row.get("SMILES") or row.get("smiles") or "").strip()
        energy_str = (row.get("best_adsorption_energy") or "").strip()
        if not smiles or not energy_str:
            skipped += 1
            continue
        try:
            energy = float(energy_str)
        except (ValueError, TypeError):
            skipped += 1
            continue

        # Only keep strong binders (< -1.3 eV) and weak binders ([-0.6, 0] eV)
        if not (energy < -1.3 or (-0.6 <= energy <= 0)):
            skipped += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue
        mol_heavy = Chem.RemoveHs(mol)
        if mol_heavy.GetNumAtoms() < 2:
            skipped += 1
            continue

        try:
            graph = MolecularGraphWithBinding.mol_to_graph(mol_heavy, binding_atom_indices=[])
        except Exception:
            skipped += 1
            continue

        graphs.append(graph)
        energies.append(energy)

    print(f"  Loaded {len(graphs)} extra molecules (skipped {skipped})")
    return graphs, energies


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


def _build_sample_weights(
    energies: List[float],
    strong_threshold: float = -1.3,
    weak_low: float = -0.6,
    weak_high: float = 0.0,
    strong_weight: float = 4.0,
    weak_weight: float = 4.0,
) -> List[float]:
    """Assign sampling weights: strong_weight for strong binders, weak_weight for weak binders, 1.0 otherwise."""
    weights = []
    for e in energies:
        if e < strong_threshold:
            weights.append(strong_weight)
        elif weak_low <= e <= weak_high:
            weights.append(weak_weight)
        else:
            weights.append(1.0)
    return weights


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


def _compute_loss_weights(
    energies: torch.Tensor,
    strong_threshold: float = -1.3,
    weak_low: float = -0.6,
    weak_high: float = 0.0,
    rare_loss_weight: float = 3.0,
) -> torch.Tensor:
    """Per-sample loss weights: rare_loss_weight for strong/weak binders, 1.0 otherwise."""
    e = energies.squeeze(-1)  # [B]
    strong = e < strong_threshold
    weak = (e >= weak_low) & (e <= weak_high)
    weights = torch.ones_like(e)
    weights[strong | weak] = rare_loss_weight
    return weights  # [B]


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    collect_predictions: bool = False,
    max_grad_norm: float = 1.0,
    rare_loss_weight: float = 3.0,
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

        loss_weights = _compute_loss_weights(
            batch_energies, rare_loss_weight=rare_loss_weight
        ).to(device)
        per_sample_loss = (predictions.squeeze(-1) - batch_energies.squeeze(-1)) ** 2
        loss = (per_sample_loss * loss_weights).mean()
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
    checkpoint_dir: str,
    val_mae: Optional[float] = None,
):
    """Save periodic epoch checkpoint (loss = validation MSE)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_mae': val_mae,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'downstream_checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    mae_str = ", val MAE {:.4f}".format(val_mae) if val_mae is not None else ""
    print("Saved checkpoint to {} (val MSE {:.4f}{})".format(checkpoint_path, loss, mae_str))


def save_best_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    save_finetuned_encoder: bool = False,
    val_mae: Optional[float] = None,
    rare_loss_weight: Optional[float] = None,
    best_metric: Optional[str] = None,
):
    """Save best model checkpoint immediately. If save_finetuned_encoder=True, also save GIN-E encoder state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_mae': val_mae,
        'rare_loss_weight': rare_loss_weight,
        'best_metric': best_metric,
    }
    
    best_path = os.path.join(checkpoint_dir, 'downstream_best_model.pt')
    torch.save(checkpoint, best_path)
    mae_str = f", val MAE {val_mae:.4f}" if val_mae is not None else ""
    print(f"Saved best downstream model (epoch {epoch}, val MSE {loss:.4f}{mae_str}) to {best_path}")

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
    dataset_name: str = "dataset",
):
    """
    Save predictions and targets to a CSV file.

    Args:
        predictions: List of predicted binding energies.
        targets: List of target binding energies.
        output_path: Path to save the CSV file.
        dataset_name: Name of the dataset (for logging).
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_binding_energy", "predicted_binding_energy", "error"])

        for target, pred in zip(targets, predictions):
            error = pred - target
            writer.writerow([target, pred, error])

    print(f"Saved {dataset_name} predictions to {output_path}")
    print(f"  Total samples: {len(predictions)}")


def compute_rmse(targets: List[float], predictions: List[float]) -> float:
    """Root mean squared error (eV) from aligned target/prediction lists."""
    t = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def build_downstream_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load downstream CSV, split, optional extra train data, and build loaders."""
    default_csv = os.path.join(
        os.path.dirname(__file__), "dataset", "prediction", "min_ads_mult1p2_struct_cleaned_merged.csv"
    )
    csv_path = (config.downstream_csv or os.environ.get("DOWNSTREAM_CSV") or default_csv).strip()
    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError(
            "Merged CSV not found: {}. Set config.downstream_csv, DOWNSTREAM_CSV, or {}".format(
                csv_path, default_csv
            )
        )
    print("\nLoading molecules from merged CSV (CID + PubChem, geometry-based binding tags)...")
    cache_path = (config.downstream_graph_cache_path or "").strip()
    graphs, energies, cids = load_adsorption_data(
        csv_path,
        use_pubchem=not config.downstream_skip_pubchem,
        pubchem_request_delay=config.pubchem_request_delay,
        pubchem_max_retries=config.pubchem_max_retries,
        pubchem_retry_base_delay=config.pubchem_retry_base_delay,
        use_graph_cache=config.downstream_use_graph_cache,
        graph_cache_path=cache_path if cache_path else None,
        prefer_smiles=config.downstream_prefer_smiles,
        skip_pubchem=config.downstream_skip_pubchem,
        zero_binding_tags=config.downstream_zero_binding_tags,
    )
    if len(graphs) == 0:
        raise RuntimeError("No molecules loaded! Check the CSV file.")

    print("\nSplitting data (70% train, 20% val, 10% test)...")
    train_graphs, train_energies, val_graphs, val_energies, test_graphs, test_energies = split_data(
        graphs,
        energies,
        train_ratio=config.downstream_train_split,
        val_ratio=config.downstream_val_split,
        test_ratio=config.downstream_test_split,
        seed=config.seed,
    )
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    extra_csv = (config.downstream_extra_csv or "").strip()
    if extra_csv and os.path.isfile(extra_csv):
        extra_graphs, extra_energies = load_extra_smiles_csv(
            extra_csv, zero_binding_tags=config.downstream_zero_binding_tags,
        )
        train_graphs.extend(extra_graphs)
        train_energies.extend(extra_energies)
        print(f"Train after extra data: {len(train_graphs)} (added {len(extra_graphs)} from extra CSV)")
    elif extra_csv:
        print(f"Warning: downstream_extra_csv not found: {extra_csv}")

    return create_data_loaders(
        train_graphs,
        train_energies,
        val_graphs,
        val_energies,
        test_graphs,
        test_energies,
        batch_size=config.downstream_batch_size,
        num_workers=config.num_workers,
    )


def build_downstream_model(config: Config, device: torch.device) -> DownstreamModel:
    """Initialize downstream model with optional pretrained GIN-E weights."""
    print("\n" + "=" * 60)
    print("Loading pretrained GIN-E encoder...")
    gin_e_encoder = GINEEncoder(
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=config.edge_feature_dim,
        node_embedding_dim=config.node_embedding_dim,
        edge_embedding_dim=config.edge_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gin_layers,
        dropout=config.dropout,
    )
    gin_e_checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    if not os.path.exists(gin_e_checkpoint_path):
        print(f"WARNING: GIN-E checkpoint not found at {gin_e_checkpoint_path}")
        print("         Will train downstream model with randomly initialized GIN-E encoder.")
        gin_e_checkpoint_path = None
    else:
        print(f"Found GIN-E checkpoint at: {gin_e_checkpoint_path}")
    print("=" * 60 + "\n")

    print("Initializing downstream model (single prediction head for binding energy)...")
    model = DownstreamModel(
        gin_e_encoder=gin_e_encoder,
        gin_e_checkpoint_path=gin_e_checkpoint_path,
        freeze_gin_e=config.freeze_pretrained_encoder,
        mlp_hidden_dim=config.downstream_mlp_hidden_dim,
        mlp_dropout=config.downstream_mlp_dropout,
        num_tasks=1,
        task_hidden_dim=config.downstream_task_hidden_dim,
        task_dropout=config.downstream_task_dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        "Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    return model


def run_downstream_training(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    rare_loss_weight: float,
    checkpoint_dir: str,
    log_dir: str,
    best_metric: str = "mae",
    save_predictions_csv: bool = True,
    save_val_predictions_csv: bool = False,
    eval_all_splits: bool = False,
) -> Dict[str, Any]:
    """
    Train binding-energy downstream model; select best checkpoint by validation MAE or MSE.

    Returns dict with best_val_mae, best_val_loss, best_epoch, test_mae, test_rmse, test_loss.
    """
    best_metric = best_metric.lower().strip()
    if best_metric not in ("mae", "loss"):
        raise ValueError("best_metric must be 'mae' or 'loss', got {!r}".format(best_metric))

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = build_downstream_model(config, device)
    criterion = nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(
        trainable_params,
        lr=config.downstream_learning_rate,
        weight_decay=config.downstream_weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.downstream_num_epochs, eta_min=1e-6
    )

    print(
        "\nStarting downstream training (rare_loss_weight={}, best_metric={})...".format(
            rare_loss_weight, best_metric
        )
    )
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0

    for epoch in range(1, config.downstream_num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.downstream_num_epochs}")
        train_loss, train_mae, _, _ = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            collect_predictions=False,
            max_grad_norm=1.0,
            rare_loss_weight=rare_loss_weight,
        )
        val_loss, val_mae, _, _ = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            collect_predictions=False,
        )
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} eV")
        print(
            f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f} eV, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
            continue

        improved = (
            val_mae < best_val_mae
            if best_metric == "mae"
            else val_loss < best_val_loss
        )
        if improved:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            save_best_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                checkpoint_dir=checkpoint_dir,
                save_finetuned_encoder=not config.freeze_pretrained_encoder,
                val_mae=val_mae,
                rare_loss_weight=rare_loss_weight,
                best_metric=best_metric,
            )

        if epoch % 10 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                checkpoint_dir=checkpoint_dir,
                val_mae=val_mae,
            )

    print("\n" + "=" * 60)
    print("Final Evaluation with Best Model")
    print("=" * 60)
    best_model_path = os.path.join(checkpoint_dir, "downstream_best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']} ({best_metric})")
    else:
        print("WARNING: No best checkpoint saved; evaluating last epoch weights.")

    def _eval_split(loader: DataLoader, name: str):
        loss, mae, preds, tgts = evaluate_model(
            model=model,
            data_loader=loader,
            criterion=criterion,
            device=device,
            desc="Evaluating {} set".format(name),
        )
        rmse = compute_rmse(tgts, preds)
        return loss, mae, rmse, preds, tgts

    if eval_all_splits:
        print("\nCollecting predictions for train, validation, and test sets...")
        train_loss, train_mae, train_rmse, train_preds, train_tgts = _eval_split(
            train_loader, "train"
        )
        val_loss, val_mae, val_rmse, val_preds, val_tgts = _eval_split(val_loader, "val")
        test_loss, test_mae, test_rmse, test_predictions, test_targets = _eval_split(
            test_loader, "test"
        )

        def compute_r2(targets, predictions):
            targets_arr = np.array(targets)
            predictions_arr = np.array(predictions)
            ss_res = np.sum((targets_arr - predictions_arr) ** 2)
            ss_tot = np.sum((targets_arr - np.mean(targets_arr)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        print("\nFinal Results (Best Model):")
        print(
            "  Train MSE: {:.4f}, MAE: {:.4f} eV, RMSE: {:.4f} eV".format(
                train_loss, train_mae, train_rmse
            )
        )
        print(
            "  Val MSE: {:.4f}, MAE: {:.4f} eV, RMSE: {:.4f} eV".format(
                val_loss, val_mae, val_rmse
            )
        )
        print(
            "  Test MSE: {:.4f}, MAE: {:.4f} eV, RMSE: {:.4f} eV".format(
                test_loss, test_mae, test_rmse
            )
        )
        print(
            "  Train R2: {:.4f}, Val R2: {:.4f}, Test R2: {:.4f}".format(
                compute_r2(train_tgts, train_preds),
                compute_r2(val_tgts, val_preds),
                compute_r2(test_targets, test_predictions),
            )
        )
        if save_predictions_csv:
            predictions_dir = os.path.join(log_dir, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)
            save_predictions(
                train_preds, train_tgts,
                os.path.join(predictions_dir, "train_predictions.csv"), "train",
            )
            save_predictions(
                val_preds, val_tgts,
                os.path.join(predictions_dir, "val_predictions.csv"), "validation",
            )
            save_predictions(
                test_predictions, test_targets,
                os.path.join(predictions_dir, "test_predictions.csv"), "test",
            )
    else:
        val_loss = val_mae = val_rmse = float("nan")
        val_preds: List[float] = []
        val_tgts: List[float] = []
        if save_val_predictions_csv:
            val_loss, val_mae, val_rmse, val_preds, val_tgts = _eval_split(val_loader, "val")
            print(
                "\nValidation set (best by val {}): MSE {:.4f}, MAE {:.4f} eV, RMSE {:.4f} eV".format(
                    best_metric.upper(), val_loss, val_mae, val_rmse
                )
            )

        test_loss, test_mae, test_predictions, test_targets = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            desc="Evaluating test set",
        )
        test_rmse = compute_rmse(test_targets, test_predictions)
        print(f"\nTest set (best by val {best_metric.upper()}):")
        print(
            "  Test MSE: {:.4f}, Test MAE: {:.4f} eV, Test RMSE: {:.4f} eV".format(
                test_loss, test_mae, test_rmse
            )
        )
        if save_predictions_csv:
            predictions_dir = os.path.join(log_dir, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)
            if save_val_predictions_csv:
                save_predictions(
                    val_preds,
                    val_tgts,
                    os.path.join(predictions_dir, "val_predictions.csv"),
                    "validation",
                )
            save_predictions(
                test_predictions,
                test_targets,
                os.path.join(predictions_dir, "test_predictions.csv"),
                "test",
            )

    print(
        "  Best epoch: {}, Best val MAE: {:.4f} eV, Best val MSE: {:.4f}".format(
            best_epoch, best_val_mae, best_val_loss
        )
    )

    return {
        "rare_loss_weight": rare_loss_weight,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_val_loss": best_val_loss,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_loss": test_loss,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
    }


def parse_downstream_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downstream binding-energy training.")
    parser.add_argument(
        "--rare-loss-weight",
        type=float,
        default=None,
        help="MSE weight for strong (E<-1.3) and weak ([-0.6,0]) binders (default: config).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for downstream_best_model.pt (default: <config.checkpoint_dir>/downstream).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Log/predictions directory (default: config.log_dir).",
    )
    parser.add_argument(
        "--best-metric",
        choices=("mae", "loss"),
        default=None,
        help="Metric for best checkpoint: val MAE or val MSE (default: config.downstream_best_metric).",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_downstream_args()
    config = Config()
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rare_loss_weight = (
        args.rare_loss_weight
        if args.rare_loss_weight is not None
        else config.downstream_rare_loss_weight
    )
    best_metric = args.best_metric or config.downstream_best_metric
    checkpoint_dir = args.checkpoint_dir or os.path.join(config.checkpoint_dir, "downstream")
    log_dir = args.log_dir or config.log_dir

    train_loader, val_loader, test_loader = build_downstream_dataloaders(config)
    metrics = run_downstream_training(
        config,
        train_loader,
        val_loader,
        test_loader,
        device,
        rare_loss_weight=rare_loss_weight,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        best_metric=best_metric,
        save_predictions_csv=True,
        eval_all_splits=True,
    )
    print("\nDownstream training completed!")
    print(
        "Best validation {}: {:.4f}".format(
            best_metric,
            metrics["best_val_mae"] if best_metric == "mae" else metrics["best_val_loss"],
        )
    )
    print("Predictions saved to: {}".format(os.path.join(log_dir, "predictions")))


if __name__ == "__main__":
    main()
