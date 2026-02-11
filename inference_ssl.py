"""
Inference script for GIN-E encoder from self-supervised learning.

Generates molecular embeddings from SMILES strings using the pretrained GIN-E encoder.

Usage:
    # Single molecule embedding
    python inference_ssl.py --smiles "CCO"
    
    # Multiple molecules
    python inference_ssl.py --smiles "CCO" "CC(=O)C" "c1ccccc1"
    
    # Process all CSV files in a folder (writes result CSVs with extended embedding columns)
    python inference_ssl.py --filtered_csv ./filtered_csv --output_dir ./filtered_csv_embeddings
"""
import os
import sys
import argparse
import csv
import glob
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from config import Config
from models.gin_e import GINEEncoder


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
                AllChem.EmbedMolecule(mol, maxAttempts=100, randomSeed=42)
        return mol
    except Exception as e:
        print(f"Error creating molecule from SMILES '{smiles}': {e}")
        return None


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


def mol_to_graph(mol: Chem.Mol) -> Data:
    """
    Convert a molecule to a PyTorch Geometric graph.
    
    Uses the same 8 node features as the GIN-E encoder:
    - atomic_num, chirality, partial_charge, hybridization,
    - coordination_num, valence_electrons, electronegativity, binding_tag
    
    Args:
        mol: RDKit molecule object.
        
    Returns:
        PyTorch Geometric Data object with node and edge features.
    """
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
        binding_tag = 0.0  # Zero for general SSL inference
        
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


class GINEEncoderInference:
    """
    Inference class for GIN-E encoder.
    Generates molecular embeddings from SMILES strings.
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        config: Config = None
    ):
        """
        Initialize the GIN-E encoder for inference.
        
        Args:
            checkpoint_path: Path to the GIN-E encoder checkpoint.
            device: Device to run inference on ('cuda' or 'cpu').
            config: Config object (if None, uses default Config).
        """
        self.config = config if config is not None else Config()
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Store embedding dimension
        self.embedding_dim = self.config.hidden_dim
    
    def _load_model(self, checkpoint_path: str) -> GINEEncoder:
        """Load the GIN-E encoder from checkpoint."""
        # Create GIN-E encoder
        model = GINEEncoder(
            node_feature_dim=self.config.node_feature_dim,
            edge_feature_dim=self.config.edge_feature_dim,
            node_embedding_dim=self.config.node_embedding_dim,
            edge_embedding_dim=self.config.edge_embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_gin_layers,
            dropout=self.config.dropout
        )
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"GIN-E checkpoint not found: {checkpoint_path}\n"
                f"Please train the encoder first using train_ssl.py"
            )
        
        print(f"Loading GIN-E encoder from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            print(f"  Loaded from epoch {epoch}, loss: {loss:.4f}" if isinstance(loss, float) else f"  Loaded from epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("  Loaded model weights")
        
        model = model.to(self.device)
        return model
    
    def encode_single(self, smiles: str) -> Tuple[Optional[np.ndarray], str]:
        """
        Generate embedding for a single molecule.
        
        Args:
            smiles: SMILES string of the molecule.
            
        Returns:
            Tuple of (embedding, status_message).
            embedding is None if encoding failed.
        """
        # Convert SMILES to molecule
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None, f"Failed to parse SMILES: {smiles}"
        
        # Convert to graph
        try:
            graph = mol_to_graph(mol)
        except Exception as e:
            return None, f"Failed to convert molecule to graph: {e}"
        
        # Run inference
        graph = graph.to(self.device)
        
        with torch.no_grad():
            embedding = self.model(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=None  # Single molecule
            )
        
        # Convert to numpy
        embedding = embedding.squeeze().cpu().numpy()
        
        return embedding, "Success"
    
    def encode_batch(
        self, 
        smiles_list: List[str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Generate embeddings for a batch of molecules.
        
        Args:
            smiles_list: List of SMILES strings.
            batch_size: Batch size for inference.
            show_progress: Whether to show progress bar.
            
        Returns:
            Tuple of (embeddings, status_list, valid_indices).
            - embeddings: numpy array of shape [num_valid, embedding_dim]
            - status_list: list of status messages for each input
            - valid_indices: indices of successfully encoded molecules
        """
        all_embeddings = []
        status_list = []
        valid_indices = []
        valid_graphs = []
        
        # Process all molecules
        iterator = tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Processing") if show_progress else enumerate(smiles_list)
        
        for i, smiles in iterator:
            mol = smiles_to_mol(smiles)
            if mol is None:
                status_list.append(f"Failed to parse SMILES: {smiles}")
                continue
            
            try:
                graph = mol_to_graph(mol)
                valid_graphs.append(graph)
                valid_indices.append(i)
                status_list.append("Success")
            except Exception as e:
                status_list.append(f"Failed to convert molecule to graph: {e}")
        
        # Batch inference
        if len(valid_graphs) > 0:
            num_batches = (len(valid_graphs) + batch_size - 1) // batch_size
            
            batch_iterator = range(num_batches)
            if show_progress:
                batch_iterator = tqdm(batch_iterator, desc="Encoding")
            
            for batch_idx in batch_iterator:
                start = batch_idx * batch_size
                end = min(start + batch_size, len(valid_graphs))
                batch_graphs = valid_graphs[start:end]
                
                batched_graph = Batch.from_data_list(batch_graphs).to(self.device)
                
                with torch.no_grad():
                    embeddings = self.model(
                        x=batched_graph.x,
                        edge_index=batched_graph.edge_index,
                        edge_attr=batched_graph.edge_attr,
                        batch=batched_graph.batch
                    )
                
                all_embeddings.append(embeddings.cpu().numpy())
            
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = np.zeros((0, self.embedding_dim))
        
        return all_embeddings, status_list, valid_indices
    
    def encode_smiles(self, smiles_list: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Simple interface to encode SMILES to embeddings.
        
        Args:
            smiles_list: List of SMILES strings.
            batch_size: Batch size for inference.
            
        Returns:
            Embeddings array of shape [num_valid, embedding_dim].
            Invalid molecules are skipped.
        """
        embeddings, _, _ = self.encode_batch(smiles_list, batch_size, show_progress=False)
        return embeddings


def save_extended_csv(
    output_path: str,
    rows: List[Dict[str, Any]],
    embeddings: np.ndarray,
    status_list: List[str],
    valid_indices: List[int],
    embedding_dim: int,
    smiles_col: str = "SMILES",
):
    """
    Save CSV with all original columns plus status and embedding columns (emb_0, ..., emb_{d-1}).
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys()) + ["status"] + [f"emb_{i}" for i in range(embedding_dim)]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        valid_idx = 0
        for i, row in enumerate(rows):
            out = dict(row)
            out["status"] = status_list[i] if i < len(status_list) else "Unknown"
            if i in valid_indices and valid_idx < len(embeddings):
                emb = embeddings[valid_idx]
                for j in range(embedding_dim):
                    out[f"emb_{j}"] = emb[j]
                valid_idx += 1
            else:
                for j in range(embedding_dim):
                    out[f"emb_{j}"] = ""
            writer.writerow(out)
    print(f"  Saved extended CSV to {output_path}")


def process_csv_with_embeddings(
    csv_path: str,
    output_path: str,
    encoder: GINEEncoderInference,
    batch_size: int,
    smiles_col: str,
) -> Tuple[int, int]:
    """
    Read a CSV, run GIN-E encoding, write result CSV with original columns + embedding columns.
    Returns (num_success, num_failed).
    """
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"  No rows in {csv_path}, skipping.")
        return 0, 0
    smiles_list = [row.get(smiles_col, "").strip() for row in rows]
    embeddings, status_list, valid_indices = encoder.encode_batch(
        smiles_list, batch_size=batch_size, show_progress=True
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_extended_csv(
        output_path,
        rows,
        embeddings,
        status_list,
        valid_indices,
        encoder.embedding_dim,
        smiles_col=smiles_col,
    )
    return len(valid_indices), len(smiles_list) - len(valid_indices)


def main():
    parser = argparse.ArgumentParser(
        description='Generate molecular embeddings using pretrained GIN-E encoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single molecule embedding
  python inference_ssl.py --smiles "CCO"
  
  # Multiple molecules
  python inference_ssl.py --smiles "CCO" "CC(=O)C" "c1ccccc1"
  
  # Process all CSV files in a folder
  python inference_ssl.py --filtered_csv ./filtered_csv --output_dir ./filtered_csv_embeddings
        """
    )
    
    # Input options
    parser.add_argument('--smiles', type=str, nargs='+', help='SMILES string(s) of molecule(s)')
    parser.add_argument('--filtered_csv', type=str, default=None,
                        help='Path to folder containing CSV files; each is processed and a result CSV with embedding columns is written')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for result CSVs when using --filtered_csv (default: filtered_csv_embeddings)')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of this process (0 to world_size-1) for splitting CSV files across workers (default: 0)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of workers for splitting CSV files (default: 1)')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to GIN-E encoder checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu, default: auto-detect)')
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('--smiles_col', type=str, default='SMILES',
                        help='Column name for SMILES in CSV (default: SMILES)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.filtered_csv is None and args.smiles is None:
        parser.error("One of --filtered_csv or --smiles is required")
    
    # Initialize encoder
    print("="*60)
    print("GIN-E Encoder Inference")
    print("="*60)
    
    encoder = GINEEncoderInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    print(f"Embedding dimension: {encoder.embedding_dim}")
    
    if args.filtered_csv:
        # Process all CSV files in the folder
        folder = os.path.abspath(args.filtered_csv)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(folder), "filtered_csv_embeddings")
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if args.world_size > 1:
            csv_files = [f for i, f in enumerate(csv_files) if i % args.world_size == args.rank]
            print(f"Rank {args.rank}/{args.world_size}: processing {len(csv_files)} CSV file(s)")
        if not csv_files:
            print(f"No CSV files found in {folder}" + (f" for rank {args.rank}" if args.world_size > 1 else ""))
        else:
            print(f"\nProcessing {len(csv_files)} CSV file(s) from {folder}")
            print(f"Output directory: {output_dir}\n")
            total_ok, total_fail, skipped = 0, 0, 0
            pbar = tqdm(csv_files, desc="CSV files", unit="file")
            for csv_path in pbar:
                basename = os.path.basename(csv_path)
                out_path = os.path.join(output_dir, basename)
                if os.path.exists(out_path):
                    pbar.set_postfix_str(f"[SKIP] {basename}")
                    skipped += 1
                    continue
                pbar.set_postfix_str(basename)
                n_ok, n_fail = process_csv_with_embeddings(
                    csv_path,
                    out_path,
                    encoder,
                    args.batch_size,
                    args.smiles_col,
                )
                total_ok += n_ok
                total_fail += n_fail
            processed = len(csv_files) - skipped
            print(f"\nSummary: {total_ok} encoded, {total_fail} failed across {processed} file(s)" + (f", {skipped} skipped (already exist)" if skipped else ""))
    
    else:
        # Single/multiple SMILES from command line
        smiles_list = args.smiles
        print(f"\nEncoding {len(smiles_list)} molecule(s)...")
        
        for smiles in smiles_list:
            embedding, status = encoder.encode_single(smiles)
            
            print(f"\n  SMILES: {smiles}")
            print(f"  Status: {status}")
            if embedding is not None:
                print(f"  Embedding shape: {embedding.shape}")
                print(f"  Embedding (first 5 dims): {embedding[:5]}")
                print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")


if __name__ == "__main__":
    main()

