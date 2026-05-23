"""
Load downstream graph cache, split train/val/test (same as downstream training),
then run t-SNE on val+test set molecules using the finetuned GIN-E encoder.
Plot format follows visualize_tsne.py (static + interactive).
Extract molecule images for val+test samples with binding atom marked by a small box.

Usage:
    python visualize_tsne_downstream.py
    python visualize_tsne_downstream.py --cache_path /path/to/cache.pt --encoder_path /path/to/gin_e_finetuned.pt --output_dir ./logs/tsne_downstream
    python visualize_tsne_downstream.py --max_samples 100  # quick test: only 100 molecules (saves encoder + t-SNE time)
    python visualize_tsne_downstream.py --no_literature  # val+test cloud only: no literature markers or legend
"""
import csv
import os
import sys
import argparse
import random
import torch
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import Config
from models.gin_e import GINEEncoder
# t-SNE figure styling (markers, colors, square axes, legend) is defined in visualize_tsne.py
# and reused here so downstream plots match the SSL script.
from visualize_tsne import (
    _split_csv_line,
    compute_tsne,
    create_static_tsne_plot,
    create_interactive_tsne_plot,
)
from train_downstream import (
    split_data,
    MolecularGraphWithBinding,
)
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image

# Default paths
DEFAULT_CACHE_PATH = "/kfs3/scratch/yeming/ai4m/prediction/cache/downstream_graph_notag_cache.pt"
DEFAULT_ENCODER_PATH = "/kfs3/scratch/yeming/ai4m/prediction/checkpoints/downstream_notag_03222026/downstream/gin_e_finetuned.pt"
DEFAULT_OUTPUT_DIR = "logs/tsne_downstream"
# Default literature CSV: server path; fallback to project-relative path
DEFAULT_LITERATURE_CSV = "/kfs3/scratch/yeming/ai4m/prediction/dataset/literature/molecule_images_by_journal/molecules_cid_smiles.csv"
if not os.path.isfile(DEFAULT_LITERATURE_CSV):
    DEFAULT_LITERATURE_CSV = os.path.join(SCRIPT_DIR, "dataset", "literature", "molecule_images_by_journal", "molecules_cid_smiles.csv")

# Approximate atomic mass for molecular weight from graph (first node feature = atomic number)
ATOMIC_MASS = {
    1: 1.008, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    11: 22.99, 12: 24.305, 13: 26.98, 14: 28.085, 15: 30.974, 16: 32.065,
    17: 35.45, 19: 39.098, 20: 40.078, 34: 78.96, 35: 79.904, 53: 126.90,
}


def load_downstream_cache(cache_path: str) -> Tuple[List, List[float], List[str]]:
    """Load graphs, energies, cids from downstream_graph_cache.pt."""
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    graphs = data["graphs"]
    energies = data["energies"]
    cids = data["cids"]
    return graphs, energies, cids


def load_literature_finetuned_tsne_csv(
    csv_path: str,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load literature molecules from CSV with columns: mol_name, SMILES,
    functional_group, journal.

    Uses the same parsing as visualize_tsne.load_literature_csv: quote-aware line splitting
    via visualize_tsne._split_csv_line; if mol_name has unquoted commas, extra fields are merged
    into the first column and trailing columns align with the header.

    Returns (mol_names, smiles_list, functional_groups, journals).
    Deduplicates by SMILES so that the same molecule appears only once in the t-SNE plot.
    """
    if not os.path.isfile(csv_path):
        return [], [], [], []
    mol_names: List[str] = []
    smiles_list: List[str] = []
    functional_groups: List[str] = []
    journals: List[str] = []
    seen_smiles: set = set()

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        content = f.read()
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return [], [], [], []

    header = _split_csv_line(lines[0])
    if len(header) < 4:
        return [], [], [], []
    header_norm = [
        h.strip().lstrip("\ufeff").lower().replace(" ", "_") for h in header
    ]
    num_cols = len(header_norm)

    NAME_KEYS = (
        "mol_name",
        "molecule_name",
        "name",
        "compound_name",
        "query_name",
    )

    def col(name: str) -> int:
        if name in header_norm:
            return header_norm.index(name)
        defaults = {"mol_name": 0, "molecule_name": 0, "smiles": 1, "functional_group": 2, "journal": 3}
        return defaults[name]

    idx_name = next(
        (header_norm.index(k) for k in NAME_KEYS if k in header_norm),
        col("mol_name"),
    )
    idx_smiles = col("smiles") if "smiles" in header_norm else None
    idx_labeled = header_norm.index("labeled_smiles") if "labeled_smiles" in header_norm else None
    idx_func = col("functional_group")
    idx_journal = col("journal")

    def row_norm_from_wide_row(row: List[str], tail_len: int) -> Dict[str, str]:
        """Merge fragments for the name column at idx_name; map tail to columns after name."""
        prefix = row[:idx_name]
        end_name = len(row) - tail_len
        name_merged = ",".join(row[idx_name:end_name]).strip()
        tail = row[end_name:]
        out: dict = {}
        for k in range(idx_name):
            out[header_norm[k]] = prefix[k].strip() if k < len(prefix) else ""
        out[header_norm[idx_name]] = name_merged
        for j in range(idx_name + 1, num_cols):
            ti = j - idx_name - 1
            out[header_norm[j]] = tail[ti].strip() if ti < len(tail) else ""
        return out

    def pick_name(rn: dict) -> str:
        for k in NAME_KEYS:
            v = (rn.get(k) or "").strip()
            if v:
                return v
        return (rn.get(header_norm[idx_name], "") or "").strip()

    def pick_smiles(rn: dict, row: List[str]) -> str:
        s = (rn.get("smiles") or "").strip()
        if s:
            return s
        if "labeled_smiles" in rn:
            return (rn.get("labeled_smiles") or "").strip()
        if idx_smiles is not None and idx_smiles < len(row):
            return (row[idx_smiles] or "").strip()
        if idx_labeled is not None and idx_labeled < len(row):
            return (row[idx_labeled] or "").strip()
        return ""

    for line in lines[1:]:
        if not line.strip():
            continue
        row = _split_csv_line(line)
        if len(row) < num_cols:
            continue

        # Quote-aware split + merge unquoted commas in the name column (same idea as visualize_tsne).
        # Anchor at idx_name so cid/other leading columns stay correct when mol_name is not first.
        if len(row) > num_cols:
            tail_len = num_cols - idx_name - 1
            if tail_len < 1:
                continue
            row_norm = row_norm_from_wide_row(row, tail_len)
            name = pick_name(row_norm)
            smiles = pick_smiles(row_norm, row).strip()
            func = (row_norm.get("functional_group") or "").strip()
            journal = (row_norm.get("journal") or "").strip()
        else:
            row_norm = {header_norm[j]: (row[j] if j < len(row) else "").strip() for j in range(num_cols)}
            name = pick_name(row_norm)
            smiles = pick_smiles(row_norm, row).strip()
            func = (row_norm.get("functional_group") or "").strip()
            journal = (row_norm.get("journal") or "").strip()

        if not smiles:
            continue
        if smiles.isdigit():
            continue
        if smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)
        mol_names.append(name or "?")
        smiles_list.append(smiles)
        functional_groups.append(func)
        journals.append(journal)
    return mol_names, smiles_list, functional_groups, journals


def build_literature_graphs(
    mol_names: List[str],
    labeled_smiles_list: List[str],
    journals: List[str],
    functional_groups: Optional[List[str]] = None,
) -> Tuple[List[Any], List[str], List[str]]:
    """
    Build PyG Data graphs for literature molecules from labeled SMILES.
    Binding atoms are those with atom map number > 0 in the labeled SMILES.
    Legend name is {mol_name}_{functional_group} when functional_group is given (so rows with
    same mol_name but different labeled_SMILES/functional_group are distinct in the t-SNE legend).
    Returns (lit_graphs, lit_names, lit_journals) for successfully built molecules.
    """
    lit_graphs: List[Any] = []
    lit_names: List[str] = []
    lit_journals: List[str] = []
    if functional_groups is None:
        functional_groups = [""] * len(mol_names)
    for i, (name, labeled_smiles, journal) in enumerate(zip(mol_names, labeled_smiles_list, journals)):
        try:
            mol = Chem.MolFromSmiles(labeled_smiles)
            if mol is None or mol.GetNumAtoms() < 2:
                continue
            # Clear atom map numbers (no binding site tagging)
            for a in mol.GetAtoms():
                a.SetAtomMapNum(0)
            mol_heavy = Chem.RemoveHs(mol)
            if mol_heavy.GetNumAtoms() < 2:
                continue
            graph = MolecularGraphWithBinding.mol_to_graph(mol_heavy, binding_atom_indices=[])
            lit_graphs.append(graph)
            func = (functional_groups[i] or "").strip()
            display_name = f"{name}_{func}" if func else name
            lit_names.append(display_name)
            lit_journals.append(journal)
        except Exception:
            continue
    return lit_graphs, lit_names, lit_journals


def _split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """Replicate train_downstream split indices (same as split_data) for indexing cids/mol_list."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )




def molecular_weight_from_graph(graph) -> float:
    """Estimate molecular weight from graph node features (x[:,0] = atomic number)."""
    an = graph.x[:, 0].cpu().numpy()
    return float(np.sum([ATOMIC_MASS.get(int(z), 12.0) for z in an]))


def load_finetuned_encoder(checkpoint_path: str, config: Config, device: torch.device) -> GINEEncoder:
    """Load GIN-E encoder from gin_e_finetuned.pt (encoder_state_dict)."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
    model = GINEEncoder(
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=config.edge_feature_dim,
        node_embedding_dim=config.node_embedding_dim,
        edge_embedding_dim=config.edge_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gin_layers,
        dropout=config.dropout,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "encoder_state_dict" in ckpt:
        model.load_state_dict(ckpt["encoder_state_dict"])
        print(f"  Loaded encoder from epoch {ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings(model: GINEEncoder, graphs: List, device: torch.device, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings; return (embeddings, invalid_mask)."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(graphs), batch_size), desc="Extracting embeddings"):
            batch_graphs = graphs[i : i + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(device)
            emb = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    nan_mask = np.isnan(embeddings).any(axis=1)
    inf_mask = np.isinf(embeddings).any(axis=1)
    invalid_mask = nan_mask | inf_mask
    if invalid_mask.any():
        print(f"  Warning: {invalid_mask.sum()} samples with NaN/Inf embeddings")
    return embeddings, invalid_mask


def main():
    parser = argparse.ArgumentParser(description="t-SNE of downstream val+test with finetuned GIN-E")
    parser.add_argument("--cache_path", default=DEFAULT_CACHE_PATH, help="Path to downstream_graph_cache.pt")
    parser.add_argument("--encoder_path", default=DEFAULT_ENCODER_PATH, help="Path to gin_e_finetuned.pt")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--n_iter", type=int, default=1000, help="t-SNE iterations")
    parser.add_argument("--learning_rate", type=float, default=200.0, help="t-SNE learning rate")
    parser.add_argument("--initialization", default="random", choices=["pca", "random"], help="t-SNE init")
    parser.add_argument("--metric", default="euclidean", help="t-SNE metric")
    parser.add_argument("--batch_size", type=int, default=512, help="Embedding batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for t-SNE (after energy filter). Subsample before embedding extraction to save compute and time (e.g. for testing). Default: use all.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: config.seed)")
    parser.add_argument("--literature_csv", default=DEFAULT_LITERATURE_CSV, help="CSV of literature molecules (mol_name, SMILES, labeled_SMILES, functional_group, journal) to plot with marker+legend. Optional.")
    parser.add_argument(
        "--hide_literature_markers",
        "--no_literature",
        dest="no_literature",
        action="store_true",
        help=(
            "Do not load literature CSV: t-SNE uses val+test cache only (no literature embeddings). "
            "Removes literature markers from plots and hides the Plotly legend."
        ),
    )
    args = parser.parse_args()

    config = Config()
    seed = args.seed if args.seed is not None else config.seed
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load cache
    print(f"\nLoading downstream graph cache: {args.cache_path}")
    graphs, energies, cids = load_downstream_cache(args.cache_path)
    print(f"  Loaded {len(graphs)} graphs")

    # Split train/val/test using the same function and ratios as train_downstream main
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    train_graphs, train_energies, val_graphs, val_energies, test_graphs, test_energies = split_data(
        graphs, energies,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    print(f"  Split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test (same as downstream training)")

    # Indices for val/test
    train_idx, val_idx, test_idx = _split_indices(len(graphs), train_ratio, val_ratio, test_ratio, seed)

    # Combine val + test for t-SNE
    val_test_graphs = val_graphs + test_graphs
    val_test_cids = [cids[i] for i in val_idx] + [cids[i] for i in test_idx]
    val_test_energies = [energies[i] for i in val_idx] + [energies[i] for i in test_idx]
    n_val, n_test = len(val_graphs), len(test_graphs)
    print(f"  Val+test: {len(val_test_graphs)} molecules ({n_val} val, {n_test} test)")

    if len(val_test_graphs) == 0:
        print("No val+test graphs. Exiting.")
        return

    # Filter to molecules with binding energy in [-2, 0] eV
    binding_energies_full = np.array(val_test_energies, dtype=np.float64)
    energy_mask = (binding_energies_full >= -2.0) & (binding_energies_full <= 0.0)
    n_before = len(val_test_graphs)
    val_test_graphs = [val_test_graphs[i] for i in range(n_before) if energy_mask[i]]
    val_test_cids = [val_test_cids[i] for i in range(n_before) if energy_mask[i]]
    val_test_energies = [val_test_energies[i] for i in range(n_before) if energy_mask[i]]
    print(f"  Filtered to binding energy in [-2, 0] eV: {len(val_test_graphs)} molecules (from {n_before})")
    if len(val_test_graphs) == 0:
        print("No molecules in [-2, 0] eV. Exiting.")
        return

    # Subsample to max_samples *before* mol building and embedding extraction (saves CSV scan + encoder time)
    if args.max_samples is not None and len(val_test_graphs) > args.max_samples:
        rng = random.Random(seed)
        indices = rng.sample(range(len(val_test_graphs)), args.max_samples)
        val_test_graphs = [val_test_graphs[i] for i in indices]
        val_test_cids = [val_test_cids[i] for i in indices]
        val_test_energies = [val_test_energies[i] for i in indices]
        print(f"  Subsampled to --max_samples={args.max_samples} for t-SNE (seed={seed})")

    # Build CID -> SMILES lookup from downstream CSV for molecule image rendering
    csv_path = (config.downstream_csv or "").strip()
    if not csv_path or not os.path.isfile(csv_path):
        csv_path = os.path.join(SCRIPT_DIR, "dataset", "prediction", "min_ads_mult1p2_struct_cleaned_merged.csv")
    cid_to_smiles: dict = {}
    if os.path.isfile(csv_path):
        print(f"\nBuilding CID -> SMILES lookup from: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fnames = list(reader.fieldnames or [])
            smiles_col = "SMILES" if "SMILES" in fnames else ("smiles" if "smiles" in fnames else None)
            if smiles_col:
                cid_need = set(int(c) for c in val_test_cids)
                for row in reader:
                    try:
                        cid = int(row["cid"])
                    except (ValueError, KeyError):
                        continue
                    if cid in cid_need:
                        s = (row.get(smiles_col) or "").strip()
                        if s:
                            cid_to_smiles[cid] = s
                    if len(cid_to_smiles) == len(cid_need):
                        break
        print(f"  Found SMILES for {len(cid_to_smiles)} / {len(val_test_cids)} molecules")

    # Load literature molecules (optional) and build graphs from labeled SMILES — styling matches visualize_tsne.py
    lit_graphs: List[Any] = []
    lit_names: List[str] = []
    lit_journals: List[str] = []
    if args.no_literature:
        print("\n--no_literature / --hide_literature_markers: skipping literature molecules, markers, and legend.")
    else:
        literature_csv = args.literature_csv
        if not os.path.isabs(literature_csv):
            literature_csv = os.path.join(SCRIPT_DIR, literature_csv)
        if os.path.isfile(literature_csv):
            mol_names, labeled_smiles_list, _functional_groups, journals_raw = load_literature_finetuned_tsne_csv(literature_csv)
            if mol_names:
                lit_graphs, lit_names, lit_journals = build_literature_graphs(
                    mol_names, labeled_smiles_list, journals_raw, functional_groups=_functional_groups
                )
                print(f"\nLoaded {len(lit_graphs)} literature molecules from {literature_csv} (same marker/legend style as visualize_tsne.py)")
            else:
                print(f"\nNo rows in literature CSV: {literature_csv}")
        else:
            print(f"\nLiterature CSV not found (skipping): {literature_csv}")

    # Combine cache + literature for embedding extraction, then t-SNE
    n_val_test = len(val_test_graphs)
    all_graphs = val_test_graphs + lit_graphs
    print(f"\nLoading finetuned encoder: {args.encoder_path}")
    encoder = load_finetuned_encoder(args.encoder_path, config, device)
    print(f"  Extracting embeddings for {len(all_graphs)} molecules ({n_val_test} cache + {len(lit_graphs)} literature)...")
    embeddings_all, invalid_mask = extract_embeddings(encoder, all_graphs, device, args.batch_size)
    val_valid = ~invalid_mask[:n_val_test]
    lit_valid = ~invalid_mask[n_val_test:] if len(lit_graphs) > 0 else np.array([], dtype=bool)
    val_test_graphs = [val_test_graphs[i] for i in range(n_val_test) if val_valid[i]]
    val_test_cids = [val_test_cids[i] for i in range(n_val_test) if val_valid[i]]
    val_test_energies = [val_test_energies[i] for i in range(n_val_test) if val_valid[i]]
    if len(lit_graphs) > 0:
        lit_names = [lit_names[i] for i in range(len(lit_names)) if lit_valid[i]]
        lit_journals = [lit_journals[i] for i in range(len(lit_journals)) if lit_valid[i]]
    embeddings_val = embeddings_all[:n_val_test][val_valid]
    embeddings_lit = embeddings_all[n_val_test:][lit_valid] if len(lit_graphs) > 0 else np.zeros((0, embeddings_all.shape[1]), dtype=embeddings_all.dtype)
    all_embeddings = np.concatenate([embeddings_val, embeddings_lit], axis=0)
    if len(val_test_graphs) == 0:
        print("No valid cache embeddings. Exiting.")
        return
    print(f"  Valid: {embeddings_val.shape[0]} cache, {embeddings_lit.shape[0]} literature")
    binding_energies = np.array(val_test_energies, dtype=np.float64)
    print(f"  Binding energy range: {binding_energies.min():.3f} - {binding_energies.max():.3f} eV")

    # t-SNE on combined embeddings (cache + literature)
    tsne_coords, scaler, tsne = compute_tsne(
        all_embeddings,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        learning_rate=args.learning_rate,
        initialization=args.initialization,
        metric=args.metric,
        random_state=seed,
    )
    n_val_clean = len(embeddings_val)
    val_test_tsne = tsne_coords[:n_val_clean]
    lit_tsne = tsne_coords[n_val_clean:] if len(embeddings_lit) > 0 else np.zeros((0, 2))

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Render canonical 2D molecule images from SMILES
    prerendered_images: List[Optional[Image.Image]] = [None] * len(val_test_graphs)
    print(f"\nRendering 2D molecule images from SMILES for {len(val_test_graphs)} samples...")
    n_rendered = 0
    for i in tqdm(range(len(val_test_graphs)), desc="Rendering molecule images"):
        cid_int = int(val_test_cids[i]) if i < len(val_test_cids) else None
        smiles = cid_to_smiles.get(cid_int) if cid_int is not None else None
        if not smiles:
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() < 2:
                continue
            mol = Chem.RemoveHs(mol)
            AllChem.Compute2DCoords(mol)
            prerendered_images[i] = Draw.MolToImage(mol, size=(400, 400))
            n_rendered += 1
        except Exception:
            continue
    print(f"  Rendered {n_rendered} / {len(val_test_graphs)} molecule images")

    # Save rendered images to disk
    images_dir = os.path.join(args.output_dir, "tsne_downstream_images")
    n_saved = sum(1 for img in prerendered_images if img is not None)
    if n_saved > 0:
        os.makedirs(images_dir, exist_ok=True)
        print(f"Saving {n_saved} molecule images to {images_dir}")
        for i, img in enumerate(prerendered_images):
            if img is not None:
                img.save(os.path.join(images_dir, f"mol_{i}.png"))
        print(f"  Saved {n_saved} images")

    static_path = os.path.join(args.output_dir, "tsne_downstream_static.png")
    title = f"t-SNE of Downstream Val+Test (Finetuned GIN-E)\n{len(val_test_graphs)} molecules, colored by Binding Energy (eV)"
    if len(lit_tsne) > 0:
        title += f"; {len(lit_tsne)} literature"
    create_static_tsne_plot(
        val_test_tsne,
        binding_energies,
        static_path,
        title=title,
        lit_tsne=lit_tsne if len(lit_tsne) > 0 else None,
        lit_names=lit_names if len(lit_tsne) > 0 else None,
        lit_journals=lit_journals if len(lit_tsne) > 0 else None,
        colorbar_label="Binding energy (eV)",
    )

    # Interactive plot — reuses the pre-rendered highlighted images for hover; literature in legend
    interactive_path = os.path.join(args.output_dir, "tsne_downstream_interactive.html")
    has_images = any(img is not None for img in prerendered_images)
    interactive_title = f"Interactive t-SNE (Downstream Val+Test, Finetuned GIN-E)<br><sub>{len(val_test_graphs)} molecules, colored by Binding Energy (eV)"
    if len(lit_tsne) > 0:
        interactive_title += f"; {len(lit_tsne)} literature"
    interactive_title += "</sub>"
    create_interactive_tsne_plot(
        val_test_tsne,
        binding_energies,
        all_embeddings[:n_val_clean],
        [],
        interactive_path,
        title=interactive_title,
        show_images=has_images,
        lit_tsne=lit_tsne if len(lit_tsne) > 0 else None,
        lit_names=lit_names if len(lit_tsne) > 0 else None,
        lit_journals=lit_journals if len(lit_tsne) > 0 else None,
        color_label="Binding energy",
        color_units="eV",
        color_fmt=".3f",
        colorbar_title="Binding energy (eV)",
        prerendered_images=prerendered_images if has_images else None,
        show_legend=len(lit_tsne) > 0 and not args.no_literature,
    )

    # Save arrays (tsne_coords and embeddings are combined cache+literature; binding_energies is cache only)
    np.save(os.path.join(args.output_dir, "tsne_coordinates.npy"), tsne_coords)
    np.save(os.path.join(args.output_dir, "embeddings.npy"), all_embeddings)
    np.save(os.path.join(args.output_dir, "binding_energies.npy"), binding_energies)

    # Save index → CID → binding energy mapping as CSV
    import csv as _csv
    metadata_path = os.path.join(args.output_dir, "molecule_metadata.csv")
    with open(metadata_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["index", "image_file", "cid", "binding_energy_eV",
                         "tsne_x", "tsne_y"])
        for i in range(len(val_test_cids)):
            has_img = prerendered_images[i] is not None
            writer.writerow([
                i,
                f"mol_{i}.png" if has_img else "",
                int(val_test_cids[i]),
                f"{binding_energies[i]:.6f}",
                f"{tsne_coords[i, 0]:.6f}",
                f"{tsne_coords[i, 1]:.6f}",
            ])

    print(f"\nSaved t-SNE outputs to {args.output_dir}")
    print(f"  Static: {static_path}")
    print(f"  Interactive: {interactive_path}")
    print(f"  Metadata CSV: {metadata_path}")


if __name__ == "__main__":
    main()
