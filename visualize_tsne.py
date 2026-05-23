"""
Generate t-SNE visualizations of molecular embeddings from pretrained GIN-E encoder.
Loads SMILES only from CSV, splits 80% train / 20% validation (same as training).
Then --max_samples (if set) chooses that many from the validation set. MolFromSmiles
is called only for those validation samples; graphs are built only for them.
"""
import csv
import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import base64
from io import BytesIO
from typing import List, Optional, Tuple

from config import Config
from dataset.ssl.molecular_graph import MolToGraphConverter
from models.gin_e import GINEEncoder


def load_smiles_from_csv(csv_file: str, max_molecules: Optional[int] = None) -> List[str]:
    """Load SMILES strings only from CSV (no MolFromSmiles yet)."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    smiles_list = []
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading SMILES"):
            s = (row.get("SMILES") or "").strip()
            if not s:
                continue
            smiles_list.append(s)
            if max_molecules and len(smiles_list) >= max_molecules:
                break
    return smiles_list


def _split_csv_line(line: str) -> List[str]:
    """Split a CSV line by commas, respecting double-quoted fields (commas inside quotes are not delimiters). Escaped quote \"\" inside a quoted field becomes one quote."""
    fields: List[str] = []
    current: List[str] = []
    in_quotes = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"':
            if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                current.append('"')
                i += 2
                continue
            in_quotes = not in_quotes
            i += 1
            continue
        if c == "," and not in_quotes:
            fields.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(c)
        i += 1
    fields.append("".join(current).strip())
    return fields


def load_literature_csv(csv_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Load literature molecules from molecules_cid_smiles.csv. Columns: molecule_name, cid, smiles, journal.
    Parses lines with quote-aware splitting so molecule names with commas (inside quotes) are never split."""
    if not os.path.exists(csv_path):
        return [], [], []
    smiles_list: List[str] = []
    names_list: List[str] = []
    journals_list: List[str] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        content = f.read()
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return [], [], []
    header = _split_csv_line(lines[0])
    if len(header) < 4:
        return [], [], []
    header_norm = [h.strip().lower().replace(" ", "_") for h in header]
    def col(name: str) -> int:
        if name in header_norm:
            return header_norm.index(name)
        return {"molecule_name": 0, "cid": 1, "smiles": 2, "journal": 3}[name]
    idx_name = col("molecule_name")
    idx_cid = col("cid")
    idx_smiles = col("smiles")
    idx_journal = col("journal")
    for line in lines[1:]:
        if not line.strip():
            continue
        row = _split_csv_line(line)
        if len(row) < 4:
            continue
        # Quote-aware split gives 4 columns when name is quoted; if unquoted name had commas we get >4
        if len(row) > 4:
            name = ",".join(row[:-3]).strip()
            cid = (row[-3] or "").strip()
            smiles = (row[-2] or "").strip()
            journal = (row[-1] or "").strip()
        else:
            name = (row[idx_name] if idx_name < len(row) else "").strip()
            cid = (row[idx_cid] if idx_cid < len(row) else "").strip()
            smiles = (row[idx_smiles] if idx_smiles < len(row) else "").strip()
            journal = (row[idx_journal] if idx_journal < len(row) else "").strip()
        if not smiles:
            continue
        if smiles.isdigit():
            continue
        smiles_list.append(smiles)
        names_list.append(name or f"CID{cid}" if cid else "?")
        journals_list.append(journal)
    return smiles_list, names_list, journals_list


def split_list(items: list, train_ratio: float = 0.8, val_ratio: float = 0.2, seed: int = 42):
    """Split a list into train/val (same indices as training process)."""
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio must equal 1.0, got {train_ratio + val_ratio}")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    indices = list(range(len(items)))
    random.shuffle(indices)
    train_size = int(len(items) * train_ratio)
    train_items = [items[i] for i in indices[:train_size]]
    val_items = [items[i] for i in indices[train_size:]]
    return train_items, val_items


def load_pretrained_encoder(config: Config, checkpoint_path: str, device: torch.device):
    """Load pretrained GIN-E encoder from checkpoint."""
    print(f"Loading pretrained encoder from {checkpoint_path}...")
    
    model = GINEEncoder(
        node_feature_dim=config.node_feature_dim,
        edge_feature_dim=config.edge_feature_dim,
        node_embedding_dim=config.node_embedding_dim,
        edge_embedding_dim=config.edge_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gin_layers,
        dropout=config.dropout
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss')
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
        print(f"  Loaded checkpoint: epoch={epoch}, loss={loss_str}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded checkpoint (no metadata)")
    
    model = model.to(device)
    model.eval()
    print(f"  Model loaded successfully")
    
    return model


def calculate_molecular_weights(molecules):
    """Calculate molecular weights for a list of RDKit molecules."""
    print(f"\nCalculating molecular weights for {len(molecules)} molecules...")
    molecular_weights = []
    for mol in tqdm(molecules, desc="Calculating molecular weights"):
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            molecular_weights.append(mw)
        else:
            molecular_weights.append(0.0)
    molecular_weights = np.array(molecular_weights)
    if len(molecular_weights) > 0:
        print(f"  Molecular weight range: {molecular_weights.min():.2f} - {molecular_weights.max():.2f} Da")
    else:
        print("  No molecules; molecular weight range N/A")
    return molecular_weights


def mol_to_base64_image(mol, size=(300, 300)):
    """Convert RDKit molecule to base64 encoded PNG image for embedding in HTML."""
    if mol is None:
        return None
    
    try:
        # Generate 2D coordinates if not present
        mol = Chem.Mol(mol)
        if mol.GetNumConformers() == 0:
            from rdkit.Chem import AllChem
            AllChem.Compute2DCoords(mol)
        
        # Draw molecule to image
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"  Warning: Failed to generate image for molecule: {e}")
        return None


def extract_embeddings(model: GINEEncoder, graphs, device: torch.device, batch_size: int = 512):
    """Extract embeddings from molecular graphs using the pretrained encoder."""
    print(f"\nExtracting embeddings for {len(graphs)} graphs...")
    
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, len(graphs), batch_size), desc="Extracting embeddings"):
            batch_graphs = graphs[i:i+batch_size]
            batch = Batch.from_data_list(batch_graphs)
            batch = batch.to(device)
            
            # Forward pass to get embeddings
            emb = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch
            )  # [batch_size, hidden_dim]
            
            embeddings.append(emb.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"  Extracted embeddings shape: {embeddings.shape}")
    
    # Check for NaN or Inf values
    nan_mask = np.isnan(embeddings).any(axis=1)
    inf_mask = np.isinf(embeddings).any(axis=1)
    invalid_mask = nan_mask | inf_mask
    
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        print(f"  Warning: Found {n_invalid} samples with NaN or Inf values ({n_invalid/len(embeddings)*100:.2f}%)")
    
    return embeddings, invalid_mask


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    learning_rate: float = 200.0,
    initialization: str = "pca",
    metric: str = "euclidean",
    random_state: int = 42
):
    """Compute t-SNE embedding."""
    print(f"\nComputing t-SNE (perplexity={perplexity}, n_iter={n_iter}, learning_rate={learning_rate}, init={initialization}, metric={metric})...")
    
    # Check for NaN or Inf values (should already be filtered, but double-check)
    nan_mask = np.isnan(embeddings).any(axis=1)
    inf_mask = np.isinf(embeddings).any(axis=1)
    invalid_mask = nan_mask | inf_mask
    
    if invalid_mask.any():
        raise ValueError(f"Input embeddings contain {invalid_mask.sum()} samples with NaN or Inf values. Please filter them before calling compute_tsne.")
    
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least 2 samples.")
    # sklearn TSNE requires perplexity < n_samples
    effective_perplexity = min(perplexity, n_samples - 1)
    if effective_perplexity < perplexity:
        print(f"  Note: perplexity capped from {perplexity} to {effective_perplexity} (n_samples={n_samples}).")
    
    # Standardize embeddings for better t-SNE performance
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Check again after scaling (shouldn't happen, but just in case)
    if np.isnan(embeddings_scaled).any() or np.isinf(embeddings_scaled).any():
        print(f"  Warning: NaN/Inf detected after scaling. Replacing with zeros...")
        embeddings_scaled = np.nan_to_num(embeddings_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute t-SNE (sklearn.manifold.TSNE)
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=n_iter,
        learning_rate=learning_rate,
        init=initialization,
        metric=metric,
        random_state=random_state,
        verbose=1
    )
    
    tsne_embedding = tsne.fit_transform(embeddings_scaled)
    
    print(f"  t-SNE embedding shape: {tsne_embedding.shape}")
    
    return tsne_embedding, scaler, tsne


# Markers and colors for literature molecules: one distinct style per CSV row (cycles if many molecules)
LITERATURE_MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', 'H', 'X']
# High-contrast, more distinguishable colors for literature markers.
LITERATURE_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#1b9e77', '#d95f02',
    '#7570b3', '#66a61e', '#e7298a', '#e6ab02', '#a6761d',
    '#17becf', '#bcbd22', '#8c564b', '#393b79', '#637939',
]
LITERATURE_MARKER_SIZE = 55  # literature marker size (static plot)


def _literature_style_for_index(idx: int) -> Tuple[str, str]:
    """Return marker/color pair with color shift to avoid repeating styles every 12 molecules."""
    n_markers = len(LITERATURE_MARKERS)
    n_colors = len(LITERATURE_COLORS)
    marker = LITERATURE_MARKERS[idx % n_markers]
    # Shift color each full marker cycle so (marker, color) pairs remain unique longer.
    color_cycle_shift = idx // n_markers
    color = LITERATURE_COLORS[(idx + color_cycle_shift) % n_colors]
    return marker, color


def create_static_tsne_plot(
    tsne_coords: np.ndarray,
    molecular_weights: np.ndarray,
    output_path: str,
    title: str = "t-SNE Visualization",
    lit_tsne: Optional[np.ndarray] = None,
    lit_names: Optional[List[str]] = None,
    lit_journals: Optional[List[str]] = None,
    colorbar_label: Optional[str] = None,
):
    """Create static matplotlib t-SNE plot. Val points colored by molecular weight (or other scalar);
    each literature molecule gets its own marker/color; legend entries show that marker and the molecule name.
    If colorbar_label is set, a color bar is drawn with that label.
    Layout: wider figure when literature legend is present so colorbar+plot+legend don't overlap."""
    print(f"\nCreating static t-SNE plot (colored by {'scalar' if colorbar_label else 'molecular weight'})...")

    has_lit = lit_tsne is not None and lit_names is not None and len(lit_tsne) > 0
    # Keep the plotting panel square; give extra figure width only for legend/colorbar.
    fig, ax = plt.subplots(figsize=(13, 10) if has_lit else (10, 10))
    ax.set_axisbelow(True)

    scatter = ax.scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=molecular_weights,
        cmap='plasma',
        alpha=0.7,
        s=2.5,
        edgecolors='none',
        label='_nolegend_'
    )
    if colorbar_label is not None:
        plt.colorbar(scatter, ax=ax, label=colorbar_label, shrink=0.8)

    x_all = np.concatenate([tsne_coords[:, 0], lit_tsne[:, 0]]) if has_lit else tsne_coords[:, 0]
    y_all = np.concatenate([tsne_coords[:, 1], lit_tsne[:, 1]]) if has_lit else tsne_coords[:, 1]
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    ax.set_xlim(x_mid - (x_span * 1.10) / 2, x_mid + (x_span * 1.10) / 2)
    ax.set_ylim(y_mid - (y_span * 1.10) / 2, y_mid + (y_span * 1.10) / 2)
    ax.set_aspect('equal', adjustable='box')
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)

    if has_lit:
        for i in range(len(lit_tsne)):
            marker, color = _literature_style_for_index(i)
            label = lit_names[i]
            ax.scatter(
                lit_tsne[i, 0],
                lit_tsne[i, 1],
                marker=marker,
                s=LITERATURE_MARKER_SIZE,
                facecolors=color,
                edgecolors='#111111',
                linewidths=0.9,
                zorder=5,
                label=label,
            )

    ax.set_title(title)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True, color='lightgray', linewidth=0.5)

    if has_lit:
        legend = ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.35 if colorbar_label else 1.02, 1.0),
            fontsize=12,
            frameon=False,
            borderaxespad=0,
        )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved static plot to {output_path}")
    plt.close()


def create_interactive_tsne_plot(
    tsne_coords: np.ndarray,
    molecular_weights: np.ndarray,
    embeddings: np.ndarray,
    molecules: list,
    output_path: str,
    title: str = "Interactive t-SNE Visualization",
    show_images: bool = False,
    lit_tsne: Optional[np.ndarray] = None,
    lit_names: Optional[List[str]] = None,
    lit_journals: Optional[List[str]] = None,
    color_label: str = "Mol. weight",
    color_units: str = "Da",
    color_fmt: str = ".1f",
    colorbar_title: Optional[str] = None,
    prerendered_images: Optional[list] = None,
    show_legend: bool = True,
):
    """Create interactive plotly t-SNE plot. If show_images=True, hover shows structure images.
    Each literature molecule has its own marker; legend shows marker + molecule name.
    color_label/color_units/color_fmt control the hover text for the color value (e.g. 'Binding energy', 'eV', '.3f').
    If colorbar_title is set, a color bar is shown with that title.
    If prerendered_images is provided (list of PIL Images or None), those are used directly
    instead of generating images from mol objects."""
    print(f"\nCreating interactive t-SNE plot (colored by {color_label})...")
    
    if show_images:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        output_basename = os.path.splitext(os.path.basename(output_path))[0]
        images_dir = os.path.join(output_dir, f"{output_basename}_images")
        os.makedirs(images_dir, exist_ok=True)
        print(f"  Saving molecular structure images to: {images_dir}")
        
        image_paths = []
        if prerendered_images is not None:
            for i, img in enumerate(prerendered_images):
                if img is not None:
                    img_path = os.path.join(images_dir, f"mol_{i}.png")
                    img.save(img_path)
                    rel_img_path = os.path.join(f"{output_basename}_images", f"mol_{i}.png")
                    image_paths.append(rel_img_path)
                else:
                    image_paths.append(None)
        else:
            for i, mol in enumerate(tqdm(molecules, desc="Generating structure images", total=len(molecules))):
                try:
                    mol = Chem.Mol(mol)
                    if mol.GetNumConformers() == 0:
                        from rdkit.Chem import AllChem
                        AllChem.Compute2DCoords(mol)
                    img = Draw.MolToImage(mol, size=(300, 300), kekulize=True)
                    img_path = os.path.join(images_dir, f"mol_{i}.png")
                    img.save(img_path)
                    rel_img_path = os.path.join(f"{output_basename}_images", f"mol_{i}.png")
                    image_paths.append(rel_img_path)
                except Exception as e:
                    print(f"  Warning: Failed to generate image for molecule {i}: {e}")
                    image_paths.append(None)
        
        hover_texts = []
        for i in range(len(tsne_coords)):
            if i < len(image_paths) and image_paths[i] is not None:
                hover_html = f'<img src="{image_paths[i]}" style="max-width:300px; max-height:300px; border:2px solid #666; border-radius:5px;">'
            else:
                hover_html = ""
            hover_texts.append(hover_html)
    else:
        hover_texts = [f"Index: {i}<br>{color_label}: {mw:{color_fmt}} {color_units}" for i, mw in enumerate(molecular_weights)]
    
    # Create scatter plot with color mapping
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tsne_coords[:, 0],
        y=tsne_coords[:, 1],
        mode='markers',
        marker=dict(
            size=5,
            color=molecular_weights,
            colorscale='Plasma',
            opacity=0.7,
            line=dict(width=0),
            showscale=colorbar_title is not None,
            colorbar=dict(title=colorbar_title) if colorbar_title else None,
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Molecules'
    ))
    
    # Literature points: unique marker/color per molecule with color shifting across marker cycles.
    _plt_to_plotly = {'o': 'circle', 's': 'square', '^': 'triangle-up', 'D': 'diamond', 'v': 'triangle-down', '<': 'triangle-left', '>': 'triangle-right', 'p': 'pentagon', 'h': 'hexagon', '*': 'star', 'H': 'hexagon2', 'X': 'x'}
    if lit_tsne is not None and lit_names is not None and len(lit_tsne) > 0:
        for i in range(len(lit_tsne)):
            marker, color = _literature_style_for_index(i)
            symbol = _plt_to_plotly.get(marker, 'circle')
            label = lit_names[i]
            fig.add_trace(go.Scatter(
                x=[lit_tsne[i, 0]],
                y=[lit_tsne[i, 1]],
                mode='markers',
                marker=dict(size=16, symbol=symbol, color=color, line=dict(width=1.1, color='#111111')),
                name=label,
                legendgroup=str(i),
            ))
    
    # Equal axis length (same scale on x and y)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Arial Black")
        ),
        xaxis=dict(
            title=dict(
                text='t-SNE Component 1',
                font=dict(size=14)
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            scaleanchor='y',
            scaleratio=1,
        ),
        yaxis=dict(
            title=dict(
                text='t-SNE Component 2',
                font=dict(size=14)
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        width=1200,
        height=1000,
        hovermode='closest',
        showlegend=show_legend,
        legend=dict(
            font=dict(size=16),
            borderwidth=0,
        ),
    )
    
    # Save interactive plot
    fig.write_html(output_path)
    print(f"  Saved interactive plot to {output_path}")


def main():
    """Main function to generate t-SNE visualizations."""
    parser = argparse.ArgumentParser(description="Generate t-SNE visualizations of molecular embeddings")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained model checkpoint (default: checkpoints/best_model.pt)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max molecules from validation set to use for t-SNE (default: None = use all validation)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity parameter (default: 30.0)")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of t-SNE iterations (default: 1000)")
    parser.add_argument("--learning_rate", type=float, default=200.0, help="t-SNE learning rate (default: 200.0)")
    parser.add_argument("--initialization", type=str, default="pca", choices=["pca", "random"], help="t-SNE initialization method (default: pca)")
    parser.add_argument("--metric", type=str, default="euclidean", help="t-SNE distance metric (default: euclidean)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots (default: logs/tsne)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for embedding extraction (default: 512)")
    parser.add_argument("--literature_csv", type=str, default="dataset/literature/molecule_images_by_journal/molecules_cid_smiles.csv", help="CSV of literature molecules (name, cid, smiles, journal) to include in t-SNE")
    parser.add_argument("--images", action="store_true", help="Show molecular structure images on hover in interactive plot")
    parser.add_argument(
        "--hide_literature_markers",
        action="store_true",
        help="Remove literature markers and legend by skipping literature molecules in plotting/t-SNE.",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(config.checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please run train_ssl.py first.")
    
    output_dir = args.output_dir if args.output_dir else os.path.join(config.log_dir, "tsne")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pretrained encoder
    model = load_pretrained_encoder(config, checkpoint_path, device)
    
    # Load SMILES only from CSV (no MolFromSmiles yet)
    print(f"\nLoading dataset...")
    print(f"Loading SMILES from {config.csv_file}...")
    smiles_list = load_smiles_from_csv(config.csv_file, max_molecules=config.max_molecules)
    print(f"Loaded {len(smiles_list)} SMILES")
    
    # Split: 80% train / 20% validation (same as training process)
    train_ratio = config.train_val_split
    val_ratio = 1.0 - train_ratio
    train_smiles, val_smiles = split_list(smiles_list, train_ratio=train_ratio, val_ratio=val_ratio, seed=config.seed)
    print(f"Split: {len(train_smiles)} train, {len(val_smiles)} validation (same as training)")
    if len(val_smiles) == 0:
        raise ValueError("Validation set is empty after split. Need more molecules in CSV.")
    
    # Optionally limit validation samples for t-SNE (--max_samples); only these will be parsed to Mol
    if args.max_samples is not None and len(val_smiles) > args.max_samples:
        print(f"Choosing {args.max_samples} SMILES from validation set for t-SNE...")
        np.random.seed(config.seed)
        subsample_indices = np.random.choice(len(val_smiles), args.max_samples, replace=False)
        val_smiles = [val_smiles[i] for i in subsample_indices]
    print(f"Using {len(val_smiles)} validation samples for t-SNE (MolFromSmiles only for these)")
    
    # MolFromSmiles only for the chosen validation SMILES; then build graphs
    print(f"\nParsing SMILES to molecules and building graphs for {len(val_smiles)} samples...")
    converter = MolToGraphConverter()
    val_graphs = []
    val_molecules = []
    n_skipped = 0
    for s in tqdm(val_smiles, desc="MolFromSmiles + graph"):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None or mol.GetNumAtoms() < 2:
                n_skipped += 1
                continue
            graph = converter.convert(mol)
            if graph is not None and graph.num_nodes >= 2:
                val_graphs.append(graph)
                val_molecules.append(mol)
            else:
                n_skipped += 1
        except Exception:
            n_skipped += 1
            continue
    if n_skipped:
        print(f"Created {len(val_graphs)} graphs (skipped {n_skipped} failed parses/conversions)")
    else:
        print(f"Created {len(val_graphs)} graphs")
    
    if len(val_graphs) == 0:
        raise ValueError("No valid graphs after conversion. Cannot run t-SNE.")
    
    # Load literature molecules (molecules_cid_smiles.csv) and add to pipeline
    lit_graphs: List = []
    lit_names: List[str] = []
    lit_journals: List[str] = []
    if args.hide_literature_markers:
        print("\n--hide_literature_markers enabled: skipping literature molecules, markers, and legend.")
    else:
        literature_csv_path = args.literature_csv
        if not os.path.isabs(literature_csv_path):
            literature_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), literature_csv_path)
        if os.path.exists(literature_csv_path):
            lit_smiles, lit_names_raw, lit_journals_raw = load_literature_csv(literature_csv_path)
            print(f"\nLoaded {len(lit_smiles)} literature molecules from {literature_csv_path}")
            for s, name, journal in zip(lit_smiles, lit_names_raw, lit_journals_raw):
                try:
                    mol = Chem.MolFromSmiles(s)
                    if mol is None or mol.GetNumAtoms() < 2:
                        continue
                    graph = converter.convert(mol)
                    if graph is not None and graph.num_nodes >= 2:
                        lit_graphs.append(graph)
                        lit_names.append(name)
                        lit_journals.append(journal)
                except Exception:
                    continue
            print(f"  Parsed {len(lit_graphs)} literature molecules for t-SNE")
        else:
            print(f"\nLiterature CSV not found: {literature_csv_path} (skipping literature molecules)")
    
    # Combine val + literature for embedding extraction
    n_val = len(val_graphs)
    all_graphs = val_graphs + lit_graphs
    
    # Calculate molecular weights (val only)
    molecular_weights = calculate_molecular_weights(val_molecules)
    
    # Extract embeddings for all (val + lit)
    embeddings, invalid_mask = extract_embeddings(model, all_graphs, device, batch_size=args.batch_size)
    
    # Filter invalid: split by val vs lit
    val_valid = ~invalid_mask[:n_val]
    lit_valid = ~invalid_mask[n_val:] if len(lit_graphs) > 0 else np.array([], dtype=bool)
    val_graphs = [val_graphs[i] for i in range(n_val) if val_valid[i]]
    val_molecules = [val_molecules[i] for i in range(n_val) if val_valid[i]]
    molecular_weights = molecular_weights[val_valid]
    val_embeddings = embeddings[:n_val][val_valid]
    if len(lit_graphs) > 0:
        lit_graphs = [lit_graphs[i] for i in range(len(lit_graphs)) if lit_valid[i]]
        lit_names = [lit_names[i] for i in range(len(lit_names)) if lit_valid[i]]
        lit_journals = [lit_journals[i] for i in range(len(lit_journals)) if lit_valid[i]]
        lit_embeddings = embeddings[n_val:][lit_valid]
    else:
        lit_embeddings = np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)
    
    if invalid_mask.any():
        print(f"\nFiltered out {invalid_mask.sum()} samples with invalid embeddings; remaining: {len(val_graphs)} val, {len(lit_graphs)} lit")
    
    # t-SNE on combined embeddings (val + lit)
    all_embeddings = np.concatenate([val_embeddings, lit_embeddings], axis=0)
    tsne_coords, scaler, tsne = compute_tsne(
        all_embeddings,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        learning_rate=args.learning_rate,
        initialization=args.initialization,
        metric=args.metric,
        random_state=config.seed
    )
    n_val_clean = len(val_embeddings)
    val_tsne = tsne_coords[:n_val_clean]
    lit_tsne = tsne_coords[n_val_clean:] if len(lit_embeddings) > 0 else np.zeros((0, 2))
    
    # Create visualizations
    static_path = os.path.join(output_dir, "tsne_static.png")
    interactive_path = os.path.join(output_dir, "tsne_interactive.html")
    
    title_suffix = f"{len(val_graphs)} molecules, colored by Molecular Weight"
    if len(lit_tsne) > 0:
        title_suffix += f"; {len(lit_tsne)} literature (unique markers)"
    
    create_static_tsne_plot(
        val_tsne,
        molecular_weights,
        static_path,
        title=f"t-SNE Visualization of Molecular Embeddings\n(Validation: {title_suffix})",
        lit_tsne=lit_tsne if len(lit_tsne) > 0 else None,
        lit_names=lit_names if len(lit_names) > 0 else None,
        lit_journals=lit_journals if len(lit_journals) > 0 else None,
    )
    
    create_interactive_tsne_plot(
        val_tsne,
        molecular_weights,
        val_embeddings,
        val_molecules,
        interactive_path,
        title=f"Interactive t-SNE Visualization of Molecular Embeddings<br><sub>Validation: {title_suffix}</sub>",
        show_images=args.images,
        lit_tsne=lit_tsne if len(lit_tsne) > 0 else None,
        lit_names=lit_names if len(lit_names) > 0 else None,
        lit_journals=lit_journals if len(lit_journals) > 0 else None,
        show_legend=len(lit_tsne) > 0 and not args.hide_literature_markers,
    )
    
    # Save t-SNE coordinates, embeddings, and molecular weights for later use (val + lit combined)
    np.save(os.path.join(output_dir, "tsne_coordinates.npy"), tsne_coords)
    np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
    np.save(os.path.join(output_dir, "molecular_weights.npy"), molecular_weights)
    print(f"\nSaved t-SNE coordinates, embeddings, and molecular weights to {output_dir}")
    
    print("\n" + "="*60)
    print("t-SNE visualization complete!")
    print(f"  Static plot: {static_path}")
    print(f"  Interactive plot: {interactive_path}")
    print("="*60)


if __name__ == "__main__":
    main()
