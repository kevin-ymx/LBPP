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
from typing import List, Optional

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
        perplexity=perplexity,
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


def create_static_tsne_plot(tsne_coords: np.ndarray, molecular_weights: np.ndarray, output_path: str, title: str = "t-SNE Visualization"):
    """Create static matplotlib t-SNE plot colored by molecular weight."""
    print(f"\nCreating static t-SNE plot (colored by molecular weight)...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=molecular_weights,
        cmap='plasma',
        alpha=0.7,
        s=3,
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Molecular Weight (Da)', fontsize=12, rotation=270, labelpad=20)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved static plot to {output_path}")
    plt.close()


def create_interactive_tsne_plot(tsne_coords: np.ndarray, molecular_weights: np.ndarray, embeddings: np.ndarray, molecules: list, output_path: str, title: str = "Interactive t-SNE Visualization", show_images: bool = False):
    """Create interactive plotly t-SNE plot colored by molecular weight. If show_images=True, hover shows structure images."""
    print(f"\nCreating interactive t-SNE plot (colored by molecular weight)...")
    
    if show_images:
        # Generate molecular structure images and save them next to the HTML output
        output_dir = os.path.dirname(os.path.abspath(output_path))
        output_basename = os.path.splitext(os.path.basename(output_path))[0]
        images_dir = os.path.join(output_dir, f"{output_basename}_images")
        os.makedirs(images_dir, exist_ok=True)
        print(f"  Saving molecular structure images to: {images_dir}")
        
        image_paths = []
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
            if image_paths[i] is not None:
                hover_html = f'<img src="{image_paths[i]}" style="max-width:300px; max-height:300px; border:2px solid #666; border-radius:5px;">'
            else:
                hover_html = ""
            hover_texts.append(hover_html)
    else:
        hover_texts = [f"Index: {i}<br>Mol. weight: {mw:.1f} Da" for i, mw in enumerate(molecular_weights)]
    
    # Create scatter plot with color mapping
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tsne_coords[:, 0],
        y=tsne_coords[:, 1],
        mode='markers',
        marker=dict(
            size=2,
            color=molecular_weights,
            colorscale='Plasma',
            colorbar=dict(
                title=dict(
                    text="Molecular Weight (Da)",
                    font=dict(size=12)
                )
            ),
            opacity=0.7,
            line=dict(width=0),
            showscale=True
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Molecules'
    ))
    
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
            gridcolor='lightgray'
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
        width=1000,
        height=800,
        hovermode='closest'
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
    
    # Calculate molecular weights
    molecular_weights = calculate_molecular_weights(val_molecules)
    
    # Extract embeddings
    embeddings, invalid_mask = extract_embeddings(model, val_graphs, device, batch_size=args.batch_size)
    
    # Filter out invalid samples (NaN/Inf embeddings)
    if invalid_mask.any():
        print(f"\nFiltering out {invalid_mask.sum()} samples with invalid embeddings...")
        valid_mask = ~invalid_mask
        val_graphs = [val_graphs[i] for i in range(len(val_graphs)) if valid_mask[i]]
        val_molecules = [val_molecules[i] for i in range(len(val_molecules)) if valid_mask[i]]
        molecular_weights = molecular_weights[valid_mask]
        embeddings = embeddings[valid_mask]
        print(f"  Remaining valid samples: {len(val_graphs)}")
    
    # Compute t-SNE
    tsne_coords, scaler, tsne = compute_tsne(
        embeddings,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        learning_rate=args.learning_rate,
        initialization=args.initialization,
        metric=args.metric,
        random_state=config.seed
    )
    
    # Create visualizations
    static_path = os.path.join(output_dir, "tsne_static.png")
    interactive_path = os.path.join(output_dir, "tsne_interactive.html")
    
    create_static_tsne_plot(
        tsne_coords,
        molecular_weights,
        static_path,
        title=f"t-SNE Visualization of Molecular Embeddings\n(Validation Set: {len(val_graphs)} molecules, colored by Molecular Weight)"
    )
    
    create_interactive_tsne_plot(
        tsne_coords,
        molecular_weights,
        embeddings,
        val_molecules,
        interactive_path,
        title=f"Interactive t-SNE Visualization of Molecular Embeddings<br><sub>Validation Set: {len(val_graphs)} molecules, colored by Molecular Weight</sub>",
        show_images=args.images
    )
    
    # Save t-SNE coordinates, embeddings, and molecular weights for later use
    np.save(os.path.join(output_dir, "tsne_coordinates.npy"), tsne_coords)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "molecular_weights.npy"), molecular_weights)
    print(f"\nSaved t-SNE coordinates, embeddings, and molecular weights to {output_dir}")
    
    print("\n" + "="*60)
    print("t-SNE visualization complete!")
    print(f"  Static plot: {static_path}")
    print(f"  Interactive plot: {interactive_path}")
    print("="*60)


if __name__ == "__main__":
    main()
