# Lewis Base Molecular Representation Learning and Binding Energy Prediction

A deep learning framework for learning molecular representations of Lewis base molecules using contrastive self-supervised learning (SSL) and predicting their binding energies on perovskite surfaces.

## Overview

This project consists of two main components:

1. **Contrastive Self-Supervised Learning (SSL)**: Pre-trains a GIN-E (Graph Isomorphism Network with Edge features) encoder to learn general molecular representations from large-scale unlabeled molecular data using contrastive learning with subgraph augmentation.

2. **Downstream Prediction Pipeline**: Fine-tunes the pretrained encoder for binding energy prediction of Lewis base molecules on perovskite surfaces, incorporating binding site information through functional group recognition.

## Features

- **Graph Neural Network Architecture**: GIN-E encoder with edge features for molecular graph representation
- **Contrastive SSL**: NT-Xent loss with subgraph removal augmentation for robust molecular embeddings
- **Binding Site Awareness**: Functional group recognition and binding atom tagging for Lewis base molecules
- **Charge-Aware Features**: Incorporates partial charges, electronegativity, and coulombic interactions
- **Flexible Inference**: Scripts for binding energy prediction

## Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kevin-ymx/LBPP.git
cd LBPP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch Geometric (if not already installed):
```bash
pip install torch-geometric
```

## Usage

Predict binding energy for a molecule given SMILES and donor type:

```bash
python inference.py --smiles "CCO" --donor_type "hydroxyl"
```

**Batch prediction from CSV**:
```bash
python inference.py --csv input.csv --output predictions.csv
```

## Model Architecture

### GIN-E Encoder

The GIN-E encoder processes molecular graphs with the following features:

**Node Features** (8 dimensions):
- Atomic number
- Chirality
- Partial charge (Gasteiger)
- Hybridization
- Coordination number
- Valence electrons
- Electronegativity
- Binding tag (1 for binding atoms, 0 otherwise)

**Edge Features** (3 dimensions):
- Bond type
- Bond direction
- Coulombic term (charge interaction)

**Architecture**:
- Node/Edge feature encoders (Linear + ReLU + LayerNorm)
- 5-layer GIN-E with batch normalization
- Graph-level mean pooling
- Final projection layer
- Output: 256-dimensional molecular embedding

### Downstream Model

The downstream model consists of:
1. **Pretrained GIN-E Encoder**: Frozen or fine-tuned
2. **Combining MLP**: 2-layer MLP (512 hidden dim) to refine embeddings
3. **Prediction Head**: 2-layer MLP (256 hidden dim) for binding energy regression

## Supported Functional Groups

The model recognizes the following Lewis base functional groups:

- `alkoxide_O`: Negatively charged oxygen
- `amide`: Amide (binding on O)
- `amide_carbonyl_O`: Amide carbonyl (binding on O)
- `amine_primary`: Primary amine nitrogen
- `amine_secondary`: Secondary amine nitrogen
- `amine_tertiary`: Tertiary amine nitrogen
- `aromatic_N_pyridinic`: Pyridinic nitrogen in aromatic ring
- `aromatic_N_pyrrolic`: Pyrrolic nitrogen
- `carbonyl_O`: Carbonyl oxygen
- `cooh_like`: Carboxylic acid-like (binding on first O)
- `ether_O`: Ether oxygen
- `hydroxyl`: Hydroxyl oxygen
- `imine`: Imine nitrogen
- `nitrile_CN`: Nitrile nitrogen
- `phenoxide_O`: Phenoxide oxygen
- `phosphine`: Phosphine phosphorus
- `p_oxide`: Phosphine oxide (binding on O)
- `sox_like`: Sulfonate-like (binding on first O)
- `sulfoxide`: Sulfoxide (binding on O)
- `thiocarbonyl`: Thiocarbonyl sulfur
- `thioether_S`: Thioether sulfur
- `thiol`: Thiol sulfur

## Data Format

### SSL Training Data
- **Format**: SDF file (`.sdf` or `.sdf.gz`)
- **Content**: 3D molecular structures with atomic coordinates
- **Source**: Large-scale molecular databases (e.g., PubChem, ZINC)

### Downstream Training Data
- **Format**: CSV file
- **Required columns**:
  - `CID`: PubChem compound ID (integer)
  - `SMILES`: SMILES string
  - `DonorType`: Functional group type (string)
  - `mlp_adsorption_energy`: Binding energy in eV (float)

**Example**:
```csv
CID,SMILES,DonorType,mlp_adsorption_energy
702,CC(=O)O,carbonyl_O,-0.45
280,CCN,amine_primary,-0.38
```

## License

[Specify your license here]

## Acknowledgments

- PyTorch Geometric for graph neural network utilities
- RDKit for molecular processing
- PubChem for molecular data

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
