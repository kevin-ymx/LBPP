# Contrastive Self-Supervised Learning of Charge-Aware Molecular Representation

This project implements a contrastive self-supervised learning framework for learning molecular representations using GIN-E (Graph Isomorphism Network with Edge features) encoder.

## Features

### 1. Molecular Graph Construction
- Extracts molecules from SDF files
- Constructs molecular graphs with rich node and edge features

**Node Features:**
- Atomic number
- Atom chirality
- Partial charges
- Hybridization
- Coordination number
- Valence electrons
- Electronegativity

**Edge Features:**
- Bond type
- Bond direction
- Coulombic term (charge-based interaction)

### 2. Graph Augmentation
- Subgraph removal with fixed ratio (25% by default)
- Generates positive pairs for contrastive learning

### 3. GIN-E Encoder
- Multi-layer Graph Isomorphism Network with edge features
- Node and edge feature encoders
- Graph-level pooling (mean or sum)

### 4. Contrastive Learning
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
- Training on batched molecular graph pairs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Edit `config.py` to set:
- SDF file path
- Model hyperparameters
- Training parameters
- Data split ratios

### Training

```bash
python train.py
```

The training script will:
1. Load molecules from the SDF file
2. Convert molecules to graphs with all features
3. Split into training and validation sets
4. Apply subgraph removal augmentation
5. Train GIN-E encoder using NT-Xent loss
6. Save checkpoints and best model

## Model Architecture

The GIN-E encoder consists of:
1. **Node Feature Encoder**: Projects raw node features to embeddings
2. **Edge Feature Encoder**: Projects raw edge features to embeddings
3. **GIN-E Layers**: Multiple layers of Graph Isomorphism Network with edge features
4. **Graph Pooling**: Mean or sum pooling for graph-level representation
5. **Final Projection**: Linear projection to final embedding space

## Citation

If you use this code, please cite the relevant papers on GIN, GIN-E, and contrastive learning for molecular representation.
