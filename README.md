# Lewis Base Binding Energy Prediction (LBPP)

A deep learning framework for predicting binding energies of Lewis base molecules on perovskite surfaces.

## Overview

This project uses contrastive self-supervised learning to pre-train a graph neural network encoder on large-scale molecular data, then fine-tunes it for binding energy prediction.

## Installation

```bash
# Clone repository
git clone <repository-url>
cd LBPP

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+, PyTorch Geometric, RDKit

## Quick Start

### Predict Binding Energy

```bash
# Single molecule
python inference.py --smiles "CCO" --donor_type "hydroxyl"

# Batch prediction
python inference.py --csv input.csv --output predictions.csv
```

### Train SSL Model

```bash
# 1. Build graph cache from molecular CSV
python dataset/ssl/build_graph_cache.py --csv_file molecules.csv --cache_dir ./cache

# 2. Train SSL encoder
python train_ssl.py
```

## Project Structure

```
LBPP/
├── config.py                 # Configuration parameters
├── train_ssl.py              # SSL training script
├── inference.py              # Binding energy prediction
├── models/
│   └── gin_e.py              # GIN-E encoder model
├── dataset/
│   ├── ssl/                  # SSL data processing
│   │   ├── build_graph_cache.py
│   │   ├── molecular_graph.py
│   │   └── augmentation.py
│   ├── prediction/           # Downstream prediction data
│   │   ├── sampling_Eb.py
│   │   └── funct_group.csv
│   └── literature/           # Literature extraction
│       └── abs_extract.py
└── checkpoints/              # Saved models
```

## Data Format

**SSL Training**: CSV file with `PUBCHEM_COMPOUND_CID` and `SMILES` columns

**Binding Energy Prediction**: CSV file with columns:
- `CID`: Compound ID
- `SMILES`: Molecular structure
- `DonorType`: Functional group type
- `mlp_adsorption_energy`: Binding energy (eV)

## License

MIT License

## Acknowledgments

PyTorch Geometric, RDKit, PubChem
