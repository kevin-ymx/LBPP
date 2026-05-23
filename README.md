# Lewis Base Binding Energy Prediction (LBPP)

A deep learning framework for predicting binding energies of Lewis base molecules on perovskite surfaces.

## Overview

This project uses contrastive self-supervised learning to pre-train a graph neural network encoder on large-scale molecular data, then fine-tunes it for binding energy prediction.

## Installation

```bash
git clone <repository-url>
cd LBPP
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+, PyTorch Geometric, RDKit. For anchor functional-group analysis, SciPy is recommended (Hungarian assignment for DFT-to-SMILES atom mapping when the anchor element appears more than once).

## Quick start

### Predict binding energy

```bash
# Single molecule
python inference.py --smiles "CCO" --donor_type "hydroxyl"

# Batch prediction
python inference.py --csv input.csv --output predictions.csv
```

### Train SSL model

```bash
# 1. Build graph cache from molecular CSV
python dataset/ssl/build_graph_cache.py --csv_file molecules.csv --cache_dir ./cache

# 2. Train SSL encoder
python train_ssl.py
```

### Binding-anchor statistics (violin plots)

`analyze_binding_anchors.py` reads a merged downstream CSV and writes violin plots plus per-group CSV summaries under `logs/binding_anchor_stats` by default (or `--output_dir`).

**Plot 1 — anchor combinations (N / O / S / P only)**  
Groups rows by the *set* of anchoring elements in {N, O, S, P} (e.g. `N`, `N+O`), using `pb_bond_encoding` and atomic numbers from `adsorbate_structure`.

**Plots 2–5 — one plot per element N, O, S, P**  
Only rows with **exactly one** anchoring atom in {N, O, S, P}. Additional anchors on other elements (e.g. H, Cl, C) are allowed and ignored. Rows with two or more anchors in {N, O, S, P} (e.g. N+O) are excluded from these plots.

Functional groups for plots 2–5 are **detected with RDKit** at the resolved anchor atom (SMILES + optional geometric mapping from DFT coordinates), **not** from the CSV `functional_group` column.

```bash
python analyze_binding_anchors.py --input path/to/merged.csv --output_dir logs/binding_anchor_stats
python analyze_binding_anchors.py --min_group_size 5
python analyze_binding_anchors.py --energy_min -3.0 --energy_max 0
```

| Flag | Meaning |
|------|---------|
| `--input` | Merged CSV path (defaults to `config.downstream_csv` if set) |
| `--output_dir` | Where PNGs and stats CSVs are written |
| `--min_group_size` | Drop groups with fewer than this many rows (default: 3) |
| `--energy_min` / `--energy_max` | Optional adsorption energy window in eV (inclusive) |

**Expected columns** (names resolved case-insensitively): `cid`, `SMILES`, `pb_bond_encoding`, `adsorption_energy`, `adsorbate_structure` (JSON with `elements.number` and ideally `coords.3d`). `functional_group` is optional and not used for plots 2–5.

**Outputs**: `violin_anchor_combinations.png`, `stats_anchor_combinations.csv`, and for each of N, O, S, P: `violin_single_anchor_<E>_by_functional_group.png` and `stats_single_anchor_<E>.csv`.

## Project structure

```
LBPP/
├── config.py                 # Configuration parameters
├── train_ssl.py              # SSL training script
├── train_downstream.py       # Downstream binding prediction training
├── inference.py              # Binding energy prediction
├── analyze_binding_anchors.py  # Anchor / functional-group violin statistics
├── visualize_tsne.py         # t-SNE of embeddings (literature overlays)
├── visualize_tsne_downstream.py
├── models/
│   └── gin_e.py              # GIN-E encoder model
├── comparison_2feat/         # Two-feature comparison experiments
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

## Data formats

**SSL training**: CSV with `PUBCHEM_COMPOUND_CID` and `SMILES` columns.

**Binding energy prediction (inference-style CSV)**: e.g. `CID`, `SMILES`, `DonorType`, `mlp_adsorption_energy` (eV).

**Merged downstream CSV** (for `analyze_binding_anchors.py` and training): includes `cid`, `SMILES`, `pb_bond_encoding`, `adsorption_energy`, `adsorbate_structure`, and related fields as produced by your merge pipeline.

## License

MIT License

## Acknowledgments

PyTorch Geometric, RDKit, PubChem
