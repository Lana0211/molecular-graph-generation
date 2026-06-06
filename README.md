# CE7076 Final Project — Molecular Graph Generation for Drug Discovery

**Course**: CE7076 Generative AI & Foundation Models in Bioinformatics  
**Track**: Graph Data (Biological Networks) — Generative Direction  

---

## Reproducing the Results

Follow these steps on a machine with an NVIDIA GPU (CUDA 12.1) and Python 3.12.

```powershell
# 1. Create and activate a virtual environment
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install PyTorch (CUDA build) first, then the rest
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Register the Jupyter kernel
python -m ipykernel install --user --name ce7076_venv --display-name "CE7076 venv (Python 3.12)"

# 4. Launch the notebook and select the "CE7076 venv" kernel
jupyter notebook notebooks/molecular_generation.ipynb
```

Then **Run All** cells in [notebooks/molecular_generation.ipynb](notebooks/molecular_generation.ipynb).
The notebook downloads the dataset automatically and runs all five sections
(data exploration → CharRNN → JTVAE → property optimization → evaluation).

**Smart caching** — every training/generation cell checks for an existing
checkpoint or CSV and *loads it instead of recomputing*. So a fresh clone trains
both models once (~1 hr for JTVAE on GPU); re-running afterwards is near-instant.

### Re-training from scratch

To force a model to re-train (or re-generate) instead of loading the cached
result, delete the relevant file(s) under `results/` before running the cell:

| To force… | Delete |
|-----------|--------|
| CharRNN re-training | `results/char_rnn/char_rnn_best.pt` |
| CharRNN re-generation | `results/char_rnn/char_rnn_generated.csv` |
| JTVAE re-training | `results/jtvae/jtvae_best.pt` |
| JTVAE re-generation | `results/jtvae/jtvae_generated.csv` |
| Rebuild JT vocabulary | `data/vocab.txt` |
| Full clean re-run | the entire `results/` folder (and `data/vocab.txt`) |

> Note: the `jtvae_epoch*.pt` and `*_history.json` files are auxiliary
> (periodic checkpoints / loss logs) and do **not** affect the load-vs-train
> decision — only `jtvae_best.pt` / `char_rnn_best.pt` do. After deleting a
> checkpoint, **restart the Jupyter kernel** before re-running so the updated
> model code is reloaded.

---

## Research Question

> Can a **graph-based generative model (JTVAE)** produce more valid, unique, and drug-like novel molecules compared to a **sequence-based baseline (CharRNN)**, when trained on the same ZINC250k benchmark dataset?

---

## Method Overview

| Model | Type | Architecture |
|-------|------|-------------|
| CharRNN | Sequence-based | 3-layer GRU, character-level SMILES |
| **JTVAE** | Graph-based | Junction Tree VAE (encoder: Tree-LSTM + MPN; decoder: auto-regressive + enum_assemble) |

**Key idea**: JTVAE decomposes each molecule into a *junction tree* of ring systems and bonds, then learns to encode/decode this structured representation. The decoder generates a sequence of cluster IDs, which are then assembled into a full molecule using `enum_assemble` — a ring-fusion algorithm that enumerates all chemically valid cluster attachments (including shared bonds between rings).

**Property optimization**: After training, we run gradient ascent in the JTVAE latent space to maximise QED (drug-likeness). A small proxy MLP is trained on encoded seed molecules, then gradient steps push latent vectors toward higher predicted QED. On 200 seed molecules: mean ΔQED = +0.076, hit rate = 53%.

> **How this relates to the original JTVAE paper** (faithful parts vs. our
> simplifications, and an honest results discussion) is documented in
> [METHODOLOGY.md](METHODOLOGY.md).

---

## Dataset

**ZINC** via the [MOSES benchmark](https://github.com/molecularsets/moses):

- Drug-like molecules already filtered upstream by MOSES (MW, atom types, ring sizes)
- We subsample **80K** molecules for training; standard train / test split provided
- JTVAE cluster vocabulary: **287** motifs (built from 50K molecules)

---

## Evaluation Metrics (MOSES standard)

| Metric | Description |
|--------|-------------|
| Validity | % parseable by RDKit |
| Uniqueness | % unique among valid |
| Novelty | % not in training set |
| FCD | Fréchet ChemNet Distance (lower = better) |
| SNN | Mean max Tanimoto to test set |
| Internal Diversity | Mean pairwise Tanimoto dissimilarity |
| QED | Drug-likeness (0–1) |
| SA | Synthetic accessibility (1–10) |
| LogP | Lipophilicity |

### Key Results

| Metric | CharRNN | JTVAE (single-bond) | JTVAE (enum_assemble) |
| ------ | ------- | ------------------- | --------------------- |
| Validity | 0.9795 | 1.0000 | **1.0000** |
| Uniqueness | 0.9996 | 1.0000 | **1.0000** |
| Novelty | 0.8433 | 1.0000 | **1.0000** |
| Internal Diversity | 0.8664 | 0.8650 | **0.9016** |
| FCD (↓) | **0.182** | 33.25 | 20.27 |
| QED | **0.802** | 0.467 | 0.587 |

JTVAE-SB = single-bond assembler (ablation baseline); JTVAE (enum_assemble) is our main model. The ablation confirms ring-fusion assembly reduces FCD by 39% vs single-bond on the same checkpoint.

---

## Project Structure

```
final_project/
├── data/
│   ├── download.py          # Downloads MOSES / ZINC250k dataset
│   ├── train.csv            # (auto-generated)
│   └── test.csv             # (auto-generated)
├── models/
│   ├── char_rnn/
│   │   ├── model.py         # CharRNN model + Vocabulary + Dataset
│   │   └── train.py         # Training script (CLI)
│   └── jtvae/
│       ├── chemutils.py     # RDKit chemistry helpers
│       ├── mol_tree.py      # Junction tree decomposition + Vocab
│       ├── mpn.py           # Message Passing Network (graph encoder)
│       ├── jtnn_vae.py      # JTVAE model
│       └── train.py         # Training script (CLI)
├── evaluation/
│   └── metrics.py           # All MOSES-compatible metrics
├── optimization/
│   └── property_optimize.py # Latent space optimization (BO / gradient)
├── notebooks/
│   └── molecular_generation.ipynb   # all-in-one notebook (sections 1–5)
├── results/                 # Generated outputs (checkpoints, CSVs, figures)
├── venv/                    # Python 3.12 virtual environment
└── requirements.txt
```

---

## Quickstart

### 1. Activate virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Register Jupyter kernel
```powershell
python -m ipykernel install --user --name venv --display-name "venv"
```

### 3. Run the all-in-one notebook
```powershell
jupyter notebook notebooks/molecular_generation.ipynb
```

### Or use CLI scripts directly

**CharRNN**:
```powershell
python -m models.char_rnn.train --data_path data/train.csv --save_dir results/char_rnn --epochs 30
```

**JTVAE**:
```powershell
python -m models.jtvae.train --data_path data/train.csv --vocab_path data/vocab.txt --save_dir results/jtvae --epochs 30 --subset 80000
```

**Property optimization** (after JTVAE training):
```powershell
python -m optimization.property_optimize --ckpt results/jtvae/jtvae_best.pt --method bo --n_iter 50
```

---

## Expected Training Time (GPU)

| Step | Time |
|------|------|
| Data download & preprocessing | ~10 min |
| Vocab building (50k molecules) | ~5 min |
| CharRNN training (30 epochs) | ~30 min |
| JTVAE training (30 epochs, 80k samples, free_bits) | ~1 hr |
| Property optimization (50 BO iterations) | ~15 min |

---

## References

1. Jin et al., "Junction Tree Variational Autoencoder for Molecular Graph Generation", ICML 2018
2. Segler et al., "Generating Focused Molecule Libraries for Drug Discovery with RNNs", ACS Cent. Sci. 2018
3. Polykovskiy et al., "Molecular Sets (MOSES): A Benchmarking Platform", Front. Pharmacol. 2020
4. Griffiths & Hernández-Lobato, "Constrained Bayesian Optimization for Automatic Chemical Design", ACS Cent. Sci. 2020
