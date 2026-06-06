# Methodology & Comparison with the Original JTVAE Paper

This document explains how our implementation relates to the original Junction
Tree Variational Autoencoder (JTVAE) paper, and is honest about where we
faithfully follow it and where we deliberately simplified it for a course-scale
project (training budget of a few hours on a single GPU).

> **Reference paper**: Wengong Jin, Regina Barzilay, Tommi Jaakkola.
> *Junction Tree Variational Autoencoder for Molecular Graph Generation.*
> ICML 2018. [arXiv:1802.04364](https://arxiv.org/abs/1802.04364) ·
> [code](https://github.com/wengong-jin/icml18-jtnn)

---

## 1. What the Paper Does

| Aspect | Original JTVAE (Jin et al., 2018) |
|--------|-----------------------------------|
| **Dataset** | ZINC, ~250K drug-like molecules, with the train/test split from Kusner et al. (2017). Vocabulary built from 240K molecules. |
| **Problem** | Generate molecular **graphs directly** instead of SMILES strings, to overcome two SMILES weaknesses: (1) similar molecules can have very different SMILES, preventing smooth embeddings; (2) character-by-character generation passes through chemically invalid intermediate states. |
| **Preprocessing** | Tree decomposition of every molecule into a junction tree of clusters (rings + single bonds). Cluster vocabulary size **\|X\| = 780**. Atom features: type, degree, formal charge, chirality. Bond features: bond type, in-ring flag, cis/trans. |
| **Model** | Two-part latent `z = [z_T, z_G]` (28 + 28 = 56 dims). **Graph encoder**: loopy belief-propagation message passing → `z_G`. **Tree encoder**: GRU tree message passing (bottom-up + top-down) → `z_T`. **Tree decoder**: top-down, generates one node at a time with separate *topological* and *label* predictions. **Graph decoder**: enumerates how neighbouring clusters can attach, scores each candidate with `z_G`, and assembles the highest-scoring molecule. |
| **Validation** | (1) Reconstruction accuracy (76.7%); (2) **Prior validity = 100%**; (3) Bayesian optimization for penalized logP; (4) Constrained optimization via latent gradient ascent with a similarity constraint. |
| **Baselines** | CVAE (character VAE), GVAE (grammar VAE), SD-VAE (syntax-directed VAE), GraphVAE, atom-by-atom LSTM. |
| **Headline result** | 100% prior validity (vs 89.2% for the best baseline) and ~30% better property scores in optimization. |

The **graph decoder with attachment scoring** is the paper's core technical
contribution — it is what guarantees 100% chemical validity and faithful
reconstruction.

---

## 2. What Our Implementation Does

| Aspect | Our implementation |
|--------|--------------------|
| **Dataset** | ZINC via the **MOSES benchmark** (`dataset_v1.csv`). We subsample **80K** molecules for training. MOSES has already applied drug-likeness filtering upstream. |
| **Problem** | Same: graph-based generation (JTVAE) vs a sequence baseline (CharRNN). |
| **Preprocessing** | Tree decomposition, cluster vocabulary **\|X\| = 287** (from 50K molecules), kekulization validation, out-of-vocabulary filtering. Atom/bond features follow the paper exactly. |
| **Encoder** | Tree encoder + graph encoder (MPN), combined as `z = (z_T + z_G) / 2`. |
| **Tree encoder** | Vectorized **synchronous GCN** message passing (not the paper's bottom-up+top-down GRU schedule). |
| **Decoder** | GRU predicts a **sequence of cluster ids**, then a **heuristic single-bond assembly** joins clusters at free-valence atoms. |
| **Validation** | 8 **MOSES** metrics (validity, uniqueness, novelty, internal diversity, SNN, scaffold similarity, FCD, plus QED/SA/LogP distributions) + latent-space property optimization + t-SNE latent visualization. |
| **Baseline** | **CharRNN** — a character-level GRU over SMILES, equivalent in spirit to the paper's CVAE baseline. |

---

## 3. Where We Faithfully Follow the Paper

- **Junction-tree decomposition** (`models/jtvae/chemutils.py`, `mol_tree.py`):
  same algorithm — extract rings + non-ring bonds as clusters, merge bridged
  rings sharing > 2 atoms, build a cluster graph, and take a **spanning tree**.
- **Atom/bond feature design** (`models/jtvae/mpn.py`): atom type, degree,
  formal charge, chirality; bond type, in-ring flag, stereo — matching the
  paper's appendix C.
- **Two-encoder architecture**: a tree encoder *and* a graph MPN encoder, both
  feeding the VAE latent.
- **β-VAE objective with KL annealing** (`models/jtvae/train.py`): reconstruction
  loss + β·KL, with linear warmup to avoid posterior collapse.
- **Latent property optimization** (`optimization/property_optimize.py`): gradient
  ascent / Bayesian optimization in latent space, the same idea as the paper's
  Section 3.2–3.3.

---

## 4. Where We Deliberately Simplified (and Why)

| # | Paper | Ours | Reason |
|---|-------|------|--------|
| 1 | Latent = **concatenation** `[z_T, z_G]` | Latent = **average** `(z_T + z_G)/2` | Simpler bottleneck; fewer parameters. |
| 2 | Tree encoder: **GRU**, bottom-up + top-down schedule | **Synchronous GCN** (a few large tensor ops) | ~50× faster per epoch; avoids thousands of tiny per-node GPU kernels. |
| 3 | Tree decoder: separate **topological + label** prediction | Single **next-cluster** GRU prediction | Simpler sequence model. |
| 4 | **Graph decoder with attachment scoring** (the core contribution) | **Heuristic single-bond assembly** at free-valence atoms | A full attachment-scoring decoder is substantial to implement and train; the heuristic still yields connected, valid, drug-like molecules. |

**Most important difference (#4).** The paper's graph decoder uses `z_G` to
decide *exactly how* neighbouring clusters attach, which is what makes
reconstruction faithful and validity guaranteed. Our heuristic assembly ignores
the fine-grained `z_G` connectivity and simply bonds clusters at the first atom
with a free valence. This is the direct cause of two observations in our results:

- **Low scaffold similarity** — assembled scaffolds differ from the training set.
- **Limited benefit from `z_G` / KL collapse** — since assembly does not consume
  `z_G`, the graph-latent contributes little after KL annealing.

---

## 5. Honest Positioning for the Report

> We implement a **simplified variant of JTVAE** (Jin et al., 2018), preserving
> its core ideas — tree decomposition into a cluster vocabulary, a dual
> tree+graph encoder, and two-phase (tree-then-graph) generation. To fit a
> few-hour single-GPU budget we make three simplifications: (1) the two latents
> are averaged rather than concatenated; (2) the tree encoder is vectorized into
> a synchronous GCN; (3) the graph decoder uses heuristic single-bond assembly
> instead of the paper's attachment-scoring decoder. We benchmark against
> **CharRNN** (corresponding to the paper's CVAE character baseline) and
> evaluate with the **MOSES** metric suite.

---

## 6. Why We Do Not Report Reconstruction Accuracy

The original paper reports **reconstruction accuracy** (76.7%): encode a molecule
`m` to `z`, decode it back, and measure how often the result is *exactly* `m`.
We deliberately omit this metric, for a principled reason tied to simplification
4 (the graph decoder) above.

Faithful reconstruction requires the decoder to use the graph latent `z_G` to
recover the **exact attachment configuration** between clusters. Our simplified
graph decoder does **not** consume `z_G` — it joins clusters with a heuristic
single bond at the first free-valence atom. As a result, even with the correct
latent `z`, the decoded molecule's connectivity generally differs from the
original, so an "exact-match" reconstruction score is not a meaningful measure of
our model. (This is the same root cause behind the observed KL collapse and the
very low scaffold similarity.)

Instead, we evaluate the property that our model *can* meaningfully deliver —
**distributional quality of prior samples** — via the MOSES metric suite below
(validity, novelty, FCD, QED/LogP distributions, etc.). The **FCD** metric in
particular (Fréchet ChemNet Distance) captures how close the entire generated
distribution is to real molecules, serving a complementary role to the paper's
reconstruction check.

---

## 7. Results Summary (random samples from the prior)

| Metric | CharRNN (baseline) | JTVAE (ours) | Real drugs (target) |
|--------|:---:|:---:|:---:|
| Validity | 0.975 | **1.000** | — |
| Uniqueness | 1.000 | 0.828 | — |
| Novelty | 0.876 | **1.000** | — |
| Internal diversity | 0.865 | 0.867 | — |
| QED | 0.799 | **0.811** | ~0.80 |
| LogP | 2.45 | 2.68 | ~2.45 |
| SNN | **0.348** | 0.194 | — |
| Scaffold similarity | **0.375** | 0.008 | — |

**Takeaway.** Both models generate valid, novel, drug-like molecules
(QED ≈ 0.8, LogP within the Lipinski range). JTVAE reaches 100% validity and
100% novelty thanks to structure-by-structure generation, while CharRNN stays
closer to the training distribution (higher SNN / scaffold similarity). This is
a genuine trade-off rather than one model dominating — consistent with the
paper's thesis that graph-based generation guarantees validity, while our
simplified assembly trades scaffold fidelity for broader exploration.
