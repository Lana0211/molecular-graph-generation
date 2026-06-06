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
| ------ | --------------------------------- |
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
| ------ | ------------------ |
| **Dataset** | ZINC via the **MOSES benchmark** (`dataset_v1.csv`). We subsample **80K** molecules for training. MOSES has already applied drug-likeness filtering upstream. |
| **Problem** | Same: graph-based generation (JTVAE) vs a sequence baseline (CharRNN). |
| **Preprocessing** | Tree decomposition, cluster vocabulary **\|X\| = 287** (from 50K molecules), kekulization validation, out-of-vocabulary filtering. Atom/bond features follow the paper exactly. |
| **Encoder** | Tree encoder + graph encoder (MPN), combined as `z = (z_T + z_G) / 2`. |
| **Tree encoder** | Vectorized **synchronous GCN** message passing (not the paper's bottom-up+top-down GRU schedule). |
| **Decoder** | GRU predicts a **sequence of cluster ids**, then `enum_assemble` (ported from Jin et al.'s original code) joins clusters by enumerating valid atom-share and ring bond-fusion attachments, selecting randomly among valid candidates. |
| **Validation** | 8 **MOSES** metrics (validity, uniqueness, novelty, internal diversity, SNN, scaffold similarity, FCD, plus QED/SA/LogP distributions) + assembler ablation (single-bond vs enum_assemble, same checkpoint) + latent-space property optimization (gradient ascent, QED proxy) + t-SNE latent visualization. |
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
| - | ----- | ---- | ------ |
| 1 | Latent = **concatenation** `[z_T, z_G]` | Latent = **average** `(z_T + z_G)/2` | Simpler bottleneck; fewer parameters. |
| 2 | Tree encoder: **GRU**, bottom-up + top-down schedule | **Synchronous GCN** (a few large tensor ops) | ~50× faster per epoch; avoids thousands of tiny per-node GPU kernels. |
| 3 | Tree decoder: separate **topological + label** prediction | Single **next-cluster** GRU prediction | Simpler sequence model. |
| 4 | **Graph decoder with attachment scoring** using `z_G` to score candidates | `enum_assemble` enumerates valid attachments (atom-share + ring bond-fusion, ported from original code) but **selects randomly** among valid candidates — no learned `z_G`-conditioned scoring | Porting the enumeration is already substantial; training the scorer would require a separate graph-matching network. |

**Most important remaining difference (#4).** The paper's graph decoder uses `z_G` to
*score* each candidate attachment and pick the best one, which is what makes
reconstruction faithful. Our implementation enumerates the same candidates using
the original `enum_assemble` logic (enabling ring fusion), but selects randomly
rather than by score. This means `z_G` still has limited influence on the final
molecule — which explains two observations in our results:

- **Low scaffold similarity for enum_assemble** — random selection among ring fusions creates
  structurally novel scaffolds not seen in ZINC, pushing scaffold similarity to 0.03.
- **FCD still high vs CharRNN** — the molecular property distribution doesn't perfectly
  match ZINC because assembly is unguided.

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

## 7. Results Summary

### 7a. Generation quality (MOSES metrics)

| Metric | CharRNN | JTVAE — single-bond | JTVAE — enum_assemble |
| ------ | ------- | ------------------- | --------------------- |
| Validity | 0.9795 | **1.0000** | **1.0000** |
| Uniqueness | 0.9996 | **1.0000** | **1.0000** |
| Novelty | 0.8433 | **1.0000** | **1.0000** |
| Internal diversity | 0.8664 | 0.8650 | **0.9016** |
| FCD (↓) | **0.182** | 33.25 | 20.27 |
| QED | **0.802** | 0.467 | 0.587 |
| SNN | **0.346** | 0.168 | 0.200 |
| Scaffold similarity | 0.391 | **0.484** | 0.034 |

The assembler ablation (columns 2 vs 3, same checkpoint) shows enum_assemble reduces FCD by 39% and raises internal diversity, confirming that ring fusion improves distributional quality. The low scaffold similarity for enum_assemble reflects novel ring systems not present in ZINC — a side effect of random candidate selection without `z_G` scoring.

### 7b. Property optimization (gradient ascent in latent space)

Starting from 200 encoded test molecules, gradient ascent on a QED proxy MLP was run for 100 steps (lr=0.02). Results at the best step (step 60):

- Mean ΔQED = **+0.076** (0.532 → 0.608)
- Hit rate (ΔQED > 0.05) = **53%**
- Best QED achieved = **0.943**
- Validity maintained at ~100% throughout optimization

This confirms the latent space is structured and meaningful (not collapsed): moving z in the direction of higher predicted QED reliably decodes into higher-QED molecules for the majority of starting points.

**Overall takeaway.** JTVAE reaches 100% validity and novelty by construction. CharRNN better matches the ZINC distribution (lower FCD, higher QED). This is a genuine trade-off: graph-based generation guarantees structural validity while our unscored assembly explores broader scaffold space.
