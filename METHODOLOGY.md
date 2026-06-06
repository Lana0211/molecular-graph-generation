# Methodology & Comparison with the Original JTVAE Paper

How our implementation relates to Jin et al. (ICML 2018), what we kept, and where we simplified for a single-GPU course project.

> **Reference**: Wengong Jin, Regina Barzilay, Tommi Jaakkola.
> *Junction Tree Variational Autoencoder for Molecular Graph Generation.* ICML 2018.
> [arXiv:1802.04364](https://arxiv.org/abs/1802.04364) · [code](https://github.com/wengong-jin/icml18-jtnn)

---

## 1. Paper vs. Our Implementation

| Aspect | Original (Jin et al., 2018) | Ours |
| ------ | --------------------------- | ---- |
| Dataset | ZINC ~250K, vocab 780 clusters | ZINC via MOSES, 80K subset, vocab 287 clusters |
| Encoder | Tree-GRU (bottom-up + top-down) + graph MPN → `z = [z_T, z_G]` (concat) | Synchronous GCN + graph MPN → `z = (z_T + z_G)/2` (average) |
| Tree decoder | Separate topological + label prediction | Single next-cluster GRU |
| Graph decoder | Enumerates cluster attachments, scores with `z_G`, picks best | `enum_assemble` enumerates same attachments (ported from original), selects randomly — no learned scoring |
| Evaluation | Reconstruction accuracy, property optimization (penalized logP) | MOSES 8-metric suite, assembler ablation, QED gradient ascent, t-SNE |
| Baseline | CVAE, GVAE, GraphVAE, atom-by-atom LSTM | CharRNN (character-level GRU, equivalent to paper's CVAE) |

---

## 2. What We Kept

- **Junction-tree decomposition**: same algorithm — extract rings + non-ring bonds as clusters, build a spanning tree. Atom/bond features match the paper's appendix C.
- **Dual encoder**: tree encoder + graph MPN encoder, both feeding the VAE latent.
- **β-VAE with KL annealing**: reconstruction loss + β·KL, linear warmup. We add `free_bits=0.5` to prevent posterior collapse (Kingma et al., 2016).
- **`enum_assemble` graph assembly**: ported from the original code — enumerates atom-share and ring bond-fusion attachments, enabling proper heterocyclic ring systems.
- **Latent property optimization**: gradient ascent / Bayesian optimization in latent space, following the paper's Section 3.2–3.3.

---

## 3. Where We Simplified (and Why)

| # | Paper | Ours | Reason |
| - | ----- | ---- | ------ |
| 1 | Latent = concatenation `[z_T, z_G]` | Latent = average `(z_T + z_G)/2` | Simpler bottleneck; fewer parameters. |
| 2 | Tree encoder: GRU, bottom-up + top-down | Synchronous GCN | ~50× faster; avoids per-node GPU kernel overhead. |
| 3 | Tree decoder: topological + label heads | Single next-cluster GRU | Simpler sequence model. |
| 4 | Graph decoder: scores candidates with `z_G` | Selects randomly among `enum_assemble` candidates | Training the scorer requires a separate graph-matching network. |

**Effect of simplification #4.** Without learned scoring, `z_G` has limited influence on the final molecule. This explains two results: (a) scaffold similarity is low — random ring-fusion selection creates novel scaffolds not in ZINC; (b) FCD remains higher than CharRNN — assembly is unguided by the graph latent.

---

## 4. Training Details

| Setting | Value |
| ------- | ----- |
| Training data | 80K ZINC molecules (MOSES split) |
| Vocab | 287 clusters, built from 50K molecules |
| Epochs | 30, batch size 32 |
| Optimizer | Adam, lr=1e-3, StepLR ×0.9 every 10 epochs |
| KL schedule | Linear warmup (β=0 for first 5 epochs → β=1.0 at epoch 30) |
| KL free bits | 0.5 nats/dim (floors total KL at ~28 nats, prevents collapse) |
| Best checkpoint | Selected by reconstruction loss after warmup (epoch 30, rec=0.689) |
| Generation | `enum_assemble`, max 12 clusters, temperature 1.2; drug-likeness filter MW≤420, logP≤5.5, QED≥0.4 |

---

## 5. Why We Don't Report Reconstruction Accuracy

The paper measures how often encode→decode returns the *exact* input molecule (76.7%). This requires `z_G` to select the correct cluster attachment — something our random assembly doesn't do. We use **FCD** (Fréchet ChemNet Distance) as the main distributional quality metric instead.

---

## 6. Results

### Generation quality (MOSES metrics)

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

Assembler ablation (same checkpoint, columns 2 vs 3): enum_assemble cuts FCD by 39% and raises internal diversity, confirming ring fusion helps. Low scaffold similarity reflects novel ring systems from random attachment selection.

### Property optimization

200 test molecules were encoded, then gradient ascent on a QED proxy MLP was run for 100 steps. At the best step (step 60): mean ΔQED = **+0.076**, hit rate (Δ > 0.05) = **53%**, best QED = **0.943**. The latent space is structured enough that pushing `z` toward higher predicted QED reliably improves actual QED for the majority of starting points.

**Takeaway**: JTVAE gives 100% validity and novelty by construction; CharRNN better matches the ZINC distribution (lower FCD, higher QED). A genuine trade-off, not one model dominating.
