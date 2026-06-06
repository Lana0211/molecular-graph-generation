# Methodology & Comparison with the Original JTVAE Paper

How our implementation relates to Jin et al. (ICML 2018), what we kept, and where we simplified for a single-GPU course project.

> **Reference**: Wengong Jin, Regina Barzilay, Tommi Jaakkola.
> *Junction Tree Variational Autoencoder for Molecular Graph Generation.* ICML 2018.
> [arXiv:1802.04364](https://arxiv.org/abs/1802.04364) · [code](https://github.com/wengong-jin/icml18-jtnn)

---

## 1. Paper vs. Our Implementation

| Aspect | Original (Jin et al., 2018) | Ours |
| ------ | --------------------------- | ---- |
| Dataset | ZINC ~250K, vocab 780 clusters | ZINC via MOSES benchmark, 80K subset, vocab 287 clusters |
| Encoder | Tree-GRU (bottom-up + top-down) + graph MPN → `z = [z_T, z_G]` (concat) | Synchronous GCN + graph MPN → `z = (z_T + z_G)/2` (average) |
| Tree decoder | Separate topological + label prediction | Single next-cluster GRU |
| Graph decoder | Enumerates cluster attachments, scores with `z_G`, picks best | `enum_assemble` enumerates same attachments (ported from original), selects **randomly** — no learned scoring |
| Evaluation | Reconstruction accuracy, property optimization (penalized logP) | MOSES 8-metric suite, assembler ablation, QED gradient ascent, t-SNE |
| Baseline | CVAE, GVAE, GraphVAE, atom-by-atom LSTM | CharRNN (character-level GRU, equivalent to paper's CVAE) |

**Key simplification**: we port the `enum_assemble` enumeration logic (enabling ring fusion) but skip the `z_G`-conditioned scorer. Candidates are chosen randomly, so `z_G` has limited influence on the final molecule. This is why scaffold similarity is low and FCD is higher than CharRNN.

---

## 2. Why We Don't Report Reconstruction Accuracy

The paper measures how often encode→decode returns the *exact* input molecule (76.7%). This requires using `z_G` to select the correct cluster attachment — something our random assembly doesn't do. Reporting this number would be misleading, so we use **FCD** (Fréchet ChemNet Distance) as the main distributional quality metric instead.

---

## 3. Results

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

Assembler ablation (same checkpoint, columns 2 vs 3): enum_assemble cuts FCD by 39% and raises diversity, confirming ring fusion helps. Low scaffold similarity reflects novel ring systems produced by random attachment selection.

### Property optimization

200 test molecules were encoded, then gradient ascent on a QED proxy MLP was run for 100 steps. At the best step (step 60): mean ΔQED = **+0.076**, hit rate (Δ > 0.05) = **53%**, best QED = **0.943**. The latent space is structured enough that pushing `z` toward higher predicted QED reliably improves actual QED for the majority of starting points.

**Takeaway**: JTVAE gives 100% validity and novelty by construction; CharRNN better matches the ZINC distribution (lower FCD, higher QED). A genuine trade-off, not one model dominating.
