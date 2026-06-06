"""
Molecular generation evaluation metrics compatible with the MOSES benchmark.

References:
  Polykovskiy et al., "Molecular Sets (MOSES): A Benchmarking Platform
  for Molecular Generation Models", Front. Pharmacol. 2020.

Metrics computed:
  - validity            : fraction of parseable SMILES
  - uniqueness          : fraction of unique molecules (among valid)
  - novelty             : fraction not appearing in training set
  - internal_diversity  : average pairwise Tanimoto dissimilarity
  - fcd                 : Fréchet ChemNet Distance (requires fcd_torch)
  - snn                 : Similarity to Nearest Neighbor (mean max Tanimoto to test)
  - scaffold_similarity : fraction of generated Murcko scaffolds seen in test
  - qed_mean/std        : drug-likeness distribution
  - sa_mean/std         : synthetic accessibility distribution
  - logp_mean/std       : lipophilicity distribution
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")   # suppress Python-level warnings
# Suppress RDKit C++ parse/kekulize messages. Generated SMILES are often
# chemically invalid by design — that is exactly what the validity metric
# measures — so we silence the per-molecule console spam during evaluation.
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def canonicalize(smi: str) -> Optional[str]:
    """Return the canonical SMILES, or None if the input is invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_mol(smi: str):
    """Attempt to parse a SMILES string; return None on failure."""
    return Chem.MolFromSmiles(smi)


def morgan_fp(mol, radius: int = 2, n_bits: int = 2048):
    """Compute a 2048-bit Morgan (ECFP4) fingerprint for similarity searches."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(fp1, fp2) -> float:
    """Tanimoto coefficient between two fingerprints (0 = dissimilar, 1 = identical)."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_scaffold(smi: str) -> Optional[str]:
    """Return the Murcko scaffold SMILES for a molecule."""
    mol = get_mol(smi)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Molecular property computation
# ---------------------------------------------------------------------------

def compute_qed(mol) -> float:
    """Quantitative Estimate of Drug-likeness (0–1; higher = more drug-like)."""
    try:
        return QED.qed(mol)
    except Exception:
        return float("nan")


def compute_sa(mol) -> float:
    """Approximate synthetic accessibility score (1–10; lower = easier to synthesise).

    Full SA score requires the RDKit contrib SA_Score.py.
    Here we use a lightweight heuristic based on stereocentres, spiro atoms,
    bridgehead atoms, and ring count.
    """
    try:
        from rdkit.Chem import rdMolDescriptors
        num_stereo     = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        rings          = mol.GetRingInfo()
        num_spiro      = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        complexity     = num_stereo + num_spiro + num_bridgehead
        # Linear proxy: complexity and ring count correlate with synthesis difficulty
        sa_proxy = 1.0 + complexity * 0.5 + max(0, rings.NumRings() - 3) * 0.3
        return min(10.0, sa_proxy)   # cap at 10 (maximum difficulty)
    except Exception:
        return float("nan")


def compute_logp(mol) -> float:
    """Wildman-Crippen LogP: estimated lipophilicity."""
    try:
        return Descriptors.MolLogP(mol)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def validity(smiles_list: List[str]) -> float:
    """Fraction of generated SMILES that can be parsed by RDKit."""
    valid = [s for s in smiles_list if get_mol(s) is not None]
    return len(valid) / len(smiles_list) if smiles_list else 0.0


def get_valid(smiles_list: List[str]) -> List[str]:
    """Filter a list to keep only RDKit-parseable SMILES."""
    return [s for s in smiles_list if get_mol(s) is not None]


def uniqueness(smiles_list: List[str]) -> float:
    """Fraction of unique canonical SMILES among all valid generated molecules."""
    valid = get_valid(smiles_list)
    if not valid:
        return 0.0
    canonical = [s for s in (canonicalize(x) for x in valid) if s]
    return len(set(canonical)) / len(canonical)


def novelty(smiles_list: List[str], train_smiles: List[str]) -> float:
    """Fraction of valid generated molecules not seen in the training set."""
    valid = get_valid(smiles_list)
    if not valid:
        return 0.0
    # Build a set of canonical training SMILES for O(1) lookup
    train_set = set(canonicalize(s) for s in train_smiles)
    train_set.discard(None)
    novel = [s for s in valid if canonicalize(s) not in train_set]
    return len(novel) / len(valid)


def internal_diversity(smiles_list: List[str], n_jobs: int = 1,
                       sample_size: int = 1000) -> float:
    """Mean pairwise Tanimoto *dissimilarity* (higher = more structurally diverse).

    O(n²) computation is subsampled to at most `sample_size` molecules.
    """
    valid = get_valid(smiles_list)
    if len(valid) < 2:
        return 0.0
    if len(valid) > sample_size:
        valid = list(np.random.default_rng(42).choice(valid, sample_size, replace=False))
    fps  = [morgan_fp(get_mol(s)) for s in valid]
    sims = [tanimoto(fps[i], fps[j])
            for i in range(len(fps)) for j in range(i + 1, len(fps))]
    return 1.0 - float(np.mean(sims)) if sims else 0.0   # dissimilarity = 1 - similarity


def snn(generated: List[str], reference: List[str],
        sample_size: int = 1000) -> float:
    """Similarity to Nearest Neighbor: mean of max Tanimoto similarity to the test set.

    Higher SNN means generated molecules are structurally closer to real drug-like
    molecules in the test set.
    """
    gen_valid = get_valid(generated)
    ref_valid = get_valid(reference)
    if not gen_valid or not ref_valid:
        return 0.0
    if len(gen_valid) > sample_size:
        gen_valid = list(np.random.default_rng(0).choice(gen_valid, sample_size, replace=False))
    gen_fps  = [morgan_fp(get_mol(s)) for s in gen_valid]
    ref_fps  = [morgan_fp(get_mol(s)) for s in ref_valid[:sample_size]]
    # For each generated molecule find its most similar reference molecule
    max_sims = [max(DataStructs.BulkTanimotoSimilarity(gfp, ref_fps)) for gfp in gen_fps]
    return float(np.mean(max_sims))


def scaffold_similarity(generated: List[str], reference: List[str]) -> float:
    """Fraction of generated Murcko scaffolds that also appear in the reference set."""
    gen_scaffolds = set(get_scaffold(s) for s in get_valid(generated)) - {None}
    ref_scaffolds = set(get_scaffold(s) for s in get_valid(reference)) - {None}
    if not gen_scaffolds:
        return 0.0
    return len(gen_scaffolds & ref_scaffolds) / len(gen_scaffolds)


def property_stats(smiles_list: List[str]) -> dict:
    """Return mean and std of QED, SA score, and LogP for valid molecules."""
    valid = get_valid(smiles_list)
    if not valid:
        return {}
    mols  = [get_mol(s) for s in valid]
    qeds  = [v for v in (compute_qed(m)  for m in mols) if not np.isnan(v)]
    sas   = [v for v in (compute_sa(m)   for m in mols) if not np.isnan(v)]
    logps = [v for v in (compute_logp(m) for m in mols) if not np.isnan(v)]
    return {
        "qed_mean":  float(np.mean(qeds)),   "qed_std":  float(np.std(qeds)),
        "sa_mean":   float(np.mean(sas)),    "sa_std":   float(np.std(sas)),
        "logp_mean": float(np.mean(logps)),  "logp_std": float(np.std(logps)),
    }


def compute_fcd(generated: List[str], reference: List[str],
                device: str = "cpu") -> float:
    """Fréchet ChemNet Distance: lower → generated distribution closer to reference.

    Requires the fcd_torch package (pip install fcd_torch).
    """
    try:
        import fcd
        # Canonicalize with RDKit ourselves (get_valid → canonicalize) instead of
        # fcd's internal canonical_smiles, which spawns a multiprocessing Pool and
        # crashes on Windows ("bootstrapping phase" error) outside __main__.
        gen_valid = [c for c in (canonicalize(s) for s in get_valid(generated)) if c]
        ref_valid = [c for c in (canonicalize(s) for s in get_valid(reference)) if c]
        if not gen_valid or not ref_valid:
            return float("nan")

        model = fcd.load_ref_model()
        # n_jobs=1 keeps prediction single-process (no Pool); device='cpu' is safe
        gen_act = fcd.get_predictions(model, gen_valid, n_jobs=1, device=device)
        ref_act = fcd.get_predictions(model, ref_valid, n_jobs=1, device=device)

        # Fréchet distance between the two activation Gaussians
        score = fcd.calculate_frechet_distance(
            mu1=gen_act.mean(0), sigma1=np.cov(gen_act.T),
            mu2=ref_act.mean(0), sigma2=np.cov(ref_act.T),
        )
        return float(score)
    except Exception as e:
        print(f"[FCD] Could not compute: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_all_metrics(
    generated: List[str],
    train: Optional[List[str]] = None,
    test:  Optional[List[str]] = None,
    compute_fcd_flag: bool = True,
) -> dict:
    """Compute all evaluation metrics and return them as a single dictionary."""
    results = {}
    results["validity"]            = validity(generated)
    results["uniqueness"]          = uniqueness(generated)
    if train is not None:
        results["novelty"]         = novelty(generated, train)
    results["internal_diversity"]  = internal_diversity(generated)
    if test is not None:
        results["snn"]             = snn(generated, test)
        results["scaffold_similarity"] = scaffold_similarity(generated, test)
        if compute_fcd_flag:
            results["fcd"]         = compute_fcd(generated, test)
    results.update(property_stats(generated))
    return results


def print_metrics(results: dict) -> None:
    """Pretty-print a metrics dictionary."""
    col_w = 24
    print("\n" + "=" * 50)
    print(f"{'Metric':<{col_w}} {'Value':>10}")
    print("-" * 50)
    for k, v in results.items():
        print(f"{k:<{col_w}} {v:>10.4f}" if isinstance(v, float)
              else f"{k:<{col_w}} {v!r:>10}")
    print("=" * 50 + "\n")
