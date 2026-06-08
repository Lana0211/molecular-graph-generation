"""
Property-guided optimization in the JTVAE latent space.

Strategy: Bayesian Optimization (BO) over the latent space z to
maximize QED (drug-likeness).  This approach follows:
  Griffiths & Hernández-Lobato, "Constrained Bayesian Optimization for
  Automatic Chemical Design Using Variational Autoencoders", ACS Cent. Sci. 2020.

Algorithm
---------
1. Encode a seed set of molecules to obtain z vectors.
2. Compute QED for each seed molecule.
3. Fit a Gaussian Process (GP) surrogate on (z, QED) pairs.
4. Use Expected Improvement (EI) to propose new z points.
5. Decode z → SMILES, compute actual QED, update GP.
6. Repeat for N iterations.

Alternatively, a simpler gradient ascent is available (--method gradient).

Usage:
  python -m optimization.property_optimize \
      --ckpt       results/jtvae/jtvae_best.pt \
      --seed_csv   data/test.csv \
      --output     results/optimized_molecules.csv \
      --n_iter     50 \
      --method     bo
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.jtvae.mol_tree import MolTree
from models.jtvae.jtnn_vae import JTVAE
from evaluation.metrics import compute_qed, compute_sa, get_mol


# ---------------------------------------------------------------------------
# Property scoring
# ---------------------------------------------------------------------------

def score_smiles(smiles_list: list[str]) -> list[float]:
    """Return QED for each SMILES (0.0 for invalid)."""
    scores = []
    for smi in smiles_list:
        mol = get_mol(smi)
        if mol is None:
            scores.append(0.0)
        else:
            scores.append(compute_qed(mol))
    return scores


# ---------------------------------------------------------------------------
# Gradient ascent in latent space
# ---------------------------------------------------------------------------

def gradient_ascent(
    model: JTVAE,
    z_init: torch.Tensor,
    n_steps: int = 200,
    lr: float = 0.01,
    temperature: float = 1.0,
) -> list[dict]:
    """Optimise z via gradient ascent on an MLP-predicted QED proxy.

    Since QED is not differentiable w.r.t. z, we train a small proxy
    network on the initial seed (z, QED) pairs and differentiate through it.
    """
    device = z_init.device
    model.eval()

    # Step 1: decode the seed latents and score them, giving (z, QED) pairs that
    # the proxy network will learn from.
    print("[opt] Collecting seed scores ...")
    seed_smiles = model._decode_z(z_init, temperature=temperature)
    seed_scores = score_smiles(seed_smiles)

    z_np  = z_init.detach().cpu().numpy()
    y_np  = np.array(seed_scores, dtype=np.float32)

    # Step 2: train a small differentiable proxy z → QED. Sigmoid bounds the
    # output to [0, 1], matching QED's range. This proxy stands in for the real
    # (non-differentiable) QED so we can take gradients w.r.t. z.
    proxy = torch.nn.Sequential(
        torch.nn.Linear(model.latent_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Sigmoid(),
    ).to(device)
    opt_proxy = torch.optim.Adam(proxy.parameters(), lr=1e-3)
    z_t = z_init.detach()
    y_t = torch.tensor(y_np, device=device).unsqueeze(1)
    for _ in range(200):   # fit the proxy by regressing predicted QED onto true QED
        opt_proxy.zero_grad()
        pred = proxy(z_t)
        loss = torch.nn.functional.mse_loss(pred, y_t)
        loss.backward()
        opt_proxy.step()

    # Step 3: ascend the proxy's gradient. z_opt is now the thing being optimised
    # (requires_grad), while the proxy weights are frozen.
    z_opt = z_init.clone().detach().requires_grad_(True)
    optimiser = torch.optim.Adam([z_opt], lr=lr)
    records = []

    print(f"[opt] Gradient ascent for {n_steps} steps ...")
    for step in range(n_steps):
        optimiser.zero_grad()
        score = proxy(z_opt).mean()
        # Optimisers minimise, so minimising -score maximises predicted QED,
        # nudging z toward regions the proxy believes are more drug-like.
        (-score).backward()
        optimiser.step()

        # Every 20 steps, decode the current z and measure the *true* QED so we
        # can track whether proxy gains translate into real improvements.
        if step % 20 == 0:
            with torch.no_grad():
                smiles_list = model._decode_z(z_opt.detach(), temperature=temperature)
            scores = score_smiles(smiles_list)
            mean_qed = np.mean([s for s in scores if s > 0] or [0])
            print(f"  step {step:4d} | proxy_score={score.item():.4f} | "
                  f"actual_QED={mean_qed:.4f}")
            for smi, sc in zip(smiles_list, scores):
                records.append({"step": step, "smiles": smi, "qed": sc})

    return records


# ---------------------------------------------------------------------------
# Bayesian Optimisation (scikit-learn GP)
# ---------------------------------------------------------------------------

def bayesian_opt(
    model: JTVAE,
    z_seed: torch.Tensor,
    n_iter: int = 50,
    temperature: float = 1.0,
    n_restarts: int = 5,
) -> list[dict]:
    """BO over the latent space with a GP surrogate."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
    except ImportError:
        print("[opt] scikit-learn not available, falling back to gradient ascent")
        return gradient_ascent(model, z_seed, temperature=temperature)

    device = z_seed.device
    model.eval()

    z_np = z_seed.detach().cpu().numpy()
    smiles_init = model._decode_z(z_seed, temperature=temperature)
    y_np = np.array(score_smiles(smiles_init), dtype=np.float64)

    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
    )

    # Seed the record log with the initial observations (iteration -1).
    records = []
    for i, (smi, sc) in enumerate(zip(smiles_init, y_np)):
        records.append({"iteration": -1, "smiles": smi, "qed": float(sc)})

    print(f"[BO] Starting {n_iter} BO iterations ...")
    for it in range(n_iter):
        # Refit the GP surrogate on all observations gathered so far.
        gp.fit(z_np, y_np)

        # Expected Improvement (EI): score random candidate latents by how much
        # they are expected to beat the best QED seen, balancing high predicted
        # mean (mu) against uncertainty (sigma) — i.e. exploit vs. explore.
        best_y = y_np.max()
        z_candidates = np.random.randn(500, z_np.shape[1])
        mu, sigma = gp.predict(z_candidates, return_std=True)
        imp   = mu - best_y - 0.01          # predicted improvement over current best
        Z     = imp / (sigma + 1e-9)        # standardised improvement
        from scipy.stats import norm
        ei    = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        next_z = z_candidates[np.argmax(ei)]   # pick the most promising candidate

        # Evaluate the true QED at the chosen latent and add it to the dataset.
        z_t   = torch.tensor(next_z, dtype=torch.float, device=device).unsqueeze(0)
        new_smi = model._decode_z(z_t, temperature=temperature)
        new_sc  = score_smiles(new_smi)[0]

        z_np  = np.vstack([z_np,   next_z])   # append the new observation so the
        y_np  = np.append(y_np,  new_sc)      # GP improves on the next iteration

        records.append({"iteration": it, "smiles": new_smi[0], "qed": new_sc})
        if it % 10 == 0:
            print(f"  iter {it:3d} | new_QED={new_sc:.4f} | best={y_np.max():.4f}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[opt] Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    vocab = ckpt["vocab"]
    model_args = ckpt.get("args", {})

    model = JTVAE(
        vocab=vocab,
        hidden_dim=model_args.get("hidden_dim", 300),
        latent_dim=model_args.get("latent_dim", 56),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("[opt] Model loaded.")

    # Encode seed molecules
    seed_df = pd.read_csv(args.seed_csv)
    seed_smiles = seed_df["SMILES"].dropna().tolist()[:args.n_seed]

    trees = []
    for smi in seed_smiles:
        t = MolTree(smi)
        valid = True
        for node in t.nodes:
            if node.smiles not in vocab:
                valid = False
                break
            node.wid = vocab[node.smiles]
        if valid and t.nodes:
            trees.append(t)
    print(f"[opt] Valid seed trees: {len(trees)}")

    if not trees:
        print("[opt] No valid seed molecules, sampling from prior instead.")
        z_seed = torch.randn(args.n_seed, model.latent_dim, device=device)
    else:
        with torch.no_grad():
            z_mean, _, _ = model.encode(trees)
        z_seed = z_mean

    # Optimise
    if args.method == "gradient":
        records = gradient_ascent(model, z_seed, n_steps=args.n_iter,
                                   temperature=args.temperature)
    else:
        records = bayesian_opt(model, z_seed, n_iter=args.n_iter,
                               temperature=args.temperature)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[opt] Saved {len(out_df)} records to {args.output}")

    top = out_df[out_df["qed"] > 0].nlargest(10, "qed")
    print("\nTop optimised molecules:")
    print(top[["smiles", "qed"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--seed_csv",    default="data/test.csv")
    p.add_argument("--output",      default="results/optimized_molecules.csv")
    p.add_argument("--n_seed",      type=int,   default=100)
    p.add_argument("--n_iter",      type=int,   default=50)
    p.add_argument("--method",      choices=["gradient", "bo"], default="bo")
    p.add_argument("--temperature", type=float, default=1.0)
    p.parse_args()
    args = p.parse_args()
    main(args)
