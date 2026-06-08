"""
Training script for the JTVAE model.

Usage (from project root with venv activated):
  python -m models.jtvae.train \
      --data_path   data/train.csv \
      --vocab_path  data/vocab.txt \
      --save_dir    results/jtvae \
      --epochs      30 \
      --batch_size  32 \
      --beta        1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Allow running as a module from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.jtvae.mol_tree import MolTree, Vocab, build_vocab
from models.jtvae.jtnn_vae import JTVAE
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _parse_smiles(smi: str) -> "tuple[MolTree | None, str]":
    """Parse a SMILES string into a MolTree.

    Returns (tree, reason) where reason is one of:
      'ok'          – successfully parsed
      'kekulize'    – RDKit cannot kekulize the molecule
      'invalid'     – RDKit cannot parse the SMILES at all
      'empty_tree'  – molecule is valid but produces an empty junction tree
    """
    from rdkit import Chem
    from rdkit.Chem.rdchem import KekulizeException
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "invalid"
    try:
        Chem.Kekulize(mol)
    except Exception:
        return None, "kekulize"
    try:
        t = MolTree(smi)
        if not t.nodes:
            return None, "empty_tree"
        return t, "ok"
    except Exception:
        return None, "invalid"


class JTVAEDataset(Dataset):
    """Pre-parse SMILES into junction trees and assign vocabulary indices."""

    def __init__(self, smiles_list: list[str], vocab: Vocab, n_workers: int = 4):
        from rdkit import RDLogger
        # Suppress RDKit per-molecule warnings; we collect stats ourselves
        RDLogger.DisableLog("rdApp.error")
        RDLogger.DisableLog("rdApp.warning")

        n_total = len(smiles_list)
        print(f"[JTVAEDataset] Parsing {n_total:,} SMILES ...")

        if n_workers > 1:
            with Pool(n_workers) as pool:
                results = pool.map(_parse_smiles, smiles_list)
        else:
            results = [_parse_smiles(s) for s in smiles_list]

        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")

        self.data     = []
        skip_counts   = {"kekulize": 0, "invalid": 0, "empty_tree": 0, "oov": 0}

        for tree, reason in results:
            if tree is None:
                skip_counts[reason] += 1
                continue
            # Skip molecules that contain clusters not in the vocabulary
            valid = True
            for node in tree.nodes:
                if node.smiles not in vocab:
                    valid = False
                    break
                node.wid = vocab[node.smiles]
            if valid:
                self.data.append(tree)
            else:
                skip_counts["oov"] += 1

        n_skip   = sum(skip_counts.values())
        n_kept   = len(self.data)
        pct_skip = 100 * n_skip / max(n_total, 1)
        print(f"[JTVAEDataset] Kept {n_kept:,} / {n_total:,}  "
              f"({100*n_kept/max(n_total,1):.1f}%)")
        print(f"[JTVAEDataset] Skipped {n_skip:,} ({pct_skip:.2f}%) — "
              f"kekulize: {skip_counts['kekulize']:,}  "
              f"invalid SMILES: {skip_counts['invalid']:,}  "
              f"empty tree: {skip_counts['empty_tree']:,}  "
              f"out-of-vocab: {skip_counts['oov']:,}")

        # Pre-compute and cache graph features for every molecule.
        # Atom-feature extraction is pure-Python and slow; doing it once here
        # (rather than every epoch inside the training loop) means each batch
        # only slices ready-made tensors. This cut per-epoch time from ~10 min
        # to ~1-2 min. The result is stashed on each tree as `_graph_cache`.
        print("[JTVAEDataset] Pre-computing graph features ...")
        from models.jtvae.mpn import atom_features, ATOM_FDIM, MAX_NB
        cached, failed = 0, 0
        for tree in self.data:
            mol = tree.mol
            if mol is None:
                tree._graph_cache = None
                failed += 1
                continue
            try:
                # Row 0 is the shared padding atom; real atoms start at index 1.
                fatoms_list = [torch.zeros(ATOM_FDIM)]
                nbr_lists   = [[]]
                idx_map     = {}     # RDKit atom idx → local cache index
                n_at        = 1
                # One feature vector per atom, recording its local index.
                for atom in mol.GetAtoms():
                    fatoms_list.append(atom_features(atom))
                    idx_map[atom.GetIdx()] = n_at
                    nbr_lists.append([])
                    n_at += 1
                # Each bond adds the two atoms to each other's neighbour list.
                for bond in mol.GetBonds():
                    g1 = idx_map[bond.GetBeginAtomIdx()]
                    g2 = idx_map[bond.GetEndAtomIdx()]
                    nbr_lists[g1].append(g2)
                    nbr_lists[g2].append(g1)
                # Pack the ragged neighbour lists into a fixed (n_at, MAX_NB) tensor.
                anbr = torch.zeros(n_at, MAX_NB, dtype=torch.long)
                for i, nbrs in enumerate(nbr_lists):
                    for k, j in enumerate(nbrs[:MAX_NB]):
                        anbr[i, k] = j
                # Keep tensors on CPU; _build_graph_batch moves them to GPU later.
                tree._graph_cache = (
                    torch.stack(fatoms_list, dim=0),   # (n_at, ATOM_FDIM)
                    anbr,                               # (n_at, MAX_NB)
                    n_at - 1,                          # num atoms (excl. padding)
                )
                cached += 1
            except Exception:
                tree._graph_cache = None   # mark for the slow fallback path
                failed += 1
        print(f"[JTVAEDataset] Graph cache: {cached:,} ok, {failed:,} failed")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def jtvae_collate(batch):
    """Pass-through collate: the model's forward handles batching internally."""
    return batch   # list of MolTree objects


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def anneal_beta(epoch: int, max_epoch: int,
                beta_max: float = 1.0, warmup: int = 5) -> float:
    """Return the KL weight β for the current epoch using linear annealing.

    The full VAE loss is rec_loss + β·KL. Turning on the KL term immediately
    tends to collapse the latent, so we ramp β up gradually:
      - epochs < warmup: β = 0, pure reconstruction (learn to use the latent).
      - after warmup:    β rises linearly from 0 to beta_max by the last epoch.
    """
    if epoch < warmup:
        return 0.0   # pure reconstruction phase, no KL pressure yet
    # Linearly interpolate from 0 (at warmup) up to beta_max (at max_epoch).
    return min(beta_max, beta_max * (epoch - warmup) / max(1, max_epoch - warmup))


def train_one_epoch(model: JTVAE, loader, optimizer, device: torch.device,
                    beta: float, free_bits: float = 0.0) -> dict:
    """Run one training epoch; return average loss statistics.

    Standard VAE training loop, with two robustness guards: batches that produce
    NaN/Inf losses are skipped, and per-batch RuntimeErrors (e.g. a degenerate
    graph) are caught so one bad molecule can't kill the whole epoch.
    """
    model.train()
    # Running sums of each loss component; divided by n_batches at the end.
    total     = {"loss": 0.0, "rec_loss": 0.0, "kl_loss": 0.0}
    n_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        try:
            # forward() returns the scalar loss plus a dict of its components.
            loss, stats = model(batch, beta=beta, free_bits=free_bits)
            if torch.isnan(loss) or torch.isinf(loss):
                continue   # skip numerically unstable batches
            loss.backward()
            # Clip gradients; the norm is large because junction-tree gradients
            # are naturally spikier than a plain RNN's.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
            optimizer.step()
            for k in total:
                total[k] += stats[k]
            n_batches += 1
        except RuntimeError as e:
            print(f"  [skip batch] {e}")   # usually OOM or degenerate graphs
            continue
    # Fail fast if every batch was skipped — otherwise we would silently
    # report loss=0 and checkpoint untrained weights (a real bug, not success).
    if n_batches == 0:
        raise RuntimeError(
            "All batches failed in this epoch — check the model, not the data. "
            "Refusing to report a fake loss of 0."
        )
    # Return per-batch averages
    return {k: v / n_batches for k, v in total.items()}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[JTVAE] Device: {device}")

    train_df     = pd.read_csv(args.data_path)
    train_smiles = train_df["SMILES"].dropna().tolist()

    # Build vocabulary from a subset to save time; reuse if file already exists
    if os.path.exists(args.vocab_path):
        with open(args.vocab_path) as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        vocab = Vocab(vocab_list)
        print(f"[JTVAE] Loaded vocab: {len(vocab)} entries")
    else:
        print("[JTVAE] Building vocabulary (this may take a few minutes) ...")
        vocab = build_vocab(train_smiles[:args.vocab_sample])
        os.makedirs(os.path.dirname(args.vocab_path) or ".", exist_ok=True)
        with open(args.vocab_path, "w") as f:
            for smi in vocab.vocab:
                f.write(smi + "\n")
        print(f"[JTVAE] Vocab size: {len(vocab)}, saved to {args.vocab_path}")

    # Reserve last 2,000 molecules for validation
    val_smiles   = train_smiles[-2000:]
    train_smiles = train_smiles[:-2000]
    if args.subset:
        train_smiles = train_smiles[:args.subset]   # optional subset for quick runs

    train_ds = JTVAEDataset(train_smiles, vocab, n_workers=args.n_workers)
    val_ds   = JTVAEDataset(val_smiles,   vocab, n_workers=0)

    # num_workers=0 avoids Windows multiprocessing issues in DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=jtvae_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=jtvae_collate, num_workers=0)

    model = JTVAE(
        vocab=vocab,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        depth_t=args.depth_t,
        depth_g=args.depth_g,
    ).to(device)
    print(f"[JTVAE] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Decay the learning rate by 10 % every 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    os.makedirs(args.save_dir, exist_ok=True)
    history   = []
    # Checkpoint on reconstruction loss, not total loss: with KL annealing the
    # total loss is artificially lowest during the beta=0 warmup, so selecting on
    # it would persist a pre-regularization (collapsed) model.
    best_rec = float("inf")

    for epoch in range(1, args.epochs + 1):
        beta     = anneal_beta(epoch, args.epochs, args.beta, warmup=args.warmup)
        t0       = time.time()
        tr_stats = train_one_epoch(model, train_loader, optimizer, device, beta,
                                   free_bits=args.free_bits)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | beta={beta:.3f} | "
            f"loss={tr_stats['loss']:.4f} | "
            f"rec={tr_stats['rec_loss']:.4f} | "
            f"kl={tr_stats['kl_loss']:.4f} | "
            f"time={time.time()-t0:.1f}s"
        )
        history.append({"epoch": epoch, "beta": beta, **tr_stats})

        # Save the best model by reconstruction loss, but only after warmup:
        # during the beta=0 warmup the latent is unregularized, so rec_loss is
        # artificially low and would persist a pre-regularization (collapsed) model.
        if epoch > args.warmup and tr_stats["rec_loss"] < best_rec:
            best_rec = tr_stats["rec_loss"]
            ckpt = os.path.join(args.save_dir, "jtvae_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "vocab": vocab, "args": vars(args)}, ckpt)
            print(f"  -> saved {ckpt}")

        # Save periodic checkpoints every 5 epochs for resumability
        if epoch % 5 == 0:
            ckpt = os.path.join(args.save_dir, f"jtvae_epoch{epoch}.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "vocab": vocab}, ckpt)

    with open(os.path.join(args.save_dir, "jtvae_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("[JTVAE] Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    default="data/train.csv")
    p.add_argument("--vocab_path",   default="data/vocab.txt")
    p.add_argument("--save_dir",     default="results/jtvae")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--hidden_dim",   type=int,   default=300)
    p.add_argument("--latent_dim",   type=int,   default=56)
    p.add_argument("--depth_t",      type=int,   default=1)
    p.add_argument("--depth_g",      type=int,   default=3)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--beta",         type=float, default=1.0)
    p.add_argument("--free_bits",    type=float, default=0.5,
                   help="Minimum KL nats per latent dim (Kingma et al. 2016). "
                        "Prevents posterior collapse; 0 disables.")
    p.add_argument("--warmup",       type=int,   default=5)
    p.add_argument("--subset",       type=int,   default=0,
                   help="Use only this many training samples (0 = all)")
    p.add_argument("--vocab_sample", type=int,   default=50000,
                   help="Number of SMILES used to build the vocabulary")
    p.add_argument("--n_workers",    type=int,   default=0)
    args = p.parse_args()
    main(args)
