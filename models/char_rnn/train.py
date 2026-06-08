"""
Training script for the CharRNN baseline model.

Usage (from project root with venv activated):
  python -m models.char_rnn.train \
      --data_path data/train.csv \
      --save_dir  results/char_rnn \
      --epochs    30 \
      --batch_size 512 \
      --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running as a top-level module from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.char_rnn.model import CharRNN, SMILESDataset, Vocabulary, collate_fn
import pandas as pd


def train_epoch(model, loader, optimizer, criterion, device, vocab):
    """Run one training epoch; return average per-token cross-entropy loss.

    One epoch = one full pass over the training data. For each batch we do the
    standard supervised loop: forward pass → compute loss → backprop → update.
    """
    model.train()   # enable dropout / training-mode behaviour
    total_loss = 0.0
    n_tokens   = 0
    for inputs, targets in loader:
        # Move the batch onto the GPU (or CPU) the model lives on.
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()          # clear gradients from the previous step
        logits, _ = model(inputs)      # forward pass -> (B, T, V) scores
        # CrossEntropyLoss expects 2-D predictions and 1-D targets, so flatten
        # the batch and time axes together: (B, T, V) -> (B*T, V), (B, T) -> (B*T).
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()                # backprop: compute gradients
        # Clip the global gradient norm to 1.0 — RNNs are prone to exploding
        # gradients, and clipping keeps training numerically stable.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()               # apply the weight update
        # The reported loss is averaged over *real* tokens only; padding
        # positions carry no information so we exclude them from the average.
        mask = targets != vocab.pad_idx
        total_loss += loss.item() * mask.sum().item()
        n_tokens   += mask.sum().item()
    return total_loss / max(n_tokens, 1)   # max(...,1) guards against /0


@torch.no_grad()   # validation never updates weights, so skip gradient tracking
def val_epoch(model, loader, criterion, device, vocab):
    """Run one validation epoch; return average per-token loss.

    Identical to train_epoch but with no backprop / optimizer step — it only
    measures how well the model generalises to held-out molecules.
    """
    model.eval()   # disable dropout for deterministic evaluation
    total_loss = 0.0
    n_tokens   = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        mask = targets != vocab.pad_idx
        total_loss += loss.item() * mask.sum().item()
        n_tokens   += mask.sum().item()
    return total_loss / max(n_tokens, 1)


def main(args):
    # Prefer the GPU when one is available; fall back to CPU otherwise.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CharRNN] Device: {device}")

    # Load all training SMILES, then carve off the last 5,000 as a validation
    # set (a simple held-out split used to monitor generalisation / early stop).
    train_df     = pd.read_csv(args.data_path)
    train_smiles = train_df["SMILES"].dropna().tolist()
    print(f"[CharRNN] Training molecules: {len(train_smiles):,}")
    val_smiles   = train_smiles[-5000:]
    train_smiles = train_smiles[:-5000]

    # Build the vocabulary and wrap both splits as tokenised datasets.
    vocab    = Vocabulary()
    train_ds = SMILESDataset(train_smiles, vocab, max_len=args.max_len)
    val_ds   = SMILESDataset(val_smiles,   vocab, max_len=args.max_len)

    # partial() bakes the pad index into collate_fn so the DataLoader can call
    # it with just the batch. DataLoaders batch + shuffle the data for us.
    _collate = partial(collate_fn, pad_idx=vocab.pad_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=0)

    # Instantiate the model and move its parameters onto the device.
    model = CharRNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"[CharRNN] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Adam optimizer = adaptive-momentum gradient descent, a robust default.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # LR scheduler: if val loss plateaus for `patience` epochs, halve the LR so
    # the model can settle into a finer minimum instead of bouncing around.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )
    # Cross-entropy is the standard classification loss; ignore_index excludes
    # padding positions so they contribute neither loss nor gradient.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")   # track the best val loss seen so far
    history  = []             # per-epoch metrics, dumped to JSON at the end

    # ---- Main training loop: one iteration per epoch ----
    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, vocab)
        va_loss = val_epoch(model, val_loader, criterion, device, vocab)
        scheduler.step(va_loss)   # let the scheduler react to the val loss
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
              f"time={elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        # Checkpoint only the best model so far (lowest val loss). This keeps the
        # final saved weights at the point of best generalisation.
        if va_loss < best_val:
            best_val = va_loss
            ckpt = os.path.join(args.save_dir, "char_rnn_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "vocab": vocab, "args": vars(args),
                        "best_val_loss": best_val}, ckpt)
            print(f"  -> saved best model to {ckpt}")

    # Persist the loss history so the notebook can plot the training curves.
    import json
    with open(os.path.join(args.save_dir, "char_rnn_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("[CharRNN] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data/train.csv")
    parser.add_argument("--save_dir",   default="results/char_rnn")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--max_len",    type=int,   default=120)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=512)
    parser.add_argument("--num_layers", type=int,   default=3)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
