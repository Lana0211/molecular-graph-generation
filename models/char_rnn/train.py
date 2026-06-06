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
    """Run one training epoch; return average per-token cross-entropy loss."""
    model.train()
    total_loss = 0.0
    n_tokens   = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        # Flatten (B, T, V) → (B*T, V) for cross-entropy
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        # Clip gradients to prevent exploding gradients in the RNN
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Count only non-padding tokens when accumulating loss
        mask = targets != vocab.pad_idx
        total_loss += loss.item() * mask.sum().item()
        n_tokens   += mask.sum().item()
    return total_loss / max(n_tokens, 1)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, vocab):
    """Run one validation epoch; return average per-token loss."""
    model.eval()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CharRNN] Device: {device}")

    # Reserve the last 5,000 molecules for validation
    train_df     = pd.read_csv(args.data_path)
    train_smiles = train_df["SMILES"].dropna().tolist()
    print(f"[CharRNN] Training molecules: {len(train_smiles):,}")
    val_smiles   = train_smiles[-5000:]
    train_smiles = train_smiles[:-5000]

    vocab    = Vocabulary()
    train_ds = SMILESDataset(train_smiles, vocab, max_len=args.max_len)
    val_ds   = SMILESDataset(val_smiles,   vocab, max_len=args.max_len)

    _collate = partial(collate_fn, pad_idx=vocab.pad_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=0)

    model = CharRNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"[CharRNN] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Halve the learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )
    # Ignore padding positions in the loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    history  = []

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, vocab)
        va_loss = val_epoch(model, val_loader, criterion, device, vocab)
        scheduler.step(va_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
              f"time={elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        # Save checkpoint whenever validation loss improves
        if va_loss < best_val:
            best_val = va_loss
            ckpt = os.path.join(args.save_dir, "char_rnn_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "vocab": vocab, "args": vars(args),
                        "best_val_loss": best_val}, ckpt)
            print(f"  -> saved best model to {ckpt}")

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
