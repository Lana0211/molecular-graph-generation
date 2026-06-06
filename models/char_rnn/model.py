"""
Character-level RNN (GRU) for SMILES generation.

Architecture:
  Embedding → multi-layer GRU → Linear → Softmax

During generation, the model samples one character at a time in an
auto-regressive fashion until the <EOS> token is produced or max length
is reached.

Reference:
  Segler et al., "Generating Focused Molecule Libraries for Drug
  Discovery with Recurrent Neural Networks", ACS Cent. Sci. 2018.
  MOSES implementation: https://github.com/molecularsets/moses
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Special control tokens used at sequence boundaries and for unknown chars
PAD = "<PAD>"   # padding token (ignored by loss)
BOS = "<BOS>"   # beginning-of-sequence token
EOS = "<EOS>"   # end-of-sequence token
UNK = "<UNK>"   # unknown character (maps all out-of-vocab chars here)

# All printable characters that appear in canonical SMILES strings
SMILES_CHARS = list(
    "#%)(+-.1032547698=@ABCFGHIKLMNOPRSTUVWYZ[\\]abcdefghilmnoprstuy"
)


class Vocabulary:
    """Maps SMILES characters ↔ integer indices."""

    def __init__(self, extra_tokens: list[str] | None = None):
        special = [PAD, BOS, EOS, UNK]
        chars = special + SMILES_CHARS + (extra_tokens or [])
        self.i2c = chars                              # index → character list
        self.c2i = {c: i for i, c in enumerate(chars)}  # character → index dict

    def __len__(self) -> int:
        return len(self.i2c)

    def encode(self, smiles: str) -> list[int]:
        """Convert a SMILES string to a list of token indices."""
        return [self.c2i.get(ch, self.c2i[UNK]) for ch in smiles]

    def decode(self, indices: list[int]) -> str:
        """Convert token indices back to a SMILES string, stopping at EOS."""
        chars = []
        for idx in indices:
            ch = self.i2c[idx]
            if ch == EOS:
                break                      # stop decoding at end-of-sequence
            if ch not in (PAD, BOS, UNK):  # skip control tokens
                chars.append(ch)
        return "".join(chars)

    @property
    def pad_idx(self) -> int:
        return self.c2i[PAD]

    @property
    def bos_idx(self) -> int:
        return self.c2i[BOS]

    @property
    def eos_idx(self) -> int:
        return self.c2i[EOS]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset


class SMILESDataset(Dataset):
    """Tokenised SMILES wrapped as (input, target) pairs for teacher-forcing."""

    def __init__(self, smiles_list: list[str], vocab: Vocabulary,
                 max_len: int = 120):
        self.vocab = vocab
        self.data: list[list[int]] = []
        for smi in smiles_list:
            enc = vocab.encode(smi)
            # Keep only sequences short enough to fit BOS + tokens + EOS
            if len(enc) <= max_len - 2:
                self.data.append([vocab.bos_idx] + enc + [vocab.eos_idx])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        # Teacher-forcing: input starts with BOS, target ends with EOS
        return seq[:-1], seq[1:]


def collate_fn(batch, pad_idx: int):
    """Pad sequences in a batch to the same length."""
    inputs, targets = zip(*batch)
    inputs  = nn.utils.rnn.pad_sequence(inputs,  batch_first=True, padding_value=pad_idx)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    return inputs, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CharRNN(nn.Module):
    """GRU-based character-level language model for SMILES generation."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # dropout only applies between layers, not on the last layer
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)  # project hidden state → logits

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding(x)              # (B, T, E)
        out, hidden = self.rnn(embed, hidden)  # (B, T, H)
        logits = self.fc(out)                  # (B, T, V)  unnormalised log-probs
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero-initialised hidden state tensor."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                           device=device)

    @torch.no_grad()
    def generate(
        self,
        vocab: Vocabulary,
        n: int = 1,
        max_len: int = 120,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> list[str]:
        """Generate `n` SMILES strings by autoregressive sampling.

        temperature > 1 → more random; temperature < 1 → more conservative.
        """
        device = torch.device(device)
        self.eval()
        self.to(device)

        generated = []
        for _ in range(n):
            hidden = self.init_hidden(1, device)
            x = torch.tensor([[vocab.bos_idx]], device=device)  # seed with BOS
            tokens = []

            for _ in range(max_len):
                logits, hidden = self.forward(x, hidden)
                logits = logits[:, -1, :] / temperature       # apply temperature
                probs  = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()  # sample
                if next_token == vocab.eos_idx:
                    break                                       # stop at EOS
                tokens.append(next_token)
                x = torch.tensor([[next_token]], device=device)  # feed back

            generated.append(vocab.decode(tokens))
        return generated
