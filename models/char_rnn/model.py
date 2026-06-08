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
    """Maps SMILES characters ↔ integer indices.

    A neural network can only consume numbers, so every SMILES character must
    be assigned a unique integer id. We keep two lookup structures that are
    inverses of each other: a list (index → char) and a dict (char → index).
    """

    def __init__(self, extra_tokens: list[str] | None = None):
        # The four control tokens MUST come first so their indices are fixed
        # and small (PAD=0, BOS=1, EOS=2, UNK=3); the rest of the code relies
        # on pad_idx/bos_idx/eos_idx below.
        special = [PAD, BOS, EOS, UNK]
        chars = special + SMILES_CHARS + (extra_tokens or [])
        self.i2c = chars                              # index → character list
        self.c2i = {c: i for i, c in enumerate(chars)}  # character → index dict

    def __len__(self) -> int:
        # Total token count = network's input embedding / output softmax size.
        return len(self.i2c)

    def encode(self, smiles: str) -> list[int]:
        """Convert a SMILES string to a list of token indices.

        Any character not in the vocabulary falls back to the UNK index so
        the lookup never crashes on unexpected input.
        """
        return [self.c2i.get(ch, self.c2i[UNK]) for ch in smiles]

    def decode(self, indices: list[int]) -> str:
        """Convert token indices back to a SMILES string, stopping at EOS.

        Control tokens are dropped so the result is a clean chemical string.
        """
        chars = []
        for idx in indices:
            ch = self.i2c[idx]
            if ch == EOS:
                break                      # stop decoding at end-of-sequence
            if ch not in (PAD, BOS, UNK):  # skip control tokens, keep real chars
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
    """Tokenised SMILES wrapped as (input, target) pairs for teacher-forcing.

    The language model is trained to predict the next character given all
    previous ones. Each molecule is stored once as the token sequence
    [BOS, ...chars..., EOS]; __getitem__ then slices it into the shifted
    input/target pair the loss needs.
    """

    def __init__(self, smiles_list: list[str], vocab: Vocabulary,
                 max_len: int = 120):
        self.vocab = vocab
        self.data: list[list[int]] = []
        for smi in smiles_list:
            enc = vocab.encode(smi)
            # Keep only sequences short enough to fit BOS + tokens + EOS within
            # max_len (the -2 reserves the two control-token slots). Longer
            # molecules are simply dropped to bound the sequence length.
            if len(enc) <= max_len - 2:
                self.data.append([vocab.bos_idx] + enc + [vocab.eos_idx])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        # Teacher-forcing: the target is the input shifted left by one position.
        #   input  = seq[:-1] = [BOS, c0, c1, ..., c_{n-1}]
        #   target = seq[1:]  = [c0,  c1, c2, ..., EOS]
        # So at each step the model sees one token and must predict the next.
        return seq[:-1], seq[1:]


def collate_fn(batch, pad_idx: int):
    """Pad variable-length sequences in a batch to a common length.

    Molecules differ in length, but a tensor must be rectangular. We pad the
    shorter sequences with pad_idx; the training loss later ignores those
    positions so padding doesn't affect gradients.
    """
    # Unzip the list of (input, target) tuples into two separate tuples.
    inputs, targets = zip(*batch)
    # pad_sequence stacks them into (batch, max_len_in_batch) tensors.
    inputs  = nn.utils.rnn.pad_sequence(inputs,  batch_first=True, padding_value=pad_idx)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    return inputs, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CharRNN(nn.Module):
    """GRU-based character-level language model for SMILES generation.

    Data flow: token ids → Embedding (lookup a dense vector per char)
    → GRU (carries context across the sequence) → Linear (score every
    possible next character). Trained with next-token prediction, then
    sampled one character at a time to produce novel SMILES.
    """

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

        # Embedding: maps each integer token id to a learnable embed_dim vector.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # GRU: the recurrent core. Stacking num_layers deepens the model;
        # batch_first means tensors are shaped (batch, time, features).
        self.rnn = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # Inter-layer dropout is only valid with >1 layer; disable otherwise
            # (PyTorch warns if dropout is set on a single-layer GRU).
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Output head: projects each hidden state to one score (logit) per token.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # B=batch, T=sequence length, E=embed dim, H=hidden dim, V=vocab size.
        embed = self.embedding(x)              # (B, T) ids -> (B, T, E) vectors
        out, hidden = self.rnn(embed, hidden)  # (B, T, H) per-step hidden states
        logits = self.fc(out)                  # (B, T, V) next-char scores
        # `hidden` is returned so generation can carry state across single steps.
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero-initialised hidden state for the start of a sequence."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                           device=device)

    @torch.no_grad()   # generation is inference only — never track gradients
    def generate(
        self,
        vocab: Vocabulary,
        n: int = 1,
        max_len: int = 120,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> list[str]:
        """Generate `n` SMILES strings by autoregressive sampling.

        Autoregressive = the model's own output at step t becomes its input at
        step t+1. `temperature` rescales the logits before sampling:
        temperature > 1 flattens the distribution (more random / novel);
        temperature < 1 sharpens it (more conservative / closer to training).
        """
        device = torch.device(device)
        self.eval()
        self.to(device)

        generated = []
        for _ in range(n):   # generate one molecule per outer-loop iteration
            hidden = self.init_hidden(1, device)
            x = torch.tensor([[vocab.bos_idx]], device=device)  # seed with BOS
            tokens = []

            for _ in range(max_len):   # emit at most max_len characters
                # Run one step; we only need the logits for the latest position.
                logits, hidden = self.forward(x, hidden)
                logits = logits[:, -1, :] / temperature       # apply temperature
                probs  = F.softmax(logits, dim=-1)            # logits -> probabilities
                # Draw the next character id from the categorical distribution.
                next_token = torch.multinomial(probs, 1).item()
                if next_token == vocab.eos_idx:
                    break                                     # molecule finished
                tokens.append(next_token)
                # Feed the sampled token back in as the next step's input.
                x = torch.tensor([[next_token]], device=device)

            # Turn the collected ids back into a SMILES string.
            generated.append(vocab.decode(tokens))
        return generated
