"""
Message Passing Network (MPN) for encoding molecular graphs.

The encoder runs Loopy Belief Propagation (LBP) on the molecular graph
and produces a graph-level embedding by summing atom hidden states.

Adapted from: wengong-jin/icml18-jtnn (MIT License)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple

from rdkit import Chem


# ---------------------------------------------------------------------------
# Feature dimensions and atom/bond feature extraction
# ---------------------------------------------------------------------------

# All element symbols encountered in ZINC250k; 'unknown' catches rare elements
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn',
             'unknown']

# Atom feature length: element (23) + degree (6) + charge (5) + chiral (4) + aromatic (1)
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1   # = 39

# Bond feature length: bond type (5) + stereo (6)
BOND_FDIM = 5 + 6   # = 11

# Maximum number of neighbouring bonds per atom (used to size adjacency tensors)
MAX_NB = 6


def onek_encoding_unk(x, allowable_set: list) -> List[int]:
    """One-hot encode x; map unknown values to the last entry of allowable_set."""
    if x not in allowable_set:
        x = allowable_set[-1]  # fall back to 'unknown' / last category
    return [int(x == s) for s in allowable_set]


def atom_features(atom: Chem.Atom) -> torch.Tensor:
    """Build a 39-dimensional feature vector for a single atom."""
    return torch.tensor(
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) +          # element type
        onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + # connectivity
        onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +  # charge
        onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3]) +     # chirality
        [int(atom.GetIsAromatic())],                               # aromaticity flag
        dtype=torch.float,
    )


def bond_features(bond: Chem.Bond) -> torch.Tensor:
    """Build an 11-dimensional feature vector for a single bond."""
    bt     = bond.GetBondType()
    stereo = int(bond.GetStereo())
    return torch.tensor(
        [
            int(bt == Chem.rdchem.BondType.SINGLE),
            int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE),
            int(bt == Chem.rdchem.BondType.AROMATIC),
            int(bond.IsInRing()),
        ] +
        onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5]),  # E/Z / cis / trans
        dtype=torch.float,
    )


# ---------------------------------------------------------------------------
# Batched molecular graph construction
# ---------------------------------------------------------------------------

def mol2graph(mol_list: List[Chem.Mol]) -> Tuple[torch.Tensor, ...]:
    """Convert a list of RDKit Mol objects into batched feature tensors.

    Index 0 in both fatoms and fbonds is reserved as a padding row.
    """
    padding   = torch.zeros(BOND_FDIM)
    atom_feats, bond_feats = [], [padding]   # bond_feats[0] = zero padding row
    atom_scope, bond_scope = [], []
    n_atoms, n_bonds = 1, 1   # start at 1 so index 0 stays as padding

    for mol in mol_list:
        a_start = n_atoms
        b_start = n_bonds

        for atom in mol.GetAtoms():
            atom_feats.append(atom_features(atom))
            n_atoms += 1

        for bond in mol.GetBonds():
            bf = bond_features(bond)
            bond_feats.extend([bf, bf])   # add both directions (a→b and b→a)
            n_bonds += 2

        # Record (start_index, length) so we can slice per-molecule later
        atom_scope.append((a_start, mol.GetNumAtoms()))
        bond_scope.append((b_start, mol.GetNumBonds() * 2))

    # Prepend a zero padding row for atoms too
    atom_feats = torch.stack([torch.zeros(ATOM_FDIM)] + atom_feats, dim=0)
    bond_feats = torch.stack(bond_feats, dim=0)
    return atom_feats, bond_feats, atom_scope, bond_scope


# ---------------------------------------------------------------------------
# MPN encoder
# ---------------------------------------------------------------------------

class MPN(nn.Module):
    """Message Passing Network: encodes each molecule as a fixed-size vector."""

    def __init__(self, hidden_dim: int = 300, depth: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth      = depth

        # GCN-style atom-based message passing:
        self.W_atom = nn.Linear(ATOM_FDIM, hidden_dim)   # atom features → hidden state
        self.W_msg  = nn.Linear(hidden_dim, hidden_dim)  # aggregated neighbour update

    def forward(self, mol_graph) -> torch.Tensor:
        """
        Args:
            mol_graph: tuple (fatoms, anbr, scope)
              fatoms : (N, ATOM_FDIM) atom features, row 0 reserved as padding
              anbr   : (N, max_nb) neighbour atom indices per atom (0 = padding)
              scope  : list of (start_index, num_atoms) per molecule
        Returns:
            mol_vecs: (batch_size, hidden_dim) – one vector per molecule
        """
        fatoms, anbr, scope = mol_graph
        device = next(self.parameters()).device
        fatoms = fatoms.to(device)
        anbr   = anbr.to(device)

        # Mask zeroes out the padding atom (index 0) so padded neighbours,
        # which all point to index 0, contribute nothing to the sum.
        mask    = torch.ones(fatoms.size(0), 1, device=device)
        mask[0] = 0.0

        # Initialise each atom hidden state from its features
        h0 = torch.relu(self.W_atom(fatoms)) * mask   # (N, H)
        h  = h0
        # `depth` rounds of neighbour aggregation with a residual to h0
        for _ in range(self.depth):
            nei = index_select_ND(h, 0, anbr).sum(dim=1)   # sum neighbour states
            h   = torch.relu(h0 + self.W_msg(nei)) * mask

        # Sum-pool atom hidden states into one vector per molecule
        mol_vecs = [h.narrow(0, st, le).sum(0) for st, le in scope]
        return torch.stack(mol_vecs, dim=0)


def index_select_ND(source: torch.Tensor, dim: int,
                    index: torch.Tensor) -> torch.Tensor:
    """Vectorised gather for 2-D index tensors (used in message aggregation)."""
    index_size = index.size()
    suffix_dim = source.size()[1:]
    flat_index = index.view(-1)
    # Clamp prevents out-of-bounds access on padding positions
    flat_index = flat_index.clamp(min=0, max=source.size(0) - 1)
    selected   = source.index_select(dim, flat_index)
    return selected.view(*index_size, *suffix_dim)
