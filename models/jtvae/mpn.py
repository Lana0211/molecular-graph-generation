"""
Message Passing Network (MPN) for encoding molecular graphs.

This is the "graph encoder" half of JTVAE. It looks at the raw atom-bond graph
(not the junction tree) and produces the graph latent z_G. Each atom starts with
a feature vector, then repeatedly mixes in information from its bonded neighbours
("message passing"); after a few rounds every atom's vector summarises its local
chemical environment. Summing those vectors gives one fixed-size embedding per
molecule, regardless of how many atoms it has.

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
    """One-hot encode x; map unknown values to the last entry of allowable_set.

    One-hot encoding = a list of 0s with a single 1 marking which category x is.
    Any value not in the list is bucketed into the final 'unknown' category so
    the feature vector always has the same fixed length.
    """
    if x not in allowable_set:
        x = allowable_set[-1]  # fall back to 'unknown' / last category
    return [int(x == s) for s in allowable_set]


def atom_features(atom: Chem.Atom) -> torch.Tensor:
    """Build a 39-dimensional feature vector for a single atom.

    Concatenates one-hot encodings of the chemical properties that matter for
    how an atom bonds: which element it is, how many neighbours it has, its
    charge, its chirality, and whether it is aromatic.
    """
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

    To process many molecules at once we concatenate all their atoms (and bonds)
    into single big tensors. The `scope` lists remember where each molecule's
    rows start and how many there are, so we can slice them apart again after
    message passing. Index 0 is reserved as an all-zero padding row that neighbour
    lookups point to when an atom has fewer than MAX_NB neighbours.
    """
    padding   = torch.zeros(BOND_FDIM)
    atom_feats, bond_feats = [], [padding]   # bond_feats[0] = zero padding row
    atom_scope, bond_scope = [], []
    n_atoms, n_bonds = 1, 1   # start at 1 so index 0 stays as padding

    for mol in mol_list:
        # Remember where this molecule's atoms/bonds begin in the flat arrays.
        a_start = n_atoms
        b_start = n_bonds

        for atom in mol.GetAtoms():
            atom_feats.append(atom_features(atom))
            n_atoms += 1

        for bond in mol.GetBonds():
            bf = bond_features(bond)
            # A chemical bond is undirected, but message passing is directional,
            # so store it twice — once for each direction (a→b and b→a).
            bond_feats.extend([bf, bf])
            n_bonds += 2

        # Record (start_index, length) so we can slice this molecule out later.
        atom_scope.append((a_start, mol.GetNumAtoms()))
        bond_scope.append((b_start, mol.GetNumBonds() * 2))

    # Stack the Python lists into tensors, prepending the atom padding row.
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
        self.depth      = depth   # number of message-passing rounds (hops)

        # Two learnable linear layers drive the message passing:
        self.W_atom = nn.Linear(ATOM_FDIM, hidden_dim)   # raw atom features → initial hidden state
        self.W_msg  = nn.Linear(hidden_dim, hidden_dim)  # transforms aggregated neighbour messages

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
        # Make sure inputs sit on the same device as the model's weights.
        device = next(self.parameters()).device
        fatoms = fatoms.to(device)
        anbr   = anbr.to(device)

        # Column mask that is 0 for the padding atom (row 0) and 1 elsewhere.
        # Multiplying by it after every layer keeps the padding atom's state at
        # zero, so neighbour sums that include padding pick up nothing.
        mask    = torch.ones(fatoms.size(0), 1, device=device)
        mask[0] = 0.0

        # h0 = each atom's starting hidden state, derived purely from its own
        # features (ReLU adds non-linearity). h is the state we iteratively update.
        h0 = torch.relu(self.W_atom(fatoms)) * mask   # (N, H)
        h  = h0
        # Each round, every atom gathers its neighbours' states, sums them, and
        # folds them back in. The residual `h0 +` keeps the atom's own identity
        # present at every hop. After `depth` rounds, h encodes a depth-hop
        # neighbourhood around each atom.
        for _ in range(self.depth):
            nei = index_select_ND(h, 0, anbr).sum(dim=1)   # sum of neighbour states
            h   = torch.relu(h0 + self.W_msg(nei)) * mask

        # Readout: sum each molecule's atom vectors into a single graph vector.
        # narrow(0, st, le) slices rows [st : st+le] = exactly this molecule.
        mol_vecs = [h.narrow(0, st, le).sum(0) for st, le in scope]
        return torch.stack(mol_vecs, dim=0)


def index_select_ND(source: torch.Tensor, dim: int,
                    index: torch.Tensor) -> torch.Tensor:
    """Vectorised gather for 2-D index tensors (used in message aggregation).

    `index` is (N, max_nb) of neighbour atom ids; this returns (N, max_nb, H) by
    looking up each id's hidden-state row in `source`. Doing it as one batched
    gather (instead of Python loops over atoms) is what makes the encoder fast.
    """
    index_size = index.size()              # remember (N, max_nb)
    suffix_dim = source.size()[1:]         # the H feature dimension
    flat_index = index.view(-1)            # flatten to 1-D for index_select
    # Clamp guards against out-of-range ids (e.g. padding) before the lookup.
    flat_index = flat_index.clamp(min=0, max=source.size(0) - 1)
    selected   = source.index_select(dim, flat_index)
    # Restore the (N, max_nb, H) shape.
    return selected.view(*index_size, *suffix_dim)
