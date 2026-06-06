"""
Chemistry utilities for junction tree VAE.

Provides helper functions for:
- Kekulized SMILES handling (explicit single/double bonds instead of aromatic)
- Ring system detection and bridged-ring merging
- Subgraph extraction and molecule copying

Adapted from: wengong-jin/icml18-jtnn (MIT License)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
import copy


# ---------------------------------------------------------------------------
# Constants used by the original JTVAE (kept for reference)
# ---------------------------------------------------------------------------

MST_MAX_WEIGHT = 100   # maximum spanning tree edge weight threshold
MAX_NCANDS     = 2000  # maximum candidate attachments to evaluate

# Allowed valences per element (used to filter invalid attachment candidates)
VALENCE_MAP = {
    "C": [4], "N": [3, 5], "O": [2], "S": [2, 4, 6],
    "P": [3, 5], "F": [1], "Cl": [1], "Br": [1], "I": [1],
}


# ---------------------------------------------------------------------------
# Core SMILES / mol helpers
# ---------------------------------------------------------------------------

def get_mol(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES and Kekulize the result (explicit bond alternation)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.Kekulize(mol)   # convert aromatic bonds to explicit single/double
    return mol


def get_smiles(mol: Chem.Mol) -> str:
    """Convert a molecule to a Kekulized SMILES string."""
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol: Chem.Mol, kekulize: bool = True) -> Optional[Chem.Mol]:
    """Round-trip a molecule through SMILES to sanitise and normalise it."""
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        return get_mol(smiles)
    except Exception:
        return None   # return None rather than raising for invalid intermediates


# ---------------------------------------------------------------------------
# Ring / clique decomposition
# ---------------------------------------------------------------------------

def get_clique_mol(mol: Chem.Mol, atom_ids: List[int]) -> Chem.Mol:
    """Extract and sanitise the subgraph induced by the given atom indices."""
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atom_ids, kekuleSmiles=True)
    except Exception:
        # Fall back to non-kekulized SMILES for unusual aromatic systems
        smiles = Chem.MolFragmentToSmiles(mol, atom_ids, kekuleSmiles=False)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy.deepcopy(new_mol)   # deepcopy before in-place sanitisation
    Chem.SanitizeMol(new_mol)
    return new_mol


def tree_decomp(mol: Chem.Mol) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Decompose a molecule into cliques (ring systems + acyclic bonds) and a clique graph.

    Algorithm:
      1. Each acyclic bond becomes a 2-atom clique.
      2. Each SSSR ring becomes a clique.
      3. Rings sharing > 2 atoms (bridged/fused systems) are merged into one clique.
      4. Clique adjacency edges are created wherever two cliques share at least one atom.

    Returns:
        cliques : list of atom-index lists, one per clique
        edges   : list of (clique_i, clique_j) adjacency pairs
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []   # trivial case: single atom → single-node tree

    cliques: List[List[int]] = []

    # Add two-atom cliques for every acyclic bond
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    # Add one clique per ring in the Smallest Set of Smallest Rings (SSSR)
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        cliques.append(list(ring))

    # Merge rings that share more than 2 atoms (bridged / spiro systems)
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for j in range(i):
            if len(cliques[j]) <= 2:
                continue
            shared = set(cliques[i]) & set(cliques[j])
            if len(shared) > 2:
                # Absorb clique j into clique i; mark j as empty to remove later
                cliques[i] += [x for x in cliques[j] if x not in cliques[i]]
                cliques[j]  = []

    cliques = [c for c in cliques if c]   # remove empty cliques from merging

    # Build the clique adjacency graph: two cliques are adjacent if they share atoms
    edges: List[Tuple[int, int]] = []
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            if set(cliques[i]) & set(cliques[j]):   # non-empty intersection → edge
                edges.append((i, j))

    # Fused-ring molecules can produce a cyclic clique graph (not a tree).
    # Prune to a BFS spanning tree so the junction tree has no cycles,
    # which prevents infinite recursion in the DFS encoder.
    edges = _bfs_spanning_tree(len(cliques), edges)

    return cliques, edges


def _bfs_spanning_tree(
    n_nodes: int, edges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return a BFS spanning tree of the undirected graph given by edges.

    This guarantees the returned edge list forms a tree (no cycles),
    which is required by the junction-tree DFS encoder.
    """
    if not edges:
        return edges

    # Build adjacency list
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    visited   = {0}
    tree_edges: List[Tuple[int, int]] = []
    queue     = deque([0])

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                tree_edges.append((u, v))
                queue.append(v)

    return tree_edges


# ---------------------------------------------------------------------------
# Atom / molecule copying and attachment helpers
# ---------------------------------------------------------------------------

def atom_equal(a1: Chem.Atom, a2: Chem.Atom) -> bool:
    """Check if two atoms are chemically equivalent (same symbol and formal charge)."""
    return (a1.GetSymbol()       == a2.GetSymbol() and
            a1.GetFormalCharge() == a2.GetFormalCharge())


def bond_match(mol1: Chem.Mol, a1: int, b1: int,
               mol2: Chem.Mol, a2: int, b2: int) -> bool:
    """Return True if the corresponding bonds in mol1 and mol2 have the same type."""
    return (mol1.GetBondBetweenAtoms(a1, b1).GetBondType() ==
            mol2.GetBondBetweenAtoms(a2, b2).GetBondType())


def copy_atom(atom: Chem.Atom) -> Chem.Atom:
    """Create a new RDKit Atom with the same symbol, charge, and explicit H count."""
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
    return new_atom


def copy_edit_mol(mol: Chem.Mol) -> Chem.RWMol:
    """Deep-copy a molecule as an editable RWMol (preserving atoms and bonds)."""
    new_mol = Chem.RWMol(Chem.MolFromSmiles(""))
    for atom in mol.GetAtoms():
        new_mol.AddAtom(copy_atom(atom))
    for bond in mol.GetBonds():
        new_mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                        bond.GetBondType())
    return new_mol


def get_inter_label(mol: Chem.Mol, a_idx: int, inter_atoms: List[int]) -> str:
    """Return a SMILES label that highlights the inter-clique attachment atom."""
    new_mol = copy_edit_mol(mol)
    for atom in new_mol.GetAtoms():
        if atom.GetIdx() not in inter_atoms:
            atom.SetAtomMapNum(0)               # clear map numbers for non-inter atoms
    new_mol.GetAtomWithIdx(a_idx).SetAtomMapNum(1)   # mark the attachment atom
    return Chem.MolToSmiles(new_mol.GetMol())


def enum_attach(cand_mol: Chem.Mol, nei_node_mol: Chem.Mol,
                nei_atom: int, att_atom: int) -> List[Chem.Mol]:
    """Enumerate chemically valid ways to attach a neighbour node to a candidate.

    Returns a list of sanitised product molecules (one per valid bond type).
    """
    cand_atom = cand_mol.GetAtomWithIdx(att_atom)
    nei_a     = nei_node_mol.GetAtomWithIdx(nei_atom)

    if not atom_equal(cand_atom, nei_a):
        return []   # attachment atom types must match

    # Collect existing bond types from the candidate attachment atom
    bond_types = [bond.GetBondType() for bond in cand_atom.GetBonds()]
    results    = []
    for btype in bond_types:
        new_mol = copy_edit_mol(cand_mol)
        try:
            new_mol.AddBond(att_atom, cand_mol.GetNumAtoms(), btype)
            new_mol.AddAtom(copy_atom(nei_a))
            mol = sanitize(new_mol.GetMol())
            if mol:
                results.append(mol)
        except Exception:
            pass   # skip invalid attachment attempts silently
    return results
