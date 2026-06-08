"""
Chemistry utilities for junction tree VAE.

Two responsibilities live here:
  1. Decomposition (encode side): break a molecule into clusters / a junction
     tree — ring detection, bridged-ring merging, subgraph extraction.
  2. Assembly (decode side): the enum_assemble family glues decoded clusters
     back into a full molecule, enumerating every valid way two clusters can
     share an atom or fuse along a bond.

Throughout, molecules are kept *kekulized* (aromatic rings written as explicit
alternating single/double bonds) because the assembly logic compares concrete
bond patterns.

Adapted from: wengong-jin/icml18-jtnn (MIT License)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional

from rdkit import Chem
import copy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cap on candidate attachment configurations enumerated per assembly step.
# Keeps generation fast on pathological clusters; value follows the original
# JTVAE (wengong-jin/icml18-jtnn).
MAX_NCAND = 2000


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

    # Step 1: every bond NOT in a ring is its own 2-atom cluster (a chain bond).
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if not bond.IsInRing():
            cliques.append([a1, a2])

    # Step 2: every ring (from RDKit's Smallest Set of Smallest Rings) is a cluster.
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        cliques.append(list(ring))

    # Step 3: merge rings sharing >2 atoms. Two rings fused over a single bond
    # share 2 atoms and stay separate (a normal fused ring); sharing 3+ atoms
    # means a bridged/spiro system that must be treated as ONE rigid cluster.
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2:
            continue
        for j in range(i):
            if len(cliques[j]) <= 2:
                continue
            shared = set(cliques[i]) & set(cliques[j])
            if len(shared) > 2:
                # Absorb clique j into clique i; blank j out, drop it below.
                cliques[i] += [x for x in cliques[j] if x not in cliques[i]]
                cliques[j]  = []

    cliques = [c for c in cliques if c]   # drop the emptied-out merged cliques

    # Step 4: connect two clusters with an edge whenever they share any atom —
    # that shared atom is the junction between them.
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
# Atom / molecule copying helpers
# ---------------------------------------------------------------------------

def atom_equal(a1: Chem.Atom, a2: Chem.Atom) -> bool:
    """Check if two atoms are chemically equivalent (same symbol and formal charge)."""
    return (a1.GetSymbol()       == a2.GetSymbol() and
            a1.GetFormalCharge() == a2.GetFormalCharge())


def ring_bond_equal(b1: Chem.Bond, b2: Chem.Bond, reverse: bool = False) -> bool:
    """Two ring bonds match if their end atoms are pairwise equivalent.

    Bond *type* is intentionally ignored: clusters are kekulized inconsistently
    across the vocabulary, so matching on atom identity (and optionally the
    reversed orientation) is what lets two rings fuse along a shared edge.
    """
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])


def copy_atom(atom: Chem.Atom) -> Chem.Atom:
    """Create a new RDKit Atom copying symbol, formal charge, and atom-map number.

    Explicit hydrogens are deliberately *not* copied — implicit valence is
    recomputed by RDKit during sanitization after the new bonds are added.
    """
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
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


# ---------------------------------------------------------------------------
# Junction-tree graph decoder: enumerate and materialise cluster attachments
#
# Faithful port of the assembly routines from the original JTVAE
# (wengong-jin/icml18-jtnn).  Given a center cluster and its neighbour
# cluster(s), enum_attach lists every chemically plausible way to glue them
# together — by sharing a single atom or by fusing a whole bond (ring fusion) —
# and enum_assemble materialises, sanitizes, and de-duplicates the candidates.
# This is what produces realistic fused-ring scaffolds (indole, quinoline, …)
# that a single inter-cluster bond can never form.
# ---------------------------------------------------------------------------

def enum_attach(ctr_mol: Chem.Mol, nei_node, amap: list, singletons: list) -> list:
    """Enumerate attachment configurations between ``ctr_mol`` and one neighbour.

    Each configuration is an ``amap``: a list of ``(nei_id, ctr_atom_idx,
    nei_atom_idx)`` triples mapping neighbour atoms onto center atoms.  Three
    neighbour shapes are handled: singleton atom, single bond, and ring
    (atom-share *and* bond-share / fusion).
    """
    nei_mol, nei_idx = nei_node.mol, nei_node.nid
    att_confs = []   # collected attachment configurations to return

    # Atoms already claimed by a singleton neighbour are off-limits for reuse.
    black_list = [atom_idx for nei_id, atom_idx, _ in amap if nei_id in singletons]
    ctr_atoms  = [a for a in ctr_mol.GetAtoms() if a.GetIdx() not in black_list]
    ctr_bonds  = list(ctr_mol.GetBonds())

    # --- Case 1: neighbour is a lone atom (no bonds) -----------------------
    # It can attach to any center atom of the same element that is still free.
    if nei_mol.GetNumBonds() == 0:                      # neighbour is a singleton atom
        nei_atom  = nei_mol.GetAtomWithIdx(0)
        used_list = [atom_idx for _, atom_idx, _ in amap]
        for atom in ctr_atoms:
            if atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                att_confs.append(amap + [(nei_idx, atom.GetIdx(), 0)])

    # --- Case 2: neighbour is a single bond (two atoms) --------------------
    # Either end atom can be the shared atom, provided the center atom has
    # enough free valence (spare hydrogens) to accept the bond.
    elif nei_mol.GetNumBonds() == 1:                    # neighbour is a single bond
        bond     = nei_mol.GetBondWithIdx(0)
        bond_val = int(bond.GetBondTypeAsDouble())      # 1=single, 2=double, ...
        b1, b2   = bond.GetBeginAtom(), bond.GetEndAtom()
        for atom in ctr_atoms:
            # A carbon without enough spare hydrogens can't form this bond.
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                continue
            if atom_equal(atom, b1):
                att_confs.append(amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())])
            elif atom_equal(atom, b2):
                att_confs.append(amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())])

    # --- Case 3: neighbour is a ring --------------------------------------
    # Two sub-cases produce the two ways rings can join.
    else:                                               # neighbour is a ring
        # (a) Spiro / single-atom share: the two rings meet at one shared atom.
        for a1 in ctr_atoms:
            for a2 in nei_mol.GetAtoms():
                if atom_equal(a1, a2):
                    if a1.GetAtomicNum() == 6 and \
                       a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                        continue
                    att_confs.append(amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())])
        # (b) Ring fusion: the two rings share a whole bond (two atoms). This is
        # what builds fused systems like indole / quinoline. Both orientations of
        # the shared edge are tried, since the bond can line up either way.
        if ctr_mol.GetNumBonds() > 1:
            for b1 in ctr_bonds:
                for b2 in nei_mol.GetBonds():
                    if ring_bond_equal(b1, b2):
                        att_confs.append(amap + [
                            (nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetBeginAtom().GetIdx()),
                            (nei_idx, b1.GetEndAtom().GetIdx(),   b2.GetEndAtom().GetIdx()),
                        ])
                    if ring_bond_equal(b1, b2, reverse=True):
                        att_confs.append(amap + [
                            (nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetEndAtom().GetIdx()),
                            (nei_idx, b1.GetEndAtom().GetIdx(),   b2.GetBeginAtom().GetIdx()),
                        ])

        # A highly symmetric ring can explode the candidate count; cap it.
        if len(att_confs) > MAX_NCAND:
            att_confs = att_confs[:MAX_NCAND]
    return att_confs


def attach_mols(ctr_mol: Chem.RWMol, neighbors: list, prev_nodes: list,
                nei_amap: dict) -> Chem.RWMol:
    """Splice neighbour atoms/bonds into ``ctr_mol`` according to ``nei_amap``."""
    prev_nids = [node.nid for node in prev_nodes]
    for nei_node in prev_nodes + neighbors:
        nei_id, nei_mol = nei_node.nid, nei_node.mol
        amap = nei_amap[nei_id]
        # Add neighbour atoms that are not already shared with the center
        for atom in nei_mol.GetAtoms():
            if atom.GetIdx() not in amap:
                amap[atom.GetIdx()] = ctr_mol.AddAtom(copy_atom(atom))

        if nei_mol.GetNumBonds() == 0:                  # singleton: carry map number
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_mol.GetAtomWithIdx(amap[0]).SetAtomMapNum(nei_atom.GetAtomMapNum())
        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtom().GetIdx()]
                a2 = amap[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
                elif nei_id in prev_nids:               # parent node wins on conflict
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
    return ctr_mol


def local_attach(ctr_mol: Chem.Mol, neighbors: list, prev_nodes: list,
                 amap_list: list) -> Chem.Mol:
    """Materialise the molecule obtained by applying ``amap_list`` to ``ctr_mol``."""
    ctr_mol  = copy_edit_mol(ctr_mol)
    nei_amap = {nei.nid: {} for nei in prev_nodes + neighbors}
    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom
    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()


def enum_assemble(node, neighbors: list, prev_nodes: list = None,
                  prev_amap: list = None) -> list:
    """Enumerate, sanitize, and de-duplicate ways to assemble ``node``.

    Returns a list of ``(smiles, mol, amap)`` candidate assemblies of the center
    ``node`` with its ``neighbors`` (each a lightweight object exposing ``.mol``
    and ``.nid``).  Rings are tried first for speed.  Bounded by ``MAX_NCAND``.
    """
    prev_nodes = prev_nodes or []
    prev_amap  = prev_amap or []
    all_attach_confs = []   # complete amaps covering every neighbour
    # Single-atom neighbours get special "claimed atom" handling in enum_attach.
    singletons = [nei.nid for nei in neighbors + prev_nodes
                  if nei.mol.GetNumAtoms() == 1]

    def search(cur_amap, depth):
        # Depth-first search that attaches neighbours one at a time. `cur_amap`
        # is the partial attachment built so far; `depth` is which neighbour
        # we're placing next.
        if len(all_attach_confs) > MAX_NCAND:
            return                                   # global safety cap
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)        # all neighbours placed
            return
        nei_node  = neighbors[depth]
        # Enumerate every way to attach this one neighbour given what's placed.
        cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)
        seen      = set()
        viable    = []
        # Keep only attachments that sanitise to a valid molecule, de-duplicated
        # by canonical SMILES, so the recursion doesn't explore equivalent paths.
        for amap in cand_amap:
            cand = local_attach(node.mol, neighbors[:depth + 1], prev_nodes, amap)
            cand = sanitize(cand)
            if cand is None:
                continue
            smi = get_smiles(cand)
            if smi in seen:
                continue
            seen.add(smi)
            viable.append(amap)
        # Recurse into each surviving partial attachment.
        for new_amap in viable:
            search(new_amap, depth + 1)

    search(prev_amap, 0)

    # Materialise every complete attachment into a final (smiles, mol, amap)
    # triple, de-duplicating once more on the fully assembled molecule.
    candidates, seen = [], set()
    for amap in all_attach_confs:
        cand = local_attach(node.mol, neighbors, prev_nodes, amap)
        try:
            # Round-trip through SMILES to canonicalise and validate.
            cand = Chem.MolFromSmiles(Chem.MolToSmiles(cand))
            if cand is None:
                continue
            smi = Chem.MolToSmiles(cand)
            if smi in seen:
                continue
            seen.add(smi)
            Chem.Kekulize(cand)
        except Exception:
            continue   # drop anything RDKit rejects
        candidates.append((smi, cand, amap))
    return candidates
