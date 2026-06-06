"""
Junction Tree (JT) construction for molecular graphs.

Each molecule is decomposed into a tree of clusters (ring systems and
single bonds).  The tree is rooted at an arbitrary node and serialised
for use as input to the JTVAE encoder.

Adapted from: wengong-jin/icml18-jtnn (MIT License)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import copy
from rdkit import Chem
from rdkit.Chem import AllChem

from models.jtvae.chemutils import get_clique_mol, get_smiles, tree_decomp


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MolTreeNode:
    """One node in the junction tree, representing a ring system or single bond."""

    def __init__(self, smiles: str, clique: List[int] | None = None):
        self.smiles: str      = smiles
        self.mol: Chem.Mol    = Chem.MolFromSmiles(smiles)   # RDKit mol for this cluster

        self.clique: List[int]          = clique or []   # atom indices in the original mol
        self.neighbors: List[MolTreeNode] = []
        self.is_leaf: bool              = True           # becomes False when neighbors are added

        self.nid: int = -1   # node ID assigned during tree construction
        self.wid: int = -1   # vocabulary index, set externally before encoding

    def add_neighbor(self, nb_node: MolTreeNode) -> None:
        self.neighbors.append(nb_node)
        self.is_leaf = False   # a node with any neighbor is not a leaf

    def recover(self, original_mol: Chem.Mol) -> str:
        """Recover the SMILES label of this node in the context of the full molecule."""
        clique = list(self.clique)
        if not self.is_leaf:
            # Include atoms from neighboring nodes to capture attachment context
            for nb in self.neighbors:
                clique.extend(nb.clique)
        clique    = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        return get_smiles(label_mol)

    def assemble(self) -> None:
        """Reorder neighbors so larger clusters come first (helps tree decoding)."""
        # Non-singleton neighbors sorted by size (largest first)
        multi    = sorted([nb for nb in self.neighbors if nb.mol.GetNumAtoms() > 1],
                          key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        # Single-atom clusters (substituents) appended at the end
        singles  = [nb for nb in self.neighbors if nb.mol.GetNumAtoms() == 1]
        self.neighbors = multi + singles


# ---------------------------------------------------------------------------
# Full molecule junction tree
# ---------------------------------------------------------------------------

class MolTree:
    """Build and store the junction tree for a single SMILES molecule."""

    def __init__(self, smiles: str):
        self.smiles: str            = smiles
        self.mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            self.nodes: List[MolTreeNode] = []  # invalid SMILES → empty tree
            return

        try:
            Chem.Kekulize(self.mol)   # convert aromatic bonds to alternating single/double
            self.nodes = self._build_tree()
        except Exception:
            # Skip molecules whose aromatic system RDKit cannot kekulize
            self.nodes = []

    def _build_tree(self) -> List[MolTreeNode]:
        cliques, edges = tree_decomp(self.mol)   # decompose into cliques and adjacency

        # Create one MolTreeNode per clique
        nodes = []
        for c in cliques:
            if len(c) == 1:
                # Single-atom clique: generate SMILES rooted at that atom
                smi = Chem.MolToSmiles(
                    Chem.MolFromSmiles(
                        Chem.MolToSmiles(Chem.RWMol(self.mol), rootedAtAtom=c[0])
                    )
                )
            else:
                cmol = get_clique_mol(self.mol, c)
                smi  = get_smiles(cmol)
            nodes.append(MolTreeNode(smi, c))

        # Wire bidirectional neighbor links from the clique adjacency list
        for i, j in edges:
            nodes[i].add_neighbor(nodes[j])
            nodes[j].add_neighbor(nodes[i])

        # Assign sequential node IDs (1-indexed; 0 is reserved as 'no node')
        for idx, node in enumerate(nodes):
            node.nid = idx + 1

        # Sort each node's neighbor list for deterministic decoding order
        for node in nodes:
            node.assemble()

        return nodes

    def size(self) -> int:
        return len(self.nodes)

    def recover(self) -> None:
        """Attach contextual SMILES labels to every node."""
        for node in self.nodes:
            node.label = node.recover(self.mol)

    def __repr__(self) -> str:
        return f"MolTree({self.smiles}, nodes={self.size()})"


# ---------------------------------------------------------------------------
# Vocabulary for junction-tree clusters
# ---------------------------------------------------------------------------

class Vocab:
    """Bidirectional map between cluster SMILES strings and integer indices."""

    def __init__(self, smiles_list: List[str]):
        self.vocab = smiles_list                                         # index → SMILES
        self.vmap: Dict[str, int] = {s: i for i, s in enumerate(smiles_list)}  # SMILES → index
        self._fp: Dict[str, Optional[object]] = {}   # fingerprint cache (unused here)

    def __getitem__(self, smiles: str) -> int:
        return self.vmap[smiles]

    def get(self, smiles: str, default: int = 0) -> int:
        return self.vmap.get(smiles, default)

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, smiles: str) -> bool:
        return smiles in self.vmap

    def get_smiles(self, idx: int) -> str:
        return self.vocab[idx]

    def get_mol(self, idx: int) -> Chem.Mol:
        """Return the Kekulized RDKit mol for a given vocab index."""
        mol = Chem.MolFromSmiles(self.vocab[idx])
        Chem.Kekulize(mol)
        return mol


def build_vocab(smiles_list: List[str]) -> Vocab:
    """Collect all unique cluster SMILES from a training corpus and build a Vocab."""
    from tqdm import tqdm
    cluster_smiles = set()
    skipped = 0
    for smi in tqdm(smiles_list, desc="Building vocab"):
        try:
            tree = MolTree(smi)
            for node in tree.nodes:
                cluster_smiles.add(node.smiles)
        except Exception:
            skipped += 1   # skip molecules that still fail after kekulize guard
    if skipped:
        print(f"[vocab] Skipped {skipped} molecules during vocab building.")
    # Sort for reproducibility across runs
    return Vocab(sorted(cluster_smiles))
