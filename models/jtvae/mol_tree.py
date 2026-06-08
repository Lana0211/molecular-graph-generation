"""
Junction Tree (JT) construction for molecular graphs.

Core idea of JTVAE: instead of generating a molecule atom-by-atom (error-prone),
collapse it into a *tree of chemical building blocks*. Each cluster node is a
ring system or a single bond; edges connect clusters that share atoms. Because
the building blocks are valid by construction, generating the tree first and
then gluing the blocks together yields chemically valid molecules.

This file turns a SMILES string into that tree (MolTree) and maintains the
cluster vocabulary (Vocab) the model decodes into.

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
        """Recover the SMILES label of this node *in context* of the full molecule.

        A cluster on its own is ambiguous — e.g. a benzene ring looks the same
        regardless of what hangs off it. To learn how clusters attach, the label
        is the sub-molecule covering this cluster plus all neighbouring clusters'
        atoms, so the attachment points are visible.
        """
        clique = list(self.clique)
        if not self.is_leaf:
            # Pull in neighbouring clusters' atoms to capture attachment context.
            for nb in self.neighbors:
                clique.extend(nb.clique)
        clique    = list(set(clique))          # dedupe shared atoms
        label_mol = get_clique_mol(original_mol, clique)   # carve out that sub-mol
        return get_smiles(label_mol)

    def assemble(self) -> None:
        """Reorder neighbors so larger clusters come first (helps tree decoding).

        Decoding attaches neighbours one at a time; placing the big ring systems
        before single-atom substituents gives a stable, deterministic order that
        the decoder can rely on.
        """
        # Multi-atom neighbours (rings / bonds), largest first.
        multi    = sorted([nb for nb in self.neighbors if nb.mol.GetNumAtoms() > 1],
                          key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        # Single-atom clusters (e.g. substituent atoms) go last.
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
            # Kekulization rewrites aromatic rings as explicit alternating
            # single/double bonds. The decomposition logic needs this concrete
            # bond pattern; some exotic aromatic systems can't be kekulized.
            Chem.Kekulize(self.mol)
            self.nodes = self._build_tree()
        except Exception:
            # Skip molecules whose aromatic system RDKit cannot kekulize.
            self.nodes = []

    def _build_tree(self) -> List[MolTreeNode]:
        # Step 1: decompose the molecule into clusters (cliques) plus the edges
        # describing which clusters share atoms. tree_decomp lives in chemutils.
        cliques, edges = tree_decomp(self.mol)

        # Step 2: create one MolTreeNode per clique, labelled by its SMILES.
        nodes = []
        for c in cliques:
            if len(c) == 1:
                # Single-atom clique: canonicalise the SMILES rooted at that atom
                # so identical single atoms always get the same label.
                smi = Chem.MolToSmiles(
                    Chem.MolFromSmiles(
                        Chem.MolToSmiles(Chem.RWMol(self.mol), rootedAtAtom=c[0])
                    )
                )
            else:
                # Multi-atom clique: extract the sub-molecule and read its SMILES.
                cmol = get_clique_mol(self.mol, c)
                smi  = get_smiles(cmol)
            nodes.append(MolTreeNode(smi, c))

        # Step 3: turn the edge list into bidirectional neighbour links so the
        # tree can be traversed from any node.
        for i, j in edges:
            nodes[i].add_neighbor(nodes[j])
            nodes[j].add_neighbor(nodes[i])

        # Step 4: assign 1-indexed node IDs (id 0 is reserved to mean "no node",
        # used as a padding / stop marker elsewhere).
        for idx, node in enumerate(nodes):
            node.nid = idx + 1

        # Step 5: canonicalise each neighbour ordering for deterministic decoding.
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
    """Collect all unique cluster SMILES from a training corpus and build a Vocab.

    The model's tree decoder is essentially a classifier over this vocabulary,
    so the vocabulary must contain every cluster the model might ever emit. We
    decompose each training molecule and union together all the cluster labels.
    """
    from tqdm import tqdm
    cluster_smiles = set()   # a set automatically de-duplicates repeated clusters
    skipped = 0
    for smi in tqdm(smiles_list, desc="Building vocab"):
        try:
            tree = MolTree(smi)
            for node in tree.nodes:
                cluster_smiles.add(node.smiles)
        except Exception:
            skipped += 1   # skip molecules that still fail after the kekulize guard
    if skipped:
        print(f"[vocab] Skipped {skipped} molecules during vocab building.")
    # Sort so the index assignment is identical across runs (reproducibility).
    return Vocab(sorted(cluster_smiles))
