"""
Junction Tree Variational Autoencoder (JTVAE).

Architecture summary
--------------------
Encoder:
  1. Tree encoder  : message passing from leaves to root over the junction tree
  2. Graph encoder : MPN over the full molecular graph
  3. Both representations are averaged → μ and log σ² via linear layers

Decoder (teacher-forcing during training):
  1. Tree decoder  : GRU predicts the next junction-tree node at each step
  2. Assembly      : predicted node SMILES are joined to form the output SMILES

Key reference:
  Jin et al., "Junction Tree Variational Autoencoder for Molecular Graph
  Generation", ICML 2018.  https://arxiv.org/abs/1802.04364

Adapted from: wengong-jin/icml18-jtnn (MIT License)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from models.jtvae.mol_tree import MolTree, Vocab, MolTreeNode
from models.jtvae.mpn import MPN, index_select_ND, ATOM_FDIM, BOND_FDIM


# ---------------------------------------------------------------------------
# Iterative DFS helper (avoids Python's recursion limit on deep/cyclic trees)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tree encoder: vectorized GCN-style message passing over the junction tree
# ---------------------------------------------------------------------------

class JTNNEncoder(nn.Module):
    """Encode a junction tree with synchronous, fully-vectorized message passing.

    The original implementation walked the tree node-by-node in Python, doing
    a handful of tiny GPU ops per node.  For a batch of ~160 nodes that meant
    >1000 kernel launches per batch (≈250 ms).  This version instead:
      1. Looks up every node embedding in one batched call.
      2. Runs `depth` rounds of synchronous neighbour aggregation as a few
         large tensor ops (one index_select + one matmul per round).
    Same GCN idea as the graph MPN, ~50x faster (≈5 ms/batch).
    """

    def __init__(self, hidden_dim: int, depth: int, embedding: nn.Embedding):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth      = depth
        self.embedding  = embedding
        self.W_msg      = nn.Linear(hidden_dim, hidden_dim)   # neighbour update

    def encode(self, tree_batch: List[MolTree]) -> Tuple[None, torch.Tensor]:
        """Return (None, root_vecs) where root_vecs is (batch, hidden_dim)."""
        device = next(self.parameters()).device

        # --- Build batched node indices + tree adjacency on CPU ---
        wids      = [0]    # index 0 = padding node
        nbr_lists = [[]]
        roots     = []     # global index of each tree's root node (0 if empty)
        n_nodes   = 1

        for tree in tree_batch:
            if not tree.nodes:
                roots.append(0)
                continue
            local_to_global = {}
            for node in tree.nodes:
                wids.append(node.wid)
                local_to_global[node.nid] = n_nodes
                nbr_lists.append([])
                n_nodes += 1
            for node in tree.nodes:
                gi = local_to_global[node.nid]
                for nb in node.neighbors:
                    nbr_lists[gi].append(local_to_global[nb.nid])
            roots.append(local_to_global[tree.nodes[0].nid])

        # Dynamic neighbour width (tree nodes can have many substituents)
        max_nb = max((len(x) for x in nbr_lists), default=1) or 1
        anbr_cpu = torch.zeros(n_nodes, max_nb, dtype=torch.long)
        for i, nbrs in enumerate(nbr_lists):
            for k, j in enumerate(nbrs[:max_nb]):
                anbr_cpu[i, k] = j

        wids_t = torch.tensor(wids, dtype=torch.long, device=device)
        anbr   = anbr_cpu.to(device)

        # --- Vectorized synchronous message passing ---
        mask    = torch.ones(n_nodes, 1, device=device)
        mask[0] = 0.0                                  # padding node contributes nothing
        h0 = self.embedding(wids_t) * mask             # (n_nodes, H)
        h  = h0
        for _ in range(self.depth):
            nei = index_select_ND(h, 0, anbr).sum(dim=1)   # aggregate neighbours
            h   = torch.relu(h0 + self.W_msg(nei)) * mask

        # --- Gather one root vector per tree ---
        root_idx  = torch.tensor(roots, dtype=torch.long, device=device)
        root_vecs = h.index_select(0, root_idx)        # (batch, H); row 0 = zeros
        return None, root_vecs


# ---------------------------------------------------------------------------
# JTVAE
# ---------------------------------------------------------------------------

class JTVAE(nn.Module):
    """Simplified JTVAE for the CE7076 final project.

    Differences from the original Jin et al. paper:
      - Tree and graph latent spaces are averaged instead of concatenated
      - Tree decoder uses a plain GRU rather than a learned topological decoder
      - Assembly is done by joining cluster SMILES (approximate but fast)

    This trades some reconstruction fidelity for ~3x faster training.
    """

    def __init__(self, vocab: Vocab, hidden_dim: int = 300,
                 latent_dim: int = 56, depth_t: int = 1, depth_g: int = 3):
        super().__init__()
        self.vocab      = vocab
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Shared embedding table for junction-tree cluster vocabulary
        self.embedding = nn.Embedding(len(vocab), hidden_dim)

        self.tree_enc = JTNNEncoder(hidden_dim, depth_t, self.embedding)  # tree encoder
        self.mpn      = MPN(hidden_dim, depth_g)                          # graph encoder

        # Separate μ / log σ² projections for tree and graph branches
        self.T_mean = nn.Linear(hidden_dim, latent_dim)
        self.T_var  = nn.Linear(hidden_dim, latent_dim)
        self.G_mean = nn.Linear(hidden_dim, latent_dim)
        self.G_var  = nn.Linear(hidden_dim, latent_dim)

        # Decoder GRU: at each step, takes (node_embedding ‖ z) and predicts next node
        self.dec_rnn = nn.GRU(hidden_dim + latent_dim, hidden_dim, batch_first=True)
        self.dec_out = nn.Linear(hidden_dim, len(vocab))   # output vocab logits

        # Project latent vector to initial GRU hidden state
        self.z_to_h  = nn.Linear(latent_dim, hidden_dim)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, mol_batch: List[MolTree]):
        """Encode a batch of molecules into (μ, log σ², tree_vecs)."""
        device = next(self.parameters()).device

        _, tree_vecs = self.tree_enc.encode(mol_batch)   # tree branch

        # Graph branch: pass trees (not raw mols) so _build_graph_batch
        # can use pre-computed graph caches when available
        trees_with_mol = [t for t in mol_batch if t.mol is not None]
        if trees_with_mol:
            graph_batch = self._build_graph_batch(trees_with_mol, device)
            graph_vecs  = self.mpn(graph_batch)
        else:
            graph_vecs = torch.zeros(len(mol_batch), self.hidden_dim, device=device)

        # Negative abs ensures log_var ≤ 0, i.e. std ≤ 1 (stabilises training)
        tree_mean    = self.T_mean(tree_vecs)
        tree_log_var = -torch.abs(self.T_var(tree_vecs))
        graph_mean   = self.G_mean(graph_vecs)
        graph_log_var= -torch.abs(self.G_var(graph_vecs))

        # Average the two branches to get the final latent distribution
        z_mean    = (tree_mean    + graph_mean)    / 2
        z_log_var = (tree_log_var + graph_log_var) / 2

        return z_mean, z_log_var, tree_vecs

    def reparametrize(self, mean: torch.Tensor,
                      log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = μ + ε·σ  (ε ~ N(0,1))."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)   # sample noise
            return mean + eps * std
        return mean   # at inference, just use the mean

    # ------------------------------------------------------------------
    # Decoding (teacher-forced, training only)
    # ------------------------------------------------------------------

    def decode_train(self, z: torch.Tensor,
                     mol_batch: List[MolTree]) -> torch.Tensor:
        """Cross-entropy loss for next-node prediction, fully vectorized.

        All node sequences are padded to a common length and run through the
        GRU in a single batched call (instead of one GRU call per tree), then
        padded positions are masked out of the loss with ignore_index=-100.
        """
        device = z.device

        # Collect per-tree node-id sequences; skip empty trees
        seqs    = [[n.wid for n in t.nodes] for t in mol_batch if t.nodes]
        z_idx   = [b for b, t in enumerate(mol_batch) if t.nodes]
        if not seqs:
            return torch.tensor(0.0, device=device, requires_grad=True)

        B       = len(seqs)
        max_len = max(len(s) for s in seqs)

        # Pad token ids to (B, max_len); pad value 0 (masked later)
        tok = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            tok[i, :len(s)] = torch.tensor(s, device=device)

        embs = self.embedding(tok)                         # (B, max_len, H)

        # Broadcast each tree's latent z across its time steps
        z_sel  = z[torch.tensor(z_idx, device=device)]     # (B, L)
        z_step = z_sel.unsqueeze(1).expand(B, max_len, self.latent_dim)
        embs_z = torch.cat([embs, z_step], dim=-1)         # (B, max_len, H+L)

        # Single batched GRU call; initial hidden from latent
        h0 = self.z_to_h(z_sel).unsqueeze(0)               # (1, B, H)
        out, _ = self.dec_rnn(embs_z, h0)                  # (B, max_len, H)
        logits = self.dec_out(out)                         # (B, max_len, V)

        # Labels = next node; last step has no target → ignore.
        # Padded positions are also set to ignore_index (-100).
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            if len(s) > 1:
                labels[i, :len(s) - 1] = torch.tensor(s[1:], device=device)

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, n: int = 1, max_nodes: int = 4,
                 temperature: float = 1.0) -> List[str]:
        """Sample latent vectors from the prior N(0,1) and decode to SMILES."""
        device = next(self.parameters()).device
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)   # sample from prior
        return self._decode_z(z, max_nodes, temperature)

    @torch.no_grad()
    def _decode_z(self, z: torch.Tensor,
                  max_nodes: int = 4,
                  temperature: float = 1.0) -> List[str]:
        """Decode a batch of latent vectors into SMILES strings."""
        device     = z.device
        results    = []

        for b in range(z.size(0)):
            h_rnn  = self.z_to_h(z[b:b+1]).unsqueeze(0)   # (1, 1, H)
            wids   = []
            cur_wid = 0   # seed with vocab[0] (simplest cluster)

            for _ in range(max_nodes):
                cur_t  = torch.tensor([cur_wid], device=device)
                emb    = self.embedding(cur_t)                         # (1, H)
                emb_z  = torch.cat([emb, z[b:b+1]], dim=-1).unsqueeze(0)  # (1,1,H+L)
                out, h_rnn = self.dec_rnn(emb_z, h_rnn)
                logits = self.dec_out(out.squeeze(0)) / temperature    # apply temperature
                probs  = F.softmax(logits, dim=-1)
                next_wid = torch.multinomial(probs, 1).item()   # stochastic sampling

                # Stop if the decoder cycles back to the seed token
                if next_wid == 0 and wids:
                    break
                wids.append(int(next_wid))
                cur_wid = int(next_wid)

            results.append(self._wids_to_smiles(wids))

        return results

    def _wids_to_smiles(self, wids: List[int], max_frags: int = 8) -> str:
        """Assemble a connected molecule from a sequence of vocab cluster indices.

        Instead of joining clusters with '.' (which yields disconnected,
        non-drug-like fragments), this greedily bonds each cluster to the
        growing molecule with a single bond between two atoms that have a free
        valence (an implicit hydrogen).  RDKit re-sanitisation then removes one
        H from each, producing a chemically valid, connected molecule.

        The original JTVAE uses explicit attachment-site matching; this is a
        pragmatic approximation that still yields realistic drug-like sizes.
        """
        from rdkit import Chem
        if not wids:
            return ""

        # Parse cluster SMILES; cap fragment count to keep molecules drug-sized
        mols = []
        for w in wids[:max_frags]:
            m = Chem.MolFromSmiles(self.vocab.get_smiles(w))
            if m is not None:
                mols.append(m)
        if not mols:
            return ""
        if len(mols) == 1:
            return Chem.MolToSmiles(mols[0])

        current = mols[0]
        for frag in mols[1:]:
            current = self._bond_fragments(current, frag) or current

        try:
            Chem.SanitizeMol(current)
            smi = Chem.MolToSmiles(current)
            # Reject if assembly left disconnected components
            return smi if "." not in smi else ""
        except Exception:
            return ""

    @staticmethod
    def _bond_fragments(mol_a, mol_b):
        """Join two molecules with one single bond between free-valence atoms.

        Returns the sanitised combined molecule, or None if no valid bond
        could be formed.
        """
        from rdkit import Chem

        def first_open_atom(mol, lo, hi):
            """Index of an atom in [lo, hi) that has at least one implicit H."""
            for idx in range(lo, hi):
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetTotalNumHs() > 0:
                    return idx
            return None

        combined = Chem.CombineMols(mol_a, mol_b)   # disconnected union
        rw       = Chem.RWMol(combined)
        n_a      = mol_a.GetNumAtoms()

        a1 = first_open_atom(rw, 0, n_a)                       # atom in mol_a
        a2 = first_open_atom(rw, n_a, rw.GetNumAtoms())        # atom in mol_b
        if a1 is None or a2 is None:
            return None   # no free valence to attach

        rw.AddBond(a1, a2, Chem.BondType.SINGLE)
        new_mol = rw.GetMol()
        try:
            Chem.SanitizeMol(new_mol)   # recomputes implicit Hs / valences
            return new_mol
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, mol_batch: List[MolTree],
                beta: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """Compute the β-VAE objective: ELBO = rec_loss + β · KL."""
        z_mean, z_log_var, _ = self.encode(mol_batch)
        z = self.reparametrize(z_mean, z_log_var)

        # KL divergence: D_KL(q(z|x) ‖ p(z)) with p(z) = N(0,I)
        kl      = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        kl_loss = kl.sum(dim=1).mean()   # sum over latent dims, mean over batch

        rec_loss = self.decode_train(z, mol_batch)   # sequence cross-entropy
        loss     = rec_loss + beta * kl_loss

        stats = {"loss": loss.item(), "rec_loss": rec_loss.item(),
                 "kl_loss": kl_loss.item()}
        return loss, stats

    # ------------------------------------------------------------------
    # Helper: build the graph batch tensor tuple for MPN
    # ------------------------------------------------------------------

    def _build_graph_batch(self, trees, device):
        """Build (fatoms, anbr, scope) for the atom-based MPN.

        Accepts a list of MolTree objects.  Uses pre-computed graph caches
        (tree._graph_cache) when available — set by JTVAEDataset.__init__ —
        falling back to on-the-fly computation for trees without a cache.

        All tensors are assembled on CPU and moved to `device` in two calls,
        avoiding the per-atom CPU→GPU overhead that caused 10-minute epochs.
        """
        from models.jtvae.mpn import atom_features, ATOM_FDIM, MAX_NB

        # padding row (index 0) — every batch shares the same zero row
        pad_atom = torch.zeros(ATOM_FDIM)
        all_fatoms = [pad_atom]
        all_anbr   = [torch.zeros(MAX_NB, dtype=torch.long)]
        scope      = []
        n_atoms    = 1   # start at 1 so index 0 stays as padding

        for tree in trees:
            cache = getattr(tree, '_graph_cache', None)
            if cache is not None:
                # Fast path: use pre-computed CPU tensors from dataset init
                mol_fatoms, mol_anbr, n_mol_atoms = cache
                # mol_fatoms[0] is the per-molecule padding row; skip it
                all_fatoms.append(mol_fatoms[1:])   # (n_mol_atoms, ATOM_FDIM)
                # Shift neighbour indices by the current global offset - 1
                # (cache was built with local indices starting at 1)
                shifted = mol_anbr[1:].clone()      # (n_mol_atoms, MAX_NB)
                mask = shifted > 0
                shifted[mask] = shifted[mask] + (n_atoms - 1)
                all_anbr.append(shifted)
                scope.append((n_atoms, n_mol_atoms))
                n_atoms += n_mol_atoms
            else:
                # Slow fallback: compute on-the-fly (molecules without cache)
                mol = tree.mol
                if mol is None:
                    continue
                a_start = n_atoms
                idx_map = {}
                mol_fatoms_list = []
                for atom in mol.GetAtoms():
                    mol_fatoms_list.append(atom_features(atom))
                    idx_map[atom.GetIdx()] = n_atoms
                    n_atoms += 1
                nbr_lists = [[] for _ in mol_fatoms_list]
                for bond in mol.GetBonds():
                    local1 = bond.GetBeginAtomIdx()
                    local2 = bond.GetEndAtomIdx()
                    g1 = idx_map[local1] - a_start
                    g2 = idx_map[local2] - a_start
                    nbr_lists[g1].append(idx_map[local2])
                    nbr_lists[g2].append(idx_map[local1])
                anbr_mol = torch.zeros(len(mol_fatoms_list), MAX_NB, dtype=torch.long)
                for i, nbrs in enumerate(nbr_lists):
                    for k, j in enumerate(nbrs[:MAX_NB]):
                        anbr_mol[i, k] = j
                all_fatoms.append(torch.stack(mol_fatoms_list))
                all_anbr.append(anbr_mol)
                scope.append((a_start, mol.GetNumAtoms()))

        # Concatenate all molecules and move to GPU in two transfers
        fatoms = torch.cat([f if f.dim() == 2 else f.unsqueeze(0)
                            for f in all_fatoms], dim=0).to(device)
        anbr   = torch.cat([a if a.dim() == 2 else a.unsqueeze(0)
                            for a in all_anbr],   dim=0).to(device)

        return fatoms, anbr, scope
