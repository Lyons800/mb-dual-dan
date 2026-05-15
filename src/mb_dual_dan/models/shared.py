"""Shared building blocks used by both the RPE baseline and the AIF agent.

Everything that doesn't differ between the two models lives here:
- KC sparse coding from real PN->KC connectome wiring
- Odor encoding into PN activation patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from mb_dual_dan.connectome.mb_extract import MBSubgraph


def _normalize_rows(W: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize so each postsynaptic neuron receives unit total input weight."""
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    D_inv = sp.diags(1.0 / row_sums)
    return (D_inv @ W).tocsr()


@dataclass
class KCSparseCoder:
    """Maps a PN activation pattern to a sparse KC code, using real PN->KC wiring.

    The PN->KC weight matrix comes from Winding's connectome; sparsity is enforced
    by a winner-take-all top-k operation (Litwin-Kumar 2017, ~5% of KCs active).
    """
    W_pn_to_kc: sp.csr_matrix       # [n_kc, n_pn]
    sparsity: float = 0.05          # fraction of KCs active per odor
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    @property
    def n_pn(self) -> int:
        return self.W_pn_to_kc.shape[1]

    @property
    def n_kc(self) -> int:
        return self.W_pn_to_kc.shape[0]

    @property
    def k_active(self) -> int:
        return max(1, int(round(self.sparsity * self.n_kc)))

    def encode(self, pn: np.ndarray) -> np.ndarray:
        """pn: [n_pn] continuous activations -> [n_kc] binary sparse code."""
        drive = self.W_pn_to_kc @ pn
        k = self.k_active
        if k >= self.n_kc:
            return (drive > 0).astype(np.float32)
        # top-k winner-take-all
        idx = np.argpartition(-drive, k - 1)[:k]
        out = np.zeros(self.n_kc, dtype=np.float32)
        out[idx] = 1.0
        return out

    @classmethod
    def from_mb(cls, mb: MBSubgraph, sparsity: float = 0.05, seed: int = 0) -> "KCSparseCoder":
        g = mb.groups()
        kc_idx, pn_idx = g["KC"], g["PN"]
        # mb.W has convention W[post, pre]; PN->KC means post=KC, pre=PN
        W = mb.W[np.ix_(kc_idx, pn_idx)].tocsr()
        W = _normalize_rows(W)
        return cls(W_pn_to_kc=W, sparsity=sparsity, rng=np.random.default_rng(seed))


def sample_odor_pattern(n_pn: int, active_frac: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random sparse PN activation pattern representing one 'odor'.

    Each odor activates a random subset of PNs with uniform [0, 1] strength.
    Larval olfaction has ~21 ORNs feeding ~28 PNs uniglomerularly; we abstract this
    by sampling directly at the PN level with a 10% active rate.
    """
    if rng is None:
        rng = np.random.default_rng()
    pn = np.zeros(n_pn, dtype=np.float32)
    n_active = max(1, int(round(active_frac * n_pn)))
    active = rng.choice(n_pn, size=n_active, replace=False)
    pn[active] = rng.uniform(0.5, 1.0, size=n_active).astype(np.float32)
    return pn
