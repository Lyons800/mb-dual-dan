"""Bennett et al. 2021 MV reinforcement-prediction-error mushroom-body model.

Single-valence reduction:
    KC      sparse code from PN->KC connectome
    MBON    rate output m = ReLU(w @ kc)
    DAN     d = r - m_hat   (prediction error from valence-opposing MBON feedback;
                              single-valence collapses to d = r - m)
    Update  delta w_i = eta * k_i * d

This is the canonical RPE baseline against which we'll test the active-inference
formulation. Weights are sparsity-masked to nonzero KC->MBON connectome edges only,
so the learnable parameters are exactly the synapses that exist in the real fly.

Reference:
    Bennett, Philippides & Nowotny, "Learning with reinforcement prediction errors
    in a model of the Drosophila mushroom body," Nat Commun 12:2569 (2021).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from brain.connectome.mb_extract import MBSubgraph
from brain.models.shared import KCSparseCoder


@dataclass
class BennettRPE:
    coder: KCSparseCoder
    w_kc_to_mbon: np.ndarray         # [n_mbon, n_kc] plastic readout weights
    mask: np.ndarray                 # [n_mbon, n_kc] {0,1} — restricts plasticity to real edges
    eta: float = 0.025               # learning rate (Bennett Fig 2-3)
    relu_floor: float = 0.0          # ReLU activation floor

    @property
    def n_mbon(self) -> int:
        return self.w_kc_to_mbon.shape[0]

    @property
    def n_kc(self) -> int:
        return self.w_kc_to_mbon.shape[1]

    def mbon_output(self, kc: np.ndarray) -> np.ndarray:
        """Return per-MBON rate. Bennett uses ReLU on a linear readout."""
        return np.maximum(self.relu_floor, self.w_kc_to_mbon @ kc)

    def population_valence(self, mbon: np.ndarray) -> float:
        """Collapse MBON population to scalar valence (mean for single-valence)."""
        return float(mbon.mean())

    def step(self, pn: np.ndarray, reward: float, learn: bool = True) -> dict:
        """One trial: encode odor, read out MBON, compute DAN error, optionally update."""
        kc = self.coder.encode(pn)
        mbon = self.mbon_output(kc)
        m_hat = self.population_valence(mbon)

        # Single-valence DAN: scalar prediction error driving all compartments equally.
        # (Bennett's full MV model has D+ and D- with opposing MBON feedback.)
        dan = reward - m_hat

        if learn:
            delta = self.eta * dan * kc[None, :]
            self.w_kc_to_mbon = np.clip(self.w_kc_to_mbon + delta * self.mask, 0.0, None)

        return {"kc": kc, "mbon": mbon, "m_hat": m_hat, "dan": dan, "n_kc_active": int(kc.sum())}

    @classmethod
    def from_mb(cls, mb: MBSubgraph, eta: float = 0.025, w_init: float = 0.05,
                sparsity: float = 0.05, seed: int = 0) -> "BennettRPE":
        coder = KCSparseCoder.from_mb(mb, sparsity=sparsity, seed=seed)
        g = mb.groups()
        kc_idx, mbon_idx = g["KC"], g["MBON"]
        # Connectome-derived KC->MBON edge mask
        block = mb.W[np.ix_(mbon_idx, kc_idx)].toarray()
        mask = (block > 0).astype(np.float32)
        w0 = mask * w_init
        return cls(coder=coder, w_kc_to_mbon=w0, mask=mask, eta=eta)
