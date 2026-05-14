"""Dual-valence MB model — Bennett 2021 MV formulation, ported to the
larval Drosophila connectome.

This is the Phase 3i upgrade from the single-valence dual-DAN model. It
implements the canonical Bennett-MV equations with parallel appetitive
and aversive memory traces, modulated by cross-valence DAN feedback:

    m+_q = ReLU(w+ @ k_q)               appetitive MBON pool
    m-_q = ReLU(w- @ k_q)               aversive  MBON pool
    m_hat = m+_q - m-_q                 net valence (-> behaviour)

    r+ = max(0,  r)                     reward channel  (non-negative)
    r- = max(0, -r)                     punishment channel (non-negative)

    d+_q = ReLU(r+ + w_M * m-_q)        appetitive DAN  (excited by aversive MBON)
    d-_q = ReLU(r- + w_M * m+_q)        aversive  DAN  (excited by appetitive MBON)

    delta w+ = eta * k * (lambda - d-)  cross-valence plasticity
    delta w- = eta * k * (lambda - d+)

Bennett 2021 (Nat Commun 12:2569) Eqs. 13 & 23. The lambda baseline
prevents weight collapse and sets the operating point where d ~ lambda
at quiescence.

The cross-valence feedback (M+ excites D-, M- excites D+) is the substrate
for Felsenberg 2018's parallel-opposing-memories phenomenology: when CS_A
flips from rewarded to unrewarded, m+ stays high (old trace persists), so
d- = ReLU(0 + w_M * high) fires and writes a NEW aversive trace into w-,
WITHOUT requiring the old w+ to extinguish first. Net m_hat = m+ - m-
collapses to ~0 quickly even though w+ hasn't decayed.

Compartment grouping uses the DAN/MBON valence tables in
brain.connectome.valence. Appetitive compartments contribute to m+ output;
aversive compartments to m-. Plasticity is applied per compartment but
read out as a pooled m+ / m- valence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain.connectome.mb_extract import MBSubgraph
from brain.connectome.valence import dan_valence, mbon_compartment_letter
from brain.models.shared import KCSparseCoder


@dataclass
class DualValenceMB:
    coder: KCSparseCoder
    # Plastic readout weights, one row per MBON, both valences in parallel.
    w_plus: np.ndarray              # [n_mbon, n_kc]  appetitive trace
    w_minus: np.ndarray             # [n_mbon, n_kc]  aversive  trace
    mask: np.ndarray                # [n_mbon, n_kc] connectome edge mask
    is_mbon_appetitive: np.ndarray  # [n_mbon] bool — appetitive-compartment MBON
    is_mbon_aversive: np.ndarray    # [n_mbon] bool — aversive-compartment MBON
    eta: float = 0.025
    w_M: float = 1.0                # cross-valence feedback strength
    lambda_baseline: float = 1.0    # Bennett-MV baseline term (Eq. 23)

    @property
    def n_kc(self) -> int:
        return self.w_plus.shape[1]

    @property
    def n_mbon(self) -> int:
        return self.w_plus.shape[0]

    def mbon_outputs(self, kc: np.ndarray) -> tuple[float, float]:
        """Pool ReLU(w+ @ kc) over appetitive MBONs and ReLU(w- @ kc) over aversive."""
        plus = np.maximum(0.0, self.w_plus @ kc)
        minus = np.maximum(0.0, self.w_minus @ kc)
        m_plus = float(plus[self.is_mbon_appetitive].sum())
        m_minus = float(minus[self.is_mbon_aversive].sum())
        return m_plus, m_minus

    def step(self, pn: np.ndarray, reward: float, learn: bool = True) -> dict:
        kc = self.coder.encode(pn)

        m_plus, m_minus = self.mbon_outputs(kc)
        m_hat = m_plus - m_minus

        r_plus = max(0.0, reward)
        r_minus = max(0.0, -reward)

        # Cross-valence DAN feedback (Bennett Eq. 13)
        d_plus = max(0.0, r_plus + self.w_M * m_minus)
        d_minus = max(0.0, r_minus + self.w_M * m_plus)

        if learn:
            # Bennett VS_lambda update (Eq. 23). Cross-valence: w+ updated by d-
            kc_outer = kc[None, :]
            delta_plus = self.eta * kc_outer * (self.lambda_baseline - d_minus)
            delta_minus = self.eta * kc_outer * (self.lambda_baseline - d_plus)
            # apply per-MBON, restricted to appetitive/aversive populations and connectome mask
            self.w_plus  = np.clip(self.w_plus  + delta_plus  * self.mask * self.is_mbon_appetitive[:, None], 0.0, None)
            self.w_minus = np.clip(self.w_minus + delta_minus * self.mask * self.is_mbon_aversive[:, None],  0.0, None)

        return {
            "kc": kc,
            "m_plus": m_plus, "m_minus": m_minus, "m_hat": m_hat,
            "d_plus": d_plus, "d_minus": d_minus,
            "r_plus": r_plus, "r_minus": r_minus,
        }

    @classmethod
    def from_mb(cls, mb: MBSubgraph, eta: float = 0.025,
                lambda_baseline: float = 1.0, w_M: float = 1.0,
                w_init: float = 0.05, sparsity: float = 0.05,
                seed: int = 0) -> "DualValenceMB":
        coder = KCSparseCoder.from_mb(mb, sparsity=sparsity, seed=seed)
        g = mb.groups()
        kc_idx, mbon_idx = g["KC"], g["MBON"]
        # connectome edge mask: only learn where wires exist
        block = mb.W[np.ix_(mbon_idx, kc_idx)].toarray()
        mask = (block > 0).astype(np.float32)

        # Per-MBON valence: based on which compartment letter it lives in,
        # cross-referenced to its DAN's functional valence.
        is_app = np.zeros(len(mbon_idx), dtype=bool)
        is_ave = np.zeros(len(mbon_idx), dtype=bool)
        for k, idx in enumerate(mbon_idx):
            letter = mbon_compartment_letter(mb.subtype[idx])
            if letter is None:
                continue
            dan_label = f"DAN-{letter}1"
            v = dan_valence(dan_label)
            if v == "appetitive":
                is_app[k] = True
            elif v == "aversive":
                is_ave[k] = True
            # MBONs in calyx (compartment a, b) or other no-DAN regions stay neutral

        w_plus  = mask * w_init
        w_minus = mask * w_init
        return cls(
            coder=coder, w_plus=w_plus, w_minus=w_minus, mask=mask,
            is_mbon_appetitive=is_app, is_mbon_aversive=is_ave,
            eta=eta, w_M=w_M, lambda_baseline=lambda_baseline,
        )

    def valence_summary(self) -> dict:
        return {
            "n_mbon_appetitive": int(self.is_mbon_appetitive.sum()),
            "n_mbon_aversive":   int(self.is_mbon_aversive.sum()),
            "n_mbon_neutral":    int((~self.is_mbon_appetitive & ~self.is_mbon_aversive).sum()),
            "n_mbon_total":      self.n_mbon,
        }
