"""Dual-DAN compartment-specialised agent.

Combines the two complementary single-channel models from Phases 1–3:

    Channel R (RPE):  Bennett 2021-style reward prediction error on KC->MBON
                      weights. Tracks reward CONTINGENCY. Drives extinction.
                      Biological anchor: appetitive PAM-cluster DANs (DAN-i1,
                      j1, k1) and aversive DL1 DANs (DAN-d1, f1, g1).

    Channel A (AIF):  Bayesian-observer surprisal -log p(KC | model). Tracks
                      perceptual NOVELTY. High on unseen patterns; habituates
                      with exposure. Biological anchor: by analogy to adult
                      PPL1-alpha'3 (Hattori 2017), candidate larval substrate
                      is DAN-c1 — the one larval DAN with no associative
                      phenotype under optogenetic activation (Saumweber 2018,
                      DAN-c1 D2R work eLife 2025).

The hybrid signal follows the canonical "precision-weighted RPE" form
discussed in Schwartenbeck 2015 / Iigaya / Yu-Dayan:

    eta_effective(t) = eta0 * (1 + lambda * normalised_surprisal(t))

Surprisal MODULATES the RPE learning rate. Novel stimuli => high surprisal =>
elevated effective learning rate. Pre-exposed familiar stimuli => low
surprisal => reduced effective rate. This naturally produces LATENT
INHIBITION (Jacob & Waddell 2022): pre-exposure to a CS *without* reward
makes it familiar, lowering surprisal, slowing subsequent CS+US learning —
a signature pure RPE cannot reproduce.

References:
    Bennett, Philippides & Nowotny 2021, Nat Commun  (RPE on KC->MBON)
    Hattori et al. 2017, Cell                         (alpha'3 novelty DAN)
    Saumweber et al. 2018, Nat Commun                 (no-pheno larval DAN-c1)
    Schwartenbeck et al. 2015, J Neurosci             (DA as precision)
    Yu & Dayan 2005, Neuron                           (unexpected uncertainty)
    Jacob & Waddell 2022, Curr Biol                   (latent inhibition in fly)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain.connectome.mb_extract import MBSubgraph
from brain.models.aif_agent import AIFAgent
from brain.models.rpe_baseline import BennettRPE


@dataclass
class DualDANAgent:
    rpe: BennettRPE                       # value/contingency channel
    aif: AIFAgent                         # novelty/perceptual channel
    eta0: float = 0.025                   # base learning rate (matches Bennett)
    lambda_precision: float = 1.5         # surprisal -> eta multiplier strength
    surprisal_baseline: float = 0.0       # tracked baseline for normalisation
    surprisal_scale: float = 1.0          # running std-dev for normalisation
    momentum: float = 0.05                # online stat update rate

    def _normalised_surprisal(self, s: float) -> float:
        """Standardise surprisal against an online baseline so the multiplier
        on eta is comparable across runs and KC sparsities."""
        # online mean / std update
        self.surprisal_baseline = (1 - self.momentum) * self.surprisal_baseline + self.momentum * s
        self.surprisal_scale = max(
            1e-6,
            (1 - self.momentum) * self.surprisal_scale + self.momentum * abs(s - self.surprisal_baseline),
        )
        z = (s - self.surprisal_baseline) / self.surprisal_scale
        # squash to [0, 1+] so eta never becomes negative
        return max(0.0, float(z))

    def step(self, pn: np.ndarray, reward: float, learn: bool = True) -> dict:
        # Run AIF perception (no learning yet so its state at observation time
        # reflects the agent's belief *prior* to seeing this trial's outcome).
        a_observe = self.aif.step(pn, reward, learn=False)
        kc = a_observe["kc"]
        surp_norm = self._normalised_surprisal(a_observe["surprisal"])

        # Precision-weighted RPE: eta scales with normalised surprisal
        original_eta = self.rpe.eta
        self.rpe.eta = self.eta0 * (1.0 + self.lambda_precision * surp_norm)
        r_out = self.rpe.step(pn, reward, learn=learn)
        self.rpe.eta = original_eta

        # AIF channel learns from the same observation
        if learn:
            a_update = self.aif.step(pn, reward, learn=True)
            # we already used a_observe for the readout; a_update updates state
        else:
            a_update = a_observe

        return {
            "kc": kc,
            "m_hat": r_out["m_hat"],
            "dan_rpe": r_out["dan"],
            "dan_aif": a_observe["dan"],         # posterior entropy
            "surprisal": a_observe["surprisal"], # raw -log p(KC)
            "surprisal_norm": surp_norm,
            "eta_effective": self.eta0 * (1.0 + self.lambda_precision * surp_norm),
        }

    @classmethod
    def from_mb(cls, mb: MBSubgraph, eta0: float = 0.025, lambda_precision: float = 1.5,
                w_init: float = 0.05, sparsity: float = 0.05, seed: int = 0) -> "DualDANAgent":
        rpe = BennettRPE.from_mb(mb, eta=eta0, w_init=w_init, sparsity=sparsity, seed=seed)
        aif = AIFAgent.from_mb(mb, sparsity=sparsity, seed=seed)
        return cls(rpe=rpe, aif=aif, eta0=eta0, lambda_precision=lambda_precision)
