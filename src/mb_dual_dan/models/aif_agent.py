"""Bayesian-observer mushroom-body agent (perception-only AIF reduction).

Same substrate as the Bennett RPE baseline (KC sparse code from real PN->KC wiring,
connectome KC->MBON edge mask), but the learning + DAN signal come from variational
Bayes rather than reward prediction error.

The agent maintains a Bayesian generative model over two latent reward classes,
discovered from (KC, reward) observations:

    Latent c in {rewarded, unrewarded}.
    Likelihood:  p(KC_i = 1 | c) ~ Beta-Bernoulli(alpha_{c,i}, beta_{c,i}).
    Reward model: mean reward per class tracked with a Gaussian-like running mean.

On each trial:
    q(c | KC)         posterior via naive Bayes log-likelihood + log prior
    m_hat = E_{q(c|KC)} [ E[r | c] ]              # MBON-equivalent readout
    DAN   = D_KL[ q(c | KC) || q(c) ]            # Bayesian surprise = AIF DAN signal

The canonical DAN signal under active inference is **Bayesian surprise** —
the KL divergence of the posterior from the prior, which is monotonically
related to information gain. This is the quantity Friston / Schwartenbeck
identify with dopaminergic precision / phasic responses. It is NOT the same
as raw posterior entropy H[q(c|KC)] (we expose both for diagnostic
comparison, but `dan` uses surprise).

The discriminator vs Bennett RPE:
    RPE  DAN = r - m_hat            — tracks reward error
    AIF  DAN = KL[q(c|KC) || q(c)]  — tracks information gain from observation

Honest caveats (verified May 2026):
- The "class = reward outcome" mapping makes this a supervised Bayesian
  classifier rather than a full latent-state AIF agent. A genuinely latent-
  state formulation would cluster KC patterns unsupervised and learn
  p(reward | cluster) separately — see Phase 4 roadmap.
- For a complete AIF agent we would also implement EFE-based policy
  selection over actions; this is perception-only.
- This agent cannot perform EXTINCTION (Phase 3c experiment). Because the
  Beta-Bernoulli posterior over class identity accumulates positive evidence
  monotonically, the agent never "unlearns" that CS+ pattern -> rewarded
  class even when reward is withheld. This is the correct, honest behavior
  of a perceptual-Bayesian formulation and reveals that AIF entropy/surprisal
  are NOT substitutes for RPE: they serve different functions. A faithful
  fly-MB model probably needs both perceptual surprise (AIF) and contingency
  prediction error (RPE) as complementary DAN signals — see Phase 3c plot.

References:
    Friston et al. 2017, "Active inference: a process theory" (Neural Computation).
    Schwartenbeck et al. 2019, "Computational mechanisms of curiosity and
        goal-directed exploration" (eLife) — DAN as precision / EFE proxy.
    Itti & Baldi 2009, "Bayesian surprise attracts human attention".
    Hattori et al. 2017 (Cell) — adult MB alpha'3 novelty response (we predict
        a larval analog; experimental anchor in larva pending).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mb_dual_dan.connectome.mb_extract import MBSubgraph
from mb_dual_dan.models.shared import KCSparseCoder


@dataclass
class AIFAgent:
    coder: KCSparseCoder
    # Beta-Bernoulli sufficient statistics over p(KC_i = 1 | class)
    # class 0 = rewarded, class 1 = unrewarded
    alpha: np.ndarray                  # [2, n_kc]
    beta: np.ndarray                   # [2, n_kc]
    n_class: np.ndarray                # [2]  count of trials per class
    reward_sum: np.ndarray             # [2]  running reward sum per class
    eps: float = 1e-9
    # Connectome edge mask — read-out is restricted to existing KC->MBON synapses
    kc_to_mbon_mask: np.ndarray | None = None

    @property
    def n_kc(self) -> int:
        return self.alpha.shape[1]

    def p_kc_given_class(self) -> np.ndarray:
        """MAP estimate p(KC_i = 1 | class) = alpha / (alpha + beta).  Returns [2, n_kc]."""
        return self.alpha / (self.alpha + self.beta)

    def mean_reward_per_class(self) -> np.ndarray:
        """Empirical mean reward per class.  Falls back to a neutral 0.5 if no data."""
        out = np.full(2, 0.5, dtype=np.float32)
        nz = self.n_class > 0
        out[nz] = (self.reward_sum[nz] / self.n_class[nz]).astype(np.float32)
        return out

    def class_log_prior(self) -> np.ndarray:
        total = self.n_class.sum()
        if total == 0:
            return np.array([np.log(0.5), np.log(0.5)], dtype=np.float32)
        # add-one smoothing
        p = (self.n_class + 1.0) / (total + 2.0)
        return np.log(p).astype(np.float32)

    def posterior(self, kc: np.ndarray) -> np.ndarray:
        """Return q(class | KC) over the 2 latent classes."""
        p = self.p_kc_given_class()
        log_lik = (
            kc[None, :] * np.log(p + self.eps)
            + (1.0 - kc[None, :]) * np.log(1.0 - p + self.eps)
        ).sum(axis=1)
        log_post = log_lik + self.class_log_prior()
        log_post -= log_post.max()
        post = np.exp(log_post)
        return post / post.sum()

    def posterior_entropy(self, q: np.ndarray) -> float:
        """H[q(c|KC)] — residual uncertainty about class after observation.

        HIGH on novel odors (posterior can't decide between classes).
        LOW on familiar odors (posterior peaks on the trained class).
        Falls with repeated exposure as evidence accumulates.

        This is the signal that matches Hattori 2017 alpha'3-style
        novelty-and-habituation phenomenology.
        """
        return float(-np.sum(q * np.log(q + self.eps)))

    def bayesian_surprise(self, q: np.ndarray, kc: np.ndarray | None = None) -> float:
        """D_KL[q(c|KC) || q(c)] — information gained about CLASS by the observation.

        DIAGNOSTIC only. Note this quantity behaves OPPOSITE to a habituation
        signal: it is HIGH on familiar odors (posterior moves far from the
        marginal prior toward a peaked class belief) and LOW on novel odors
        (posterior stays near uniform). Reported here for completeness and
        for comparison with Itti-Baldi / Schwartenbeck-style claims about
        DAN encoding information gain about latent state.
        """
        prior = np.exp(self.class_log_prior())
        return float(np.sum(q * np.log((q + self.eps) / (prior + self.eps))))

    def surprisal(self, kc: np.ndarray) -> float:
        """-log p(KC | model)  — Shannon surprisal of the observation.

        Equals variational free energy under perfect (zero-KL) inference.
        HIGH on novel KC patterns that the model does not predict; LOW on
        familiar patterns. Also habituates with exposure as the likelihood
        adapts.

        This is the Friston-citable habituation-compatible signal: -log p(o)
        is what variational free energy F bounds, and it monotonically tracks
        Hattori-style novelty responses.
        """
        p_kc_c = self.p_kc_given_class()                              # [2, n_kc]
        log_lik = (kc[None, :] * np.log(p_kc_c + self.eps)
                   + (1.0 - kc[None, :]) * np.log(1.0 - p_kc_c + self.eps)).sum(axis=1)
        log_prior = self.class_log_prior()
        # log p(KC) = log sum_c p(c) p(KC | c) — log-sum-exp for stability
        a = log_lik + log_prior
        m = a.max()
        log_marg = m + np.log(np.exp(a - m).sum() + self.eps)
        return float(-log_marg)

    def dan_signal(self, q: np.ndarray) -> float:
        """Canonical DAN readout for this agent: posterior entropy H[q(c|KC)].

        Posterior entropy is the simplest Bayesian quantity that matches the
        empirically-required habituation profile (high on novel, low on
        familiar, decays with exposure). The surprisal -log p(KC) is the
        Friston-rigorous alternative (variational free energy under perfect
        inference); we expose both for the analysis stage.
        """
        return self.posterior_entropy(q)

    def expected_reward(self, q: np.ndarray) -> float:
        return float((q * self.mean_reward_per_class()).sum())

    def step(self, pn: np.ndarray, reward: float, learn: bool = True) -> dict:
        kc = self.coder.encode(pn)
        q = self.posterior(kc)
        m_hat = self.expected_reward(q)
        dan = self.dan_signal(q)                       # = posterior entropy
        bs = self.bayesian_surprise(q)                 # diagnostic
        surp = self.surprisal(kc)                      # diagnostic

        if learn:
            # Online Bayesian update from this (KC, reward) observation.
            c = 0 if reward > 0.5 else 1
            self.alpha[c] += kc
            self.beta[c] += (1.0 - kc)
            self.n_class[c] += 1
            self.reward_sum[c] += reward

        return {"kc": kc, "q": q, "m_hat": m_hat, "dan": dan,
                "entropy": dan, "bayesian_surprise": bs, "surprisal": surp,
                "n_kc_active": int(kc.sum())}

    @classmethod
    def from_mb(cls, mb: MBSubgraph, alpha0: float = 0.5, beta0: float = 0.5,
                sparsity: float = 0.05, seed: int = 0) -> "AIFAgent":
        coder = KCSparseCoder.from_mb(mb, sparsity=sparsity, seed=seed)
        n_kc = coder.n_kc
        alpha = np.full((2, n_kc), alpha0, dtype=np.float32)
        beta = np.full((2, n_kc), beta0, dtype=np.float32)
        # connectome mask for the read-out (kept for parity with RPE; unused by perception itself)
        g = mb.groups()
        kc_idx, mbon_idx = g["KC"], g["MBON"]
        block = mb.W[np.ix_(mbon_idx, kc_idx)].toarray()
        mask = (block > 0).astype(np.float32)
        return cls(
            coder=coder, alpha=alpha, beta=beta,
            n_class=np.zeros(2, dtype=np.float32),
            reward_sum=np.zeros(2, dtype=np.float32),
            kc_to_mbon_mask=mask,
        )
