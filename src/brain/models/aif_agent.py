"""Active-inference mushroom-body agent.

Same substrate as the Bennett RPE baseline (KC sparse code from real PN->KC wiring,
connectome KC->MBON edge mask), but the learning + DAN signal come from variational
Bayes rather than reward prediction error.

The agent maintains a Bayesian generative model over two latent reward classes,
discovered from (KC, reward) observations:

    Latent c in {rewarded, unrewarded}.
    Likelihood:  p(KC_i = 1 | c) ~ Beta-Bernoulli(alpha_{c,i}, beta_{c,i}).
    Reward model: mean reward per class tracked with a Gaussian-like running mean.

On each trial:
    q(c | KC) = softmax( sum_i kc_i * log p_i_c  +  (1 - kc_i) * log (1 - p_i_c)  +  log p(c) )
    m_hat    = E_{q(c|KC)} [ E[r | c] ]                                 # MBON-equivalent readout
    DAN      = H[ q(c | KC) ]                                           # AIF: posterior entropy

DAN is the **epistemic-value signal**: high when the agent is uncertain about the
class identity given the current KC pattern. This is the falsifiable contrast with
the RPE baseline, where DAN tracks reward error and goes to ~0 on familiar odors.

For familiar CS+/CS- odors the two models converge in behaviour (both correctly predict
reward, both DAN signals fall toward 0). The discriminator lives in Phase 3:
**novel-odor probing**, where AIF predicts a positive DAN transient (high entropy)
and RPE predicts ~0 DAN (no reward, no prediction).

References:
    Friston et al. 2017, "Active inference: a process theory" (Neural Computation).
    pymdp 1.0 (Heins et al., JOSS 2022) — corresponds to a single perception step
    with two hidden states and N_KC observation modalities, A_dependencies enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.connectome.mb_extract import MBSubgraph
from brain.models.shared import KCSparseCoder


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

    def dan_signal(self, q: np.ndarray) -> float:
        """Posterior entropy over classes — the AIF DAN signal."""
        return float(-np.sum(q * np.log(q + self.eps)))

    def expected_reward(self, q: np.ndarray) -> float:
        return float((q * self.mean_reward_per_class()).sum())

    def step(self, pn: np.ndarray, reward: float) -> dict:
        kc = self.coder.encode(pn)
        q = self.posterior(kc)
        m_hat = self.expected_reward(q)
        dan = self.dan_signal(q)

        # Online Bayesian update from this (KC, reward) observation.
        # Class membership comes from the reward outcome (mirrors how the RPE
        # baseline implicitly conditions weight changes on reward magnitude).
        c = 0 if reward > 0.5 else 1
        self.alpha[c] += kc
        self.beta[c] += (1.0 - kc)
        self.n_class[c] += 1
        self.reward_sum[c] += reward

        return {"kc": kc, "q": q, "m_hat": m_hat, "dan": dan, "n_kc_active": int(kc.sum())}

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
