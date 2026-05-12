"""Novel-odor probing task.

Protocol:
    1. Train both models on CS+/CS- conditioning until they converge.
    2. Probe phase (no learning):
        a. Familiar probe — re-present CS+ and CS- without reward, multiple repeats.
        b. Novel probe   — present N never-seen-before PN patterns once each.
        c. Mixture probe — present CS+/CS- mixtures at varying ratios.

The active-inference prediction is that DAN responses to novel odors are
high (posterior entropy) even though no reward is delivered, whereas the RPE
baseline has nothing to flag the novelty with — DAN tracks (r - m_hat) only,
so unfamiliar stimuli with no reward produce ~0 DAN.

Mixtures of trained odors are the second discriminator: AIF predicts the
DAN signal peaks near 50/50 mixture (maximum posterior entropy), while RPE
DAN interpolates monotonically as m_hat varies linearly with KC overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain.models.shared import sample_odor_pattern
from brain.tasks.conditioning import CSPair


@dataclass
class ProbeSet:
    familiar: list[tuple[np.ndarray, str]]   # (pattern, label)
    novel: list[tuple[np.ndarray, str]]
    mixtures: list[tuple[np.ndarray, float]] # (pattern, mixture_fraction)


def make_probes(cs: CSPair, n_pn: int, n_novel: int = 12, n_familiar_reps: int = 4,
                mixture_steps: int = 9, active_frac: float = 0.1, seed: int = 99) -> ProbeSet:
    rng = np.random.default_rng(seed)
    familiar = []
    for _ in range(n_familiar_reps):
        familiar.append((cs.cs_plus, "CS+"))
        familiar.append((cs.cs_minus, "CS-"))

    novel = []
    for i in range(n_novel):
        pat = sample_odor_pattern(n_pn, active_frac=active_frac, rng=rng)
        novel.append((pat, f"novel-{i}"))

    # Mixtures: linear blend of CS+ and CS- PN patterns at evenly-spaced ratios.
    mixtures = []
    for a in np.linspace(0.0, 1.0, mixture_steps):
        mix = (1.0 - a) * cs.cs_plus + a * cs.cs_minus
        mixtures.append((mix.astype(np.float32), float(a)))

    return ProbeSet(familiar=familiar, novel=novel, mixtures=mixtures)
