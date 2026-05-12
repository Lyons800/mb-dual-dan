"""Appetitive odor conditioning task.

Two fixed PN activation patterns are presented across trials:
    CS+   paired with reward = +1
    CS-   unrewarded         (reward = 0)

Trials are interleaved (CS+, CS-, CS+, CS-, ...). The model should:
    - Increase MBON response to CS+ across trials (acquisition).
    - Show no change in MBON response to CS-.
    - DAN error to CS+ should decay toward zero as prediction catches up.

This is the bog-standard fly olfactory conditioning paradigm
(Schleyer/Gerber larval protocols, Saumweber 2017 manual).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain.models.shared import sample_odor_pattern


@dataclass
class CSPair:
    cs_plus: np.ndarray   # [n_pn] fixed PN pattern for the rewarded odor
    cs_minus: np.ndarray  # [n_pn] fixed PN pattern for the unrewarded odor


def make_cs_pair(n_pn: int, active_frac: float = 0.1, seed: int = 1) -> CSPair:
    rng = np.random.default_rng(seed)
    return CSPair(
        cs_plus=sample_odor_pattern(n_pn, active_frac=active_frac, rng=rng),
        cs_minus=sample_odor_pattern(n_pn, active_frac=active_frac, rng=rng),
    )


def conditioning_trials(cs: CSPair, n_trials: int, reward_value: float = 1.0):
    """Yield (odor_pattern, reward, label) tuples alternating CS+/CS-."""
    for t in range(n_trials):
        if t % 2 == 0:
            yield cs.cs_plus, reward_value, "CS+"
        else:
            yield cs.cs_minus, 0.0, "CS-"


def extinction_trials(cs: CSPair, n_trials: int):
    """After acquisition, present CS+ WITHOUT reward (extinction) interleaved
    with CS-. Tests whether the RPE rule can unlearn an acquired association
    when the reward contingency reverses, and whether the AIF agent's
    uncertainty rises when prior predictions are violated.
    """
    for t in range(n_trials):
        if t % 2 == 0:
            yield cs.cs_plus, 0.0, "CS+(ext)"
        else:
            yield cs.cs_minus, 0.0, "CS-"
