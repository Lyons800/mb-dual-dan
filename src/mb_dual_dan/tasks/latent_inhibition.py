"""Latent inhibition task — the classical discriminator for RPE vs hybrid models.

Protocol (Jacob & Waddell 2022 style, adapted for our minimal setting):

    Group: pre-exposed   1) Pre-exposure  N_pre trials of CS+ alone (no reward)
                         2) Acquisition   N_acq trials of CS+/CS- (CS+ paired)
    Group: control       1) Pre-exposure  N_pre trials of a DIFFERENT odor alone
                         2) Acquisition   N_acq trials of CS+/CS- (CS+ paired)

Behaviour:
    Pure RPE  — no mechanism to slow CS+ acquisition based on prior exposure.
                Both groups acquire at the same rate.
    AIF       — pre-exposure habituates surprisal for CS+ in the pre-exposed
                group; under the dual model this lowers eta_effective on CS+
                trials in the acquisition phase, slowing learning.
    Dual      — predicts a clear LI effect: pre-exposed group acquires SLOWER.

The clean discriminator: acquisition slope on CS+ in the pre-exposed group
vs control group, under each model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mb_dual_dan.models.shared import sample_odor_pattern
from mb_dual_dan.tasks.conditioning import CSPair


@dataclass
class LISchedule:
    """A latent-inhibition trial schedule for one group."""
    pre_exposure: list[tuple[np.ndarray, float, str]]  # (odor, reward, label)
    acquisition: list[tuple[np.ndarray, float, str]]


def make_li_schedule(cs: CSPair, n_pn: int,
                     n_preexposure: int = 20,
                     n_acquisition: int = 40,
                     preexpose_cs_plus: bool = True,
                     seed: int = 7) -> LISchedule:
    """Generate a pre-exposure -> acquisition schedule.

    If `preexpose_cs_plus=True` (pre-exposed group), the CS+ odor is shown
    repeatedly without reward in the pre-exposure phase. Otherwise an
    unrelated control odor is shown (control group).
    """
    rng = np.random.default_rng(seed)
    pre = []
    if preexpose_cs_plus:
        target = cs.cs_plus
    else:
        target = sample_odor_pattern(n_pn, active_frac=0.1, rng=rng)
    for _ in range(n_preexposure):
        pre.append((target, 0.0, "pre"))

    # Standard alternating CS+/CS- acquisition phase
    acq = []
    for t in range(n_acquisition):
        if t % 2 == 0:
            acq.append((cs.cs_plus, 1.0, "CS+"))
        else:
            acq.append((cs.cs_minus, 0.0, "CS-"))
    return LISchedule(pre_exposure=pre, acquisition=acq)
