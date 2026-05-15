"""Multi-seed runners and parameter sweeps for robustness analysis.

These tools let experiments be run across multiple random seeds for
mean +/- std variance bands, and across parameter ranges for sensitivity
analysis (tornado plots and dose-response curves).

Use case: take any of the experiments/0X_*.py protocols and wrap them in
these multi-seed runs so the paper figures have proper variance bands
instead of single-seed curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SeedRun:
    """Result of running an experiment across multiple seeds."""
    curves: np.ndarray         # [n_seeds, n_trials] (or higher rank if multi-output)
    seeds: np.ndarray
    mean: np.ndarray           # [n_trials]
    std: np.ndarray            # [n_trials]
    lo_95: np.ndarray          # 2.5th percentile
    hi_95: np.ndarray          # 97.5th percentile


def run_seeds(experiment_fn: Callable[[int], np.ndarray], seeds: list[int]) -> SeedRun:
    """Run `experiment_fn(seed)` for each seed; return mean + variance bands.

    The function should return a 1-D array of outputs (e.g., m_hat per trial).
    """
    curves = np.array([experiment_fn(s) for s in seeds])
    return SeedRun(
        curves=curves,
        seeds=np.array(seeds),
        mean=curves.mean(axis=0),
        std=curves.std(axis=0),
        lo_95=np.percentile(curves, 2.5, axis=0),
        hi_95=np.percentile(curves, 97.5, axis=0),
    )


@dataclass
class ParamSweep:
    """Result of sweeping a single parameter."""
    param_values: np.ndarray
    metric_mean: np.ndarray
    metric_std: np.ndarray


def sweep_param(experiment_fn: Callable[[float, int], float],
                param_values: list[float],
                seeds: list[int]) -> ParamSweep:
    """Sweep `param_values`; for each value, run across all seeds and
    summarise the scalar metric returned by `experiment_fn(param, seed)`."""
    means, stds = [], []
    for p in param_values:
        vals = np.array([experiment_fn(p, s) for s in seeds])
        means.append(vals.mean())
        stds.append(vals.std())
    return ParamSweep(
        param_values=np.array(param_values),
        metric_mean=np.array(means),
        metric_std=np.array(stds),
    )
