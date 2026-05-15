"""Phase 3g regression tests: robustness analysis tools."""

import numpy as np

from mb_dual_dan.robustness import run_seeds, sweep_param


def test_run_seeds_returns_means_and_bands():
    def expt(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(loc=1.0, scale=0.2, size=10)
    sr = run_seeds(expt, list(range(20)))
    assert sr.curves.shape == (20, 10)
    assert abs(sr.mean.mean() - 1.0) < 0.1
    assert (sr.lo_95 < sr.mean).all()
    assert (sr.hi_95 > sr.mean).all()


def test_sweep_param_recovers_monotonic():
    """Sweeping a parameter that drives a monotonic metric should yield
    monotonic means with non-zero variance across seeds."""
    def expt(param: float, seed: int) -> float:
        rng = np.random.default_rng(seed * 31 + int(1000 * param))
        return param + rng.normal(0, 0.05)
    sweep = sweep_param(expt, [0.0, 0.5, 1.0, 1.5, 2.0], list(range(10)))
    diffs = np.diff(sweep.metric_mean)
    assert (diffs > 0).all(), "sweep should be monotonically increasing"
