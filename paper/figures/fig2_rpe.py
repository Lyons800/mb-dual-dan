"""Paper Figure 2: RPE acquisition + extinction on the larval MB connectome.

Layout (1 x 3):
  A — Acquisition curve with N=20 seed variance bands
  B — DAN prediction error decay across acquisition + extinction
  C — Extinction kinetics showing signed-weight requirement

The claim this figure supports: signed-weight RPE is necessary and
sufficient for acquisition + extinction phenomena.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.rpe_baseline import BennettRPE
from brain.robustness import run_seeds
from brain.tasks.conditioning import conditioning_trials, extinction_trials, make_cs_pair

OUT = Path(__file__).parent / "fig2_rpe.png"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "legend.frameon": False, "figure.dpi": 150,
})

C_PLUS, C_MINUS = "#2266aa", "#cc3333"


def run_acquisition(mb, seed: int, n_trials: int = 60) -> dict:
    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=seed + 1)
    log = {"label": [], "m_hat": [], "dan": []}
    for odor, reward, label in conditioning_trials(cs, n_trials):
        out = rpe.step(odor, reward)
        log["label"].append(label); log["m_hat"].append(out["m_hat"]); log["dan"].append(out["dan"])
    return {k: np.array(v) for k, v in log.items()}, rpe, cs


def run_acq_then_ext(mb, seed: int, n_acq=60, n_ext=40) -> dict:
    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=seed + 1)
    log = {"phase": [], "label": [], "m_hat": [], "dan": []}
    for odor, reward, label in conditioning_trials(cs, n_acq):
        out = rpe.step(odor, reward)
        log["phase"].append("acq"); log["label"].append(label)
        log["m_hat"].append(out["m_hat"]); log["dan"].append(out["dan"])
    for odor, reward, label in extinction_trials(cs, n_ext):
        out = rpe.step(odor, reward)
        log["phase"].append("ext"); log["label"].append(label)
        log["m_hat"].append(out["m_hat"]); log["dan"].append(out["dan"])
    return {k: np.array(v) for k, v in log.items()}


def main():
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    seeds = list(range(20))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # --- Panel A: acquisition with seed variance, CS+ and CS-
    def cs_plus_curve(s):
        log, _, _ = run_acquisition(mb, s)
        return log["m_hat"][np.array(log["label"]) == "CS+"]
    def cs_minus_curve(s):
        log, _, _ = run_acquisition(mb, s)
        return log["m_hat"][np.array(log["label"]) == "CS-"]
    sr_plus  = run_seeds(cs_plus_curve, seeds)
    sr_minus = run_seeds(cs_minus_curve, seeds)
    ax = axes[0]
    x = np.arange(len(sr_plus.mean))
    ax.plot(x, sr_plus.mean,  "-", color=C_PLUS,  lw=1.5, label=r"CS$^+$")
    ax.fill_between(x, sr_plus.lo_95, sr_plus.hi_95, color=C_PLUS, alpha=0.25, linewidth=0)
    ax.plot(x, sr_minus.mean, "-", color=C_MINUS, lw=1.5, label=r"CS$^-$")
    ax.fill_between(x, sr_minus.lo_95, sr_minus.hi_95, color=C_MINUS, alpha=0.25, linewidth=0)
    ax.set_title("A   RPE acquisition  (N=20 seeds)", loc="left", fontweight="bold")
    ax.set_xlabel("CS trial"); ax.set_ylabel(r"$\hat{m}$")
    ax.legend(loc="lower right"); ax.set_ylim(-0.05, 1.05)

    # --- Panel B: DAN error decay
    def dan_plus(s):
        log, _, _ = run_acquisition(mb, s)
        return log["dan"][np.array(log["label"]) == "CS+"]
    sr_dan = run_seeds(dan_plus, seeds)
    ax = axes[1]
    x = np.arange(len(sr_dan.mean))
    ax.plot(x, sr_dan.mean, "-", color="#264653", lw=1.5)
    ax.fill_between(x, sr_dan.lo_95, sr_dan.hi_95, color="#264653", alpha=0.25, linewidth=0)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_title(r"B   RPE DAN error  ($r - \hat{m}$)", loc="left", fontweight="bold")
    ax.set_xlabel(r"CS$^+$ trial"); ax.set_ylabel("DAN")
    ax.set_ylim(-0.1, 1.0)

    # --- Panel C: acquisition -> extinction
    def get_acq_ext(s):
        log = run_acq_then_ext(mb, s)
        plus_mask = np.array([l.startswith("CS+") for l in log["label"]])
        return log["m_hat"][plus_mask]
    sr_ae = run_seeds(get_acq_ext, seeds)
    ax = axes[2]
    x = np.arange(len(sr_ae.mean))
    rev_start = 30  # n_acq / 2 since CS+ trials are every other
    ax.plot(x, sr_ae.mean, "-", color=C_PLUS, lw=1.5)
    ax.fill_between(x, sr_ae.lo_95, sr_ae.hi_95, color=C_PLUS, alpha=0.25, linewidth=0)
    ax.axvspan(rev_start - 0.5, len(x) - 0.5, color="#f4f4f4", zorder=-1)
    ax.text(rev_start + 0.5, 1.05, "extinction", fontsize=7, style="italic", color="#666")
    ax.set_title(r"C   Acquisition $\to$ extinction  (CS$^+$ only)", loc="left", fontweight="bold")
    ax.set_xlabel(r"CS$^+$ trial"); ax.set_ylabel(r"$\hat{m}$")
    ax.set_ylim(-0.05, 1.15)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
