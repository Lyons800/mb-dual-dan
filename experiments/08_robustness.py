"""Phase 3g: robustness analysis.

Three deliverables:
  1. Multi-seed variance bands on the headline LI plot (20 seeds).
  2. Lambda sensitivity sweep — how LI effect scales with the precision-
     weighting strength `lambda_precision` in the Dual-DAN model.
  3. Eta tuning for Dual-Valence to match Mancini's quantitative 3:1
     acquisition-vs-reversal ratio (acquisition ~3 cycles, reversal ~1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.dual_dan import DualDANAgent
from mb_dual_dan.models.dual_valence import DualValenceMB
from mb_dual_dan.models.rpe_baseline import BennettRPE
from mb_dual_dan.robustness import run_seeds, sweep_param
from mb_dual_dan.tasks.conditioning import conditioning_trials, make_cs_pair, reversal_trials
from mb_dual_dan.tasks.latent_inhibition import make_li_schedule

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------- multi-seed LI variance ----------

def li_curve_for_seed(model_name: str, preexpose: bool, mb, seed: int) -> np.ndarray:
    """Return the CS+ m_hat curve in the acquisition phase, after pre-exposure."""
    if model_name == "RPE":
        agent = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    elif model_name == "Dual-DAN":
        agent = DualDANAgent.from_mb(mb, eta0=0.025, lambda_precision=1.5,
                                     sparsity=0.05, seed=seed)
    else:
        raise ValueError(model_name)
    n_pn = agent.coder.n_pn if hasattr(agent, "coder") else agent.rpe.coder.n_pn
    cs = make_cs_pair(n_pn, active_frac=0.1, seed=seed + 1)
    sched = make_li_schedule(cs, n_pn, n_preexposure=20, n_acquisition=40,
                             preexpose_cs_plus=preexpose, seed=seed + 11)
    for odor, reward, _ in sched.pre_exposure:
        agent.step(odor, reward)
    out = []
    for odor, reward, label in sched.acquisition:
        agent.step(odor, reward)
        # probe CS+ each trial without learning
        out.append(agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"])
    return np.array(out)


def li_multiseed_plot(mb, seeds, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, ["RPE", "Dual-DAN"]):
        for preexpose, color, label in [(False, "C0", "control"), (True, "C3", "pre-exposed CS+")]:
            sr = run_seeds(lambda s: li_curve_for_seed(model, preexpose, mb, s), seeds)
            x = np.arange(len(sr.mean))
            ax.plot(x, sr.mean, "-", color=color, label=label)
            ax.fill_between(x, sr.lo_95, sr.hi_95, color=color, alpha=0.2)
        ax.set_title(f"{model}  (N={len(seeds)} seeds)")
        ax.set_xlabel("acquisition trial")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("m_hat on CS+")
    fig.suptitle("Phase 3g: latent inhibition with seed-variance bands", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")


# ---------- lambda sweep on dual-DAN LI strength ----------

def li_effect_at_lambda(lam: float, seed: int, mb) -> float:
    """Return LI effect = AUC(control) - AUC(pre-exposed) over acquisition.

    A positive value means control acquires CS+ faster (the canonical LI
    signature). AUC integrates the difference across all acquisition trials,
    so saturating curves don't artificially zero the metric.
    """
    def cs_plus_curve(preexpose: bool, s: int) -> np.ndarray:
        agent = DualDANAgent.from_mb(mb, eta0=0.025, lambda_precision=lam,
                                     sparsity=0.05, seed=s)
        n_pn = agent.rpe.coder.n_pn
        cs = make_cs_pair(n_pn, active_frac=0.1, seed=s + 1)
        sched = make_li_schedule(cs, n_pn, n_preexposure=20, n_acquisition=40,
                                 preexpose_cs_plus=preexpose, seed=s + 11)
        for odor, reward, _ in sched.pre_exposure:
            agent.step(odor, reward)
        plus_log = []
        for odor, reward, label in sched.acquisition:
            agent.step(odor, reward)
            if label == "CS+":
                plus_log.append(agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"])
        return np.array(plus_log)

    ctrl = cs_plus_curve(False, seed)
    pre = cs_plus_curve(True, seed)
    return float(ctrl.mean() - pre.mean())


def lambda_sweep_plot(mb, lams, seeds, path):
    sweep = sweep_param(lambda lam, s: li_effect_at_lambda(lam, s, mb),
                        list(lams), list(seeds))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sweep.param_values, sweep.metric_mean, yerr=sweep.metric_std,
                fmt="o-", capsize=4)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel(r"$\lambda$ (precision-weighting strength)")
    ax.set_ylabel("LI effect  (control_slope - preexposed_slope)")
    ax.set_title(f"Phase 3g: LI scales monotonically with lambda  (N={len(seeds)} seeds per point)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")
    return sweep


# ---------- eta tuning for Mancini 3:1 ratio ----------

def reversal_kinetics(eta: float, seed: int, mb) -> tuple[int, int]:
    """Return (trials_to_acq, trials_to_rev) for the dual-valence model.

    Uses fractional-of-peak thresholds: acquisition is when m_hat first
    reaches 50% of its eventual acquisition-phase peak; reversal is when
    m_hat first drops to 50% of that peak (the half-life crossing).
    Mancini's data: acq ~3 cycles, rev ~1 cycle => target ratio ~3:1.
    """
    agent = DualValenceMB.from_mb(mb, eta=eta, lambda_baseline=1.0, w_M=1.0,
                                  w_init=0.05, sparsity=0.05, seed=seed)
    n_pn = agent.coder.n_pn
    cs = make_cs_pair(n_pn, active_frac=0.1, seed=seed + 1)

    # acquisition phase
    acq_curve = []
    for odor, reward, _ in conditioning_trials(cs, 60):
        agent.step(odor, reward)
        acq_curve.append(agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"])
    acq_curve = np.array(acq_curve)
    peak = float(acq_curve.max())
    if peak <= 0.05:
        return -1, -1
    half = peak * 0.5
    t_acq_idx = np.flatnonzero(acq_curve >= half)
    t_acq = int(t_acq_idx[0]) if len(t_acq_idx) else -1

    # reversal phase — m_hat should drop from peak back through half
    rev_curve = []
    for odor, reward, _ in reversal_trials(cs, 60):
        agent.step(odor, reward)
        rev_curve.append(agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"])
    rev_curve = np.array(rev_curve)
    t_rev_idx = np.flatnonzero(rev_curve <= half)
    t_rev = int(t_rev_idx[0]) if len(t_rev_idx) else -1
    return t_acq, t_rev


def eta_tuning_plot(mb, etas, seeds, path):
    acq_means, rev_means, ratios = [], [], []
    acq_stds, rev_stds = [], []
    for eta in etas:
        acqs, revs = [], []
        for s in seeds:
            a, r = reversal_kinetics(eta, s, mb)
            if a >= 0 and r >= 0:
                acqs.append(a); revs.append(r)
        if acqs and revs:
            acq_means.append(np.mean(acqs)); acq_stds.append(np.std(acqs))
            rev_means.append(np.mean(revs)); rev_stds.append(np.std(revs))
            ratios.append(np.mean(acqs) / max(np.mean(revs), 1e-3))
        else:
            acq_means.append(np.nan); rev_means.append(np.nan); ratios.append(np.nan)
            acq_stds.append(0); rev_stds.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.errorbar(etas, acq_means, yerr=acq_stds, fmt="o-", color="C0", label="acquisition")
    ax.errorbar(etas, rev_means, yerr=rev_stds, fmt="s-", color="C3", label="reversal")
    ax.set_xscale("log"); ax.set_xlabel(r"$\eta$ (learning rate)"); ax.set_ylabel("trials to threshold")
    ax.set_title("Dual-Valence kinetics vs eta")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(etas, ratios, "o-", color="C2")
    ax.axhline(3.0, color="k", ls="--", label="Mancini 2019 ratio (3:1)")
    ax.set_xscale("log"); ax.set_xlabel(r"$\eta$"); ax.set_ylabel("acq / rev ratio")
    ax.set_title("Target: ratio ~3")
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"Phase 3g: tuning eta to match Mancini's 3:1 acquisition-vs-reversal ratio  (N={len(seeds)} seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")
    return list(zip(etas, acq_means, rev_means, ratios))


def main():
    c = load_winding()
    mb = extract_mb(c, include_pns=True)

    print("=== 1. LI multi-seed variance ===")
    seeds = list(range(20))
    li_multiseed_plot(mb, seeds, RESULTS_DIR / "08_li_seed_variance.png")

    print("\n=== 2. Lambda sweep on Dual-DAN LI ===")
    lams = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    sweep_seeds = list(range(8))    # fewer seeds because each is a full LI run
    lambda_sweep_plot(mb, lams, sweep_seeds, RESULTS_DIR / "08_lambda_sweep.png")

    print("\n=== 3. Eta tuning for Dual-Valence Mancini ratio ===")
    etas = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.025]
    eta_seeds = list(range(5))
    table = eta_tuning_plot(mb, etas, eta_seeds, RESULTS_DIR / "08_eta_tuning.png")
    print(f"\n{'eta':>8} {'acq':>8} {'rev':>8} {'ratio':>8}")
    for eta, a, r, ratio in table:
        print(f"{eta:>8.4f} {a:>8.2f} {r:>8.2f} {ratio:>8.2f}")


if __name__ == "__main__":
    main()
