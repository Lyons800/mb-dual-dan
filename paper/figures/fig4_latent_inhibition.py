"""Paper Figure 4: latent inhibition — the precision-weighted RPE signature.

Layout (1 x 3, with insets):
  A — RPE          control and pre-exposed curves overlap (structural no-LI)
  B — Dual-DAN     dramatic separation, tight variance across 20 seeds
  C — lambda sweep  dose-response: LI effect scales monotonically with
                    precision-weighting strength. Pure RPE is the lambda=0 limit.

This is the figure that uniquely identifies precision-weighting as a
necessary architectural ingredient.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.dual_dan import DualDANAgent
from brain.models.rpe_baseline import BennettRPE
from brain.robustness import run_seeds, sweep_param
from brain.tasks.conditioning import make_cs_pair
from brain.tasks.latent_inhibition import make_li_schedule

OUT = Path(__file__).parent / "fig4_latent_inhibition.png"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 150,
})

C_CTRL = "#2266aa"
C_PRE  = "#cc3333"


def li_curve(model_name, preexpose, mb, seed):
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
        out.append(agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"])
    return np.array(out)


def li_auc_diff(lam, seed, mb):
    """AUC difference (control - pre-exposed) at given lambda."""
    def cs_plus_run(preexpose, s):
        agent = DualDANAgent.from_mb(mb, eta0=0.025, lambda_precision=lam,
                                     sparsity=0.05, seed=s)
        n_pn = agent.rpe.coder.n_pn
        cs = make_cs_pair(n_pn, active_frac=0.1, seed=s + 1)
        sched = make_li_schedule(cs, n_pn, n_preexposure=20, n_acquisition=40,
                                 preexpose_cs_plus=preexpose, seed=s + 11)
        for odor, reward, _ in sched.pre_exposure:
            agent.step(odor, reward)
        log = []
        for odor, reward, label in sched.acquisition:
            agent.step(odor, reward)
            if label == "CS+":
                log.append(agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"])
        return np.array(log)

    ctrl = cs_plus_run(False, seed)
    pre = cs_plus_run(True, seed)
    return float(ctrl.mean() - pre.mean())


def main():
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    seeds = list(range(20))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # Panels A, B: RPE and Dual-DAN with seed variance
    for ax, model, title in [
        (axes[0], "RPE", "A   RPE — no LI"),
        (axes[1], "Dual-DAN", "B   Dual-DAN — LI from precision-weighting"),
    ]:
        for preexpose, color, label in [(False, C_CTRL, "control"),
                                          (True,  C_PRE,  "pre-exposed CS$^+$")]:
            sr = run_seeds(lambda s: li_curve(model, preexpose, mb, s), seeds)
            x = np.arange(len(sr.mean))
            ax.plot(x, sr.mean, "-", color=color, lw=1.5, label=label)
            ax.fill_between(x, sr.lo_95, sr.hi_95, color=color, alpha=0.25, linewidth=0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("acquisition trial")
        ax.set_ylabel(r"$\hat{m}$ on CS$^+$")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc="lower right")

    # Panel C: lambda sweep
    lams = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    sweep_seeds = list(range(8))
    sweep = sweep_param(lambda lam, s: li_auc_diff(lam, s, mb), lams, sweep_seeds)
    ax = axes[2]
    ax.errorbar(sweep.param_values, sweep.metric_mean, yerr=sweep.metric_std,
                fmt="o-", color="#264653", lw=1.5, capsize=3, ms=5)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"$\lambda$  (precision-weighting strength)")
    ax.set_ylabel("LI effect  (AUC$_{ctrl}$ − AUC$_{pre}$)")
    ax.set_title(r"C   LI vs $\lambda$ dose-response", loc="left", fontweight="bold")
    ax.set_ylim(-0.05, max(sweep.metric_mean) * 1.15)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
