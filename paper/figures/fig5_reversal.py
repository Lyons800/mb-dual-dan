"""Paper Figure 5: reversal-learning across four models on the larval MB connectome.

Layout (2 x 2):
  A — RPE             slow symmetric reversal (single trace)
  B — AIF             cannot reverse (Beta-Bernoulli evidence locks in)
  C — Dual-Valence    fast reversal via parallel opposing memories
  D — POM mechanism   w+ persists while w- grows in parallel during reversal

Style: serif fonts, consistent palette, panel letters, journal-grade layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.aif_agent import AIFAgent
from brain.models.dual_valence import DualValenceMB
from brain.models.rpe_baseline import BennettRPE
from brain.tasks.conditioning import conditioning_trials, make_cs_pair, reversal_trials

OUT = Path(__file__).parent / "fig5_reversal.png"

# Journal-grade defaults
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

PAL = {
    "CS_A":  "#2266aa",
    "CS_B":  "#aa3322",
    "w+":    "#2a9d8f",
    "w-":    "#e76f51",
    "m_hat": "#264653",
}


def run_one_model(model, mb, cs, n_acq=30, n_rev=30):
    log = {"phase": [], "m_a": [], "m_b": [], "m_plus_a": [], "m_minus_a": []}
    for odor, reward, _ in conditioning_trials(cs, n_acq):
        model.step(odor, reward)
        out_a = model.step(cs.cs_plus, 0.0, learn=False)
        out_b = model.step(cs.cs_minus, 0.0, learn=False)
        log["phase"].append("acq")
        log["m_a"].append(out_a["m_hat"])
        log["m_b"].append(out_b["m_hat"])
        log["m_plus_a"].append(out_a.get("m_plus", np.nan))
        log["m_minus_a"].append(out_a.get("m_minus", np.nan))
    for odor, reward, _ in reversal_trials(cs, n_rev):
        model.step(odor, reward)
        out_a = model.step(cs.cs_plus, 0.0, learn=False)
        out_b = model.step(cs.cs_minus, 0.0, learn=False)
        log["phase"].append("rev")
        log["m_a"].append(out_a["m_hat"])
        log["m_b"].append(out_b["m_hat"])
        log["m_plus_a"].append(out_a.get("m_plus", np.nan))
        log["m_minus_a"].append(out_a.get("m_minus", np.nan))
    return {k: np.array(v) for k, v in log.items()}


def main():
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    cs = make_cs_pair(BennettRPE.from_mb(mb, seed=0).coder.n_pn, active_frac=0.1, seed=1)

    rpe_log = run_one_model(BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=0), mb, cs)
    aif_log = run_one_model(AIFAgent.from_mb(mb, sparsity=0.05, seed=0), mb, cs)
    dv_log = run_one_model(DualValenceMB.from_mb(mb, eta=0.005, lambda_baseline=1.0, w_M=1.0,
                                                  w_init=0.05, sparsity=0.05, seed=0), mb, cs)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))

    def panel(ax, log, title):
        t = np.arange(len(log["m_a"]))
        rev_start = int(np.argmax(log["phase"] == "rev"))
        ax.plot(t, log["m_a"], "-", color=PAL["CS_A"], lw=1.5, label=r"$\hat{m}$(CS$_A$)")
        ax.plot(t, log["m_b"], "-", color=PAL["CS_B"], lw=1.5, label=r"$\hat{m}$(CS$_B$)")
        ax.axvspan(rev_start - 0.5, len(t) - 0.5, color="#f4f4f4", zorder=-1)
        ax.text(rev_start + 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
                "reversal", fontsize=7, style="italic", color="#666")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("trial")
        ax.set_ylabel(r"$\hat{m}$")
        ax.legend(loc="lower right")
        ax.set_ylim(-0.3, 1.4)
        ax.axhline(0, color="k", lw=0.4, ls=":")

    panel(axes[0, 0], rpe_log, "A   RPE — slow symmetric reversal")
    panel(axes[0, 1], aif_log, "B   AIF — cannot reverse")
    panel(axes[1, 0], dv_log,  "C   Dual-Valence — fast reversal via POM")

    # Panel D: trace dynamics for Dual-Valence
    ax = axes[1, 1]
    t = np.arange(len(dv_log["m_a"]))
    rev_start = int(np.argmax(dv_log["phase"] == "rev"))
    ax.plot(t, dv_log["m_plus_a"], "-", color=PAL["w+"], lw=1.5, label=r"$m^+$ (appetitive)")
    ax.plot(t, dv_log["m_minus_a"], "-", color=PAL["w-"], lw=1.5, label=r"$m^-$ (aversive)")
    ax.plot(t, dv_log["m_a"], "--", color=PAL["m_hat"], lw=1.2, label=r"$\hat{m} = m^+ - m^-$")
    ax.axvspan(rev_start - 0.5, len(t) - 0.5, color="#f4f4f4", zorder=-1)
    ax.text(rev_start + 0.5, ax.get_ylim()[1] * 0.95, "reversal", fontsize=7, style="italic", color="#666")
    ax.set_title("D   POM mechanism — parallel traces on CS$_A$", loc="left", fontweight="bold")
    ax.set_xlabel("trial")
    ax.set_ylabel("MBON pool output")
    ax.legend(loc="upper right")
    ax.axhline(0, color="k", lw=0.4, ls=":")

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
