"""Phase 3f experiment: reversal learning, Mancini 2019 paradigm.

Acquisition phase:  CS_A -> reward, CS_B -> nothing      (30 trials)
Reversal phase:     CS_A -> nothing, CS_B -> reward      (30 trials)

For each model, track m_hat(CS_A) and m_hat(CS_B) across both phases.
Compare KINETICS of acquisition vs reversal.

Predictions:
    Pure RPE       — symmetric kinetics. Same eta governs unlearning of A
                     and learning of B; reversal takes about the same number
                     of trials as the original acquisition.

    AIF (Bayesian) — once a class identity is locked in by accumulated
                     Beta-Bernoulli evidence, the agent struggles to revise.
                     Expect slow / incomplete reversal.

    Dual           — surprisal is LOW for both patterns during reversal
                     (both are familiar), so eta_effective ~ eta_0. Reversal
                     should be slower than initial acquisition (which had a
                     surprisal boost). This is the OPPOSITE of Mancini's
                     finding (~1 cycle reversal vs 3 cycles acquisition).

    The Mancini-style fast reversal is biologically explained by
    PARALLEL OPPOSING MEMORIES (Felsenberg 2018) — a new aversive trace
    forms IN PARALLEL with the original appetitive trace, rather than
    overwriting it. None of our three current models implement parallel
    opposing memories directly — this experiment exposes that limitation
    and motivates Phase 3i (valence-resolved compartments).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.aif_agent import AIFAgent
from brain.models.dual_dan import DualDANAgent
from brain.models.rpe_baseline import BennettRPE
from brain.tasks.conditioning import conditioning_trials, make_cs_pair, reversal_trials

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def fresh_agents(mb, seed):
    return {
        "RPE":  BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed),
        "AIF":  AIFAgent.from_mb(mb, sparsity=0.05, seed=seed),
        "Dual": DualDANAgent.from_mb(mb, eta0=0.025, lambda_precision=1.5, sparsity=0.05, seed=seed),
    }


def run(mb, cs, seed: int = 0, n_acq: int = 30, n_rev: int = 30):
    agents = fresh_agents(mb, seed)
    out = {name: {"phase": [], "label": [], "m_a": [], "m_b": []} for name in agents}

    # Acquisition: probe CS_A and CS_B on every trial (no-learn probes) for clean curves
    for t, (odor, reward, label) in enumerate(conditioning_trials(cs, n_acq)):
        for name, agent in agents.items():
            # learn on the actual trial
            agent.step(odor, reward, learn=True)
            # probe both patterns without learning
            pa = agent.step(cs.cs_plus, 0.0, learn=False)
            pb = agent.step(cs.cs_minus, 0.0, learn=False)
            out[name]["phase"].append("acq")
            out[name]["label"].append(label)
            out[name]["m_a"].append(pa["m_hat"])
            out[name]["m_b"].append(pb["m_hat"])

    # Reversal: contingencies swap
    for t, (odor, reward, label) in enumerate(reversal_trials(cs, n_rev)):
        for name, agent in agents.items():
            agent.step(odor, reward, learn=True)
            pa = agent.step(cs.cs_plus, 0.0, learn=False)
            pb = agent.step(cs.cs_minus, 0.0, learn=False)
            out[name]["phase"].append("rev")
            out[name]["label"].append(label)
            out[name]["m_a"].append(pa["m_hat"])
            out[name]["m_b"].append(pb["m_hat"])

    return {name: {k: np.array(v) for k, v in d.items()} for name, d in out.items()}


def trials_to_cross(log, threshold: float = 0.5, ascending: bool = True) -> int:
    """Number of trials until the m_hat curve crosses `threshold`."""
    if ascending:
        idx = np.flatnonzero(log >= threshold)
    else:
        idx = np.flatnonzero(log <= threshold)
    return int(idx[0]) if len(idx) > 0 else -1


def main():
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    cs = make_cs_pair(BennettRPE.from_mb(mb, seed=0).coder.n_pn, active_frac=0.1, seed=1)
    results = run(mb, cs, seed=0)

    print(f"{'model':>6} {'phase':>4} {'trials to A>=0.5':>17} {'trials to B<=0.5':>17}")
    print("-" * 55)
    for name, log in results.items():
        acq_mask = log["phase"] == "acq"
        rev_mask = log["phase"] == "rev"
        m_a_acq = log["m_a"][acq_mask]
        m_a_rev = log["m_a"][rev_mask]
        m_b_acq = log["m_b"][acq_mask]
        m_b_rev = log["m_b"][rev_mask]
        # Acquisition kinetics: how fast m_a (the rewarded one) rises above 0.5
        t_a_up = trials_to_cross(m_a_acq, threshold=0.5, ascending=True)
        # Reversal kinetics: how fast m_a (now unrewarded) drops below 0.5
        t_a_down = trials_to_cross(m_a_rev, threshold=0.5, ascending=False)
        print(f"{name:>6} {'acq':>4} {t_a_up:>17d} {'-':>17s}")
        print(f"{name:>6} {'rev':>4} {'-':>17s} {t_a_down:>17d}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (name, log) in zip(axes, results.items()):
        t = np.arange(len(log["m_a"]))
        rev_start = int(np.argmax(log["phase"] == "rev"))
        ax.plot(t, log["m_a"], "o-", color="C0", label="m_hat(CS_A)", ms=3)
        ax.plot(t, log["m_b"], "o-", color="C3", label="m_hat(CS_B)", ms=3)
        ax.axvline(rev_start - 0.5, color="k", ls="--", lw=1)
        ax.text(rev_start, 1.04, "reversal", fontsize=9, ha="left")
        ax.set_title(name); ax.set_xlabel("trial"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("m_hat")
    fig.suptitle("Phase 3f: reversal learning — CS_A was rewarded, CS_B unrewarded; reversal swaps contingency",
                 y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / "07_reversal.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
