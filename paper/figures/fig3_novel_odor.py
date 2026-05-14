"""Paper Figure 3: novel-odor signaling and the AIF / Bayesian-observer channel.

Layout (1 x 3):
  A — |DAN| comparison: familiar vs novel for RPE and AIF
  B — Distribution of AIF DAN responses across novel odors (reaches log 2)
  C — Habituation curve on repeated novel-odor exposure (Hattori signature)

This figure supports: a Bayesian surprisal channel is required for
novelty/familiarity signaling and the Hattori-2017-α'3 habituation profile.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.aif_agent import AIFAgent
from brain.models.rpe_baseline import BennettRPE
from brain.tasks.conditioning import conditioning_trials, make_cs_pair
from brain.tasks.novel_odor import make_probes

OUT = Path(__file__).parent / "fig3_novel_odor.png"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "legend.frameon": False, "figure.dpi": 150,
})

C_RPE = "#88aacc"
C_AIF = "#cc7733"


def trained_pair(mb, seed: int):
    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    aif = AIFAgent.from_mb(mb, sparsity=0.05, seed=seed)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=seed + 1)
    for odor, reward, _ in conditioning_trials(cs, 60):
        rpe.step(odor, reward); aif.step(odor, reward)
    return rpe, aif, cs


def main(n_seeds: int = 10, n_novel: int = 20):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)

    # Collect |DAN| on familiar and novel across seeds
    rpe_fam, rpe_nov = [], []
    aif_fam, aif_nov = [], []
    for s in range(n_seeds):
        rpe, aif, cs = trained_pair(mb, s)
        probes = make_probes(cs, rpe.coder.n_pn, n_novel=n_novel, n_familiar_reps=2, seed=s + 99)
        for pn, _ in probes.familiar:
            rpe_fam.append(abs(rpe.step(pn, 0.0, learn=False)["dan"]))
            aif_fam.append(abs(aif.step(pn, 0.0, learn=False)["dan"]))
        for pn, _ in probes.novel:
            rpe_nov.append(abs(rpe.step(pn, 0.0, learn=False)["dan"]))
            aif_nov.append(abs(aif.step(pn, 0.0, learn=False)["dan"]))

    rpe_fam = np.array(rpe_fam); rpe_nov = np.array(rpe_nov)
    aif_fam = np.array(aif_fam); aif_nov = np.array(aif_nov)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # --- Panel A: bar chart |DAN| means for familiar vs novel
    ax = axes[0]
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, [rpe_fam.mean(), rpe_nov.mean()], w, yerr=[rpe_fam.std(), rpe_nov.std()],
           color=C_RPE, capsize=3, label="RPE")
    ax.bar(x + w/2, [aif_fam.mean(), aif_nov.mean()], w, yerr=[aif_fam.std(), aif_nov.std()],
           color=C_AIF, capsize=3, label="AIF (Bayesian)")
    ax.axhline(np.log(2), color="k", lw=0.5, ls=":", label=r"max possible H = $\log 2$")
    ax.set_xticks(x); ax.set_xticklabels(["familiar (CS$^+$, CS$^-$)", "novel"])
    ax.set_ylabel(r"$|$DAN$|$")
    ax.set_title("A   DAN response to familiar vs novel", loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 0.85)

    # --- Panel B: AIF DAN distribution over novel odors (bimodal: low for class-match, high for ambiguous)
    ax = axes[1]
    ax.hist(aif_nov, bins=15, color=C_AIF, alpha=0.7, edgecolor="white")
    ax.axvline(np.log(2), color="k", lw=1.0, ls="--", label=r"$\log 2$ (max entropy)")
    ax.set_xlabel("AIF DAN on novel odor"); ax.set_ylabel("count")
    ax.set_title(f"B   Distribution across {len(aif_nov)} novel probes", loc="left", fontweight="bold")
    ax.legend(loc="upper left")

    # --- Panel C: habituation across repeated exposure to one novel odor
    # Pick an ambiguous novel odor and re-present 10 times with learning on
    rpe, aif, cs = trained_pair(mb, seed=0)
    probes = make_probes(cs, rpe.coder.n_pn, n_novel=20, seed=42)
    # find a novel that gives max initial DAN
    init_dans = []
    for pn, _ in probes.novel:
        init_dans.append(aif.step(pn, 0.0, learn=False)["dan"])
    best = int(np.argmax(init_dans))
    target_pn = probes.novel[best][0]
    track = []
    for _ in range(12):
        out = aif.step(target_pn, 0.0, learn=True)
        track.append(out["dan"])
    ax = axes[2]
    ax.plot(np.arange(len(track)), track, "o-", color=C_AIF, lw=1.5, ms=5)
    ax.axhline(np.log(2), color="k", lw=0.5, ls=":")
    ax.set_xlabel("exposure"); ax.set_ylabel("AIF DAN")
    ax.set_title("C   Habituation to repeated novel exposure", loc="left", fontweight="bold")
    ax.set_ylim(-0.05, np.log(2) * 1.1)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
