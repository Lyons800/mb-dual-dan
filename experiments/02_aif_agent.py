"""Phase 2: AIF agent on the same CS+/CS- conditioning task as the RPE baseline.

Same substrate (Winding connectome PN->KC wiring), same task, same seed.
Plots a head-to-head comparison: MBON readout (m_hat) and DAN signal,
RPE vs AIF, for CS+ and CS- across trials.

Expected results:
    - Both models acquire CS+ correctly (m_hat -> ~1.0 for CS+).
    - CS- stays low for both.
    - DAN signals look qualitatively different:
        * RPE DAN  = reward - m_hat  -> magnitude decreases as model learns.
        * AIF DAN  = H[q(class|KC)]   -> starts high (uncertain), decreases as
          the agent gathers evidence to disambiguate CS+ from CS-.
    - On familiar odors both DAN signals approach zero (Phase 3 will introduce
      novel odors where AIF DAN spikes back up but RPE DAN does not).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.aif_agent import AIFAgent
from mb_dual_dan.models.rpe_baseline import BennettRPE
from mb_dual_dan.tasks.conditioning import conditioning_trials, make_cs_pair

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_both(n_trials: int = 60, seed: int = 0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    aif = AIFAgent.from_mb(mb, sparsity=0.05, seed=seed)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=seed + 1)

    log = {"trial": [], "label": [],
           "rpe_m": [], "rpe_dan": [],
           "aif_m": [], "aif_dan": []}
    for t, (odor, reward, label) in enumerate(conditioning_trials(cs, n_trials)):
        r_out = rpe.step(odor, reward)
        a_out = aif.step(odor, reward)
        log["trial"].append(t)
        log["label"].append(label)
        log["rpe_m"].append(r_out["m_hat"])
        log["rpe_dan"].append(r_out["dan"])
        log["aif_m"].append(a_out["m_hat"])
        log["aif_dan"].append(a_out["dan"])

    return {k: np.array(v) if k != "label" else np.array(v) for k, v in log.items()}


def plot(log: dict, path: Path) -> None:
    t = log["trial"]
    lbl = log["label"]
    plus = lbl == "CS+"
    minus = lbl == "CS-"

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)

    # MBON readout
    ax = axes[0, 0]
    ax.plot(t[plus], log["rpe_m"][plus], "o-", color="C2", label="CS+")
    ax.plot(t[minus], log["rpe_m"][minus], "o-", color="C3", label="CS-")
    ax.set_title("RPE  —  MBON readout m_hat")
    ax.set_ylabel("m_hat")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t[plus], log["aif_m"][plus], "o-", color="C2", label="CS+")
    ax.plot(t[minus], log["aif_m"][minus], "o-", color="C3", label="CS-")
    ax.set_title("AIF  —  MBON readout m_hat = E[r | KC]")
    ax.legend()
    ax.grid(alpha=0.3)

    # DAN signal — DIFFERENT MEANINGS
    ax = axes[1, 0]
    ax.plot(t[plus], log["rpe_dan"][plus], "o-", color="C2", label="CS+")
    ax.plot(t[minus], log["rpe_dan"][minus], "o-", color="C3", label="CS-")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title("RPE  —  DAN = reward - m_hat  (prediction error)")
    ax.set_ylabel("DAN")
    ax.set_xlabel("trial")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t[plus], log["aif_dan"][plus], "o-", color="C2", label="CS+")
    ax.plot(t[minus], log["aif_dan"][minus], "o-", color="C3", label="CS-")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title("AIF  —  DAN = H[q(class | KC)]  (posterior entropy)")
    ax.set_xlabel("trial")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("RPE vs AIF on larval MB connectome — CS+/CS- conditioning", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")


def main():
    log = run_both(n_trials=60, seed=0)
    plus = log["label"] == "CS+"

    print(f"trials run         : {len(log['trial'])}")
    print(f"RPE m_hat CS+ end  : {log['rpe_m'][plus][-3:].round(3).tolist()}")
    print(f"AIF m_hat CS+ end  : {log['aif_m'][plus][-3:].round(3).tolist()}")
    print(f"RPE DAN   CS+ end  : {log['rpe_dan'][plus][-3:].round(3).tolist()}")
    print(f"AIF DAN   CS+ end  : {log['aif_dan'][plus][-3:].round(3).tolist()}")

    plot(log, RESULTS_DIR / "02_rpe_vs_aif.png")


if __name__ == "__main__":
    main()
