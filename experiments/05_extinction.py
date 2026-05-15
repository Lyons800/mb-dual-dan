"""Phase 3c rigor experiment: extinction.

After CS+/CS- acquisition, present CS+ without reward. A faithful RPE rule
should extinguish (m_hat -> 0) — this requires signed weights, which is why
we removed the w >= 0 clip in the rigor pass. The AIF agent's posterior over
class identity will adapt as the reward contingency changes.

Failure mode this guards against: if the RPE model can only INCREASE weights
(w clamped to >= 0), extinction is impossible and the comparison to AIF is
unfair — AIF would 'beat' RPE only because RPE cannot represent negative
predictions, not because of any genuine theoretical advantage.

This experiment runs:
    Phase A — acquisition: 60 trials of CS+/CS-, reward on CS+
    Phase B — extinction:  40 trials of CS+/CS-, NO reward on CS+
And plots m_hat and DAN across both phases for both models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.aif_agent import AIFAgent
from mb_dual_dan.models.rpe_baseline import BennettRPE
from mb_dual_dan.tasks.conditioning import conditioning_trials, extinction_trials, make_cs_pair

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run(n_acq: int = 60, n_ext: int = 40, seed: int = 0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    aif = AIFAgent.from_mb(mb, sparsity=0.05, seed=seed)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=seed + 1)

    log = {"trial": [], "phase": [], "label": [],
           "rpe_m": [], "rpe_dan": [],
           "aif_m": [], "aif_dan": [], "aif_surp": []}

    t = 0
    for odor, reward, label in conditioning_trials(cs, n_acq):
        r = rpe.step(odor, reward); a = aif.step(odor, reward)
        log["trial"].append(t); log["phase"].append("acq"); log["label"].append(label)
        log["rpe_m"].append(r["m_hat"]); log["rpe_dan"].append(r["dan"])
        log["aif_m"].append(a["m_hat"]); log["aif_dan"].append(a["dan"])
        log["aif_surp"].append(a["surprisal"])
        t += 1

    for odor, reward, label in extinction_trials(cs, n_ext):
        r = rpe.step(odor, reward); a = aif.step(odor, reward)
        log["trial"].append(t); log["phase"].append("ext"); log["label"].append(label)
        log["rpe_m"].append(r["m_hat"]); log["rpe_dan"].append(r["dan"])
        log["aif_m"].append(a["m_hat"]); log["aif_dan"].append(a["dan"])
        log["aif_surp"].append(a["surprisal"])
        t += 1

    return {k: np.array(v) for k, v in log.items()}


def plot(log, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    t = log["trial"]
    is_plus = np.array([l.startswith("CS+") for l in log["label"]])
    is_minus = np.array([l == "CS-" for l in log["label"]])
    ext_start = int(np.argmax(log["phase"] == "ext"))

    ax = axes[0]
    ax.plot(t[is_plus], log["rpe_m"][is_plus], "o-", color="C0", label="RPE  CS+", ms=3)
    ax.plot(t[is_plus], log["aif_m"][is_plus], "o-", color="C1", label="AIF  CS+", ms=3)
    ax.plot(t[is_minus], log["rpe_m"][is_minus], ".", color="C0", alpha=0.5, label="RPE  CS-")
    ax.plot(t[is_minus], log["aif_m"][is_minus], ".", color="C1", alpha=0.5, label="AIF  CS-")
    ax.axvline(ext_start - 0.5, color="k", ls="--", lw=1)
    ax.set_ylabel("m_hat"); ax.set_title("Acquisition -> Extinction"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax.text(ext_start, 1.02, "extinction begins", fontsize=9, ha="left", color="k")

    ax = axes[1]
    ax.plot(t[is_plus], log["rpe_dan"][is_plus], "o-", color="C0", label="RPE  CS+", ms=3)
    ax.plot(t[is_minus], log["rpe_dan"][is_minus], ".", color="C0", alpha=0.5, label="RPE  CS-")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axvline(ext_start - 0.5, color="k", ls="--", lw=1)
    ax.set_ylabel("RPE DAN (= r - m_hat)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t[is_plus], log["aif_dan"][is_plus], "o-", color="C1", label="AIF DAN (H[q])", ms=3)
    ax.plot(t[is_plus], log["aif_surp"][is_plus], "x-", color="C3", label="AIF surprisal", ms=3, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axvline(ext_start - 0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("trial"); ax.set_ylabel("AIF DAN"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Phase 3c: extinction probe (RPE with signed weights, AIF posterior dynamics)", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")


def main():
    log = run()
    ext_mask = log["phase"] == "ext"
    plus_mask = np.array([l.startswith("CS+") for l in log["label"]])
    plus_ext = ext_mask & plus_mask

    rpe_m_ext = log["rpe_m"][plus_ext]
    aif_m_ext = log["aif_m"][plus_ext]
    print(f"=== extinction phase, CS+ trials ===")
    print(f"  RPE m_hat at extinction start : {rpe_m_ext[0]:.3f}")
    print(f"  RPE m_hat at extinction end   : {rpe_m_ext[-1]:.3f}")
    print(f"  AIF m_hat at extinction start : {aif_m_ext[0]:.3f}")
    print(f"  AIF m_hat at extinction end   : {aif_m_ext[-1]:.3f}")
    print(f"  -> RPE extinguished by {rpe_m_ext[0] - rpe_m_ext[-1]:.3f}")
    print(f"  -> AIF extinguished by {aif_m_ext[0] - aif_m_ext[-1]:.3f}")

    plot(log, RESULTS_DIR / "05_extinction.png")


if __name__ == "__main__":
    main()
