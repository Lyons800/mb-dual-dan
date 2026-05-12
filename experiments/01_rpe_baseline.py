"""Phase 1: train the Bennett 2021 RPE baseline on a CS+/CS- conditioning task.

Substrate: real larval Drosophila MB connectome (Winding 2023). The KC sparse code
comes from real PN->KC wiring; KC->MBON weights are plastic but restricted to the
KC->MBON edge mask from the connectome.

Expected behaviour:
    - MBON response to CS+ rises over trials (acquisition).
    - MBON response to CS- stays flat.
    - DAN error on CS+ decays toward zero as the readout learns.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.rpe_baseline import BennettRPE
from brain.tasks.conditioning import conditioning_trials, make_cs_pair

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run(n_trials: int = 60, seed: int = 0) -> dict:
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    model = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)

    cs = make_cs_pair(model.coder.n_pn, active_frac=0.1, seed=seed + 1)

    log = {"trial": [], "label": [], "m_hat": [], "dan": [], "n_kc_active": []}
    for t, (odor, reward, label) in enumerate(conditioning_trials(cs, n_trials)):
        out = model.step(odor, reward)
        log["trial"].append(t)
        log["label"].append(label)
        log["m_hat"].append(out["m_hat"])
        log["dan"].append(out["dan"])
        log["n_kc_active"].append(out["n_kc_active"])

    return {k: np.array(v) if k != "label" else v for k, v in log.items()}


def plot(log: dict, path: Path) -> None:
    trials = log["trial"]
    labels = np.array(log["label"])
    m_hat = log["m_hat"]
    dan = log["dan"]

    cs_plus_mask = labels == "CS+"
    cs_minus_mask = labels == "CS-"

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(trials[cs_plus_mask], m_hat[cs_plus_mask], "o-", label="CS+", color="C2")
    axes[0].plot(trials[cs_minus_mask], m_hat[cs_minus_mask], "o-", label="CS-", color="C3")
    axes[0].set_ylabel("MBON pop. mean")
    axes[0].set_title("Bennett RPE on larval MB connectome")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(trials[cs_plus_mask], dan[cs_plus_mask], "o-", label="CS+ (DAN error)", color="C2")
    axes[1].plot(trials[cs_minus_mask], dan[cs_minus_mask], "o-", label="CS- (DAN error)", color="C3")
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_ylabel("DAN  (r - m_hat)")
    axes[1].set_xlabel("trial")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def main():
    log = run(n_trials=60, seed=0)

    cs_plus_mask = np.array(log["label"]) == "CS+"
    m_plus = log["m_hat"][cs_plus_mask]
    dan_plus = log["dan"][cs_plus_mask]
    print(f"trials run        : {len(log['trial'])}")
    print(f"KCs active / odor : {int(log['n_kc_active'].mean())}")
    print(f"m_hat CS+ first 3 : {m_plus[:3].round(3).tolist()}")
    print(f"m_hat CS+ last  3 : {m_plus[-3:].round(3).tolist()}")
    print(f"DAN   CS+ first 3 : {dan_plus[:3].round(3).tolist()}")
    print(f"DAN   CS+ last  3 : {dan_plus[-3:].round(3).tolist()}")

    plot(log, RESULTS_DIR / "01_rpe_acquisition.png")


if __name__ == "__main__":
    main()
