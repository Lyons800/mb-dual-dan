"""Phase 3d experiment: latent inhibition as the dual-DAN discriminator.

Compare three agents on the LI paradigm:
    1. Pure RPE (Bennett 2021 single-valence on connectome)
    2. Pure AIF (Bayesian observer with posterior entropy DAN)
    3. Dual-DAN (RPE with eta gated by AIF surprisal)

For each: run two groups (pre-exposed CS+ vs control pre-exposure) and
measure the CS+ acquisition slope during the post-pre-exposure phase.
A latent-inhibition effect means pre-exposed CS+ acquires SLOWER than
control. Only the Dual model is expected to show this clearly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.aif_agent import AIFAgent
from brain.models.dual_dan import DualDANAgent
from brain.models.rpe_baseline import BennettRPE
from brain.tasks.conditioning import make_cs_pair
from brain.tasks.latent_inhibition import make_li_schedule

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def fresh_agents(mb, seed):
    return {
        "RPE":  BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed),
        "AIF":  AIFAgent.from_mb(mb, sparsity=0.05, seed=seed),
        "Dual": DualDANAgent.from_mb(mb, eta0=0.025, lambda_precision=1.5,
                                     w_init=0.05, sparsity=0.05, seed=seed),
    }


def run_group(agent, schedule):
    log = []
    for odor, reward, label in schedule.pre_exposure:
        out = agent.step(odor, reward)
        log.append({"phase": "pre", "label": label, "m_hat": out["m_hat"]})
    for odor, reward, label in schedule.acquisition:
        out = agent.step(odor, reward)
        log.append({"phase": "acq", "label": label, "m_hat": out["m_hat"]})
    return log


def cs_plus_acq_curve(log) -> np.ndarray:
    """Extract m_hat on CS+ trials during the acquisition phase only."""
    return np.array([row["m_hat"] for row in log
                     if row["phase"] == "acq" and row["label"] == "CS+"])


def main(seed: int = 0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    cs = make_cs_pair(mb.W.shape[0] if False else BennettRPE.from_mb(mb, seed=seed).coder.n_pn,
                      active_frac=0.1, seed=seed + 1)

    results = {}  # results[(model, group)] = m_hat curve

    for group_name, preexpose in [("pre-exposed", True), ("control", False)]:
        schedule = make_li_schedule(cs, mb.W.shape[0] if False else BennettRPE.from_mb(mb, seed=seed).coder.n_pn,
                                    n_preexposure=20, n_acquisition=40,
                                    preexpose_cs_plus=preexpose, seed=seed + 11)
        agents = fresh_agents(mb, seed)
        for model_name, agent in agents.items():
            log = run_group(agent, schedule)
            curve = cs_plus_acq_curve(log)
            results[(model_name, group_name)] = curve

    # ---- summary stats ----
    print(f"{'model':>6} {'group':>12} {'m_hat[0]':>10} {'m_hat[5]':>10} {'m_hat[-1]':>10} {'slope':>8}")
    print("-" * 60)
    for (model, group), curve in results.items():
        slope = float(curve[5] - curve[0]) if len(curve) >= 6 else float("nan")
        print(f"{model:>6} {group:>12} {curve[0]:>10.3f} {curve[5]:>10.3f} {curve[-1]:>10.3f} {slope:>8.3f}")

    print()
    print("=== Latent inhibition score (slope_control - slope_preexposed) ===")
    for model in ("RPE", "AIF", "Dual"):
        s_ctrl = float(results[(model, "control")][5] - results[(model, "control")][0])
        s_pre = float(results[(model, "pre-exposed")][5] - results[(model, "pre-exposed")][0])
        print(f"  {model:>6}: control_slope={s_ctrl:.3f}, preexposed_slope={s_pre:.3f}, LI_effect={s_ctrl - s_pre:+.3f}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, ["RPE", "AIF", "Dual"]):
        ctrl = results[(model, "control")]
        pre  = results[(model, "pre-exposed")]
        x = np.arange(len(ctrl))
        ax.plot(x, ctrl, "o-", color="C0", label="control")
        ax.plot(x, pre,  "o-", color="C3", label="pre-exposed CS+")
        ax.set_title(f"{model}: CS+ acquisition after pre-exposure")
        ax.set_xlabel("CS+ trial (in acquisition phase)")
        ax.set_ylabel("m_hat")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Phase 3d: latent inhibition — only the Dual model slows CS+ acquisition after pre-exposure",
                 y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / "06_latent_inhibition.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")

    return results


if __name__ == "__main__":
    main()
