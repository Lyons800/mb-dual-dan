"""Phase 3 — the discriminating experiment.

Train RPE and AIF on CS+/CS- conditioning, then probe with:
    1. Familiar odors (CS+/CS-) without reward — both models should be calm.
    2. Novel odors never seen during training — AIF DAN should spike; RPE flat.
    3. Mixtures of CS+ and CS- across the full ratio space — AIF DAN should
       peak near 0.5 (maximum entropy); RPE DAN should vary monotonically.

If the headline AIF prediction holds, the novel-probe panel of the output
plot will show distinctly elevated DAN for AIF compared to RPE.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.aif_agent import AIFAgent
from brain.models.rpe_baseline import BennettRPE
from brain.tasks.conditioning import conditioning_trials, make_cs_pair
from brain.tasks.novel_odor import make_probes

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def train(model, cs, n_trials: int) -> None:
    for odor, reward, _ in conditioning_trials(cs, n_trials):
        model.step(odor, reward, learn=True)


def probe(model, patterns: list[tuple[np.ndarray, object]]) -> dict:
    out = {"label": [], "m_hat": [], "dan": []}
    for pn, label in patterns:
        r = model.step(pn, reward=0.0, learn=False)   # no reward, no update
        out["label"].append(label)
        out["m_hat"].append(r["m_hat"])
        out["dan"].append(r["dan"])
    return {k: np.array(v) if k != "label" else v for k, v in out.items()}


def main(n_train: int = 60, seed: int = 0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    aif = AIFAgent.from_mb(mb, sparsity=0.05, seed=seed)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=seed + 1)

    train(rpe, cs, n_train)
    train(aif, cs, n_train)

    probes = make_probes(cs, rpe.coder.n_pn, n_novel=12, n_familiar_reps=4, mixture_steps=11, seed=seed + 42)

    rpe_fam = probe(rpe, probes.familiar)
    aif_fam = probe(aif, probes.familiar)
    rpe_nov = probe(rpe, probes.novel)
    aif_nov = probe(aif, probes.novel)
    rpe_mix = probe(rpe, probes.mixtures)
    aif_mix = probe(aif, probes.mixtures)

    # ---- summary stats ----
    print("=== Familiar probe (no reward) ===")
    print(f"  RPE DAN  mean: {np.mean(np.abs(rpe_fam['dan'])):.4f}")
    print(f"  AIF DAN  mean: {np.mean(np.abs(aif_fam['dan'])):.4f}")
    print()
    print("=== Novel probe (no reward) ===")
    print(f"  RPE DAN  mean: {np.mean(np.abs(rpe_nov['dan'])):.4f}   max: {np.max(np.abs(rpe_nov['dan'])):.4f}")
    print(f"  AIF DAN  mean: {np.mean(np.abs(aif_nov['dan'])):.4f}   max: {np.max(np.abs(aif_nov['dan'])):.4f}")
    print(f"  AIF/RPE  ratio: {np.mean(np.abs(aif_nov['dan'])) / max(np.mean(np.abs(rpe_nov['dan'])), 1e-6):.2f}x")
    print()
    print("=== Mixture probe (no reward) ===")
    print(f"  RPE DAN at mix 0.5: {rpe_mix['dan'][len(rpe_mix['dan'])//2]:.4f}")
    print(f"  AIF DAN at mix 0.5: {aif_mix['dan'][len(aif_mix['dan'])//2]:.4f}")
    print(f"  AIF DAN argmax frac: {probes.mixtures[int(np.argmax(aif_mix['dan']))][1]:.2f}")

    # ---- plotting ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))

    def _bar(ax, dans_rpe, dans_aif, labels, title):
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, np.abs(dans_rpe), w, label="RPE |DAN|", color="C0")
        ax.bar(x + w/2, np.abs(dans_aif), w, label="AIF |DAN|", color="C1")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    _bar(axes[0, 0], rpe_fam["dan"], aif_fam["dan"], rpe_fam["label"], "Familiar probe")
    _bar(axes[0, 1], rpe_nov["dan"], aif_nov["dan"], [str(l) for l in rpe_nov["label"]], "Novel-odor probe")

    # Mixtures
    fracs = [m[1] for m in probes.mixtures]
    ax = axes[0, 2]
    ax.plot(fracs, np.abs(rpe_mix["dan"]), "o-", color="C0", label="RPE |DAN|")
    ax.plot(fracs, np.abs(aif_mix["dan"]), "o-", color="C1", label="AIF |DAN|")
    ax.set_xlabel("mix fraction (0=CS+, 1=CS-)"); ax.set_ylabel("|DAN|")
    ax.set_title("Mixture probe (DAN vs blend)"); ax.legend(); ax.grid(alpha=0.3)

    # Bottom row: m_hat readouts on same probes
    ax = axes[1, 0]
    ax.bar(np.arange(len(rpe_fam["label"])) - 0.2, rpe_fam["m_hat"], 0.4, label="RPE m_hat", color="C0")
    ax.bar(np.arange(len(aif_fam["label"])) + 0.2, aif_fam["m_hat"], 0.4, label="AIF m_hat", color="C1")
    ax.set_xticks(np.arange(len(rpe_fam["label"]))); ax.set_xticklabels(rpe_fam["label"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Familiar m_hat"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.bar(np.arange(len(rpe_nov["label"])) - 0.2, rpe_nov["m_hat"], 0.4, label="RPE m_hat", color="C0")
    ax.bar(np.arange(len(aif_nov["label"])) + 0.2, aif_nov["m_hat"], 0.4, label="AIF m_hat", color="C1")
    ax.set_xticks(np.arange(len(rpe_nov["label"]))); ax.set_xticklabels([str(l) for l in rpe_nov["label"]], rotation=45, ha="right", fontsize=8)
    ax.set_title("Novel m_hat"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 2]
    ax.plot(fracs, rpe_mix["m_hat"], "o-", color="C0", label="RPE m_hat")
    ax.plot(fracs, aif_mix["m_hat"], "o-", color="C1", label="AIF m_hat")
    ax.set_xlabel("mix fraction (0=CS+, 1=CS-)"); ax.set_ylabel("m_hat")
    ax.set_title("Mixture m_hat"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Phase 3: novel-odor + mixture probes after CS+/CS- training", y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / "03_novel_odor_probe.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")

    return rpe_fam, aif_fam, rpe_nov, aif_nov, rpe_mix, aif_mix, probes


if __name__ == "__main__":
    main()
