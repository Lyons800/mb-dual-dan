"""Phase 3b: discover MB compartments from the bipartite MBIN <-> MBON KC overlap.

Each larval MB compartment is a (DAN, MBON, KC-pool) triplet (Eichler 2017,
Saumweber 2018) — a DAN that modulates plasticity at the KC->MBON synapses
sharing the DAN's KC contact zone. So compartments appear in the wiring as
high-Jaccard pairs of (MBIN, MBON) on their shared KC pool.

We compute the full [n_mbin x n_mbon] Jaccard matrix, hierarchically reorder
both axes, threshold for candidate compartments, and visualize.

Expected: a clear block-diagonal structure in the reordered Jaccard heatmap,
plus a handful of high-overlap (MBIN, MBON) pairs that correspond to canonical
compartment-level DAN-MBON couplings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.connectome.compartments import discover_bipartite

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main(threshold: float = 0.4):
    c = load_winding()
    mb = extract_mb(c, include_pns=False)
    bi = discover_bipartite(mb, threshold=threshold)

    n_mbin, n_mbon = bi.jaccard.shape
    print(f"MBIN x MBON Jaccard matrix: {n_mbin} x {n_mbon}")
    print(f"Mean overlap          : {bi.jaccard.mean():.3f}")
    print(f"Max  overlap          : {bi.jaccard.max():.3f}")
    print(f"Pairs over threshold {threshold}: {len(bi.pairings)}")

    # How many MBINs / MBONs participate in at least one high-overlap pair?
    in_pair_mbin = set(i for i, _, _ in bi.pairings)
    in_pair_mbon = set(j for _, j, _ in bi.pairings)
    print(f"MBINs in some compartment pair: {len(in_pair_mbin)}/{n_mbin}")
    print(f"MBONs in some compartment pair: {len(in_pair_mbon)}/{n_mbon}")

    # Show top 12 highest-overlap pairs
    print("\nTop 12 highest-overlap (MBIN, MBON) pairs:")
    for i, j, score in bi.pairings[:12]:
        print(f"  MBIN[{i:>2}] (id={bi.mbin_ids[i]:>9}) <->  MBON[{j:>2}] (id={bi.mbon_ids[j]:>9})   Jaccard={score:.3f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 1. Raw Jaccard heatmap (original order)
    ax = axes[0]
    im = ax.imshow(bi.jaccard, cmap="magma", vmin=0, vmax=bi.jaccard.max(), aspect="auto")
    ax.set_title("MBIN x MBON Jaccard\n(original order)")
    ax.set_xlabel("MBON index"); ax.set_ylabel("MBIN index")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. Reordered heatmap (compartments emerge)
    ax = axes[1]
    J_ord = bi.jaccard[np.ix_(bi.mbin_order, bi.mbon_order)]
    im = ax.imshow(J_ord, cmap="magma", vmin=0, vmax=bi.jaccard.max(), aspect="auto")
    ax.set_title("Reordered by hierarchical clustering\nblock-diagonal => compartments")
    ax.set_xlabel("MBON (reordered)"); ax.set_ylabel("MBIN (reordered)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. Threshold + pair counts
    ax = axes[2]
    high = (J_ord >= threshold).astype(np.float32)
    ax.imshow(high, cmap="binary", aspect="auto")
    ax.set_title(f"Jaccard >= {threshold}  ({len(bi.pairings)} pairs)")
    ax.set_xlabel("MBON (reordered)"); ax.set_ylabel("MBIN (reordered)")

    fig.suptitle("Phase 3b — bipartite compartment recovery from larval MB connectome", y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / "04_compartments.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
