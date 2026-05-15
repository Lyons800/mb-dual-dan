"""Paper Figure 1: substrate, compartment recovery, and the four-architecture lineup.

Layout (2 x 2):
  A — Substrate composition (MB subgraph cell counts)
  B — DAN x MBON shared-KC Jaccard heatmap (hierarchical reorder)
  C — Top canonical pairs with subtype labels (DAN-i1 <-> MBON-i1, etc.)
  D — Architecture schematic: four agents on the same substrate

This figure grounds the paper: shows the substrate exists, has discoverable
compartment structure, and supports a clean four-way model comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.connectome.compartments import discover_bipartite

OUT = Path(__file__).parent / "fig1_substrate.png"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "legend.frameon": False, "figure.dpi": 150,
})


def main():
    c = load_winding()
    mb = extract_mb(c, include_pns=True)

    fig = plt.figure(figsize=(12.5, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.0, 1.2])

    # --- Panel A: cell counts in the MB subgraph
    ax = fig.add_subplot(gs[0, 0])
    g = mb.groups()
    labels = ["KC", "MBON", "DAN", "OAN", "MB-FBN", "MB-FFN", "PN"]
    counts = [len(g[k]) for k in labels]
    colors = ["#2266aa", "#cc7733", "#cc3333", "#666699", "#aabbaa", "#bbaaaa", "#cccccc"]
    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors, edgecolor="white")
    for i, v in enumerate(counts):
        ax.text(v + 2, i, str(v), va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("number of neurons in MB subgraph")
    ax.set_title("A   MB subgraph composition  (Winding 2023)", loc="left", fontweight="bold")
    ax.invert_yaxis()
    ax.text(0.95, 0.05,
            f"total: {mb.W.shape[0]} neurons,\n{mb.W.nnz:,} edges,\n{int(mb.W.sum()):,} synapses",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, style="italic", color="#666")

    # --- Panel B: DAN x MBON Jaccard heatmap
    ax = fig.add_subplot(gs[0, 1])
    bi = discover_bipartite(mb, threshold=0.4, dans_only=True)
    J_ord = bi.jaccard[np.ix_(bi.mbin_order, bi.mbon_order)]
    im = ax.imshow(J_ord, cmap="magma", vmin=0, vmax=bi.jaccard.max(), aspect="auto")
    ax.set_title("B   DAN $\\times$ MBON KC-pool Jaccard  (reordered)", loc="left", fontweight="bold")
    ax.set_xlabel("MBON (clustered)"); ax.set_ylabel("DAN (clustered)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")

    # --- Panel C: top canonical pairs
    ax = fig.add_subplot(gs[1, 0])
    g_all = mb.groups()
    dan_idx, mbon_idx = g_all["DAN"], g_all["MBON"]
    top = bi.pairings[:8]
    rows = []
    for i, j, score in top:
        dan = mb.subtype[dan_idx[i]]
        mbon = mb.subtype[mbon_idx[j]]
        # take primary label after semicolon split
        mbon_primary = mbon.split(";")[0].strip()
        rows.append((dan, mbon_primary, score))

    ax.axis("off")
    ax.set_title("C   Top DAN--MBON pairs  (Jaccard $\\geq 0.7$)", loc="left", fontweight="bold")
    table_data = [["DAN", "MBON", "Jaccard"]]
    for dan, mbon, score in rows:
        # highlight canonical within-compartment pairs (matching letter)
        dan_letter = dan.split("-")[1][0] if "-" in dan else ""
        mbon_letter = mbon.split("-")[1][0] if "-" in mbon else ""
        canonical = (dan_letter and mbon_letter and dan_letter == mbon_letter)
        marker = "  *" if canonical else "   "
        table_data.append([dan + marker, mbon, f"{score:.3f}"])
    tbl = ax.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.32, 0.5, 0.18])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0, 1.4)
    # bold header
    for k in range(3):
        tbl[(0, k)].set_text_props(fontweight="bold")
        tbl[(0, k)].set_facecolor("#eeeeee")
    ax.text(0.5, 0.05, "*  canonical within-compartment pair (DAN-x $\\leftrightarrow$ MBON-x)",
            transform=ax.transAxes, ha="center", fontsize=7, style="italic", color="#666")

    # --- Panel D: architecture schematic — text-based grid showing the 4 models
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("D   Four agents on a shared connectome substrate", loc="left", fontweight="bold")

    # Stylised "boxes" for each agent layer
    def box(x0, y0, w, h, text, color, ax):
        rect = plt.Rectangle((x0, y0), w, h, facecolor=color, edgecolor="#333",
                             linewidth=0.8, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    # shared substrate row
    box(0.05, 0.78, 0.9, 0.10, "PN $\\to$ KC sparse code  (real Winding wiring)",
        "#dde3ec", ax)
    # divider
    ax.annotate("", xy=(0.5, 0.74), xytext=(0.5, 0.78),
                arrowprops=dict(arrowstyle="-", color="#333", lw=0.6))

    # four model rows
    box(0.05, 0.58, 0.21, 0.14, "RPE\nKC$\\to$MBON\nsigned weights", "#a5c7e3", ax)
    box(0.28, 0.58, 0.21, 0.14, "AIF\nBayesian\nposterior + DAN", "#e8b384", ax)
    box(0.51, 0.58, 0.21, 0.14, "Dual-DAN\n$\\eta_{eff}=\\eta_0(1+\\lambda\\sigma)$", "#cba0d6", ax)
    box(0.74, 0.58, 0.21, 0.14, "Dual-Valence\nM$^+$/M$^-$ POM", "#a0d6a8", ax)

    # behavioural outputs row
    box(0.05, 0.32, 0.21, 0.14, "Acquisition\n+ Extinction\n(Fig. 2)", "#f5f5f5", ax)
    box(0.28, 0.32, 0.21, 0.14, "Novelty\n+ Habituation\n(Fig. 3)", "#f5f5f5", ax)
    box(0.51, 0.32, 0.21, 0.14, "Latent\nInhibition\n(Fig. 4)", "#f5f5f5", ax)
    box(0.74, 0.32, 0.21, 0.14, "Fast Reversal\nvia POM\n(Fig. 5)", "#f5f5f5", ax)

    for x in [0.155, 0.385, 0.615, 0.845]:
        ax.annotate("", xy=(x, 0.46), xytext=(x, 0.58),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=0.6))

    ax.text(0.5, 0.18, "Each ingredient is individually necessary for its phenomenon —\n"
                       "removing any one breaks at least one signature.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, style="italic", color="#333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
