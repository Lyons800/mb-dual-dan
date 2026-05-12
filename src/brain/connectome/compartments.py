"""Discover mushroom-body compartments from connectome wiring alone.

The larval MB has ~10 anatomically-defined compartments (Eichler et al. 2017,
Saumweber et al. 2018) — each a triplet of (DAN, MBON, shared KC pool) that
performs semi-independent associative learning.

If the wiring is biologically meaningful, we should be able to *recover* this
compartment structure from the connectome alone, with no anatomical labels —
purely by clustering MBINs and MBONs by their shared KC-contact patterns.

Method:
    1. Project each MBIN to its KC-contact vector (binary, length n_kc).
    2. Project each MBON to its KC-input vector.
    3. Stack into one matrix X of shape [n_mbin + n_mbon, n_kc].
    4. Hierarchical clustering with cosine distance on the rows of X.
    5. Cut at K clusters; each is a candidate compartment.

A clean result: the cell-by-cell similarity matrix shows block-diagonal
structure with one block per compartment, and each block contains both MBINs
and MBONs (a DAN-MBON pair with a shared KC pool — Aso's compartment unit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from brain.connectome.mb_extract import MBSubgraph


@dataclass
class Compartments:
    X: np.ndarray            # [n_mbin + n_mbon, n_kc] KC-projection vectors (binary)
    is_mbin: np.ndarray      # [n_mbin + n_mbon] True for MBIN rows
    cell_ids: np.ndarray     # [n_mbin + n_mbon] neuron ids from the connectome
    cluster: np.ndarray      # [n_mbin + n_mbon] cluster label per cell
    Z: np.ndarray            # linkage matrix
    distance: np.ndarray     # squareform pairwise distance
    order: np.ndarray        # row ordering for block-diagonal display
    k: int

    def cluster_summary(self) -> list[dict]:
        out = []
        for c in sorted(np.unique(self.cluster)):
            members = np.flatnonzero(self.cluster == c)
            n_mbin = int(self.is_mbin[members].sum())
            n_mbon = len(members) - n_mbin
            shared_kc = (self.X[members].sum(axis=0) > 0).sum()
            out.append({
                "compartment": int(c),
                "size": len(members),
                "n_mbin": n_mbin,
                "n_mbon": n_mbon,
                "kc_pool_size": int(shared_kc),
            })
        return out


@dataclass
class BipartiteCompartments:
    """Bipartite view: MBIN x MBON similarity matrix, reordered so true
    compartments appear as 1:1 (or N:M) blocks where DANs and MBONs share KCs."""
    jaccard: np.ndarray            # [n_mbin, n_mbon] Jaccard overlap on KC pools
    mbin_ids: np.ndarray
    mbon_ids: np.ndarray
    mbin_order: np.ndarray         # ordering for plotting
    mbon_order: np.ndarray
    pairings: list[tuple[int, int, float]]   # (mbin_row, mbon_col, jaccard) high-overlap pairs


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a * b).sum())
    union = float((a + b > 0).sum())
    return inter / union if union > 0 else 0.0


def discover_bipartite(mb: MBSubgraph, threshold: float = 0.4,
                       dans_only: bool = True) -> BipartiteCompartments:
    """Build the MBIN x MBON Jaccard matrix on shared KC pools, plus a
    reordering that makes block-diagonal compartment structure visible.

    Pairs above `threshold` are flagged as candidate (DAN, MBON) compartments —
    they have substantial KC pool overlap. Sets of mutually-overlapping
    MBINs+MBONs form many-to-many compartment groups.

    `dans_only` (default True) restricts the analysis to dopaminergic MBINs only
    (DAN-c1/d1/f1/g1/i1/j1/k1 in Winding's annotations). This matters because
    Eichler 2017 shows larval DANs are 1:1 with anatomical compartments, while
    OANs and "other MBINs" (APL, DPM-like, unknown-NT) have promiscuous KC
    contacts that contaminate compartment recovery if included.
    """
    g = mb.groups()
    kc_idx, mbon_idx = g["KC"], g["MBON"]
    mbin_idx = g["DAN"] if dans_only else g["MBIN"]
    mbin_kc = (mb.W[np.ix_(kc_idx, mbin_idx)].toarray().T > 0).astype(np.float32)  # [n_mbin, n_kc]
    mbon_kc = (mb.W[np.ix_(mbon_idx, kc_idx)].toarray() > 0).astype(np.float32)    # [n_mbon, n_kc]

    n_mbin, n_mbon = mbin_kc.shape[0], mbon_kc.shape[0]
    J = np.zeros((n_mbin, n_mbon), dtype=np.float32)
    for i in range(n_mbin):
        for j in range(n_mbon):
            J[i, j] = _jaccard(mbin_kc[i], mbon_kc[j])

    # Reorder via hierarchical clustering on rows then on cols.
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist
    mbin_order = leaves_list(linkage(pdist(mbin_kc, metric="cosine"), method="average"))
    mbon_order = leaves_list(linkage(pdist(mbon_kc, metric="cosine"), method="average"))

    pairings = [(int(i), int(j), float(J[i, j]))
                for i in range(n_mbin) for j in range(n_mbon)
                if J[i, j] >= threshold]
    pairings.sort(key=lambda x: -x[2])

    return BipartiteCompartments(
        jaccard=J,
        mbin_ids=mb.neuron_ids[mbin_idx],
        mbon_ids=mb.neuron_ids[mbon_idx],
        mbin_order=mbin_order,
        mbon_order=mbon_order,
        pairings=pairings,
    )


def compartment_groups(bi: BipartiteCompartments) -> list[dict]:
    """Group high-Jaccard pairings into connected components of the bipartite graph.

    Each connected component is a candidate compartment-like cluster: a set of
    MBINs that share KC pools with a set of MBONs (transitively). Note that
    connected components are an upper bound — a single "hub" cell can fuse what
    are anatomically distinct compartments into one component. For tight
    canonical compartments use a higher Jaccard threshold (>= 0.7).
    """
    import networkx as nx
    G = nx.Graph()
    for i, j, s in bi.pairings:
        G.add_node(("mbin", i)); G.add_node(("mbon", j))
        G.add_edge(("mbin", i), ("mbon", j), weight=s)
    comps = list(nx.connected_components(G))
    comps.sort(key=lambda x: -len(x))
    out = []
    for k, comp in enumerate(comps):
        mbins = sorted([i for kind, i in comp if kind == "mbin"])
        mbons = sorted([j for kind, j in comp if kind == "mbon"])
        out.append({
            "id": k + 1,
            "n_mbin": len(mbins),
            "n_mbon": len(mbons),
            "mbin_idx": mbins,
            "mbon_idx": mbons,
        })
    return out


def discover_compartments(mb: MBSubgraph, k: int = 20, method: str = "average",
                          metric: str = "cosine") -> Compartments:
    """Recover compartments by clustering MBINs + MBONs in KC-projection space.

    The default k=20 corresponds to ~10 compartments x 2 hemispheres.
    """
    g = mb.groups()
    kc_idx, mbon_idx, mbin_idx = g["KC"], g["MBON"], g["MBIN"]
    # KC-projection vectors per cell
    mbin_proj = (mb.W[np.ix_(kc_idx, mbin_idx)].toarray().T > 0).astype(np.float32)   # [n_mbin, n_kc]
    mbon_proj = (mb.W[np.ix_(mbon_idx, kc_idx)].toarray() > 0).astype(np.float32)     # [n_mbon, n_kc]
    X = np.vstack([mbin_proj, mbon_proj])

    # cell metadata
    is_mbin = np.concatenate([np.ones(len(mbin_idx), dtype=bool),
                              np.zeros(len(mbon_idx), dtype=bool)])
    cell_ids = np.concatenate([mb.neuron_ids[mbin_idx], mb.neuron_ids[mbon_idx]])

    # pairwise distance + hierarchical clustering
    dvec = pdist(X, metric=metric)
    Z = linkage(dvec, method=method)
    clust = fcluster(Z, t=k, criterion="maxclust")
    distance = squareform(dvec)

    # row order from linkage (block-diagonal display)
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)

    return Compartments(X=X, is_mbin=is_mbin, cell_ids=cell_ids,
                        cluster=clust, Z=Z, distance=distance, order=order, k=k)
