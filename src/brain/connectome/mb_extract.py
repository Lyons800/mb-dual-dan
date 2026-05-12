"""Extract the mushroom-body subgraph from the larval connectome.

Returns a sub-connectome containing KCs, MBONs, MBINs (DANs + OANs + modulatory),
plus the projection neurons (PNs) that feed odor information into the calyx.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from brain.connectome.loader import Connectome


@dataclass
class MBSubgraph:
    W: sp.csr_matrix          # [n, n] post×pre, unsigned synapse counts within subgraph
    neuron_ids: np.ndarray    # [n]
    celltype: np.ndarray      # [n] string array
    side: np.ndarray          # [n] "L"/"R"
    subtype: np.ndarray       # [n] subtype label (e.g., "DAN-c1", "OAN-e1", "MBIN-b1")

    # canonical group masks
    is_kc: np.ndarray
    is_mbon: np.ndarray
    is_mbin: np.ndarray       # supersets DANs + OANs + other modulatory
    is_dan: np.ndarray        # DANs only (subset of is_mbin) — labels DAN-c1/d1/f1/g1/i1/j1/k1
    is_oan: np.ndarray        # OANs only
    is_pn: np.ndarray
    is_fbn: np.ndarray        # MB-FBN feedback
    is_ffn: np.ndarray        # MB-FFN feedforward

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def groups(self) -> dict[str, np.ndarray]:
        return {
            "KC":  np.flatnonzero(self.is_kc),
            "MBON": np.flatnonzero(self.is_mbon),
            "MBIN": np.flatnonzero(self.is_mbin),
            "DAN":  np.flatnonzero(self.is_dan),
            "OAN":  np.flatnonzero(self.is_oan),
            "PN":  np.flatnonzero(self.is_pn),
            "MB-FBN": np.flatnonzero(self.is_fbn),
            "MB-FFN": np.flatnonzero(self.is_ffn),
        }


def extract_mb(c: Connectome, include_pns: bool = True) -> MBSubgraph:
    a = c.annotations
    types = {"KC", "MBON", "MBIN", "MB-FBN", "MB-FFN"}
    if include_pns:
        types |= {"PN"}

    mask_keep = a.celltype.isin(types)
    sub_ids = a.loc[mask_keep, "neuron_id"].to_numpy()
    sub_types = a.loc[mask_keep, "celltype"].to_numpy()
    sub_side = a.loc[mask_keep, "side"].to_numpy()
    sub_subtype = a.loc[mask_keep, "subtype"].to_numpy()

    # positional indices into the full matrix
    lookup = {nid: i for i, nid in enumerate(c.neuron_ids)}
    keep_idx = np.array([lookup[i] for i in sub_ids if i in lookup], dtype=np.int64)
    keep_set = set(sub_ids.tolist())
    final_mask = np.array([i in keep_set for i in sub_ids], dtype=bool)
    sub_ids = sub_ids[final_mask]
    sub_types = sub_types[final_mask]
    sub_side = sub_side[final_mask]
    sub_subtype = sub_subtype[final_mask]

    W_sub = c.W[keep_idx, :][:, keep_idx].tocsr()

    is_kc   = sub_types == "KC"
    is_mbon = sub_types == "MBON"
    is_mbin = sub_types == "MBIN"
    is_pn   = sub_types == "PN"
    is_fbn  = sub_types == "MB-FBN"
    is_ffn  = sub_types == "MB-FFN"

    # Subtype-based refinements within MBIN: DANs vs OANs vs other
    sub_str = np.array([str(s) if s is not None else "" for s in sub_subtype])
    is_dan = is_mbin & np.array([s.startswith("DAN-") for s in sub_str])
    is_oan = is_mbin & np.array([s.startswith("OAN-") for s in sub_str])

    return MBSubgraph(
        W=W_sub, neuron_ids=sub_ids, celltype=sub_types, side=sub_side,
        subtype=sub_str,
        is_kc=is_kc, is_mbon=is_mbon, is_mbin=is_mbin,
        is_dan=is_dan, is_oan=is_oan,
        is_pn=is_pn, is_fbn=is_fbn, is_ffn=is_ffn,
    )


def mb_summary(mb: MBSubgraph) -> str:
    g = mb.groups()
    lines = [f"Mushroom-body subgraph: {mb.n} neurons, {mb.W.nnz:,} nonzero edges"]
    for name, idx in g.items():
        l = int(np.sum(mb.side[idx] == "L"))
        r = int(np.sum(mb.side[idx] == "R"))
        lines.append(f"  {name:8s} n={len(idx):4d}  (L={l}, R={r})")

    # cross-group edge counts
    lines.append("")
    lines.append("Edge counts between groups (rows=post, cols=pre):")
    names = list(g.keys())
    header = "post\\pre  " + "  ".join(f"{n:>7s}" for n in names)
    lines.append(header)
    for post_name in names:
        post_idx = g[post_name]
        row = [f"{post_name:8s}"]
        for pre_name in names:
            pre_idx = g[pre_name]
            block = mb.W[post_idx, :][:, pre_idx]
            row.append(f"{block.nnz:>7d}")
        lines.append("  ".join(row))
    return "\n".join(lines)
