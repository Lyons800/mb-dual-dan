"""Load the Winding 2023 larval Drosophila connectome.

Returns a sparse signed weight matrix plus per-neuron annotations.

Signs are NOT in Supplementary-Data-S1 — see nt_signs.py for default conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Cell-type macro groups in Winding's `annotations.csv`.
# `MBIN` aggregates DANs + OANs + other modulatory MB inputs.
MB_TYPES = {"KC", "MBON", "MBIN", "MB-FBN", "MB-FFN"}


@dataclass
class Connectome:
    W: sp.csr_matrix              # [N, N] with W[post, pre] = pre->post synapse count
    neuron_ids: np.ndarray        # [N] integer IDs from the connectivity matrix
    annotations: pd.DataFrame     # long-format: one row per neuron, columns: id, side, celltype, ...

    @property
    def N(self) -> int:
        return self.W.shape[0]

    def by_celltype(self, types: set[str] | str) -> np.ndarray:
        """Return positional indices into `neuron_ids` for the requested cell types."""
        if isinstance(types, str):
            types = {types}
        ids = self.annotations.loc[self.annotations.celltype.isin(types), "neuron_id"].to_numpy()
        # Some annotated IDs may be absent from the matrix (unpaired bilateral, etc.)
        lookup = {nid: i for i, nid in enumerate(self.neuron_ids)}
        return np.array([lookup[i] for i in ids if i in lookup], dtype=np.int64)


def _annotations_long(annot: pd.DataFrame) -> pd.DataFrame:
    """Winding's annotations.csv has paired `left_id`/`right_id` columns.

    Flatten to one row per neuron with a `side` label.
    """
    cols = ["celltype", "additional_annotations", "level_7_cluster"]
    left = annot.rename(columns={"left_id": "neuron_id"})[["neuron_id", *cols]].copy()
    left["side"] = "L"
    right = annot.rename(columns={"right_id": "neuron_id"})[["neuron_id", *cols]].copy()
    right["side"] = "R"
    out = pd.concat([left, right], ignore_index=True)
    # 'no pair' marks unpaired bilateral/midline rows — coerce to NaN and drop.
    out["neuron_id"] = pd.to_numeric(out["neuron_id"], errors="coerce")
    out = out.dropna(subset=["neuron_id"]).reset_index(drop=True)
    out["neuron_id"] = out["neuron_id"].astype(np.int64)
    out = out.drop_duplicates(subset=["neuron_id"], keep="first").reset_index(drop=True)
    return out


def load_winding(data_dir: Path | str | None = None) -> Connectome:
    """Load the Winding 2023 larval connectome from `data/winding_2023/extracted/Supplementary-Data-S1/`."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[3] / "data" / "winding_2023" / "extracted" / "Supplementary-Data-S1"
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Connectome dir not found: {data_dir}")

    # Winding's CSV is W_raw[pre, post] (rows = presynaptic ids, cols = postsynaptic ids).
    # We transpose to the standard W[post, pre] convention so y = W @ x gives postsynaptic activity.
    M = pd.read_csv(data_dir / "all-all_connectivity_matrix.csv", index_col=0)
    pre_ids = M.index.astype(np.int64).to_numpy()
    post_ids = M.columns.astype(np.int64).to_numpy()
    assert np.array_equal(post_ids, pre_ids), "Expected square symmetric id ordering on rows/cols"
    W = sp.csr_matrix(M.values.T.astype(np.float32))
    ids = post_ids

    annot_raw = pd.read_csv(data_dir / "annotations.csv")
    annot = _annotations_long(annot_raw)

    # Filter annotations to ids actually in the matrix
    annot = annot[annot.neuron_id.isin(ids)].reset_index(drop=True)

    return Connectome(W=W, neuron_ids=ids, annotations=annot)


def summary(c: Connectome) -> str:
    counts = c.annotations.celltype.value_counts()
    nnz = c.W.nnz
    total_syn = int(c.W.sum())
    lines = [
        f"Winding 2023 larval connectome",
        f"  neurons in matrix : {c.N}",
        f"  annotated neurons : {len(c.annotations)}",
        f"  nonzero edges     : {nnz:,}",
        f"  total synapses    : {total_syn:,}",
        f"  mean weight (nz)  : {total_syn / max(nnz, 1):.2f}",
        f"",
        f"Top cell types:",
    ]
    for ct, n in counts.head(20).items():
        lines.append(f"  {ct:20s} {n:5d}")
    return "\n".join(lines)
