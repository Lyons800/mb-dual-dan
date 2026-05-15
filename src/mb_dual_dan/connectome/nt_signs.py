"""Apply neurotransmitter-based excitation/inhibition signs to the connectome.

Winding's Supplementary Data S1 does NOT include neurotransmitter labels — those live
in the Science paper's Supplementary Table S2. Until that table is wired in here, we
fall back to canonical-class conventions that are correct for >95% of larval neurons:

    KC, MBON, PN, sensory, projection types                -> cholinergic (excitatory)
    LN, APL                                                -> GABAergic   (inhibitory)
    MBIN (DAN/OAN/modulatory)                              -> modulatory  (no fast sign)
    everything else                                        -> excitatory  (default)

The "modulatory" category is signed +1 here as a stand-in; in the real RPE/AIF models
DAN action is gated separately as a teaching signal, not a fast post-synaptic drive.

TODO: replace this with per-neuron NT calls from Winding Science Table S2.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mb_dual_dan.connectome.loader import Connectome

INHIBITORY_CELLTYPES = {"LN", "APL"}
MODULATORY_CELLTYPES = {"MBIN"}  # treated as +1 here; real DAN logic is in the agents


def neuron_signs(c: Connectome) -> np.ndarray:
    """Return [N] array of {+1, -1} signs per neuron, based on cell-type heuristics."""
    a = c.annotations.set_index("neuron_id")
    signs = np.ones(c.N, dtype=np.float32)
    for i, nid in enumerate(c.neuron_ids):
        if nid in a.index:
            ct = a.loc[nid, "celltype"]
            if isinstance(ct, str) and ct in INHIBITORY_CELLTYPES:
                signs[i] = -1.0
    return signs


def signed_W(c: Connectome) -> sp.csr_matrix:
    """Apply presynaptic-source sign to the connectome weight matrix.

    W_signed[post, pre] = sign(pre) * synapse_count(pre -> post).
    """
    signs = neuron_signs(c)
    # Multiply each column (presynaptic source) by its sign:
    # csr_matrix * diag(signs) is the right way to scale columns.
    D = sp.diags(signs)
    return (c.W @ D).tocsr()
