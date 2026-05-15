"""Smoke tests for the connectome loader.

These check the *Winding 2023* invariants — if the published numbers ever change,
these tests pin down what we're working against.
"""

import numpy as np

from mb_dual_dan.connectome import load_winding, extract_mb
from mb_dual_dan.connectome.nt_signs import neuron_signs, signed_W


def test_winding_loads():
    c = load_winding()
    assert c.N == 2952
    assert c.W.nnz == 110_677
    # Winding 2023: ~352K total synapses
    assert 350_000 < c.W.sum() < 360_000


def test_mb_counts():
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    g = mb.groups()
    # Expected paired counts (L+R combined)
    assert len(g["KC"]) == 144         # 73L + 71R
    assert len(g["MBON"]) == 48        # 24L + 24R
    assert len(g["MBIN"]) == 28        # 14L + 14R (DAN+OAN+modulatory)
    assert len(g["PN"]) == 206         # 103L + 103R


def test_kc_to_mbon_is_dominant_learning_pathway():
    """Sanity check on weight matrix orientation: KC->MBON should be the largest
    cross-group edge count within the MB."""
    c = load_winding()
    mb = extract_mb(c, include_pns=False)  # exclude PNs so they don't dominate
    g = mb.groups()
    kc, mbon = g["KC"], g["MBON"]
    # W[post, pre], so KC->MBON is mb.W[mbon, kc]
    kc_to_mbon_edges = mb.W[np.ix_(mbon, kc)].nnz
    mbon_to_kc_edges = mb.W[np.ix_(kc, mbon)].nnz
    assert kc_to_mbon_edges > mbon_to_kc_edges
    assert kc_to_mbon_edges > 1000  # known: 3032


def test_signs_apply():
    c = load_winding()
    signs = neuron_signs(c)
    assert signs.shape == (c.N,)
    assert set(np.unique(signs).tolist()).issubset({-1.0, 1.0})
    # signed_W preserves shape
    Ws = signed_W(c)
    assert Ws.shape == c.W.shape
