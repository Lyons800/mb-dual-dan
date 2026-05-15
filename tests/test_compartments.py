"""Phase 3b regression tests: pin down compartment-discovery numbers.

These run on DANs only (14 cells: DAN-c1, d1, f1, g1, i1, j1, k1 x 2 hemispheres),
which is the biologically faithful set per Eichler 2017 — OANs and other MBINs
have promiscuous KC contacts and contaminate compartment recovery.
"""

import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.connectome.compartments import compartment_groups, discover_bipartite


def test_dan_jaccard_matrix_shape():
    c = load_winding()
    mb = extract_mb(c, include_pns=False)
    bi = discover_bipartite(mb, threshold=0.6, dans_only=True)
    assert bi.jaccard.shape == (14, 48)


def test_dan_top_pair_is_canonical_compartment():
    """The highest-Jaccard pairing should be a within-compartment DAN-MBON
    pair (e.g., DAN-i1 with MBON-i1) — direct evidence the connectome
    encodes Eichler 2017's compartmental organization."""
    c = load_winding()
    mb = extract_mb(c, include_pns=False)
    bi = discover_bipartite(mb, threshold=0.7, dans_only=True)
    assert bi.jaccard.max() > 0.75
    i, j, score = bi.pairings[0]
    g = mb.groups()
    dan_label = mb.subtype[g["DAN"][i]]
    mbon_label = mb.subtype[g["MBON"][j]]
    # both should be MBON-x / DAN-x with overlapping compartment letter
    assert dan_label.startswith("DAN-")
    assert mbon_label.startswith("MBON-")
    # extract the compartment letter (e.g., DAN-i1 -> i)
    dan_letter = dan_label.split("-")[1][0]
    # mbon_label may be like "MBON-i1" or "MBON-h1; MBON-h2" — check ANY contains the letter
    mbon_letters = {p.split("-")[1][0] for p in mbon_label.split("; ") if p.startswith("MBON-")}
    assert dan_letter in mbon_letters, f"{dan_label} should pair within compartment, got {mbon_label}"


def test_tight_dan_compartments_recoverable():
    """At Jaccard >= 0.7 enough tight DAN-MBON pairs survive to reveal structure."""
    c = load_winding()
    mb = extract_mb(c, include_pns=False)
    bi = discover_bipartite(mb, threshold=0.7, dans_only=True)
    assert len(bi.pairings) >= 10


def test_mbins_only_shows_kc_overlap_contamination():
    """When we DON'T restrict to DANs, OANs/MBIN-others inflate the matrix
    and break clean compartment recovery — this is precisely why we default
    dans_only=True."""
    c = load_winding()
    mb = extract_mb(c, include_pns=False)
    bi_all = discover_bipartite(mb, threshold=0.6, dans_only=False)
    bi_dan = discover_bipartite(mb, threshold=0.6, dans_only=True)
    # Including OANs / other MBINs adds rows -> more pairs above the same threshold
    assert bi_all.jaccard.shape == (28, 48)
    assert bi_dan.jaccard.shape == (14, 48)
    assert len(bi_all.pairings) > len(bi_dan.pairings)
